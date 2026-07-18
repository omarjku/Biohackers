"""
calibration.py — Platt scaling + the no-call gate
Owner: Waji (pipeline)

predictor.py emits a raw logistic-regression probability. That number is not a
confidence: L2-regularized logistic regression on a wide, sparse feature matrix
is systematically over-confident near 0 and 1, and `class_weight="balanced"`
shifts the whole curve away from the true prevalence. Reporting it to a
clinician as "87% confident" would be a lie with a decimal point on it.

This module does two things:

1. Platt scaling, fitted on the CALIBRATION slice of the grouped split — never
   train (optimistically biased, the model has already seen those rows) and
   never test (leaks, and then the reliability plot is meaningless). splits.py
   carves out the third slice precisely so this file has somewhere honest to
   fit. See GroupedSplit.calibration.

2. The no-call gate. A prediction is withheld when:
     a. the calibrated probability sits in an ambiguous band (default 0.3-0.7),
     b. the genome is out-of-distribution vs. what the model trained on, or
     c. Platt scaling could not be fitted at all for that drug.
   Target-gene absence is NOT handled here — predictor.py already turns that
   into "not_applicable", which is a stronger statement than "no_call" and must
   not be downgraded to one.

On out-of-distribution detection: the obvious signal — "this genome's cluster
was never seen in training" — is useless here, because a grouped split puts
EVERY test genome in an unseen cluster by construction. Flagging on that would
no-call the entire test set. So novelty is measured in feature space instead:
Hamming distance from the query genome's AMR profile to the nearest training
genome, compared against how far apart training genomes are from each other.
A genome carrying a combination of resistance determinants unlike anything in
training is one the model cannot honestly score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from data_io import Dataset
from predictor import Predictor
from schemas import Prediction

# Ambiguous band. Inside this range the model is not committing to an answer, so
# neither do we. The brief suggests 0.3-0.7; it is deliberately wide, because the
# cost of a confidently wrong antibiotic call is not symmetric with the cost of
# saying "test this one in the lab".
NO_CALL_LOW = 0.30
NO_CALL_HIGH = 0.70

# A query genome is out-of-distribution if its nearest training genome is further
# away than this percentile of the training set's own nearest-neighbour distances.
# 99 rather than 100: a single weird training genome should not stretch the
# envelope wide enough to wave everything through.
OOD_PERCENTILE = 99.0

# Probabilities are clipped before the logit transform so a saturated 0.0/1.0
# does not produce an infinite feature for the Platt fit.
_EPS = 1e-6


class CalibrationError(ValueError):
    """Raised when calibration is asked for something it cannot honestly do."""


def _logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


# --------------------------------------------------------------------------
# 1. Per-drug calibrator
# --------------------------------------------------------------------------


@dataclass
class DrugCalibrator:
    """Platt scaling + an OOD envelope for a single antibiotic."""

    drug: str
    platt: LogisticRegression | None          # None when it could not be fitted
    train_profiles: np.ndarray                 # training feature matrix, 0/1
    feature_names: list[str]
    ood_threshold: float
    n_calibration: int = 0
    fit_failure: str | None = None             # why platt is None, if it is

    @property
    def is_fitted(self) -> bool:
        return self.platt is not None

    def calibrate(self, raw_probability: float) -> float:
        """Raw model output -> calibrated probability of resistance."""
        if self.platt is None:
            # Uncalibrated passthrough. Callers must not report this as a
            # confidence — every prediction built from it is forced to no_call
            # in apply(), so it only ever reaches a human as "no call".
            return float(raw_probability)
        x = _logit(raw_probability).reshape(1, 1)
        return float(self.platt.predict_proba(x)[0, 1])

    def distance_to_training(self, row: pd.Series) -> float:
        """Hamming distance from this AMR profile to its nearest neighbour in train."""
        profile = row.reindex(self.feature_names).fillna(0).to_numpy(dtype=float)
        return float(np.abs(self.train_profiles - profile).sum(axis=1).min())

    def is_out_of_distribution(self, row: pd.Series) -> bool:
        return self.distance_to_training(row) > self.ood_threshold


def _ood_threshold(train_profiles: np.ndarray, calibration_profiles: np.ndarray) -> float:
    """
    How far from training data is "too far"?

    The tempting reference is the training set's own internal spread — how far
    each training genome sits from its nearest neighbour in training. That is
    far too tight. Training genomes share clusters with each other, so their
    mutual distances are tiny (often zero, for duplicated AMR profiles), and
    measuring against them flags most of a grouped test set as novel.

    The calibration slice is the honest reference instead. It sits in clusters
    the model never trained on — exactly the deployment condition — so the
    spread of its distances to training data is what "an unfamiliar but still
    scoreable genome" actually looks like. OOD means further out than that.
    """
    if len(train_profiles) == 0 or len(calibration_profiles) == 0:
        return float("inf")  # nothing to characterise a spread with; never flag

    distances = np.array(
        [
            np.abs(train_profiles - profile).sum(axis=1).min()
            for profile in calibration_profiles
        ]
    )
    # Floor of 1 so a degenerate all-identical calibration slice cannot collapse
    # the threshold to 0 and flag every genome differing by a single gene.
    return float(max(np.percentile(distances, OOD_PERCENTILE), 1.0))


# --------------------------------------------------------------------------
# 2. Fitting
# --------------------------------------------------------------------------


@dataclass
class Calibrator:
    """All per-antibiotic calibrators, fitted against a Predictor's own splits."""

    per_drug: dict[str, DrugCalibrator] = field(default_factory=dict)

    @classmethod
    def fit(cls, dataset: Dataset, predictor: Predictor) -> Calibrator:
        """
        Fit Platt scaling for every drug the predictor has a model for.

        Reuses predictor.splits, so the calibration slice is guaranteed to be the
        same cluster-disjoint slice that was held out when the model was fitted.
        Rebuilding a split here would silently break that guarantee.
        """
        calibrator = cls()
        for drug, model in predictor.models.items():
            split = predictor.splits.get(drug)
            if split is None:
                raise CalibrationError(
                    f"{drug}: predictor has a model but no retained split, so there "
                    "is no held-out slice to calibrate on."
                )

            X, y, _ = dataset.xy_for_drug(drug)
            train_profiles = X.iloc[split.train].to_numpy(dtype=float)

            X_cal, y_cal = X.iloc[split.calibration], y[split.calibration]
            raw = np.array(
                [model.probability_resistant(row) for _, row in X_cal.iterrows()]
            )

            platt: LogisticRegression | None = None
            failure: str | None = None
            if len(np.unique(y_cal)) < 2:
                failure = (
                    f"calibration slice holds only class {np.unique(y_cal).tolist()} "
                    f"across {len(y_cal)} rows — Platt scaling needs both classes"
                )
            else:
                platt = LogisticRegression(solver="lbfgs")
                platt.fit(_logit(raw).reshape(-1, 1), y_cal)

            calibrator.per_drug[drug] = DrugCalibrator(
                drug=drug,
                platt=platt,
                train_profiles=train_profiles,
                feature_names=list(X.columns),
                ood_threshold=_ood_threshold(
                    train_profiles, X_cal.to_numpy(dtype=float)
                ),
                n_calibration=len(y_cal),
                fit_failure=failure,
            )
        return calibrator

    # ----------------------------------------------------------------------
    # 3. Applying: raw Prediction -> calibrated, gated Prediction
    # ----------------------------------------------------------------------

    def apply(self, prediction: Prediction, row: pd.Series) -> Prediction:
        """
        Overwrite the placeholder confidence and apply the no-call gate.

        Returns a new Prediction; the input is left alone.
        """
        cal = self.per_drug.get(prediction.drug)
        if cal is None:
            raise CalibrationError(f"No calibrator fitted for {prediction.drug!r}")

        probability = cal.calibrate(prediction.confidence)

        call = prediction.call
        reason: str | None = None

        if prediction.call == "not_applicable":
            # The drug's target is absent. That is a stronger, deterministic
            # statement than "we are unsure" — never downgrade it to no_call.
            pass
        elif not cal.is_fitted:
            call = "no_call"
            reason = f"Confidence could not be calibrated for this drug ({cal.fit_failure})."
        elif cal.is_out_of_distribution(row):
            call = "no_call"
            reason = (
                "This genome's resistance-gene profile is unlike anything in the "
                "training data, so the model's confidence cannot be trusted."
            )
        elif NO_CALL_LOW <= probability <= NO_CALL_HIGH:
            call = "no_call"
            reason = (
                f"Calibrated probability {probability:.2f} falls in the ambiguous "
                f"band ({NO_CALL_LOW:.2f}-{NO_CALL_HIGH:.2f})."
            )
        else:
            call = "likely_to_fail" if probability >= 0.5 else "likely_to_work"

        return prediction.model_copy(
            update={"call": call, "confidence": probability, "no_call_reason": reason}
        )


def predict_calibrated(
    dataset: Dataset,
    predictor: Predictor,
    calibrator: Calibrator,
    genome_id: str,
    drugs: list[str] | None = None,
) -> list[Prediction]:
    """The full path: features -> model -> calibration -> no-call gate."""
    row = dataset.features.loc[genome_id]
    return [
        calibrator.apply(prediction, row)
        for prediction in predictor.predict(dataset, genome_id, drugs=drugs)
    ]


# --------------------------------------------------------------------------
# 4. Diagnostics — consumed by evaluation.py for reliability plots
# --------------------------------------------------------------------------


def calibration_report(
    dataset: Dataset,
    predictor: Predictor,
    calibrator: Calibrator,
) -> pd.DataFrame:
    """
    Per-drug Brier score before and after calibration, on the held-out TEST slice.

    Brier going down is the evidence that Platt scaling did something. If it goes
    up, the calibration slice was too small or unrepresentative — report that
    rather than quietly shipping worse numbers.
    """
    rows = []
    for drug, model in predictor.models.items():
        split = predictor.splits[drug]
        cal = calibrator.per_drug[drug]

        X, y, _ = dataset.xy_for_drug(drug)
        X_test, y_test = X.iloc[split.test], y[split.test]

        raw = np.array([model.probability_resistant(r) for _, r in X_test.iterrows()])
        calibrated = np.array([cal.calibrate(p) for p in raw])
        n_ood = sum(cal.is_out_of_distribution(r) for _, r in X_test.iterrows())

        rows.append(
            {
                "drug": drug,
                "n_test": len(y_test),
                "n_calib": cal.n_calibration,
                "brier_raw": round(float(np.mean((raw - y_test) ** 2)), 4),
                "brier_cal": round(float(np.mean((calibrated - y_test) ** 2)), 4),
                "pct_ambiguous": round(
                    100
                    * float(
                        np.mean((calibrated >= NO_CALL_LOW) & (calibrated <= NO_CALL_HIGH))
                    ),
                    1,
                ),
                "pct_ood": round(100 * n_ood / len(y_test), 1),
                "fitted": cal.is_fitted,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from data_io import load_dataset

    ds = load_dataset(Path(__file__).parent.parent / "data" / "synthetic")
    predictor = Predictor.fit(ds)
    calibrator = Calibrator.fit(ds, predictor)

    print("Calibration on held-out test slice")
    print(calibration_report(ds, predictor, calibrator).to_string(index=False))

    sample = ds.features.index[0]
    print(f"\nCalibrated predictions for {sample}")
    for prediction in predict_calibrated(ds, predictor, calibrator, sample):
        line = (
            f"  {prediction.drug:<16} {prediction.call:<16} "
            f"p(R)={prediction.confidence:.2f}  "
            f"target={prediction.target_gate_status}"
        )
        if prediction.no_call_reason:
            line += f"\n      -> {prediction.no_call_reason}"
        print(line)
