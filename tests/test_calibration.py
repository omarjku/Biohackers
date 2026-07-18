"""
Run with: pytest tests/test_calibration.py -v

Calibration is the module a demo can most easily lie through: it produces a
number that looks authoritative, printed next to a drug name, in front of a
clinician. The tests below are mostly about what must NEVER happen — a
not_applicable silently downgraded to a no_call, an uncalibrated probability
escaping as a confidence, a no_call without a reason attached.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from calibration import (
    NO_CALL_HIGH,
    NO_CALL_LOW,
    CalibrationError,
    Calibrator,
    calibration_report,
    predict_calibrated,
)
from data_io import load_dataset
from predictor import Predictor

DATA_DIR = Path(__file__).parent.parent / "data" / "synthetic"


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATA_DIR)


@pytest.fixture(scope="module")
def fitted(dataset):
    predictor = Predictor.fit(dataset)
    return predictor, Calibrator.fit(dataset, predictor)


@pytest.fixture(scope="module")
def all_predictions(dataset, fitted):
    predictor, calibrator = fitted
    out = []
    for genome_id in dataset.features.index:
        out.extend(predict_calibrated(dataset, predictor, calibrator, genome_id))
    return out


# --------------------------------------------------------------------------
# The safety invariants
# --------------------------------------------------------------------------


def test_not_applicable_is_never_downgraded_to_no_call(dataset, fitted):
    """
    Target-gene absence is a deterministic statement, stronger than "unsure".
    If calibration ever converts it to no_call, the whole point of the target
    gate is lost — a missing drug target would come back as "we couldn't decide"
    rather than "this drug has nothing to bind to".
    """
    predictor, calibrator = fitted
    seen_any = False
    for genome_id in dataset.features.index:
        raw = predictor.predict(dataset, genome_id)
        row = dataset.features.loc[genome_id]
        for prediction in raw:
            if prediction.call != "not_applicable":
                continue
            seen_any = True
            assert calibrator.apply(prediction, row).call == "not_applicable"
    assert seen_any, "fixture no longer exercises the not_applicable path"


def test_every_no_call_carries_a_reason(all_predictions):
    """A withheld prediction the UI cannot explain is worse than no prediction."""
    for prediction in all_predictions:
        if prediction.call == "no_call":
            assert prediction.no_call_reason, f"{prediction.drug}: no_call without a reason"
            assert len(prediction.no_call_reason) > 20


def test_non_no_calls_carry_no_reason(all_predictions):
    """Stale reasons on a decided call would render as contradictory UI text."""
    for prediction in all_predictions:
        if prediction.call not in ("no_call",):
            assert prediction.no_call_reason is None


def test_ambiguous_band_is_always_withheld(all_predictions):
    """
    Nothing inside the ambiguous band may be reported as a decision. This is
    rule 4 of the brief — a real no-call option, not a forced yes/no.
    """
    for prediction in all_predictions:
        if prediction.call in ("likely_to_work", "likely_to_fail"):
            assert not (NO_CALL_LOW <= prediction.confidence <= NO_CALL_HIGH), (
                f"{prediction.drug}: decided '{prediction.call}' at "
                f"p={prediction.confidence:.3f}, inside the no-call band"
            )


def test_call_direction_matches_probability(all_predictions):
    """y=1 is RESISTANT, so a high probability must mean likely_to_fail."""
    for prediction in all_predictions:
        if prediction.call == "likely_to_fail":
            assert prediction.confidence > NO_CALL_HIGH
        elif prediction.call == "likely_to_work":
            assert prediction.confidence < NO_CALL_LOW


def test_confidence_stays_a_probability(all_predictions):
    for prediction in all_predictions:
        assert 0.0 <= prediction.confidence <= 1.0


def test_calibration_overwrites_the_raw_placeholder(dataset, fitted):
    """
    predictor.py emits a raw sigmoid as a placeholder. If calibration silently
    passed it through, the demo would report an uncalibrated number as a
    confidence — the exact failure this module exists to prevent.
    """
    predictor, calibrator = fitted
    genome_id = dataset.features.index[0]
    raw = {p.drug: p.confidence for p in predictor.predict(dataset, genome_id)}
    calibrated = {
        p.drug: p.confidence
        for p in predict_calibrated(dataset, predictor, calibrator, genome_id)
    }
    assert any(
        abs(raw[drug] - calibrated[drug]) > 1e-9 for drug in raw
    ), "calibrated probabilities are identical to raw ones — Platt scaling is a no-op"


def test_apply_does_not_mutate_its_input(dataset, fitted):
    predictor, calibrator = fitted
    genome_id = dataset.features.index[0]
    prediction = predictor.predict(dataset, genome_id)[0]
    before = prediction.model_copy(deep=True)
    calibrator.apply(prediction, dataset.features.loc[genome_id])
    assert prediction == before


# --------------------------------------------------------------------------
# Platt scaling actually doing its job
# --------------------------------------------------------------------------


def test_calibration_improves_brier_on_held_out_test(dataset, fitted):
    """
    The whole justification for the third split. If Brier gets worse, the
    calibration slice is unrepresentative and the confidence numbers should not
    be shipped — so this failing is informative, not just annoying.
    """
    predictor, calibrator = fitted
    report = calibration_report(dataset, predictor, calibrator)
    worse = report[report["brier_cal"] > report["brier_raw"]]
    assert worse.empty, (
        "calibration made these drugs worse:\n"
        f"{worse[['drug', 'brier_raw', 'brier_cal', 'n_calib']].to_string(index=False)}"
    )


def test_platt_is_monotonic(fitted):
    """
    Platt scaling must preserve ranking — it rescales confidence, it does not
    reorder genomes. A non-monotonic map would mean a genome with more
    resistance evidence scoring as less likely to be resistant.
    """
    _, calibrator = fitted
    probes = np.linspace(0.01, 0.99, 40)
    for drug, cal in calibrator.per_drug.items():
        mapped = [cal.calibrate(p) for p in probes]
        assert mapped == sorted(mapped) or mapped == sorted(mapped, reverse=True), (
            f"{drug}: Platt mapping is not monotonic"
        )


def test_calibrator_fitted_for_every_model(dataset, fitted):
    predictor, calibrator = fitted
    assert set(calibrator.per_drug) == set(predictor.models)
    for drug, cal in calibrator.per_drug.items():
        assert cal.is_fitted, f"{drug}: Platt fit failed ({cal.fit_failure})"


# --------------------------------------------------------------------------
# Out-of-distribution gate
# --------------------------------------------------------------------------


def test_ood_does_not_swallow_the_test_set(dataset, fitted):
    """
    A grouped split puts every test genome in an unseen cluster. An OOD rule
    keyed on cluster novelty would therefore no-call the entire test set, which
    is how the first version of this gate behaved (25-68% flagged). The envelope
    is derived from the calibration slice for exactly this reason.
    """
    predictor, calibrator = fitted
    report = calibration_report(dataset, predictor, calibrator)
    assert (report["pct_ood"] < 25.0).all(), (
        "OOD gate is flagging an implausible share of held-out genomes:\n"
        f"{report[['drug', 'pct_ood']].to_string(index=False)}"
    )


def test_wildly_novel_genome_is_flagged_ood(dataset, fitted):
    """The gate must still fire on something genuinely unlike the training data."""
    _, calibrator = fitted
    cal = next(iter(calibrator.per_drug.values()))
    # Every AMR determinant present at once — not a profile any real genome in
    # the training set carries.
    everything = dataset.features.iloc[0].copy()
    everything[:] = 1
    assert cal.is_out_of_distribution(everything)


def test_training_genome_is_not_flagged_ood(dataset, fitted):
    """A genome the model trained on cannot be novel to it."""
    predictor, calibrator = fitted
    drug = dataset.drugs[0]
    X, _, _ = dataset.xy_for_drug(drug)
    train_rows = predictor.splits[drug].train
    cal = calibrator.per_drug[drug]
    for position in train_rows[:20]:
        assert not cal.is_out_of_distribution(X.iloc[position])


# --------------------------------------------------------------------------
# Failing loudly
# --------------------------------------------------------------------------


def test_unknown_drug_raises(dataset, fitted):
    predictor, calibrator = fitted
    prediction = predictor.predict(dataset, dataset.features.index[0])[0]
    orphan = prediction.model_copy(update={"drug": "Nonexistentmycin"})
    with pytest.raises(CalibrationError, match="No calibrator"):
        calibrator.apply(orphan, dataset.features.loc[dataset.features.index[0]])
