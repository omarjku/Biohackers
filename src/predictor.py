"""
predictor.py — Module 02 core: features -> raw predictions
Owner: Waji (pipeline)

One L2-regularized logistic regression per antibiotic, fit on the train slice
of a grouped split. Emits schemas.Prediction objects; calibration.py then
replaces the placeholder confidence with a Platt-scaled probability and applies
the no-call gate on top.

Three things here are deliberate and worth not "simplifying" away:

1. The target gate runs BEFORE the model, and overrides it. If a drug's target
   gene is absent, the call is "not_applicable" regardless of what the model
   says. The brief is explicit: absence of resistance markers must never be
   reported as "likely to work" when the drug has nothing to bind to.

2. A missing feature COLUMN is not an absent gene. If the annotator never
   looked for ftsI, we know nothing about ftsI — that is target_gate_status
   "unknown", not "absent". Conflating the two silently manufactures
   not_applicable calls out of incomplete annotation.

3. evidence_category never gets promoted to "known_gene_or_mutation" on the
   strength of a model coefficient. A large coefficient means correlation in
   our training set — possibly a hitchhiker gene riding along with a real
   mechanism. Only genes on Moncef's curated list count as hard evidence;
   everything else the model leans on is "statistical_association".

Confidence emitted here is the raw, UNCALIBRATED sigmoid output. It is a
placeholder so explainer.py has a field to read. Do not report it as a
confidence anywhere — that is calibration.py's job.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from data_io import Dataset
from schemas import Prediction, SupportingFeature
from splits import GroupedSplit, grouped_split

# Inverse regularization strength. Small C = strong L2 = fewer features carrying
# large weights. AMR feature matrices are wide and sparse relative to the number
# of genomes, so the default C=1.0 overfits happily; 0.1 keeps coefficients
# readable, which matters because they end up in front of a clinician.
DEFAULT_C = 0.1

# How many statistically-associated genes to surface as supporting evidence.
# This is a UI budget, not a statistical threshold.
MAX_STATISTICAL_FEATURES = 3

# Feature names encode point mutations as GENE_SUBSTITUTION, e.g. "gyrA_S83L".
_MUTATION_PATTERN = re.compile(r"^(?P<gene>.+?)_(?P<mutation>[A-Z]\d+[A-Z])$")


class PredictorError(ValueError):
    """Raised when a model is asked for something it cannot honestly answer."""


def _parse_feature_name(name: str) -> tuple[str, str | None]:
    """'gyrA_S83L' -> ('gyrA', 'S83L');  'blaCTX-M-15' -> ('blaCTX-M-15', None)."""
    match = _MUTATION_PATTERN.match(name)
    if match:
        return match.group("gene"), match.group("mutation")
    return name, None


def _known_resistance_genes(drug: str) -> list[str]:
    """
    Curated resistance genes for a drug, from Moncef's lookup.

    Returns [] when drug_database.py has not filled this in yet, which degrades
    every prediction to "statistical_association" — the honest failure mode. We
    would rather under-claim evidence than assert a mechanism nobody curated.
    """
    try:
        import drug_database
    except ImportError:
        return []
    table = getattr(drug_database, "KNOWN_RESISTANCE_GENES", {})
    return list(table.get(drug, []))


# --------------------------------------------------------------------------
# 1. The deterministic target gate
# --------------------------------------------------------------------------


def target_gate(row: pd.Series, target_genes: list[str] | None) -> str:
    """
    Can this drug's target even be found in this genome?

    Returns "present" / "absent" / "unknown", matching
    schemas.Prediction.target_gate_status.

    "absent" requires that we LOOKED for every target gene and at least one came
    back missing. A drug with several targets needs all of them: if
    ciprofloxacin's parC is gone, gyrA alone does not make the drug viable.
    """
    if not target_genes:
        return "unknown"  # drug not in the lookup — no gate to apply

    unscanned = [g for g in target_genes if g not in row.index]
    if unscanned:
        return "unknown"  # annotator never looked; absence of data, not of gene

    if all(row[g] == 1 for g in target_genes):
        return "present"
    return "absent"


# --------------------------------------------------------------------------
# 2. One model per antibiotic
# --------------------------------------------------------------------------


@dataclass
class DrugModel:
    """A fitted logistic regression for a single antibiotic, plus its context."""

    drug: str
    estimator: LogisticRegression
    feature_names: list[str]
    target_genes: list[str] | None
    known_genes: list[str] = field(default_factory=list)
    # Every drug's target genes, not just this one's. Targets are housekeeping
    # genes carried by nearly every genome, so they are near-constant columns:
    # whatever coefficient they pick up is an artefact, and showing "gyrA
    # detected" as evidence for gentamicin is actively misleading. Excluded
    # from evidence only — they stay in the model's feature matrix.
    evidence_exclusions: frozenset[str] = frozenset()
    n_train: int = 0
    n_train_clusters: int = 0

    def probability_resistant(self, row: pd.Series) -> float:
        """P(resistant) for one genome. Uncalibrated."""
        x = row.reindex(self.feature_names).fillna(0).to_numpy(dtype=float)
        return float(self.estimator.predict_proba(x.reshape(1, -1))[0, 1])

    def positive_drivers(self, row: pd.Series) -> list[str]:
        """Features present in this genome that push the model toward resistant."""
        present = [
            (name, coef)
            for name, coef in zip(self.feature_names, self.estimator.coef_[0])
            if coef > 0 and row.get(name, 0) == 1
        ]
        present.sort(key=lambda pair: -pair[1])
        return [name for name, _ in present]


def fit_drug_model(
    dataset: Dataset,
    drug: str,
    train_rows: np.ndarray,
    groups: np.ndarray,
    C: float = DEFAULT_C,
) -> DrugModel:
    """Fit one antibiotic's model on the given training row positions."""
    X, y, _ = dataset.xy_for_drug(drug)
    X_train, y_train = X.iloc[train_rows], y[train_rows]

    if len(np.unique(y_train)) < 2:
        raise PredictorError(
            f"{drug}: training split contains only class {np.unique(y_train)[0]}. "
            "A one-class fit cannot be calibrated or evaluated — either the drug "
            "has too few labels, or the grouped split put every resistant genome "
            "into one cluster."
        )

    # class_weight balanced: resistance prevalence varies wildly by drug, and an
    # unweighted fit on a 5%-resistant drug learns to answer "susceptible" always.
    # L2 is the solver default; naming it explicitly is deprecated as of
    # sklearn 1.8, so the regularization is set through C alone.
    estimator = LogisticRegression(
        C=C,
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
    )
    estimator.fit(X_train.to_numpy(dtype=float), y_train)

    all_targets = {
        gene
        for entry in dataset.drug_targets.values()
        for gene in entry.get("target_genes", [])
    }

    return DrugModel(
        drug=drug,
        estimator=estimator,
        feature_names=list(X.columns),
        target_genes=dataset.target_genes(drug),
        known_genes=_known_resistance_genes(drug),
        evidence_exclusions=frozenset(all_targets),
        n_train=len(train_rows),
        n_train_clusters=len(set(groups[train_rows])),
    )


# --------------------------------------------------------------------------
# 3. Feature matrix + models -> Predictions
# --------------------------------------------------------------------------


@dataclass
class Predictor:
    """All per-antibiotic models for one species."""

    models: dict[str, DrugModel] = field(default_factory=dict)
    splits: dict[str, GroupedSplit] = field(default_factory=dict)

    @classmethod
    def fit(
        cls,
        dataset: Dataset,
        drugs: list[str] | None = None,
        C: float = DEFAULT_C,
        seed: int = 0,
    ) -> Predictor:
        """
        Fit one model per drug, each on its own grouped split.

        Splits are per-drug because coverage is ragged — the genomes tested for
        ciprofloxacin are not the set tested for gentamicin, so one global split
        would not be cluster-disjoint for every drug. Each split is retained so
        calibration.py can reuse the matching calibration slice without refitting.
        """
        predictor = cls()
        for drug in drugs or dataset.drugs:
            _, y, groups = dataset.xy_for_drug(drug)
            split = grouped_split(groups, y=y, seed=seed)
            predictor.models[drug] = fit_drug_model(
                dataset, drug, split.train, groups, C=C
            )
            predictor.splits[drug] = split
        return predictor

    def predict(
        self,
        dataset: Dataset,
        genome_id: str,
        drugs: list[str] | None = None,
    ) -> list[Prediction]:
        """Every antibiotic's prediction for one genome."""
        if genome_id not in dataset.features.index:
            raise PredictorError(f"Unknown genome_id {genome_id!r}")

        row = dataset.features.loc[genome_id]
        species = str(dataset.genomes.loc[genome_id, "species"])
        return [
            self._predict_one(row, genome_id, species, drug)
            for drug in (drugs or sorted(self.models))
        ]

    def _predict_one(
        self,
        row: pd.Series,
        genome_id: str,
        species: str,
        drug: str,
    ) -> Prediction:
        model = self.models.get(drug)
        if model is None:
            raise PredictorError(f"No model fitted for {drug!r}")

        gate = target_gate(row, model.target_genes)
        probability = model.probability_resistant(row)
        known_present = [g for g in model.known_genes if row.get(g, 0) == 1]

        # Evidence tiering. Curated genes outrank the model, always.
        if known_present:
            evidence = "known_gene_or_mutation"
            supporting = [_feature(name, curated=True) for name in known_present]
        else:
            drivers = [
                name
                for name in model.positive_drivers(row)
                if name not in model.evidence_exclusions
            ][:MAX_STATISTICAL_FEATURES]
            if drivers:
                evidence = "statistical_association"
                supporting = [_feature(name, curated=False) for name in drivers]
            else:
                evidence = "no_known_signal"
                supporting = []

        if gate == "absent":
            # Overrides the model entirely — see module docstring, point 1.
            call = "not_applicable"
        else:
            call = "likely_to_fail" if probability >= 0.5 else "likely_to_work"

        return Prediction(
            sample_id=genome_id,
            species=species,
            drug=drug,
            call=call,
            confidence=probability,  # UNCALIBRATED — calibration.py overwrites
            evidence_category=evidence,
            supporting_features=supporting,
            target_gate_status=gate,
            no_call_reason=None,  # set by calibration.py
        )


def _feature(name: str, curated: bool) -> SupportingFeature:
    gene, mutation = _parse_feature_name(name)
    note = (
        "Curated resistance determinant for this drug."
        if curated
        else "Statistical association in training data only — not a confirmed mechanism."
    )
    return SupportingFeature(gene=gene, mutation=mutation, note=note)


if __name__ == "__main__":
    from data_io import load_dataset

    ds = load_dataset(Path(__file__).parent.parent / "data" / "synthetic")
    predictor = Predictor.fit(ds)

    print("Fitted models")
    for drug, model in predictor.models.items():
        gate = "gated" if model.target_genes else "NO GATE (drug absent from lookup)"
        print(
            f"  {drug:<16} n_train={model.n_train:<5} "
            f"clusters={model.n_train_clusters:<4} "
            f"known_genes={len(model.known_genes)}  {gate}"
        )

    sample = ds.features.index[0]
    print(f"\nPredictions for {sample}")
    for prediction in predictor.predict(ds, sample):
        genes = ", ".join(f.gene for f in prediction.supporting_features) or "-"
        print(
            f"  {prediction.drug:<16} {prediction.call:<16} "
            f"p(R)={prediction.confidence:.2f}  "
            f"target={prediction.target_gate_status:<8} "
            f"{prediction.evidence_category:<24} {genes}"
        )
