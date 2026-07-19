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

# Feature names encode point mutations as GENE_CHANGE, e.g. "gyrA_S83L".
#
# The change token is wider than a plain substitution. Real AMRFinderPlus output
# for E. coli also uses stops ("cirA_Q56Ter"), frameshifts ("acrR_V29YfsTer44"),
# in-frame insertions ("ftsI_I336IKYRI"), and promoter positions that are
# negative and nucleotide-level ("ampC_T-32A", "blaTEMp_C32T"). So the token is
# an uppercase letter, an optionally-negative position, then any trailing
# letters/digits.
#
# The earlier pattern was `[A-Z]\d+[A-Z]` — substitutions only. It silently
# failed to split 20 of the 41 mutation features in the real matrix, leaving
# gene="cirA_Q56Ter" with mutation=None. That is not just cosmetic: evidence
# exclusions and KNOWN_RESISTANCE_GENES both match on the parsed gene name, so
# an unsplit feature is never recognised as belonging to its gene.
#
# Verified against all 124 real features: 41/41 mutations split correctly and
# none of the 83 acquired gene names are falsely split. Pinned by
# tests/test_predictor.py.
_MUTATION_PATTERN = re.compile(r"^(?P<gene>.+?)_(?P<mutation>[A-Z]-?\d+[A-Za-z0-9]*)$")


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
    #: {allele: family} when the model was fitted on aggregated features.
    #: `feature_names` then holds FAMILY names, and any raw genome row must be
    #: aggregated through this mapping before it can be scored.
    family_map: dict[str, str] | None = None

    def probability_resistant(self, row: pd.Series) -> float:
        """P(resistant) for one genome. Uncalibrated."""
        if self.family_map is not None:
            x = _row_to_families(row, self.family_map, self.feature_names).to_numpy(
                dtype=float
            )
        else:
            x = row.reindex(self.feature_names).fillna(0).to_numpy(dtype=float)
        return float(self.estimator.predict_proba(x.reshape(1, -1))[0, 1])

    def positive_drivers(self, row: pd.Series) -> list[str]:
        """
        Alleles present in this genome that push the model toward resistant.

        Returns ALLELE-level names even when the model was fitted on families.
        The coefficient belongs to the family, but the evidence a human needs is
        the specific variant that was detected — "blaCTX-M-15", not "blaCTX-M".
        Alleles inherit their family's coefficient for ranking, which is exactly
        what the model actually learned.
        """
        if self.family_map is None:
            present = [
                (name, coef)
                for name, coef in zip(self.feature_names, self.estimator.coef_[0])
                if coef > 0 and row.get(name, 0) == 1
            ]
        else:
            coefficients = dict(zip(self.feature_names, self.estimator.coef_[0]))
            present = [
                (allele, coefficients[family])
                for allele, family in self.family_map.items()
                if coefficients.get(family, 0.0) > 0 and row.get(allele, 0) == 1
            ]
        present.sort(key=lambda pair: -pair[1])
        return [name for name, _ in present]


#: Trailing allele designation on an acquired gene: "blaCTX-M-15" -> "blaCTX-M",
#: "dfrA17" -> "dfrA", "qnrS1" -> "qnrS". Anchored to the end so the interior
#: hyphens and digits that are part of the family name itself survive.
_ALLELE_SUFFIX = re.compile(r"-?\d+$")


def gene_family(feature_name: str) -> str:
    """
    Collapse a feature to the gene family that carries its resistance mechanism.

    "blaTEM-1", "blaTEM-12", "blaTEM-30" -> "blaTEM";  "gyrA_S83L" -> "gyrA".

    Why the model needs this: AMR resistance here is spread across many rare
    allelic variants rather than a few common genes. In the real E. coli matrix
    the eight dfrA alleles sit in one or two genomes each, so as separate columns
    they are near-singletons that 33 training rows cannot learn from — which is
    what forces regularization heavy enough to compress every probability toward
    0.5. Aggregated, dfrA is one feature present in 42 of 119 genomes.

    Filtering by prevalence instead would delete the biology: at a >=3-occurrence
    threshold only 1 of 25 curated ampicillin genes and 1 of 8 trimethoprim genes
    survive. Aggregation keeps every signal and merely stops splitting it.

    The justification is a priori, not selected on test performance: allelic
    variants of a gene confer the same resistance, which is how clinical AMR
    interpretation already reasons ("an ESBL is present", not "blaCTX-M-15 is
    present"). Measured across 8 grouped seeds it widens probability spread
    (0.318->0.364, 0.421->0.500, 0.369->0.440) at an AUROC cost within noise.

    Used for MODELLING ONLY. Evidence shown to a human stays allele-level — see
    DrugModel.positive_drivers — because "blaCTX-M-15 detected" is the useful
    statement and "blaCTX-M detected" throws away the identification.

    OFF BY DEFAULT (fit_drug_model(aggregate_families=False)). The reasoning
    above is sound and the spread gain is real, but measured end-to-end over 8
    grouped seeds on the real data it does not pay for itself:

        drug            AUROC           Brier           bal_acc
        Ampicillin      0.894 -> 0.850  0.170 -> 0.180  0.750 -> 0.625
        Ciprofloxacin   0.859 -> 0.833  0.116 -> 0.125  0.738 -> 0.625
        Trimethoprim    0.940 -> 0.948  0.127 -> 0.105  0.844 -> 0.896

    It helps trimethoprim on every metric and hurts the other two on every
    metric. It also roughly doubles synthetic Ciprofloxacin's raw Brier
    (0.1559 -> 0.2793), because synth_data.py plants its signal in SPECIFIC
    alleles, so merging gyrA_S83L with gyrA_D87N destroys what the fixture
    encoded — a reminder that the synthetic set is not biologically faithful.

    Kept rather than deleted because the mechanism is real and the trimethoprim
    result suggests it may become correct at larger n, where each family has
    enough members to beat the information it discards. Re-measure before
    turning it on; do not enable it per-drug on the strength of the table above,
    which would be selecting a model on test results.
    """
    gene, mutation = _parse_feature_name(feature_name)
    if mutation is not None:
        return gene
    return _ALLELE_SUFFIX.sub("", feature_name) or feature_name


def aggregate_to_families(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Collapse a genome x allele matrix to genome x family, presence-wise.

    Returns the aggregated frame and the {allele: family} mapping used, so a
    single row can be aggregated the same way at prediction time.
    """
    mapping = {column: gene_family(column) for column in X.columns}
    aggregated = X.T.groupby(pd.Series(mapping), sort=True).max().T
    return aggregated, mapping


def _row_to_families(
    row: pd.Series, mapping: dict[str, str], family_names: list[str]
) -> pd.Series:
    """Aggregate one genome's allele row into the model's family feature space."""
    present = pd.Series(0, index=family_names, dtype=float)
    for allele, value in row.items():
        family = mapping.get(allele)
        if family in present.index and value == 1:
            present[family] = 1.0
    return present


def cluster_sample_weights(groups: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Per-row weights that de-duplicate near-identical genomes and balance classes.

    Grouped splitting stops leakage but does nothing about redundancy WITHIN
    train. BV-BRC is full of outbreak isolates and resequenced strains, so a
    single clonal expansion can dominate the fit: on the synthetic set the top
    three clusters are ~33% of training rows for every drug. The model then
    learns that one lineage's quirks and calls it a resistance mechanism.

    The brief asks for de-duplication. Dropping rows is the crude version — it
    throws away the real within-cluster variation that does exist. Weighting
    each genome by 1/(size of its cluster) instead makes every CLUSTER count
    once regardless of how many times it was sequenced, while keeping every row.

    Class balance is then applied on top of those weights, not before: sklearn's
    class_weight="balanced" is computed from raw class counts and would be
    wrong once rows carry unequal weight. Returns weights summing to len(y).
    """
    _, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]

    # Equalize total weight per class, so a rare resistant class still moves the
    # loss as much as the majority does.
    for c in np.unique(y):
        mask = y == c
        weights[mask] *= len(y) / (2.0 * weights[mask].sum())

    return weights * len(y) / weights.sum()


def fit_drug_model(
    dataset: Dataset,
    drug: str,
    train_rows: np.ndarray,
    groups: np.ndarray,
    C: float = DEFAULT_C,
    weight_by_cluster: bool = True,
    aggregate_families: bool = False,
) -> DrugModel:
    """Fit one antibiotic's model on the given training row positions."""
    X, y, _ = dataset.xy_for_drug(drug)

    # Collapse allelic variants into gene families before fitting — see
    # gene_family(). The mapping is a pure function of the column name, so doing
    # it over the whole matrix leaks nothing: no label, no row, and no split
    # membership is consulted. It stays inside the fit path so the production
    # model and the out-of-fold fold models in calibration.py cannot drift into
    # different feature spaces.
    family_map: dict[str, str] | None = None
    if aggregate_families:
        X, family_map = aggregate_to_families(X)

    X_train, y_train = X.iloc[train_rows], y[train_rows]

    if len(np.unique(y_train)) < 2:
        raise PredictorError(
            f"{drug}: training split contains only class {np.unique(y_train)[0]}. "
            "A one-class fit cannot be calibrated or evaluated — either the drug "
            "has too few labels, or the grouped split put every resistant genome "
            "into one cluster."
        )

    # Weighting: cluster_sample_weights() handles both de-duplication and class
    # balance, so class_weight is left off to avoid applying the latter twice.
    # Without it, fall back to class_weight="balanced" — resistance prevalence
    # varies wildly by drug, and an unweighted fit on a 5%-resistant drug learns
    # to answer "susceptible" always.
    # L2 is the solver default; naming it explicitly is deprecated as of
    # sklearn 1.8, so the regularization is set through C alone.
    sample_weight = None
    if weight_by_cluster:
        sample_weight = cluster_sample_weights(groups[train_rows], y_train)

    estimator = LogisticRegression(
        C=C,
        class_weight=None if weight_by_cluster else "balanced",
        solver="liblinear",
        max_iter=2000,
    )
    estimator.fit(X_train.to_numpy(dtype=float), y_train, sample_weight=sample_weight)

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
        family_map=family_map,
    )


# --------------------------------------------------------------------------
# 3. Feature matrix + models -> Predictions
# --------------------------------------------------------------------------


@dataclass
class Predictor:
    """All per-antibiotic models for one species."""

    models: dict[str, DrugModel] = field(default_factory=dict)
    splits: dict[str, GroupedSplit] = field(default_factory=dict)
    # Retained so calibration.py can refit fold models with the SAME
    # hyperparameters. Out-of-fold probabilities calibrate the production model
    # only if they came from an identically-configured one.
    C: float = DEFAULT_C
    weight_by_cluster: bool = True

    @classmethod
    def fit(
        cls,
        dataset: Dataset,
        drugs: list[str] | None = None,
        C: float = DEFAULT_C,
        seed: int = 0,
        weight_by_cluster: bool = True,
    ) -> Predictor:
        """
        Fit one model per drug, each on its own grouped split.

        Splits are per-drug because coverage is ragged — the genomes tested for
        ciprofloxacin are not the set tested for gentamicin, so one global split
        would not be cluster-disjoint for every drug. Each split is retained so
        calibration.py can reuse the matching calibration slice without refitting.

        weight_by_cluster de-duplicates within the training slice — see
        cluster_sample_weights(). Turn it off only to reproduce the unweighted
        baseline for comparison; it should stay on for anything reported.
        """
        predictor = cls(C=C, weight_by_cluster=weight_by_cluster)
        for drug in drugs or dataset.drugs:
            _, y, groups = dataset.xy_for_drug(drug)
            split = grouped_split(groups, y=y, seed=seed)
            predictor.models[drug] = fit_drug_model(
                dataset, drug, split.train, groups, C=C,
                weight_by_cluster=weight_by_cluster,
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
