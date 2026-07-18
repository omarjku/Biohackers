"""
Run with: pytest tests/test_predictor.py -v

Covers the three things predictor.py's docstring says must not be "simplified"
away — the target gate running ahead of the model, a missing feature column
meaning "unknown" rather than "absent", and evidence never being promoted to
known_gene_or_mutation on the strength of a coefficient — plus the per-cluster
de-duplication weights.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_io import load_dataset
from predictor import (
    Predictor,
    PredictorError,
    cluster_sample_weights,
    target_gate,
)

DATA_DIR = Path(__file__).parent.parent / "data" / "synthetic"


@pytest.fixture(scope="module")
def dataset():
    return load_dataset(DATA_DIR)


@pytest.fixture(scope="module")
def predictor(dataset):
    return Predictor.fit(dataset)


# --------------------------------------------------------------------------
# De-duplication weights
# --------------------------------------------------------------------------


def test_each_cluster_gets_equal_total_weight():
    """
    The point of the weights: one clonal expansion sequenced 50 times must not
    outvote 50 distinct lineages. Within a class, every cluster's rows should
    sum to the same total weight regardless of cluster size.
    """
    groups = np.array(["A"] * 50 + ["B"] * 5 + ["C"] * 1)
    y = np.zeros(56, dtype=int)
    y[:28] = 1  # both classes present, spread across the big cluster

    weights = cluster_sample_weights(groups, y)
    totals = {g: weights[groups == g].sum() for g in ("A", "B", "C")}
    # A carries both classes so it gets two clusters' worth; B and C one each.
    assert totals["B"] == pytest.approx(totals["C"], rel=1e-6)
    assert totals["A"] > totals["B"]


def test_big_cluster_rows_are_downweighted_individually():
    groups = np.array(["big"] * 40 + ["small"] * 2)
    y = np.array([1] * 20 + [0] * 20 + [1, 0])
    weights = cluster_sample_weights(groups, y)
    assert weights[groups == "big"].max() < weights[groups == "small"].min()


def test_weights_are_class_balanced():
    """Total weight per class must be equal, or a rare resistant class is ignored."""
    groups = np.array([f"CL-{i // 3}" for i in range(60)])
    y = np.array([1] * 6 + [0] * 54)  # 10% resistant
    weights = cluster_sample_weights(groups, y)
    assert weights[y == 1].sum() == pytest.approx(weights[y == 0].sum(), rel=1e-6)


def test_weights_sum_to_row_count():
    """Keeps C comparable between weighted and unweighted fits."""
    groups = np.array([f"CL-{i // 4}" for i in range(40)])
    y = np.array([1, 0] * 20)
    weights = cluster_sample_weights(groups, y)
    assert weights.sum() == pytest.approx(len(y))


def test_uniform_clusters_give_uniform_weights():
    """With no redundancy and balanced classes, weighting must be a no-op."""
    groups = np.array([f"CL-{i}" for i in range(20)])
    y = np.array([1, 0] * 10)
    weights = cluster_sample_weights(groups, y)
    assert weights == pytest.approx(np.ones(20))


def test_weighting_changes_the_fit(dataset):
    """A flag that silently does nothing is worse than no flag."""
    weighted = Predictor.fit(dataset, drugs=["Gentamicin"], weight_by_cluster=True)
    plain = Predictor.fit(dataset, drugs=["Gentamicin"], weight_by_cluster=False)
    assert not np.allclose(
        weighted.models["Gentamicin"].estimator.coef_,
        plain.models["Gentamicin"].estimator.coef_,
    )


# --------------------------------------------------------------------------
# The target gate — deterministic, and ahead of the model
# --------------------------------------------------------------------------


def test_missing_column_is_unknown_not_absent():
    """
    Absence of data is not absence of gene. If the annotator never looked for
    ftsI, conflating that with "ftsI is gone" manufactures not_applicable calls
    out of incomplete annotation.
    """
    row = pd.Series({"blaCTX-M-15": 1})
    assert target_gate(row, ["ftsI"]) == "unknown"


def test_gate_absent_only_when_all_targets_were_scanned():
    row = pd.Series({"gyrA": 1, "parC": 0})
    assert target_gate(row, ["gyrA", "parC"]) == "absent"
    assert target_gate(row, ["gyrA"]) == "present"
    # parC scanned but gyrB never looked for -> unknown, not absent
    assert target_gate(row, ["gyrA", "parC", "gyrB"]) == "unknown"


def test_gate_needs_every_target_present():
    """One surviving target does not make the drug viable."""
    row = pd.Series({"gyrA": 1, "parC": 0})
    assert target_gate(row, ["gyrA", "parC"]) == "absent"


def test_no_targets_means_no_gate():
    row = pd.Series({"gyrA": 1})
    assert target_gate(row, None) == "unknown"
    assert target_gate(row, []) == "unknown"


def test_absent_target_overrides_the_model(dataset, predictor):
    """
    The gate runs before the model and wins. Never report "likely to work" from
    absence of resistance markers when the drug has nothing to bind to.
    """
    genome_id = dataset.features.index[0]
    row = dataset.features.loc[genome_id].copy()
    drug = next(d for d in predictor.models if predictor.models[d].target_genes)
    for gene in predictor.models[drug].target_genes:
        row[gene] = 0

    species = str(dataset.genomes.loc[genome_id, "species"])
    prediction = predictor._predict_one(row, genome_id, species, drug)
    assert prediction.target_gate_status == "absent"
    assert prediction.call == "not_applicable"


# --------------------------------------------------------------------------
# Evidence tiering
# --------------------------------------------------------------------------


def test_evidence_never_promoted_without_curated_genes(dataset, predictor):
    """
    drug_database.KNOWN_RESISTANCE_GENES is still empty, so every prediction
    must degrade to statistical_association or no_known_signal. A large model
    coefficient is correlation — possibly a hitchhiker gene — and must never be
    dressed up as a known mechanism.
    """
    for genome_id in list(dataset.features.index)[:40]:
        for prediction in predictor.predict(dataset, genome_id):
            if prediction.evidence_category == "known_gene_or_mutation":
                assert prediction.supporting_features
                curated = predictor.models[prediction.drug].known_genes
                assert curated, (
                    f"{prediction.drug}: claimed known_gene_or_mutation with no "
                    "curated gene list — evidence was promoted by coefficient"
                )


def _all_target_genes(dataset) -> set[str]:
    return {
        gene
        for entry in dataset.drug_targets.values()
        for gene in entry.get("target_genes", [])
    }


def test_bare_target_gene_presence_excluded_from_evidence(dataset, predictor):
    """
    Target genes are near-universal housekeeping genes, so a coefficient on their
    mere PRESENCE is an artefact. Showing "gyrA detected" as evidence for
    gentamicin is misleading and must never surface.
    """
    targets = _all_target_genes(dataset)
    for genome_id in list(dataset.features.index)[:40]:
        for prediction in predictor.predict(dataset, genome_id):
            bare = {
                f.gene for f in prediction.supporting_features if f.mutation is None
            }
            assert not (bare & targets), (
                f"{prediction.drug} surfaced bare presence of a target gene: "
                f"{sorted(bare & targets)}"
            )


def test_target_gene_mutations_are_still_allowed_as_evidence(dataset, predictor):
    """
    The exclusion is on presence, NOT on point mutations within a target gene.
    parC S80I and gyrA S83L are the canonical fluoroquinolone resistance
    mechanisms — suppressing them would hide the best evidence we have. This
    test exists because a previous version of the sibling test above matched on
    parsed gene name and would have forced exactly that mistake.
    """
    targets = _all_target_genes(dataset)
    found = set()
    for genome_id in list(dataset.features.index)[:60]:
        for prediction in predictor.predict(dataset, genome_id):
            for feature in prediction.supporting_features:
                if feature.gene in targets and feature.mutation:
                    found.add(f"{feature.gene}_{feature.mutation}")
    assert found, (
        "no target-gene mutation ever surfaced as evidence — the exclusion is "
        "probably matching on gene name instead of raw feature name"
    )


def test_mutation_names_are_split(dataset, predictor):
    """'gyrA_S83L' must render as gene gyrA, mutation S83L."""
    seen = False
    for genome_id in list(dataset.features.index)[:60]:
        for prediction in predictor.predict(dataset, genome_id):
            for feature in prediction.supporting_features:
                assert "_" not in feature.gene or feature.mutation is None
                if feature.mutation:
                    seen = True
                    assert feature.mutation[0].isalpha()
                    assert feature.mutation[-1].isalpha()
    assert seen, "fixture no longer exercises the mutation-parsing path"


# --------------------------------------------------------------------------
# Failing loudly
# --------------------------------------------------------------------------


def test_unknown_genome_raises(dataset, predictor):
    with pytest.raises(PredictorError, match="Unknown genome_id"):
        predictor.predict(dataset, "NOT-A-GENOME")


def test_unfitted_drug_raises(dataset, predictor):
    with pytest.raises(PredictorError, match="No model fitted"):
        predictor.predict(dataset, dataset.features.index[0], drugs=["Nonexistentmycin"])


def test_every_prediction_validates_against_the_schema(dataset, predictor):
    """Prediction is a pydantic model — construction already enforces the contract."""
    for prediction in predictor.predict(dataset, dataset.features.index[0]):
        assert prediction.call in (
            "likely_to_work",
            "likely_to_fail",
            "no_call",
            "not_applicable",
        )
        assert 0.0 <= prediction.confidence <= 1.0
        assert prediction.target_gate_status in ("present", "absent", "unknown")
