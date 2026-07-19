"""
data_io.py — loading + validation of Module 01 inputs
Owner: Waji (pipeline)

Implements the input side of the interface: see docs/DATA_CONTRACT.md.
schemas.py defines what the pipeline emits; this defines what it consumes.

Everything downstream (splits, predictor, calibration) reads a Dataset produced
here, so synthetic and real data are fully interchangeable.

Label convention: positive class (y=1) is RESISTANT ("likely to fail").
Resistance is the event being detected, so recall/PR-AUC on the positive class
is the number the brief actually cares about.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RESISTANT = 1
SUSCEPTIBLE = 0

_PHENOTYPE_TO_Y = {"R": RESISTANT, "S": SUSCEPTIBLE}


class ContractViolation(ValueError):
    """Raised when input data does not match docs/DATA_CONTRACT.md."""


@dataclass
class Dataset:
    """A validated bundle of the four contract files."""

    features: pd.DataFrame          # index=genome_id, binary AMR columns
    labels: pd.DataFrame            # long: genome_id, drug, phenotype, [mic]
    genomes: pd.DataFrame           # index=genome_id, species, cluster_id
    drug_targets: dict[str, dict]   # drug -> {"target_genes": [...], "note": ...}

    @property
    def drugs(self) -> list[str]:
        return sorted(self.labels["drug"].unique())

    @property
    def feature_names(self) -> list[str]:
        return list(self.features.columns)

    def xy_for_drug(self, drug: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Training data for one antibiotic.

        Returns (X, y, groups) aligned row-for-row, restricted to genomes that
        were actually tested against this drug. `groups` carries cluster_id and
        must be passed to every split and every cross-validation call — that is
        what prevents near-identical genomes leaking across train and test.
        """
        rows = self.labels[self.labels["drug"] == drug]
        if rows.empty:
            raise ContractViolation(f"No labels found for drug {drug!r}")

        ids = rows["genome_id"].to_numpy()
        X = self.features.loc[ids]
        y = rows["phenotype"].map(_PHENOTYPE_TO_Y).to_numpy()
        groups = self.genomes.loc[ids, "cluster_id"].to_numpy()
        return X, y, groups

    def target_genes(self, drug: str) -> list[str] | None:
        """Target genes for a drug, or None if the drug is not in the lookup."""
        entry = self.drug_targets.get(drug)
        return list(entry["target_genes"]) if entry else None


def load_dataset(data_dir: str | Path) -> Dataset:
    """Load and validate the four contract files from a directory."""
    data_dir = Path(data_dir)

    # genome_id is ALWAYS a string, never inferred. BV-BRC ids look numeric
    # ("562.100145"), so pandas types the column float64 and silently drops
    # trailing zeros: "562.65180" becomes 562.6518 and no longer matches its own
    # label row. 228 of the 2,154 ids in data/genome_id_list.csv (10.6%) are
    # affected. Synthetic fixtures use "GEN-0001" and never exposed this.
    features = pd.read_csv(
        data_dir / "features.csv", index_col="genome_id", dtype={"genome_id": str}
    )
    labels = pd.read_csv(data_dir / "labels.csv", dtype={"genome_id": str})
    genomes = pd.read_csv(
        data_dir / "genomes.csv", index_col="genome_id", dtype={"genome_id": str}
    )

    targets_path = data_dir / "drug_targets.json"
    drug_targets = json.loads(targets_path.read_text()) if targets_path.exists() else {}

    dataset = Dataset(features, labels, genomes, drug_targets)
    validate(dataset)
    return dataset


def validate(dataset: Dataset) -> None:
    """Fail loudly on contract violations rather than training a broken model."""
    features, labels, genomes = dataset.features, dataset.labels, dataset.genomes

    if features.index.has_duplicates:
        dupes = features.index[features.index.duplicated()].unique().tolist()
        raise ContractViolation(f"Duplicate genome_id in features.csv: {dupes[:5]}")

    if features.isna().any().any():
        bad = features.columns[features.isna().any()].tolist()
        raise ContractViolation(f"features.csv has missing values in: {bad[:5]}")

    non_binary = [c for c in features.columns if not set(features[c].unique()) <= {0, 1}]
    if non_binary:
        raise ContractViolation(
            f"features.csv must be binary 0/1; non-binary columns: {non_binary[:5]}"
        )

    missing_cols = {"genome_id", "drug", "phenotype"} - set(labels.columns)
    if missing_cols:
        raise ContractViolation(f"labels.csv is missing columns: {sorted(missing_cols)}")

    bad_phenotypes = set(labels["phenotype"].unique()) - set(_PHENOTYPE_TO_Y)
    if bad_phenotypes:
        raise ContractViolation(
            f"labels.csv phenotype must be R or S; found {sorted(bad_phenotypes)}. "
            "Intermediate 'I' should be resolved or dropped in Module 01."
        )

    unknown = set(labels["genome_id"]) - set(features.index)
    if unknown:
        raise ContractViolation(
            f"labels.csv references genome_ids absent from features.csv: {sorted(unknown)[:5]}"
        )

    if "cluster_id" not in genomes.columns:
        raise ContractViolation(
            "genomes.csv needs a cluster_id column. Use organizer cluster IDs if provided, "
            "otherwise generate them from FASTA with src/splits.py:cluster_genomes()."
        )

    ungrouped = set(features.index) - set(genomes.index[genomes["cluster_id"].notna()])
    if ungrouped:
        raise ContractViolation(
            f"{len(ungrouped)} genomes have no cluster_id, so they cannot be split safely: "
            f"{sorted(ungrouped)[:5]}"
        )


def summarize(dataset: Dataset) -> pd.DataFrame:
    """Per-drug counts and class balance — the first thing to look at on new data."""
    rows = []
    for drug in dataset.drugs:
        sub = dataset.labels[dataset.labels["drug"] == drug]
        n_r = int((sub["phenotype"] == "R").sum())
        rows.append(
            {
                "drug": drug,
                "n": len(sub),
                "n_resistant": n_r,
                "n_susceptible": len(sub) - n_r,
                "pct_resistant": round(100 * n_r / len(sub), 1),
                "has_target_gate": drug in dataset.drug_targets,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    ds = load_dataset(Path(__file__).parent.parent / "data" / "synthetic")
    print(f"{len(ds.features)} genomes, {len(ds.feature_names)} features, "
          f"{ds.genomes['cluster_id'].nunique()} clusters\n")
    print(summarize(ds).to_string(index=False))
