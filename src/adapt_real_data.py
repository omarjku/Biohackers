"""
adapt_real_data.py — Module 01 real outputs -> the four contract files
Owner: Waji (pipeline)

Moncef's AMRFinderPlus run produced `data/raw/files.zip` (feature matrix, gene
metadata, target-gate results, a label subset) plus `data/raw/genome_clusters.csv`.
Those are real and correct, but they are not in the shape `data_io.load_dataset()`
validates against — different column names, different phenotype encoding, drug
names in a different case, clusters in a separate file.

This module does that translation and nothing else. It does not recompute
features and it does not touch the biology; if a number is wrong here it is
wrong upstream in genome_reader.py.

Run:  python src/adapt_real_data.py
Out:  data/processed/{features,labels,genomes}.csv + drug_targets.json
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw"
OUT_DIR = REPO_ROOT / "data" / "processed"

SPECIES = "Escherichia coli"

# Drug names arrive in two cases: the BV-BRC label pulls use lowercase
# ("ciprofloxacin"), drug_database.DRUG_TARGET_MAP uses title case
# ("Ciprofloxacin"). Title case is canonical — it is what DATA_CONTRACT.md
# and drug_database.py already use. Normalising here is what stops the
# target-gate lookup silently missing on every row.
def canonical_drug(name: str) -> str:
    return str(name).strip().title()


_PHENOTYPE_MAP = {
    "resistant": "R",
    "susceptible": "S",
}


def _read_zip_csvs(zip_path: Path) -> dict[str, pd.DataFrame]:
    with zipfile.ZipFile(zip_path) as z:
        return {
            Path(n).stem: pd.read_csv(io.BytesIO(z.read(n)))
            for n in z.namelist()
            if n.endswith(".csv")
        }


def build_features(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix -> contract shape: genome_id index, strictly binary."""
    features = feature_matrix.copy()
    features["genome_id"] = features["genome_id"].astype(str)
    features = features.set_index("genome_id")
    features = features.fillna(0).astype(int)
    return features.sort_index()


def build_labels(labels_sampled: pd.DataFrame, keep_genomes: set[str]) -> pd.DataFrame:
    """
    Long-format labels for the genomes we have features for.

    Source is `data/labels_sampled.csv`, NOT the `labels_final_urgent.csv` inside
    files.zip. The latter is trimmed to exactly 40 rows per drug; the sampled file
    carries 150 valid rows for these same genomes and every dropped row has a real
    lab_method (Broth dilution / Disk diffusion). The trim looks like rounding, not
    quality control, so we take the wider set.

    Rows with a blank lab_method are dropped — CLAUDE.md requires
    laboratory-measured results only, and a blank method cannot be confirmed as one.
    Intermediate phenotypes are dropped rather than guessed (DATA_CONTRACT.md §2).
    """
    labels = labels_sampled.copy()
    labels["genome_id"] = labels["genome_id"].astype(str)
    labels = labels[labels["genome_id"].isin(keep_genomes)]

    blank_method = labels["lab_method"].isna() | (
        labels["lab_method"].astype(str).str.strip() == ""
    )
    labels = labels[~blank_method]

    labels["drug"] = labels["antibiotic"].map(canonical_drug)
    labels["phenotype"] = (
        labels["phenotype"].astype(str).str.strip().str.lower().map(_PHENOTYPE_MAP)
    )
    labels = labels[labels["phenotype"].notna()]

    labels = labels.drop_duplicates(subset=["genome_id", "drug"])
    return labels[["genome_id", "drug", "phenotype", "lab_method"]].sort_values(
        ["genome_id", "drug"]
    ).reset_index(drop=True)


def build_genomes(clusters: pd.DataFrame, keep_genomes: set[str]) -> pd.DataFrame:
    """genome_id -> species + cluster_id, the grouping that drives every split."""
    genomes = clusters.copy()
    genomes["genome_id"] = genomes["genome_id"].astype(str)
    genomes = genomes[genomes["genome_id"].isin(keep_genomes)]
    genomes["species"] = SPECIES
    genomes = genomes.set_index("genome_id")
    return genomes[["species", "cluster_id"]].sort_index()


def build_drug_targets(drugs: list[str]) -> dict[str, dict]:
    """
    DRUG_TARGET_MAP (comma-joined string) -> contract JSON (list per drug).

    Only drugs we actually have labels for are emitted. A drug absent from the
    output gets target_gate_status="unknown" downstream, which is the correct
    honest default — absence of a lookup entry is not absence of a gene.
    """
    from drug_database import DRUG_TARGET_MAP

    normalised = {canonical_drug(k): v for k, v in DRUG_TARGET_MAP.items()}
    targets: dict[str, dict] = {}
    for drug in drugs:
        raw = normalised.get(drug)
        if not raw:
            continue
        genes = [g.strip() for g in str(raw).split(",") if g.strip()]
        if genes:
            targets[drug] = {"target_genes": genes}
    return targets


def adapt(raw_dir: Path = RAW_DIR, out_dir: Path = OUT_DIR) -> dict[str, Path]:
    csvs = _read_zip_csvs(raw_dir / "files.zip")
    feature_matrix = csvs["feature_matrix_real"]
    clusters = pd.read_csv(raw_dir / "genome_clusters.csv")
    labels_sampled = pd.read_csv(
        REPO_ROOT / "data" / "labels_sampled.csv", encoding="utf-8-sig"
    )

    features = build_features(feature_matrix)
    keep = set(features.index)
    labels = build_labels(labels_sampled, keep)
    genomes = build_genomes(clusters, keep)
    drug_targets = build_drug_targets(sorted(labels["drug"].unique()))

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "features": out_dir / "features.csv",
        "labels": out_dir / "labels.csv",
        "genomes": out_dir / "genomes.csv",
        "drug_targets": out_dir / "drug_targets.json",
    }
    features.to_csv(paths["features"])
    labels.to_csv(paths["labels"], index=False)
    genomes.to_csv(paths["genomes"])
    paths["drug_targets"].write_text(json.dumps(drug_targets, indent=2))
    return paths


def _report(out_dir: Path) -> None:
    """Print what came out, and flag anything that makes the data unusable as-is."""
    import data_io

    ds = data_io.load_dataset(out_dir)
    n_clusters = ds.genomes["cluster_id"].nunique()
    print(
        f"{len(ds.features)} genomes, {len(ds.feature_names)} features, "
        f"{n_clusters} clusters\n"
    )
    print(data_io.summarize(ds).to_string(index=False))

    sizes = ds.genomes["cluster_id"].value_counts()
    print(f"\ncluster sizes: {sizes.to_dict()}")
    if n_clusters < 10:
        print(
            f"\nWARNING: {n_clusters} clusters is too few for a grouped "
            "train/calibration/test split — there are not enough independent "
            "groups to fill three splits. Re-cluster from the FASTA files with "
            "splits.py at a threshold that actually separates E. coli "
            "(Mash 0.05 / ~95% ANI over-merges a single species)."
        )


if __name__ == "__main__":
    paths = adapt()
    print(f"wrote {len(paths)} files to {OUT_DIR.relative_to(REPO_ROOT)}\n")
    _report(OUT_DIR)
