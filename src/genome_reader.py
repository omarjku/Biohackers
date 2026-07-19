"""
genome_reader.py — Module 01: FASTA -> features
Owner: Moncef (biology)
Must produce, per genome: AMR gene/mutation presence (binary feature matrix)
via AMRFinderPlus, feeding into predictor.py.
See schemas.py for the Prediction shape this eventually feeds into
(specifically: supporting_features, and — via drug_database.py — target_gate_status).
"""
import subprocess
from pathlib import Path

import pandas as pd

from drug_database import DRUG_TARGET_MAP  # TODO(Moncef): confirm import path matches repo layout


# ---------------------------------------------------------------------
# Step 1: run AMRFinderPlus on one genome
# ---------------------------------------------------------------------
def run_amrfinder(fasta_path: Path, output_dir: Path, organism: str = "Escherichia") -> Path:
    """Run AMRFinderPlus on a single genome FASTA, return path to its result TSV."""
    genome_id = fasta_path.stem
    out_path = output_dir / f"{genome_id}.amrfinder.tsv"
    subprocess.run(
        ["amrfinder", "-n", str(fasta_path), "--organism", organism,
         "-o", str(out_path), "--threads", "2"],
        check=True, capture_output=True, text=True,
    )
    return out_path


# ---------------------------------------------------------------------
# Step 2: parse one AMRFinderPlus TSV into (genes_found, evidence_by_gene)
# Column names match AMRFinderPlus v4.2.7 output (confirmed against real
# results - see gene_metadata_real.csv for reference values).
# ---------------------------------------------------------------------
def parse_amrfinder_result(tsv_path: Path) -> dict:
    """
    Returns:
        {
            "genes_found": set of gene/mutation names detected,
            "evidence": {gene_name: "acquired_gene" | "point_mutation" | "disrupted_gene"},
        }
    """
    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    df = df[df["Type"].str.upper() == "AMR"]

    genes_found = set()
    evidence = {}
    for _, row in df.iterrows():
        gene = str(row["Element symbol"]).strip()
        subtype = str(row.get("Subtype", "")).strip().upper()
        genes_found.add(gene)
        if subtype == "POINT_DISRUPT":
            evidence[gene] = "disrupted_gene"
        elif subtype == "POINT":
            evidence[gene] = "point_mutation"
        else:
            evidence[gene] = "acquired_gene"

    return {"genes_found": genes_found, "evidence": evidence}


# ---------------------------------------------------------------------
# Step 3: build the full binary feature matrix across many genomes
# ---------------------------------------------------------------------
def build_feature_matrix(amrfinder_results_dir: Path) -> tuple[pd.DataFrame, dict]:
    """
    Returns:
        matrix_df: genomes x features binary DataFrame, index = genome_id
        evidence_by_genome: {genome_id: {gene_name: evidence_type}}
    """
    tsv_files = sorted(amrfinder_results_dir.glob("*.amrfinder.tsv"))
    per_genome = {}
    evidence_by_genome = {}
    all_features = set()

    for f in tsv_files:
        genome_id = f.stem.replace(".amrfinder", "")
        parsed = parse_amrfinder_result(f)
        per_genome[genome_id] = parsed["genes_found"]
        evidence_by_genome[genome_id] = parsed["evidence"]
        all_features |= parsed["genes_found"]

    all_features = sorted(all_features)
    matrix_df = pd.DataFrame(0, index=sorted(per_genome.keys()), columns=all_features, dtype=int)
    for genome_id, genes in per_genome.items():
        matrix_df.loc[genome_id, list(genes)] = 1
    matrix_df.index.name = "genome_id"

    return matrix_df, evidence_by_genome


# ---------------------------------------------------------------------
# Step 4: deterministic target gate — checks if a drug's target is disrupted
# ---------------------------------------------------------------------
def check_target_gate(genome_id: str, antibiotic: str, evidence_by_genome: dict) -> str:
    """
    Returns target_gate_status: "present" or "absent".
    "absent" = the drug's target gene(s) were found DISRUPTED in this genome
    (AMRFinderPlus POINT_DISRUPT) - NOT simply "not mentioned in the hit list",
    since AMRFinderPlus only reports genes that are notable (acquired/mutated/
    disrupted), not routine intact essential genes.
    TODO(Moncef): confirm target_gate_status string values against schemas.py
    (currently guessing "present"/"absent" - may need to match an enum there)
    """
    target_genes = DRUG_TARGET_MAP.get(antibiotic, "").split(",")
    genome_evidence = evidence_by_genome.get(genome_id, {})

    for feature_name, evidence_type in genome_evidence.items():
        if evidence_type == "disrupted_gene":
            if any(feature_name.startswith(f"{tg}_") for tg in target_genes if tg):
                return "absent"
    return "present"


# ---------------------------------------------------------------------
# Step 5: supporting_features for a given (genome, antibiotic) prediction
# TODO(Moncef): confirm exact field/shape needed in schemas.py's Prediction —
# currently returning a list of {feature, evidence_type} dicts as a guess.
# ---------------------------------------------------------------------
def get_supporting_features(genome_id: str, matrix_df: pd.DataFrame, evidence_by_genome: dict) -> list[dict]:
    row = matrix_df.loc[genome_id]
    present_features = row[row == 1].index.tolist()
    return [
        {"feature": feat, "evidence_type": evidence_by_genome.get(genome_id, {}).get(feat, "unknown")}
        for feat in present_features
    ]
