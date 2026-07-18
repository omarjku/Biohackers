"""
synth_data.py — generate contract-shaped fixtures so Module 02 can be built now
Owner: Waji (pipeline)

Writes the four files described in docs/DATA_CONTRACT.md into data/synthetic/.
Real data is then a path change, not a rewrite:

    load_dataset("data/synthetic")   ->   load_dataset("data/processed")

THIS IS NOT BIOLOGY. Gene names and resistance rules are plausible-looking but
invented, tuned to exercise the pipeline's edge cases. Nothing measured here
says anything about real resistance, and no metric computed on it belongs in
the final writeup — it is scaffolding for testing code paths only.

What it deliberately builds in, so the pipeline is tested against the cases
that actually matter:

  1. Cluster structure with hitchhiker genes. Genomes in a cluster share
     near-identical gene content, including non-causal markers that correlate
     with resistance WITHIN a cluster. A random split therefore scores much
     higher than a grouped split — that gap is the leak the brief warns about,
     and here it is reproducible on demand.
  2. Missing drug targets. A few genomes lack a drug's target gene entirely,
     exercising the deterministic gate (-> "not_applicable", never
     "likely_to_work").
  3. Ragged coverage. Not every genome is tested against every drug, matching
     BV-BRC and the long-format labels.csv contract.
  4. Label noise. Resistance is probabilistic given genotype, so perfect
     scores are impossible and calibration has something real to correct.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SPECIES = "Escherichia coli"

# --- Feature vocabulary --------------------------------------------------
# CAUSAL genes actually drive the phenotype below. HITCHHIKER genes drive
# nothing but travel with clusters — they are the trap that a random split
# falls into and a grouped split does not.

CAUSAL_GENES = {
    "Ceftriaxone": ["blaCTX-M-15", "blaSHV-12"],
    "Ciprofloxacin": ["gyrA_S83L", "gyrA_D87N", "parC_S80I"],
    "Gentamicin": ["aac(3)-IIa", "aac(6')-Ib-cr"],
    "Meropenem": ["blaKPC-2", "blaNDM-1"],
}

HITCHHIKER_GENES = [
    "blaTEM-1", "sul1", "sul2", "tet(A)", "tet(B)",
    "dfrA17", "qnrS1", "mph(A)", "catA1", "aadA1",
]

# Target genes for the deterministic gate. Present in almost every genome —
# absence is rare and is exactly what the gate must catch.
TARGET_GENES = {
    "Ceftriaxone": ["ftsI", "mrdA"],
    "Ciprofloxacin": ["gyrA", "parC"],
    "Gentamicin": ["rpsL"],
    "Meropenem": ["ftsI", "mrdA"],
}

TARGET_NOTES = {
    "Ceftriaxone": "penicillin-binding proteins",
    "Ciprofloxacin": "DNA gyrase / topoisomerase IV",
    "Gentamicin": "30S ribosomal subunit",
    "Meropenem": "penicillin-binding proteins",
}

# Meropenem is left OUT of drug_targets.json on purpose — see write_dataset().

# Probability a genome carries each gene, before cluster inheritance.
BASE_PREVALENCE = {
    "blaCTX-M-15": 0.35, "blaSHV-12": 0.12,
    "gyrA_S83L": 0.30, "gyrA_D87N": 0.18, "parC_S80I": 0.22,
    "aac(3)-IIa": 0.20, "aac(6')-Ib-cr": 0.15,
    "blaKPC-2": 0.08, "blaNDM-1": 0.05,
}
HITCHHIKER_PREVALENCE = 0.30

# Given genotype, how often the phenotype is actually resistant.
P_RESISTANT_WITH_MECHANISM = 0.92
P_RESISTANT_WITHOUT = 0.04

# How faithfully a cluster member copies its founder's gene content.
# Low = tight clusters = a bigger random-vs-grouped gap.
GENE_FLIP_RATE = 0.03

# Fraction of (genome, drug) pairs with no lab result.
UNTESTED_RATE = 0.15

# How often a genome is missing a given drug's target gene.
TARGET_ABSENT_RATE = 0.04


def _cluster_sizes(n_genomes: int, n_clusters: int, rng: np.random.Generator) -> list[int]:
    """
    Skewed cluster sizes: a few large outbreak clusters, a long tail of singletons.

    This shape is what makes grouped splitting non-trivial — one huge cluster can
    swallow an entire split if it is allocated carelessly.
    """
    weights = rng.pareto(1.5, n_clusters) + 1
    sizes = np.maximum(1, np.round(weights / weights.sum() * n_genomes).astype(int))

    # Correct rounding drift back to exactly n_genomes.
    while sizes.sum() > n_genomes:
        sizes[sizes.argmax()] -= 1
    while sizes.sum() < n_genomes:
        sizes[sizes.argmin()] += 1
    return sizes.tolist()


def generate(
    n_genomes: int = 600,
    n_clusters: int = 45,
    seed: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Build the four contract objects in memory."""
    rng = np.random.default_rng(seed)

    causal_flat = [g for genes in CAUSAL_GENES.values() for g in genes]
    target_flat = sorted({g for genes in TARGET_GENES.values() for g in genes})
    feature_names = causal_flat + HITCHHIKER_GENES + target_flat

    genome_ids: list[str] = []
    cluster_ids: list[str] = []
    rows: list[dict[str, int]] = []

    sizes = _cluster_sizes(n_genomes, n_clusters, rng)
    counter = 0

    for c_idx, size in enumerate(sizes):
        cluster_id = f"CL-{c_idx:04d}"

        # The founder defines the cluster's gene content...
        founder = {
            gene: int(rng.random() < BASE_PREVALENCE.get(gene, HITCHHIKER_PREVALENCE))
            for gene in causal_flat + HITCHHIKER_GENES
        }
        for gene in target_flat:
            founder[gene] = int(rng.random() > TARGET_ABSENT_RATE)

        # ...and members inherit it with only small deviations. This is what
        # makes cluster membership so predictive, and why splitting on it matters.
        for _ in range(size):
            counter += 1
            genome_id = f"GEN-{counter:04d}"
            genome = {
                gene: (1 - val) if rng.random() < GENE_FLIP_RATE else val
                for gene, val in founder.items()
            }

            # A point mutation cannot exist without the gene it sits in.
            for mutation in ("gyrA_S83L", "gyrA_D87N"):
                if not genome["gyrA"]:
                    genome[mutation] = 0
            if not genome["parC"]:
                genome["parC_S80I"] = 0

            genome_ids.append(genome_id)
            cluster_ids.append(cluster_id)
            rows.append(genome)

    features = pd.DataFrame(rows, index=pd.Index(genome_ids, name="genome_id"))
    features = features[feature_names].astype(int)

    genomes = pd.DataFrame(
        {"species": SPECIES, "cluster_id": cluster_ids},
        index=pd.Index(genome_ids, name="genome_id"),
    )

    # --- phenotypes ---
    label_rows = []
    for genome_id in genome_ids:
        row = features.loc[genome_id]
        for drug, causal in CAUSAL_GENES.items():
            if rng.random() < UNTESTED_RATE:
                continue  # not tested against this drug — no row at all

            # A drug whose target is missing cannot be meaningfully tested;
            # the gate handles these, so they carry no training label.
            if not all(row[gene] for gene in TARGET_GENES[drug]):
                continue

            has_mechanism = any(row[gene] for gene in causal)
            p = P_RESISTANT_WITH_MECHANISM if has_mechanism else P_RESISTANT_WITHOUT
            resistant = rng.random() < p

            # MIC is informational only; the model trains on phenotype.
            mic = float(rng.choice([8, 16, 32, 64])) if resistant else float(
                rng.choice([0.03, 0.06, 0.12, 0.25])
            )
            label_rows.append(
                {
                    "genome_id": genome_id,
                    "drug": drug,
                    "phenotype": "R" if resistant else "S",
                    "mic": mic,
                }
            )

    labels = pd.DataFrame(label_rows)

    # Meropenem is intentionally omitted: the contract says a drug missing from
    # this file gets target_gate_status="unknown" and is still predicted, and
    # that branch needs a test case.
    drug_targets = {
        drug: {"target_genes": genes, "note": TARGET_NOTES[drug]}
        for drug, genes in TARGET_GENES.items()
        if drug != "Meropenem"
    }

    return features, labels, genomes, drug_targets


def write_dataset(out_dir: str | Path, **kwargs) -> Path:
    """Generate and write the four contract files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features, labels, genomes, drug_targets = generate(**kwargs)

    features.to_csv(out_dir / "features.csv")
    labels.to_csv(out_dir / "labels.csv", index=False)
    genomes.to_csv(out_dir / "genomes.csv")
    (out_dir / "drug_targets.json").write_text(json.dumps(drug_targets, indent=2))

    return out_dir


if __name__ == "__main__":
    out = write_dataset(Path(__file__).parent.parent / "data" / "synthetic")
    print(f"wrote fixtures to {out}\n")

    # Round-trip through the real loader, so generation failures surface here
    # rather than deep inside training.
    from data_io import load_dataset, summarize

    ds = load_dataset(out)
    print(
        f"{len(ds.features)} genomes, {len(ds.feature_names)} features, "
        f"{ds.genomes['cluster_id'].nunique()} clusters\n"
    )
    print(summarize(ds).to_string(index=False))
