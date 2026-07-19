"""
fasta_pipeline.py — live demo path: a raw FASTA -> per-drug predictions
Owner: Hazem (branch data/bvbrc-scale)

The model in data/processed was trained on BV-BRC NDARO gene calls (looked up by
genome id). A genome someone UPLOADS has no id to look up, so we annotate it
ourselves with AMRFinderPlus (genome_reader.run_amrfinder) and then translate its
gene names into the model's vocabulary — the "name bridge" below.

Why a bridge is needed: NDARO (what the model learned) and AMRFinderPlus (what we
run on an upload) are the same underlying gene catalog, but spell genes at
different granularity — AMRFinderPlus says `blaCTX-M-15`, the model column is
`CTX-M`. bridge_row() maps one to the other so an upload lands on the features the
model actually weights.

Honest limitation carried over from the training data: the model's NDARO features
are mutation-blind, so even though AMRFinderPlus DOES detect gyrA/parC point
mutations on an upload, the model has no strong column for them. Ciprofloxacin
therefore leans on weaker signal and honestly no-calls more often — which is the
intended, safe behavior, not a bug.

Public entry point:
    analyze_fasta(path) -> list[Prediction]   # calibrated, gated, ready to explain
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

import genome_reader
from calibration import Calibrator, predict_calibrated
from data_io import Dataset, load_dataset
from predictor import Predictor
from schemas import Prediction

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DATA_DIR = REPO_ROOT / "data" / "processed"
SPECIES = "Escherichia coli"

# A finished E. coli assembly is ~4.5-5.5 Mb. Anything far below this is a gene,
# an amplicon (e.g. a 16S rRNA sequence, ~1.5 kb), or a truncated file — NOT a
# whole genome. Scoring it is unsafe: with no AMR genes found, an empty feature
# vector looks identical to a genuinely susceptible genome, so the model returns
# a confident (and wrong) "likely to work". QC lets the app warn before that.
GENOME_MIN_BP = 1_000_000


def genome_qc(fasta_path: str | Path) -> dict:
    """
    Cheap structural sanity check on an uploaded FASTA (no annotation needed).
    Returns {total_bp, n_contigs, plausible_genome, message}.
    """
    fasta_path = Path(fasta_path)
    total_bp = 0
    n_contigs = 0
    with fasta_path.open() as fh:
        for line in fh:
            if line.startswith(">"):
                n_contigs += 1
            else:
                total_bp += len(line.strip())

    plausible = total_bp >= GENOME_MIN_BP and n_contigs > 0
    if plausible:
        message = ""
    elif n_contigs == 0:
        message = "No FASTA sequence records were found in this file."
    else:
        message = (
            f"This file is only {total_bp:,} bp across {n_contigs} record(s) — "
            f"that is a gene or partial sequence, not a whole-genome assembly "
            f"(E. coli genomes are ~4.5-5.5 Mb). Any prediction below is unreliable: "
            f"with no genome to scan, an absence of resistance genes is "
            f"indistinguishable from a truly susceptible strain. Upload a complete "
            f"E. coli assembly for a trustworthy result."
        )
    return {
        "total_bp": total_bp,
        "n_contigs": n_contigs,
        "plausible_genome": plausible,
        "message": message,
    }

# Curated causal AMR families, in the model's NDARO vocabulary. drug_database.py
# KNOWN_RESISTANCE_GENES is Moncef's, keyed by AMRFinderPlus names — those never
# match the NDARO family tokens the model trained on, so the predictor labels even
# a textbook ESBL "statistical_association". Rather than edit that module (and its
# tests), we carry the family-token knowledge here and upgrade the evidence when a
# genuinely causal determinant is present. Acquired enzymes only; intrinsic
# BlaEC/ampC are excluded so susceptible carriers are not over-claimed.
KNOWN_CAUSAL_FAMILIES: dict[str, set[str]] = {
    "Ampicillin": {
        "CTX-M", "TEM", "SHV", "OXA-1", "OXA-10", "OXA-2", "OXA-48",
        "CMY/CMY-2/CFE/LAT", "CMY-1/MOX", "DHA/MOR",
        "KPC", "NDM", "VIM", "IMP", "GES", "VEB", "CARB/PSE",
    },
    # NOTE: "Quinolone" (the acquired qnr family) is deliberately NOT here. The
    # bridge routes gyrA/parC TARGET mutations onto that column, and calling a
    # chromosomal QRDR mutation a "known qnr determinant" would be mechanism
    # conflation. Only a genuine acquired fluoroquinolone-acetylating gene counts
    # as a known determinant here; everything else stays statistical_association.
    "Ciprofloxacin": {"AAC-Ib-cr"},
    "Trimethoprim": {"dhfrI", "dhfrV", "dfrA", "dfrD", "dfrA12"},
}

# --------------------------------------------------------------------------
# The name bridge: AMRFinderPlus Element symbol -> ONE model feature column
# --------------------------------------------------------------------------
# The training features (NDARO) encode each gene as a single family token
# ("blaCTX-M-15" was stored as "CTX-M", "sul1" as "Sulfonamide"), and those
# family tokens are where the model's signal actually sits (CTX-M in 386 genomes,
# the rarer exact-name column "blaCTX-M" in only a handful). So each AMRFinderPlus
# gene must map to the ONE dominant family token, not to every related column —
# setting several synonyms at once builds a feature vector no real genome has and
# trips the out-of-distribution no-call gate. Rules are ordered specific -> general
# and the FIRST match wins.
_BRIDGE_RULES: list[tuple[str, str]] = [
    # ---- beta-lactamases (ampicillin) ----
    (r"^(bla)?ctx-m", "CTX-M"),
    (r"^(bla)?tem", "TEM"),
    (r"^(bla)?shv", "SHV"),
    (r"^(bla)?oxa-?10", "OXA-10"),
    (r"^(bla)?oxa-?48", "OXA-48"),
    (r"^(bla)?oxa-?2$", "OXA-2"),
    (r"^(bla)?oxa", "OXA-1"),
    (r"^(bla)?cmy", "CMY/CMY-2/CFE/LAT"),
    (r"^(bla)?(dha|mor)", "DHA/MOR"),
    (r"^(bla)?kpc", "KPC"),
    (r"^(bla)?ndm", "NDM"),
    (r"^(bla)?vim", "VIM"),
    (r"^(bla)?imp", "IMP"),
    (r"^(bla)?ges", "GES"),
    (r"^(bla)?veb", "VEB"),
    (r"^(bla)?(carb|pse)", "CARB/PSE"),
    (r"^ampc", "ampC"),
    # ---- trimethoprim / sulfonamide ----
    (r"^sul", "Sulfonamide"),
    (r"^(dfr|dhfr)", "dhfrI"),
    # ---- tetracycline ----
    (r"^tet", "Tet"),
    # ---- macrolide ----
    (r"^(mph|mrx)", "Mph"),
    (r"^erm", "Erm"),
    (r"^ere", "EreB"),
    (r"^mef", "Mef"),
    (r"^msr", "Msr"),
    (r"^lnu", "Lnu/Lnu"),
    # ---- phenicol ----
    (r"^catb", "CatB"),
    (r"^cat", "CatA1/CatA4"),
    (r"^cml", "CmlA"),
    (r"^flor", "FloR"),
    # ---- aminoglycoside ----
    (r"^aac\(6'\)-ib-cr", "AAC-Ib-cr"),   # also fluoroquinolone-acetylating
    (r"^aac", "AAC-II,III,IV,VI,VIII,IX,X"),
    (r"^(aad|ant)", "ANT-Ia"),
    (r"^(aph\(3''\)|aph\(6\)|str[ab])", "APH-Ic/APH-Id"),
    (r"^aph", "APH-I"),
    # ---- fluoroquinolone (model is mutation-blind; weak by design) ----
    (r"^(qnr|gyr|par)", "Quinolone"),
    # ---- other ----
    (r"^fosa", "fosA"),
    (r"^mcr", "colistin"),
    (r"^qac", "QacE"),
]

_COMPILED = [(re.compile(pat), col) for pat, col in _BRIDGE_RULES]

# Intrinsic to essentially every E. coli (chromosomal AmpC), present in 92% of
# training genomes. AMRFinderPlus only reports it when notably altered, so we set
# it for any E. coli upload — otherwise every bridged row looks unusual for
# lacking the single most common training feature.
_INTRINSIC_COLUMNS = ("BlaEC",)


def bridge_row(genes_found: set[str], model_columns: list[str]) -> pd.Series:
    """
    Turn a set of AMRFinderPlus Element symbols into a binary feature row indexed
    by the model's columns, mapping each gene to one dominant family token.
    Everything the bridge cannot place stays 0.
    """
    colset = set(model_columns)
    lower_to_col = {c.lower(): c for c in model_columns}
    row = pd.Series(0, index=model_columns, dtype=int)

    for col in _INTRINSIC_COLUMNS:
        if col in colset:
            row[col] = 1

    for sym in genes_found:
        base = sym.split("_")[0].strip().lower()   # gyrA_S83L -> gyra
        target: str | None = None

        # 1. first family rule that matches wins — routes to the dominant family
        #    token where the model's signal actually sits.
        for rx, col in _COMPILED:
            if rx.match(base) and col in colset:
                target = col
                break

        # 2. fallback: the base name is itself a model column (uncovered genes)
        if target is None and base in lower_to_col:
            target = lower_to_col[base]

        if target is not None:
            row[target] = 1

    return row


# --------------------------------------------------------------------------
# Scan a FASTA -> AMRFinderPlus genes
# --------------------------------------------------------------------------
def scan_fasta(fasta_path: str | Path, work_dir: Path | None = None) -> set[str]:
    """Run AMRFinderPlus on one FASTA and return the set of AMR Element symbols."""
    fasta_path = Path(fasta_path)
    work_dir = work_dir or fasta_path.parent
    tsv = genome_reader.run_amrfinder(fasta_path, work_dir, organism="Escherichia")
    return genome_reader.parse_amrfinder_result(tsv)["genes_found"]


# --------------------------------------------------------------------------
# The trained engine (fit once, reuse across uploads)
# --------------------------------------------------------------------------
@dataclass
class Engine:
    training: Dataset
    predictor: Predictor
    calibrator: Calibrator

    @property
    def columns(self) -> list[str]:
        return list(self.training.features.columns)


@lru_cache(maxsize=1)
def load_engine(data_dir: str | Path = MODEL_DATA_DIR) -> Engine:
    """Fit predictor + calibrator on the scaled real dataset. Cached per process."""
    ds = load_dataset(data_dir)
    predictor = Predictor.fit(ds, seed=0)
    calibrator = Calibrator.fit(ds, predictor)
    return Engine(ds, predictor, calibrator)


def _single_genome_dataset(row: pd.Series, engine: Engine, sample_id: str) -> Dataset:
    """Wrap one bridged feature row as a Dataset so the normal predict path runs."""
    features = pd.DataFrame([row.values], index=[sample_id], columns=row.index)
    features.index.name = "genome_id"
    genomes = pd.DataFrame(
        {"species": [SPECIES], "cluster_id": ["UPLOAD"]},
        index=pd.Index([sample_id], name="genome_id"),
    )
    labels = pd.DataFrame(columns=["genome_id", "drug", "phenotype"])
    return Dataset(features, labels, genomes, engine.training.drug_targets)


def analyze_fasta(
    fasta_path: str | Path,
    drugs: list[str] | None = None,
    data_dir: str | Path = MODEL_DATA_DIR,
) -> list[Prediction]:
    """
    Full live path for an uploaded genome:
        FASTA -> AMRFinderPlus -> name bridge -> model -> calibration -> no-call gate.
    Returns calibrated, gated Predictions (ready for the explainer).
    """
    fasta_path = Path(fasta_path)
    engine = load_engine(data_dir)
    genes = scan_fasta(fasta_path)
    row = bridge_row(genes, engine.columns)
    sample_id = fasta_path.stem
    upload_ds = _single_genome_dataset(row, engine, sample_id)
    preds = predict_calibrated(
        upload_ds, engine.predictor, engine.calibrator, sample_id, drugs=drugs
    )
    present = set(row[row == 1].index)
    return [_upgrade_evidence(p, present) for p in preds]


def _upgrade_evidence(pred: Prediction, present_families: set[str]) -> Prediction:
    """
    Promote evidence to a known mechanism when a curated causal family is present.

    The predictor can only recognise curated genes it was handed in AMRFinderPlus
    naming; in the NDARO vocabulary that recognition misses, so a real ESBL comes
    back as "statistical_association". We correct that here for a "likely to fail"
    call, and surface the actual causal determinant(s) as the evidence.
    """
    if pred.call != "likely_to_fail":
        return pred
    causal = KNOWN_CAUSAL_FAMILIES.get(pred.drug, set()) & present_families
    if not causal:
        return pred
    from schemas import SupportingFeature

    return pred.model_copy(
        update={
            "evidence_category": "known_gene_or_mutation",
            "supporting_features": [SupportingFeature(gene=g) for g in sorted(causal)],
        }
    )


def report_for_fasta(
    fasta_path: str | Path,
    drugs: list[str] | None = None,
    use_llm: bool = False,
    data_dir: str | Path = MODEL_DATA_DIR,
) -> list[dict]:
    """
    One-call entry point for the Streamlit app: an uploaded FASTA -> the frontend
    JSON array (drug / drug_class / underlying_state / confidence / target_marker /
    locus_id / bio_explanation / stat_explanation). `explainer.DISCLAIMER` holds
    the mandatory lab-confirmation banner the UI must show alongside these.
    """
    import explainer

    preds = analyze_fasta(fasta_path, drugs=drugs, data_dir=data_dir)
    return explainer.explain_report(preds, use_llm=use_llm)


if __name__ == "__main__":
    import sys

    preds = analyze_fasta(sys.argv[1])
    for p in preds:
        genes = ", ".join(f.gene for f in p.supporting_features) or "—"
        print(f"{p.drug:15} {p.call:16} conf={p.confidence:.2f}  gate={p.target_gate_status}  [{genes}]")
