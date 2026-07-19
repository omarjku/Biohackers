"""
pipeline.py — the module app.py imports as its backend.
Owner: Hazem (branch data/bvbrc-scale)

app.py calls `pipeline.run(uploaded_fasta) -> list[Prediction]`; when this module
is absent it falls back to hard-coded mock TB data. This wires the real live path
instead: an uploaded FASTA is annotated with AMRFinderPlus, bridged into the
model's feature vocabulary, scored, calibrated, and gated — see fasta_pipeline.

Requires the amrfinder binary on PATH (the project's conda env provides it). If it
is missing, fasta_pipeline raises a clear GenomeReaderError, which the app surfaces.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import fasta_pipeline
from schemas import Prediction

# The system covers exactly these — declared here so the UI can state scope honestly.
SUPPORTED_DRUGS = ["Ampicillin", "Ciprofloxacin", "Trimethoprim"]
SUPPORTED_SPECIES = "Escherichia coli"


def _materialize(fasta_file) -> Path:
    """A Streamlit UploadedFile or a path -> a real file path on disk."""
    if hasattr(fasta_file, "getvalue"):
        name = getattr(fasta_file, "name", "upload.fna")
        tmp_dir = Path(tempfile.mkdtemp(prefix="genome_firewall_"))
        fasta_path = tmp_dir / f"{Path(name).stem}.fna"
        fasta_path.write_bytes(fasta_file.getvalue())
        return fasta_path
    return Path(fasta_file)


def run(fasta_file, drugs: list[str] | None = None) -> list[Prediction]:
    """
    Run the live pipeline on an uploaded FASTA.

    `fasta_file` may be a Streamlit UploadedFile (has .getvalue()/.name) or a path.
    """
    fasta_path = _materialize(fasta_file)
    return fasta_pipeline.analyze_fasta(fasta_path, drugs=drugs or SUPPORTED_DRUGS)


def analyze(fasta_file, drugs: list[str] | None = None) -> tuple[dict, list[Prediction]]:
    """
    Like run(), but also returns a structural QC report for the FASTA — computed
    from the same on-disk copy, so the file is written only once. The app uses the
    QC to warn when an upload is not a plausible whole-genome assembly.
    Returns (qc_dict, predictions).
    """
    fasta_path = _materialize(fasta_file)
    qc = fasta_pipeline.genome_qc(fasta_path)
    preds = fasta_pipeline.analyze_fasta(fasta_path, drugs=drugs or SUPPORTED_DRUGS)
    return qc, preds
