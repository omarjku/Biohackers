"""
genome_reader.py — Module 01: FASTA -> features
Owner: Moncef (biology)

Turns a directory of assembled genomes into the binary AMR feature matrix that
predictor.py consumes, by running AMRFinderPlus over each FASTA and pivoting the
hits into one row per genome, one column per gene/mutation.

Run:  python src/genome_reader.py --fasta-dir data/raw/fasta --out-dir data/raw
Out:  features.csv          (genome_id x AMR feature, binary — DATA_CONTRACT.md §1)
      gene_metadata.csv     (feature_name, gene_symbol, evidence_type, amr_class,
                             amr_subclass — drives evidence tiering in predictor.py)

The AMRFinderPlus TSVs are cached per genome, so re-running is cheap and a failed
batch resumes instead of starting over. Delete the TSV to force a re-scan.

See schemas.py for the Prediction shape this eventually feeds into.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# AMRFinderPlus column names, v4.2.7. Confirmed against the real run behind
# data/raw/files.zip. Older versions used "Gene symbol" instead of
# "Element symbol"; _column() below tolerates both so a version bump on someone
# else's machine does not silently produce an empty matrix.
_SYMBOL_COLUMNS = ("Element symbol", "Gene symbol")
_TYPE_COLUMNS = ("Type", "Element type")
_SUBTYPE_COLUMNS = ("Subtype", "Element subtype")
_CLASS_COLUMNS = ("Class",)
_SUBCLASS_COLUMNS = ("Subclass",)

DEFAULT_ORGANISM = "Escherichia"
DEFAULT_THREADS_PER_JOB = 2


class GenomeReaderError(RuntimeError):
    """Raised when annotation cannot produce a trustworthy feature matrix."""


def _column(df: pd.DataFrame, candidates: tuple[str, ...], required: bool = True) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    if required:
        raise GenomeReaderError(
            f"AMRFinderPlus output has none of the expected columns {candidates}. "
            f"Found: {list(df.columns)[:10]}. Check the AMRFinderPlus version."
        )
    return None


# ---------------------------------------------------------------------
# Step 1: run AMRFinderPlus on one genome
# ---------------------------------------------------------------------
def run_amrfinder(
    fasta_path: Path,
    output_dir: Path,
    organism: str = DEFAULT_ORGANISM,
    threads: int = DEFAULT_THREADS_PER_JOB,
    force: bool = False,
) -> Path:
    """
    Run AMRFinderPlus on a single genome FASTA, return the path to its result TSV.

    `--organism` is not optional for this project. Without it AMRFinderPlus
    reports acquired genes only and skips point-mutation screening, which is the
    dominant ciprofloxacin resistance mechanism in E. coli — gyrA/parC would
    vanish from the matrix and the drug would look like it has no signal.

    Cached: an existing non-empty TSV is reused unless `force=True`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{fasta_path.stem}.amrfinder.tsv"

    if out_path.exists() and out_path.stat().st_size > 0 and not force:
        return out_path

    if shutil.which("amrfinder") is None:
        raise GenomeReaderError(
            "amrfinder not found on PATH. Install AMRFinderPlus "
            "(https://github.com/ncbi/amr) and run `amrfinder_update` once to "
            "fetch the database, or run it via the ncbi/amr Docker image."
        )

    try:
        subprocess.run(
            ["amrfinder", "-n", str(fasta_path), "--organism", organism,
             "-o", str(out_path), "--threads", str(threads)],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as exc:
        out_path.unlink(missing_ok=True)  # never leave a truncated TSV to be cached
        raise GenomeReaderError(
            f"amrfinder failed on {fasta_path.name} (exit {exc.returncode}): "
            f"{(exc.stderr or '').strip()[:500]}"
        ) from exc

    return out_path


def run_amrfinder_batch(
    fasta_paths: list[Path],
    output_dir: Path,
    organism: str = DEFAULT_ORGANISM,
    jobs: int = 8,
    threads: int = DEFAULT_THREADS_PER_JOB,
    force: bool = False,
) -> tuple[list[Path], dict[str, str]]:
    """
    Annotate many genomes concurrently. Returns (tsv_paths, {genome_id: error}).

    Failures are collected rather than raised: on a multi-thousand-genome run one
    bad assembly should not discard the other 2,000 results. The caller decides
    what an acceptable failure rate is.

    Threads, not processes — each worker spends its time waiting on an
    amrfinder subprocess, so the GIL is not the constraint. Total CPU load is
    roughly jobs * threads; keep that at or below your core count.
    """
    done: list[Path] = []
    errors: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(run_amrfinder, p, output_dir, organism, threads, force): p
            for p in fasta_paths
        }
        for i, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            try:
                done.append(future.result())
            except GenomeReaderError as exc:
                errors[path.stem] = str(exc)
            print(f"\r  annotated {i}/{len(fasta_paths)} "
                  f"({len(errors)} failed)", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return done, errors


# ---------------------------------------------------------------------
# Step 2: parse one AMRFinderPlus TSV
# ---------------------------------------------------------------------
def parse_amrfinder_result(tsv_path: Path) -> dict:
    """
    Parse one result TSV.

    Returns:
        {
            "genes_found": set of gene/mutation names detected,
            "evidence": {gene_name: "acquired_gene"|"point_mutation"|"disrupted_gene"},
            "annotations": {gene_name: {"gene_symbol", "amr_class", "amr_subclass"}},
        }

    A genome with no AMR hits is a valid result, not an error — it yields an
    empty set and becomes an all-zero row. Treating it as a failure would
    silently drop exactly the susceptible genomes the model needs.
    """
    if tsv_path.stat().st_size == 0:
        return {"genes_found": set(), "evidence": {}, "annotations": {}}

    df = pd.read_csv(tsv_path, sep="\t", dtype=str).fillna("")
    if df.empty:
        return {"genes_found": set(), "evidence": {}, "annotations": {}}

    symbol_col = _column(df, _SYMBOL_COLUMNS)
    type_col = _column(df, _TYPE_COLUMNS)
    subtype_col = _column(df, _SUBTYPE_COLUMNS, required=False)
    class_col = _column(df, _CLASS_COLUMNS, required=False)
    subclass_col = _column(df, _SUBCLASS_COLUMNS, required=False)

    df = df[df[type_col].str.upper() == "AMR"]

    genes_found: set[str] = set()
    evidence: dict[str, str] = {}
    annotations: dict[str, dict] = {}

    for _, row in df.iterrows():
        gene = str(row[symbol_col]).strip()
        if not gene:
            continue
        subtype = str(row[subtype_col]).strip().upper() if subtype_col else ""

        genes_found.add(gene)
        if subtype == "POINT_DISRUPT":
            evidence[gene] = "disrupted_gene"
        elif subtype == "POINT":
            evidence[gene] = "point_mutation"
        else:
            evidence[gene] = "acquired_gene"

        # Point-mutation features arrive as "gyrA_S83L"; the bare gene symbol is
        # the part before the underscore, matching DATA_CONTRACT.md §1.
        annotations[gene] = {
            "gene_symbol": gene.split("_")[0] if evidence[gene] != "acquired_gene" else gene,
            "amr_class": str(row[class_col]).strip() if class_col else "",
            "amr_subclass": str(row[subclass_col]).strip() if subclass_col else "",
        }

    return {"genes_found": genes_found, "evidence": evidence, "annotations": annotations}


# ---------------------------------------------------------------------
# Step 3: build the binary feature matrix across many genomes
# ---------------------------------------------------------------------
def build_feature_matrix(
    amrfinder_results_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Returns:
        matrix_df:     genomes x features, binary, index = genome_id
        metadata_df:   one row per feature (feature_name, gene_symbol,
                       evidence_type, amr_class, amr_subclass)
        evidence_by_genome: {genome_id: {gene_name: evidence_type}}
    """
    tsv_files = sorted(amrfinder_results_dir.glob("*.amrfinder.tsv"))
    if not tsv_files:
        raise GenomeReaderError(
            f"No *.amrfinder.tsv files in {amrfinder_results_dir}. Run the "
            "annotation step first."
        )

    per_genome: dict[str, set[str]] = {}
    evidence_by_genome: dict[str, dict] = {}
    feature_evidence: dict[str, str] = {}
    feature_annotations: dict[str, dict] = {}

    for f in tsv_files:
        genome_id = f.name.replace(".amrfinder.tsv", "")
        parsed = parse_amrfinder_result(f)
        per_genome[genome_id] = parsed["genes_found"]
        evidence_by_genome[genome_id] = parsed["evidence"]
        feature_evidence.update(parsed["evidence"])
        feature_annotations.update(parsed["annotations"])

    all_features = sorted(feature_evidence)
    matrix_df = pd.DataFrame(
        0, index=sorted(per_genome), columns=all_features, dtype=int
    )
    for genome_id, genes in per_genome.items():
        if genes:  # .loc[id, []] on an empty list is not a safe no-op
            matrix_df.loc[genome_id, sorted(genes)] = 1
    matrix_df.index.name = "genome_id"

    metadata_df = pd.DataFrame(
        [
            {
                "feature_name": feat,
                "gene_symbol": feature_annotations[feat]["gene_symbol"],
                "evidence_type": feature_evidence[feat],
                "amr_class": feature_annotations[feat]["amr_class"],
                "amr_subclass": feature_annotations[feat]["amr_subclass"],
            }
            for feat in all_features
        ]
    )

    _validate_matrix(matrix_df)
    return matrix_df, metadata_df, evidence_by_genome


def _validate_matrix(matrix_df: pd.DataFrame) -> None:
    """Fail here rather than let data_io reject the file three steps later."""
    if matrix_df.index.has_duplicates:
        dupes = matrix_df.index[matrix_df.index.duplicated()].unique().tolist()
        raise GenomeReaderError(f"Duplicate genome_id in feature matrix: {dupes[:5]}")
    if matrix_df.isna().any().any():
        raise GenomeReaderError("Feature matrix contains missing values")
    bad = [c for c in matrix_df.columns if not set(matrix_df[c].unique()) <= {0, 1}]
    if bad:
        raise GenomeReaderError(f"Feature matrix is not binary in columns: {bad[:5]}")


# ---------------------------------------------------------------------
# The target gate and supporting-feature construction both live in
# predictor.py. They are NOT duplicated here.
#
# predictor.target_gate() returns the full "present"/"absent"/"unknown" set that
# schemas.Prediction requires and is covered by tests/test_predictor.py;
# predictor._feature() builds schemas.SupportingFeature objects. Earlier drafts
# of this module carried its own versions of both. They disagreed with
# predictor's — one defaulted to "present" for an unseen genome, and the other
# returned plain dicts keyed "feature"/"evidence_type" rather than the
# gene/mutation/note model — so they were removed rather than fixed. Two
# implementations of a safety gate is how the two answers silently diverge.
#
# Worth preserving from that work, because it is correct and non-obvious:
# AMRFinderPlus only reports a gene when something notable happened to it
# (acquired, mutated, or POINT_DISRUPT). It does NOT report an intact essential
# gene. So for a target gene, "not in the hit list" means wildtype/intact, NOT
# missing — which is exactly why predictor.target_gate() returns "unknown"
# rather than "absent" when the column is not there.
# ---------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    parser.add_argument("--fasta-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tsv-dir", type=Path, default=None,
                        help="Where AMRFinderPlus TSVs are cached "
                             "(default: <out-dir>/amrfinder)")
    parser.add_argument("--organism", default=DEFAULT_ORGANISM)
    parser.add_argument("--jobs", type=int, default=8,
                        help="Genomes annotated concurrently")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS_PER_JOB,
                        help="Threads per amrfinder job")
    parser.add_argument("--force", action="store_true",
                        help="Re-annotate even if a cached TSV exists")
    parser.add_argument("--max-failures", type=float, default=0.05,
                        help="Abort if more than this fraction of genomes fail")
    args = parser.parse_args(argv)

    fastas = sorted(
        p for ext in ("*.fna", "*.fasta", "*.fa") for p in args.fasta_dir.glob(ext)
    )
    if not fastas:
        raise GenomeReaderError(f"No FASTA files found in {args.fasta_dir}")

    tsv_dir = args.tsv_dir or (args.out_dir / "amrfinder")
    print(f"annotating {len(fastas)} genomes "
          f"({args.jobs} concurrent x {args.threads} threads)...", file=sys.stderr)
    _, errors = run_amrfinder_batch(
        fastas, tsv_dir, args.organism, args.jobs, args.threads, args.force
    )

    if errors:
        rate = len(errors) / len(fastas)
        print(f"\n{len(errors)} genome(s) failed to annotate:", file=sys.stderr)
        for gid, msg in list(errors.items())[:5]:
            print(f"  {gid}: {msg[:160]}", file=sys.stderr)
        if rate > args.max_failures:
            raise GenomeReaderError(
                f"{rate:.1%} of genomes failed to annotate, above the "
                f"{args.max_failures:.1%} threshold. Not writing a partial matrix."
            )

    matrix_df, metadata_df, _ = build_feature_matrix(tsv_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = args.out_dir / "features.csv"
    metadata_path = args.out_dir / "gene_metadata.csv"
    matrix_df.to_csv(matrix_path)
    metadata_df.to_csv(metadata_path, index=False)

    print(f"\n{len(matrix_df)} genomes x {len(matrix_df.columns)} features")
    print(f"  {matrix_path}")
    print(f"  {metadata_path}")
    print("\nfeatures by evidence type:")
    print(metadata_df["evidence_type"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
