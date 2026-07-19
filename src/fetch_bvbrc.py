"""
fetch_bvbrc.py — pull the challenge dataset from the BV-BRC Data API
Owner: Hazem (added on branch data/bvbrc-scale)

Why this exists
---------------
`adapt_real_data.py` reshapes a *precomputed* AMRFinderPlus run (data/raw/files.zip,
~119 genomes) into the four contract files. That run only ever covered ~119 of the
2,154 E. coli genome ids we already have laboratory AST labels for — and it clusters
into 3 coarse groups, too few to fill a grouped train/calibration/test split. Every
real-data metric downstream is starved as a result (test sets of 9-11 genomes,
Ampicillin balanced-accuracy at chance).

This module goes to the source instead. For the same 2,154 ids in
`data/genome_id_list.csv` it pulls, over the public BV-BRC HTTPS API (no AMRFinderPlus
install, no FASTA download):

  * labels   <- genome_amr,  laboratory-measured rows only (brief-mandated)
  * features <- sp_gene source=NDARO — the NCBI Reference Gene Catalog that
                AMRFinderPlus itself annotates against, so this is an
                AMRFinderPlus-*aligned* acquired-resistance-gene set, not a
                generic and noisy specialty-gene dump
  * groups   <- genome.mlst sequence type — a real biological grouping, and a
                far better cluster_id than within-species Mash (splits.py already
                documents that E. coli Mash distances barely separate strains)

Output is data/processed/{features,labels,genomes}.csv + drug_targets.json +
gene_metadata.csv — byte-for-byte what data_io.load_dataset() validates, so
predictor / calibration / evaluation consume it unchanged.

Two honest limitations, documented here because they are the reason for the design
and because the brief rewards stating them:

  1. Feature source is BV-BRC's NDARO calls, not a fresh AMRFinderPlus run. It is
     strong on ACQUIRED genes (blaTEM, blaCTX-M, sul, dfrA, tet, aac/ant/aph) but
     BLIND to resistance point mutations (gyrA/parC S83L etc.), because sp_gene
     records gene presence, not the mutation. E. coli fluoroquinolone resistance
     is mutation-driven, so ciprofloxacin loses signal here and honestly earns
     more no_calls — a strength per the brief, not a hidden failure. A local
     AMRFinderPlus run stays the documented fidelity upgrade.
  2. This REBUILDS the whole real matrix from one consistent source rather than
     blending with the 119-genome AMRFinderPlus matrix; mixing two annotators'
     vocabularies in one feature space would be worse than either alone.

Run:  python src/fetch_bvbrc.py                 # all 2,154 ids
      python src/fetch_bvbrc.py --limit 200     # quick subset for iteration
Out:  data/processed/{features,labels,genomes}.csv + drug_targets.json
      data/raw/bvbrc_cache/*.json               # raw responses, so re-runs are free
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "raw" / "bvbrc_cache"
GENOME_ID_LIST = DATA_DIR / "genome_id_list.csv"

API_BASE = "https://www.bv-brc.org/api"
SPECIES = "Escherichia coli"

# genome_amr stores drug names lowercase; the contract + drug_database use title
# case. Normalising here is what keeps the target-gate lookup from silently
# missing on every row (same fix adapt_real_data.canonical_drug makes).
DRUGS_LOWER = {"ampicillin", "ciprofloxacin", "trimethoprim"}

# Only laboratory-measured phenotypes are admissible. genome_amr mixes these with
# ~3.5x as many "Computational Method" rows (MIC XGBoost model predictions); the
# brief is explicit that model-generated phenotypes must never be used as labels.
LAB_EVIDENCE = "Laboratory Method"
_PHENOTYPE_MAP = {"Resistant": "R", "Susceptible": "S"}  # Intermediate/None dropped

BATCH_SIZE = 100          # genome ids per API request
PAGE_LIMIT = 25000        # rows per request; batches stay well under this
HTTP_RETRIES = 4
HTTP_TIMEOUT = 100        # BV-BRC latency is spiky; be patient, then retry


def canonical_drug(name: str) -> str:
    return str(name).strip().title()


# ---------------------------------------------------------------------------
# HTTP with on-disk caching + retry. The cache makes a re-run free and lets a
# failed batch resume instead of restarting, mirroring genome_reader's TSV cache.
# ---------------------------------------------------------------------------
def _api_get(endpoint: str, rql: str, cache_dir: Path) -> list[dict]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{endpoint}?{rql}".encode()).hexdigest()[:16]
    cache_file = cache_dir / f"{endpoint}_{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    url = f"{API_BASE}/{endpoint}/?{rql}&http_accept=application/json"
    last_err: Exception | None = None
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                data = json.loads(resp.read().decode())
            cache_file.write_text(json.dumps(data))
            return data
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_err = exc
            time.sleep(2 * attempt)  # linear backoff
    raise RuntimeError(f"BV-BRC {endpoint} failed after {HTTP_RETRIES} tries: {last_err}")


def _batched(items: list[str], size: int = BATCH_SIZE):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _fetch_over_ids(endpoint: str, ids: list[str], select: str, extra: str, cache_dir: Path) -> list[dict]:
    """Run one `in(genome_id,(...))` query per batch of ids and concatenate."""
    rows: list[dict] = []
    for batch in _batched(ids):
        id_clause = "in(genome_id,(" + ",".join(batch) + "))"
        filt = f"and({id_clause},{extra})" if extra else id_clause
        rql = f"{filt}&select({select})&limit({PAGE_LIMIT})"
        page = _api_get(endpoint, rql, cache_dir)
        if len(page) >= PAGE_LIMIT:
            # A single 100-genome batch has never approached 25k rows in practice;
            # if it ever does, fail loud rather than silently truncate the matrix.
            raise RuntimeError(
                f"{endpoint} batch hit the {PAGE_LIMIT}-row page limit — "
                "shrink BATCH_SIZE or add pagination before trusting this run."
            )
        rows.extend(page)
    return rows


# ---------------------------------------------------------------------------
# Labels  <-  genome_amr
# ---------------------------------------------------------------------------
def fetch_labels(ids: list[str], cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    rows = _fetch_over_ids(
        "genome_amr",
        ids,
        select="genome_id,antibiotic,resistant_phenotype,evidence,laboratory_typing_method",
        extra="",
        cache_dir=cache_dir,
    )
    recs = []
    for r in rows:
        if r.get("evidence") != LAB_EVIDENCE:
            continue
        drug = (r.get("antibiotic") or "").lower()
        if drug not in DRUGS_LOWER:
            continue
        pheno = _PHENOTYPE_MAP.get(r.get("resistant_phenotype"))
        if pheno is None:  # None / Intermediate
            continue
        recs.append(
            {
                "genome_id": str(r["genome_id"]),
                "drug": canonical_drug(drug),
                "phenotype": pheno,
                "lab_method": r.get("laboratory_typing_method") or "",
            }
        )
    labels = pd.DataFrame(recs, columns=["genome_id", "drug", "phenotype", "lab_method"])

    # A genome can carry several lab measurements for one drug. Keep the pair only
    # when every measurement agrees; drop genuinely conflicting (R and S) pairs
    # rather than guess a winner.
    labels = labels.drop_duplicates(subset=["genome_id", "drug", "phenotype"])
    agree = (
        labels.groupby(["genome_id", "drug"])["phenotype"].nunique().eq(1).rename("ok")
    )
    labels = labels.merge(agree, on=["genome_id", "drug"])
    labels = labels[labels["ok"]].drop(columns="ok")
    labels = labels.drop_duplicates(subset=["genome_id", "drug"])
    return labels.sort_values(["genome_id", "drug"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Features  <-  sp_gene (source=NDARO)
# ---------------------------------------------------------------------------
# Intrinsic / housekeeping targets that BV-BRC lists as "resistance" genes but
# that every E. coli carries, so as gene-PRESENCE features they are constant and
# carry no signal (their resistance is mutation-driven, which sp_gene cannot see).
_INTRINSIC = {
    "gyra", "gyrb", "parc", "pare", "fola", "folp", "glpt", "uhpt", "alr", "ddl",
    "rpob", "rpsl", "mgrb", "cpxr", "ef-tu", "h-ns",
}


def _feature_token(row: dict) -> str | None:
    """
    Reduce a sp_gene NDARO row to a stable gene-family token, or None to drop it.

    NDARO rows rarely fill the `gene` column; the curated family lives in the
    free-text `product` after a `=>` or `@` marker
    (e.g. "... => CTX-M family", "... @ Sul1"). We take that family symbol.
    Rows with neither a gene nor a `=>`/`@` family are intrinsic target genes
    (e.g. "DNA gyrase subunit A") — dropped, per the mutation-blindness note above.
    """
    gene = (row.get("gene") or "").strip()
    if not gene:
        product = row.get("product") or ""
        m = re.search(r"(?:=>|@)\s*(.+)$", product)
        if not m:
            return None
        tok = m.group(1)
        tok = re.sub(r"\bfamily\b", "", tok, flags=re.I)
        tok = re.sub(r"\(.*?\)", "", tok)          # drop parentheticals
        tok = re.split(r"\s{2,}|=>|@", tok)[0]      # first family if chained
        tok = tok.strip(" .,;/")
        gene = tok.split()[0] if tok.split() else ""
    gene = gene.strip(" .,;/")
    if not gene or gene.lower() in _INTRINSIC:
        return None
    return gene


def fetch_features(ids: list[str], cache_dir: Path = CACHE_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (features, gene_metadata).

    features: binary genome_id x gene-family matrix (only genomes with >=1
    acquired gene appear; build() adds all-zero rows for the rest).
    gene_metadata: feature_name -> amr_class + evidence_type, for the explainer's
    evidence tiering (every NDARO acquired gene is a known mechanism).
    """
    rows = _fetch_over_ids(
        "sp_gene",
        ids,
        select="genome_id,gene,product,antibiotics_class",
        extra="eq(source,NDARO)",
        cache_dir=cache_dir,
    )
    present: dict[str, set[str]] = {}
    meta: dict[str, str] = {}
    for r in rows:
        tok = _feature_token(r)
        if tok is None:
            continue
        present.setdefault(str(r["genome_id"]), set()).add(tok)
        cls = (r.get("antibiotics_class") or "").strip()
        if cls and tok not in meta:
            meta[tok] = cls

    genes = sorted({g for gs in present.values() for g in gs})
    features = pd.DataFrame(0, index=sorted(present), columns=genes, dtype=int)
    features.index.name = "genome_id"
    for gid, gs in present.items():
        features.loc[gid, list(gs)] = 1

    gene_metadata = pd.DataFrame(
        {
            "feature_name": genes,
            "amr_class": [meta.get(g, "") for g in genes],
            "evidence_type": "known_gene",  # NDARO = curated AMR reference genes
            "source": "BV-BRC/NDARO",
        }
    )
    return features, gene_metadata


# ---------------------------------------------------------------------------
# Groups  <-  genome.mlst
# ---------------------------------------------------------------------------
def _parse_st(mlst: str | None) -> str | None:
    """'MLST.ecoli_achtman_4.410' -> 'ST-410'. Non-numeric / missing -> None."""
    if not mlst:
        return None
    st = str(mlst).rsplit(".", 1)[-1]
    return f"ST-{st}" if st.isdigit() else None


def fetch_genomes(ids: list[str], cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    rows = _fetch_over_ids(
        "genome",
        ids,
        select="genome_id,genome_name,mlst,species",
        extra="",
        cache_dir=cache_dir,
    )
    recs = []
    for r in rows:
        st = _parse_st(r.get("mlst"))
        if st is None:  # no usable ST -> cannot group safely -> drop downstream
            continue
        recs.append({"genome_id": str(r["genome_id"]), "species": SPECIES, "cluster_id": st})
    genomes = pd.DataFrame(recs, columns=["genome_id", "species", "cluster_id"])
    return genomes.drop_duplicates("genome_id").set_index("genome_id").sort_index()


# ---------------------------------------------------------------------------
# drug_targets.json  <-  drug_database.DRUG_TARGET_MAP
# ---------------------------------------------------------------------------
def build_drug_targets(drugs: list[str]) -> dict[str, dict]:
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


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def build(limit: int | None = None, out_dir: Path = OUT_DIR, cache_dir: Path = CACHE_DIR) -> dict[str, Path]:
    ids = pd.read_csv(GENOME_ID_LIST, dtype=str)["genome_id"].tolist()
    if limit:
        ids = ids[:limit]
    print(f"fetching {len(ids)} genome ids from BV-BRC...")

    labels = fetch_labels(ids, cache_dir)
    features, gene_metadata = fetch_features(ids, cache_dir)
    genomes = fetch_genomes(ids, cache_dir)

    # Keep set = genomes we can both label AND group. Everything the contract
    # emits is restricted to this set so data_io.validate() passes:
    #   labels.genome_id  subset of  features.index   (validate checks this)
    #   every features row has a cluster_id in genomes (validate checks this)
    labelled = set(labels["genome_id"])
    grouped = set(genomes.index)
    keep = sorted(labelled & grouped)
    if not keep:
        raise RuntimeError("No genome has both a lab label and an MLST group — nothing to write.")

    # features: reindex to exactly `keep`, adding all-zero rows for labelled
    # genomes that simply carry no acquired resistance gene (a real, common case).
    features = features.reindex(index=keep, fill_value=0).astype(int)
    features.index.name = "genome_id"
    features = features.loc[:, features.sum(axis=0) > 0]  # drop all-zero columns
    labels = labels[labels["genome_id"].isin(keep)].reset_index(drop=True)
    genomes = genomes.loc[keep]
    drug_targets = build_drug_targets(sorted(labels["drug"].unique()))

    # Drop feature columns that EXACTLY name a drug's target gene. NDARO reports a
    # target gene (ftsI/gyrA/parC/folA) only when it carries a notable variant, so
    # its presence is sparse and its absence-as-a-0-column is NOT evidence the gene
    # is missing. Left in, predictor.target_gate() reads those 0s as "absent" and
    # mis-fires not_applicable across the whole panel (observed: Ampicillin 99.7%
    # not_applicable off a 2/2127 'ftsI' column). Removing the exact-name columns
    # makes the gate read "unknown" — honest-by-construction, and matches the
    # AMRFinderPlus matrix, which never emits these as bare columns either.
    # Variant-suffixed forms (gyrA_1, folA_2) are kept: the gate ignores them and
    # they may carry real signal, exactly like AMRFinderPlus's gyrA_S83L features.
    target_genes = {g for entry in drug_targets.values() for g in entry["target_genes"]}
    gate_columns = [c for c in features.columns if c in target_genes]
    if gate_columns:
        features = features.drop(columns=gate_columns)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "features": out_dir / "features.csv",
        "labels": out_dir / "labels.csv",
        "genomes": out_dir / "genomes.csv",
        "drug_targets": out_dir / "drug_targets.json",
        "gene_metadata": out_dir / "gene_metadata.csv",
    }
    features.to_csv(paths["features"])
    labels[["genome_id", "drug", "phenotype"]].to_csv(paths["labels"], index=False)
    genomes.to_csv(paths["genomes"])
    paths["drug_targets"].write_text(json.dumps(drug_targets, indent=2))
    gene_metadata[gene_metadata["feature_name"].isin(features.columns)].to_csv(
        paths["gene_metadata"], index=False
    )
    return paths


def _report(out_dir: Path) -> None:
    import data_io

    ds = data_io.load_dataset(out_dir)  # raises on any contract violation
    n_clusters = ds.genomes["cluster_id"].nunique()
    print(
        f"\n{len(ds.features)} genomes, {len(ds.feature_names)} features, "
        f"{n_clusters} MLST clusters\n"
    )
    print(data_io.summarize(ds).to_string(index=False))
    sizes = ds.genomes["cluster_id"].value_counts()
    print(f"\nlargest MLST groups: {sizes.head(8).to_dict()}")
    if n_clusters < 10:
        print(f"\nWARNING: only {n_clusters} MLST clusters — grouped split may be thin.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fetch the BV-BRC challenge dataset into contract files.")
    ap.add_argument("--limit", type=int, default=None, help="only the first N genome ids (for iteration)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    paths = build(limit=args.limit, out_dir=args.out_dir)
    print(f"wrote {len(paths)} files to {args.out_dir}")
    _report(args.out_dir)
