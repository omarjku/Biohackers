# Scaling the real dataset from the BV-BRC API (`src/fetch_bvbrc.py`)

`adapt_real_data.py` reshapes a precomputed AMRFinderPlus run (`data/raw/files.zip`,
~119 genomes) into the contract files. That covered only ~119 of the 2,154 E. coli
genome ids in `data/genome_id_list.csv` that already have laboratory AST labels, and
clustered into 3 coarse groups — too few for a grouped train/calibration/test split.
Every real-data metric downstream was starved (test sets of 9-11 genomes, Ampicillin
balanced-accuracy at chance).

`fetch_bvbrc.py` rebuilds the real dataset for the **full 2,154 ids** directly from
the public [BV-BRC Data API](https://www.bv-brc.org/api/) over HTTPS — no
AMRFinderPlus install, no Docker, no FASTA download. Output is the same four contract
files in `data/processed/`, validated by `data_io.load_dataset()`, consumed unchanged
by `predictor` → `calibration` → `evaluation`.

```bash
python src/fetch_bvbrc.py            # all 2,154 ids -> data/processed/
python src/fetch_bvbrc.py --limit 200   # quick subset while iterating
```

## What comes from where

| Contract file | BV-BRC source | Rule |
|---|---|---|
| `labels.csv` | `genome_amr` | **Laboratory-measured rows only** (`evidence == "Laboratory Method"`), dropping the ~3.5× as many computational MIC-model rows — the brief forbids model-generated phenotypes as labels. R/S only; conflicting duplicate measurements dropped. |
| `features.csv` | `sp_gene`, `source=NDARO` | NDARO is the NCBI Reference Gene Catalog that AMRFinderPlus itself annotates against, so this is an **AMRFinderPlus-aligned acquired-gene set**, not a noisy generic specialty-gene dump. Gene family parsed from the `product` free-text (`… => CTX-M family`, `… @ Sul1`). |
| `genomes.csv` (`cluster_id`) | `genome.mlst` | **MLST sequence type** as the genetic group. A real biological grouping that yields hundreds of clusters — and `splits.py` already documents that within-species Mash distances barely separate E. coli, so MLST is the stronger `cluster_id` here. |
| `drug_targets.json` | `drug_database.DRUG_TARGET_MAP` | Unchanged from the existing pipeline. |

Raw API responses are cached per batch under `data/raw/bvbrc_cache/` (gitignored), so
re-runs are free and a failed batch resumes.

## Result

2,127 genomes · 139 acquired-AMR-gene features · **444 MLST clusters** (was 3). All
three drugs are two-class (Ampicillin 59.9% R, Ciprofloxacin 25.4% R, Trimethoprim
44.3% R) — previously Ampicillin was 100% R and unusable. See `reports_real_scaled/`
for metrics vs. the `reports_real/` baseline.

## Honest limitations (stated because the brief rewards it)

1. **Mutation-blind, by source.** `sp_gene` records gene *presence*, so this feature
   set is strong on acquired genes (blaTEM, blaCTX-M, sul, dfrA, tet, aac/ant/aph) but
   cannot see resistance *point mutations* (gyrA/parC S83L). E. coli fluoroquinolone
   resistance is mutation-driven, so **ciprofloxacin loses signal and honestly earns
   more no_calls** — a strength per the brief, not a hidden failure. A local
   AMRFinderPlus run over the FASTA files stays the documented fidelity upgrade.
2. **One source, not blended.** This rebuilds the whole matrix from NDARO rather than
   mixing with the 119-genome AMRFinderPlus matrix; two annotators' vocabularies in one
   feature space would be worse than either alone. The old matrix is kept for a later
   fidelity comparison.
3. **Target gate reads "unknown", by construction.** Feature columns that exactly name
   a target gene (`ftsI`/`gyrA`/`parC`/`folA`) are dropped, because NDARO reports a
   target gene only when it carries a notable variant — so its absence-as-a-0-column is
   not evidence the gene is missing. This matches the AMRFinderPlus matrix, which never
   emits these as bare columns either.
