# Reproducing the results

Every number in `CLAUDE.md`, `reports_real/` and `reports_real_scaled/`, from a
fresh clone. Read `DATA_CONTRACT.md` first if you want the column-level shape;
this file is the runbook.

**There are two real datasets, and they are different feature vocabularies —
never compare a number across them without saying which is which.**

| | `reports_real/` | `reports_real_scaled/` |
|---|---|---|
| Built by | `adapt_real_data.py` from `files.zip` | `fetch_bvbrc.py` from the BV-BRC API |
| Annotator | AMRFinderPlus (local run) | BV-BRC `sp_gene`, source NDARO |
| Vocabulary | ALLELE symbols (`blaTEM-1`) | GENE-FAMILY tokens (`TEM`) |
| Size | 119 genomes · 124 features | 2,127 genomes · 139 features |
| Grouping | Mash single-linkage, 102 clusters | MLST sequence type, 444 clusters |
| Test slice | 9–11 rows | 174–384 rows |

`reports_real_scaled/` is the stronger result and the one to quote. §8 covers it
and `BVBRC_DATA.md` is its reference. Both write to `data/processed/`, so
**running one overwrites the other's input** — rebuild before re-evaluating.

There are two datasets and they are never mixed. **Real** (E. coli, BV-BRC,
lab-measured) produces every result we quote. **Synthetic** is a seeded fixture
generator for exercising code paths and for clearly-labelled methodology demos —
its labels come from a made-up rule, so a model trained on them learns that rule.
Never train across both: only 17 of the 124 real features exist in the 24-feature
synthetic set, only 1 of 4 synthetic drugs overlaps the real ones, and near-miss
names (`aac(6')-Ib-cr` vs `aac(6')-Ib-cr5`) align silently and wrongly.

---

## 0. Setup

```bash
pip install -r requirements.txt
python -m pytest tests/ -q          # 118 tests, no API key needed
```

An OpenAI key is only needed for `explainer.py`. Nothing in the reproduction
path below calls an API.

---

## 1. Real data — clone to numbers

```bash
python src/adapt_real_data.py                    # data/raw/ -> data/processed/
python -c "
from pathlib import Path
import sys; sys.path.insert(0, 'src')
from evaluation import run_full_evaluation
run_full_evaluation(Path('data/processed'), Path('reports_real'))
"
```

Expected after step one:

```
119 genomes, 124 features, 102 clusters

         drug  n  n_resistant  n_susceptible  pct_resistant  has_target_gate
   Ampicillin 41           23             18           56.1             True
Ciprofloxacin 51           12             39           23.5             True
 Trimethoprim 51           20             31           39.2             True
```

**If you see 3 clusters instead of 102, stop.** See §3 — the run will complete
and produce numbers, and they will not be the ones we report.

Step two writes `metrics.csv` (the 8-seed headline table), `metrics_seed0.csv`
(single split, for the plots only — not for quoting), `leakage_comparison.csv`
(grouped vs. random), and `evaluation.png` (four-panel dashboard, including the
reliability plot).

---

## 2. What ships in the repo, and what doesn't

Tracked (`data/raw/` is otherwise gitignored; these are explicit exceptions):

| File | What it is |
|---|---|
| `files.zip` | AMRFinderPlus outputs: `feature_matrix_real.csv`, `gene_metadata_real.csv`, `target_gate_real.csv`, `labels_final_urgent.csv` |
| `genome_clusters_mash.csv` | 102 sequence-derived clusters — **the one the results use** |
| `genome_clusters.csv` | 3 coarse clusters (60/58/1), phylogroup-level; fallback only, not usable |

Not tracked: `data/raw/fasta/` (187 MB compressed, over GitHub's 100 MB
per-file limit) and `data/processed/` (derived — rebuild it, don't commit it).

**So the reproducible boundary is `files.zip`, not the FASTAs.** Everything
downstream of AMRFinderPlus reproduces exactly from a clone. Regenerating
`files.zip` itself requires the FASTAs and a working AMRFinderPlus, and §4
covers what that takes. Nobody has run that path end to end on a clean machine.

---

## 3. The clustering file, and why it matters

`adapt_real_data.py` prefers `genome_clusters_mash.csv` and silently falls back
to `genome_clusters.csv` if it is missing. The fallback has 3 clusters of
60/58/1 — two usable groups cannot fill three splits, so the grouped
train/calibration/test split degenerates and the results are not comparable to
anything we report. `adapt_real_data.py` prints a warning below 10 clusters.
Heed it.

Regenerating the mash file needs the FASTAs, which are not in the repo:

```python
import sys; sys.path.insert(0, 'src')
from splits import cluster_genomes
cluster_genomes('data/raw/fasta').to_csv('data/raw/genome_clusters_mash.csv')
```

Budget ~19 minutes for 119 genomes. The MinHash is pure Python and O(n²) in
sketch comparisons; at 2,154 genomes that extrapolates to roughly 5.7 hours, so
scaling means swapping in the real `mash`/`sourmash` binary.

**Threshold 0.02 (~98% ANI) is deliberate and dataset-specific.** Re-derive it
on any new collection. Measured sweep on this data — clusters at each threshold:
0.05 → 2, 0.03 → 54, 0.02 → 102, 0.01 → 117. The 0.05 default (~95% ANI) is the
*species* boundary and does not survive single-species data: a quarter of real
E. coli pairs sit below it, so single-linkage chained 118 of 119 genomes into one
cluster. The sweep is inline in `splits.py`.

---

## 4. Rebuilding features from FASTA (not yet reproducible here)

Documented for whoever scales the dataset; treat the numbers as estimates.

1. Fetch genomes from BV-BRC (`bv-brc.org`) for the ids in
   `data/genome_id_list.csv` — 2,154 ids, of which **only 119 currently have
   FASTAs**. Estimated ~11 GB for the full set, and this step is manual: there
   is no FASTA download script. (`src/fetch_bvbrc.py` is not it — that pulls
   *precomputed annotations* over the API and skips FASTAs and AMRFinderPlus
   entirely. See §8. This section is the higher-fidelity path you would take to
   recover point mutations, which the API route structurally cannot see.)
2. Annotate:
   `python src/genome_reader.py --fasta-dir <dir> --out-dir <dir>`.
   Runs concurrently, caches each TSV so a failed batch resumes, and emits
   `features.csv` + `gene_metadata.csv`.
3. Re-cluster with `cluster_genomes()` (§3).
4. Re-run §1.

**AMRFinderPlus is not installed in this environment and Docker's daemon is not
running here.** The shipped matrix came from Moncef's machine. Validate the
toolchain on ~5 genomes before committing to a full run.

Use `lab_method` to keep laboratory-measured results only — BV-BRC's general
phenotype fields may be model-generated. `adapt_real_data.py` drops blank-method
rows and intermediate phenotypes rather than guessing them.

---

## 5. Synthetic path

```bash
python src/synth_data.py      # regenerate fixtures (seed=7)
python src/evaluation.py      # -> reports/
```

This produces the random-vs-grouped leakage table. **Label it a methodology
demonstration whenever it is shown** — it is not a result about E. coli, and it
does not reproduce on the real data (see `CLAUDE.md`).

---

## 6. Reading the output honestly

- **Multi-seed mean ± sd only.** `metrics_seed0.csv` exists for the plots and is
  labelled not-for-quoting. On the 119-genome set a single split swings wildly
  (test slices of 9–11 genomes); the scaled set is far steadier (174–384) but the
  rule stands.
- **`scoreable` is part of the result.** It counts seeds where the answered rows
  held both classes. Decision metrics are undefined otherwise and report as NaN
  rather than a flattering number.
- **Quote Ampicillin's 1.000s with the denominator attached** — that is
  `reports_real/`, roughly five answered rows per seed, which is what a
  near-perfect single-gene rule looks like at this sample size rather than a
  solved problem. The scaled run puts the same drug at a believable 0.930 ±0.010
  over 375 rows. Prefer the scaled figure; it is the one that survives scrutiny.
- **Say which cluster weighting a quoted leakage gap came from.**
  `weight_by_cluster` changes the conclusion, not just the decimals.
- **The leakage gap does not reproduce on real data, at either scale.** Scaled
  gaps are −0.001 / −0.012 / −0.030, i.e. grouped splits score the same or
  slightly better than random ones. Report that finding rather than leading with
  the synthetic table; "we measured our own headline claim and it did not hold
  here, and here is why" is a stronger position than a number that breaks under
  questioning. The why: this collection has little clonal redundancy, so a
  grouped split is nearly a random split. A set built from outbreak isolates
  would behave completely differently.

---

## 7. Biosecurity check

```bash
python verify_patch.py        # 13 PASS / 3 GAP / 0 FAIL
```

Run before any demo. No control is broken; the three GAPs are inert or
incomplete controls and print with owners. This system explains existing
resistance only — it never designs, modifies, or suggests changes to an organism.

---

## 8. Scaled real data — the numbers to quote

```bash
python src/fetch_bvbrc.py            # all 2,154 ids -> data/processed/
python -c "
from pathlib import Path
import sys; sys.path.insert(0, 'src')
from evaluation import run_full_evaluation
run_full_evaluation(Path('data/processed'), Path('reports_real_scaled'))
"
```

Reference: `docs/BVBRC_DATA.md`. No AMRFinderPlus, no Docker, no FASTA
download — responses cache per batch under `data/raw/bvbrc_cache/`, so a failed
batch resumes and re-runs are free. Expect 2,127 genomes · 139 features · 444
MLST clusters.

**This overwrites `data/processed/`.** To go back to the 119-genome
AMRFinderPlus set, re-run `python src/adapt_real_data.py` (§1).

Two caveats that are live as of 2026-07-19, both owned outside the pipeline:

- **Ciprofloxacin has no known-gene evidence on this feature set.** Thirteen of
  its eighteen curated genes are point mutations that `sp_gene` structurally
  cannot see — honest by construction. The other five (`qnrA1`, `qnrB6`,
  `qnrB19`, `qnrS1`, `aac(6')-Ib-cr5`) *are* in the source but are destroyed by
  `fetch_bvbrc._feature_token()`, which takes the first word of a product
  string and so collapses `Quinolone resistance protein QnrB10` to
  `Quinolone`. That part is a bug, not a limitation. Owner: Omar / Hazem.
- **`cluster_id` is MLST here, not Mash.** The two fail in opposite directions:
  Mash single-linkage over-merges (safe — over-merging cannot leak), MLST
  over-splits, because single-locus variants get distinct ST numbers and can
  land on both sides of a split. The ≈0 leakage gaps are consistent with
  genuinely low redundancy but *also* with ST failing to separate near-identical
  genomes. Unmeasured; worth one check before defending the grouped-split claim.
