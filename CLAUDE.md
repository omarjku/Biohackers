# Genome Firewall

An AI defense system against superbugs — Hack-Nation Global AI Hackathon, Challenge 06 (OpenAI track), in
collaboration with MIT Club of Northern California and MIT Club of Germany.

Source docs, one level up in the parent directory: `../1784397351993-06-Hack-Nation-Genome-Firewall.docx-1-.pdf`
(official challenge brief) and `../genome_firewall_battleplan.pdf` (team's execution plan). Read these for full
detail — this file summarizes what matters for writing code.

## What we're building

Take a reconstructed, quality-checked bacterial genome (FASTA, ONE supported species) and predict, per
antibiotic: **likely to work / likely to fail / no-call**, each with a calibrated confidence score, an evidence
category, and the supporting genes/DNA changes. This is decision support only — never a treatment decision,
never organism design or modification.

## Scope

- IN SCOPE: quality-checked FASTA in → per-antibiotic prediction + confidence + evidence out.
- OUT OF SCOPE: sample collection, reading DNA from blood, species identification, genome reconstruction,
  separating multiple bacteria in one sample. Assume isolation/sequencing/assembly are already done.
- Strictly defensive: never generate, design, modify, or suggest changes to an organism.

## Architecture (3 modules + evaluation)

1. **Genome Reader** (`genome_reader.py`) — FASTA → binary AMR feature matrix. Default annotation tool:
   AMRFinderPlus (NCBI, public-domain). One row per genome, one column per AMR gene/mutation.
2. **Predictor** (`predictor.py`, `calibration.py`, `drug_database.py`) — features → per-antibiotic prediction.
   - Baseline model: one L2-regularized logistic regression per antibiotic (CPU, fast, interpretable).
   - Deterministic target gate: if the antibiotic's molecular target gene is absent, flag "not applicable" —
     never report "likely to work" from absence of resistance markers alone.
   - De-duplication before train/test split: cluster by sequence homology (e.g. Mash ~95% identity) so
     near-identical genomes never leak across train/test. Threshold choice must be documented and justified.
   - Calibration: Platt scaling on a held-out calibration set; validate with reliability plots.
   - No-call gate: return no-call when calibrated probability is in an ambiguous band (e.g. 0.3–0.7), the
     genome is out-of-distribution vs. training data, or the drug target is missing.
3. **Decision Report** (`app.py`, `explainer.py`) — Streamlit or Gradio app. Per-antibiotic card: drug, verdict,
   calibrated confidence, evidence category — (i) known resistance gene/DNA change detected, (ii) statistical
   association only, (iii) no known resistance signal. Every page must show a mandatory disclaimer: "This is a
   research prototype. All results must be confirmed with standard laboratory testing." GPT-4/OpenAI API may
   be used to turn structured predictions into clinician-readable explanations (genuine value-add, not a gimmick).
4. **Evaluation** (`evaluation.py`) — offline metrics on held-out, grouped test data.

Actual layout (as of 2026-07-19; paths are relative to this file's directory, `*` = still a TODO stub):

```
  data/
    labels_sampled.csv          # 2,400 rows, the working label set
    genome_id_list.csv          # 2,154 genome ids to fetch
    {Ampicillin,Ciprofloxacin,Trimethoprim} {Resistant,Susceptible}.csv
    raw/                        # files.zip (real AMRFinderPlus matrix + metadata),
                                # genome_clusters.csv, genome_clusters_mash.csv, fasta/ (gitignored)
    processed/                  # the four contract files, built by adapt_real_data.py (gitignored)
    synthetic/                  # seeded fixtures (seed=7): features/labels/genomes/drug_targets
  docs/DATA_CONTRACT.md   # column contract data_io.py validates against
  src/
    schemas.py            # Prediction / ExplanationResult dataclasses — the shared contract
    data_io.py            # contract loader + validation
    splits.py             # MinHash/Mash clustering + grouped train/calib/test split
    synth_data.py         # synthetic fixture generator
    adapt_real_data.py    # Moncef's real outputs -> the four contract files
    predictor.py          # Module 02: per-drug logistic regression, target gate, evidence tiering
    calibration.py        # Platt scaling (+OOF below 40 rows) + no-call gate + OOD envelope
    evaluation.py         # multi-seed metrics, per-cluster breakdown, leakage comparison, dashboard
    explainer.py          # GPT-4 explanation generation
    genome_reader.py      # Module 01: AMRFinderPlus -> feature matrix (has a CLI)
    drug_database.py      # antibiotic -> target gene + curated resistance genes
    app.py              * # Streamlit frontend
  tests/                  # 101 tests: splits, predictor, calibration, genome_reader, explainer
  reports/                # synthetic evaluation output
  reports_real/           # real-data evaluation output
  models/                 # EMPTY — nothing trained/saved yet
  requirements.txt
  README.md
```

## Non-negotiable rules

1. Never touch resistance-gene knowledge to *design* an organism — this system explains existing resistance
   only.
2. Always evaluate on a **grouped** split (by genetic similarity/cluster), never a random row split — random
   splits leak near-identical genomes across train/test and inflate scores. This is a common failure mode judges
   explicitly penalize.
3. Never present a raw SHAP/feature-importance score as proof of biological causation. Separate "known
   resistance gene/mutation" evidence from "statistical association only" evidence.
4. Every prediction needs a real no-call option — don't force yes/no on weak or conflicting evidence.
5. The mandatory "confirm with standard lab testing" disclaimer must appear on every result, always.
6. Prefer doing one species + a few antibiotics well (calibrated, honest, no-call-aware) over claiming broad
   coverage with forced answers.

## Metrics to report (on held-out data)

- Balanced accuracy per antibiotic, plus recall for resistant and susceptible cases separately.
- F1, AUROC, PR-AUC per drug (PR-AUC matters under class imbalance).
- Brier score + reliability plot; no-call rate and accuracy of the remaining (non-no-call) predictions.
- Performance broken down by genetically related group, including unseen groups where possible.

## Data & tools

- Primary genome/resistance data source: BV-BRC (ex-PATRIC), `bv-brc.org`. Use organizer-pinned,
  laboratory-measured test results — not general phenotype fields (may be model-generated).
- Default annotator: AMRFinderPlus (`github.com/ncbi/amr`). Alternatives referenced in the brief: ResFinder,
  cAMRah, XTree.
- Stack: Python, scikit-learn, AMRFinderPlus, Streamlit, OpenAI GPT-4 API ($50/team credit available).
- Kaggle mirrors are tutorial-only — never treat as a verified benchmark without documented license/source.

## Current status (2026-07-19)

Status below is repo-wide, not one person's worklist — it was last refreshed from branch
`person-a/real-data-integration` (branched off `origin/main` at `54e81ef`). If you are on another
branch, treat file-level claims here as a baseline and check the tree before relying on them.

**Real data has landed and the pipeline runs end to end on it.** Moncef's AMRFinderPlus run
produced `data/raw/files.zip` (feature matrix, gene metadata, target-gate results) plus
`data/raw/genome_clusters.csv`. `src/adapt_real_data.py` converts those into the four contract
files in `data/processed/`: **119 E. coli genomes x 124 AMR features, 143 label rows**
(Ampicillin 41, Ciprofloxacin 51, Trimethoprim 51). Run `python src/adapt_real_data.py`, then
evaluate with `run_full_evaluation(Path("data/processed"), Path("reports_real"))`.

The dataset is small enough that this is a working pipeline, not a competitive model — see
"What the real numbers actually say" below before quoting anything.

Built and running: `schemas.py` (shared Prediction contract), `data_io.py` (contract loader +
validation, against `docs/DATA_CONTRACT.md`), `splits.py` (MinHash/Mash clustering, grouped
train/calibration/test split), `synth_data.py` + `data/synthetic/` (seeded fixtures, seed=7),
`explainer.py` + `tests/test_explainer.py`, `predictor.py` (per-drug logistic regression, target
gate, evidence tiering).

Also built: `calibration.py` (Platt scaling on the third split, no-call gate, OOD envelope),
`evaluation.py` (per-drug metrics, per-cluster breakdown, random-vs-grouped comparison, and a
four-panel dashboard — run `python src/evaluation.py`, writes to `reports/`), and
`verify_patch.py` at the repo root (biosecurity compliance harness: de-duplication, target gate,
LLM constraints, disclaimer — run `python verify_patch.py`).

`genome_reader.py` and `drug_database.py` are implemented (Moncef, plus fixes below).
`genome_reader.py` now has a CLI — `python src/genome_reader.py --fasta-dir <dir> --out-dir <dir>`
— that annotates concurrently, caches each TSV so a failed batch resumes, and emits both
`features.csv` and `gene_metadata.csv`. **`app.py` is still a one-line stub (UI owner)** and
`models/` is still empty.

Labels live in `data/` (E. coli taxon 562). `labels_sampled.csv` is the working set: 2,400 rows,
columns `genome_id,genome_name,antibiotic,phenotype,lab_method` — filter on `lab_method` to keep
laboratory-measured results only. `genome_id_list.csv` holds 2,154 genome ids; **only 119 have
FASTAs so far**, so scaling up means fetching the rest from BV-BRC and running the
`genome_reader.py` CLI over them. Note the synthetic drugs
(Ceftriaxone/Ciprofloxacin/Gentamicin/Meropenem) are NOT the real label drugs — only Ciprofloxacin
overlaps.

**Never train on synthetic and real data together.** Measured: only 17 of the 124 real features
exist in the 24-feature synthetic set, only 1 of 4 synthetic drugs overlaps the real ones, and
near-miss names (`aac(6')-Ib-cr` vs `aac(6')-Ib-cr5`) would align silently and wrongly. More
fundamentally `synth_data.py` is a fixture generator seeded at 7 to exercise code paths, never a
validated biological simulator — its labels come from a made-up rule, so a model trained on them
learns that rule. Synthetic is for test fixtures and clearly-labelled methodology demos only.

### The headline number

**This table is SYNTHETIC. It does not reproduce on real data — see the next section. Label it
as a methodology demonstration whenever it is shown, never as a result about E. coli.**

Same model, random split vs. grouped split, balanced accuracy on synthetic data, mean ± sd over 8
seeds. Regenerate with `python src/evaluation.py`.

| drug | random | grouped | gap |
|---|---|---|---|
| Meropenem | 0.770 ±0.066 | 0.676 ±0.039 | **+0.094** |
| Ciprofloxacin | 0.902 ±0.030 | 0.842 ±0.049 | **+0.060** |
| Gentamicin | 0.946 ±0.025 | 0.887 ±0.035 | **+0.059** |
| Ceftriaxone | 0.934 ±0.028 | 0.911 ±0.014 | +0.023 |

The gap is the pitch, but state it honestly: it is modest, and it is largest on the drug we predict
worst (Meropenem, grouped balanced accuracy 0.676 — barely above useful). Report the grouped
numbers as the real ones.

**This table was re-measured on 2026-07-19 after per-cluster sample weights landed (`5bc80ae`),
and it replaces the previous version** (Meropenem +0.132, Gentamicin +0.109, Ciprofloxacin 0.000,
Ceftriaxone −0.001). That version was not wrong when written — it was measured with
`weight_by_cluster=False`, which was the only behaviour that existed then. Setting that flag False
today still reproduces it (Meropenem +0.144, Gentamicin +0.110, Ciprofloxacin −0.018, Ceftriaxone
−0.002), so the two tables are the same experiment under different weighting, not a bug fix.

What changed and why it matters: de-duplication weighting moves *both* columns, and it moved the
qualitative story. Under the old unweighted default the gap was real for two drugs and absent for
two — the honest caveat was "it doesn't reproduce everywhere." With cluster weighting on, every
drug shows a positive gap, because down-weighting redundant clusters costs the grouped model some
accuracy on the drugs whose signal came from a few large clades (Ciprofloxacin grouped 0.900 →
0.842) while raising the leaky random baseline. **Always state which weighting a quoted gap came
from** — the flag changes the conclusion, not just the decimals.

**These numbers replace an earlier, inflated table** (Ciprofloxacin +0.308, Gentamicin +0.233).
That version was measured before `grouped_split()` was label-aware, when the allocator was leaving
Ciprofloxacin at 40% resistant in train against 70% in test. Most of that apparent "leakage
penalty" was the grouped model being scored against a differently-balanced test set, not leakage.
Do not resurrect the old figures — if anyone quotes +0.308, it came from our own bug. Always
average over several seeds; single-seed gaps still swing by ±0.05.

### What the real numbers actually say

Real data, 8 grouped seeds. Regenerate with `run_full_evaluation(Path("data/processed"),
Path("reports_real"))`. `scoreable` counts seeds where the answered rows held BOTH classes —
decision metrics are undefined otherwise and are reported as NaN rather than a flattering number.

| drug | coverage | bal_acc | scoreable | recall_R | AUROC | brier raw → cal |
|---|---|---|---|---|---|---|
| Ampicillin | 25.0 ±17.4 | 0.500 ±0.000 | **1 of 8** | 1.000 | 0.894 ±0.104 | 0.194 → 0.171 |
| Ciprofloxacin | 80.7 ±8.4 | 0.821 ±0.175 | 7 of 8 | 0.643 ±0.350 | 0.859 ±0.144 | 0.183 → 0.108 |
| Trimethoprim | 42.5 ±22.2 | 0.700 ±0.245 | 5 of 8 | 0.400 ±0.490 | 0.940 ±0.062 | 0.193 → 0.142 |

How to read this honestly:

- **The models are real.** AUROC 0.86–0.94 means the biology is being learned. Calibration now
  improves Brier on all three drugs.
- **The decisions are weak and seed-dependent.** Test slices are 9–11 genomes, so those standard
  deviations are the story, not the means. Never quote a single split.
- **Ampicillin is degenerate.** It only ever answers genomes that are truly resistant — measured
  over 8 seeds, ZERO test rows ever landed below the 0.30 no-call floor (p5 of its calibrated
  distribution is 0.368). So its answered slice is single-class almost always, `recall_S` is 0 by
  construction, and `bal_acc` 0.500 is the honest reading. When it does answer resistant it is
  right 23 of 24 times. The fix is more data, not a wider band.
- **`recall_R` is the weak side everywhere.** The models miss resistant cases far more often than
  they misclassify susceptible ones. In this domain that is the dangerous direction.

**The leakage gap does NOT reproduce on real data.** Measured mean gap is ≈ −0.01 with enormous
spread (±0.234 on Ciprofloxacin random). That is not a failure — it follows from the clustering:
only ONE genome pair sits below Mash 0.002, so this sample has almost no clonal redundancy and a
grouped split is nearly a random split. A collection built from outbreak isolates would behave
completely differently. Say this out loud rather than hiding it; "we measured our own headline
claim and it didn't hold here, and here is why" is a stronger position than a number that breaks
under questioning.

### Decisions made (don't silently reverse)

- Positive class y=1 is RESISTANT ("likely to fail"). Resistance is the event being detected.
- **Mash threshold 0.02 (~98% ANI)**, still deliberately over-merging. Was 0.05 (~95% ANI, the
  *species* boundary), which does not survive single-species data: a quarter of all real E. coli
  pairs sit below 0.05, so single-linkage chained 118 of 119 genomes into ONE cluster and no
  grouped split was possible. Measured sweep (clusters at each threshold: 0.05→2, 0.03→54,
  0.02→102, 0.01→117) is inline in `splits.py`. Re-derive this on any new dataset.
- **Calibration is a held-out third split whenever that split is big enough**, and below
  `MIN_CALIBRATION_ROWS = 40` it is augmented with out-of-fold predictions over the training rows
  (grouped folds, so no genome is scored by a model that saw its cluster). Platt on train is still
  biased and Platt on test still leaks — neither is what this does. Why the change: the real
  calibration slices are 7–8 rows, and a sigmoid fitted on that collapses onto the slice's own base
  rate, which caused 100% no-call on two drugs and `recall_R` exactly 0.000 on the third while the
  same raw probabilities scored 0.875/0.833/0.771 balanced accuracy at a plain 0.5 threshold. Why
  it is conditional: out-of-fold rows come from models trained on less data, so Platt over-sharpens
  — on synthetic (61–77 row slices) unconditional pooling made Ciprofloxacin's Brier WORSE, 0.1559
  → 0.1687. That regression set the threshold and
  `tests/test_calibration.py::test_calibration_improves_brier_on_held_out_test` pins it. **Do not
  weaken that test to make a change pass — it is the canary.**
- **Report multi-seed mean ± sd, never a single split.** `evaluation.multi_seed_metrics()` is the
  headline table; the seed-0 table is printed only for the dashboard plots and is labelled
  not-for-quoting. At 9–11 test rows per drug a single split swings wildly.
- **Decision metrics are NaN when the answered slice is single-class.** `balanced_accuracy_score`
  does not raise on single-class `y_true` — it silently degrades to plain accuracy. That reported
  ampicillin at 0.917 alongside `recall_R` 1.000 and `recall_S` 0.000, three numbers that cannot
  all be true. Guarded by `evaluation._decision_metrics_defined()`.
- **Gene-family aggregation is available but OFF** (`fit_drug_model(aggregate_families=False)`).
  Collapsing allelic variants (`blaTEM-1/-12/-30` → `blaTEM`) is biologically sound and widens
  probability spread, but measured end-to-end it helps Trimethoprim on every metric and hurts the
  other two on every metric, and it doubles synthetic Ciprofloxacin's raw Brier because
  `synth_data.py` plants signal in specific alleles. Full table in `predictor.gene_family`'s
  docstring. Re-measure before enabling; do NOT enable per-drug on the strength of that table,
  which would be selecting a model on test results.
- Missing feature *column* means `target_gate_status="unknown"`, never `"absent"`. Absence of data
  is not absence of gene.
- `evidence_category` is never promoted to `known_gene_or_mutation` by coefficient size. Only genes
  in `drug_database.KNOWN_RESISTANCE_GENES` count; everything else is `statistical_association`.
- All drugs' target genes are excluded from displayed evidence (they are near-universal
  housekeeping genes, so their coefficients are artefacts). They stay in the feature matrix.
- `predictor.py` emits raw uncalibrated probability in `confidence` as a placeholder. Never report
  it — `calibration.py` overwrites it.

### Known gaps to fix

- **THE binding constraint: 143 label rows.** Everything weak about the model is downstream of
  this. Only 119 of the 2,154 genome ids have FASTAs, giving 25–33 training rows and 7–8
  calibration rows per drug. Nothing in the modelling layer substitutes for more genomes — fetch
  the rest from BV-BRC and run the `genome_reader.py` CLI. Estimated cost at 2,154 genomes: ~11 GB
  download, and note `splits.py`'s pure-Python MinHash took 19 minutes for 119 genomes (≈5.7 hours
  extrapolated), so scaling also means swapping it for the real `mash`/`sourmash` binary.
- ~~No de-duplication.~~ **Done.** `cluster_sample_weights()` weights each genome by 1/(cluster
  size) and equalises class weight on top, so every cluster counts once regardless of how often it
  was sequenced. Note this already applies class balancing — do NOT add `class_weight="balanced"`
  on top, it would be applied twice.
- **AMRFinderPlus is not installed in this environment** and Docker's daemon is not running. The
  119-genome matrix came from Moncef's machine. Anyone scaling the dataset needs the toolchain
  working locally first — validate on ~5 genomes before committing to a full run.
- ~~`grouped_split()` balances cluster size, not label.~~ **Fixed.** The allocator now sends each
  cluster to the split furthest behind on its per-class targets, weighted by the cluster's own
  class mix. Ciprofloxacin went from 40/35/70 to 45.4/44.9/45.7 percent resistant across
  train/calibration/test, split sizes still ~65/15/20. Do NOT rewrite the deficit as a
  squared-deviation-from-target cost: that rewards hitting a target, so the small splits fill
  first and train collapses to ~34% of rows. Measured, and pinned by `tests/test_splits.py`.
  (Relative vs. absolute class deficits, by contrast, were measured equivalent — spread 0.023 vs
  0.021. An earlier note here claimed absolute deficits starve the small splits; that was
  asserted without measurement and was wrong.)
- ~~`drug_database.KNOWN_RESISTANCE_GENES` does not exist.~~ **Done.** 25 ampicillin, 18
  ciprofloxacin, 8 trimethoprim genes, derived from AMRFinderPlus `amr_class`/`amr_subclass` then
  narrowed by hand; all 51 resolve against the real matrix. Nonspecific efflux/porin regulators
  (`acrR`, `marR`, `ompC`) and cefiderocol-uptake `cirA` truncations are deliberately excluded and
  report as `statistical_association`.
- **The target gate cannot fire on this feature matrix, for any of the three drugs.** Not a bug:
  AMRFinderPlus only reports an essential gene when it is mutated or disrupted, so no bare
  `ftsI`/`gyrA`/`parC`/`folA` column exists and `predictor.target_gate()` correctly returns
  `"unknown"` rather than `"absent"`. Present it as honest-by-construction, not as a filter doing
  active work. (`DRUG_TARGET_MAP`'s ampicillin entry was `pbp3,pbp1A,pbp1B,pbp2` — AMRFinderPlus
  emits none of those symbols in any form; it calls PBP3 `ftsI`. Fixed.)
- **`genome_reader.check_target_gate()` is deprecated dead code** with two bugs (title-case drug
  keys that never match the lowercase labels, and no `"unknown"` branch so an unseen genome falls
  through to `"present"`). `predictor.target_gate()` is the real one. Do not wire the other up.

## Team roles and file ownership

The team works this repo in parallel on separate branches. Ownership is by file:

| Owner | Scope | Owns |
|---|---|---|
| **Waji** (`syedwajiulhassan715-rgb`) | ML pipeline: de-duplication, per-drug logistic regression, calibration, no-call logic | `data_io.py`, `splits.py`, `synth_data.py`, `predictor.py`, `calibration.py` |
| **Moncef** (`Moncefzack`) | Genome Reader + biology: BV-BRC data pulls, AMRFinderPlus setup, FASTA parsing, feature matrix, drug-target lookup, biological validation | `genome_reader.py`, `drug_database.py`, `data/*.csv` |
| **Hazem** (`Hazem Kassem`) | Interface contract + GPT-4 explanation layer | `schemas.py`, `explainer.py`, `tests/test_explainer.py` |
| **UI owner** | Streamlit frontend, visualizations | `app.py` |
| Unassigned | Evaluation output, metrics, plots | `evaluation.py` |

Notes on reading `git log` here: Hazem's initial commit created the five one-line TODO stubs
(`app.py`, `calibration.py`, `drug_database.py`, `evaluation.py`, `genome_reader.py`) as scaffolding
for other people — **authoring a stub does not mean owning the implementation**, so use the table
above, not `git blame`, to decide whose file something is. `omarjku` owns the GitHub repo
(`omarjku/Biohackers`) and PR merges route through his fork, which makes branches look like his work;
he has authored no commits so far.

Shared, changed by agreement only: `schemas.py` (the cross-module contract — Hazem's file, but
everyone depends on it), `docs/DATA_CONTRACT.md`, `requirements.txt`, this file.

### If you are working with Claude on this repo

- **Do the task in front of you, not the whole backlog.** The "Known gaps to fix" and the stub list
  are a repo-wide picture so you understand how your piece fits — they are not a to-do list for
  whoever reads them first. Someone else is already assigned to each one.
- **Don't implement or refactor another owner's file to unblock yourself.** If you need something
  that doesn't exist yet (e.g. `drug_database.KNOWN_RESISTANCE_GENES`), code against the documented
  interface, stub it locally if you must, and flag the dependency — don't fill it in on their
  branch's behalf. Two people implementing the same file is the main merge risk here.
- **`schemas.py` is the seam between modules.** Read it before touching anything cross-module.
  Changing a field there breaks other people's in-flight work, so propose it rather than doing it.
- **The non-negotiable rules and the "Decisions made" list are team-wide and already settled.**
  They apply to every branch. If you think one is wrong, raise it — don't quietly reverse it in
  your module.
- **Keep this file current when the picture changes** — a module going from stub to working, a new
  decision, a resolved gap. It's how everyone else's Claude stays accurate.
