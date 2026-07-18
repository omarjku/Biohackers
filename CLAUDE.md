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
    raw/                        # EMPTY — no real feature matrix yet
    synthetic/                  # seeded fixtures (seed=7): features/labels/genomes/drug_targets
  docs/DATA_CONTRACT.md   # column contract data_io.py validates against
  src/
    schemas.py            # Prediction / ExplanationResult dataclasses — the shared contract
    data_io.py            # contract loader + validation
    splits.py             # MinHash/Mash clustering + grouped train/calib/test split
    synth_data.py         # synthetic fixture generator
    predictor.py          # Module 02: per-drug logistic regression, target gate, evidence tiering
    explainer.py          # GPT-4 explanation generation
    genome_reader.py    * # Module 01: AMRFinderPlus -> feature matrix
    drug_database.py    * # antibiotic -> target gene mapping
    calibration.py      * # Platt scaling + no-call logic
    evaluation.py       * # metrics + plots
    app.py              * # Streamlit frontend
  tests/test_explainer.py # only module under test so far
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
`person-a/ml-pipeline` at `4a7b2d9`. If you are on another branch, treat file-level claims here as
a baseline and check the tree before relying on them.

Built and running: `schemas.py` (shared Prediction contract), `data_io.py` (contract loader +
validation, against `docs/DATA_CONTRACT.md`), `splits.py` (MinHash/Mash clustering, grouped
train/calibration/test split), `synth_data.py` + `data/synthetic/` (seeded fixtures, seed=7),
`explainer.py` + `tests/test_explainer.py`, `predictor.py` (per-drug logistic regression, target
gate, evidence tiering).

Still one-line TODO stubs, no implementation — each with its owner: `calibration.py` (Waji),
`genome_reader.py` and `drug_database.py` (Moncef, both blocking downstream work), `app.py` (UI
owner), `evaluation.py` (unassigned). `models/` is empty — nothing has been trained and saved.

**`data/raw/` is empty — there is no real feature matrix yet.** Labels exist in `data/` (E. coli
taxon 562; ampicillin, ciprofloxacin, trimethoprim). `labels_sampled.csv` is the working set:
2,400 rows, columns `genome_id,genome_name,antibiotic,phenotype,lab_method` — filter on
`lab_method` to keep laboratory-measured results only. `genome_id_list.csv` holds the 2,154 genome
ids still to fetch from BV-BRC; the six `<Drug> <Phenotype>.csv` files are the unsampled BV-BRC
pulls behind it. Everything currently runs on synthetic fixtures. Note the synthetic drugs
(Ceftriaxone/Ciprofloxacin/Gentamicin/Meropenem) are NOT the real label drugs — only Ciprofloxacin
overlaps.

### The headline number

Same model, random split vs. grouped split, balanced accuracy on synthetic data:

| drug | random | grouped | gap |
|---|---|---|---|
| Ciprofloxacin | 0.956 | 0.648 | **+0.308** |
| Gentamicin | 0.933 | 0.699 | +0.233 |
| Meropenem | 0.803 | 0.690 | +0.113 |
| Ceftriaxone | 0.925 | 0.923 | +0.003 |

This gap is the pitch. Report the grouped numbers as the real ones. Ceftriaxone shows the leak is
not universal — don't claim it is.

### Decisions made (don't silently reverse)

- Positive class y=1 is RESISTANT ("likely to fail"). Resistance is the event being detected.
- Mash threshold 0.05 (~95% ANI), deliberately over-merging. Rationale is inline in `splits.py`.
- Calibration is a third split, separate from train and test — Platt on train is biased, Platt on
  test leaks.
- Missing feature *column* means `target_gate_status="unknown"`, never `"absent"`. Absence of data
  is not absence of gene.
- `evidence_category` is never promoted to `known_gene_or_mutation` by coefficient size. Only genes
  in `drug_database.KNOWN_RESISTANCE_GENES` count; everything else is `statistical_association`.
- All drugs' target genes are excluded from displayed evidence (they are near-universal
  housekeeping genes, so their coefficients are artefacts). They stay in the feature matrix.
- `predictor.py` emits raw uncalibrated probability in `confidence` as a placeholder. Never report
  it — `calibration.py` overwrites it.

### Known gaps to fix

- **No de-duplication.** The brief asks for it; we do grouped *splitting*, which stops leakage, but
  large clusters still dominate training (biggest = 19% of train rows). Consider per-cluster
  sample weights.
- **`grouped_split()` balances cluster size, not label.** Ciprofloxacin lands at 40% resistant in
  train vs 70% in test. Metrics are unstable across seeds because of it; report across several
  seeds or make the allocator label-aware.
- **`drug_database.KNOWN_RESISTANCE_GENES` does not exist yet** — that exact symbol is what
  `predictor.py` imports. Until Moncef adds it, every prediction is `statistical_association`.

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
