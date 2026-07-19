# Genome Firewall

AI defense system predicting antibiotic resistance from bacterial genome data.
Hack-Nation Global AI Hackathon — Challenge 06 (OpenAI track).

## Pipeline

```
FASTA upload
    |
    v  pipeline.py -> fasta_pipeline.py
genome_reader.py (AMRFinderPlus scan) + name-bridge   (Module 01)
    |                                                  + drug_database.py
    v  binary AMR feature row
predictor.py + calibration.py          (Module 02 — per-drug LR, Platt, no-call gate)
    |
    v  list[schemas.Prediction]
explainer.py  explain_report()         (Module 03, NL layer — bio + stat, evidence)
    |
    v  frontend JSON
app.py                                 (Module 03, Streamlit cards)
```

The offline pipeline scores genomes looked up by id from `data/processed/`
(built by `fetch_bvbrc.py`); the live path above scores an uploaded FASTA by
annotating it with AMRFinderPlus and bridging its gene names onto the same
feature vocabulary. `evaluation.py` runs offline against the held-out test set
and feeds the presentation deck, independent of the live app.

## The interface contract

`src/schemas.py` defines `Prediction` and `ExplanationResult` — the exact
data shape passed between modules. Everyone builds against this from hour 0;
nobody blocks on anyone else's module being finished. See
`data/synthetic/sample_predictions.json` for hand-written example predictions
covering every case (clean pass, clean fail, statistical-association-only,
two no-call variants, target-gate not-applicable) — the fixtures every module
is tested against.

Changing a field in `schemas.py` breaks every module downstream of it —
flag it to the team before editing.

## Scope & limitations (what we do and do NOT cover)

- **Species:** *Escherichia coli* only (single-species prototype).
- **Antibiotics:** Ampicillin, Ciprofloxacin, Trimethoprim.
- **Out of scope:** every other species and antibiotic; sample collection,
  sequencing, and genome reconstruction; and any organism design or modification
  (this tool is strictly defensive — it only predicts and explains resistance that
  already exists).
- **Known limitation:** the shipped features are acquired-gene calls and are
  mutation-blind, so **Ciprofloxacin** (driven by gyrA/parC point mutations) has
  lower resistant-recall (~0.76) and returns `no-call` more often — reported
  honestly, not hidden. Every result must be confirmed by standard lab testing.

Held-out performance (8 grouped MLST splits, 2,127 genomes): Ampicillin
bal-acc 0.94 · Ciprofloxacin 0.85 · Trimethoprim 0.92. See `reports_real_scaled/`.

## Run the demo

```bash
# one-time env (AMRFinderPlus + deps) and model-data build — see docs/LIVE_DEMO.md
streamlit run src/app.py     # upload a FASTA -> per-antibiotic report
```

`docs/LIVE_DEMO.md` covers the AMRFinderPlus conda env and building `data/processed`.

## Setup (tests / development)

```bash
pip install -r requirements.txt
cp .env.example .env   # optional OpenAI key (LLM explanations are off by default)
pytest tests/ -q       # 129 tests, no API key needed
```

To reproduce the real-data results from a fresh clone, follow
`docs/REPRODUCING.md` — it is the runbook, and it is explicit about which step
of the pipeline ships in the repo and which needs the FASTAs.

## Team roles

| Person | Module | Files |
|---|---|---|
| A (Pipeline Lead) | Module 02 — ML core, dedup, calibration, no-call | `predictor.py`, `calibration.py` |
| B (Biology/EEE) | Module 01 — genome reader, target lookup | `genome_reader.py`, `drug_database.py` |
| Hazem | Module 03, NL layer — GPT-4 + template explanations | `explainer.py`, `schemas.py` |
| UI person | Module 03, frontend | `app.py` |
| D | Evaluation + pitch | `evaluation.py`, presentation |
