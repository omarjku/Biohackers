# Genome Firewall

AI defense system predicting antibiotic resistance from bacterial genome data.
Hack-Nation Global AI Hackathon — Challenge 06 (OpenAI track).

## Pipeline

```
FASTA file
    |
    v
genome_reader.py + drug_database.py   (Module 01 — Person B)
    |
    v
predictor.py + calibration.py          (Module 02 — Person A)
    |
    v  list[schemas.Prediction]
    v
explainer.py                           (Module 03, NL layer — Hazem)
    |
    v  schemas.ExplanationResult
    v
app.py                                 (Module 03, UI — [UI person])
```

`evaluation.py` (Person D) runs offline against the held-out test set and
feeds the presentation deck, independent of the live app.

## The interface contract

`src/schemas.py` defines `Prediction` and `ExplanationResult` — the exact
data shape passed between modules. Everyone builds against this from hour 0;
nobody blocks on anyone else's module being finished. See
`data/synthetic/sample_predictions.json` for hand-written example predictions
covering every case (clean pass, clean fail, statistical-association-only,
two no-call variants, target-gate not-applicable) — use these to build and
test against before the real model exists.

Changing a field in `schemas.py` breaks every module downstream of it —
flag it to the team before editing.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your OpenAI key
pytest tests/ -v       # template-path tests, no API key needed
python src/explainer.py  # manual run against synthetic fixtures
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
