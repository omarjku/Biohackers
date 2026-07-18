# Input Data Contract (Module 01 → Module 02)

`src/schemas.py` defines what comes **out** of the pipeline (`Prediction`).
This document defines what goes **in** — the exact files Person B (Module 01) hands to
Person A (Module 02).

Person A builds against these four files. Synthetic data (`src/synth_data.py`) produces
files in exactly this shape, so the ML pipeline is built and tested before real data
exists. When real data arrives it is a path change, not a rewrite.

**If a field here needs to change, say so in the team channel first** — same rule as
`schemas.py`.

---

## 1. `features.csv` — the AMR feature matrix

One row per genome, one column per AMR gene or mutation. Values are binary.

| genome_id | blaCTX-M-15 | blaTEM-1 | gyrA_S83L | aac(3)-IIa | ... |
|---|---|---|---|---|---|
| GEN-0001 | 1 | 0 | 1 | 0 | ... |
| GEN-0002 | 0 | 0 | 0 | 1 | ... |

- `genome_id` — string, unique, matches the other files.
- Every other column — `0` (not detected) or `1` (detected). No missing values.
- Column names come from the AMRFinderPlus **Element symbol** field. Point mutations use
  `gene_mutation` form (e.g. `gyrA_S83L`).
- Column names must be stable across the train and test sets. A feature seen only at
  prediction time is ignored; a feature missing at prediction time is treated as `0`.

## 2. `labels.csv` — resistance phenotypes (long format)

| genome_id | drug | phenotype | mic |
|---|---|---|---|
| GEN-0001 | Ceftriaxone | R | 32 |
| GEN-0001 | Ciprofloxacin | S | 0.03 |
| GEN-0002 | Ceftriaxone | S | 0.12 |

- `phenotype` — `R` (resistant → "likely to fail") or `S` (susceptible → "likely to work").
  Intermediate `I` should be resolved or dropped by Person B, not passed through.
- `mic` — optional, informational only. The model trains on `phenotype`.
- **Long format is deliberate.** Not every genome is tested against every drug in BV-BRC,
  so a wide matrix would be full of holes. Long format represents "not tested" by simply
  having no row, which is exactly right.
- Use organizer-pinned, laboratory-measured results only — never general phenotype
  fields, which may contain model-generated predictions.

## 3. `genomes.csv` — metadata and genetic grouping

| genome_id | species | cluster_id |
|---|---|---|
| GEN-0001 | Escherichia coli | CL-07 |
| GEN-0002 | Escherichia coli | CL-07 |

- `species` — needed for the `species` field of `Prediction`.
- `cluster_id` — **the most important column in this file.** Genomes that are near-identical
  share a `cluster_id`. Splitting is done by cluster so that near-identical genomes never
  land in both train and test.

**Where `cluster_id` comes from, in priority order:**

1. Organizer-provided cluster/group IDs, if the challenge dataset includes them. These are
   authoritative — use them and say so in the writeup.
2. Otherwise Person A computes them from the FASTA files via MinHash sketching
   (`src/splits.py`).

→ **Person B: if the organizer files contain no cluster IDs, Person A needs access to the
FASTA files.** The feature matrix alone cannot produce genome similarity. This is the one
dependency that can silently invalidate the grouped-split results.

## 4. `drug_targets.json` — the deterministic target gate

```json
{
  "Ceftriaxone": {
    "target_genes": ["ftsI", "mrdA"],
    "note": "penicillin-binding proteins"
  },
  "Ciprofloxacin": {
    "target_genes": ["gyrA", "parC"],
    "note": "DNA gyrase / topoisomerase IV"
  }
}
```

- Maps each drug to the gene(s) encoding its molecular target.
- Drives the deterministic gate: if a drug's target is absent from the genome, the result is
  `not_applicable` with `target_gate_status="absent"` — never `likely_to_work`. The brief
  requires this explicitly, so that "no resistance gene found" is never confused with "this
  drug will work".
- If a drug is missing from this file, its `target_gate_status` is `"unknown"` and the
  pipeline still predicts — but the gate cannot protect that drug, so coverage should be
  stated honestly in the writeup.

---

## File locations

```
data/
  raw/          # organizer files, gitignored
  processed/    # the four files above, gitignored
  synthetic/    # generated equivalents, committed as test fixtures
```

## Validation

`src/data_io.py` loads and validates all four files, and raises a clear error on contract
violations (unknown genome_id, non-binary features, phenotype outside {R,S}, genomes
missing a cluster). Run it against real data the moment it lands — it is designed to fail
loudly and early rather than silently producing a broken model.
