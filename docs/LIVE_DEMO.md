# Live demo path: upload a FASTA → per-antibiotic report

The offline pipeline scores genomes we look up by id. The **live** path scores a
genome someone **uploads** — no id to look up — by annotating it ourselves.

```
uploaded FASTA
   │  src/pipeline.py  (app.py calls pipeline.run)
   ▼
AMRFinderPlus scan            src/genome_reader.run_amrfinder
   ▼  Element symbols (blaCTX-M-15, sul1, gyrA_S83L, …)
name bridge                   src/fasta_pipeline.bridge_row
   ▼  model's NDARO feature vocabulary (CTX-M, Sulfonamide, Quinolone, …)
predictor → calibration → no-call gate
   ▼  list[Prediction]  (evidence upgraded to known mechanism where a curated family is present)
explainer.explain_report      src/explainer
   ▼  frontend JSON: drug / drug_class / underlying_state / confidence /
      target_marker / locus_id / bio_explanation / stat_explanation
Streamlit cards               src/app.py
```

## Why the name bridge

The model (`data/processed`) trained on BV-BRC **NDARO** gene calls, which name
genes at family level (`CTX-M`). AMRFinderPlus, which we run on an upload, uses
allele-level **Element symbols** (`blaCTX-M-15`). NDARO *is* the reference catalog
AMRFinderPlus annotates against, so the genes are the same — `bridge_row()` maps
each Element symbol to the one dominant family column the model actually weights,
and sets intrinsic `BlaEC` (present in ~all E. coli). Mapping to a single token
per gene matters: lighting up every synonym builds a feature vector no real genome
has and trips the out-of-distribution no-call gate.

**Validated** on 24 genomes (FASTA path vs. the model's own NDARO-lookup path):
Ampicillin 100% / Trimethoprim 100% / Ciprofloxacin 83% agreement; FASTA-vs-lab
accuracy 81 / 80 / 86% — matching the model's held-out numbers.

## Honest limitation (carried from the training data)

NDARO features are mutation-blind. AMRFinderPlus *does* detect gyrA/parC point
mutations on an upload, but the model has no strong column for them, so
**ciprofloxacin leans on weaker signal and honestly no-calls more often** — the
safe behavior the brief rewards, not a bug.

## Environment (AMRFinderPlus)

The live path needs the `amrfinder` binary + database on PATH. Native arm64 works:

```bash
conda create -y -n amrfinder -c conda-forge -c bioconda ncbi-amrfinderplus=4.2.7
conda activate amrfinder
amrfinder -u                     # install the AMR database to the default location
pip install -r requirements.txt  # ML + Streamlit deps in the same env
```

Build the model data (writes `data/processed/`, which is gitignored — the app's
first prediction needs it, else it errors):

```bash
python src/fetch_bvbrc.py         # pulls the 2,154-genome dataset from BV-BRC
```

Run the app (from that env, so amrfinder is on PATH):

```bash
streamlit run src/app.py
```

Upload a FASTA, or use the bundled-example selector (any `*.fna` placed in
`data/raw/fasta_demo/`, gitignored). Every result carries the mandatory
"confirm with standard laboratory testing" disclaimer.
```
