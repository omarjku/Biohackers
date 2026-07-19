# ⚠️ Baseline only — do NOT quote these numbers

This directory holds the **pre-scaling** evaluation from the original ~119-genome
AMRFinderPlus matrix. Test sets are tiny (n_test 9–11 per drug), so several
metrics read as a naked `1.000 ± 0.000` — an artifact of small n, not real
performance.

**Quote `reports_real_scaled/` instead** — the current model over 2,127 genomes,
8 grouped MLST splits (Ampicillin bal-acc 0.930 ± 0.010, Ciprofloxacin
0.853 ± 0.011, Trimethoprim 0.915 ± 0.019). This directory is kept only for the
before/after scaling comparison.
