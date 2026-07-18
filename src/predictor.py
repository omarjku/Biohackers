"""
predictor.py — Module 02 core: features -> raw predictions
Owner: Person A (Pipeline Lead)

De-duplication (Mash clustering), one logistic regression per antibiotic.
Must output objects matching schemas.Prediction (evidence_category and
target_gate_status included) — calibration.py then adds calibrated
confidence + no_call_reason on top of this.
"""

# TODO(Person A): dedup, per-antibiotic LogisticRegression, produce raw Predictions
