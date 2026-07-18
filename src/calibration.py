"""
calibration.py — Platt scaling + no-call logic
Owner: Person A (Pipeline Lead)

Added AFTER predictor.py has trained models (needs a held-out calibration
set + Mash distances). Fills in the final `confidence` and `no_call_reason`
fields on each schemas.Prediction — see explainer.py, which already expects
these fields to exist (stub them with placeholders until this lands).

no_call triggers per the brief:
  1. calibrated probability in [0.3, 0.7]
  2. genome dissimilar from training data (high Mash distance)
  3. drug target not found (-> handled as "not_applicable", not "no_call")
"""

# TODO(Person A): CalibratedClassifierCV (Platt scaling) + no-call gate
