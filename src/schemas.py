"""
schemas.py — THE INTERFACE CONTRACT

This file is the single source of truth for the data shape passed between modules:

    genome_reader.py + drug_database.py  (Person B)
                |
                v
    predictor.py + calibration.py        (Person A)
                |
                v  produces a list[Prediction]
                v
    explainer.py                         (Hazem — you)
                |
                v  produces explanation text
                v
    app.py                               (UI person)

Rule: nobody changes this file alone. If a field needs to change, say so in the
team channel first — everyone downstream of that field breaks silently otherwise.
"""

from pydantic import BaseModel, Field
from typing import Literal, Optional


#: The mandatory biosecurity disclaimer (non-negotiable rule 5). Defined here on
#: the shared contract so it is the single source of truth; explainer.py imports
#: it rather than re-declaring the wording.
DISCLAIMER = (
    "This is a research prototype. All results must be confirmed with "
    "standard laboratory testing."
)


class SupportingFeature(BaseModel):
    """One piece of genomic evidence backing a prediction."""
    gene: str
    mutation: Optional[str] = None
    note: Optional[str] = None  # e.g. "ESBL, hydrolyzes cephalosporins"


class Prediction(BaseModel):
    """
    One antibiotic's prediction for one genome.
    Produced by predictor.py / calibration.py. Consumed by explainer.py.
    """
    sample_id: str
    species: str
    drug: str

    call: Literal["likely_to_work", "likely_to_fail", "no_call", "not_applicable"]
    # "not_applicable" is set when target_gate_status == "absent" — it overrides
    # everything else, per the brief: absence of a drug's target must never be
    # reported as "likely to work".

    confidence: float = Field(ge=0.0, le=1.0)
    # Platt-scaled probability. Only meaningful once calibration.py is wired in;
    # until then, teams can stub this at 0.5 or any placeholder — explainer.py
    # doesn't care where the number came from, only that the field exists.

    evidence_category: Literal[
        "known_gene_or_mutation",   # (i) hard genomic evidence
        "statistical_association",  # (ii) model pattern, not a confirmed mechanism
        "no_known_signal",          # (iii) nothing found
    ]

    supporting_features: list[SupportingFeature] = []

    target_gate_status: Literal["present", "absent", "unknown"]

    no_call_reason: Optional[str] = None
    # Filled only when call == "no_call". Added by Person A after calibration
    # is wired up — until then, leave as None or a placeholder string.

    disclaimer: str = DISCLAIMER
    # Non-negotiable rule 5: every result must carry the "confirm with lab
    # testing" disclaimer. Defaulted so any consumer that renders a Prediction
    # directly (app.py, a JSON export, evaluation output) — not only the
    # explainer path — ships the disclaimer. Backward-compatible: existing
    # constructors omit it and get the default.


class ExplanationResult(BaseModel):
    """What explainer.py hands back to app.py."""
    sample_id: str
    drug: str
    explanation_text: str
    disclaimer: str
    confidence_label: str  # e.g. "78%"
