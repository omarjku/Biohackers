"""
drug_database.py — antibiotic -> molecular target lookup
Owner: Person B (Biology/EEE)

Deterministic gate: if a drug's target gene is absent from the genome,
predictor.py must set call="not_applicable", target_gate_status="absent"
(see schemas.py) rather than "likely_to_work".
"""

# TODO(Person B): drug -> target gene/protein lookup table
DRUG_TARGET_MAP: dict[str, str] = {
    # "Ceftriaxone": "penicillin-binding proteins",
    # ...
}
