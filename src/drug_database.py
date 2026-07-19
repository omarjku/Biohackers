
"""
drug_database.py — antibiotic -> molecular target lookup
Owner: Moncef (biology)
Deterministic gate: if a drug's target gene is absent from the genome,
predictor.py must set call="not_applicable", target_gate_status="absent"
(see schemas.py) rather than "likely_to_work".
"""

# Target genes match the "Element symbol" / gene_symbol values AMRFinderPlus
# reports (see gene_metadata_real.csv). Multiple genes = drug has multiple
# valid targets; gate passes if ANY one of them is intact (not disrupted).
#
# NOTE: AMRFinderPlus only reports these genes when something notable
# happens to them (acquired resistance gene, resistance mutation, or a
# DISRUPTED/broken gene - subtype "POINT_DISRUPT"). It does NOT report a
# gene just because it's present and normal - these are essential genes,
# so "not in the AMRFinderPlus hit list" usually means wildtype/intact,
# NOT missing. predictor.py's gate should check specifically for a
# POINT_DISRUPT hit on these genes, not simple absence - see
# target_gate_real.csv for the reference implementation.
DRUG_TARGET_MAP: dict[str, str] = {
    "Ampicillin": "pbp3,pbp1A,pbp1B,pbp2",      # penicillin-binding proteins (beta-lactam target)
    "Ciprofloxacin": "gyrA,parC",                # DNA gyrase / topoisomerase IV (fluoroquinolone target)
    "Trimethoprim": "folA",                      # dihydrofolate reductase (trimethoprim target)
}
