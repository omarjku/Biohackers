
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
    "Ampicillin": "ftsI",                        # PBP3 (beta-lactam target)
    "Ciprofloxacin": "gyrA,parC",                # DNA gyrase / topoisomerase IV (fluoroquinolone target)
    "Trimethoprim": "folA",                      # dihydrofolate reductase (trimethoprim target)
}

# Ampicillin was originally "pbp3,pbp1A,pbp1B,pbp2". AMRFinderPlus emits none of
# those symbols — checked against all 124 features in gene_metadata_real.csv, in
# both bare and gene_MUTATION form. It calls PBP3 `ftsI` (see ftsI_I336IKYRI,
# ftsI_N337NYRIN), so the original entry could never match anything.
#
# Measured consequence of the note above: on this feature matrix the gate cannot
# fire for ANY of the three drugs. No bare `ftsI`/`gyrA`/`parC`/`folA` column
# exists, because AMRFinderPlus only reports these essential genes when they are
# mutated or disrupted. predictor.target_gate() therefore returns "unknown" for
# all three — never "absent" — which is the correct reading: we never scanned for
# an intact copy, and absence of data is not absence of gene. Report the gate as
# in place and honest-by-construction, not as a filter that is doing active work.


# --------------------------------------------------------------------------
# Curated resistance genes, per drug. Consumed by predictor._known_resistance_genes().
# --------------------------------------------------------------------------
#
# Membership here is what promotes a feature's evidence_category from
# "statistical_association" to "known_gene_or_mutation", so it is a claim about
# mechanism and the bar is deliberately high. Derived from AMRFinderPlus
# amr_class/amr_subclass annotations, then narrowed by hand to mechanisms that
# are specifically established for each drug.
#
# Deliberately EXCLUDED, though a purely class-based filter would sweep them in:
#   - cirA_* truncations: annotated BETA-LACTAM/CEFIDEROCOL, but cirA is an iron
#     transporter and its loss is a cefiderocol-uptake mechanism, not ampicillin.
#   - acrR_*, marR_*, ompC_*: efflux and porin regulators. Real resistance
#     contributors, but broad-spectrum and not drug-specific, so naming them as
#     the mechanism for one drug overstates what is known.
# These still enter the model as features; they simply report as
# "statistical_association" rather than as a curated mechanism.
KNOWN_RESISTANCE_GENES: dict[str, list[str]] = {
    # Beta-lactamases hydrolysing ampicillin, plus the ampC promoter mutation
    # that derepresses the chromosomal enzyme, plus PBP3 target modifications.
    "Ampicillin": [
        "blaTEM", "blaTEM-1", "blaTEM-12", "blaTEM-30",
        "blaTEMp_C32T", "blaTEMp_G162T",
        "blaCTX-M-1", "blaCTX-M-14", "blaCTX-M-15", "blaCTX-M-27", "blaCTX-M-55",
        "blaSHV-1", "blaSHV-12",
        "blaOXA-1", "blaOXA-10",
        "blaCMY-2", "blaCMY-42",
        "blaKPC-2", "blaKPC-3",
        "blaNDM-1", "blaNDM-5", "blaNDM-7",
        "ampC_T-32A",
        "ftsI_I336IKYRI", "ftsI_N337NYRIN",
    ],
    # Quinolone-resistance-determining-region mutations in the drug's own
    # targets, plasmid-borne qnr protection, and the cr variant of aac(6')-Ib
    # which acetylates ciprofloxacin directly.
    "Ciprofloxacin": [
        "gyrA_S83L", "gyrA_S83A", "gyrA_D87N", "gyrA_D87G",
        "parC_S80I", "parC_E84V", "parC_E84G", "parC_A56T",
        "parE_S458A", "parE_E460K", "parE_I355T", "parE_I529L", "parE_L416F",
        "qnrA1", "qnrB6", "qnrB19", "qnrS1",
        "aac(6')-Ib-cr5",
    ],
    # Acquired dihydrofolate reductases — an alternative DHFR that trimethoprim
    # does not inhibit. The whole dfrA family is well established for this drug.
    "Trimethoprim": [
        "dfrA1", "dfrA5", "dfrA7", "dfrA8",
        "dfrA12", "dfrA14", "dfrA17", "dfrA27",
    ],
}
