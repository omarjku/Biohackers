"""
Tests for the live FASTA demo path.

The name bridge (fasta_pipeline.bridge_row) and the frontend report
(explainer.explain_report / report_item) are pure and tested here in full. The
AMRFinderPlus scan and the fitted model are exercised by the live validation in
the branch writeup, not in CI — they need the amrfinder binary and the
gitignored data/processed, neither of which belongs in a unit test.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import fasta_pipeline as fp  # noqa: E402
import explainer  # noqa: E402
from schemas import Prediction, SupportingFeature  # noqa: E402

# A realistic slice of the trained model's 139 NDARO columns.
MODEL_COLS = [
    "BlaEC", "CTX-M", "TEM", "SHV", "OXA-1", "CMY/CMY-2/CFE/LAT", "ampC",
    "Sulfonamide", "dhfrI", "Tet", "Mph", "CatB", "CatA1/CatA4",
    "AAC-Ib-cr", "AAC-II,III,IV,VI,VIII,IX,X", "ANT-Ia", "APH-I", "APH-Ic/APH-Id",
    "Quinolone", "QacE", "sul1", "tet(A)", "blaCTX-M",
]


class TestBridge:
    def test_maps_genes_to_dominant_family_tokens(self):
        genes = {"blaCTX-M-15", "blaOXA-1", "sul1", "dfrA17", "tet(A)",
                 "aac(6')-Ib-cr5", "mph(A)", "catB3", "aadA5", "gyrA_S83L"}
        on = set(fp.bridge_row(genes, MODEL_COLS).pipe(lambda s: s[s == 1].index))
        assert {"CTX-M", "OXA-1", "Sulfonamide", "dhfrI", "Tet",
                "AAC-Ib-cr", "Mph", "CatB", "ANT-Ia", "Quinolone"} <= on

    def test_prefers_family_token_over_rare_exact_column(self):
        # 'sul1' and 'tet(A)' exist as rare columns but signal lives in the family
        row = fp.bridge_row({"sul1", "tet(A)"}, MODEL_COLS)
        assert row["Sulfonamide"] == 1 and row["Tet"] == 1

    def test_does_not_over_activate_synonyms(self):
        # one gene must not light up every related column (that trips the OOD gate)
        row = fp.bridge_row({"sul1"}, MODEL_COLS)
        assert row["sul1"] == 0  # rare exact col stays off; only the family token fires

    def test_sets_intrinsic_blaEC_for_ecoli(self):
        assert fp.bridge_row(set(), MODEL_COLS)["BlaEC"] == 1

    def test_ignores_unknown_and_point_mutation_suffixes(self):
        row = fp.bridge_row({"someUnknownGene", "gyrA_D87N"}, MODEL_COLS)
        assert row["Quinolone"] == 1              # gyrA point -> Quinolone
        assert row.drop("BlaEC").sum() == 1        # nothing else lit besides intrinsic


def _pred(drug, call, conf, evidence, genes=(), gate="unknown", reason=None):
    return Prediction(
        sample_id="G", species="Escherichia coli", drug=drug, call=call,
        confidence=conf, evidence_category=evidence,
        supporting_features=[SupportingFeature(gene=g) for g in genes],
        target_gate_status=gate, no_call_reason=reason,
    )


class TestReportItem:
    def test_fail_known_gene(self):
        item = explainer.report_item(_pred("Ampicillin", "likely_to_fail", 0.94,
                                           "known_gene_or_mutation", ["CTX-M", "OXA-1"]))
        assert item["underlying_state"] == "Likely to fail"
        assert item["confidence"] == 0.94
        assert item["target_marker"] == "CTX-M; OXA-1"
        assert item["locus_id"] == "ftsI"
        assert "known" in item["bio_explanation"].lower()

    def test_work_reports_susceptibility_confidence_and_wildtype(self):
        item = explainer.report_item(_pred("Ciprofloxacin", "likely_to_work", 0.06,
                                           "no_known_signal"))
        assert item["underlying_state"] == "Likely to work"
        assert item["confidence"] == pytest.approx(0.94)   # 1 - P(resistant)
        assert item["target_marker"] == "Wild-Type"

    def test_statistical_is_labelled_not_causal(self):
        item = explainer.report_item(_pred("Ampicillin", "likely_to_fail", 0.8,
                                           "statistical_association", ["Tet"]))
        assert "not rest on a proven mechanism" in item["bio_explanation"]
        assert "association" in item["stat_explanation"].lower()

    def test_no_call_is_ambiguous(self):
        item = explainer.report_item(_pred("Trimethoprim", "no_call", 0.45,
                                           "statistical_association", reason="ambiguous band"))
        assert item["underlying_state"] == "No-call"
        assert item["target_marker"] == "Ambiguous"

    def test_not_applicable_maps_to_no_call_with_na_marker(self):
        item = explainer.report_item(_pred("Ampicillin", "not_applicable", 0.5,
                                           "no_known_signal", gate="absent"))
        assert item["underlying_state"] == "No-call"
        assert item["target_marker"] == "N/A"

    def test_exact_frontend_keys_present(self):
        item = explainer.report_item(_pred("Ampicillin", "likely_to_work", 0.1, "no_known_signal"))
        assert set(item) == {"drug", "drug_class", "underlying_state", "confidence",
                             "target_marker", "locus_id", "bio_explanation", "stat_explanation"}
