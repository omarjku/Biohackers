"""
Tests for fetch_bvbrc.py — BV-BRC Data API -> the four contract files.

The HTTP layer is never touched: `_fetch_over_ids` is monkeypatched to return
canned rows in genuine BV-BRC JSON shape (genome_amr / sp_gene / genome), taken
from real responses observed on www.bv-brc.org, not invented. Everything from the
raw row onward — the laboratory-evidence filter, the NDARO product->family
parsing, MLST->cluster, and the contract-consistency logic in build() — is tested
for real, and build()'s output is validated through data_io.load_dataset().
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import fetch_bvbrc as fb  # noqa: E402
import data_io  # noqa: E402


# --- real-shape fixture rows -------------------------------------------------

def _amr(gid, drug, pheno, evidence="Laboratory Method", method="Broth dilution"):
    return {
        "genome_id": gid,
        "antibiotic": drug,
        "resistant_phenotype": pheno,
        "evidence": evidence,
        "laboratory_typing_method": method,
    }


def _sp(gid, gene=None, product="", cls=""):
    return {"genome_id": gid, "gene": gene, "product": product, "antibiotics_class": cls}


def _genome(gid, mlst):
    return {"genome_id": gid, "genome_name": "Escherichia coli", "mlst": mlst, "species": "Escherichia coli"}


class TestFeatureToken:
    def test_prefers_explicit_gene_symbol(self):
        assert fb._feature_token(_sp("g", gene="blaTEM-1")) == "blaTEM-1"

    def test_parses_family_after_arrow(self):
        row = _sp("g", product="Class A beta-lactamase (EC 3.5.2.6) => CTX-M family")
        assert fb._feature_token(row) == "CTX-M"

    def test_parses_family_after_at_marker(self):
        row = _sp("g", product="Dihydropteroate synthase type-2 @ Sul1")
        assert fb._feature_token(row) == "Sul1"

    def test_drops_intrinsic_target_without_family(self):
        # "DNA gyrase subunit A" has no => family and is a mutation-driven target
        assert fb._feature_token(_sp("g", product="DNA gyrase subunit A (EC 5.99.1.3)")) is None

    def test_drops_intrinsic_even_when_named(self):
        assert fb._feature_token(_sp("g", gene="gyrA")) is None


class TestParseSt:
    def test_extracts_sequence_type(self):
        assert fb._parse_st("MLST.ecoli_achtman_4.410") == "ST-410"

    def test_missing_or_nonnumeric_is_none(self):
        assert fb._parse_st(None) is None
        assert fb._parse_st("MLST.ecoli_achtman_4.-") is None


class TestFetchLabels:
    def _patch(self, monkeypatch, rows):
        monkeypatch.setattr(fb, "_fetch_over_ids", lambda *a, **k: rows)

    def test_keeps_only_lab_measured_rs(self, monkeypatch):
        rows = [
            _amr("1", "ampicillin", "Resistant"),
            _amr("2", "ampicillin", "Susceptible", evidence="Computational Method"),  # dropped
            _amr("3", "ampicillin", "Intermediate"),  # dropped
            _amr("4", "ampicillin", None),             # dropped
            _amr("5", "vancomycin", "Resistant"),      # not a target drug
        ]
        self._patch(monkeypatch, rows)
        out = fb.fetch_labels(["1"])
        assert list(out["genome_id"]) == ["1"]
        assert out.iloc[0]["phenotype"] == "R"
        assert out.iloc[0]["drug"] == "Ampicillin"  # title-cased

    def test_drops_conflicting_pairs_keeps_agreeing(self, monkeypatch):
        rows = [
            _amr("1", "ciprofloxacin", "Resistant"),
            _amr("1", "ciprofloxacin", "Susceptible"),   # conflict with above -> both dropped
            _amr("2", "ciprofloxacin", "Resistant"),
            _amr("2", "ciprofloxacin", "Resistant"),      # agreeing duplicate -> one kept
        ]
        self._patch(monkeypatch, rows)
        out = fb.fetch_labels(["1", "2"])
        assert list(out["genome_id"]) == ["2"]
        assert len(out) == 1


class TestFetchFeatures:
    def test_binary_matrix_and_metadata(self, monkeypatch):
        rows = [
            _sp("1", gene="blaTEM-1", cls="beta-lactam"),
            _sp("1", product="... => CTX-M family", cls="beta-lactam"),
            _sp("2", product="... => Tet(A)", cls="tetracycline"),
            _sp("2", gene="gyrA"),  # intrinsic -> dropped
        ]
        monkeypatch.setattr(fb, "_fetch_over_ids", lambda *a, **k: rows)
        features, meta = fb.fetch_features(["1", "2"])
        assert set(features.columns) == {"blaTEM-1", "CTX-M", "Tet"}
        assert features.loc["1", "CTX-M"] == 1
        assert features.loc["2", "CTX-M"] == 0
        assert set(features.values.flatten()) <= {0, 1}
        assert (meta["evidence_type"] == "known_gene").all()


class TestBuildIsContractValid:
    def test_build_output_passes_data_io_validation(self, monkeypatch, tmp_path):
        amr = [
            _amr("562.1", "ampicillin", "Resistant"),
            _amr("562.1", "ciprofloxacin", "Susceptible"),
            _amr("562.2", "ampicillin", "Susceptible"),
            _amr("562.2", "ciprofloxacin", "Resistant"),
            _amr("562.3", "ampicillin", "Resistant"),
        ]
        sp = [
            _sp("562.1", gene="blaTEM-1", cls="beta-lactam"),
            _sp("562.2", product="... => CTX-M family", cls="beta-lactam"),
            # 562.3 has no acquired gene -> must still appear as an all-zero row
        ]
        gen = [_genome("562.1", "MLST.x.11"), _genome("562.2", "MLST.x.69"), _genome("562.3", "MLST.x.131")]

        def fake(endpoint, *a, **k):
            return {"genome_amr": amr, "sp_gene": sp, "genome": gen}[endpoint]

        monkeypatch.setattr(fb, "_fetch_over_ids", fake)
        id_list = tmp_path / "idlist.csv"
        pd.DataFrame({"genome_id": ["562.1", "562.2", "562.3"]}).to_csv(id_list, index=False)
        monkeypatch.setattr(fb, "GENOME_ID_LIST", id_list)

        fb.build(out_dir=tmp_path, cache_dir=tmp_path / "cache")
        ds = data_io.load_dataset(tmp_path)  # raises on any contract violation

        assert set(ds.features.index) == {"562.1", "562.2", "562.3"}
        assert ds.features.loc["562.3"].sum() == 0  # susceptible genome, zero row, still present
        assert ds.genomes.loc["562.1", "cluster_id"] == "ST-11"
        assert "Ampicillin" in ds.drug_targets  # target map wired through

    def test_exact_target_gene_column_is_dropped(self, monkeypatch, tmp_path):
        """A bare 'ftsI' column would make the Ampicillin gate mis-fire
        not_applicable across the panel; build() must drop it (but keep variants)."""
        amr = [_amr("562.1", "ampicillin", "Resistant"), _amr("562.2", "ampicillin", "Susceptible")]
        sp = [
            _sp("562.1", gene="ftsI"),            # exact target gene -> must be dropped
            _sp("562.1", gene="ftsI_N337NYRIN"),  # variant form -> must be kept
            _sp("562.2", gene="blaTEM-1"),
        ]
        gen = [_genome("562.1", "MLST.x.11"), _genome("562.2", "MLST.x.69")]
        monkeypatch.setattr(fb, "_fetch_over_ids", lambda endpoint, *a, **k: {"genome_amr": amr, "sp_gene": sp, "genome": gen}[endpoint])
        id_list = tmp_path / "idlist.csv"
        pd.DataFrame({"genome_id": ["562.1", "562.2"]}).to_csv(id_list, index=False)
        monkeypatch.setattr(fb, "GENOME_ID_LIST", id_list)

        fb.build(out_dir=tmp_path, cache_dir=tmp_path / "cache")
        ds = data_io.load_dataset(tmp_path)
        assert "ftsI" not in ds.features.columns          # exact target gene dropped
        assert "ftsI_N337NYRIN" in ds.features.columns    # variant kept
