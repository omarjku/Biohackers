"""
Tests for genome_reader.py — AMRFinderPlus output -> binary feature matrix.

The annotation step itself needs the amrfinder binary, so the subprocess calls
are mocked. Everything downstream of the TSV is tested for real, against TSVs
written in genuine AMRFinderPlus v4.2.7 layout.

Feature names and mutation nomenclature here are taken from the real E. coli run
behind data/raw/files.zip, not invented — that is the point of the fixtures.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import genome_reader as gr  # noqa: E402

HEADER = "Name\tElement symbol\tElement name\tType\tSubtype\tClass\tSubclass\n"


def _tsv(tmp_path: Path, genome_id: str, rows: list[str]) -> Path:
    path = tmp_path / f"{genome_id}.amrfinder.tsv"
    path.write_text(HEADER + "".join(rows))
    return path


ACQUIRED = "g\tblaTEM-1\tbeta-lactamase\tAMR\tAMR\tBETA-LACTAM\tBETA-LACTAM\n"
POINT = "g\tgyrA_S83L\tquinolone resistance\tAMR\tPOINT\tQUINOLONE\tQUINOLONE\n"
DISRUPT = "g\tcirA_Q56Ter\tdisrupted\tAMR\tPOINT_DISRUPT\tBETA-LACTAM\tCEFIDEROCOL\n"
VIRULENCE = "g\tespX1\tvirulence factor\tVIRULENCE\tVIRULENCE\t\t\n"


class TestParse:
    def test_classifies_the_three_evidence_types(self, tmp_path):
        path = _tsv(tmp_path, "g1", [ACQUIRED, POINT, DISRUPT])
        parsed = gr.parse_amrfinder_result(path)
        assert parsed["evidence"] == {
            "blaTEM-1": "acquired_gene",
            "gyrA_S83L": "point_mutation",
            "cirA_Q56Ter": "disrupted_gene",
        }

    def test_non_amr_rows_are_excluded(self, tmp_path):
        path = _tsv(tmp_path, "g1", [ACQUIRED, VIRULENCE])
        parsed = gr.parse_amrfinder_result(path)
        assert parsed["genes_found"] == {"blaTEM-1"}

    def test_header_only_tsv_is_a_valid_empty_result(self, tmp_path):
        """A genome with no AMR hits is susceptible, not a failure."""
        path = _tsv(tmp_path, "clean", [])
        parsed = gr.parse_amrfinder_result(path)
        assert parsed["genes_found"] == set()

    def test_zero_byte_tsv_is_a_valid_empty_result(self, tmp_path):
        path = tmp_path / "empty.amrfinder.tsv"
        path.write_text("")
        assert gr.parse_amrfinder_result(path)["genes_found"] == set()

    def test_mutation_gene_symbol_strips_the_change(self, tmp_path):
        path = _tsv(tmp_path, "g1", [POINT, DISRUPT])
        ann = gr.parse_amrfinder_result(path)["annotations"]
        assert ann["gyrA_S83L"]["gene_symbol"] == "gyrA"
        assert ann["cirA_Q56Ter"]["gene_symbol"] == "cirA"

    def test_acquired_gene_symbol_is_left_whole(self, tmp_path):
        """aac(6')-Ib-cr5 must not be truncated — it has no mutation suffix."""
        row = "g\taac(6')-Ib-cr5\tacetyltransferase\tAMR\tAMR\tQUINOLONE\tQUINOLONE\n"
        path = _tsv(tmp_path, "g1", [row])
        ann = gr.parse_amrfinder_result(path)["annotations"]
        assert ann["aac(6')-Ib-cr5"]["gene_symbol"] == "aac(6')-Ib-cr5"

    def test_older_amrfinder_column_names_still_parse(self, tmp_path):
        """v3 called it 'Gene symbol'; a version bump must not empty the matrix."""
        path = tmp_path / "old.amrfinder.tsv"
        path.write_text(
            "Name\tGene symbol\tType\tSubtype\tClass\tSubclass\n"
            "g\tblaTEM-1\tAMR\tAMR\tBETA-LACTAM\tBETA-LACTAM\n"
        )
        assert gr.parse_amrfinder_result(path)["genes_found"] == {"blaTEM-1"}

    def test_unrecognised_layout_fails_loudly(self, tmp_path):
        path = tmp_path / "weird.amrfinder.tsv"
        path.write_text("colA\tcolB\nx\ty\n")
        with pytest.raises(gr.GenomeReaderError, match="expected columns"):
            gr.parse_amrfinder_result(path)


class TestFeatureMatrix:
    def test_matrix_is_binary_and_keeps_hitless_genomes(self, tmp_path):
        _tsv(tmp_path, "g1", [ACQUIRED, POINT])
        _tsv(tmp_path, "g2", [])
        matrix, metadata, _ = gr.build_feature_matrix(tmp_path)

        assert sorted(matrix.index) == ["g1", "g2"]
        assert set(matrix.values.ravel()) <= {0, 1}
        assert matrix.loc["g2"].sum() == 0
        assert matrix.loc["g1", "gyrA_S83L"] == 1
        assert len(metadata) == 2

    def test_feature_union_across_genomes(self, tmp_path):
        _tsv(tmp_path, "g1", [ACQUIRED])
        _tsv(tmp_path, "g2", [POINT])
        matrix, _, _ = gr.build_feature_matrix(tmp_path)
        assert list(matrix.columns) == ["blaTEM-1", "gyrA_S83L"]
        assert matrix.loc["g1", "gyrA_S83L"] == 0
        assert matrix.loc["g2", "blaTEM-1"] == 0

    def test_genome_id_comes_from_filename_not_tsv_contents(self, tmp_path):
        _tsv(tmp_path, "562.100145", [ACQUIRED])
        matrix, _, _ = gr.build_feature_matrix(tmp_path)
        assert list(matrix.index) == ["562.100145"]

    def test_empty_directory_fails_loudly(self, tmp_path):
        with pytest.raises(gr.GenomeReaderError, match="No .*amrfinder.tsv"):
            gr.build_feature_matrix(tmp_path)

    def test_output_satisfies_the_data_contract(self, tmp_path):
        """The matrix must load through data_io without a contract violation."""
        import data_io

        _tsv(tmp_path, "g1", [ACQUIRED, POINT])
        _tsv(tmp_path, "g2", [])
        matrix, _, _ = gr.build_feature_matrix(tmp_path)

        out = tmp_path / "processed"
        out.mkdir()
        matrix.to_csv(out / "features.csv")
        pd.DataFrame(
            [{"genome_id": "g1", "drug": "Ampicillin", "phenotype": "R"},
             {"genome_id": "g2", "drug": "Ampicillin", "phenotype": "S"}]
        ).to_csv(out / "labels.csv", index=False)
        pd.DataFrame(
            [{"genome_id": "g1", "species": "Escherichia coli", "cluster_id": "CL-0001"},
             {"genome_id": "g2", "species": "Escherichia coli", "cluster_id": "CL-0002"}]
        ).to_csv(out / "genomes.csv", index=False)

        ds = data_io.load_dataset(out)
        assert len(ds.features) == 2


class TestAnnotation:
    def test_cached_tsv_is_reused_without_calling_amrfinder(self, tmp_path, monkeypatch):
        cached = _tsv(tmp_path, "g1", [ACQUIRED])

        def explode(*a, **k):
            raise AssertionError("amrfinder was called despite a cached TSV")

        monkeypatch.setattr(gr.subprocess, "run", explode)
        assert gr.run_amrfinder(Path("g1.fna"), tmp_path) == cached

    def test_missing_binary_gives_an_actionable_error(self, tmp_path, monkeypatch):
        monkeypatch.setattr(gr.shutil, "which", lambda _: None)
        with pytest.raises(gr.GenomeReaderError, match="github.com/ncbi/amr"):
            gr.run_amrfinder(tmp_path / "novel.fna", tmp_path)

    def test_organism_flag_is_always_passed(self, tmp_path, monkeypatch):
        """Without --organism there is no point-mutation screening, so no
        gyrA/parC, so ciprofloxacin looks like it has no signal at all."""
        seen = {}

        def fake_run(cmd, **kwargs):
            seen["cmd"] = cmd
            (tmp_path / "novel.amrfinder.tsv").write_text(HEADER + ACQUIRED)
            return None

        monkeypatch.setattr(gr.shutil, "which", lambda _: "/usr/bin/amrfinder")
        monkeypatch.setattr(gr.subprocess, "run", fake_run)
        gr.run_amrfinder(tmp_path / "novel.fna", tmp_path, organism="Escherichia")
        assert "--organism" in seen["cmd"]
        assert seen["cmd"][seen["cmd"].index("--organism") + 1] == "Escherichia"

    def test_failed_run_leaves_no_partial_tsv_to_be_cached(self, tmp_path, monkeypatch):
        import subprocess as sp

        def fake_run(cmd, **kwargs):
            out = Path(cmd[cmd.index("-o") + 1])
            out.write_text(HEADER)  # truncated output before the crash
            raise sp.CalledProcessError(1, cmd, stderr="database not found")

        monkeypatch.setattr(gr.shutil, "which", lambda _: "/usr/bin/amrfinder")
        monkeypatch.setattr(gr.subprocess, "run", fake_run)

        with pytest.raises(gr.GenomeReaderError, match="database not found"):
            gr.run_amrfinder(tmp_path / "novel.fna", tmp_path)
        assert not (tmp_path / "novel.amrfinder.tsv").exists()

    def test_batch_collects_failures_instead_of_aborting(self, tmp_path, monkeypatch):
        """One bad assembly must not discard the other genomes' results."""
        def fake_run(fasta, out_dir, *a, **k):
            if fasta.stem == "bad":
                raise gr.GenomeReaderError("assembly rejected")
            return _tsv(out_dir, fasta.stem, [ACQUIRED])

        monkeypatch.setattr(gr, "run_amrfinder", fake_run)
        paths = [tmp_path / f"{n}.fna" for n in ("g1", "bad", "g2")]
        done, errors = gr.run_amrfinder_batch(paths, tmp_path, jobs=2)

        assert len(done) == 2
        assert list(errors) == ["bad"]
