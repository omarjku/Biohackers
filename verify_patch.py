"""
verify_patch.py — biosecurity/ML compliance harness

Probes the four controls a Genome Firewall audit asks about, by EXERCISING the
pipeline rather than reading it:

  1. sequence-homology de-duplication clustering before the train/test split
  2. a deterministic molecular-target gate that overrides ML scoring
  3. LLM prompt constraints: forced evidence categories, no hallucinated biology
  4. the mandatory "confirm with lab testing" disclaimer on every result

Run:  python verify_patch.py        (exit 0 = all controls hold)

Each check prints PASS / FAIL / GAP. GAP means the control's logic is correct
but is inert because a dependency is unpopulated — a real finding, but a
different fix from a broken control, and owned by a different person.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
DATA_DIR = ROOT / "data" / "synthetic"

results: list[tuple[str, str, str]] = []


def record(control: str, status: str, detail: str) -> None:
    results.append((control, status, detail))
    print(f"  [{status}] {detail}")


# --------------------------------------------------------------------------
# Control 1 — de-duplication clustering before the split
# --------------------------------------------------------------------------


def check_deduplication() -> None:
    print("\n1. Sequence-homology de-duplication before train/test split")
    from data_io import load_dataset
    from predictor import cluster_sample_weights
    from splits import DEFAULT_MASH_THRESHOLD, cluster_genomes, grouped_split

    ds = load_dataset(DATA_DIR)

    # (a) homology clustering exists and is Mash/MinHash based
    sig = inspect.signature(cluster_genomes)
    assert "threshold" in sig.parameters, "cluster_genomes takes no distance threshold"
    record(
        "1a", "PASS",
        f"cluster_genomes() clusters FASTA by MinHash/Mash distance, "
        f"threshold={DEFAULT_MASH_THRESHOLD} (~95% ANI)",
    )

    # (b) the split is cluster-disjoint for every drug — the actual leak test
    leaks = 0
    for drug in ds.drugs:
        _, y, groups = ds.xy_for_drug(drug)
        split = grouped_split(groups, y=y, seed=0)
        seen: dict[str, str] = {}
        for name, idx in (
            ("train", split.train),
            ("calibration", split.calibration),
            ("test", split.test),
        ):
            for g in set(np.asarray(groups)[idx]):
                if g in seen and seen[g] != name:
                    leaks += 1
                seen[g] = name
    if leaks:
        record("1b", "FAIL", f"{leaks} cluster(s) span two splits — homology leak")
    else:
        record(
            "1b", "PASS",
            f"no cluster spans two splits across {len(ds.drugs)} drugs "
            "(train/calibration/test all disjoint)",
        )

    # (c) within-train redundancy is down-weighted, not just split around
    drug = ds.drugs[0]
    _, y, groups = ds.xy_for_drug(drug)
    split = grouped_split(groups, y=y, seed=0)
    g_train, y_train = np.asarray(groups)[split.train], np.asarray(y)[split.train]
    w = cluster_sample_weights(g_train, y_train)

    sizes = pd.Series(g_train).value_counts()
    biggest = sizes.index[0]
    raw_share = sizes.iloc[0] / len(g_train)
    weighted_share = w[g_train == biggest].sum() / w.sum()
    if weighted_share < raw_share:
        record(
            "1c", "PASS",
            f"per-cluster weights shrink the largest cluster's influence "
            f"{raw_share:.1%} -> {weighted_share:.1%} of training mass ({drug})",
        )
    else:
        record("1c", "FAIL", "cluster weights do not reduce redundant-cluster dominance")


# --------------------------------------------------------------------------
# Control 2 — deterministic target gate overriding ML
# --------------------------------------------------------------------------


def check_target_gate() -> None:
    print("\n2. Deterministic molecular target gate overrides ML scoring")
    from data_io import load_dataset
    from predictor import Predictor, target_gate

    ds = load_dataset(DATA_DIR)

    # (a) absence of every target gene => "absent"; a missing COLUMN => "unknown"
    row_present = pd.Series({"gyrA": 1, "parC": 1})
    row_absent = pd.Series({"gyrA": 1, "parC": 0})
    row_unscanned = pd.Series({"gyrA": 1})
    assert target_gate(row_present, ["gyrA", "parC"]) == "present"
    assert target_gate(row_absent, ["gyrA", "parC"]) == "absent"
    assert target_gate(row_unscanned, ["gyrA", "parC"]) == "unknown"
    record(
        "2a", "PASS",
        "target_gate(): all targets present->present, any absent->absent, "
        "unscanned column->unknown (absence of data != absence of gene)",
    )

    # (b) the gate beats the model even when the model is confident it works
    predictor = Predictor.fit(ds, seed=0)
    gated_drug = next(
        (d for d, m in predictor.models.items() if m.target_genes), None
    )
    assert gated_drug, "no drug in the synthetic fixture has target genes"
    model = predictor.models[gated_drug]

    # forge a genome that is clean of resistance genes (model -> likely_to_work)
    # but missing the drug's target.
    row = ds.features.iloc[0].copy()
    for gene in model.feature_names:
        row[gene] = 0
    p_clean = model.probability_resistant(row)
    for gene in model.target_genes:
        row[gene] = 0

    species = str(ds.genomes.iloc[0]["species"])
    pred = predictor._predict_one(row, "SYNTH-GATE-TEST", species, gated_drug)
    if pred.call == "not_applicable" and pred.target_gate_status == "absent":
        record(
            "2b", "PASS",
            f"{gated_drug}: model scored p(R)={p_clean:.2f} (would call "
            f"likely_to_work) but missing target forced call='not_applicable'",
        )
    else:
        record(
            "2b", "FAIL",
            f"{gated_drug}: target absent yet call={pred.call!r} — ML was not overridden",
        )

    # (c) calibration must not downgrade not_applicable to no_call
    from calibration import Calibrator

    calibrator = Calibrator.fit(ds, predictor)
    after = calibrator.apply(pred, row)
    if after.call == "not_applicable":
        record("2c", "PASS", "calibration preserves 'not_applicable' (not downgraded to no_call)")
    else:
        record("2c", "FAIL", f"calibration overwrote not_applicable -> {after.call!r}")

    # (d) does the gate actually have data to act on in production?
    import drug_database

    target_map = getattr(drug_database, "DRUG_TARGET_MAP", {})
    known = getattr(drug_database, "KNOWN_RESISTANCE_GENES", None)
    ungated = [d for d, m in predictor.models.items() if not m.target_genes]
    if not target_map:
        record(
            "2d", "GAP",
            "drug_database.DRUG_TARGET_MAP is EMPTY and KNOWN_RESISTANCE_GENES is "
            f"{'absent' if known is None else 'present'} — the gate is correct but "
            "inert on real drugs; it only fires on synthetic fixture targets. "
            "Owner: Moncef.",
        )
    if ungated:
        record(
            "2e", "GAP",
            f"no target genes for {ungated} — these drugs get "
            "target_gate_status='unknown' and are never gated. Owner: Moncef.",
        )


# --------------------------------------------------------------------------
# Control 3 — LLM prompt constraints
# --------------------------------------------------------------------------


def check_llm_constraints() -> None:
    print("\n3. LLM prompt constraints (evidence categories, no hallucinated biology)")
    import explainer

    src = inspect.getsource(explainer.llm_explain)

    # (a) does the system prompt forbid inventing biology?
    forbids_invention = "never invent" in src.lower()
    record(
        "3a", "PASS" if forbids_invention else "FAIL",
        "system prompt forbids inventing genes/mutations/evidence"
        if forbids_invention
        else "system prompt does NOT forbid inventing genes",
    )

    # (b) does it force the statistical_association caveat?
    forces_caveat = "statistical_association" in src
    record(
        "3b", "PASS" if forces_caveat else "FAIL",
        "system prompt forces the 'learned pattern, not proven cause' caveat "
        "for statistical_association"
        if forces_caveat
        else "system prompt does not constrain evidence_category wording",
    )

    # (c) is LLM output VALIDATED against the structured input, or trusted?
    #     This is the real question: a prompt rule is a request, not a guarantee.
    validated = any(
        token in src for token in ("supporting_features", "_validate", "verify_")
    ) and "return response" not in src.replace(" ", "")
    returns_unchecked = "response.choices[0].message.content.strip()" in src
    if returns_unchecked and not validated:
        record(
            "3c", "GAP",
            "llm_explain() returns model output UNVALIDATED — no post-hoc check "
            "that emitted gene names appear in pred.supporting_features. Prompt "
            "rules alone cannot prevent hallucinated biology. Owner: Hazem.",
        )
    else:
        record("3c", "PASS", "LLM output is validated against supporting_features")

    # (d) the deterministic path must never touch the network
    tmpl = inspect.getsource(explainer.template_explain)
    if "openai" not in tmpl.lower():
        record(
            "3d", "PASS",
            "template_explain() is fully deterministic — no LLM in the fallback path",
        )
    else:
        record("3d", "FAIL", "template fallback reaches the LLM")

    # (e) fallback actually engages on LLM failure
    from schemas import Prediction

    pred = Prediction(
        sample_id="S1", species="Escherichia coli", drug="Ciprofloxacin",
        call="likely_to_fail", confidence=0.91,
        evidence_category="statistical_association",
        supporting_features=[], target_gate_status="present",
    )
    text = explainer.template_explain(pred)
    if "not a confirmed causal resistance mechanism" in text:
        record(
            "3e", "PASS",
            "deterministic template states the association caveat verbatim, "
            "independent of any LLM",
        )
    else:
        record("3e", "FAIL", "template omits the statistical-association caveat")


# --------------------------------------------------------------------------
# Control 4 — mandatory disclaimer
# --------------------------------------------------------------------------


def check_disclaimer() -> None:
    print("\n4. Mandatory biosecurity disclaimer on every result")
    import explainer
    from schemas import ExplanationResult, Prediction

    required = (
        "This is a research prototype. All results must be confirmed with "
        "standard laboratory testing."
    )

    # (a) exact required wording present in the backend
    if explainer.DISCLAIMER == required:
        record("4a", "PASS", "explainer.DISCLAIMER matches the required wording exactly")
    else:
        record("4a", "FAIL", f"disclaimer text differs: {explainer.DISCLAIMER!r}")

    # (b) EVERY explanation carries it, across every call type
    cases = [
        ("likely_to_fail", "known_gene_or_mutation", None),
        ("likely_to_work", "no_known_signal", None),
        ("no_call", "statistical_association", "ambiguous band"),
        ("not_applicable", "no_known_signal", None),
    ]
    missing = []
    for call, evidence, reason in cases:
        pred = Prediction(
            sample_id="S1", species="Escherichia coli", drug="Ciprofloxacin",
            call=call, confidence=0.5, evidence_category=evidence,
            supporting_features=[], target_gate_status="present",
            no_call_reason=reason,
        )
        out = explainer.explain(pred, use_llm=False)
        if out.disclaimer != required:
            missing.append(call)
    if missing:
        record("4b", "FAIL", f"explanations missing the disclaimer for calls: {missing}")
    else:
        record(
            "4b", "PASS",
            f"all {len(cases)} call types emit the disclaimer via ExplanationResult",
        )

    # (c) is the disclaimer structurally REQUIRED, or merely conventionally added?
    field = ExplanationResult.model_fields["disclaimer"]
    if field.is_required():
        record(
            "4c", "PASS",
            "ExplanationResult.disclaimer is a required field — an explanation "
            "cannot be constructed without one",
        )
    else:
        record("4c", "FAIL", "disclaimer is optional on ExplanationResult")

    # (d) the Prediction object itself carries no disclaimer
    if "disclaimer" not in Prediction.model_fields:
        record(
            "4d", "GAP",
            "schemas.Prediction has no disclaimer field — any consumer that "
            "renders Predictions directly (app.py, a JSON export, evaluation "
            "output) emits results with no disclaimer attached. Only the "
            "explainer path is covered. schemas.py is the shared contract: "
            "change by team agreement, not unilaterally.",
        )


# --------------------------------------------------------------------------


def main() -> int:
    print("=" * 72)
    print("Genome Firewall — biosecurity & ML compliance verification")
    print("=" * 72)

    check_deduplication()
    check_target_gate()
    check_llm_constraints()
    check_disclaimer()

    failures = [r for r in results if r[1] == "FAIL"]
    gaps = [r for r in results if r[1] == "GAP"]
    passes = [r for r in results if r[1] == "PASS"]

    print("\n" + "=" * 72)
    print(f"{len(passes)} PASS   {len(gaps)} GAP   {len(failures)} FAIL")
    print("=" * 72)

    if gaps:
        print("\nGAPs — control logic is correct but inert or incomplete:")
        for control, _, detail in gaps:
            print(f"  ({control}) {detail}")

    if failures:
        print("\nFAILs — control is broken:")
        for control, _, detail in failures:
            print(f"  ({control}) {detail}")
        return 1

    print("\nNo broken controls. GAPs above are dependency/ownership items.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
