"""
explainer.py — Module 03, natural-language layer
Owner: Hazem

Input:  a Prediction (see schemas.py) — produced by predictor.py / calibration.py
Output: an ExplanationResult — consumed by app.py

Two paths:
  1. template_explain()  -> deterministic, free, zero API risk. Always works.
  2. llm_explain()       -> GPT-4 rephrases the template into clinician-readable
                            prose. The system prompt forbids inventing genes, and
                            the frontend report path (report_item, use_llm) runs a
                            post-hoc guard (_contains_unlisted_gene) that rejects
                            LLM output naming an AMR gene absent from
                            supporting_features and falls back to the template.

The LIVE APP calls explain_report with use_llm=False, so the deterministic
template path is what ships by default; the LLM is an optional enhancement.
explain()/report_item() fall back to the template on any failure (timeout, rate
limit, no credits), hedging the shared $50 OpenAI credit running out mid-demo.
"""

import os
import re
from schemas import Prediction, ExplanationResult

DISCLAIMER = (
    "This is a research prototype. All results must be confirmed with "
    "standard laboratory testing."
)

# --- 1. Template fallback ---------------------------------------------------

def _feature_list(pred: Prediction) -> str:
    if not pred.supporting_features:
        return "no supporting genomic features on record"
    parts = []
    for f in pred.supporting_features:
        s = f.gene
        if f.mutation:
            s += f" ({f.mutation})"
        if f.note:
            s += f" — {f.note}"
        parts.append(s)
    return "; ".join(parts)


TEMPLATES = {
    ("likely_to_fail", "known_gene_or_mutation"):
        "{drug}: LIKELY TO FAIL against this {species} sample "
        "(confidence {confidence:.0%}). Known resistance mechanism detected: "
        "{features}.",
    ("likely_to_fail", "statistical_association"):
        "{drug}: LIKELY TO FAIL (confidence {confidence:.0%}), based on a "
        "statistical association with {features}. This is a model-learned "
        "pattern, not a confirmed causal resistance mechanism.",
    ("likely_to_work", "no_known_signal"):
        "{drug}: LIKELY TO WORK against this {species} sample "
        "(confidence {confidence:.0%}). No resistance signal was found, and "
        "the drug's molecular target is present.",
    ("likely_to_work", "known_gene_or_mutation"):
        "{drug}: LIKELY TO WORK (confidence {confidence:.0%}), though note: "
        "{features}.",
    ("no_call", None):
        "{drug}: NO CONFIDENT PREDICTION for this sample. Reason: "
        "{no_call_reason}.",
    ("not_applicable", None):
        "{drug}: this drug's molecular target was not found in the genome — "
        "not applicable to this organism.",
}


def template_explain(pred: Prediction) -> str:
    key = (pred.call, pred.evidence_category)
    template = TEMPLATES.get(key) or TEMPLATES.get((pred.call, None))
    if template is None:
        template = "{drug}: {call} (confidence {confidence:.0%})."
    return template.format(
        drug=pred.drug,
        species=pred.species,
        confidence=_display_confidence(pred.confidence),
        features=_feature_list(pred),
        no_call_reason=pred.no_call_reason or "insufficient evidence",
        call=pred.call,
    )


# --- 2. GPT-4 rephrasing (optional enhancement) -----------------------------

def llm_explain(pred: Prediction, base_text: str) -> str:
    from openai import OpenAI  # imported lazily so the template path never needs it

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    system_prompt = (
        "You rewrite a structured antibiotic-resistance prediction into one "
        "short, clear paragraph for a clinician. Rules: "
        "(1) Only restate facts given to you — never invent genes, mutations, "
        "or evidence not present in the input. "
        "(2) Never state a treatment decision. "
        "(3) If evidence_category is 'statistical_association', make clear "
        "this is a learned pattern, not a proven biological cause. "
        "(4) Keep it to 2-3 sentences."
    )

    user_prompt = (
        f"Structured result: {pred.model_dump_json()}\n\n"
        f"Plain-language draft to refine: {base_text}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # cheap default; swap to gpt-4o if budget allows
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()


# --- 3. Public entry point ---------------------------------------------------

def explain(pred: Prediction, use_llm: bool = True) -> ExplanationResult:
    base_text = template_explain(pred)

    text = base_text
    if use_llm and os.environ.get("OPENAI_API_KEY"):
        try:
            text = llm_explain(pred, base_text)
        except Exception as e:  # noqa: BLE001 — any API failure falls back safely
            text = base_text

    return ExplanationResult(
        sample_id=pred.sample_id,
        drug=pred.drug,
        explanation_text=text,
        disclaimer=DISCLAIMER,
        confidence_label=f"{_display_confidence(pred.confidence):.0%}",
    )


# --- 4. Frontend report JSON (Module 03 contract with the Streamlit app) --------
#
# The app consumes a list of these objects (see data/synthetic/sample_predictions
# JSON spec). Shape and the EXACT capitalised state strings are fixed by the
# frontend; do not change them without the UI owner.

DRUG_CLASS = {
    "Ampicillin": "Beta-lactam (aminopenicillin)",
    "Ciprofloxacin": "Fluoroquinolone",
    "Trimethoprim": "Folate-pathway antagonist (DHFR inhibitor)",
}

# The molecular target locus the drug acts on — populates the genetic tracking box.
DRUG_LOCUS = {
    "Ampicillin": "ftsI",
    "Ciprofloxacin": "gyrA",
    "Trimethoprim": "folA",
}

# call -> the exact string the frontend expects (capitalisation matters).
STATE_LABEL = {
    "likely_to_fail": "Likely to fail",
    "likely_to_work": "Likely to work",
    "no_call": "No-call",
    "not_applicable": "No-call",  # 3-state UI; target absence is explained in text
}


def _gene_names(pred: Prediction) -> str:
    """Comma-joined gene (+mutation) names only — no notes, for clean prose."""
    if not pred.supporting_features:
        return "no supporting genomic features"
    return ", ".join(
        f.gene + (f" {f.mutation}" if f.mutation else "") for f in pred.supporting_features
    )


def _state_confidence(pred: Prediction) -> float:
    """
    Confidence IN THE REPORTED STATE, as a raw 0-1 float.

    pred.confidence is the calibrated P(resistant). For a "likely to work" call
    the number the user should see is the susceptibility confidence, 1 - P(R)
    (matching the frontend's 0.942 "Likely to work" example). For no-call we show
    the raw probability so the ambiguity is visible.
    """
    if pred.call == "likely_to_work":
        return round(1.0 - pred.confidence, 3)
    return round(pred.confidence, 3)


def _markers(pred: Prediction) -> tuple[str, str]:
    """(target_marker, locus_id) for the genetic tracking boxes."""
    locus = DRUG_LOCUS.get(pred.drug, "N/A")
    if pred.call == "not_applicable":
        return "N/A", locus
    if pred.call == "no_call":
        return "Ambiguous", locus
    if pred.supporting_features:
        genes = [f.gene + (f" {f.mutation}" if f.mutation else "") for f in pred.supporting_features]
        return "; ".join(genes[:3]), locus
    # No determinant found. We say "None detected", NOT "Wild-Type": AMRFinderPlus
    # only reports an essential target gene when it is altered, so we never
    # confirmed an intact/wild-type copy — absence of a call is not a positive
    # wild-type observation.
    return "None detected", locus


def _display_confidence(probability: float) -> float:
    """
    Confidence as it may be SHOWN to a human — never 0% and never 100%.

    Platt is fitted on 131-288 held-out rows here, which cannot resolve a
    probability past roughly two significant figures, so `{p:.0%}` rendering a
    calibrated 0.9999998 as "100%" is false precision on a research prototype.
    Clamping inward keeps the printed number defensible and errs toward
    understating confidence, which is the safe direction in this domain.

    Display only — never feed this back into a metric or a decision threshold.
    """
    return min(max(probability, 0.01), 0.99)


def _bio_stat_text(pred: Prediction) -> tuple[str, str]:
    """
    (bio_explanation, stat_explanation) — the honest split the brief requires:
    a KNOWN causal mechanism is described only when the evidence is a known gene;
    a statistical association is labelled as a learned pattern, never as proof.
    Only genes already in supporting_features are named (no hallucinated biology).
    """
    drug = pred.drug
    klass = DRUG_CLASS.get(drug, "this antibiotic class")
    locus = DRUG_LOCUS.get(drug, "the drug target")
    feats = _gene_names(pred)
    conf = _display_confidence(pred.confidence)

    if pred.call == "not_applicable":
        return (
            f"The molecular target for {drug} ({locus}) was not detected in this "
            "genome, so susceptibility cannot be assessed from sequence.",
            "No probability is reported: the deterministic target gate overrides "
            "the statistical model when the drug has no target to act on.",
        )

    if pred.call == "no_call":
        reason = pred.no_call_reason or "the evidence is weak or conflicting"
        return (
            f"Evidence for {drug} is inconclusive, so no confident biological "
            f"interpretation is made. {reason}",
            # Use the actual gate reason rather than a generic dual-reason line —
            # an out-of-distribution no-call can have a high probability, and
            # asserting it "does not clear the no-call margin" would contradict the
            # number shown.
            f"A safe prediction cannot be computed for this genome. {reason}",
        )

    if pred.call == "likely_to_fail":
        if pred.evidence_category == "known_gene_or_mutation":
            return (
                f"Detected a known {klass} resistance determinant: {feats}. This is "
                "an established mechanism, so the drug is predicted to fail.",
                f"The calibrated model assigns {conf:.0%} probability of resistance, "
                "consistent with the detected determinant.",
            )
        return (
            f"No confirmed causal resistance mutation at the {locus} target was "
            "detected; the prediction does not rest on a proven mechanism.",
            f"The model links {feats} to resistance as a learned statistical "
            f"pattern (association, not proven cause); calibrated probability {conf:.0%}.",
        )

    # likely_to_work
    if pred.evidence_category == "no_known_signal":
        return (
            f"No known {klass} resistance determinant was detected in this genome. "
            f"(Note: the annotator reports the {locus} target only when it is "
            "altered, so this is absence of a resistance signal, not a confirmed "
            "intact target.)",
            f"Calibrated susceptibility confidence is {1 - conf:.0%}; the genomic "
            "profile sits within the susceptible range seen in training.",
        )
    return (
        f"Although {feats} was noted, the model still predicts {drug} works; "
        "interpret with caution and confirm by lab testing.",
        f"Calibrated resistance probability is low ({conf:.0%}) despite the noted "
        "feature(s).",
    )


def report_item(pred: Prediction, use_llm: bool = True) -> dict:
    """One antibiotic -> the frontend JSON object (see the app's JSON spec).

    LLM refinement is ON by default; it falls back to the deterministic template
    whenever no OPENAI_API_KEY is set or the API call fails, so this is always safe.
    """
    bio, stat = _bio_stat_text(pred)
    if use_llm and os.environ.get("OPENAI_API_KEY"):
        bio, stat = _llm_refine_report(pred, bio, stat)
    marker, locus = _markers(pred)
    return {
        "drug": pred.drug,
        "drug_class": DRUG_CLASS.get(pred.drug, "Unknown"),
        "underlying_state": STATE_LABEL[pred.call],
        "confidence": _state_confidence(pred),
        "target_marker": marker,
        "locus_id": locus,
        "bio_explanation": bio,
        "stat_explanation": stat,
    }


def explain_report(preds: list[Prediction], use_llm: bool = True) -> list[dict]:
    """Full per-genome report array for the frontend. LLM on by default (safe fallback)."""
    return [report_item(p, use_llm=use_llm) for p in preds]


# --- 5. Overall AI clinical summary (synthesises all drugs into one narrative) ---

def _template_summary(preds: list[Prediction]) -> str:
    """Deterministic one-paragraph synthesis — the always-safe fallback."""
    if not preds:
        return "No antibiotics were evaluated for this sample."
    works = [p.drug for p in preds if p.call == "likely_to_work"]
    fails = [p.drug for p in preds if p.call == "likely_to_fail"]
    holds = [p.drug for p in preds if p.call in ("no_call", "not_applicable")]
    species = preds[0].species
    bits = []
    if works:
        bits.append(f"predicted to remain effective against this {species} sample: "
                    f"{', '.join(works)}")
    if fails:
        bits.append(f"predicted to fail (resistance indicated): {', '.join(fails)}")
    if holds:
        bits.append(f"no confident call (insufficient or ambiguous evidence): "
                    f"{', '.join(holds)}")
    body = "; ".join(bits) if bits else "no confident calls could be made"
    return (f"Across {len(preds)} antibiotics, {body}. "
            "These are sequence-based predictions and must be confirmed by "
            "standard laboratory susceptibility testing before any clinical use.")


def clinical_summary(preds: list[Prediction], use_llm: bool = True) -> str:
    """
    A short clinician-facing synthesis of the whole report. LLM-refined when a key
    is available, otherwise the deterministic template. Honesty-guarded: an LLM
    summary that names a gene absent from any prediction is rejected.
    """
    base = _template_summary(preds)
    if not (use_llm and os.environ.get("OPENAI_API_KEY") and preds):
        return base
    try:
        from openai import OpenAI

        # Bounded: an analysis makes 4 calls (3 drugs + summary) inside the request
        # path. The SDK default is 600s x 2 retries, so on bad conference wifi the
        # "falls back to the template" promise above never gets a chance to fire.
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=8.0, max_retries=1)
        compact = [
            {"drug": p.drug, "call": p.call, "evidence_category": p.evidence_category,
             "confidence": round(p.confidence, 3),
             "genes": [f.gene for f in p.supporting_features]}
            for p in preds
        ]
        system = (
            "You write a 2-3 sentence clinical summary of an antibiotic-resistance "
            "report for a clinician. Rules: only restate facts in the input; never "
            "invent genes or mutations; never state a treatment decision or dosing; "
            "keep known-mechanism vs statistical-association honesty; end by noting "
            "results need laboratory confirmation."
        )
        import json as _json
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": _json.dumps(compact)}],
            temperature=0.2,
            max_tokens=220,
        )
        text = resp.choices[0].message.content.strip()
        # honesty guard against the union of all listed genes
        allowed = {f.gene.lower() for p in preds for f in p.supporting_features}
        allowed |= {DRUG_LOCUS.get(p.drug, "").lower() for p in preds}
        for tok in _AMR_TOKEN.findall(text):
            t = tok.lower()
            if t not in allowed and not any(t in a or a in t for a in allowed if a):
                return base
        return text or base
    except Exception:  # noqa: BLE001 — any failure falls back to the template
        return base


def _llm_refine_report(pred: Prediction, bio: str, stat: str) -> tuple[str, str]:
    """Optional GPT polish. Same honesty guardrails as llm_explain; falls back."""
    try:
        from openai import OpenAI

        # Bounded: an analysis makes 4 calls (3 drugs + summary) inside the request
        # path. The SDK default is 600s x 2 retries, so on bad conference wifi the
        # "falls back to the template" promise above never gets a chance to fire.
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=8.0, max_retries=1)
        system = (
            "You refine two short antibiotic-resistance explanation strings for a "
            "clinician. Rules: only restate facts in the input, never invent genes "
            "or mutations; keep the biological vs statistical distinction; never "
            "state a treatment decision; one or two sentences each. Return exactly "
            "two lines: 'BIO: ...' then 'STAT: ...'."
        )
        user = f"Structured result: {pred.model_dump_json()}\nBIO draft: {bio}\nSTAT draft: {stat}"
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=220,
        )
        out = resp.choices[0].message.content.strip().splitlines()
        new_bio = next((l[4:].strip() for l in out if l.upper().startswith("BIO:")), bio)
        new_stat = next((l[5:].strip() for l in out if l.upper().startswith("STAT:")), stat)
        # honesty guard: reject a refinement that names a gene not in the input
        if _contains_unlisted_gene(pred, new_bio + " " + new_stat):
            return bio, stat
        return new_bio, new_stat
    except Exception:  # noqa: BLE001
        return bio, stat


#: Broad AMR-gene token detector: bla-enzymes, named beta-lactamase families
#: (CTX-M, NDM, KPC, OXA, VIM, IMP, SHV, TEM, CMY, GES, VEB), and the common
#: acquired-gene stems (aac/aad/aph/ant, sul, dfr/dhfr, tet, mph, erm, qnr, cat,
#: mcr, fosA), each optionally with an allele suffix. Catches the families the
#: old regex missed (a hallucinated "NDM-1"/"KPC-3"/"OXA-48" now trips the guard).
_AMR_TOKEN = re.compile(
    r"\b(?:bla[A-Za-z0-9\-]+"
    r"|(?:CTX-M|NDM|KPC|OXA|VIM|IMP|SHV|TEM|CMY|GES|VEB|DHA|PSE|CARB|MOX)(?:-?\d+)?"
    r"|(?:aac|aad|aph|ant|sul|dfr|dhfr|tet|mph|erm|qnr|cat|mcr|fos|arr|aadA|strA|strB)"
    r"[A-Za-z0-9()'\-]*"
    r"|gyrA|parC|parE|gyrB)\b",
    re.IGNORECASE,
)


def _contains_unlisted_gene(pred: Prediction, text: str) -> bool:
    """True if text names an AMR gene token absent from supporting_features."""
    allowed = {f.gene.lower() for f in pred.supporting_features}
    allowed |= {DRUG_LOCUS.get(pred.drug, "").lower()}
    for tok in _AMR_TOKEN.findall(text):
        t = tok.lower()
        if t not in allowed and not any(t in a or a in t for a in allowed if a):
            return True
    return False


if __name__ == "__main__":
    # Quick manual test against the synthetic fixtures — run this file directly:
    #   python src/explainer.py
    import json
    from pathlib import Path

    fixture_path = Path(__file__).parent.parent / "data/synthetic/sample_predictions.json"
    raw = json.loads(fixture_path.read_text())

    for item in raw:
        pred = Prediction(**item)
        result = explain(pred, use_llm=False)  # template-only, no API key needed
        print(f"[{result.sample_id}] {result.explanation_text}")
        print(f"    disclaimer: {result.disclaimer}\n")
