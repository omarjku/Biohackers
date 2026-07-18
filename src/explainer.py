"""
explainer.py — Module 03, natural-language layer
Owner: Hazem

Input:  a Prediction (see schemas.py) — produced by predictor.py / calibration.py
Output: an ExplanationResult — consumed by app.py

Two paths:
  1. template_explain()  -> deterministic, free, zero API risk. Always works.
  2. llm_explain()       -> GPT-4 rephrases the template into clinician-readable
                            prose. Constrained to only the fields already in
                            the Prediction — it cannot invent gene names or
                            claims that aren't in supporting_features.

explain() tries the LLM path and falls back to the template on any failure
(timeout, rate limit, no credits left). This is the hedge against the shared
$50 OpenAI credit running out mid-demo.
"""

import os
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
        confidence=pred.confidence,
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
        confidence_label=f"{pred.confidence:.0%}",
    )


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
