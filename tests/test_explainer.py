"""
Run with: pytest tests/test_explainer.py -v
Tests the template path only (no API key required) — this is what CI / your
teammates can run without touching OpenAI credits.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from schemas import Prediction
from explainer import explain, template_explain


FIXTURES = json.loads(
    (Path(__file__).parent.parent / "data/synthetic/sample_predictions.json").read_text()
)


def _load(sample_id: str) -> Prediction:
    for item in FIXTURES:
        if item["sample_id"] == sample_id:
            return Prediction(**item)
    raise ValueError(f"no fixture {sample_id}")


def test_known_gene_mentions_gene_name():
    pred = _load("SIM-001")
    text = template_explain(pred)
    assert "blaCTX-M-15" in text
    assert "LIKELY TO FAIL" in text


def test_statistical_association_is_hedged():
    pred = _load("SIM-003")
    text = template_explain(pred)
    assert "statistical" in text.lower() or "not a confirmed" in text.lower()


def test_no_call_includes_reason():
    pred = _load("SIM-004")
    text = template_explain(pred)
    assert "uncertain range" in text


def test_not_applicable_overrides_call():
    pred = _load("SIM-006")
    assert pred.target_gate_status == "absent"
    text = template_explain(pred)
    assert "not applicable" in text.lower()


def test_disclaimer_always_present():
    for item in FIXTURES:
        pred = Prediction(**item)
        result = explain(pred, use_llm=False)
        assert "confirmed with" in result.disclaimer.lower()
        assert "laboratory" in result.disclaimer.lower()


def test_confidence_label_is_percentage():
    pred = _load("SIM-001")
    result = explain(pred, use_llm=False)
    assert result.confidence_label == "94%"
