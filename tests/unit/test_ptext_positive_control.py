import json
from random import Random

from scripts.run_ptext_positive_control import (
    interpretation,
    stable_select,
    summarize_panel_a,
    summarize_panel_b,
)


def test_stable_selection_is_order_invariant():
    rows = [
        {"review_id": str(index), "excel_row": index, "text": f"text-{index}"}
        for index in range(20)
    ]
    shuffled = rows[:]
    Random(20261018).shuffle(shuffled)
    assert stable_select(rows, 5) == stable_select(shuffled, 5)


def test_panel_a_pass_requires_all_ten_label_matches():
    rows = [
        {"expected_label": "NORMAL" if index < 5 else "SUSPICIOUS", "label_match": True}
        for index in range(10)
    ]
    summary = summarize_panel_a(rows)
    assert summary == {
        "status": "PASS",
        "sample_count": 10,
        "label_match_count": 10,
        "expected_counts": {"NORMAL": 5, "SUSPICIOUS": 5},
        "reproduced_counts": {"NORMAL": 5, "SUSPICIOUS": 5},
    }
    rows[-1]["label_match"] = False
    assert summarize_panel_a(rows)["status"] == "FAIL"


def test_panel_b_summary_is_descriptive_by_human_label():
    rows = [
        {"human_label": "NORMAL", "current_predicted_label": "NORMAL"},
        {"human_label": "NORMAL", "current_predicted_label": "SUSPICIOUS"},
        {"human_label": "SUSPICIOUS", "current_predicted_label": "NORMAL"},
        {"human_label": "SUSPICIOUS", "current_predicted_label": "SUSPICIOUS"},
    ]
    summary = summarize_panel_b(rows)
    assert summary["NORMAL"] == {
        "sample_count": 2,
        "model_NORMAL_count": 1,
        "model_SUSPICIOUS_count": 1,
        "model_unavailable_count": 0,
    }
    assert summary["SUSPICIOUS"] == {
        "sample_count": 2,
        "model_NORMAL_count": 1,
        "model_SUSPICIOUS_count": 1,
        "model_unavailable_count": 0,
    }


def test_provenance_and_panel_separation_are_json_serializable():
    panel_a = {"status": "PASS"}
    panel_b = {
        "SUSPICIOUS": {"model_NORMAL_count": 3, "model_SUSPICIOUS_count": 1}
    }
    result = interpretation(panel_a, panel_b)
    assert result["recommended_priority"] == "candidate expansion"
    serialized = json.dumps({
        "panel_a": {"provenance": "previous-model expected"},
        "panel_b": {"provenance": "human-reviewed exploratory"},
        "interpretation": result,
    })
    assert "previous-model expected" in serialized
    assert "human-reviewed exploratory" in serialized
