from __future__ import annotations

from titan_v4.metrics.external_validation import combined_summary


def test_combined_summary_does_not_use_cascade_as_primary_source():
    payload = combined_summary(
        {
            "module": "arrhythmia_primary9",
            "evidence_status": "REPRODUCED_FULL_672_INFERENCE",
            "evaluation_status": "FULL_672",
            "accuracy": 0.69345,
        },
        {"module": "pathology_primary5", "evidence_status": "LEGACY_NOT_REPRODUCED", "accuracy": 0.90256},
        {"module": "cascade_safety_annex", "evidence_status": "LEGACY_NOT_REPRODUCED", "role": "safety_annex"},
    )
    assert payload["label_set"] == "final external validation label set"
    assert payload["external_training_allowed"] is False
    assert payload["external_threshold_tuning_allowed"] is False
    assert payload["cascade_annex"]["role"] == "safety_annex"
    assert payload["arrhythmia_primary9"]["evidence_status"] == "REPRODUCED_FULL_672_INFERENCE"
    assert "not the official hidden Challenge test set" in payload["evaluation_scope"]

