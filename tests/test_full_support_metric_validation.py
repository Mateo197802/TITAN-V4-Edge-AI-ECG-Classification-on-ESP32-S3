from __future__ import annotations

from titan_v4.metrics.external_validation import combined_summary


def test_combined_summary_does_not_use_cascade_as_primary_source():
    payload = combined_summary(
        {
            "module": "arrhythmia_primary9",
            "evidence_status": "REPRODUCED_SINGLE_LABEL_DIAGNOSTIC_NOT_EXTERNAL_VALIDATION",
            "evaluation_status": "FULL_672_DIAGNOSTIC_ONLY",
            "accuracy": 0.69345,
            "official_challenge_metric": False,
            "external_validation_claim": False,
        },
        {"module": "pathology_primary5", "evidence_status": "LEGACY_NOT_REPRODUCED", "accuracy": 0.90256},
        {"module": "cascade_safety_annex", "evidence_status": "LEGACY_NOT_REPRODUCED", "role": "safety_annex"},
    )
    assert payload["label_set"] == "frozen 672-record source-header single-label diagnostic set"
    assert payload["external_training_allowed"] is False
    assert payload["external_threshold_tuning_allowed"] is False
    assert payload["cascade_annex"]["role"] == "safety_annex"
    assert payload["arrhythmia_primary9"]["evidence_status"] == "REPRODUCED_SINGLE_LABEL_DIAGNOSTIC_NOT_EXTERNAL_VALIDATION"
    assert payload["metrics_comparable_across_protocols"] is False
    assert "not the official Challenge metric or hidden test set" in payload["evaluation_scope"]

