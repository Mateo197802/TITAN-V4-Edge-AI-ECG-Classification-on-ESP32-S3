from __future__ import annotations

import csv

from titan_v4.evaluation.pathology_evidence import build_audit


def test_readiness_audit_keeps_historical_claim_separate_from_cedia_followup(tmp_path):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"weights")
    labels = tmp_path / "labels.csv"
    with labels.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["pathology_label_names"])
        writer.writeheader()
        writer.writerow({"pathology_label_names": ""})

    report = build_audit(
        {"accuracy": 0.90256, "macro_f1": 0.66874, "primary_pathologies": ["IMI"]},
        {"pathology_weight": 0.0, "pathology_pos_weight_stats": {"labeled_windows": 0}},
        checkpoint_path=checkpoint,
        label_path=labels,
        prediction_path=tmp_path / "historical-predictions.csv",
        cedia_candidate={"model": {"checkpoint_sha256": "separate"}},
        cedia_evaluation={
            "dataset": {"version": "1.0.3"},
            "checkpoint": {"sha256": "separate"},
            "calibration_records": 273,
            "test_records": 406,
            "per_label_accuracy": 0.715,
            "macro_f1": 0.414,
            "fixed_threshold_sensitivities": {
                "0.5": {"per_label_accuracy": 0.732, "macro_f1": 0.418},
            },
        },
    )

    assert report["evidence_status"] == "HISTORICAL_METRIC_NOT_REPRODUCIBLE_FROM_AVAILABLE_ARTIFACTS"
    assert report["scope"]["historical_metric_reproduced"] is False
    assert report["separate_cedia_candidate"]["follow_up_evaluation"]["test_records"] == 406
    assert report["separate_cedia_candidate"]["historical_primary5_claim_linked"] is False
