from __future__ import annotations

import importlib

import pytest


def test_current_predictions_are_scored_against_both_aligned_label_tables():
    module = importlib.import_module("titan_v4.reproducibility.legacy_comparison")
    predictions = [
        {"source_label": "source", "record_id": "r1", "true_label": "AFIB", "predicted_label": "AFIB"},
        {"source_label": "source", "record_id": "r2", "true_label": "SB", "predicted_label": "AFIB"},
    ]
    current = [
        {"source": "source", "record_id": "R1", "rhythm_label": "AFIB"},
        {"source": "source", "record_id": "R2", "rhythm_label": "SB"},
    ]
    legacy = [
        {"source": "source", "record_id": "r1", "rhythm_label": "AFIB"},
        {"source": "source", "record_id": "r2", "rhythm_label": "AFIB"},
    ]
    stored_report = {
        "total_records": 2,
        "correct_predictions": 2,
        "accuracy": 1.0,
        "macro_f1": 1.0,
        "weighted_f1": 1.0,
    }

    report, rows = module.build_comparison(
        predictions,
        current,
        legacy,
        historical_report=stored_report,
        expected_records=2,
    )

    assert report["current_checkpoint_vs_rebuilt_labels"]["correct_predictions"] == 1
    assert report["current_checkpoint_vs_historical_labels"]["correct_predictions"] == 2
    assert report["label_table_comparison"]["different_label_count"] == 1
    assert rows[1]["historical_label"] == "AFIB"


def test_comparison_fails_closed_when_record_keys_differ():
    module = importlib.import_module("titan_v4.reproducibility.legacy_comparison")
    prediction = [{"source_label": "source", "record_id": "r1", "true_label": "AFIB", "predicted_label": "AFIB"}]
    current = [{"source": "source", "record_id": "r1", "rhythm_label": "AFIB"}]
    legacy = [{"source": "source", "record_id": "r2", "rhythm_label": "AFIB"}]
    stored_report = {"total_records": 1, "correct_predictions": 1, "accuracy": 1, "macro_f1": 1, "weighted_f1": 1}

    with pytest.raises(ValueError, match="record keys differ"):
        module.build_comparison(
            prediction, current, legacy, historical_report=stored_report, expected_records=None
        )


def test_comparison_fails_closed_when_stored_truth_does_not_match_current_labels():
    module = importlib.import_module("titan_v4.reproducibility.legacy_comparison")
    prediction = [{"source_label": "source", "record_id": "r1", "true_label": "SB", "predicted_label": "AFIB"}]
    labels = [{"source": "source", "record_id": "r1", "rhythm_label": "AFIB"}]
    stored_report = {"total_records": 1, "correct_predictions": 1, "accuracy": 1, "macro_f1": 1, "weighted_f1": 1}

    with pytest.raises(ValueError, match="stored truth differs"):
        module.build_comparison(
            prediction, labels, labels, historical_report=stored_report, expected_records=1
        )


def test_pathology_audit_never_scores_aggregate_without_targets_and_predictions(tmp_path):
    module = importlib.import_module("titan_v4.evaluation.pathology_evidence")
    label_path = tmp_path / "labels.csv"
    label_path.write_text("record_id,pathology_label_names\nr1,\n", encoding="utf-8")
    checkpoint_path = tmp_path / "model.pth"
    checkpoint_path.write_bytes(b"test checkpoint")
    prediction_path = tmp_path / "missing_predictions.csv"
    historical = {
        "accuracy": 0.90256,
        "accuracy_definition": "legacy per-label accuracy",
        "macro_f1": 0.66874,
        "primary_pathologies": ["IMI", "ALMI", "ILMI", "LAE", "ISC_"],
        "checkpoint_sha256": "not recorded",
    }
    training = {
        "pathology_head": True,
        "pathology_class_names": ["IMI", "ASMI", "LVH", "ISC_", "ISCAL", "NST_", "ILMI", "AMI", "ALMI", "LAE"],
        "pathology_weight": 0,
        "pathology_pos_weight_stats": {"labeled_windows": 0},
        "pathology_sidecar": "not included; original local path removed",
    }
    cedia_candidate = {
        "model": {
            "checkpoint_sha256": "e9a44e4eea8ebb8f89d5e32909ae4afcc96442d1fa8eacb1e57a73bfa498353b",
            "pathology_weight": 0.2,
            "pathology_labeled_windows": 1811,
        },
        "pathology_label_map_sha256": "e5f677944efc18db42248db4130fe51dc9ae863a68b8056a3542f52b786b8409",
        "pathology_prediction_artifacts_found": False,
    }

    report = module.build_audit(
        historical,
        training,
        checkpoint_path=checkpoint_path,
        label_path=label_path,
        prediction_path=prediction_path,
        cedia_candidate=cedia_candidate,
    )

    assert report["evaluation_performed"] is False
    assert report["scope"]["historical_metric_reproduced"] is False
    assert report["checked_candidate_reference_table"]["records_with_any_primary5_pathology_label"] == 0
    assert report["historical_record_level_prediction_file"]["available"] is False
    assert "per-class thresholds and calibration split for the historical aggregate" in report["missing_evidence"]
    assert report["separate_cedia_candidate"]["model"]["checkpoint_sha256"] == cedia_candidate["model"]["checkpoint_sha256"]
    assert report["separate_cedia_candidate"]["matches_distributed_checkpoint"] is False
    assert report["separate_cedia_candidate"]["pathology_prediction_artifacts_found"] is False
