from __future__ import annotations

import json
import csv
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_recomputed_primary9_inference_metrics_are_exact():
    report = json.loads(
        (ROOT / "outputs/reproduced/primary9/primary9_recomputed_report.json")
        .read_text(encoding="utf-8")
    )
    assert report["label_set"] == "final external validation label set"
    assert report["evaluation_status"] == "FULL_672"
    assert report["total_records"] == 672
    assert report["correct_predictions"] == 466
    assert report["accuracy"] == 0.6934523809523809
    assert report["macro_f1"] == 0.6888407765536525
    assert report["weighted_f1"] == 0.6895899672373766
    assert {"AFIB", "SB", "STACH", "NSR", "PVC", "RBBB", "LBBB", "PAC", "1AVB"} <= set(
        report["classification_report"]
    )


def test_older_primary9_report_is_marked_superseded():
    report = json.loads(
        (ROOT / "outputs/gold_master_external_validation/arrhythmia_primary9/primary9_external_validation_report.json")
        .read_text(encoding="utf-8")
    )
    assert report["evidence_status"] == "LEGACY_SUPERSEDED_NOT_REPRODUCED_FROM_CURRENT_LABELS"
    assert report["superseded_by"] == "outputs/reproduced/primary9/primary9_recomputed_report.json"


def test_primary9_confusion_matrix_shape():
    report = json.loads(
        (ROOT / "outputs/reproduced/primary9/primary9_recomputed_report.json")
        .read_text(encoding="utf-8")
    )
    matrix = report["confusion_matrix"]
    assert len(matrix) == 9
    assert all(len(row) == 9 for row in matrix)
    assert sum(sum(row) for row in matrix) == 672


def test_primary9_run_manifest_records_hashes_and_environment():
    output_dir = ROOT / "outputs/reproduced/primary9"
    manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["evaluation_status"] == "FULL_672"
    assert manifest["record_count"] == 672
    assert len(manifest["code_files"]) >= 8
    assert manifest["runtime"]["deterministic_algorithms"] is True
    assert manifest["source_dataset"]["version"] == "1.0.3"
    assert manifest["source_dataset_counts"]["ptb-xl"] == 378
    for relative_path, expected_hash in manifest["code_files"].items():
        actual_hash = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest().upper()
        assert actual_hash == expected_hash
    for filename, expected_hash in manifest["outputs"].items():
        actual_hash = hashlib.sha256((output_dir / filename).read_bytes()).hexdigest().upper()
        assert actual_hash == expected_hash


def test_prediction_artifact_contains_all_records_and_nine_probabilities():
    predictions = list(
        csv.DictReader(
            (ROOT / "outputs/reproduced/primary9/primary9_record_predictions.csv").open(
                "r", newline="", encoding="utf-8"
            )
        )
    )
    assert len(predictions) == 672
    probability_fields = [field for field in predictions[0] if field.startswith("prob_")]
    assert len(probability_fields) == 9
    assert all(abs(sum(float(row[field]) for field in probability_fields) - 1.0) < 1e-6 for row in predictions)

