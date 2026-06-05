from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_primary9_external_metrics_are_exact():
    report = json.loads(
        (ROOT / "outputs/gold_master_external_validation/arrhythmia_primary9/primary9_external_validation_report.json")
        .read_text(encoding="utf-8")
    )
    assert report["label_set"] == "final external validation label set"
    assert report["total_records"] == 672
    assert report["correct_predictions"] == 605
    assert report["accuracy"] == 0.9002976190476191
    assert report["macro_f1"] == 0.8781916793252358
    assert report["weighted_f1"] == 0.9008533778242968
    assert set(report["classification_report"]) == {"AFIB", "SB", "STACH", "NSR", "PVC", "RBBB", "LBBB", "PAC", "1AVB"}


def test_primary9_confusion_matrix_shape():
    report = json.loads(
        (ROOT / "outputs/gold_master_external_validation/arrhythmia_primary9/primary9_external_validation_report.json")
        .read_text(encoding="utf-8")
    )
    matrix = report["confusion_matrix"]
    assert len(matrix) == 9
    assert all(len(row) == 9 for row in matrix)
    assert sum(sum(row) for row in matrix) == 672

