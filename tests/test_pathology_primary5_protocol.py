from __future__ import annotations

import json
import csv
from pathlib import Path

from titan_v4.evaluation import pathology_primary5_protocol as protocol


ROOT = Path(__file__).resolve().parents[1]


def test_pathology_external_metrics_are_exact():
    report = json.loads(
        (ROOT / "outputs/gold_master_external_validation/pathology_primary5/pathology_primary5_external_summary.json")
        .read_text(encoding="utf-8")
    )
    assert report["label_set"] == "historical Primary-5 target set; provenance incomplete"
    assert report["primary_pathologies"] == ["IMI", "ALMI", "ILMI", "LAE", "ISC_"]
    assert report["accuracy"] == 0.90256
    assert "per-label accuracy" in report["accuracy_definition"]
    assert report["macro_f1"] == 0.66874
    assert report["external_training_allowed"] is False
    assert report["external_threshold_tuning_allowed"] is False
    assert report["evidence_status"] == "LEGACY_NOT_REPRODUCED_FROM_RECORD_LEVEL_PREDICTIONS"


def test_fixed_threshold_summary_uses_only_test_rows_and_unweighted_class_means():
    summarize = getattr(protocol, "summarize_fixed_threshold", None)
    assert callable(summarize)

    report = summarize(
        [
            {"split": "test", "true_IMI": 1, "score_IMI": 0.8, "true_ALMI": 0, "score_ALMI": 0.2},
            {"split": "test", "true_IMI": 0, "score_IMI": 0.9, "true_ALMI": 1, "score_ALMI": 0.7},
            {"split": "calibration", "true_IMI": 1, "score_IMI": 0.0, "true_ALMI": 1, "score_ALMI": 0.0},
        ],
        threshold=0.5,
        pathologies=("IMI", "ALMI"),
    )

    assert report["test_records"] == 2
    assert report["per_label_accuracy"] == 0.75
    assert report["macro_f1"] == (2 / 3 + 1) / 2
    assert report["per_class"]["IMI"]["accuracy"] == 0.5
    assert report["per_class"]["ALMI"]["accuracy"] == 1.0


def test_cli_recomputes_fixed_threshold_sensitivities_from_prediction_rows(tmp_path):
    labels = protocol.PRIMARY5_PATHOLOGIES
    fields = ["record_id", "split"] + [field for name in labels for field in (f"true_{name}", f"score_{name}")]
    predictions = tmp_path / "scores.csv"
    with predictions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record_id, split, truth, score in (
            ("00001", "calibration", 1, 0.9),
            ("00002", "test", 1, 0.8),
            ("00003", "test", 0, 0.2),
        ):
            row = {"record_id": record_id, "split": split}
            for name in labels:
                row[f"true_{name}"] = truth
                row[f"score_{name}"] = score if truth else 0.1
            writer.writerow(row)

    output_dir = tmp_path / "report"
    assert protocol.main([
        "--predictions_csv", str(predictions),
        "--out_dir", str(output_dir),
        "--fixed-threshold", "0.50",
        "--fixed-threshold", "0.65",
    ]) == 0

    report = json.loads((output_dir / "pathology_primary5_report.json").read_text(encoding="utf-8"))
    assert report["fixed_threshold_sensitivities"]["0.5"]["test_records"] == 2
    assert report["fixed_threshold_sensitivities"]["0.65"]["per_label_accuracy"] == 1.0
    assert b"\r\n" not in (output_dir / "pathology_primary5_per_class.csv").read_bytes()
    assert b"\r\n" not in (output_dir / "pathology_primary5_predictions.csv").read_bytes()

