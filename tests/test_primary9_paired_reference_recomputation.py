from __future__ import annotations

import csv
import hashlib
import importlib
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def _module():
    module_path = ROOT / "scripts" / "recompute_primary9_paired_reference.py"
    assert module_path.is_file(), "Paired-reference recomputation script is missing"
    return importlib.import_module("scripts.recompute_primary9_paired_reference")


def test_original_cedia_manifest_is_matched_to_history_by_record_id(tmp_path):
    package_module = importlib.import_module("scripts.package_cedia_primary9_gold_evidence")
    manifest = tmp_path / "original_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("record_id", "rhythm_label_name"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            (
                {"record_id": "r1", "rhythm_label_name": "AFIB"},
                {"record_id": "r2", "rhythm_label_name": "NSR"},
            )
        )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()

    evidence = package_module._original_label_manifest_provenance(
        manifest,
        {"r1": "AFIB", "r2": "NSR"},
        expected_sha256=digest,
        expected_records=2,
    )

    assert evidence["matched_record_ids"] == 2
    assert evidence["source_label_field"] == "rhythm_label_name"
    assert evidence["matched_label_field"] == "original_rhythm_label"
    assert package_module.write_bundle_hash_manifest(tmp_path) == 1
    assert package_module._read_csv(tmp_path / "SHA256SUMS.csv")[0]["relative_path"] == "original_manifest.csv"


def test_same_predictions_are_scored_against_explicit_reference_columns():
    module = _module()
    predictions = [
        {"record_id": "r1", "predicted_label": "A"},
        {"record_id": "r2", "predicted_label": "B"},
        {"record_id": "r3", "predicted_label": "A"},
        {"record_id": "r4", "predicted_label": "B"},
    ]
    labels = [
        {"record_id": "r1", "original_rhythm_label": "A", "final_rhythm_label": "A"},
        {"record_id": "r2", "original_rhythm_label": "B", "final_rhythm_label": "A"},
        {"record_id": "r3", "original_rhythm_label": "B", "final_rhythm_label": "B"},
        {"record_id": "r4", "original_rhythm_label": "B", "final_rhythm_label": "B"},
    ]

    original = module.score_predictions_by_reference(
        predictions, labels, ["A", "B"], "original_rhythm_label"
    )
    final = module.score_predictions_by_reference(
        predictions, labels, ["A", "B"], "final_rhythm_label"
    )

    assert original["correct"] == 3
    assert original["accuracy"] == 0.75
    assert final["correct"] == 2
    assert final["accuracy"] == 0.5
    assert original["confusion_matrix_true_rows_predicted_columns"] != final[
        "confusion_matrix_true_rows_predicted_columns"
    ]


def test_cedia_evidence_reports_original_and_final_results_separately(tmp_path):
    module = _module()
    bundle = ROOT / "outputs/reviewer_verification/arrhythmia_primary9/cedia_gold_2026-05-26"
    report, per_class = module.build_evidence(bundle, ROOT / "models/gold_master/gold_master_primary9_model.pth")

    assert report["source_original_label_manifest"]["records_matching"] == 672
    assert report["source_original_label_manifest"]["source_label_field"] == "rhythm_label_name"
    assert report["source_original_label_manifest"]["matched_label_field"] == "original_rhythm_label"
    assert report["historical_cedia_source_report"]["original_reference"]["correct"] == 476
    assert not report["historical_cedia_source_report"]["original_reference_confusion_matrix_available"]
    assert report["historical_cedia_source_report"]["updated_reference"]["correct"] == 605
    assert report["historical_accuracy_reconciliation"] == {
        "original_correct": 476,
        "updated_records": 129,
        "updated_rows_equal_recorded_top1": 129,
        "updated_rows_differ_from_original_label": 129,
        "final_correct_reconciled": 605,
        "final_accuracy_reconciled": 605 / 672,
        "scope": "Aggregate arithmetic only; this does not reproduce the model inference run.",
    }
    assert report["fixed_checkpoint_results"]["original_external_labels"]["correct"] == 466
    assert report["fixed_checkpoint_results"]["updated_external_labels"]["correct"] == 568
    assert not report["clinical_review_provenance"]["clinical_review_claimed_in_source_artifacts"]
    assert math.isclose(
        report["fixed_checkpoint_results"]["original_external_labels"]["macro_f1"],
        0.6888407765536525,
    )
    assert report["checkpoint"]["run_manifest_sha256"] != report["checkpoint"]["sha256"]
    assert len(per_class) == 18

    module.write_evidence(report, per_class, tmp_path)
    assert (tmp_path / "primary9_paired_reference_metrics.json").is_file()
    assert (tmp_path / "primary9_per_class_paired_metrics.csv").is_file()


def test_cli_accepts_repository_relative_paths(tmp_path, monkeypatch):
    module = _module()
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "recompute_primary9_paired_reference.py",
            "--bundle",
            "outputs/reviewer_verification/arrhythmia_primary9/cedia_gold_2026-05-26",
            "--checkpoint",
            "models/gold_master/gold_master_primary9_model.pth",
            "--out-dir",
            str(tmp_path),
        ],
    )

    assert module.main() == 0
    assert (tmp_path / "primary9_paired_reference_metrics.json").is_file()
    assert (tmp_path / "primary9_per_class_paired_metrics.csv").is_file()
    hashes = module._read_csv(tmp_path / "SHA256SUMS.csv")
    assert {row["file"] for row in hashes} == {
        "primary9_paired_reference_metrics.json",
        "primary9_per_class_paired_metrics.csv",
    }
