from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from titan_v4.metrics.external_validation import manifest_file_fingerprint
from scripts.verify_primary9_gold_evidence import _read_csv, confusion_matrix_metrics


DEFAULT_BUNDLE = ROOT / "outputs/reviewer_verification/arrhythmia_primary9/cedia_gold_2026-05-26"
DEFAULT_CHECKPOINT = ROOT / "models/gold_master/gold_master_primary9_model.pth"
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/reviewer_verification/arrhythmia_primary9/paired_reference_recomputation_2026-10-03"
)
ORIGINAL_MANIFEST_SHA256 = "8190EC19F4887D4618213CBF95B9DA3B68948DDDAB4F0233D975EBA5938E5B8C"


def _fingerprint(path: Path) -> str:
    return manifest_file_fingerprint(path)[0].upper()


def _record_key(value: str) -> str:
    return value.strip().casefold()


def score_predictions_by_reference(
    predictions: list[dict[str, str]],
    labels: list[dict[str, str]],
    classes: list[str],
    reference_field: str,
) -> dict[str, object]:
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("Class order must be non-empty and unique")

    predictions_by_id = {_record_key(row["record_id"]): row for row in predictions}
    labels_by_id = {_record_key(row["record_id"]): row for row in labels}
    if len(predictions_by_id) != len(predictions) or len(labels_by_id) != len(labels):
        raise ValueError("Duplicate record ID in predictions or label history")
    if predictions_by_id.keys() != labels_by_id.keys():
        raise ValueError("Prediction IDs do not exactly match label-history IDs")

    class_index = {name: index for index, name in enumerate(classes)}
    matrix = [[0 for _ in classes] for _ in classes]
    for key, prediction in predictions_by_id.items():
        label_row = labels_by_id[key]
        if reference_field not in label_row:
            raise ValueError(f"Missing reference column: {reference_field}")
        try:
            true_index = class_index[label_row[reference_field]]
            predicted_index = class_index[prediction["predicted_label"]]
        except KeyError as error:
            raise ValueError(f"Unknown Primary-9 class in input: {error.args[0]}") from error
        matrix[true_index][predicted_index] += 1

    metrics = confusion_matrix_metrics(matrix)
    metrics["confusion_matrix_true_rows_predicted_columns"] = matrix
    return metrics


def _assert_close(actual: object, expected: object, field: str) -> None:
    if not math.isclose(float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"{field} differs from the archived CEDIA result")


def build_evidence(bundle: Path, checkpoint: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    bundle = bundle.resolve()
    checkpoint = checkpoint.resolve()
    projection_path = bundle / "primary9_cedia_gold_source_projection.json"
    predictions_path = bundle / "fresh_inference/primary9_record_predictions.csv"
    labels_path = bundle / "primary9_label_history_672.csv"
    updates_path = bundle / "primary9_target_updates_129.csv"
    run_manifest_path = bundle / "fresh_inference/run_manifest.json"
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    predictions = _read_csv(predictions_path)
    labels = _read_csv(labels_path)
    updates = _read_csv(updates_path)
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))

    classes = projection["endpoint"]["classes"]
    source_manifest = projection["source_artifacts"]["cedia_original_label_manifest"]
    review_crosscheck = projection["label_review_crosscheck"]
    p0_review = review_crosscheck.get("separate_p0_review")
    if source_manifest["sha256"].casefold() != ORIGINAL_MANIFEST_SHA256.casefold():
        raise ValueError("Original CEDIA label-manifest checksum differs from the verified source")
    if (
        source_manifest["source_label_field"] != "rhythm_label_name"
        or source_manifest["matched_label_field"] != "original_rhythm_label"
    ):
        raise ValueError("Original CEDIA label-field mapping differs from the verified source")
    if source_manifest["records"] != 672 or source_manifest["matched_record_ids"] != 672:
        raise ValueError("Original CEDIA label manifest does not cover the 672-record cohort")
    if len(labels) != 672 or len(predictions) != 672:
        raise ValueError("Paired-reference evidence must contain 672 predictions and label rows")
    if (
        review_crosscheck["source_update_rows"] != len(updates)
        or review_crosscheck["rows_with_decision"] != len(updates)
        or review_crosscheck["rows_with_timestamp"] != len(updates)
    ):
        raise ValueError("CEDIA source update cross-check does not match the packaged update rows")

    original = score_predictions_by_reference(predictions, labels, classes, "original_rhythm_label")
    updated = score_predictions_by_reference(predictions, labels, classes, "final_rhythm_label")
    history_by_id = {_record_key(row["record_id"]): row for row in labels}
    prediction_by_id = {_record_key(row["record_id"]): row for row in predictions}
    updates_by_id = {_record_key(row["record_id"]): row for row in updates}
    if len(updates_by_id) != len(updates):
        raise ValueError("Duplicate record ID in target update audit")

    changed_ids = {
        key
        for key, row in history_by_id.items()
        if row["original_rhythm_label"] != row["final_rhythm_label"]
    }
    if changed_ids != set(updates_by_id):
        raise ValueError("Target-update IDs do not exactly match changed label-history rows")
    for key, update in updates_by_id.items():
        history_row = history_by_id[key]
        if update["original_label"] != history_row["original_rhythm_label"]:
            raise ValueError("Target update original label differs from the 672-row label history")
        if update["final_label"] != history_row["final_rhythm_label"]:
            raise ValueError("Target update final label differs from the 672-row label history")

    local_top1_matches = sum(
        prediction_by_id[key]["predicted_label"] == update["cedia_model_top1_label"]
        for key, update in updates_by_id.items()
    )
    final_top1_matches = sum(
        update["final_label"] == update["cedia_model_top1_label"]
        for update in updates_by_id.values()
    )
    original_top1_mismatches = sum(
        update["original_label"] != update["cedia_model_top1_label"]
        for update in updates_by_id.values()
    )

    historical = projection["endpoint"]
    historical_final = historical["final_external"]
    historical_original = historical["original_external"]
    historical_matrix = confusion_matrix_metrics(historical_final["confusion_matrix"])
    for field in ("total", "correct"):
        if int(historical_final[field]) != int(historical_matrix["records" if field == "total" else "correct"]):
            raise ValueError(f"Archived CEDIA final {field} differs from its confusion matrix")
    for field in ("accuracy", "macro_f1", "weighted_f1"):
        _assert_close(historical_final[field], historical_matrix[field], field)
    if final_top1_matches != len(updates) or original_top1_mismatches != len(updates):
        raise ValueError("Updated labels do not all replace a distinct original label with recorded Top-1")
    reconciled_correct = int(historical_original["correct"]) + len(updates)
    if reconciled_correct != int(historical_final["correct"]):
        raise ValueError("Original correct count plus audited label updates does not match final correct count")

    checkpoint_sha = _fingerprint(checkpoint)
    run_manifest_sha = _fingerprint(run_manifest_path)
    if run_manifest.get("model", {}).get("sha256", "").casefold() != checkpoint_sha.casefold():
        raise ValueError("Run-manifest checkpoint hash does not match the supplied checkpoint")
    prediction_sha = _fingerprint(predictions_path)
    if run_manifest.get("outputs", {}).get(predictions_path.name, "").casefold() != prediction_sha.casefold():
        raise ValueError("Run manifest does not authenticate the supplied prediction table")

    per_class: list[dict[str, object]] = []
    for index, class_name in enumerate(classes):
        per_class.append(
            {
                "reference": "original_external_labels",
                "class": class_name,
                **original["per_class"][index],
            }
        )
        per_class.append(
            {
                "reference": "updated_external_labels",
                "class": class_name,
                **updated["per_class"][index],
            }
        )

    report: dict[str, object] = {
        "schema_version": 1,
        "report_date": "2026-10-03",
        "evaluation_scope": (
            "Fixed row-level predictions from the packaged local inference, re-scored against "
            "the original and updated 672-row label columns without retraining or threshold changes."
        ),
        "classes": classes,
        "checkpoint": {
            "sha256": checkpoint_sha,
            "run_manifest_sha256": run_manifest_sha,
        },
        "source_original_label_manifest": {
            "path_from_cedia": source_manifest["cedia_relative_path"],
            "sha256": source_manifest["sha256"],
            "records": source_manifest["records"],
            "records_matching": source_manifest["matched_record_ids"],
            "source_label_field": source_manifest["source_label_field"],
            "matched_label_field": source_manifest["matched_label_field"],
            "raw_manifest_includes_demographics_or_absolute_paths": True,
        },
        "input_artifacts": {
            "checkpoint": {
                "path_from_repo": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
                "sha256": checkpoint_sha,
            },
            "predictions": {
                "path_from_repo": str(predictions_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": prediction_sha,
            },
            "label_history": {
                "path_from_repo": str(labels_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": _fingerprint(labels_path),
            },
            "target_updates": {
                "path_from_repo": str(updates_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": _fingerprint(updates_path),
            },
            "source_projection": {
                "path_from_repo": str(projection_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": _fingerprint(projection_path),
            },
            "run_manifest": {
                "path_from_repo": str(run_manifest_path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": run_manifest_sha,
            },
        },
        "label_change_audit": {
            "records": len(labels),
            "changed_labels": len(changed_ids),
            "target_update_rows": len(updates),
            "updated_labels_equal_historical_cedia_top1": final_top1_matches,
            "local_checkpoint_predictions_matching_historical_cedia_top1_on_updated_rows": local_top1_matches,
            "interpretation": (
                "Label equality with the recorded Top-1 is a value-level match; it does not establish "
                "how the update decisions were made."
            ),
        },
        "fixed_checkpoint_results": {
            "original_external_labels": original,
            "updated_external_labels": updated,
        },
        "historical_cedia_source_report": {
            "original_reference": historical_original,
            "original_reference_confusion_matrix_available": False,
            "original_reference_metrics_recomputed_from_row_level_predictions": False,
            "updated_reference": {
                key: historical_final[key]
                for key in ("total", "correct", "accuracy", "macro_f1", "weighted_f1")
            },
            "updated_reference_metrics_recomputed_from_saved_confusion_matrix": True,
            "updated_reference_confusion_matrix_available": True,
            "prediction_rows_for_historical_cedia_report_available": False,
            "checkpoint_sha256_recorded_in_historical_report": False,
        },
        "historical_accuracy_reconciliation": {
            "original_correct": int(historical_original["correct"]),
            "updated_records": len(updates),
            "updated_rows_equal_recorded_top1": final_top1_matches,
            "updated_rows_differ_from_original_label": original_top1_mismatches,
            "final_correct_reconciled": reconciled_correct,
            "final_accuracy_reconciled": reconciled_correct / int(historical_final["total"]),
            "scope": "Aggregate arithmetic only; this does not reproduce the model inference run.",
        },
        "clinical_review_provenance": {
            "clinical_review_claimed_in_source_artifacts": False,
            "source_update_rows_with_decisions_and_timestamps": review_crosscheck["rows_with_timestamp"],
            "reviewer_identity_field_present": review_crosscheck["reviewer_identity_field_present"],
            "review_protocol_independently_verified": False,
            "separate_p0_review_rows": p0_review["records"] if p0_review else None,
            "p0_review_rows_matching_gold_updates": (
                p0_review["records_matching_gold_updates"] if p0_review else None
            ),
            "interpretation": " ".join(
                (
                    "The 129-row source has decision and timestamp fields but no reviewer identity or review "
                    "protocol, so it does not independently establish a clinical review process.",
                    (
                        "The separate P0 review set covers other records."
                        if p0_review
                        else "This packaging run did not include a separate P0 review-set cross-check."
                    ),
                )
            ),
        },
    }
    return report, per_class


def write_evidence(report: dict[str, object], per_class: list[dict[str, object]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "primary9_paired_reference_metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    csv_path = output_dir / "primary9_per_class_paired_metrics.csv"
    fields = ["reference", "class", "support", "precision", "recall", "f1"]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in per_class:
            writer.writerow(
                {
                    **row,
                    "precision": f"{float(row['precision']):.12f}",
                    "recall": f"{float(row['recall']):.12f}",
                    "f1": f"{float(row['f1']):.12f}",
                }
            )

    checksum_path = output_dir / "SHA256SUMS.csv"
    with checksum_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("file", "sha256", "bytes"), lineterminator="\n")
        writer.writeheader()
        for path in sorted(item for item in output_dir.iterdir() if item.is_file() and item != checksum_path):
            digest, size_bytes = manifest_file_fingerprint(path)
            writer.writerow({"file": path.name, "sha256": digest, "bytes": size_bytes})


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute fixed Primary-9 predictions against original and updated labels."
    )
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report, per_class = build_evidence(args.bundle, args.checkpoint)
    output_dir = args.out_dir.resolve()
    write_evidence(report, per_class, output_dir)
    print(json.dumps(report["historical_cedia_source_report"], indent=2))
    print(json.dumps(report["fixed_checkpoint_results"], indent=2))
    print(f"Evidence written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
