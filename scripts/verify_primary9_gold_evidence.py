from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from titan_v4.metrics.external_validation import manifest_file_fingerprint

DEFAULT_BUNDLE = ROOT / "outputs/reviewer_verification/arrhythmia_primary9/cedia_gold_2026-05-26"
DEFAULT_CHECKPOINT = ROOT / "models/gold_master/gold_master_primary9_model.pth"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _bundle_fingerprint(path: Path) -> tuple[str, int]:
    return manifest_file_fingerprint(path)


def confusion_matrix_metrics(matrix: list[list[int]]) -> dict[str, object]:
    if not matrix or any(len(row) != len(matrix) for row in matrix):
        raise ValueError("Confusion matrix must be non-empty and square")
    if any(not isinstance(value, int) or value < 0 for row in matrix for value in row):
        raise ValueError("Confusion matrix counts must be non-negative integers")

    supports = [sum(row) for row in matrix]
    predicted_supports = [sum(matrix[row][column] for row in range(len(matrix))) for column in range(len(matrix))]
    per_class: list[dict[str, float | int]] = []
    for index, support in enumerate(supports):
        true_positive = matrix[index][index]
        precision_denominator = predicted_supports[index]
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / support if support else 0.0
        f1_denominator = precision + recall
        f1 = (2 * precision * recall / f1_denominator) if f1_denominator else 0.0
        per_class.append(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )

    total = sum(supports)
    correct = sum(matrix[index][index] for index in range(len(matrix)))
    return {
        "records": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": math.fsum(float(row["f1"]) for row in per_class) / len(per_class),
        "weighted_f1": math.fsum(float(row["f1"]) * int(row["support"]) for row in per_class) / total if total else 0.0,
        "per_class": per_class,
    }


def _close(left: object, right: object) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)


def _record_key(record_id: str) -> str:
    return record_id.strip().casefold()


def _verify_file_hashes(bundle: Path) -> int:
    checksum_file = bundle / "SHA256SUMS.csv"
    rows = _read_csv(checksum_file)
    expected = {row["relative_path"]: row for row in rows}
    if len(expected) != len(rows):
        raise ValueError("Duplicate path in bundle SHA256SUMS.csv")
    actual_paths = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and path != checksum_file
    }
    if actual_paths != set(expected):
        raise ValueError("Bundle file list does not match SHA256SUMS.csv")
    for relative_path, row in expected.items():
        path = bundle / Path(relative_path)
        digest, size_bytes = _bundle_fingerprint(path)
        if digest.casefold() != row["sha256"].casefold() or size_bytes != int(row["bytes"]):
            raise ValueError(f"Bundle hash or size mismatch: {relative_path}")
    return len(rows)


def verify_bundle(bundle: Path = DEFAULT_BUNDLE, checkpoint: Path = DEFAULT_CHECKPOINT) -> dict[str, object]:
    projection = json.loads((bundle / "primary9_cedia_gold_source_projection.json").read_text(encoding="utf-8"))
    labels = _read_csv(bundle / "primary9_final_labels_672.csv")
    history = _read_csv(bundle / "primary9_label_history_672.csv")
    updates = _read_csv(bundle / "primary9_target_updates_129.csv")
    per_class_rows = _read_csv(bundle / "primary9_cedia_per_class_metrics.csv")
    prediction_dir = bundle / "fresh_inference"
    predictions = _read_csv(prediction_dir / "primary9_record_predictions.csv")
    local_report = json.loads((prediction_dir / "primary9_recomputed_report.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((prediction_dir / "run_manifest.json").read_text(encoding="utf-8"))

    endpoint = projection["endpoint"]
    classes = endpoint["classes"]
    historical = endpoint["final_external"]
    matrix_metrics = confusion_matrix_metrics(historical["confusion_matrix"])
    if len(labels) != 672 or len(history) != 672 or len(predictions) != 672:
        raise ValueError("The packaged Primary-9 evaluation must contain 672 rows")
    if len(classes) != 9 or [row["class"] for row in per_class_rows] != classes:
        raise ValueError("Primary-9 class order differs across the source report and per-class table")
    if len(updates) != endpoint["updated_label_rows"]:
        raise ValueError("Target update count does not match the source report")

    for field, calculated in (("total", matrix_metrics["records"]), ("correct", matrix_metrics["correct"])):
        if int(historical[field]) != calculated:
            raise ValueError(f"Historical {field} does not agree with the saved confusion matrix")
    for field in ("accuracy", "macro_f1", "weighted_f1"):
        if not _close(historical[field], matrix_metrics[field]):
            raise ValueError(f"Historical {field} does not agree with the saved confusion matrix")

    supports = {name: int(matrix_metrics["per_class"][index]["support"]) for index, name in enumerate(classes)}
    history_by_id = {_record_key(row["record_id"]): row for row in history}
    if len(history_by_id) != len(history):
        raise ValueError("Duplicate record ID in label history")
    labels_by_key = {
        (item["source"].casefold(), _record_key(item["record_id"])): item
        for item in labels
    }
    if len(labels_by_key) != len(labels):
        raise ValueError("Duplicate source/record key in final labels")
    final_counts: dict[str, int] = {}
    for row in history:
        final_counts[row["final_rhythm_label"]] = final_counts.get(row["final_rhythm_label"], 0) + 1
        label = labels_by_key.get((row["source"].casefold(), _record_key(row["record_id"])))
        if label is None or label["rhythm_label"] != row["final_rhythm_label"]:
            raise ValueError("Inference labels do not match the 672-row label history")
    if final_counts != supports:
        raise ValueError("Final label supports differ from the historical confusion matrix")

    update_by_id = {_record_key(row["record_id"]): row for row in updates}
    if len(update_by_id) != len(updates):
        raise ValueError("Duplicate record ID in target update audit")
    final_top1_matches = 0
    for key, update in update_by_id.items():
        history_row = history_by_id.get(key)
        if history_row is None:
            raise ValueError("Target update row is missing from the full label history")
        if update["original_label"] != history_row["original_rhythm_label"]:
            raise ValueError("Original target update label differs from the full label history")
        if update["final_label"] != history_row["final_rhythm_label"]:
            raise ValueError("Final target update label differs from the full label history")
        final_top1_matches += update["final_label"] == update["cedia_model_top1_label"]

    prediction_by_id = {_record_key(row["record_id"]): row for row in predictions}
    if len(prediction_by_id) != len(predictions):
        raise ValueError("Duplicate record ID in the local inference table")
    local_top1_matches = 0
    local_correct = 0
    for row in predictions:
        history_row = history_by_id.get(_record_key(row["record_id"]))
        if history_row is None or row["true_label"] != history_row["final_rhythm_label"]:
            raise ValueError("Local inference row truth differs from the CEDIA final label history")
        local_correct += row["true_label"] == row["predicted_label"]
    for key, update in update_by_id.items():
        local_top1_matches += prediction_by_id[key]["predicted_label"] == update["cedia_model_top1_label"]

    if int(local_report["total_records"]) != len(predictions) or int(local_report["correct_predictions"]) != local_correct:
        raise ValueError("Local inference JSON metrics do not match the saved row-level predictions")
    if not _close(local_report["accuracy"], local_correct / len(predictions)):
        raise ValueError("Local inference accuracy does not match row-level predictions")
    local_matrix_metrics = confusion_matrix_metrics(local_report["confusion_matrix"])
    for field in ("accuracy", "macro_f1", "weighted_f1"):
        if not _close(local_report[field], local_matrix_metrics[field]):
            raise ValueError(f"Local inference {field} does not agree with its confusion matrix")
    if run_manifest["model"]["sha256"].casefold() != _sha256(checkpoint).casefold():
        raise ValueError("Packaged inference checkpoint does not match the repository checkpoint")
    if run_manifest["label_file"]["sha256"].casefold() != _bundle_fingerprint(
        bundle / "primary9_final_labels_672.csv"
    )[0].casefold():
        raise ValueError("Packaged inference labels do not match the run manifest")
    for filename, expected_hash in run_manifest["outputs"].items():
        if _bundle_fingerprint(prediction_dir / filename)[0].casefold() != expected_hash.casefold():
            raise ValueError(f"Local inference output hash mismatch: {filename}")

    hashes_checked = _verify_file_hashes(bundle)
    return {
        "historical_source_result": {
            "records": matrix_metrics["records"],
            "correct": matrix_metrics["correct"],
            "accuracy": matrix_metrics["accuracy"],
            "macro_f1": matrix_metrics["macro_f1"],
            "weighted_f1": matrix_metrics["weighted_f1"],
            "recalculated_from_saved_confusion_matrix": True,
        },
        "label_update_audit": {
            "records_in_full_label_history": len(history),
            "updated_rows": len(updates),
            "final_labels_equal_cedia_model_top1": final_top1_matches,
            "local_rerun_predictions_equal_cedia_model_top1": local_top1_matches,
        },
        "fresh_local_inference": {
            "checkpoint_sha256": run_manifest["model"]["sha256"],
            "records": len(predictions),
            "correct": local_correct,
            "accuracy": local_report["accuracy"],
            "macro_f1": local_report["macro_f1"],
            "weighted_f1": local_report["weighted_f1"],
            "matches_historical_report": (
                local_correct == matrix_metrics["correct"]
                and _close(local_report["macro_f1"], matrix_metrics["macro_f1"])
            ),
        },
        "bundle_files_sha256_verified": hashes_checked,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify CEDIA historical Primary-9 aggregates and the separate local rerun.")
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()
    print(json.dumps(verify_bundle(args.bundle, args.checkpoint), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
