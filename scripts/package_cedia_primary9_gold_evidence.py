from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from titan_v4.metrics.external_validation import manifest_file_fingerprint

DEFAULT_OUTPUT = ROOT / "outputs/reviewer_verification/arrhythmia_primary9/cedia_gold_2026-05-26"
LABEL_FIELDS = ("record_id", "source", "split", "rhythm_label")
HISTORY_FIELDS = (
    "record_id",
    "source",
    "split",
    "original_rhythm_label",
    "final_rhythm_label",
    "label_action",
)
UPDATE_FIELDS = (
    "record_id",
    "source",
    "original_label",
    "cedia_model_top1_label",
    "model_top1_confidence",
    "final_label",
    "label_update_decision",
    "label_update_metric_action",
    "final_metric_action",
)
RUN_FILES = (
    "primary9_record_predictions.csv",
    "primary9_recomputed_report.json",
    "run_manifest.json",
    "source_files_sha256.csv",
)
ORIGINAL_MANIFEST_SHA256 = "8190EC19F4887D4618213CBF95B9DA3B68948DDDAB4F0233D975EBA5938E5B8C"
BUNDLE_README = """# CEDIA Gold Primary-9 Evidence

This bundle separates the archived CEDIA aggregate from a fresh inference run performed with the repository code.

## Archived CEDIA Result

The source report records two references for the same 672 records. Against original labels it reports 476/672 correct, 70.8333% accuracy, 70.6844% macro-F1, and 70.6975% weighted-F1. Against final updated labels it reports 605/672 correct, 90.0298% accuracy, 87.8192% macro-F1, and 90.0853% weighted-F1. Only the final-reference confusion matrix is saved and can be recalculated.

The original labels come from `DATASETS_CURADOS/RHYTHM_PRIMARY_V2/manifest_external_test.csv`. All 672 IDs and labels are cross-checked against `original_rhythm_label` in the sanitized history. Its raw source manifest is not redistributed because it contains demographic fields and absolute paths. The target-update audit records CEDIA Top-1 separately from the original and final labels. The source report does not record a checkpoint hash or historical row-level predictions.

## Fresh Inference

`fresh_inference/` contains predictions, probabilities, source-file hashes, a recomputed report, and a run manifest for the repository checkpoint whose SHA-256 matches the Primary-9 Gold checkpoint available in CEDIA. The same predictions score 466/672 against original labels and 568/672 against final labels. The final-reference metrics are 84.5238% accuracy, 82.3519% macro-F1, and 84.6412% weighted-F1. This is a separate inference result, not the source of the archived matrix.

Run the local verifier from the repository root:

```powershell
python scripts/verify_primary9_gold_evidence.py
python scripts/recompute_primary9_paired_reference.py
```

To run inference again in PowerShell:

```powershell
$out = Join-Path $env:TEMP "titan-primary9-cedia-final-rerun"
python scripts/run_primary9_inference.py `
  --labels outputs/reviewer_verification/arrhythmia_primary9/cedia_gold_2026-05-26/primary9_final_labels_672.csv `
  --source-manifest data/external_validation/record_source_manifest.csv `
  --checkpoint models/gold_master/gold_master_primary9_model.pth `
  --data-root data/cache/physionet/challenge-2021/1.0.3 `
  --out-dir $out --download-workers 6 --batch-size 32
```

The runner fetches the 672 records if they are not cached. The dataset release is available at https://physionet.org/content/challenge-2021/1.0.3/ (DOI 10.13026/34va-7q14; CC BY 4.0).

The CEDIA-derived tables omit age, sex, diagnosis codes, absolute paths, and free-text notes. ECG signals are not redistributed in this repository.
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _record_key(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    return PurePosixPath(normalized).name.casefold()


def _original_label_manifest_provenance(
    manifest_path: Path,
    expected_labels: dict[str, str],
    *,
    expected_sha256: str = ORIGINAL_MANIFEST_SHA256,
    expected_records: int = 672,
) -> dict[str, object]:
    rows = _read_csv(manifest_path)
    if len(rows) != expected_records:
        raise ValueError("CEDIA original-label manifest does not have the expected record count")
    required_fields = {"record_id", "rhythm_label_name"}
    if not required_fields.issubset(rows[0]):
        raise ValueError("CEDIA original-label manifest is missing required fields")
    if _sha256(manifest_path).casefold() != expected_sha256.casefold():
        raise ValueError("CEDIA original-label manifest SHA-256 differs from the verified source")

    source_labels: dict[str, str] = {}
    for row in rows:
        key = _record_key(row["record_id"])
        if not key or key in source_labels:
            raise ValueError("CEDIA original-label manifest has an empty or duplicate record ID")
        source_labels[key] = row["rhythm_label_name"].strip()
    expected = {_record_key(key): value for key, value in expected_labels.items()}
    if source_labels.keys() != expected.keys():
        raise ValueError("CEDIA original-label manifest record IDs do not match the final manifest")
    if any(source_labels[key] != expected[key] for key in source_labels):
        raise ValueError("CEDIA original labels do not match the original labels in the final manifest")

    return {
        "artifact": "CEDIA V4 Primary-9 original-label manifest, before final label updates",
        "cedia_relative_path": "DATASETS_CURADOS/RHYTHM_PRIMARY_V2/manifest_external_test.csv",
        "sha256": _sha256(manifest_path),
        "records": len(rows),
        "matched_record_ids": len(source_labels),
        "source_label_field": "rhythm_label_name",
        "matched_label_field": "original_rhythm_label",
        "raw_manifest_includes_demographics_or_absolute_paths": True,
    }


def _json_write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def write_bundle_hash_manifest(output: Path) -> int:
    checksum_rows: list[dict[str, object]] = []
    for path in sorted(item for item in output.rglob("*") if item.is_file() and item.name != "SHA256SUMS.csv"):
        digest, size_bytes = manifest_file_fingerprint(path)
        checksum_rows.append(
            {
                "relative_path": path.relative_to(output).as_posix(),
                "sha256": digest,
                "bytes": size_bytes,
            }
        )
    with (output / "SHA256SUMS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("relative_path", "sha256", "bytes"), lineterminator="\n")
        writer.writeheader()
        writer.writerows(checksum_rows)
    return len(checksum_rows)


def package(args: argparse.Namespace) -> dict[str, object]:
    report = json.loads(args.cedia_report.read_text(encoding="utf-8"))
    source_rows = _read_csv(args.cedia_manifest)
    raw_updates = _read_csv(args.cedia_updates)
    per_class_rows = _read_csv(args.per_class_metrics)
    local_predictions = _read_csv(args.inference_dir / "primary9_record_predictions.csv")
    local_report = json.loads((args.inference_dir / "primary9_recomputed_report.json").read_text(encoding="utf-8"))
    local_run = json.loads((args.inference_dir / "run_manifest.json").read_text(encoding="utf-8"))
    label_rows = _read_csv(args.labels)
    class_order = list(report["final_external"]["classification_report"])

    required_manifest = {"record_id", "source", "split", "original_rhythm_label", "final_rhythm_label", "label_action"}
    if not required_manifest.issubset(source_rows[0]):
        raise ValueError("CEDIA final manifest is missing required label fields")
    if len(source_rows) != 672 or len(label_rows) != 672 or len(local_predictions) != 672:
        raise ValueError("Primary-9 Gold evidence must contain exactly 672 records")
    if len(raw_updates) != report["updated_rows"]:
        raise ValueError("CEDIA update-row count differs from the source report")

    history_by_id: dict[str, dict[str, str]] = {}
    history_rows: list[dict[str, str]] = []
    for source in source_rows:
        safe_id = source["record_id"].strip()
        key = _record_key(safe_id)
        if not safe_id or key in history_by_id:
            raise ValueError("CEDIA final manifest contains an empty or duplicate record key")
        history = {field: source.get(field, "").strip() for field in HISTORY_FIELDS}
        history["record_id"] = safe_id
        history_by_id[key] = history
        history_rows.append(history)

    labels_by_key = {(row["source"].casefold(), _record_key(row["record_id"])): row for row in label_rows}
    if len(labels_by_key) != len(label_rows):
        raise ValueError("Final label input contains duplicate source/record keys")
    for history in history_rows:
        key = (history["source"].casefold(), _record_key(history["record_id"]))
        label = labels_by_key.get(key)
        if label is None or label["rhythm_label"] != history["final_rhythm_label"]:
            raise ValueError("Inference labels do not match the CEDIA final label manifest")

    original_manifest_provenance = _original_label_manifest_provenance(
        args.cedia_original_manifest,
        {key: row["original_rhythm_label"] for key, row in history_by_id.items()},
    )

    updates: list[dict[str, str]] = []
    updates_by_id: dict[str, dict[str, str]] = {}
    for row in raw_updates:
        key = _record_key(row.get("record_id", ""))
        history = history_by_id.get(key)
        if history is None or key in updates_by_id:
            raise ValueError("CEDIA updates cannot be uniquely joined to the final manifest")
        if row["original_external_label"] != history["original_rhythm_label"]:
            raise ValueError("CEDIA update original label disagrees with the final manifest")
        if row["final_label"] != history["final_rhythm_label"]:
            raise ValueError("CEDIA update final label disagrees with the final manifest")
        update = {
            "record_id": history["record_id"],
            "source": history["source"],
            "original_label": row["original_external_label"],
            "cedia_model_top1_label": row["model_top1_label"],
            "model_top1_confidence": row["model_top1_confidence"],
            "final_label": row["final_label"],
            "label_update_decision": row["label_update_decision"],
            "label_update_metric_action": row["label_update_metric_action"],
            "final_metric_action": row["final_metric_action"],
        }
        updates.append(update)
        updates_by_id[key] = update

    if len(updates) != report["updated_rows"]:
        raise ValueError("Not all CEDIA label updates were matched")
    top1_matches = sum(row["final_label"] == row["cedia_model_top1_label"] for row in updates)

    predictions_by_id = {
        _record_key(row["record_id"]): row
        for row in local_predictions
    }
    if len(predictions_by_id) != len(local_predictions):
        raise ValueError("Local inference predictions contain duplicate record IDs")
    local_top1_matches = 0
    for key, update in updates_by_id.items():
        prediction = predictions_by_id.get(key)
        if prediction is None or prediction["true_label"] != update["final_label"]:
            raise ValueError("Local inference rows do not align with the updated Gold records")
        local_top1_matches += prediction["predicted_label"] == update["cedia_model_top1_label"]

    if local_run["model"]["sha256"].casefold() != _sha256(args.checkpoint).casefold():
        raise ValueError("Inference manifest checkpoint hash does not match the supplied checkpoint")
    if local_run["label_file"]["sha256"].casefold() != _sha256(args.labels).casefold():
        raise ValueError("Inference manifest label hash does not match the supplied labels")
    if _sha256(args.checkpoint).casefold() != "bc7ba03d0d6d40e823fdb9bb261ebe29ae8b7a74c97d1c98e7140bca4d2d305b".casefold():
        raise ValueError("The supplied checkpoint is not the verified Primary-9 Gold checkpoint")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    run_dir = output / "fresh_inference"
    run_dir.mkdir()
    labels_output = output / "primary9_final_labels_672.csv"
    shutil.copyfile(args.labels, labels_output)
    _write_csv(output / "primary9_label_history_672.csv", HISTORY_FIELDS, history_rows)
    _write_csv(output / "primary9_target_updates_129.csv", UPDATE_FIELDS, updates)
    _write_csv(
        output / "primary9_cedia_per_class_metrics.csv",
        tuple(per_class_rows[0].keys()),
        per_class_rows,
    )
    for name in RUN_FILES:
        shutil.copyfile(args.inference_dir / name, run_dir / name)

    packaged_run_path = run_dir / "run_manifest.json"
    packaged_run = json.loads(packaged_run_path.read_text(encoding="utf-8"))
    packaged_run["label_file"]["name"] = "../primary9_final_labels_672.csv"
    packaged_run["label_file"]["sha256"] = manifest_file_fingerprint(labels_output)[0]
    packaged_run["model"]["checkpoint"] = "../../../../../models/gold_master/gold_master_primary9_model.pth"
    packaged_run["source_manifest"]["name"] = "../../../../../data/external_validation/record_source_manifest.csv"
    packaged_run["source_manifest"]["sha256"] = manifest_file_fingerprint(
        ROOT / "data/external_validation/record_source_manifest.csv"
    )[0]
    for relative_path in packaged_run["code_files"]:
        code_path = ROOT / relative_path
        if code_path.is_file():
            packaged_run["code_files"][relative_path] = manifest_file_fingerprint(code_path)[0]
    for filename in packaged_run["outputs"]:
        packaged_run["outputs"][filename] = manifest_file_fingerprint(run_dir / filename)[0]
    packaged_run["hash_semantics"] = (
        "Text-file SHA-256 and sizes use UTF-8 universal-newline normalization; "
        "binary artifacts are hashed as raw bytes."
    )
    _json_write(packaged_run_path, packaged_run)

    source_projection = {
        "schema": "TITAN_V4_CEDIA_PRIMARY9_GOLD_EVIDENCE_V1",
        "dataset": {
            "name": "PhysioNet/Computing in Cardiology Challenge 2021",
            "version": "1.0.3",
            "doi": "10.13026/34va-7q14",
            "license": "CC BY 4.0",
        },
        "source_artifacts": {
            "cedia_original_label_manifest": original_manifest_provenance,
            "cedia_report": {
                "artifact": "CEDIA V4 Gold Primary-9 source report, 2026-05-26",
                "cedia_relative_path": "03_OUTPUTS/01_ARRHYTHMIA_RHYTHM_OUTPUTS/2026-05-26_FINAL_EXTERNAL_GATE/primary9_final_external_report.json",
                "matching_evidence_copy": "04_EVIDENCIA_AUDITORIA/02_ARRITMIAS_PRIMARY9_FULL_SUPPORT/2026-05-26_FINAL_EXTERNAL_GATE/primary9_final_external_report.json",
                "sha256": _sha256(args.cedia_report),
            },
            "cedia_final_label_manifest": {
                "artifact": "CEDIA V4 final 672-record label manifest, 2026-05-26",
                "cedia_relative_path": "03_OUTPUTS/2026-05-26_GOLD_MASTER_EXTERNAL_GATE/04_EXTERNAL_VALIDATION_SET/primary9_final_external_validation_manifest.csv",
                "sha256": _sha256(args.cedia_manifest),
            },
            "cedia_label_updates": {
                "artifact": "CEDIA V4 Primary-9 target update table, 2026-05-26",
                "cedia_relative_path": "03_OUTPUTS/01_ARRHYTHMIA_RHYTHM_OUTPUTS/2026-05-26_FINAL_EXTERNAL_GATE/primary9_final_label_updates.csv",
                "sha256": _sha256(args.cedia_updates),
            },
            "cedia_per_class_metrics": {
                "artifact": "CEDIA V4 Gold Primary-9 per-class metrics, 2026-05-26",
                "cedia_relative_path": "03_OUTPUTS/01_ARRHYTHMIA_RHYTHM_OUTPUTS/2026-05-26_FINAL_EXTERNAL_GATE/primary9_final_per_class_metrics.csv",
                "sha256": _sha256(args.per_class_metrics),
            },
            "checkpoint": {
                "cedia_relative_path": "03_OUTPUTS/2026-05-26_GOLD_MASTER_EXTERNAL_GATE/01_GOLD_MASTER_PTH/gold_master_primary9_model.pth",
                "sha256": _sha256(args.checkpoint),
                "source_report_records_checkpoint_sha256": False,
            },
        },
        "endpoint": {
            "classes": class_order,
            "records": report["final_external"]["total"],
            "updated_label_rows": report["updated_rows"],
            "original_external": report["original_external"],
            "final_external": report["final_external"],
            "confusion_matrix_axis_order": "true labels by rows; predicted labels by columns",
        },
        "label_update_audit": {
            "updates_matched_to_full_manifest": len(updates),
            "original_labels_match_full_manifest": len(updates),
            "final_labels_match_full_manifest": len(updates),
            "final_labels_equal_cedia_model_top1": top1_matches,
            "local_rerun_predictions_equal_cedia_model_top1": local_top1_matches,
            "local_rerun_updated_rows": len(updates),
        },
        "label_review_crosscheck": {
            "source_update_rows": len(raw_updates),
            "rows_with_decision": sum(bool(row.get("label_update_decision", "").strip()) for row in raw_updates),
            "rows_with_timestamp": sum(bool(row.get("label_update_timestamp", "").strip()) for row in raw_updates),
            "reviewer_identity_field_present": any(
                "reviewer" in field.casefold() for field in raw_updates[0]
            ),
        },
        "fresh_inference": {
            "checkpoint_sha256": local_run["model"]["sha256"],
            "records": local_report["total_records"],
            "correct_predictions": local_report["correct_predictions"],
            "accuracy": local_report["accuracy"],
            "macro_f1": local_report["macro_f1"],
            "weighted_f1": local_report["weighted_f1"],
            "report_path": "fresh_inference/primary9_recomputed_report.json",
            "predictions_path": "fresh_inference/primary9_record_predictions.csv",
            "run_manifest_path": "fresh_inference/run_manifest.json",
        },
        "redactions": [
            "Removed age, sex, diagnosis codes, header paths, and absolute CEDIA paths.",
            "Replaced source update-table path identifiers with the matched PhysioNet record ID.",
            "Excluded free-text update notes.",
        ],
    }
    if args.cedia_p0_review_decisions:
        p0_rows = _read_csv(args.cedia_p0_review_decisions)
        p0_ids = [_record_key(row.get("record_id", "")) for row in p0_rows]
        if any(not key for key in p0_ids) or len(set(p0_ids)) != len(p0_ids):
            raise ValueError("Separate P0 review table has empty or duplicate record IDs")
        source_projection["label_review_crosscheck"]["separate_p0_review"] = {
            "cedia_relative_path": "DATASETS_CURADOS/RHYTHM_MANUAL_CURATION_QUEUE/ADJUDICATED_FIRST_PASS/p0_review_decisions.csv",
            "sha256": _sha256(args.cedia_p0_review_decisions),
            "records": len(p0_rows),
            "records_matching_gold_updates": len(set(p0_ids) & set(updates_by_id)),
        }
    _json_write(output / "primary9_cedia_gold_source_projection.json", source_projection)
    (output / "README.md").write_text(BUNDLE_README, encoding="utf-8")

    write_bundle_hash_manifest(output)

    return {
        "output_dir": str(output),
        "historical_correct": report["final_external"]["correct"],
        "historical_records": report["final_external"]["total"],
        "historical_accuracy": report["final_external"]["accuracy"],
        "historical_macro_f1": report["final_external"]["macro_f1"],
        "updates_equal_cedia_top1": top1_matches,
        "local_rerun_correct": local_report["correct_predictions"],
        "local_rerun_records": local_report["total_records"],
        "local_rerun_macro_f1": local_report["macro_f1"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Package sanitized CEDIA Primary-9 Gold source evidence and a fresh local rerun.")
    parser.add_argument("--cedia-report", type=Path, required=True)
    parser.add_argument("--cedia-manifest", type=Path, required=True)
    parser.add_argument("--cedia-original-manifest", type=Path, required=True)
    parser.add_argument("--cedia-updates", type=Path, required=True)
    parser.add_argument("--cedia-p0-review-decisions", type=Path)
    parser.add_argument("--per-class-metrics", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True, help="The exact sanitized labels file used for the fresh local rerun.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/gold_master/gold_master_primary9_model.pth")
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(package(args), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
