#!/usr/bin/env python3
"""Evaluate the CEDIA pathology checkpoint on patient-disjoint PTB-XL records.

Run this from a TITAN V4 audit checkout staged on CEDIA, with the original CEDIA
project, checkpoint, PTB-XL data, and Python environment available.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from titan_v4.data.label_registry_pathology import (  # noqa: E402
    NUM_PATHOLOGY_CLASSES,
    PATHOLOGY_CLASS_NAMES,
    pathology_labels_from_scp_codes,
)
from titan_v4.evaluation.cedia_pathology_cohort import (  # noqa: E402
    build_patient_disjoint_cohorts,
    normalize_ptbxl_record_id,
    ptbxl_split_record_ids,
    ptbxl_split_record_ids_by_source,
    resolve_ptbxl_low_resolution_record,
)
from titan_v4.evaluation.cedia_checkpoint import (  # noqa: E402
    inspect_cedia_checkpoint_shapes,
)
from titan_v4.evaluation.pathology_primary5_protocol import (  # noqa: E402
    PRIMARY5_PATHOLOGIES,
    evaluate_pathology_primary5,
    summarize_fixed_threshold,
)


PTBXL_103_DATABASE_SHA256 = "7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b"
PTBXL_103_SCP_SHA256 = "ad05b0b1fcae83bb1230755ad9cfc7c96f303feddc08a4a9ad5bdc9ca63bac8f"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_artifact_sha256_manifest(output_dir: Path) -> Path:
    manifest_path = output_dir / "SHA256SUMS.csv"
    artifacts = sorted(
        path
        for path in output_dir.iterdir()
        if path.is_file() and path != manifest_path
    )
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("relative_path", "sha256", "size_bytes"),
            lineterminator="\n",
        )
        writer.writeheader()
        for artifact in artifacts:
            writer.writerow(
                {
                    "relative_path": artifact.name,
                    "sha256": sha256_file(artifact),
                    "size_bytes": artifact.stat().st_size,
                }
            )
    return manifest_path


def _record_id_from_database_row(row: dict[str, str]) -> str:
    return f"{int(row['ecg_id']):05d}"


def _pathology_sidecar(path: Path) -> tuple[dict[str, tuple[int, ...]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if tuple(payload.get("class_names", ())) != tuple(PATHOLOGY_CLASS_NAMES):
        raise ValueError("CEDIA sidecar class order does not match the repository pathology registry")
    records: dict[str, tuple[int, ...]] = {}
    for record_path, item in payload.get("records", {}).items():
        labels = tuple(sorted(int(index) for index in item.get("labels", ())))
        if not labels:
            continue
        record_id = normalize_ptbxl_record_id(record_path)
        if record_id in records and records[record_id] != labels:
            raise ValueError(f"Conflicting sidecar labels for PTB-XL record {record_id}")
        records[record_id] = labels
    return records, payload


def _load_ptbxl_metadata(
    database_csv: Path,
) -> tuple[dict[str, tuple[int, ...]], dict[str, str], dict[str, str]]:
    labels_by_record: dict[str, tuple[int, ...]] = {}
    patient_by_record: dict[str, str] = {}
    filename_lr_by_record: dict[str, str] = {}
    with database_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"ecg_id", "patient_id", "scp_codes", "filename_lr"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"PTB-XL metadata is missing columns: {sorted(required - set(reader.fieldnames or ()))}")
        for row in reader:
            record_id = _record_id_from_database_row(row)
            if record_id in filename_lr_by_record:
                raise ValueError(f"Duplicate PTB-XL metadata record ID: {record_id}")
            patient_id = str(int(float(row["patient_id"])))
            patient_by_record[record_id] = patient_id
            filename_lr_by_record[record_id] = row["filename_lr"]
            try:
                scp_codes = ast.literal_eval(row.get("scp_codes") or "{}")
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"Invalid PTB-XL SCP metadata for record {record_id}") from exc
            if not isinstance(scp_codes, dict):
                raise ValueError(f"PTB-XL SCP metadata is not a mapping for record {record_id}")
            labels = tuple(pathology_labels_from_scp_codes(scp_codes))
            if labels:
                labels_by_record[record_id] = labels
    return labels_by_record, patient_by_record, filename_lr_by_record


def _available_records(
    record_root: Path,
    filename_lr_by_record: dict[str, str],
) -> dict[str, Path]:
    records: dict[str, Path] = {}
    for record_id, filename_lr in filename_lr_by_record.items():
        try:
            record_base = resolve_ptbxl_low_resolution_record(record_root, filename_lr)
        except FileNotFoundError:
            continue
        resolved_id = normalize_ptbxl_record_id(record_base.name)
        if resolved_id != record_id:
            raise ValueError(
                f"PTB-XL filename_lr ID does not match metadata ecg_id: {record_id} != {resolved_id}"
            )
        if record_id in records:
            raise ValueError(f"Duplicate PTB-XL low-resolution record ID: {record_id}")
        records[record_id] = record_base
    if not records:
        raise FileNotFoundError(f"No WFDB headers found under {record_root}")
    return records


def _ptbxl_split_ids(paths: list[str]) -> set[str]:
    return ptbxl_split_record_ids(paths)


def _write_scores_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["record_id", "split"]
    for name in PATHOLOGY_CLASS_NAMES:
        fieldnames.extend((f"true_{name}", f"score_{name}"))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_input_manifest(
    path: Path,
    record_ids: tuple[str, ...],
    split_by_record: dict[str, str],
    record_bases: dict[str, Path],
    read_failures: dict[str, dict[str, str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["record_id", "split", "status", "error_type", "header_sha256", "signal_sha256"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for record_id in record_ids:
            record_base = record_bases[record_id]
            header = record_base.with_suffix(".hea")
            signal = record_base.with_suffix(".dat")
            writer.writerow(
                {
                    "record_id": record_id,
                    "split": split_by_record[record_id],
                    "status": "excluded_input_read_error" if record_id in read_failures else "evaluated",
                    "error_type": read_failures.get(record_id, {}).get("error_type", ""),
                    "header_sha256": sha256_file(header) if header.is_file() else "",
                    "signal_sha256": sha256_file(signal) if signal.is_file() else "",
                }
            )


def _preflight_windows(
    record_ids: tuple[str, ...],
    *,
    record_bases: dict[str, Path],
    evaluator: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    windows: dict[str, Any] = {}
    failures: dict[str, dict[str, str]] = {}
    for index, record_id in enumerate(record_ids, start=1):
        try:
            windows[record_id] = evaluator.read_first_window(str(record_bases[record_id])).to("cpu")
        except Exception as exc:
            failures[record_id] = {"error_type": type(exc).__name__}
            print(f"Excluded unreadable PTB-XL record {record_id}: {type(exc).__name__}", flush=True)
        if index % 100 == 0 or index == len(record_ids):
            print(f"Preflighted {index}/{len(record_ids)} records", flush=True)
    return windows, failures


def _predict(
    record_ids: tuple[str, ...],
    *,
    windows_by_record: dict[str, Any],
    labels_by_record: dict[str, tuple[int, ...]],
    split_by_record: dict[str, str],
    model: Any,
    batch_size: int,
    num_threads: int,
) -> list[dict[str, Any]]:
    import numpy as np
    import torch

    torch.set_num_threads(num_threads)
    output_rows: list[dict[str, Any]] = []
    for start in range(0, len(record_ids), batch_size):
        batch_ids = record_ids[start : start + batch_size]
        windows = [windows_by_record[record_id] for record_id in batch_ids]
        signals = torch.cat(windows, dim=0)
        with torch.inference_mode():
            output = model(signals)
        if not isinstance(output, (tuple, list)) or len(output) < 4:
            raise ValueError("CEDIA model output does not contain the pathology head at index 3")
        logits = output[3]
        if logits.ndim != 2 or logits.shape != (len(batch_ids), NUM_PATHOLOGY_CLASSES):
            raise ValueError(f"Unexpected pathology logit shape: {tuple(logits.shape)}")
        probabilities = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
        for row_index, record_id in enumerate(batch_ids):
            positive = set(labels_by_record[record_id])
            row: dict[str, Any] = {
                "record_id": record_id,
                "split": split_by_record[record_id],
            }
            for class_index, class_name in enumerate(PATHOLOGY_CLASS_NAMES):
                row[f"true_{class_name}"] = int(class_index in positive)
                row[f"score_{class_name}"] = float(probabilities[row_index, class_index])
            output_rows.append(row)
        print(f"Evaluated {min(start + len(batch_ids), len(record_ids))}/{len(record_ids)} records", flush=True)
    return output_rows


def _load_cedia_evaluator(code_dir: Path) -> Any:
    code_dir = code_dir.resolve()
    source_dir = code_dir.parent
    for directory in (source_dir, code_dir):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    import evaluate_holdout_14_classes

    return evaluate_holdout_14_classes


def _load_cedia_checkpoint(
    weights: Path,
    code_dir: Path,
    training_summary: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    import torch

    source_dir = code_dir.parent.resolve()
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    from titan_v4_net import TitanV4Max

    try:
        payload = torch.load(str(weights), map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(str(weights), map_location="cpu")
    state_dict = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    if not isinstance(state_dict, dict):
        raise ValueError("CEDIA checkpoint does not contain a model state dictionary")
    state_dict = {str(key).replace("module.", ""): value for key, value in state_dict.items()}
    spec = inspect_cedia_checkpoint_shapes(state_dict)
    if spec.pathology_classes != int(training_summary.get("pathology_num_classes", -1)):
        raise ValueError("Checkpoint pathology head size disagrees with its training summary")
    if spec.pathology_classes != NUM_PATHOLOGY_CLASSES:
        raise ValueError("Checkpoint pathology head size disagrees with the frozen label registry")
    if int(training_summary.get("morphology_dim", 0)) != spec.morphology_dim:
        raise ValueError("Checkpoint morphology input dimension disagrees with its training summary")
    model = TitanV4Max(
        in_channels=6,
        num_rhythm=spec.rhythm_classes,
        num_pathology=spec.pathology_classes,
        morphology_dim=spec.morphology_dim,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to("cpu")
    model.eval()
    return model, {
        "architecture": spec.architecture,
        "encoder_dim": spec.encoder_dim,
        "rhythm_classes": spec.rhythm_classes,
        "pathology_classes": spec.pathology_classes,
        "morphology_dim": spec.morphology_dim,
        "state_dict_strict_load": True,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    project_root = args.project_root.resolve()
    ptbxl_root = project_root / "DATA" / "ptb-xl"
    database_csv = ptbxl_root / "ptbxl_database.csv"
    scp_csv = ptbxl_root / "scp_statements.csv"
    sidecar_json = ptbxl_root / "ptbxl_pathology_label_map.json"
    record_root = ptbxl_root / "records100"
    weights = args.weights.resolve()
    training_summary_path = args.training_summary.resolve()
    split_metadata_path = args.split_metadata.resolve()
    code_dir = args.cedia_code_dir.resolve()

    database_hash = sha256_file(database_csv)
    scp_hash = sha256_file(scp_csv)
    if database_hash != PTBXL_103_DATABASE_SHA256 or scp_hash != PTBXL_103_SCP_SHA256:
        raise ValueError("PTB-XL database files do not match the pinned v1.0.3 checksums")

    training_summary = json.loads(training_summary_path.read_text(encoding="utf-8"))
    class_names = tuple(training_summary.get("pathology_class_names", ()))
    if class_names != tuple(PATHOLOGY_CLASS_NAMES):
        raise ValueError("Checkpoint training summary class order differs from this evaluation registry")
    if int(training_summary.get("pathology_num_classes", -1)) != NUM_PATHOLOGY_CLASSES:
        raise ValueError("Checkpoint training summary pathology head size is not 10")

    labels_by_record, patient_by_record, filename_lr_by_record = _load_ptbxl_metadata(database_csv)
    record_bases = _available_records(record_root, filename_lr_by_record)
    labels_by_record = {
        record_id: labels
        for record_id, labels in labels_by_record.items()
        if record_id in record_bases
    }
    sidecar_labels, sidecar_payload = _pathology_sidecar(sidecar_json)
    mismatches = [
        record_id
        for record_id, labels in sidecar_labels.items()
        if labels_by_record.get(record_id) != labels
    ]
    if mismatches:
        raise ValueError(f"PTB-XL SCP label reconstruction disagrees with the training sidecar for {len(mismatches)} records")

    split_metadata = json.loads(split_metadata_path.read_text(encoding="utf-8"))
    train_paths = split_metadata.get("train_record_paths", [])
    validation_paths = split_metadata.get("val_record_paths", [])
    train_ids_by_source = ptbxl_split_record_ids_by_source(train_paths)
    validation_ids_by_source = ptbxl_split_record_ids_by_source(validation_paths)
    train_ids = set().union(*train_ids_by_source.values())
    validation_ids = set().union(*validation_ids_by_source.values())
    candidate_cohorts = build_patient_disjoint_cohorts(
        labels_by_record,
        train_record_ids=train_ids,
        validation_record_ids=validation_ids,
        patient_by_record=patient_by_record,
        num_classes=NUM_PATHOLOGY_CLASSES,
    )
    for record_id in (*candidate_cohorts.calibration_record_ids, *candidate_cohorts.test_record_ids):
        if record_id not in record_bases:
            raise FileNotFoundError(f"Missing PTB-XL signal for selected record {record_id}")

    candidate_split_by_record = {
        **{record_id: "calibration" for record_id in candidate_cohorts.calibration_record_ids},
        **{record_id: "test" for record_id in candidate_cohorts.test_record_ids},
    }
    evaluator = _load_cedia_evaluator(code_dir)
    candidate_ids = tuple(sorted(candidate_split_by_record))
    windows_by_record, read_failures = _preflight_windows(
        candidate_ids,
        record_bases=record_bases,
        evaluator=evaluator,
    )
    available_labels = {
        record_id: labels
        for record_id, labels in labels_by_record.items()
        if record_id not in read_failures
    }
    cohorts = build_patient_disjoint_cohorts(
        available_labels,
        train_record_ids=train_ids,
        validation_record_ids=validation_ids,
        patient_by_record=patient_by_record,
        num_classes=NUM_PATHOLOGY_CLASSES,
    )
    split_by_record = {
        **{record_id: "calibration" for record_id in cohorts.calibration_record_ids},
        **{record_id: "test" for record_id in cohorts.test_record_ids},
    }
    evaluation_ids = tuple(sorted(split_by_record))
    model, architecture = _load_cedia_checkpoint(weights, code_dir, training_summary)
    rows = _predict(
        evaluation_ids,
        windows_by_record=windows_by_record,
        labels_by_record=labels_by_record,
        split_by_record=split_by_record,
        model=model,
        batch_size=args.batch_size,
        num_threads=args.num_threads,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = output_dir / "pathology_primary5_scores.csv"
    input_manifest_path = output_dir / "pathology_input_records.csv"
    report_path = output_dir / "pathology_primary5_report.json"
    _write_scores_csv(scores_path, rows)
    _write_input_manifest(
        input_manifest_path,
        candidate_ids,
        candidate_split_by_record,
        record_bases,
        read_failures,
    )

    report = evaluate_pathology_primary5(rows, out_dir=output_dir)
    report["fixed_threshold_sensitivities"] = {
        str(threshold): summarize_fixed_threshold(rows, threshold=threshold)
        for threshold in (0.50, 0.65)
    }
    report["evidence_status"] = "CEDIA_CHECKPOINT_EVALUATED_ON_PATIENT_DISJOINT_PTBXL_HOLDOUT"
    report["dataset"] = {
        "name": "PTB-XL",
        "version": "1.0.3",
        "doi": "10.13026/kfzx-aw45",
        "metadata_sha256_verified": database_hash,
        "scp_statements_sha256_verified": scp_hash,
        "available_100hz_records": len(record_bases),
        "label_mapping": "PTB-XL scp_codes mapped by the repository's frozen 10-class pathology registry",
        "training_sidecar_reconstructed_records": len(sidecar_labels),
        "training_sidecar_exact_label_matches": len(sidecar_labels) - len(mismatches),
    }
    report["cohort_protocol"] = {
        "checkpoint_training_records_with_ptbxl_paths": len(train_ids),
        "checkpoint_validation_records_with_ptbxl_paths": len(validation_ids),
        "checkpoint_training_ids_by_path_type": {
            source: len(ids) for source, ids in train_ids_by_source.items()
        },
        "checkpoint_validation_ids_by_path_type": {
            source: len(ids) for source, ids in validation_ids_by_source.items()
        },
        "checkpoint_training_pathology_records_in_sidecar": len(set(sidecar_labels) & train_ids),
        "checkpoint_training_pathology_labeled_windows_from_summary": training_summary.get(
            "pathology_pos_weight_stats", {}
        ).get("labeled_windows"),
        "calibration_records": len(cohorts.calibration_record_ids),
        "calibration_patients": cohorts.calibration_patient_count,
        "test_records": len(cohorts.test_record_ids),
        "test_patients": cohorts.test_patient_count,
        "candidate_calibration_records_before_input_preflight": len(candidate_cohorts.calibration_record_ids),
        "candidate_test_records_before_input_preflight": len(candidate_cohorts.test_record_ids),
        "excluded_unreadable_inputs": {
            "total": len(read_failures),
            "by_split": {
                split: sum(candidate_split_by_record[record_id] == split for record_id in read_failures)
                for split in ("calibration", "test")
            },
            "by_error_type": {
                error_type: sum(item["error_type"] == error_type for item in read_failures.values())
                for error_type in sorted({item["error_type"] for item in read_failures.values()})
            },
        },
        "excluded_records": cohorts.excluded_counts,
        "calibration_patient_disjoint_from_training": True,
        "test_patient_disjoint_from_training_and_validation": True,
        "challenge_hr_alias_resolution": "PTB-XL Challenge HR##### suffix maps to the matching PTB-XL ecg_id; alias IDs are included in the patient-overlap exclusion.",
        "threshold_selection": "per-label F1 optimization on calibration records only; fixed before test evaluation",
        "fixed_thresholds_are_sensitivity_checks": [0.50, 0.65],
        "model_selection_metric": training_summary.get("checkpoint_metric"),
        "selected_epoch": training_summary.get("best_epoch"),
    }
    report["checkpoint"] = {
        "sha256": sha256_file(weights),
        "training_summary_sha256": sha256_file(training_summary_path),
        "pathology_loss_weight": training_summary.get("pathology_weight"),
        **architecture,
    }
    report["input_sha256"] = {
        "split_metadata": sha256_file(split_metadata_path),
        "training_label_sidecar": sha256_file(sidecar_json),
        "cedia_evaluator": sha256_file(code_dir / "evaluate_holdout_14_classes.py"),
        "cedia_preprocessing": sha256_file(code_dir.parent / "data_loader_ultra.py"),
        "cedia_model_definition": sha256_file(code_dir.parent / "titan_v4_net.py"),
        "cedia_training_utilities": sha256_file(code_dir.parent / "training_utils.py"),
        "cedia_training_script": sha256_file(code_dir.parent / "train_v4_lite.py"),
        "public_label_registry": sha256_file(REPOSITORY_ROOT / "src" / "titan_v4" / "data" / "label_registry_pathology.py"),
        "public_cohort_protocol": sha256_file(REPOSITORY_ROOT / "src" / "titan_v4" / "evaluation" / "cedia_pathology_cohort.py"),
        "public_checkpoint_loader": sha256_file(REPOSITORY_ROOT / "src" / "titan_v4" / "evaluation" / "cedia_checkpoint.py"),
        "public_metric_protocol": sha256_file(REPOSITORY_ROOT / "src" / "titan_v4" / "evaluation" / "pathology_primary5_protocol.py"),
        "evaluation_script": sha256_file(Path(__file__).resolve()),
    }
    report["runtime"] = {
        "python": platform.python_version(),
        "numpy": importlib.metadata.version("numpy"),
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "scipy": importlib.metadata.version("scipy"),
        "torch": importlib.metadata.version("torch"),
        "wfdb": importlib.metadata.version("wfdb"),
        "device": "cpu",
        "inference_batch_size": args.batch_size,
        "torch_threads": args.num_threads,
    }
    report["historical_reference"] = {
        "reported_per_label_accuracy": 0.90256,
        "reported_macro_f1": 0.66874,
        "comparable_as_reproduction": False,
        "note": "Historical checkpoint, exact cohort, and threshold lineage are not recorded; this is a separate, newly evaluated CEDIA checkpoint.",
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_manifest_path = _write_artifact_sha256_manifest(output_dir)
    print(json.dumps({
        "report": str(report_path),
        "artifact_hash_manifest": str(output_manifest_path),
        "calibration_records": len(cohorts.calibration_record_ids),
        "test_records": len(cohorts.test_record_ids),
        "per_label_accuracy_calibrated": report["per_label_accuracy"],
        "macro_f1_calibrated": report["macro_f1"],
        "per_label_accuracy_at_0.50": report["fixed_threshold_sensitivities"]["0.5"]["per_label_accuracy"],
        "macro_f1_at_0.50": report["fixed_threshold_sensitivities"]["0.5"]["macro_f1"],
        "per_label_accuracy_at_0.65": report["fixed_threshold_sensitivities"]["0.65"]["per_label_accuracy"],
        "macro_f1_at_0.65": report["fixed_threshold_sensitivities"]["0.65"]["macro_f1"],
    }, indent=2), flush=True)
    return report


def _parse_args() -> argparse.Namespace:
    default_project = Path.home() / "V4_CEDIA"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_project)
    parser.add_argument("--weights", type=Path, default=default_project / "03_OUTPUTS" / "titan_v4_lite_weights_best.pth")
    parser.add_argument("--training-summary", type=Path, default=default_project / "03_OUTPUTS" / "training_summary.json")
    parser.add_argument("--split-metadata", type=Path, default=default_project / "03_OUTPUTS" / "split_metadata.json")
    parser.add_argument("--cedia-code-dir", type=Path, default=default_project / "01_CODIGO_FUENTE" / "AUDITORIA")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--num-threads", type=int, default=8)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_threads < 1:
        parser.error("--batch-size and --num-threads must be positive")
    return args


if __name__ == "__main__":
    evaluate(_parse_args())
