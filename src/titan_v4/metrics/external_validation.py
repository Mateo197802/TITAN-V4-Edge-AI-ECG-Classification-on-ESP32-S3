"""External validation utilities for the TITAN V4 evidence package."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


PRIMARY9_CLASSES = ("AFIB", "SB", "STACH", "NSR", "PVC", "RBBB", "LBBB", "PAC", "1AVB")
PATHOLOGY5_CLASSES = ("IMI", "ALMI", "ILMI", "LAE", "ISC_")


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(str(source))
    return json.loads(source.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def primary9_summary(report: dict[str, Any]) -> dict[str, Any]:
    required = {"total_records", "correct_predictions", "accuracy", "macro_f1", "weighted_f1"}
    missing = sorted(required - set(report))
    if missing:
        raise ValueError(f"Primary-9 report missing fields: {missing}")
    if int(report["total_records"]) != 672:
        raise ValueError("Primary-9 external validation must evaluate exactly 672 records")
    return {
        "module": "arrhythmia_primary9",
        "records": int(report["total_records"]),
        "correct_predictions": int(report["correct_predictions"]),
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro_f1"]),
        "weighted_f1": float(report["weighted_f1"]),
        "classes": list(PRIMARY9_CLASSES),
    }


def pathology_summary(report: dict[str, Any]) -> dict[str, Any]:
    required = {"accuracy", "macro_f1", "primary_pathologies"}
    missing = sorted(required - set(report))
    if missing:
        raise ValueError(f"Pathology report missing fields: {missing}")
    classes = tuple(str(value) for value in report["primary_pathologies"])
    if classes != PATHOLOGY5_CLASSES:
        raise ValueError(f"Unexpected Primary-5 pathology class order: {classes}")
    return {
        "module": "pathology_primary5",
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro_f1"]),
        "classes": list(PATHOLOGY5_CLASSES),
    }


def cascade_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "module": "cascade_safety_annex",
        "role": "safety_annex",
        "diagnostic_subset_records": int(report["diagnostic_subset_records"]),
        "coverage": float(report["coverage"]),
        "quarantine_rate": float(report["quarantine_rate"]),
        "accepted_predictions": int(report["action_counts"]["accepted_prediction"]),
    }


def combined_summary(primary9: dict[str, Any], pathology: dict[str, Any], cascade: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": "COMBINED_EXTERNAL_VALIDATION_SUMMARY_V1",
        "label_set": "final external validation label set",
        "external_training_allowed": False,
        "external_threshold_tuning_allowed": False,
        "arrhythmia_primary9": primary9,
        "pathology_primary5": pathology,
        "cascade_annex": cascade,
    }


def verify_hash_manifest(root: str | Path, manifest_csv: str | Path) -> list[dict[str, str]]:
    root_path = Path(root)
    manifest_path = Path(manifest_csv)
    failures: list[dict[str, str]] = []
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            relative_path = row["relative_path"]
            expected = row["sha256"].upper()
            target = root_path / relative_path
            if not target.is_file():
                failures.append({"relative_path": relative_path, "error": "missing_file"})
                continue
            actual = sha256_file(target)
            if actual != expected:
                failures.append({"relative_path": relative_path, "expected": expected, "actual": actual})
    return failures

