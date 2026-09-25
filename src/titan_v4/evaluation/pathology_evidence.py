"""Evidence readiness checks for historical Primary-5 pathology metrics."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from titan_v4.metrics.external_validation import sha256_file


PRIMARY5 = {"IMI", "ALMI", "ILMI", "LAE", "ISC_"}


def _count_primary5_labels(path: Path) -> tuple[int, int]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "pathology_label_names" not in rows[0]:
        return len(rows), 0
    labeled = 0
    for row in rows:
        names = {value.strip() for value in row.get("pathology_label_names", "").split(",") if value.strip()}
        if names & PRIMARY5:
            labeled += 1
    return len(rows), labeled


def build_audit(
    historical: dict[str, Any],
    training: dict[str, Any],
    *,
    checkpoint_path: Path,
    label_path: Path,
    prediction_path: Path,
    cedia_candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    label_rows, candidate_labeled_records = _count_primary5_labels(label_path)
    pathology_stats = training.get("pathology_pos_weight_stats") or {}
    sidecar = training.get("pathology_sidecar")
    metadata = {
        "head_present_in_model_configuration": training.get("pathology_head"),
        "classes": training.get("pathology_class_names", []),
        "loss_weight": training.get("pathology_weight"),
        "training_windows_with_pathology_labels": pathology_stats.get("labeled_windows"),
        "sidecar_included_with_release": sidecar != "not included; original local path removed",
    }
    missing = []
    if not prediction_path.is_file():
        missing.append("record-level per-class pathology scores/predictions for the historical evaluation")
    if candidate_labeled_records == 0:
        missing.append("record-level Primary-5 reference labels in the checked 672-record candidate table")
    if historical.get("checkpoint_sha256") in (None, "", "not recorded"):
        missing.append("checkpoint hash for the historical aggregate")
    if not historical.get("thresholds") and not historical.get("threshold_provenance"):
        missing.append("per-class thresholds and calibration split for the historical aggregate")

    checkpoint_sha = sha256_file(checkpoint_path)
    report = {
        "protocol": "PATHOLOGY_PRIMARY5_REPRODUCIBILITY_AUDIT_V1",
        "evidence_status": "HISTORICAL_METRIC_NOT_REPRODUCIBLE_FROM_AVAILABLE_ARTIFACTS",
        "evaluation_performed": False,
        "reason": "The checked artifacts do not provide a record-level target/prediction pair for scoring the historical claim.",
        "historical_claim": {
            "reported_per_label_accuracy": historical.get("accuracy"),
            "accuracy_definition": historical.get("accuracy_definition"),
            "reported_macro_f1": historical.get("macro_f1"),
            "label_set": historical.get("primary_pathologies"),
            "checkpoint_sha256": historical.get("checkpoint_sha256"),
        },
        "distributed_checkpoint": {
            "filename": checkpoint_path.name,
            "sha256": checkpoint_sha,
            "training_metadata": metadata,
        },
        "checked_candidate_reference_table": {
            "filename": label_path.name,
            "sha256": sha256_file(label_path),
            "record_count": label_rows,
            "records_with_any_primary5_pathology_label": candidate_labeled_records,
            "note": "This table is not asserted to be the unavailable original pathology reference table.",
        },
        "historical_record_level_prediction_file": {
            "expected_filename": prediction_path.name,
            "available": prediction_path.is_file(),
            "sha256": sha256_file(prediction_path) if prediction_path.is_file() else None,
        },
        "missing_evidence": missing,
        "interpretation": (
            "The 90.26% per-label accuracy and 66.87% macro-F1 remain unverified historical aggregates. "
            "The current checkpoint's training summary reports pathology loss weight 0 and zero pathology-labeled "
            "windows, but this metadata alone does not establish the full prior weight lineage. It is not valid "
            "to score the historical claim by inventing negative labels or reusing rhythm-only labels."
        ),
        "scope": {
            "historical_metric_reproduced": False,
            "current_checkpoint_pathology_performance_claim": False,
            "clinical_validation": False,
        },
    }
    if cedia_candidate is not None:
        cedia_model = cedia_candidate.get("model", {})
        cedia_checkpoint_sha = cedia_model.get("checkpoint_sha256")
        report["separate_cedia_candidate"] = {
            **cedia_candidate,
            "matches_distributed_checkpoint": (
                str(cedia_checkpoint_sha or "").upper() == checkpoint_sha.upper()
            ),
            "historical_primary5_claim_linked": False,
        }
    return report
