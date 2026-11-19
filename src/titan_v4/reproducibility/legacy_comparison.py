"""Evidence comparison for current predictions and historical label tables."""

from __future__ import annotations

from typing import Iterable

from titan_v4.reproducibility.inference import primary9_metrics


def _record_key(row: dict[str, str], source_field: str, id_field: str) -> tuple[str, str]:
    source = row.get(source_field, "").strip().casefold()
    record_id = row.get(id_field, "").strip().upper()
    if not source or not record_id:
        raise ValueError(f"Missing source or record ID in row: {row}")
    return source, record_id


def _index_rows(
    rows: Iterable[dict[str, str]], *, source_field: str, id_field: str, label: str
) -> dict[tuple[str, str], dict[str, str]]:
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = _record_key(row, source_field, id_field)
        if key in indexed:
            raise ValueError(f"Duplicate record in {label}: {key}")
        indexed[key] = row
    return indexed


def build_comparison(
    prediction_rows: list[dict[str, str]],
    current_label_rows: list[dict[str, str]],
    legacy_label_rows: list[dict[str, str]],
    *,
    historical_report: dict[str, object],
    expected_records: int | None = 672,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    predictions = _index_rows(
        prediction_rows, source_field="source_label", id_field="record_id", label="predictions"
    )
    current = _index_rows(
        current_label_rows, source_field="source", id_field="record_id", label="current labels"
    )
    legacy = _index_rows(
        legacy_label_rows, source_field="source", id_field="record_id", label="legacy labels"
    )
    if set(predictions) != set(current) or set(predictions) != set(legacy):
        missing_current = sorted(set(predictions) - set(current))
        extra_current = sorted(set(current) - set(predictions))
        missing_legacy = sorted(set(predictions) - set(legacy))
        extra_legacy = sorted(set(legacy) - set(predictions))
        raise ValueError(
            "Prediction/current/legacy record keys differ: "
            f"current_missing={missing_current[:5]}, current_extra={extra_current[:5]}, "
            f"legacy_missing={missing_legacy[:5]}, legacy_extra={extra_legacy[:5]}"
        )
    if expected_records is not None and len(predictions) != expected_records:
        raise ValueError(f"Expected {expected_records} records, found {len(predictions)}")
    if int(historical_report.get("total_records", -1)) != len(predictions):
        raise ValueError("Historical aggregate support does not match the aligned prediction records")
    historical_correct = int(historical_report["correct_predictions"])

    current_truth: list[str] = []
    legacy_truth: list[str] = []
    predicted: list[str] = []
    comparison_rows: list[dict[str, object]] = []
    changed_labels: list[dict[str, str]] = []
    for key in sorted(predictions):
        prediction = predictions[key]
        current_label = current[key]["rhythm_label"].strip()
        legacy_label = legacy[key]["rhythm_label"].strip()
        predicted_label = prediction["predicted_label"].strip()
        if prediction.get("true_label", "").strip() != current_label:
            raise ValueError(f"Prediction artifact's stored truth differs from current labels for {key}")
        current_truth.append(current_label)
        legacy_truth.append(legacy_label)
        predicted.append(predicted_label)
        if current_label != legacy_label:
            changed_labels.append(
                {"source_label": prediction["source_label"], "record_id": prediction["record_id"]}
            )
        row: dict[str, object] = {
            "record_id": prediction["record_id"],
            "source_label": prediction["source_label"],
            "source_dataset": prediction.get("source_dataset", ""),
            "source_path": prediction.get("source_path", ""),
            "rebuilt_label": current_label,
            "historical_label": legacy_label,
            "current_checkpoint_prediction": predicted_label,
            "correct_on_rebuilt_label": predicted_label == current_label,
            "correct_on_historical_label": predicted_label == legacy_label,
        }
        row.update({name: score for name, score in prediction.items() if name.startswith("prob_")})
        comparison_rows.append(row)

    historical_correct_now = sum(row["correct_on_historical_label"] is True for row in comparison_rows)
    report = {
        "protocol": "PRIMARY9_CURRENT_CHECKPOINT_VS_HISTORICAL_LABELS_V1",
        "evidence_status": "CURRENT_CHECKPOINT_SCORED_ON_BOTH_LABEL_TABLES; HISTORICAL_605_RESULT_NOT_REPRODUCED",
        "record_count": len(predictions),
        "current_checkpoint_vs_rebuilt_labels": primary9_metrics(current_truth, predicted),
        "current_checkpoint_vs_historical_labels": primary9_metrics(legacy_truth, predicted),
        "historical_stored_aggregate": {
            "correct_predictions": historical_correct,
            "accuracy": float(historical_report["accuracy"]),
            "macro_f1": float(historical_report["macro_f1"]),
            "weighted_f1": float(historical_report["weighted_f1"]),
            "original_record_level_predictions_available": False,
            "exact_checkpoint_and_evaluation_lineage_available": False,
        },
        "label_table_comparison": {
            "different_label_count": len(changed_labels),
            "different_label_record_ids": changed_labels,
            "current_checkpoint_correct_on_historical_labels": historical_correct_now,
            "historical_aggregate_correct_prediction_gap": historical_correct_now - historical_correct,
        },
        "interpretation": (
            "This recomputes the distributed checkpoint's predictions against the exact historical label table. "
            "It does not recreate the historical aggregate because its original predictions and evaluation lineage "
            "are unavailable. The result neither proves nor disproves that an earlier, different model and protocol "
            "produced that aggregate."
        ),
        "scope": {
            "official_challenge_metric": False,
            "independent_external_validation_claim": False,
            "checkpoint_training_overlap": "unknown",
            "upstream_partition": "PhysioNet Challenge 2021 v1.0.3 training",
        },
    }
    return report, comparison_rows
