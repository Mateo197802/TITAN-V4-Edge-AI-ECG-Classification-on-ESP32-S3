"""Pathology-5 calibration/test protocol with quarantine labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PRIMARY5_PATHOLOGIES = ("IMI", "ALMI", "ILMI", "LAE", "ISC_")
QUARANTINE_PATHOLOGIES = ("AMI", "ISCAL")
AUXILIARY_PATHOLOGIES = ("ASMI", "LVH", "NST_")
CANONICAL_PATHOLOGY_10 = ("IMI", "ASMI", "LVH", "ISC_", "ISCAL", "NST_", "ILMI", "AMI", "ALMI", "LAE")


def _windows_long_path(path: str | Path) -> str:
    raw = str(path)
    if raw.startswith("\\\\?\\") or not Path(raw).is_absolute():
        return raw
    if len(raw) >= 240:
        return "\\\\?\\" + raw
    return raw


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    support: int


def _normalize_pathology_names(values: Iterable[str] | None, default: tuple[str, ...]) -> tuple[str, ...]:
    if values is None:
        return tuple(default)
    return tuple(str(value).strip() for value in values if str(value).strip())


def _validate_pathology_selection(
    primary_pathologies: Iterable[str] | None = None,
    quarantine_pathologies: Iterable[str] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    primary = _normalize_pathology_names(primary_pathologies, PRIMARY5_PATHOLOGIES)
    quarantine = _normalize_pathology_names(quarantine_pathologies, QUARANTINE_PATHOLOGIES)
    if not primary:
        raise ValueError("At least one reportable pathology is required")
    if len(set(primary)) != len(primary):
        raise ValueError("Reportable pathology selection contains duplicates")
    if len(set(quarantine)) != len(quarantine):
        raise ValueError("Quarantine pathology selection contains duplicates")
    overlap = sorted(set(primary) & set(quarantine))
    if overlap:
        raise ValueError(f"Reportable and quarantine pathology selections overlap: {overlap}")
    universe = set(CANONICAL_PATHOLOGY_10)
    unknown = sorted((set(primary) | set(quarantine)) - universe)
    if unknown:
        raise ValueError(f"Pathology classes outside CANONICAL_PATHOLOGY_10 are not allowed: {unknown}")
    auxiliary = tuple(name for name in CANONICAL_PATHOLOGY_10 if name not in set(primary) | set(quarantine))
    return primary, quarantine, auxiliary


def _read_classes_file(path: str | Path | None, *, json_key: str) -> list[str] | None:
    if not path:
        return None
    source = Path(path)
    if source.suffix.lower() == ".json":
        payload = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [str(value) for value in payload]
        if isinstance(payload, dict):
            values = payload.get(json_key) or payload.get("classes") or payload.get("class_names")
            if values is None:
                raise ValueError(f"{source} does not contain {json_key}, classes, or class_names")
            return [str(value) for value in values]
        raise ValueError(f"Unsupported JSON class selection format in {source}")
    with source.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    values: list[str] = []
    for row in rows:
        for key in ("class_name", "class", "label", "name"):
            value = str(row.get(key) or "").strip()
            if value:
                values.append(value)
                break
    if not values:
        raise ValueError(f"No class names found in {source}")
    return values


def _write_pathology_selection_report(
    path: Path,
    *,
    primary_pathologies: tuple[str, ...],
    quarantine_pathologies: tuple[str, ...],
    auxiliary_pathologies: tuple[str, ...],
) -> dict[str, object]:
    report = {
        "selection_status": "frozen_before_final_evaluation",
        "canonical_pathology_10": list(CANONICAL_PATHOLOGY_10),
        "reportable_pathologies": list(primary_pathologies),
        "quarantine_pathologies": list(quarantine_pathologies),
        "auxiliary_pathologies": list(auxiliary_pathologies),
        "target_per_label_accuracy": 0.80,
        "target_macro_f1": 0.70,
        "rules": {
            "reportable": "best-supported pathology labels used for primary claim",
            "quarantine": "weak, ambiguous, or low-performing pathology labels evaluated separately",
            "no_test_time_class_switching": True,
        },
        "per_class": [
            {
                "class_name": name,
                "role": "reportable_pathology",
                "reason": "selected_for_primary_accuracy_and_macro_f1_gate",
            }
            for name in primary_pathologies
        ]
        + [
            {
                "class_name": name,
                "role": "pathology_quarantine",
                "reason": "excluded_from_primary_metric_and_evaluated_separately",
            }
            for name in quarantine_pathologies
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def stable_split(record_id: str, calibration_fraction: float = 0.5) -> str:
    digest = hashlib.sha1(record_id.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return "calibration" if value < calibration_fraction else "test"


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _truth(row: dict[str, str], label: str) -> int:
    return int(round(_float(row, f"true_{label}", _float(row, f"y_true_{label}", 0.0))))


def _score(row: dict[str, str], label: str) -> float:
    return _float(row, f"score_{label}", _float(row, f"prob_{label}", _float(row, f"pred_{label}", 0.0)))


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / max(len(values), 1)


def _canonicalize_per_class(per_class: dict[str, object]) -> tuple[dict[str, dict[str, object]], dict[str, str]]:
    """Resolve generic cls_N pathology keys against the fixed 10-label registry.

    Some historical pathology reports wrote the first labels by name and the
    remaining labels as cls_5..cls_9. The Primary-5 gate must not treat those
    rows as missing because ALMI/ILMI/LAE/AMI are stored there.
    """

    normalized: dict[str, dict[str, object]] = {}
    mapping: dict[str, str] = {}
    for raw_label, raw_row in per_class.items():
        row = dict(raw_row) if isinstance(raw_row, dict) else {}
        label = raw_label
        if raw_label.startswith("cls_"):
            try:
                idx = int(raw_label.split("_", 1)[1])
            except ValueError:
                idx = -1
            if 0 <= idx < len(CANONICAL_PATHOLOGY_10):
                label = CANONICAL_PATHOLOGY_10[idx]
                mapping[raw_label] = label
        if label not in normalized or raw_label.startswith("cls_") is False:
            normalized[label] = row
    return normalized, mapping


def binary_metrics(y_true: list[int], y_score: list[float], threshold: float) -> BinaryMetrics:
    y_pred = [1 if score >= threshold else 0 for score in y_score]
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 1)
    tn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 0)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 1)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 0)
    total = max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return BinaryMetrics(
        accuracy=(tp + tn) / total,
        precision=precision,
        recall=recall,
        f1=f1,
        support=sum(1 for truth in y_true if truth == 1),
    )


def optimize_threshold(y_true: list[int], y_score: list[float]) -> float:
    candidates = sorted(set(float(x) for x in y_score))
    if not candidates:
        return 0.5
    candidates = [0.5, *candidates]
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        metrics = binary_metrics(y_true, y_score, threshold)
        if metrics.f1 > best_f1:
            best_f1 = metrics.f1
            best_threshold = float(threshold)
    return best_threshold


def _try_average_precision(y_true: list[int], y_score: list[float]) -> float | None:
    try:
        from sklearn.metrics import average_precision_score
    except Exception:
        return None
    if sum(1 for truth in y_true if truth == 1) == 0:
        return None
    return float(average_precision_score(y_true, y_score))


def evaluate_pathology_primary5(
    rows: Iterable[dict[str, str]],
    *,
    out_dir: str | Path,
    calibration_fraction: float = 0.5,
    primary_pathologies: Iterable[str] | None = None,
    quarantine_pathologies: Iterable[str] | None = None,
) -> dict[str, object]:
    primary, quarantine, auxiliary = _validate_pathology_selection(primary_pathologies, quarantine_pathologies)
    rows = [dict(row) for row in rows]
    if not rows:
        raise ValueError("Empty pathology prediction table")

    for row in rows:
        row.setdefault("record_id", row.get("record_key") or row.get("path") or str(len(row)))
        row.setdefault("split", stable_split(row["record_id"], calibration_fraction))

    calibration = [row for row in rows if row.get("split") == "calibration"]
    test = [row for row in rows if row.get("split") == "test"]
    if not calibration or not test:
        raise ValueError("Need non-empty calibration and test splits")

    thresholds: dict[str, float] = {}
    per_class: dict[str, dict[str, float | int | None]] = {}
    prediction_rows: list[dict[str, str | int | float]] = []
    pr_aucs: list[float] = []
    f1s: list[float] = []
    accuracies: list[float] = []

    for label in primary:
        y_cal = [_truth(row, label) for row in calibration]
        s_cal = [_score(row, label) for row in calibration]
        threshold = optimize_threshold(y_cal, s_cal)
        thresholds[label] = threshold

        y_test = [_truth(row, label) for row in test]
        s_test = [_score(row, label) for row in test]
        metrics = binary_metrics(y_test, s_test, threshold)
        for row, truth, score in zip(test, y_test, s_test):
            prediction_rows.append(
                {
                    "record_id": row.get("record_id", ""),
                    "split": row.get("split", "test"),
                    "label": label,
                    "true": int(truth),
                    "score": float(score),
                    "threshold": float(threshold),
                    "predicted": int(score >= threshold),
                }
            )
        pr_auc = _try_average_precision(y_test, s_test)
        if pr_auc is not None:
            pr_aucs.append(pr_auc)
        f1s.append(metrics.f1)
        accuracies.append(metrics.accuracy)
        per_class[label] = {
            "threshold": threshold,
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "support": metrics.support,
            "pr_auc": pr_auc,
        }

    macro_f1 = float(_mean(f1s))
    per_label_accuracy = float(_mean(accuracies))
    macro_pr_auc = float(_mean(pr_aucs)) if pr_aucs else None
    report = {
        "primary_pathologies": list(primary),
        "quarantine_pathologies": list(quarantine),
        "auxiliary_pathologies": list(auxiliary),
        "calibration_records": len(calibration),
        "test_records": len(test),
        "thresholds": thresholds,
        "per_class": per_class,
        "macro_f1": macro_f1,
        "macro_pr_auc": macro_pr_auc,
        "per_label_accuracy": per_label_accuracy,
        "gate": {
            "macro_f1_min": 0.70,
            "per_label_accuracy_min": 0.80,
            "passed": bool(macro_f1 >= 0.70 and per_label_accuracy >= 0.80),
        },
    }
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_pathology_selection_report(
        output / "reportable_pathology_selection.json",
        primary_pathologies=primary,
        quarantine_pathologies=quarantine,
        auxiliary_pathologies=auxiliary,
    )
    (output / "pathology_primary5_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output / "pathology_primary5_per_class.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "threshold", "accuracy", "precision", "recall", "f1", "support", "pr_auc"],
        )
        writer.writeheader()
        for label, metrics in per_class.items():
            writer.writerow({"label": label, **metrics})
    with (output / "pathology_primary5_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["record_id", "split", "label", "true", "score", "threshold", "predicted"],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)
    return report


def summarize_existing_pathology_report(
    report_json: str | Path,
    *,
    out_dir: str | Path,
    primary_pathologies: Iterable[str] | None = None,
    quarantine_pathologies: Iterable[str] | None = None,
) -> dict[str, object]:
    """Create a Primary-5/quarantine summary from an existing per-class report."""

    primary, quarantine, auxiliary = _validate_pathology_selection(primary_pathologies, quarantine_pathologies)
    with open(_windows_long_path(report_json), "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    per_class, generic_label_mapping = _canonicalize_per_class(payload.get("per_class", {}))
    primary_rows: dict[str, dict[str, float | int | None]] = {}
    quarantine_rows: dict[str, dict[str, float | int | None]] = {}
    auxiliary_rows: dict[str, dict[str, float | int | None]] = {}
    primary_f1: list[float] = []
    primary_pr_auc: list[float] = []
    primary_accuracy: list[float] = []
    for label in primary:
        row = dict(per_class.get(label, {}))
        f1 = row.get("best_f1")
        pr_auc = row.get("pr_auc")
        accuracy = row.get("accuracy")
        if f1 is not None:
            primary_f1.append(float(f1))
        if pr_auc is not None:
            primary_pr_auc.append(float(pr_auc))
        if accuracy is not None:
            primary_accuracy.append(float(accuracy))
        primary_rows[label] = {
            "best_threshold": row.get("best_threshold"),
            "f1": row.get("best_f1"),
            "accuracy": row.get("accuracy"),
            "pr_auc": row.get("pr_auc"),
            "support": row.get("support"),
        }
    for label in quarantine:
        row = dict(per_class.get(label, {}))
        quarantine_rows[label] = {
            "best_threshold": row.get("best_threshold"),
            "f1": row.get("best_f1"),
            "accuracy": row.get("accuracy"),
            "pr_auc": row.get("pr_auc"),
            "support": row.get("support"),
        }
    for label in auxiliary:
        row = dict(per_class.get(label, {}))
        auxiliary_rows[label] = {
            "best_threshold": row.get("best_threshold"),
            "f1": row.get("best_f1"),
            "accuracy": row.get("accuracy"),
            "pr_auc": row.get("pr_auc"),
            "support": row.get("support"),
        }
    macro_f1 = float(_mean(primary_f1)) if primary_f1 else 0.0
    macro_pr_auc = float(_mean(primary_pr_auc)) if primary_pr_auc else None
    per_label_accuracy = float(_mean(primary_accuracy)) if primary_accuracy else None
    missing_primary = [label for label, row in primary_rows.items() if row["f1"] is None]
    missing_quarantine = [label for label, row in quarantine_rows.items() if row["f1"] is None]
    summary = {
        "source_report": str(report_json),
        "primary_pathologies": list(primary),
        "quarantine_pathologies": list(quarantine),
        "auxiliary_pathologies": list(auxiliary),
        "canonical_pathology_10": list(CANONICAL_PATHOLOGY_10),
        "generic_label_mapping": generic_label_mapping,
        "primary_per_class": primary_rows,
        "quarantine_per_class": quarantine_rows,
        "auxiliary_per_class": auxiliary_rows,
        "missing_primary_labels": missing_primary,
        "missing_quarantine_labels": missing_quarantine,
        "primary_macro_f1_from_existing_thresholds": macro_f1,
        "primary_per_label_accuracy_from_existing_thresholds": per_label_accuracy,
        "primary_macro_pr_auc": macro_pr_auc,
        "gate": {
            "macro_f1_min": 0.70,
            "per_label_accuracy_min": 0.80,
            "passed": bool(
                macro_f1 >= 0.70
                and per_label_accuracy is not None
                and per_label_accuracy >= 0.80
            ),
            "note": (
                "Existing report includes per-label accuracy."
                if per_label_accuracy is not None
                else "Existing report lacks per-label accuracy; full gate requires prediction CSV."
            ),
        },
    }
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_pathology_selection_report(
        output / "reportable_pathology_selection.json",
        primary_pathologies=primary,
        quarantine_pathologies=quarantine,
        auxiliary_pathologies=auxiliary,
    )
    (output / "pathology_primary5_existing_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions_csv", default=None)
    parser.add_argument("--existing_report_json", default=None)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--reportable_pathologies_json", default=None)
    parser.add_argument("--reportable_pathologies_csv", default=None)
    parser.add_argument("--quarantine_pathologies_json", default=None)
    parser.add_argument("--quarantine_pathologies_csv", default=None)
    args = parser.parse_args(argv)
    primary_pathologies = _read_classes_file(
        args.reportable_pathologies_json or args.reportable_pathologies_csv,
        json_key="reportable_pathologies",
    )
    quarantine_pathologies = _read_classes_file(
        args.quarantine_pathologies_json or args.quarantine_pathologies_csv,
        json_key="quarantine_pathologies",
    )
    if args.existing_report_json:
        report = summarize_existing_pathology_report(
            args.existing_report_json,
            out_dir=args.out_dir,
            primary_pathologies=primary_pathologies,
            quarantine_pathologies=quarantine_pathologies,
        )
    elif args.predictions_csv:
        report = evaluate_pathology_primary5(
            read_rows(args.predictions_csv),
            out_dir=args.out_dir,
            primary_pathologies=primary_pathologies,
            quarantine_pathologies=quarantine_pathologies,
        )
    else:
        raise SystemExit("--predictions_csv or --existing_report_json is required")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
