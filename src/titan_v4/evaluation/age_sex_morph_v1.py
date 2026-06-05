"""Age/sex morphology evidence gate.

This protocol evaluates whether an age/sex claim is supported by real metadata
and morphology-derived predictions, not by imputed defaults or source leakage.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


def _windows_long_path(path: str | Path) -> str:
    raw = str(path)
    if raw.startswith("\\\\?\\") or not Path(raw).is_absolute():
        return raw
    if len(raw) >= 240:
        return "\\\\?\\" + raw
    return raw


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _sex_to_int(value: str) -> int | None:
    text = str(value or "").strip().lower()
    if text in {"m", "male", "masculino", "1", "1.0"}:
        return 1
    if text in {"f", "female", "femenino", "0", "0.0"}:
        return 0
    return None


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / max(len(values), 1)


def reject_imputed_metadata(rows: Iterable[dict[str, str]]) -> None:
    bad = []
    for idx, row in enumerate(rows):
        flag = str(row.get("metadata_source") or row.get("imputed") or "").strip().lower()
        if flag in {"imputed", "default", "synthetic", "1", "true"}:
            bad.append(idx)
    if bad:
        raise ValueError(f"Age/sex metadata contains imputed/default rows: {bad[:10]}")


def sex_f1(y_true: list[int], y_pred: list[int]) -> float:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def source_only_baseline(rows: list[dict[str, str]]) -> dict[str, float]:
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_source[str(row.get("source") or "unknown")].append(row)
    pred_sex: list[int] = []
    true_sex: list[int] = []
    age_errors: list[float] = []
    for _source, group in by_source.items():
        sexes = [_sex_to_int(str(row.get("true_sex") or row.get("sex"))) for row in group]
        sexes = [sex for sex in sexes if sex is not None]
        majority = 1 if sum(sexes) >= len(sexes) / 2 else 0
        ages = [_float(row, "true_age", _float(row, "age", -1.0)) for row in group]
        ages = [age for age in ages if age >= 0]
        mean_age = float(_mean(ages)) if ages else 0.0
        for row in group:
            sex = _sex_to_int(str(row.get("true_sex") or row.get("sex")))
            if sex is not None:
                true_sex.append(sex)
                pred_sex.append(majority)
            age = _float(row, "true_age", _float(row, "age", -1.0))
            if age >= 0:
                age_errors.append(abs(mean_age - age))
    return {
        "sex_f1": sex_f1(true_sex, pred_sex) if true_sex else 0.0,
        "age_mae": float(_mean(age_errors)) if age_errors else 0.0,
    }


def evaluate_age_sex_morphology(rows: Iterable[dict[str, str]], *, out_dir: str | Path) -> dict[str, object]:
    rows = [dict(row) for row in rows]
    if not rows:
        raise ValueError("Empty AGE_SEX_MORPH input")
    reject_imputed_metadata(rows)

    true_sex: list[int] = []
    pred_sex: list[int] = []
    age_errors: list[float] = []
    for row in rows:
        sex = _sex_to_int(str(row.get("true_sex") or row.get("sex")))
        if sex is not None:
            score = _float(row, "morph_pred_sex_score", _float(row, "pred_sex_score", 0.5))
            true_sex.append(sex)
            pred_sex.append(1 if score >= 0.5 else 0)
        age = _float(row, "true_age", _float(row, "age", -1.0))
        if age >= 0:
            pred_age = _float(row, "morph_pred_age", _float(row, "pred_age", age))
            age_errors.append(abs(pred_age - age))

    morph = {
        "sex_f1": sex_f1(true_sex, pred_sex) if true_sex else 0.0,
        "age_mae": float(_mean(age_errors)) if age_errors else 0.0,
    }
    source = source_only_baseline(rows)
    report = {
        "records": len(rows),
        "metadata_policy": "real_metadata_only",
        "morphology_metrics": morph,
        "source_only_baseline": source,
        "gate": {
            "sex_f1_min": 0.75,
            "age_mae_max": 10.0,
            "beats_source_only": bool(morph["sex_f1"] > source["sex_f1"] and morph["age_mae"] < source["age_mae"]),
            "passed": bool(
                morph["sex_f1"] >= 0.75
                and morph["age_mae"] < 10.0
                and morph["sex_f1"] > source["sex_f1"]
                and morph["age_mae"] < source["age_mae"]
            ),
        },
    }
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "age_sex_morph_v1_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return report


def summarize_existing_age_sex_report(
    report_json: str | Path,
    *,
    out_dir: str | Path,
    predictions_csv: str | Path | None = None,
) -> dict[str, object]:
    with open(_windows_long_path(report_json), "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    metrics = payload.get("metrics", {})
    morphology_dim = int(payload.get("morphology_dim", 0) or 0)
    sex_f1 = float(metrics.get("biometry_sex_f1", 0.0) or 0.0)
    age_mae = float(metrics.get("biometry_age_mae_years", 999.0) or 999.0)
    source_baseline = None
    beats_source_only = False
    if predictions_csv:
        source_baseline = source_only_baseline(read_rows(predictions_csv))
        beats_source_only = bool(sex_f1 > source_baseline["sex_f1"] and age_mae < source_baseline["age_mae"])
    report = {
        "source_report": str(report_json),
        "predictions_csv": str(predictions_csv) if predictions_csv else None,
        "records_evaluated": int(payload.get("records_evaluated", 0) or 0),
        "morphology_dim": morphology_dim,
        "current_metrics": {
            "sex_f1": sex_f1,
            "age_mae_years": age_mae,
            "sex_accuracy": metrics.get("biometry_sex_acc"),
            "age_within10_acc": metrics.get("biometry_age_within10_acc"),
        },
        "source_only_baseline": source_baseline,
        "gate": {
            "sex_f1_min": 0.75,
            "age_mae_max": 10.0,
            "requires_morphology_dim_gt_0": True,
            "beats_source_only": beats_source_only,
            "passed": bool(
                morphology_dim > 0
                and sex_f1 >= 0.75
                and age_mae < 10.0
                and (not predictions_csv or beats_source_only)
            ),
        },
        "decision": "exploratory_only_until_morphology_probe_beats_source_baseline",
    }
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "age_sex_existing_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", default=None)
    parser.add_argument("--existing_report_json", default=None)
    parser.add_argument("--predictions_csv", default=None)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args(argv)
    if args.existing_report_json:
        report = summarize_existing_age_sex_report(
            args.existing_report_json,
            out_dir=args.out_dir,
            predictions_csv=args.predictions_csv,
        )
    elif args.input_csv:
        report = evaluate_age_sex_morphology(read_rows(args.input_csv), out_dir=args.out_dir)
    else:
        raise SystemExit("--input_csv or --existing_report_json is required")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
