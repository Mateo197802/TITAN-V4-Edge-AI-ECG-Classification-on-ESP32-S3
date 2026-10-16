from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from titan_v4.metrics.external_validation import sha256_file
from titan_v4.reproducibility.legacy_comparison import build_comparison


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = ROOT / "outputs/reproduced/primary9/primary9_record_predictions.csv"
DEFAULT_CURRENT_LABELS = ROOT / "data/external_validation/final_external_validation_labels.csv"
DEFAULT_LEGACY_LABELS = ROOT / "outputs/gold_master_external_validation/validation_set/final_external_validation_labels.csv"
DEFAULT_HISTORICAL_REPORT = ROOT / "outputs/gold_master_external_validation/arrhythmia_primary9/primary9_external_validation_report.json"
DEFAULT_RUN_MANIFEST = ROOT / "outputs/reproduced/primary9/run_manifest.json"
DEFAULT_SOURCE_MANIFEST = ROOT / "data/external_validation/record_source_manifest.csv"
DEFAULT_CHECKPOINT = ROOT / "models/gold_master/gold_master_primary9_model.pth"
DEFAULT_OUTPUT = ROOT / "outputs/reviewer_verification/arrhythmia_primary9"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty comparison table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score one frozen Primary-9 checkpoint prediction table against current and legacy labels."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--current-labels", type=Path, default=DEFAULT_CURRENT_LABELS)
    parser.add_argument("--legacy-labels", type=Path, default=DEFAULT_LEGACY_LABELS)
    parser.add_argument("--historical-report", type=Path, default=DEFAULT_HISTORICAL_REPORT)
    parser.add_argument("--run-manifest", type=Path, default=DEFAULT_RUN_MANIFEST)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    historical = json.loads(args.historical_report.read_text(encoding="utf-8"))
    report, rows = build_comparison(
        read_csv(args.predictions),
        read_csv(args.current_labels),
        read_csv(args.legacy_labels),
        historical_report=historical,
    )
    run_manifest = json.loads(args.run_manifest.read_text(encoding="utf-8"))
    prediction_sha = sha256_file(args.predictions)
    current_labels_sha = sha256_file(args.current_labels)
    source_manifest_sha = sha256_file(args.source_manifest)
    checkpoint_sha = sha256_file(args.checkpoint)
    if run_manifest.get("outputs", {}).get(args.predictions.name) != prediction_sha:
        raise ValueError("Run manifest does not authenticate the prediction CSV supplied for comparison")
    if run_manifest.get("label_file", {}).get("sha256") != current_labels_sha:
        raise ValueError("Run manifest does not authenticate the current label CSV supplied for comparison")
    if run_manifest.get("source_manifest", {}).get("sha256") != source_manifest_sha:
        raise ValueError("Run manifest does not authenticate the source manifest supplied for comparison")
    if run_manifest.get("model", {}).get("sha256") != checkpoint_sha:
        raise ValueError("Run manifest does not authenticate the checkpoint supplied for comparison")

    report.update(
        {
            "checkpoint_sha256": checkpoint_sha,
            "prediction_csv_sha256": prediction_sha,
            "current_label_csv_sha256": current_labels_sha,
            "historical_label_csv_sha256": sha256_file(args.legacy_labels),
            "historical_aggregate_json_sha256": sha256_file(args.historical_report),
            "source_manifest_sha256": source_manifest_sha,
            "inference_run_manifest_sha256": sha256_file(args.run_manifest),
        }
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table_path = args.out_dir / "primary9_current_vs_historical_label_predictions.csv"
    report_path = args.out_dir / "primary9_current_vs_historical_label_report.json"
    _write_csv(table_path, rows)
    report["comparison_csv_sha256"] = sha256_file(table_path)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Evidence written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
