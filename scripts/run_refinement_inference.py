"""Repeat the refinement evaluation directly from pinned public WFDB records."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from titan_v4.reproducibility.inference import load_primary9_model, predict_primary9, primary9_metrics
from titan_v4.data.primary9_registry import PRIMARY9_CLASSES
from titan_v4.reproducibility.refinement import load_tensor, read_manifest, sha256

BUNDLE = ROOT / "outputs/reviewer_verification/partial_backbone_refinement_2026-10-12"


def key(row):
    return row["cohort"], row["dataset"], row["ecg_id"]


def compare_predictions(rows, p, expected_rows, tolerance):
    expected = {key(row): row for row in expected_rows}
    if len(expected) != len(expected_rows):
        raise ValueError("Expected predictions contain duplicate record keys")
    if len({key(row) for row in rows}) != len(rows):
        raise ValueError("Source manifest contains duplicate record keys")
    reference = []
    for row in rows:
        match = expected[key(row)]
        if match["reference"] != row["reference_label"]:
            raise ValueError("Reference label changed between manifests")
        reference.append([float(match[f"prob_{label}"]) for label in PRIMARY9_CLASSES])
    reference = np.asarray(reference)
    maximum = float(np.max(np.abs(p - reference)))
    top1 = int(np.sum(p.argmax(1) == reference.argmax(1)))
    return {"records": len(rows), "identical_top1": top1, "max_probability_abs_difference": maximum,
            "probability_tolerance": tolerance, "passed": top1 == len(rows) and maximum <= tolerance}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=BUNDLE / "manifests/followup_records.csv")
    parser.add_argument("--checkpoint", type=Path, default=BUNDLE / "model/partial_backbone_candidate.pth")
    parser.add_argument("--expected-predictions", type=Path, default=BUNDLE / "evaluation/followup_predictions.csv")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/cache/physionet")
    parser.add_argument("--output", type=Path, default=ROOT / "data/cache/refinement_reproduced")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--probability-tolerance", type=float, default=2e-5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Use a fresh output directory: {args.output}")
    if args.workers < 1 or args.probability_tolerance < 0:
        parser.error("workers must be positive and tolerance nonnegative")
    rows = read_manifest(args.manifest)
    if not rows:
        raise ValueError("Empty source manifest")
    torch.set_num_threads(2)
    model = load_primary9_model(args.checkpoint, device=args.device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    if parameters != 1_143_751:
        raise ValueError("Original parameter count changed")
    def read(row):
        return load_tensor(row, args.data_root, download=not args.offline)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        signals = np.stack(list(executor.map(read, rows)))
    p = predict_primary9(model, signals, batch_size=128)
    comparison = compare_predictions(rows, p, read_manifest(args.expected_predictions), args.probability_tolerance)
    labels = [PRIMARY9_CLASSES[index] for index in p.argmax(1)]
    metrics = {cohort: primary9_metrics(
        [row["reference_label"] for row in rows if row["cohort"] == cohort],
        [label for row, label in zip(rows, labels, strict=True) if row["cohort"] == cohort])
        for cohort in sorted({row["cohort"] for row in rows})}
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "records": len(rows),
              "source_header_signal_and_tensor_hashes_verified": len(rows), "parameters": parameters,
              "checkpoint_sha256": sha256(args.checkpoint), "manifest_sha256": sha256(args.manifest),
              "comparison_with_cedia": comparison, "cohorts": metrics,
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "numpy": np.__version__, "device": args.device}}
    args.output.mkdir(parents=True)
    fields = ("cohort", "dataset", "ecg_id", "reference", "phase3_prediction",
              *(f"prob_{label}" for label in PRIMARY9_CLASSES))
    with (args.output / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows):
            output = {"cohort": row["cohort"], "dataset": row["dataset"], "ecg_id": row["ecg_id"],
                      "reference": row["reference_label"], "phase3_prediction": labels[index]}
            output.update({f"prob_{label}": float(p[index,j]) for j,label in enumerate(PRIMARY9_CLASSES)})
            writer.writerow(output)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "records": len(rows), "comparison": comparison}, indent=2))
    return 0 if comparison["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
