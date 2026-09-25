from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from titan_v4.evaluation.pathology_evidence import build_audit
from titan_v4.metrics.external_validation import sha256_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HISTORICAL_REPORT = ROOT / "outputs/gold_master_external_validation/pathology_primary5/pathology_primary5_external_summary.json"
DEFAULT_TRAINING_SUMMARY = ROOT / "models/gold_master/gold_master_primary9_training_summary.json"
DEFAULT_CHECKPOINT = ROOT / "models/gold_master/gold_master_primary9_model.pth"
DEFAULT_LABELS = ROOT / "outputs/gold_master_external_validation/validation_set/final_external_validation_labels.csv"
DEFAULT_PREDICTIONS = ROOT / "outputs/gold_master_external_validation/pathology_primary5/pathology_primary5_predictions.csv"
DEFAULT_CEDIA_EVIDENCE = ROOT / "reports/evidence/cedia-pathology-crosscheck.json"
DEFAULT_OUTPUT = ROOT / "outputs/reviewer_verification/pathology_primary5/pathology_reproducibility_audit.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit whether the historical Primary-5 pathology metric can be reproduced.")
    parser.add_argument("--historical-report", type=Path, default=DEFAULT_HISTORICAL_REPORT)
    parser.add_argument("--training-summary", type=Path, default=DEFAULT_TRAINING_SUMMARY)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--candidate-labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--historical-predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--cedia-evidence", type=Path, default=DEFAULT_CEDIA_EVIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    historical = json.loads(args.historical_report.read_text(encoding="utf-8"))
    training = json.loads(args.training_summary.read_text(encoding="utf-8"))
    report = build_audit(
        historical,
        training,
        checkpoint_path=args.checkpoint,
        label_path=args.candidate_labels,
        prediction_path=args.historical_predictions,
        cedia_candidate=json.loads(args.cedia_evidence.read_text(encoding="utf-8")),
    )
    report["historical_report_sha256"] = sha256_file(args.historical_report)
    report["training_summary_sha256"] = sha256_file(args.training_summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Evidence written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
