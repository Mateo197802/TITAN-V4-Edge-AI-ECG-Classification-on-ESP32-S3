from __future__ import annotations

import csv
import json
from pathlib import Path

from titan_v4.metrics.external_validation import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def test_primary9_legacy_comparison_artifacts_are_complete_and_hash_linked():
    evidence_dir = ROOT / "outputs/reviewer_verification/arrhythmia_primary9"
    report_path = evidence_dir / "primary9_current_vs_historical_label_report.json"
    table_path = evidence_dir / "primary9_current_vs_historical_label_predictions.csv"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    with table_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert report["record_count"] == 672
    assert report["current_checkpoint_vs_rebuilt_labels"]["correct_predictions"] == 466
    assert report["current_checkpoint_vs_historical_labels"]["correct_predictions"] == 568
    assert report["historical_stored_aggregate"]["correct_predictions"] == 605
    assert report["label_table_comparison"]["different_label_count"] == 129
    assert len(rows) == 672
    assert report["comparison_csv_sha256"] == sha256_file(table_path)
    assert report["prediction_csv_sha256"] == sha256_file(
        ROOT / "outputs/reproduced/primary9/primary9_record_predictions.csv"
    )
    assert report["scope"]["independent_external_validation_claim"] is False


def test_pathology_audit_keeps_cedia_checkpoint_separate_from_published_checkpoint():
    audit_path = ROOT / "outputs/reviewer_verification/pathology_primary5/pathology_reproducibility_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    cedia = audit["separate_cedia_candidate"]

    assert audit["evaluation_performed"] is False
    assert audit["scope"]["historical_metric_reproduced"] is False
    assert audit["distributed_checkpoint"]["training_metadata"]["loss_weight"] == 0
    assert cedia["model"]["pathology_loss_weight"] == 0.2
    assert cedia["matches_distributed_checkpoint"] is False
    assert cedia["historical_primary5_claim_linked"] is False
    assert cedia["output_inventory"]["pathology_prediction_or_primary5_artifact_found"] is False
