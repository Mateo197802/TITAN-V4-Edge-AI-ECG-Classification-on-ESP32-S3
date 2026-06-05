from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_cascade_annex_metrics_are_reported_separately():
    report = json.loads(
        (ROOT / "outputs/gold_master_external_validation/cascade_annex/cascade_safety_annex_report.json")
        .read_text(encoding="utf-8")
    )
    assert report["role"] == "safety_annex"
    assert report["diagnostic_subset_records"] == 211
    assert report["coverage"] == 0.5071090047393365
    assert report["quarantine_rate"] == 0.4928909952606635
    assert report["action_counts"]["accepted_prediction"] == 107

