from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_pathology_external_metrics_are_exact():
    report = json.loads(
        (ROOT / "outputs/gold_master_external_validation/pathology_primary5/pathology_primary5_external_summary.json")
        .read_text(encoding="utf-8")
    )
    assert report["label_set"] == "final external validation label set"
    assert report["primary_pathologies"] == ["IMI", "ALMI", "ILMI", "LAE", "ISC_"]
    assert report["accuracy"] == 0.90256
    assert report["macro_f1"] == 0.66874
    assert report["external_training_allowed"] is False
    assert report["external_threshold_tuning_allowed"] is False

