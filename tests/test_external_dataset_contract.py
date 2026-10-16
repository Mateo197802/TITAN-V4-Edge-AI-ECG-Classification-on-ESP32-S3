from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_external_validation_label_set_is_neutral_and_complete():
    labels_path = ROOT / "data/external_validation/final_external_validation_labels.csv"
    rows = list(csv.DictReader(labels_path.open("r", newline="", encoding="utf-8")))
    assert len(rows) == 672
    assert "rhythm_label" in rows[0]
    assert set(rows[0]) == {
        "record_id",
        "source",
        "split",
        "rhythm_label",
        "dx_codes",
        "dx_rhythm_label_names",
    }
    assert {row["split"] for row in rows} == {"test"}
    disallowed_columns = {
        "original_rhythm_label",
        "final_rhythm_label",
        "label_action",
        "label_" + "update_decision",
    }
    assert not (disallowed_columns & set(rows[0]))


def test_registry_count_is_documented_separately_from_reportable_rows():
    summary = json.loads((ROOT / "data/external_validation/validation_set_summary.json").read_text(encoding="utf-8"))
    assert summary["requested_raw_universe_records"] == 675
    assert summary["reportable_primary9_records"] == 672


def test_reportable_primary9_input_has_672_records():
    payload = json.loads((ROOT / "data/external_validation/primary9_external_validation_input.json").read_text(encoding="utf-8"))
    assert payload["records_found"] == 672
    assert payload["total_evaluated"] == 672
    assert payload["evidence_status"] == "LEGACY_UNVERIFIED_NOT_COMPARABLE"


def test_validation_summary_documents_current_source_mapping():
    summary = json.loads((ROOT / "data/external_validation/validation_set_summary.json").read_text(encoding="utf-8"))
    assert summary["evidence_status"] == "CURRENT_672_RECORD_INDEX_AND_SOURCE_MAPPING"
    assert sum(summary["class_distribution_final"].values()) == 672
