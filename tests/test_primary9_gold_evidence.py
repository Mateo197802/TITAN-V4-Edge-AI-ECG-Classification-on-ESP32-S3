from __future__ import annotations

import csv
import importlib
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_confusion_matrix_metrics_recompute_standard_binary_result():
    module = importlib.import_module("scripts.verify_primary9_gold_evidence")
    metrics = module.confusion_matrix_metrics([[3, 1], [1, 3]])

    assert metrics["records"] == 8
    assert metrics["correct"] == 6
    assert metrics["accuracy"] == 0.75
    assert math.isclose(metrics["macro_f1"], 0.75)
    assert math.isclose(metrics["weighted_f1"], 0.75)


def test_bundle_hashes_are_stable_across_text_line_endings(tmp_path):
    module = importlib.import_module("scripts.verify_primary9_gold_evidence")
    lf_file = tmp_path / "evidence.json"
    crlf_file = tmp_path / "evidence-windows.json"
    lf_file.write_bytes(b'{\n  "records": 672\n}\n')
    crlf_file.write_bytes(b'{\r\n  "records": 672\r\n}\r\n')

    assert module._bundle_fingerprint(lf_file) == module._bundle_fingerprint(crlf_file)


def test_cedia_gold_aggregate_and_fresh_inference_are_verified_separately():
    module = importlib.import_module("scripts.verify_primary9_gold_evidence")
    result = module.verify_bundle()

    assert result["historical_source_result"]["correct"] == 605
    assert result["historical_source_result"]["records"] == 672
    assert math.isclose(result["historical_source_result"]["accuracy"], 0.9002976190476191)
    assert math.isclose(result["historical_source_result"]["macro_f1"], 0.8781916793252358)
    assert result["historical_source_result"]["recalculated_from_saved_confusion_matrix"] is True
    assert result["label_update_audit"]["updated_rows"] == 129
    assert result["label_update_audit"]["final_labels_equal_cedia_model_top1"] == 129
    assert result["label_update_audit"]["local_rerun_predictions_equal_cedia_model_top1"] == 110
    assert result["fresh_local_inference"]["correct"] == 568
    assert result["fresh_local_inference"]["matches_historical_report"] is False


def test_published_gold_evidence_excludes_private_source_fields_and_internal_gate_terms():
    bundle = ROOT / "outputs/reviewer_verification/arrhythmia_primary9/cedia_gold_2026-05-26"
    projection_text = (bundle / "primary9_cedia_gold_source_projection.json").read_text(encoding="utf-8")
    p5 = json.loads((ROOT / "reports/evidence/pathology-primary5-gold-source.json").read_text(encoding="utf-8"))

    assert "/home/" not in projection_text.casefold()
    assert "gavilanes" not in projection_text.casefold()
    assert "kevin.landazuri" not in projection_text.casefold()
    assert re.search(r"\bgates?\b", projection_text, flags=re.IGNORECASE) is None
    assert re.search(r"\bgates?\b", json.dumps(p5), flags=re.IGNORECASE) is None

    with (bundle / "primary9_label_history_672.csv").open(encoding="utf-8", newline="") as handle:
        history_header = next(csv.reader(handle))
    with (bundle / "primary9_target_updates_129.csv").open(encoding="utf-8", newline="") as handle:
        update_header = next(csv.reader(handle))
    forbidden = {"age", "sex", "dx_codes", "record_path", "header_path", "label_update_notes"}
    assert forbidden.isdisjoint(history_header)
    assert forbidden.isdisjoint(update_header)
