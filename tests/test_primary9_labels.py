from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from titan_v4.reproducibility.labels import (
    parse_dx_codes_from_comments,
    primary9_label_from_dx_codes,
    primary9_names_from_dx_codes,
)


ROOT = Path(__file__).resolve().parents[1]


def test_wfdb_dx_parser_accepts_standard_comment_forms():
    assert parse_dx_codes_from_comments(["Age: 55", "# Dx: 426783006, 427084000"]) == [
        "426783006",
        "427084000",
    ]


def test_primary9_label_uses_header_order_and_pr_interval_fallback():
    assert primary9_label_from_dx_codes(["59118001", "164889003"]) == "RBBB"
    assert primary9_label_from_dx_codes(["164947007", "426783006"]) == "1AVB"
    assert primary9_label_from_dx_codes(["164947007", "426783006", "427084000"]) == "STACH"


def test_primary9_label_table_is_minimal_complete_and_matches_dx_codes():
    labels = list(
        csv.DictReader(
            (ROOT / "data/external_validation/final_external_validation_labels.csv").open(
                "r", newline="", encoding="utf-8"
            )
        )
    )
    assert len(labels) == 672
    assert set(labels[0]) == {
        "record_id",
        "source",
        "split",
        "rhythm_label",
        "dx_codes",
        "dx_rhythm_label_names",
    }
    assert Counter(row["rhythm_label"] for row in labels) == {
        "AFIB": 46,
        "SB": 38,
        "STACH": 147,
        "NSR": 38,
        "PVC": 134,
        "RBBB": 47,
        "LBBB": 30,
        "PAC": 146,
        "1AVB": 46,
    }
    for row in labels:
        codes = row["dx_codes"].split("|")
        assert primary9_label_from_dx_codes(codes) == row["rhythm_label"]
        assert "|".join(primary9_names_from_dx_codes(codes)) == row["dx_rhythm_label_names"]


def test_source_manifest_resolves_all_former_data_test_rows():
    rows = list(
        csv.DictReader(
            (ROOT / "data/external_validation/record_source_manifest.csv").open(
                "r", newline="", encoding="utf-8"
            )
        )
    )
    formerly_unresolved = [row for row in rows if row["source_label"] == "data_test"]
    assert len(formerly_unresolved) == 312
    assert all(row["record_path"].startswith(f"training/{row['dataset']}/g") for row in formerly_unresolved)
    assert all(row["physionet_release"] == "1.0.3" for row in rows)
    assert all(row["upstream_partition"] == "training" for row in rows)
    assert all(row["data_license"] == "CC BY 4.0" for row in rows)
    assert Counter(row["dataset"] for row in rows) == {
        "cpsc_2018": 56,
        "cpsc_2018_extra": 42,
        "georgia": 56,
        "ptb-xl": 378,
        "chapman_shaoxing": 70,
        "ningbo": 70,
    }
