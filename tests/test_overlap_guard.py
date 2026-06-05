from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_external_record_ids_are_unique():
    labels_path = ROOT / "data/external_validation/final_external_validation_labels.csv"
    rows = list(csv.DictReader(labels_path.open("r", newline="", encoding="utf-8")))
    ids = [row["record_id"].strip().lower() for row in rows]
    assert len(ids) == len(set(ids))

