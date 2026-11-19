from __future__ import annotations

import pytest

from titan_v4.evaluation import cedia_pathology_cohort as cohort
from titan_v4.evaluation.cedia_pathology_cohort import (
    build_patient_disjoint_cohorts,
    normalize_ptbxl_record_id,
    ptbxl_split_record_ids,
    ptbxl_split_record_ids_by_source,
)


def test_cohorts_exclude_training_patients_from_calibration_and_test():
    labels = {f"{record_id:05d}": [record_id % 10] for record_id in range(1, 10)}
    patients = {
        "00001": "p1",
        "00002": "p2",
        "00003": "p3",
        "00004": "p1",
        "00005": "p4",
        "00006": "p5",
        "00007": "p2",
        "00008": "p6",
        "00009": "p7",
    }

    cohorts = build_patient_disjoint_cohorts(
        labels,
        train_record_ids=["00001", "00002"],
        validation_record_ids=["00003", "00004", "00005"],
        patient_by_record=patients,
    )

    assert cohorts.calibration_record_ids == ("00003", "00005")
    assert cohorts.test_record_ids == ("00006", "00008", "00009")
    assert cohorts.calibration_patient_count == 2
    assert cohorts.test_patient_count == 3
    assert cohorts.excluded_counts == {
        "labeled_training_records": 2,
        "checkpoint_train_validation_record_id_overlap": 0,
        "calibration_patient_overlap_with_training": 1,
        "test_patient_overlap_with_training_or_validation": 1,
    }


def test_train_validation_record_overlap_is_excluded_from_calibration_and_reported():
    cohorts = build_patient_disjoint_cohorts(
        {"00001": [0], "00002": [1], "00003": [2]},
        train_record_ids=["00001"],
        validation_record_ids=["00001", "00002"],
        patient_by_record={"00001": "p1", "00002": "p2", "00003": "p3"},
    )

    assert cohorts.calibration_record_ids == ("00002",)
    assert cohorts.test_record_ids == ("00003",)
    assert cohorts.excluded_counts["checkpoint_train_validation_record_id_overlap"] == 1


def test_cohort_builder_rejects_missing_patient_metadata():
    with pytest.raises(ValueError, match="missing patient IDs"):
        build_patient_disjoint_cohorts(
            {"00001": [0]},
            train_record_ids=[],
            validation_record_ids=[],
            patient_by_record={},
        )


@pytest.mark.parametrize("path", ["00008_lr", r"C:\ptb-xl\records100\00008_lr", "00008_lr.hea"])
def test_ptbxl_record_id_normalizes_record_paths(path):
    assert normalize_ptbxl_record_id(path) == "00008"


def test_ptbxl_record_id_rejects_non_ptbxl_identifier():
    with pytest.raises(ValueError, match="PTB-XL record identifier"):
        normalize_ptbxl_record_id("record-abc")


def test_low_resolution_metadata_path_disambiguates_duplicate_record_ids(tmp_path):
    resolve_record = getattr(cohort, "resolve_ptbxl_low_resolution_record", None)
    assert callable(resolve_record)

    record_root = tmp_path / "records100"
    low_resolution = record_root / "00000" / "00008_lr"
    high_resolution = record_root / "00000" / "00008_hr"
    low_resolution.with_suffix(".hea").parent.mkdir(parents=True)
    low_resolution.with_suffix(".hea").write_text("low", encoding="utf-8")
    high_resolution.with_suffix(".hea").write_text("high", encoding="utf-8")

    resolved = resolve_record(
        record_root,
        "records100/00000/00008_lr",
    )

    assert resolved == low_resolution


def test_split_id_extraction_maps_challenge_ptbxl_aliases_and_ignores_other_sources():
    paths = [
        "/data/training/ptb-xl/g10/HR09066",
        "/data/training/PTB-XL/records100/00000/00008_lr",
        r"C:\data\PTB-XL\records100\00000\00009_lr.hea",
        "/data/other/records100/00000/00010_lr",
    ]

    assert ptbxl_split_record_ids(paths) == {"00008", "00009", "09066"}
    assert ptbxl_split_record_ids_by_source(paths) == {
        "records100": {"00008", "00009"},
        "challenge_hr_alias": {"09066"},
    }
