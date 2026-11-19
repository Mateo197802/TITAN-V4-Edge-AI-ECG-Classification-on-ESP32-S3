"""Patient-disjoint calibration and holdout cohorts for the CEDIA pathology run."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence


_PTBXL_RECORD_RE = re.compile(r"^(\d{5})(?:_(?:lr|hr))?$", re.IGNORECASE)


@dataclass(frozen=True)
class PathologyCohorts:
    calibration_record_ids: tuple[str, ...]
    test_record_ids: tuple[str, ...]
    calibration_patient_count: int
    test_patient_count: int
    excluded_counts: dict[str, int]


def normalize_ptbxl_record_id(value: str) -> str:
    """Normalize PTB-XL ECG IDs and WFDB record paths to five-digit IDs."""

    basename = str(value).strip().replace("\\", "/").rsplit("/", 1)[-1]
    if basename.lower().endswith(".hea"):
        basename = basename[:-4]
    match = _PTBXL_RECORD_RE.fullmatch(basename)
    if not match:
        raise ValueError(f"Invalid PTB-XL record identifier: {value!r}")
    return match.group(1)


def resolve_ptbxl_low_resolution_record(record_root: Path, filename_lr: str) -> Path:
    """Resolve the exact WFDB base path named by PTB-XL's ``filename_lr``."""

    parts = PurePosixPath(str(filename_lr).strip().replace("\\", "/")).parts
    records100_index = next(
        (index for index, part in enumerate(parts) if part.lower() == "records100"),
        None,
    )
    if records100_index is None or records100_index == len(parts) - 1:
        raise ValueError(f"PTB-XL filename_lr must point inside records100: {filename_lr!r}")

    record_base = Path(record_root).joinpath(*parts[records100_index + 1 :])
    if not record_base.name.lower().endswith("_lr"):
        raise ValueError(f"PTB-XL filename_lr must identify a low-resolution record: {filename_lr!r}")
    normalize_ptbxl_record_id(record_base.name)
    if not record_base.with_suffix(".hea").is_file():
        raise FileNotFoundError(record_base.with_suffix(".hea"))
    return record_base


def ptbxl_split_record_ids_by_source(paths: Iterable[str]) -> dict[str, set[str]]:
    """Resolve numeric PTB-XL paths and Challenge ``HR#####`` aliases separately."""

    record_ids = {"records100": set(), "challenge_hr_alias": set()}
    for raw_path in paths:
        parts = PurePosixPath(str(raw_path).strip().replace("\\", "/")).parts
        lowered = tuple(part.lower() for part in parts)
        if "ptb-xl" not in lowered:
            continue
        if "records100" in lowered:
            record_ids["records100"].add(normalize_ptbxl_record_id(parts[-1]))
            continue
        alias = re.fullmatch(r"HR(\d{5})(?:\.hea)?", parts[-1], flags=re.IGNORECASE)
        if alias:
            record_ids["challenge_hr_alias"].add(alias.group(1))
    return record_ids


def ptbxl_split_record_ids(paths: Iterable[str]) -> set[str]:
    """Extract all PTB-XL split IDs, including Challenge ``HR#####`` aliases."""

    ids_by_source = ptbxl_split_record_ids_by_source(paths)
    return ids_by_source["records100"] | ids_by_source["challenge_hr_alias"]


def build_patient_disjoint_cohorts(
    labels_by_record: Mapping[str, Sequence[int]],
    *,
    train_record_ids: Iterable[str],
    validation_record_ids: Iterable[str],
    patient_by_record: Mapping[str, str | int],
    num_classes: int = 10,
) -> PathologyCohorts:
    """Use validation patients for calibration and unseen patients for test.

    The calibration cohort contains labeled validation records whose patients do
    not occur in training. The test cohort contains labeled records absent from
    both checkpoint splits and whose patients occur in neither split.
    """

    normalized_labels: dict[str, tuple[int, ...]] = {}
    for raw_record_id, raw_labels in labels_by_record.items():
        record_id = normalize_ptbxl_record_id(raw_record_id)
        labels = tuple(int(label) for label in raw_labels)
        if any(label < 0 or label >= num_classes for label in labels):
            raise ValueError(f"Pathology class index out of range for {record_id}")
        if len(set(labels)) != len(labels):
            raise ValueError(f"Duplicate pathology class index for {record_id}")
        if record_id in normalized_labels and normalized_labels[record_id] != labels:
            raise ValueError(f"Conflicting pathology labels for {record_id}")
        normalized_labels[record_id] = labels

    train_ids = {normalize_ptbxl_record_id(record_id) for record_id in train_record_ids}
    validation_ids = {normalize_ptbxl_record_id(record_id) for record_id in validation_record_ids}
    overlap = train_ids & validation_ids

    normalized_patients: dict[str, str] = {}
    for raw_record_id, raw_patient_id in patient_by_record.items():
        record_id = normalize_ptbxl_record_id(raw_record_id)
        patient_id = str(raw_patient_id).strip()
        if not patient_id or patient_id.lower() == "nan":
            continue
        if record_id in normalized_patients and normalized_patients[record_id] != patient_id:
            raise ValueError(f"Conflicting patient IDs for {record_id}")
        normalized_patients[record_id] = patient_id

    needed_ids = train_ids | validation_ids | set(normalized_labels)
    missing = needed_ids - normalized_patients.keys()
    if missing:
        raise ValueError(f"PTB-XL patient metadata missing patient IDs for {len(missing)} records")

    train_patients = {normalized_patients[record_id] for record_id in train_ids}
    validation_patients = {normalized_patients[record_id] for record_id in validation_ids}
    labeled_ids = set(normalized_labels)

    calibration_candidates = labeled_ids & validation_ids
    calibration_ids = tuple(
        sorted(
            record_id
            for record_id in calibration_candidates
            if normalized_patients[record_id] not in train_patients
        )
    )

    test_candidates = labeled_ids - train_ids - validation_ids
    excluded_test_ids = {
        record_id
        for record_id in test_candidates
        if normalized_patients[record_id] in train_patients | validation_patients
    }
    test_ids = tuple(sorted(test_candidates - excluded_test_ids))

    if not calibration_ids or not test_ids:
        raise ValueError("Both patient-disjoint calibration and test cohorts must be non-empty")

    calibration_patients = {normalized_patients[record_id] for record_id in calibration_ids}
    test_patients = {normalized_patients[record_id] for record_id in test_ids}
    if calibration_patients & test_patients:
        raise ValueError("Calibration and test patients overlap")

    return PathologyCohorts(
        calibration_record_ids=calibration_ids,
        test_record_ids=test_ids,
        calibration_patient_count=len(calibration_patients),
        test_patient_count=len(test_patients),
        excluded_counts={
            "labeled_training_records": len(labeled_ids & train_ids),
            "checkpoint_train_validation_record_id_overlap": len(overlap),
            "calibration_patient_overlap_with_training": len(calibration_candidates) - len(calibration_ids),
            "test_patient_overlap_with_training_or_validation": len(excluded_test_ids),
        },
    )
