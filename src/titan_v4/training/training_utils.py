"""
Shared training and evaluation utilities for TITAN V4.

The key invariant is record-level separation: all sliding windows generated
from the same WFDB record must live in exactly one split.
"""
from __future__ import annotations

import collections
import csv
import glob
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from label_registry import (
    NUM_RHYTHM_CLASSES,
    RHYTHM_CLASS_NAMES,
    exclusion_reason_for_codes,
    normalize_record_path,
    parse_dx_codes_from_header,
    primary_label_from_codes,
)

DEFAULT_UNKNOWN_POLICY = "skip"


@dataclass(frozen=True)
class RecordSplit:
    train_records: List[str]
    val_records: List[str]
    record_labels: Dict[str, int]
    excluded_records: Dict[str, str]
    train_label_counts: Dict[int, int]
    val_label_counts: Dict[int, int]
    val_fraction: float
    seed: int


def parse_dx_codes(record_base: str) -> List[str]:
    return parse_dx_codes_from_header(record_base)


def extract_known_labels(record_base: str, mapping: Mapping[str, int]) -> List[int]:
    labels = []
    for code in parse_dx_codes(record_base):
        if code in mapping:
            labels.append(int(mapping[code]))
    return labels


def _resolve_virtual_source(sidecar_path: str, source_record: str) -> str:
    if os.path.isabs(source_record):
        return source_record
    sidecar_dir = os.path.dirname(sidecar_path)
    data_root = os.path.dirname(sidecar_dir) if os.path.basename(sidecar_dir) == "virtual_windows" else sidecar_dir
    return os.path.join(data_root, source_record)


def virtual_window_key(source_record: str, window_start: int) -> str:
    return f"virtual::{normalize_record_path(source_record)}|w:{int(window_start)}"


def load_virtual_window_label_maps(data_dirs: Sequence[str]) -> tuple[Dict[str, int], Dict[str, str]]:
    virtual_labels: Dict[str, int] = {}
    virtual_groups: Dict[str, str] = {}
    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            continue
        pattern = os.path.join(data_dir, "**", "*_window_label_map.json")
        for sidecar_path in glob.glob(pattern, recursive=True):
            try:
                with open(sidecar_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            windows = payload.get("windows", {})
            if not isinstance(windows, dict):
                continue
            for _window_id, meta in windows.items():
                if not isinstance(meta, dict) or "label" not in meta:
                    continue
                source_record = meta.get("source_record")
                if not source_record:
                    continue
                label = int(meta["label"])
                if label < 0 or label >= NUM_RHYTHM_CLASSES:
                    continue
                class_name = meta.get("class_name")
                if class_name is not None and str(class_name) != RHYTHM_CLASS_NAMES[label]:
                    continue
                source_path = _resolve_virtual_source(sidecar_path, str(source_record))
                window_start = int(meta.get("window_start", 0))
                key = virtual_window_key(source_path, window_start)
                virtual_labels[key] = label
                group_id = meta.get("group_id") or normalize_record_path(source_path)
                virtual_groups[key] = str(group_id)
    return virtual_labels, virtual_groups


def load_sidecar_label_maps(data_dirs: Sequence[str]) -> Dict[str, int]:
    sidecar_labels: Dict[str, int] = {}
    sidecar_groups: Dict[str, str] = {}
    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            continue
        pattern = os.path.join(data_dir, "**", "*_label_map.json")
        for sidecar_path in glob.glob(pattern, recursive=True):
            sidecar_root = os.path.dirname(sidecar_path)
            try:
                with open(sidecar_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            records = payload.get("records", {})
            if not isinstance(records, dict):
                continue
            for relative_record, meta in records.items():
                if not isinstance(meta, dict) or "label" not in meta:
                    continue
                label = int(meta["label"])
                if label < 0 or label >= NUM_RHYTHM_CLASSES:
                    continue
                class_name = meta.get("class_name")
                if class_name is not None and str(class_name) != RHYTHM_CLASS_NAMES[label]:
                    continue
                key = normalize_record_path(os.path.join(sidecar_root, relative_record))
                sidecar_labels[key] = label
                group_id = meta.get("group_id")
                if group_id:
                    sidecar_groups[key] = str(group_id)
                    source_record = meta.get("source_record")
                    if source_record:
                        sidecar_groups[normalize_record_path(str(source_record))] = str(group_id)
    return sidecar_labels, sidecar_groups


def select_primary_label(
    record_base: str,
    mapping: Mapping[str, int],
    unknown_policy: str = DEFAULT_UNKNOWN_POLICY,
) -> int | None:
    codes = parse_dx_codes(record_base)
    label = primary_label_from_codes(codes, mapping=mapping)
    if label is not None:
        return int(label)
    if unknown_policy == "skip":
        return None
    raise ValueError(f"unknown_policy no soportada: {unknown_policy}")


def discover_record_labels(
    data_dirs: Sequence[str],
    mapping: Mapping[str, int],
    unknown_policy: str = DEFAULT_UNKNOWN_POLICY,
) -> tuple[Dict[str, int], Dict[str, str]]:
    record_labels: Dict[str, int] = {}
    excluded_records: Dict[str, str] = {}
    sidecar_labels, _sidecar_groups = load_sidecar_label_maps(data_dirs)
    virtual_labels, _virtual_groups = load_virtual_window_label_maps(data_dirs)
    record_labels.update(virtual_labels)
    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            continue
        pattern = os.path.join(data_dir, "**", "*.hea")
        for header in sorted(glob.glob(pattern, recursive=True)):
            record_base = header[:-4]
            key = normalize_record_path(record_base)
            if key in sidecar_labels:
                record_labels[key] = int(sidecar_labels[key])
                continue
            label = select_primary_label(record_base, mapping, unknown_policy=unknown_policy)
            if label is None:
                excluded_records[key] = exclusion_reason_for_codes(parse_dx_codes(record_base), mapping=mapping)
                continue
            record_labels[key] = int(label)
    return record_labels, excluded_records


def load_sidecar_groups(data_dirs: Sequence[str]) -> Dict[str, str]:
    """Return mapping derived_record_path -> group_id for sidecar-defined records.

    group_id is used to keep derived segments from the same original record
    together in the same split to avoid leakage.
    """
    _labels, groups = load_sidecar_label_maps(data_dirs)
    _virtual_labels, virtual_groups = load_virtual_window_label_maps(data_dirs)
    groups.update(virtual_groups)
    return groups


def _count(values: Iterable[int]) -> Dict[int, int]:
    return dict(sorted(collections.Counter(int(v) for v in values).items()))


def build_record_splits(
    data_dirs: Sequence[str],
    mapping: Mapping[str, int],
    val_fraction: float = 0.15,
    seed: int = 42,
    unknown_policy: str = DEFAULT_UNKNOWN_POLICY,
) -> RecordSplit:
    if not 0.0 < val_fraction < 0.5:
        raise ValueError("val_fraction must be > 0 and < 0.5")

    record_labels, excluded_records = discover_record_labels(data_dirs, mapping, unknown_policy)
    sidecar_groups = load_sidecar_groups(data_dirs)

    # Group-aware split: derived segment records with the same group_id must
    # stay together in the same split to prevent leakage.
    group_to_records: Dict[str, List[str]] = collections.defaultdict(list)
    record_to_group: Dict[str, str] = {}
    for record in record_labels:
        gid = sidecar_groups.get(record, record)
        record_to_group[record] = gid
        group_to_records[gid].append(record)

    group_label: Dict[str, int] = {}
    for gid, records in group_to_records.items():
        labels = [int(record_labels[r]) for r in records if r in record_labels]
        if not labels:
            continue
        # Virtual windows from one long record may legitimately contain several
        # rhythm labels. Split assignment stays group-aware; the majority label
        # is used only for coarse stratification of the source group.
        group_label[gid] = collections.Counter(labels).most_common(1)[0][0]

    by_label: Dict[int, List[str]] = collections.defaultdict(list)
    for gid, label in group_label.items():
        by_label[int(label)].append(gid)

    rng = np.random.default_rng(seed)
    train_records: List[str] = []
    val_records: List[str] = []

    for label in sorted(by_label):
        groups = sorted(by_label[label])
        shuffled = list(rng.permutation(groups))
        if len(shuffled) < 2:
            for gid in shuffled:
                train_records.extend(group_to_records[gid])
            continue
        n_val = max(1, int(round(len(shuffled) * val_fraction)))
        n_val = min(n_val, len(shuffled) - 1)

        # Ensure at least one derived-segment group lands in validation when present.
        # This avoids val coverage collapsing for rare rhythms represented mostly by
        # sidecar-derived segments.
        def is_derived_group(gid: str) -> bool:
            return any("segments_wfdb" in rec.lower() for rec in group_to_records.get(gid, []))

        derived_groups = [gid for gid in shuffled if is_derived_group(gid)]
        if derived_groups and not any(is_derived_group(gid) for gid in shuffled[:n_val]):
            # Prefer the derived group with the most derived records (typically the
            # longest episode), so validation gets enough windows.
            pick = max(derived_groups, key=lambda gid: len(group_to_records.get(gid, [])))
            idx = shuffled.index(pick)
            shuffled[0], shuffled[idx] = shuffled[idx], shuffled[0]

        for gid in shuffled[:n_val]:
            val_records.extend(group_to_records[gid])
        for gid in shuffled[n_val:]:
            train_records.extend(group_to_records[gid])

    return RecordSplit(
        train_records=sorted(train_records),
        val_records=sorted(val_records),
        record_labels=record_labels,
        excluded_records=excluded_records,
        train_label_counts=_count(record_labels[r] for r in train_records),
        val_label_counts=_count(record_labels[r] for r in val_records),
        val_fraction=float(val_fraction),
        seed=int(seed),
    )


def _portable_basename(raw_path: str) -> str:
    normalized = str(raw_path or "").strip().replace("\\", "/").rstrip("/")
    if not normalized:
        return ""
    return normalized.split("/")[-1]


def _resolve_manifest_record_base(row: Mapping[str, str], data_root: str | None = None) -> str:
    """Resolve a manifest row generated on Windows or Linux to this machine."""
    source = str(row.get("source", "")).strip()
    data_root_path = Path(data_root).resolve() if data_root else None
    candidates: list[Path] = []

    for raw_key in ("header_path", "record_base"):
        basename = _portable_basename(str(row.get(raw_key, "")).strip())
        if basename and source and data_root_path is not None:
            stem = Path(basename).with_suffix("").name
            candidates.append(data_root_path / source / stem)

    for raw_key in ("record_base", "header_path"):
        raw = str(row.get(raw_key, "")).strip()
        if not raw:
            continue
        path = Path(raw)
        if raw_key == "header_path" and path.suffix.lower() == ".hea":
            path = path.with_suffix("")
        candidates.append(path)

    for candidate in candidates:
        if Path(str(candidate) + ".hea").is_file():
            return normalize_record_path(str(candidate))

    if candidates:
        return normalize_record_path(str(candidates[0]))
    raise ValueError("manifest row without usable record_base/header_path")


def _read_manifest_records(manifest_csv: str, data_root: str | None) -> Dict[str, int]:
    records: Dict[str, int] = {}
    with open(manifest_csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_label = str(row.get("rhythm_label", "")).strip()
            if raw_label == "":
                continue
            label = int(float(raw_label))
            if label < 0 or label >= NUM_RHYTHM_CLASSES:
                continue
            record = _resolve_manifest_record_base(row, data_root=data_root)
            previous = records.get(record)
            if previous is not None and previous != label:
                raise ValueError(f"Conflicting rhythm labels for manifest record: {record}")
            records[record] = label
    return records


def load_record_split_from_manifests(
    train_manifest_csv: str,
    val_manifest_csv: str,
    *,
    data_root: str | None = None,
    seed: int = 42,
) -> RecordSplit:
    """Build a RecordSplit from reviewed manifest CSV files."""
    train_labels = _read_manifest_records(train_manifest_csv, data_root=data_root)
    val_labels = _read_manifest_records(val_manifest_csv, data_root=data_root)
    overlap = sorted(set(train_labels).intersection(val_labels))
    if overlap:
        preview = ", ".join(overlap[:5])
        raise ValueError(f"Manifest train/val overlap detected: {preview}")

    record_labels: Dict[str, int] = {}
    record_labels.update(train_labels)
    record_labels.update(val_labels)

    total = len(train_labels) + len(val_labels)
    val_fraction = float(len(val_labels) / total) if total else 0.0
    return RecordSplit(
        train_records=sorted(train_labels),
        val_records=sorted(val_labels),
        record_labels=record_labels,
        excluded_records={},
        train_label_counts=_count(train_labels.values()),
        val_label_counts=_count(val_labels.values()),
        val_fraction=val_fraction,
        seed=int(seed),
    )


def split_window_indices_by_record(dataset, split: RecordSplit):
    train_set = set(split.train_records)
    val_set = set(split.val_records)
    train_indices = []
    val_indices = []
    skipped = 0

    for idx, (record_base, _window_start) in enumerate(dataset.windows):
        if hasattr(dataset, "split_key_for_index"):
            key = dataset.split_key_for_index(idx)
        else:
            key = normalize_record_path(record_base)
        if key in train_set:
            train_indices.append(idx)
        elif key in val_set:
            val_indices.append(idx)
        else:
            skipped += 1

    return train_indices, val_indices, skipped


def labels_for_indices(dataset, indices: Sequence[int], record_labels: Mapping[str, int]) -> List[int]:
    labels = []
    for idx in indices:
        if hasattr(dataset, "label_for_index"):
            labels.append(int(dataset.label_for_index(idx)))
            continue
        record_base, _window_start = dataset.windows[idx]
        labels.append(int(record_labels[normalize_record_path(record_base)]))
    return labels


def compute_class_weights(
    labels: Sequence[int],
    num_classes: int,
    max_weight: float = 20.0,
    min_weight: float = 0.05,
) -> List[float]:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes)
    total = float(counts.sum())
    weights = np.zeros(num_classes, dtype=np.float32)
    if total == 0:
        return weights.tolist()

    present = counts > 0
    weights[present] = total / (present.sum() * counts[present])
    mean_present = float(weights[present].mean()) if present.any() else 1.0
    if mean_present > 0:
        weights[present] = weights[present] / mean_present
    weights[present] = np.clip(weights[present], float(min_weight), max_weight)
    return weights.tolist()


def compute_sample_weights(
    labels: Sequence[int],
    num_classes: int,
    max_weight: float = 20.0,
    min_weight: float = 0.05,
) -> List[float]:
    class_weights = compute_class_weights(
        labels,
        num_classes,
        max_weight=max_weight,
        min_weight=min_weight,
    )
    return [float(class_weights[int(label)]) for label in labels]


def compute_balanced_sample_weights(labels: Sequence[int], num_classes: int) -> List[float]:
    """Return per-sample weights that give each present class equal sampler mass."""
    labels_array = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels_array, minlength=num_classes)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = 1.0 / counts[present].astype(np.float32)
    return [float(weights[int(label)]) for label in labels_array]


def compute_tempered_sample_weights(labels: Sequence[int], num_classes: int, alpha: float = 0.5) -> List[float]:
    """Return inverse-frequency weights softened by alpha.

    alpha=0 behaves like uniform per-sample weighting.
    alpha=1 behaves like inverse-frequency class balancing.
    Values between them reduce rare-class oversampling and false positives.
    """
    labels_array = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels_array, minlength=num_classes).astype(np.float64)
    weights = np.zeros(num_classes, dtype=np.float64)
    present = counts > 0
    alpha = float(alpha)
    weights[present] = np.power(1.0 / counts[present], alpha)
    return [float(weights[int(label)]) for label in labels_array]


def save_split_metadata(split: RecordSplit, path: str, class_names: Sequence[str] | None = None) -> None:
    payload = {
        "seed": split.seed,
        "val_fraction": split.val_fraction,
        "train_records": len(split.train_records),
        "val_records": len(split.val_records),
        "excluded_records": len(split.excluded_records),
        "train_record_paths": split.train_records,
        "val_record_paths": split.val_records,
        "excluded_record_reasons": split.excluded_records,
        "train_label_counts": split.train_label_counts,
        "val_label_counts": split.val_label_counts,
    }
    if class_names:
        payload["class_names"] = list(class_names)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
