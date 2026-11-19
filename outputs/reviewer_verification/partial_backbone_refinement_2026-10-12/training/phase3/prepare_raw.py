"""Rebuild the frozen train/validation and follow-up signals from source files."""
from __future__ import annotations

import csv
from collections import Counter
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import wfdb

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
PROJECT = STUDY.parent
REPO = PROJECT / "TITAN-V4-Audit"
EXPANSION = PROJECT / "TITAN-V4-Expansion-2026-10-11"
BASES = {
    "ptb": PROJECT / "TITAN-V4-Improvement-2026-10-11" / "inputs" / "waveforms",
    "ptb672": PROJECT / "TITAN-V4-Improvement-2026-10-11" / "inputs" / "original672_waveforms",
    "source": PROJECT / "TITAN-V4-Specialists-2026-10-11" / "inputs" / "source",
    "external": PROJECT / "TITAN-V4-Backbone-2026-10-11" / "inputs" / "external_source",
    "new_ptb": EXPANSION / "inputs" / "ptb",
    "new_source": EXPANSION / "inputs" / "source",
    "new_js": EXPANSION / "inputs" / "js",
}
CLASSES = ["AFIB", "SB", "STACH", "NSR", "PVC", "RBBB", "LBBB", "PAC", "1AVB"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def source_path(row: dict[str, str]) -> tuple[Path, Path]:
    try:
        base = BASES[row["base"]]
    except KeyError as exc:
        raise ValueError(f"Unrecognized source base: {row.get('base')}") from exc
    record = base / row["record_path"]
    header = record.with_suffix(".hea")
    if not header.is_file():
        raise FileNotFoundError(header)
    expected = row["signal_sha256"].lower()
    matches = [p for suffix in (".dat", ".mat") if (p := record.with_suffix(suffix)).is_file()
               and digest(p) == expected]
    if len(matches) != 1:
        raise ValueError(f"Signal file/hash mismatch for {row['record_path']}: {len(matches)} matches")
    if digest(header) != row["header_sha256"].lower():
        raise ValueError(f"Header hash mismatch for {row['record_path']}")
    return record, matches[0]


def build_waveforms(rows: list[dict[str, str]]) -> tuple[np.ndarray, list[str]]:
    sys.path.insert(0, str(REPO / "src"))
    from titan_v4.reproducibility.physionet import preprocess_primary9_signal

    tensors: list[np.ndarray] = []
    verified: list[str] = []
    for index, row in enumerate(rows, 1):
        record_path, _signal_path = source_path(row)
        record = wfdb.rdrecord(str(record_path))
        tensor = preprocess_primary9_signal(record.p_signal, record.sig_name, record.fs)
        if tensor.shape != (6, 1250) or tensor.dtype != np.float32:
            raise ValueError(f"Input contract mismatch for {row['record_path']}: {tensor.shape} {tensor.dtype}")
        tensor_hash = hashlib.sha256(tensor.astype("<f4", copy=False).tobytes()).hexdigest()
        if tensor_hash != row["tensor_sha256"].lower():
            raise ValueError(f"Preprocessed tensor hash mismatch for {row['record_path']}")
        tensors.append(tensor)
        verified.append(tensor_hash)
        if index % 1000 == 0:
            print(f"Verified {index}/{len(rows)} source signals", flush=True)
    return np.stack(tensors), verified


def prepare_split(role: str) -> dict[str, object]:
    rows = read_rows(STUDY / "phase2" / f"{role}.csv")
    features = dict(np.load(STUDY / "phase2" / f"{role}.npz", allow_pickle=False))
    if len(rows) != len(features["y"]):
        raise ValueError(f"{role}: manifest and feature counts differ")
    for index, row in enumerate(rows):
        if row["ecg_id"] != str(features["ecg_id"][index]):
            raise ValueError(f"{role}: record order differs at {index}")
        if row["reference_label"] != CLASSES[int(features["y"][index])]:
            raise ValueError(f"{role}: reference label differs at {row['ecg_id']}")
    waveforms, tensor_hashes = build_waveforms(rows)
    packed = {
        "x": waveforms,
        "embedding": features["x"].astype(np.float32, copy=False),
        "y": features["y"],
        "dataset": features["dataset"],
        "ecg_id": features["ecg_id"],
        "gold": features["gold"],
        "tensor_sha256": np.asarray(tensor_hashes),
    }
    if role == "train":
        packed["teacher"] = features["teacher"]
    path = HERE / f"{role}.npz"
    if path.exists():
        raise FileExistsError(f"Refusing to replace {path}")
    np.savez_compressed(path, **packed)
    shutil.copy2(STUDY / "phase2" / f"{role}.csv", HERE / f"{role}.csv")
    return {"records": len(rows), "datasets": dict(sorted(Counter(r["dataset"] for r in rows).items())),
            "tensor_hashes_verified": len(tensor_hashes), "npz_sha256": digest(path)}


def prepare_evaluation() -> dict[str, object]:
    rows: list[dict[str, str]] = []
    for path in sorted((STUDY / "evaluation").glob("*_predictions.csv")):
        for row in read_rows(path):
            rows.append({"cohort": path.stem.removesuffix("_predictions"), **row})
    if len(rows) != 2171:
        raise ValueError(f"Expected 2,171 frozen follow-up records, found {len(rows)}")
    waveforms, tensor_hashes = build_waveforms(rows)
    labels = [row["reference_label"] for row in rows]
    if any(label not in CLASSES for label in labels):
        raise ValueError("Evaluation contains a label outside Primary-9")
    packed = {
        "x": waveforms,
        "y": np.asarray([CLASSES.index(label) for label in labels], dtype=np.int64),
        "dataset": np.asarray([row["dataset"] for row in rows]),
        "cohort": np.asarray([row["cohort"] for row in rows]),
        "ecg_id": np.asarray([row["ecg_id"] for row in rows]),
        "gold_label": np.asarray([CLASSES.index(row["gold_label"]) for row in rows], dtype=np.int64),
        "phase2_label": np.asarray([CLASSES.index(row["selected_label"]) for row in rows], dtype=np.int64),
        "previous_partial_label": np.asarray([CLASSES.index(row["previous_partial_label"]) for row in rows], dtype=np.int64),
        "tensor_sha256": np.asarray(tensor_hashes),
    }
    path = HERE / "evaluation.npz"
    if path.exists():
        raise FileExistsError(f"Refusing to replace {path}")
    np.savez_compressed(path, **packed)
    with (HERE / "evaluation.csv").open("x", newline="", encoding="utf-8") as handle:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return {"records": len(rows), "cohorts": dict(sorted(Counter(r["cohort"] for r in rows).items())),
            "tensor_hashes_verified": len(tensor_hashes), "npz_sha256": digest(path)}


def main() -> None:
    if any((HERE / name).exists() for name in ("train.npz", "validation.npz", "evaluation.npz")):
        raise FileExistsError("Phase-3 data already exists; refusing to overwrite")
    train = prepare_split("train")
    validation = prepare_split("validation")
    evaluation = prepare_evaluation()
    report = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_preprocessor": "TITAN-V4-Audit/src/titan_v4/reproducibility/physionet.py:preprocess_primary9_signal",
        "input_contract": {"leads": ["I", "II", "III", "aVR", "aVL", "aVF"], "fs_hz": 125,
                           "samples": 1250, "dtype": "float32"},
        "train": train,
        "validation": validation,
        "follow_up_only": evaluation,
        "labels_modified": False,
    }
    receipt = HERE / "raw_data_receipt.json"
    with receipt.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
