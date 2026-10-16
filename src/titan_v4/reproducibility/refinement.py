"""Read versioned refinement records while enforcing their frozen file/tensor hashes."""
from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path, PurePosixPath
import tempfile
from urllib.parse import quote

import numpy as np

from titan_v4.reproducibility.physionet import _http_bytes, preprocess_primary9_signal


RELEASES = {
    "challenge-2021": ("1.0.3", "10.13026/34va-7q14"),
    "ptb-xl": ("1.0.3", "10.13026/kfzx-aw45"),
    "ecg-arrhythmia": ("1.0.0", "10.13026/wgex-er52"),
}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def source_release(row: dict[str, str]) -> str:
    path = PurePosixPath(row["record_path"])
    if path.is_absolute() or ".." in path.parts or "\\" in row["record_path"] or path.suffix:
        raise ValueError("Unsafe record path")
    if not path.parts:
        raise ValueError("Empty record path")
    if path.parts[0] == "training" and len(path.parts) == 4:
        if path.parts[1] != row["dataset"] or row["dataset"] not in {"cpsc_2018", "georgia"}:
            raise ValueError("Challenge source/path mismatch")
        return "challenge-2021"
    if path.parts[0] == "records500" and len(path.parts) == 3 and row["dataset"] == "ptb-xl":
        return "ptb-xl"
    if path.parts[0] == "WFDBRecords" and len(path.parts) == 4 and row["dataset"] == "ecg-arrhythmia":
        return "ecg-arrhythmia"
    raise ValueError("Unsupported refinement source/path")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_file(target: Path, url: str, expected: str, download: bool) -> None:
    if target.is_file():
        if sha256(target) != expected.lower():
            raise ValueError(f"Source hash mismatch: {target.name}")
        return
    if not download:
        raise FileNotFoundError(target)
    payload = _http_bytes(url)
    if hashlib.sha256(payload).hexdigest() != expected.lower():
        raise ValueError(f"Downloaded source hash mismatch: {target.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def ensure_record(row: dict[str, str], data_root: Path, *, download: bool = True) -> Path:
    release = source_release(row)
    version, doi = RELEASES[release]
    if row.get("source_release") != release or row.get("version") != version or row.get("doi") != doi:
        raise ValueError("Source manifest release/version/DOI is not pinned")
    relative = PurePosixPath(row["record_path"])
    signal = PurePosixPath(row["signal_filename"])
    if signal.name != str(signal) or signal.stem != relative.name or signal.suffix not in {".mat", ".dat"}:
        raise ValueError("Unsafe or mismatched signal filename")
    root = (data_root / release / version).resolve()
    base = root.joinpath(*relative.parts)
    if not base.resolve().is_relative_to(root):
        raise ValueError("Record path escapes the source cache")
    url = f"https://physionet.org/files/{release}/{version}/"
    _ensure_file(base.with_suffix(".hea"), url + quote(row["record_path"], safe="/") + ".hea",
                 row["header_sha256"], download)
    signal_path = base.parent / signal.name
    signal_relative = relative.parent / signal.name
    _ensure_file(signal_path, url + quote(signal_relative.as_posix(), safe="/"), row["signal_sha256"], download)
    return base


def load_tensor(row: dict[str, str], data_root: Path, *, download: bool = True) -> np.ndarray:
    import wfdb

    base = ensure_record(row, data_root, download=download)
    header = wfdb.rdheader(str(base))
    if set(header.file_name) != {row["signal_filename"]}:
        raise ValueError("WFDB header refers to an unmanifested signal file")
    record = wfdb.rdrecord(str(base), physical=True)
    tensor = preprocess_primary9_signal(record.p_signal, record.sig_name, record.fs)
    digest = hashlib.sha256(np.asarray(tensor, dtype="<f4").tobytes()).hexdigest()
    if digest != row["tensor_sha256"].lower():
        raise ValueError(f"Frozen preprocessing hash differs for {row['ecg_id']}")
    return tensor
