from __future__ import annotations

import hashlib
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from math import gcd
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen

import numpy as np
from scipy.signal import butter, lfilter, resample_poly


PHYSIONET_RELEASE = "1.0.3"
PHYSIONET_RELEASE_BASE = f"https://physionet.org/files/challenge-2021/{PHYSIONET_RELEASE}"
PHYSIONET_TRAINING_BASE = f"{PHYSIONET_RELEASE_BASE}/training"
PHYSIONET_RELEASE_DOI = "10.13026/34va-7q14"
PHYSIONET_DATA_LICENSE = "CC BY 4.0"
TARGET_FS = 125
TARGET_SAMPLES = 1250
FILTER_CONTEXT_SECONDS = 2
FRONTAL_LEADS = ("I", "II", "III", "AVR", "AVL", "AVF")

SOURCE_DATASETS = {
    "training/cpsc_2018": ("cpsc_2018", re.compile(r"^A\d+$", re.IGNORECASE)),
    "training/cpsc_2018_extra": ("cpsc_2018_extra", re.compile(r"^Q\d+$", re.IGNORECASE)),
    "training/georgia": ("georgia", re.compile(r"^E\d+$", re.IGNORECASE)),
    "training/ptb-xl": ("ptb-xl", re.compile(r"^HR\d+$", re.IGNORECASE)),
    "training/chapman_shaoxing": ("chapman_shaoxing", re.compile(r"^JS\d+$", re.IGNORECASE)),
    "training/ningbo": ("ningbo", re.compile(r"^JS\d+$", re.IGNORECASE)),
}
DATA_TEST_PREFIXES = (
    (re.compile(r"^A\d+$", re.IGNORECASE), "cpsc_2018"),
    (re.compile(r"^E\d+$", re.IGNORECASE), "georgia"),
    (re.compile(r"^HR\d+$", re.IGNORECASE), "ptb-xl"),
)


class _DirectoryLinks(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        href = dict(attrs).get("href")
        if href:
            self.hrefs.append(href)


def dataset_for_record(source_label: str, record_id: str) -> str:
    """Resolve a label-table source to the exact dataset family using its record ID."""
    source = str(source_label or "").strip().casefold()
    record = str(record_id or "").strip().upper()
    if not record:
        raise ValueError("Empty record_id in external validation labels")

    if source == "data_test":
        for pattern, dataset in DATA_TEST_PREFIXES:
            if pattern.fullmatch(record):
                return dataset
        raise ValueError(f"Cannot resolve data_test source from record ID {record!r}")

    entry = SOURCE_DATASETS.get(source)
    if entry is None:
        raise ValueError(f"Unsupported validation source {source_label!r}")
    dataset, pattern = entry
    if not pattern.fullmatch(record):
        raise ValueError(f"Record ID {record!r} does not match source {source_label!r}")
    return dataset


def resolve_indexed_records(
    dataset: str,
    record_ids: Iterable[str],
    group_indexes: Mapping[str, Iterable[str]],
) -> dict[str, str]:
    """Resolve record IDs only through official PhysioNet RECORDS indexes."""
    if dataset not in {value[0] for value in SOURCE_DATASETS.values()}:
        raise ValueError(f"Unsupported PhysioNet dataset directory {dataset!r}")

    requested = [str(record_id).strip().upper() for record_id in record_ids]
    if any(not record_id for record_id in requested):
        raise ValueError("Record IDs must not be empty")
    targets = set(requested)
    found: dict[str, str] = {}

    def group_order(name: str) -> tuple[int, str]:
        match = re.fullmatch(r"g(\d+)", name.casefold())
        if not match:
            raise ValueError(f"Invalid PhysioNet group name {name!r}")
        return int(match.group(1)), name

    for group in sorted(group_indexes, key=group_order):
        group_order(group)
        for raw_record in group_indexes[group]:
            record_id = PurePosixPath(str(raw_record).strip()).stem.upper()
            if record_id not in targets:
                continue
            relative_path = f"training/{dataset}/{group}/{record_id}"
            previous = found.get(record_id)
            if previous is not None and previous != relative_path:
                raise ValueError(
                    f"Record {record_id} appears in multiple official record indexes: "
                    f"{previous} and {relative_path}"
                )
            found[record_id] = relative_path

    missing = sorted(targets - set(found))
    if missing:
        raise ValueError(
            f"Records not present in the official record indexes for {dataset}: {', '.join(missing)}"
        )
    return {record_id: found[record_id] for record_id in dict.fromkeys(requested)}


def _http_bytes(url: str, *, timeout: int = 45, retries: int = 3) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": "TITAN-V4-reproducibility/1.0"})
            with urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise OSError(f"HTTP {response.status} for {url}")
                return response.read()
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(0.5 * (attempt + 1))
    raise OSError(f"Could not retrieve {url}: {last_error}")


def fetch_record_indexes(dataset: str, *, workers: int = 8) -> dict[str, list[str]]:
    """Fetch all official RECORDS lists for one dataset in the pinned release."""
    if dataset not in {value[0] for value in SOURCE_DATASETS.values()}:
        raise ValueError(f"Unsupported PhysioNet dataset directory {dataset!r}")
    source_url = f"{PHYSIONET_TRAINING_BASE}/{dataset}/"
    parser = _DirectoryLinks()
    parser.feed(_http_bytes(source_url).decode("utf-8"))
    groups = sorted(
        {href.strip("/") for href in parser.hrefs if re.fullmatch(r"g\d+/?", href)},
        key=lambda name: int(name[1:]),
    )
    if not groups:
        raise ValueError(f"No group directories found at {source_url}")

    def load_group(group: str) -> tuple[str, list[str]]:
        index_url = f"{source_url}{group}/RECORDS"
        body = _http_bytes(index_url).decode("utf-8-sig")
        records = [line.strip() for line in body.splitlines() if line.strip()]
        return group, records

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        results = list(executor.map(load_group, groups))
    return dict(results)


def build_record_source_rows(
    label_rows: Iterable[Mapping[str, str]], *, workers: int = 8
) -> list[dict[str, str]]:
    rows = [dict(row) for row in label_rows]
    targets: dict[str, list[str]] = {}
    datasets_by_row: list[str] = []
    for row in rows:
        dataset = dataset_for_record(row.get("source", ""), row.get("record_id", ""))
        datasets_by_row.append(dataset)
        targets.setdefault(dataset, []).append(row["record_id"].strip().upper())

    resolved: dict[str, dict[str, str]] = {}
    for dataset, record_ids in targets.items():
        indexes = fetch_record_indexes(dataset, workers=workers)
        resolved[dataset] = resolve_indexed_records(dataset, record_ids, indexes)

    output: list[dict[str, str]] = []
    for row, dataset in zip(rows, datasets_by_row, strict=True):
        record_id = row["record_id"].strip().upper()
        output.append(
            {
                "record_id": record_id,
                "source_label": row["source"].strip(),
                "dataset": dataset,
                "physionet_release": PHYSIONET_RELEASE,
                "upstream_partition": "training",
                "record_path": resolved[dataset][record_id],
                "release_doi": PHYSIONET_RELEASE_DOI,
                "data_license": PHYSIONET_DATA_LICENSE,
                "resolution_basis": (
                    "record_id_prefix_and_official_RECORDS_index"
                    if row["source"].strip().casefold() == "data_test"
                    else "source_field_and_official_RECORDS_index"
                ),
            }
        )
    if len(output) != len(rows):
        raise AssertionError("Source manifest does not preserve all label rows")
    return output


def _safe_record_path(record_path: str) -> PurePosixPath:
    path = PurePosixPath(record_path)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"Unsafe source record path {record_path!r}")
    if len(path.parts) != 4 or path.parts[0] != "training" or not re.fullmatch(r"g\d+", path.parts[2]):
        raise ValueError(f"Invalid source record path {record_path!r}")
    dataset = path.parts[1]
    if dataset not in {value[0] for value in SOURCE_DATASETS.values()}:
        raise ValueError(f"Unsupported dataset path {record_path!r}")
    return path


def _write_download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    payload = _http_bytes(url)
    temporary.write_bytes(payload)
    os.replace(temporary, target)


def ensure_local_wfdb_header(
    record_path: str,
    data_root: str | Path,
    *,
    download: bool = True,
) -> tuple[Path, str]:
    """Ensure a record header exists and matches its indexed WFDB record ID."""
    relative = _safe_record_path(record_path)
    base = Path(data_root).joinpath(*relative.parts)
    header_path = base.with_suffix(".hea")
    if not header_path.is_file():
        if not download:
            raise FileNotFoundError(f"Missing local WFDB header for {record_path}")
        url = f"{PHYSIONET_RELEASE_BASE}/{quote(record_path, safe='/')}.hea"
        _write_download(url, header_path)

    header_text = header_path.read_text(encoding="ascii", errors="strict")
    first_line = next((line.strip() for line in header_text.splitlines() if line.strip()), "")
    header_id = first_line.split()[0] if first_line else ""
    if header_id.casefold() != relative.name.casefold():
        raise ValueError(
            f"WFDB header ID {header_id!r} does not match expected {relative.name!r}"
        )

    digest = hashlib.sha256(header_path.read_bytes()).hexdigest().upper()
    return base, digest


def ensure_local_wfdb_record(
    record_path: str,
    data_root: str | Path,
    *,
    download: bool = True,
) -> tuple[Path, dict[str, str]]:
    """Ensure one record's WFDB header/signal files exist; return base path and SHA-256s."""
    relative = _safe_record_path(record_path)
    base, _header_digest = ensure_local_wfdb_header(record_path, data_root, download=download)
    header_path = base.with_suffix(".hea")
    header_text = header_path.read_text(encoding="ascii", errors="strict")

    signal_files: set[PurePosixPath] = set()
    for line in header_text.splitlines()[1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        filename = stripped.split()[0]
        if filename == "~":
            continue
        file_path = PurePosixPath(filename)
        if file_path.is_absolute() or any(part in {"", ".", ".."} for part in file_path.parts):
            raise ValueError(f"Unsafe WFDB signal filename {filename!r} in {record_path}")
        signal_files.add(file_path)
    if not signal_files:
        raise ValueError(f"WFDB header has no signal files: {record_path}")

    paths_to_hash = [header_path]
    for signal_file in sorted(signal_files):
        target = base.parent.joinpath(*signal_file.parts)
        if not target.is_file():
            if not download:
                raise FileNotFoundError(f"Missing WFDB signal file {target.name} for {record_path}")
            url = urljoin(
                f"{PHYSIONET_RELEASE_BASE}/{quote(record_path, safe='/')}.hea",
                quote(signal_file.as_posix(), safe="/"),
            )
            _write_download(url, target)
        paths_to_hash.append(target)

    hashes: dict[str, str] = {}
    for path in paths_to_hash:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[path.relative_to(Path(data_root)).as_posix()] = digest.hexdigest().upper()
    return base, hashes


def preprocess_primary9_signal(
    signals: np.ndarray,
    signal_names: Iterable[str],
    sample_rate: float,
) -> np.ndarray:
    """Apply the frozen first-window 6-lead, 125-Hz preprocessing contract."""
    raw = np.asarray(signals)
    names = [str(name).strip() for name in signal_names]
    if raw.ndim != 2 or raw.shape[1] != len(names) or raw.shape[0] == 0:
        raise ValueError("Signals must be [samples, leads] and match signal_names")
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive finite number")

    normalized_names = ["".join(ch for ch in name.upper() if ch.isalnum()) for name in names]
    lead_indices: list[int] = []
    missing: list[str] = []
    for lead in FRONTAL_LEADS:
        aliases = {lead}
        index = next((i for i, name in enumerate(normalized_names) if name in aliases), None)
        if index is None:
            missing.append(lead)
        else:
            lead_indices.append(index)
    if missing or len(set(lead_indices)) != len(FRONTAL_LEADS):
        missing_names = missing or list(FRONTAL_LEADS)
        raise ValueError(f"Missing required frontal leads: {', '.join(missing_names)}")

    fs = float(sample_rate)
    context_samples = int(FILTER_CONTEXT_SECONDS * fs)
    requested_samples = int((TARGET_SAMPLES / TARGET_FS + FILTER_CONTEXT_SECONDS) * fs)
    selected = np.nan_to_num(
        raw[:requested_samples, lead_indices].T.astype(np.float32, copy=True),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    pad_right = max(0, requested_samples - selected.shape[1])
    selected = np.pad(selected, ((0, 0), (context_samples, pad_right)), mode="constant")

    nyquist = fs / 2.0
    high = min(45.0 / nyquist, 0.99)
    low = 0.5 / nyquist
    if not 0 < low < high < 1:
        raise ValueError(f"Unsupported sampling frequency for 0.5-45 Hz filtering: {sample_rate}")
    b, a = butter(3, [low, high], btype="band")
    for channel in range(selected.shape[0]):
        if np.std(selected[channel]) > 1e-6:
            selected[channel] = lfilter(b, a, selected[channel])

    rate_fraction = __import__("fractions").Fraction(str(sample_rate)).limit_denominator(10000)
    up = TARGET_FS * rate_fraction.denominator
    down = rate_fraction.numerator
    divisor = gcd(up, down)
    up //= divisor
    down //= divisor
    if up != 1 or down != 1:
        selected = resample_poly(selected, up, down, axis=1)

    crop_start = int(round(context_samples * TARGET_FS / fs))
    window = selected[:, crop_start : crop_start + TARGET_SAMPLES]
    if window.shape[1] < TARGET_SAMPLES:
        window = np.pad(window, ((0, 0), (0, TARGET_SAMPLES - window.shape[1])), mode="constant")
    else:
        window = window[:, :TARGET_SAMPLES]

    mean = window.mean(axis=1, keepdims=True)
    std = window.std(axis=1, keepdims=True)
    std[std < 1e-6] = 1.0
    window = (window - mean) / std
    return np.nan_to_num(window, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
