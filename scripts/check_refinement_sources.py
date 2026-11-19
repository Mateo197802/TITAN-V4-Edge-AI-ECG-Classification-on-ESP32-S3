"""Check record hashes against official release indexes and pin public manifests."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from titan_v4.reproducibility.physionet import _http_bytes
from titan_v4.reproducibility.refinement import RELEASES, source_release

BUNDLE = ROOT / "outputs/reviewer_verification/partial_backbone_refinement_2026-10-12"


def fetch_index(url):
    with urlopen(Request(url, headers={"Range": "bytes=0-0"}), timeout=15) as response:
        match = re.fullmatch(r"bytes 0-0/(\d+)", response.headers.get("Content-Range", ""))
        response.read(1)
        if response.status != 206 or match is None:
            return _http_bytes(url, timeout=15, retries=2)
        length = int(match.group(1))
    chunk_size = 512 * 1024
    ranges = [(start, min(length - 1, start + chunk_size - 1)) for start in range(0, length, chunk_size)]
    def fetch(bounds):
        start, end = bounds
        for attempt in range(2):
            try:
                with urlopen(Request(url, headers={"Range": f"bytes={start}-{end}"}), timeout=15) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{length}":
                        raise ValueError("Upstream returned an unexpected byte range")
                    payload = response.read()
                    if len(payload) != end - start + 1:
                        raise ValueError("Incomplete upstream byte range")
                    return start, payload
            except OSError:
                if attempt:
                    raise
    parts = {}
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(fetch, bounds) for bounds in ranges]
        for future in as_completed(futures):
            start, payload = future.result()
            parts[start] = payload
            print(f"Verified index byte ranges: {len(parts)}/{len(ranges)}", flush=True)
    payload = b"".join(parts[start] for start, _end in ranges)
    if len(payload) != length:
        raise ValueError("Incomplete upstream hash index")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--index-root", type=Path, default=ROOT / "data/cache/refinement_source_indexes")
    parser.add_argument("--refresh-release", choices=tuple(RELEASES), action="append", default=[])
    args = parser.parse_args()
    indexes = {}
    receipts = {}
    for release, (version, doi) in RELEASES.items():
        url = f"https://physionet.org/files/{release}/{version}/SHA256SUMS.txt"
        cached = args.index_root / release / version / "SHA256SUMS.txt"
        capture = cached.with_name("capture.json")
        print(f"Checking official hash index: {release} {version}", flush=True)
        if cached.is_file() and release not in args.refresh_release:
            payload = cached.read_bytes()
        else:
            payload = fetch_index(url)
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(payload)
        index = {}
        for line in payload.decode("utf-8-sig").splitlines():
            if not line.strip():
                continue
            digest, name = line.split(None, 1)
            if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
                raise ValueError("Malformed upstream checksum entry")
            name = name.strip().lstrip("*")
            if name.startswith("./"):
                name = name[2:]
            index[name] = digest.lower()
        indexes[release] = index
        receipts[release] = {"version": version, "doi": doi, "index_url": url,
                             "index_sha256": hashlib.sha256(payload).hexdigest(), "matched_records": 0}
        if capture.is_file() and release not in args.refresh_release:
            provenance = json.loads(capture.read_text(encoding="utf-8"))
            if (provenance["url"] != url
                or provenance["captured_sha256"] != hashlib.sha256(payload).hexdigest()
                or provenance["captured_bytes"] != len(payload)):
                raise ValueError("Archived official index capture differs from its provenance")
            receipts[release]["index_capture"] = provenance
    total = 0
    for name in ("train_records.csv", "validation_records.csv", "followup_records.csv"):
        path = BUNDLE / "manifests" / name
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fields = list(reader.fieldnames)
        for row in rows:
            release = source_release(row)
            index = indexes[release]
            record = row["record_path"]
            if index.get(record + ".hea") != row["header_sha256"].lower():
                raise ValueError(f"Official header hash mismatch: {record}")
            signal_names = [record + ext for ext in (".dat", ".mat")
                            if index.get(record + ext) == row["signal_sha256"].lower()]
            if len(signal_names) != 1:
                raise ValueError(f"Official signal hash mismatch: {record}")
            version, doi = RELEASES[release]
            pins = {"source_release": release, "version": version, "doi": doi,
                    "signal_filename": PurePosixPath(signal_names[0]).name}
            if args.write:
                row.update(pins)
            elif any(row.get(key) != value for key, value in pins.items()):
                raise ValueError(f"Manifest pin differs: {record}")
            receipts[release]["matched_records"] += 1
            total += 1
        if args.write:
            fields = list(dict.fromkeys([*fields, "source_release", "signal_filename"]))
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(rows)
    if args.write:
        receipt = {"verified_utc": datetime.now(timezone.utc).isoformat(), "records": total, "releases": receipts}
        (BUNDLE / "manifests/official_source_hash_verification.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"verified_records": total, "releases": receipts}, indent=2))


if __name__ == "__main__":
    main()
