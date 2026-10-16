from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from titan_v4.reproducibility.labels import (
    parse_dx_codes_from_comments,
    primary9_label_from_dx_codes,
    primary9_names_from_dx_codes,
)
from titan_v4.reproducibility.physionet import ensure_local_wfdb_header


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "data/external_validation/validation_record_index.csv"
DEFAULT_SOURCE_MANIFEST = ROOT / "data/external_validation/record_source_manifest.csv"
DEFAULT_DATA_ROOT = ROOT / "data/cache/physionet/challenge-2021/1.0.3"
DEFAULT_OUTPUT = ROOT / "data/external_validation/final_external_validation_labels.csv"
FIELDS = ("record_id", "source", "split", "rhythm_label", "dx_codes", "dx_rhythm_label_names")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _key(row: dict[str, str], source_field: str) -> tuple[str, str]:
    return row[source_field].strip().casefold(), row["record_id"].strip().casefold()


def _render(rows: list[dict[str, str]]) -> str:
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild the 672 Primary-9 labels from the pinned PhysioNet WFDB headers."
    )
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offline", action="store_true", help="Require all headers under --data-root.")
    parser.add_argument("--check", action="store_true", help="Compare with the checked-in label table.")
    args = parser.parse_args()

    try:
        index_rows = _read_csv(args.index)
        source_rows = _read_csv(args.source_manifest)
        if len(index_rows) != 672 or len(source_rows) != 672:
            raise ValueError("The record index and source manifest must each contain exactly 672 rows")
        source_keys = [_key(row, "source_label") for row in source_rows]
        index_keys = [_key(row, "source") for row in index_rows]
        if len(set(source_keys)) != len(source_keys) or set(source_keys) != set(index_keys):
            raise ValueError("Record index and source manifest do not identify the same unique records")
        sources = {_key(row, "source_label"): row for row in source_rows}
        import wfdb

        output_rows: list[dict[str, str]] = []
        for index, row in enumerate(index_rows, start=1):
            source = sources[_key(row, "source")]
            record_base, _header_sha256 = ensure_local_wfdb_header(
                source["record_path"], args.data_root, download=not args.offline
            )
            header = wfdb.rdheader(str(record_base))
            codes = parse_dx_codes_from_comments(header.comments)
            output_rows.append(
                {
                    "record_id": row["record_id"].strip().upper(),
                    "source": row["source"].strip(),
                    "split": row["split"].strip(),
                    "rhythm_label": primary9_label_from_dx_codes(codes),
                    "dx_codes": "|".join(codes),
                    "dx_rhythm_label_names": "|".join(primary9_names_from_dx_codes(codes)),
                }
            )
            if index % 100 == 0 or index == len(index_rows):
                print(f"Labeled {index}/{len(index_rows)} records")
    except (ImportError, OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))

    counts = Counter(row["rhythm_label"] for row in output_rows)
    text = _render(output_rows)
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8-sig") != text:
            print(f"Label table differs from official headers or is missing: {args.output}")
            return 1
        print(f"Verified labels for {len(output_rows)} records against PhysioNet headers.")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8", newline="")
        print(f"Wrote {len(output_rows)} label rows to {args.output}.")
    for label, count in sorted(counts.items()):
        print(f"{label}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
