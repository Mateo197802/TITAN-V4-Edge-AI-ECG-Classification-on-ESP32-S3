from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from titan_v4.reproducibility.physionet import build_record_source_rows


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = Path("data/external_validation/validation_record_index.csv")
DEFAULT_OUTPUT = Path("data/external_validation/record_source_manifest.csv")
FIELDS = (
    "record_id",
    "source_label",
    "dataset",
    "physionet_release",
    "upstream_partition",
    "release_doi",
    "data_license",
    "record_path",
    "resolution_basis",
)


def read_label_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No validation labels found in {path}")
    required = {"record_id", "source", "split"}
    missing = sorted(required - set(rows[0]))
    if missing:
        raise ValueError(f"Validation label CSV is missing columns: {missing}")
    return rows


def render_manifest(rows: list[dict[str, str]]) -> str:
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Resolve every validation record against the pinned PhysioNet Challenge 2021 indexes."
    )
    parser.add_argument("--labels", type=Path, default=ROOT / DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=ROOT / DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--check", action="store_true", help="Compare against the checked-in manifest without writing.")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    try:
        rows = build_record_source_rows(read_label_rows(args.labels), workers=args.workers)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    manifest = render_manifest(rows)
    counts = Counter(row["dataset"] for row in rows)
    if args.check:
        if not args.output.is_file():
            print(f"Manifest is missing: {args.output}")
            return 1
        if args.output.read_text(encoding="utf-8") != manifest:
            print("The checked-in source manifest differs from the official PhysioNet record indexes.")
            return 1
        print(f"Verified {len(rows)} records against PhysioNet Challenge 2021 {rows[0]['physionet_release']}.")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(manifest, encoding="utf-8", newline="")
        print(f"Wrote {len(rows)} source mappings to {args.output}.")
    for dataset, count in sorted(counts.items()):
        print(f"{dataset}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
