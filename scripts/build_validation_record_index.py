from __future__ import annotations

import argparse
import csv
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "data/external_validation/final_external_validation_labels.csv"
DEFAULT_OUTPUT = ROOT / "data/external_validation/validation_record_index.csv"
FIELDS = ("record_id", "source", "split")


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Label file is missing required fields: {sorted(missing)}")
        rows = [{field: row[field].strip() for field in FIELDS} for row in reader]
    if len(rows) != 672:
        raise ValueError(f"Expected 672 evaluation record IDs, found {len(rows)}")
    keys = [(row["source"].casefold(), row["record_id"].casefold()) for row in rows]
    if len(set(keys)) != len(keys) or any(not record_id for _, record_id in keys):
        raise ValueError("Evaluation record IDs must be non-empty and unique by source")
    return rows


def render(rows: list[dict[str, str]]) -> str:
    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract the source/split index for 672 evaluation records.")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        text = render(load_rows(args.labels))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != text:
            print(f"Record index differs from {args.labels} or is missing: {args.output}")
            return 1
        print(f"Verified 672 record IDs in {args.output}.")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8", newline="")
        print(f"Wrote 672 record IDs to {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
