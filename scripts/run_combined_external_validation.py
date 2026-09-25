from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from titan_v4.metrics.external_validation import (
    cascade_summary,
    combined_summary,
    load_json,
    pathology_summary,
    primary9_summary,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the combined external validation summary.")
    parser.add_argument("--write", action="store_true", help="Write the combined summary JSON to disk.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/reproduced/combined/combined_reproduction_summary.json"),
    )
    args = parser.parse_args()

    primary9 = primary9_summary(
        load_json("outputs/reproduced/primary9/primary9_recomputed_report.json")
    )
    pathology = pathology_summary(
        load_json("outputs/gold_master_external_validation/pathology_primary5/pathology_primary5_external_summary.json")
    )
    cascade = cascade_summary(
        load_json("outputs/gold_master_external_validation/cascade_annex/cascade_safety_annex_report.json")
    )
    summary = combined_summary(primary9, pathology, cascade)
    if args.write:
        write_json(args.out, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
