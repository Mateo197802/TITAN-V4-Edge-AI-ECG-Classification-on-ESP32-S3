from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from titan_v4.metrics.external_validation import load_json, pathology_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Read and validate the Primary-5 pathology external summary.")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("outputs/gold_master_external_validation/pathology_primary5/pathology_primary5_external_summary.json"),
    )
    args = parser.parse_args()
    print(json.dumps(pathology_summary(load_json(args.report)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
