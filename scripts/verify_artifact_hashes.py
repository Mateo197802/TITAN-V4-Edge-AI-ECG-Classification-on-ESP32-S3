from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from titan_v4.metrics.external_validation import verify_hash_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify SHA256 hashes for public validation artifacts.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/gold_master_external_validation/artifact_hashes/ARTIFACT_HASHES.csv"),
    )
    args = parser.parse_args()
    failures = verify_hash_manifest(args.root, args.manifest)
    print(json.dumps({"passed": not failures, "failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
