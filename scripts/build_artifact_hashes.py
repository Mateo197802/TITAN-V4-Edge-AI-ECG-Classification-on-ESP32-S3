from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from titan_v4.metrics.external_validation import manifest_file_fingerprint


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = Path("outputs/gold_master_external_validation/artifact_hashes/ARTIFACT_HASHES.csv")
FIELDS = ("relative_path", "sha256", "size_bytes")


def repository_files(root: Path, manifest: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    try:
        excluded = manifest.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("The artifact manifest must be inside the repository root.") from exc
    return sorted(path for path in result.stdout.splitlines() if path != excluded)


def build_rows(root: Path, manifest: Path) -> list[dict[str, str]]:
    rows = []
    for relative_path in repository_files(root, manifest):
        target = root / relative_path
        if not target.is_file():
            continue
        digest, size_bytes = manifest_file_fingerprint(target)
        rows.append({"relative_path": relative_path, "sha256": digest, "size_bytes": str(size_bytes)})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Check or rebuild the repository artifact SHA256 manifest.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--write", action="store_true", help="Replace the manifest with current file hashes.")
    args = parser.parse_args()
    manifest = args.manifest if args.manifest.is_absolute() else args.root / args.manifest
    try:
        expected_rows = build_rows(args.root, manifest)
    except ValueError as exc:
        parser.error(str(exc))

    if args.write:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS, quoting=csv.QUOTE_ALL, lineterminator="\n")
            writer.writeheader()
            writer.writerows(expected_rows)
        print(f"Wrote {len(expected_rows)} hashes to {manifest}")
        return 0

    with manifest.open("r", newline="", encoding="utf-8") as handle:
        recorded_rows = list(csv.DictReader(handle))
    recorded = {row["relative_path"]: row for row in recorded_rows}
    expected = {row["relative_path"]: row for row in expected_rows}
    changed = sorted(
        path for path in set(recorded) | set(expected) if recorded.get(path) != expected.get(path)
    )
    if changed:
        print(f"{len(changed)} manifest entries differ:")
        print("\n".join(changed))
        print("Review the changes, then pass --write to update the manifest.")
        return 1

    print(f"Manifest matches {len(expected_rows)} repository files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
