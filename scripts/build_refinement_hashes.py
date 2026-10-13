"""Refresh byte-level checksums for the refinement evidence bundle."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from titan_v4.reproducibility.refinement import sha256

BUNDLE = ROOT / "outputs/reviewer_verification/partial_backbone_refinement_2026-10-12"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    path = BUNDLE / "SHA256SUMS.json"
    files = [p for p in sorted(BUNDLE.rglob("*")) if p.is_file() and p != path
             and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    if any(p.suffix in {".hea", ".dat", ".mat"} for p in files):
        raise ValueError("Source ECG waveform/header files must stay in the untracked data cache")
    expected = {p.relative_to(BUNDLE).as_posix():sha256(p) for p in files}
    if args.write:
        path.write_text(json.dumps(expected,indent=2) + "\n",encoding="utf-8")
    elif json.loads(path.read_text(encoding="utf-8")) != expected:
        raise ValueError("Bundle checksums differ; review the files before --write")
    print(f"{'Wrote' if args.write else 'Verified'} {len(expected)} bundle hashes")


if __name__ == "__main__":
    main()
