"""Prepare a fresh CEDIA weight-interpolation replay from reconstructed inputs."""
import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from titan_v4.reproducibility.refinement import sha256

BUNDLE = ROOT / "outputs/reviewer_verification/partial_backbone_refinement_2026-10-12"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-inputs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    input_manifest = json.loads((args.training_inputs / "INPUT_SHA256.json").read_text(encoding="utf-8"))
    for name in ("validation.npz", "evaluation.npz"):
        if sha256(args.training_inputs / name) != input_manifest[name]:
            raise ValueError("Reconstructed input differs from its checksum: " + name)
    if args.output_dir.exists():
        raise FileExistsError("Use a fresh interpolation directory")
    args.output_dir.mkdir(parents=True)
    for name in ("validation.npz", "evaluation.npz"):
        shutil.copy2(args.training_inputs / name, args.output_dir / name)
    for name in ("contract.py", "model_core.py", "titan_v4_net.py", "run_merge.py", "job.sbatch", "protocol.json",
                 "donor_partial.pth", "donor_expanded.pth"):
        shutil.copy2(BUNDLE / "training/weight_interpolation" / name, args.output_dir / name)
    shutil.copy2(BUNDLE / "model/partial_backbone_candidate.pth", args.output_dir / "base.pth")
    hashes = {path.name: sha256(path) for path in sorted(args.output_dir.iterdir()) if path.is_file()}
    (args.output_dir / "INPUT_SHA256.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prepared_input_files": len(hashes), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
