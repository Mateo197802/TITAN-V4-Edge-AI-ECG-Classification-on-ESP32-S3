"""Rebuild the CEDIA training inputs from public signals and frozen teacher sidecars."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from titan_v4.data.primary9_registry import PRIMARY9_CLASSES
from titan_v4.reproducibility.refinement import load_tensor, read_manifest, sha256

BUNDLE = ROOT / "outputs/reviewer_verification/partial_backbone_refinement_2026-10-12"


def build_split(rows, features, signals):
    if len(rows) != len(features["y"]) or features["x"].shape != (len(rows), 192):
        raise ValueError("Frozen feature sidecar dimensions differ")
    if signals.shape != (len(rows), 6, 1250) or signals.dtype != np.float32:
        raise ValueError("Signal input dimensions/dtype differ")
    if (not np.issubdtype(features["y"].dtype, np.integer)
        or np.any(features["y"] < 0) or np.any(features["y"] >= len(PRIMARY9_CLASSES))):
        raise ValueError("Frozen targets are outside Primary-9")
    for name in ("gold", "teacher"):
        if name not in features:
            continue
        p = features[name]
        if p.shape != (len(rows), 9) or not np.isfinite(p).all() or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=2e-6, rtol=0):
            raise ValueError("Invalid frozen probability sidecar")
    for index, row in enumerate(rows):
        if (row["ecg_id"] != str(features["ecg_id"][index])
            or row["dataset"] != str(features["dataset"][index])
            or row["reference_label"] != PRIMARY9_CLASSES[int(features["y"][index])]):
            raise ValueError("Feature/manifest record ordering or labels differ")
    packed = {"x": signals, "embedding": features["x"].astype(np.float32, copy=False),
              "y": features["y"], "dataset": features["dataset"], "ecg_id": features["ecg_id"],
              "gold": features["gold"], "tensor_sha256": np.asarray([row["tensor_sha256"] for row in rows])}
    if "teacher" in features:
        packed["teacher"] = features["teacher"]
    return packed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/cache/physionet")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if args.output_dir.exists():
        raise FileExistsError("Use a new reproduction directory")
    args.output_dir.mkdir(parents=True)
    receipts = {}
    def waveforms(rows):
        def read(row):
            return load_tensor(row, args.data_root, download=not args.offline)
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            tensors = []
            for index, tensor in enumerate(executor.map(read, rows), 1):
                tensors.append(tensor)
                if index % 1000 == 0:
                    print(f"Verified {index}/{len(rows)} source tensors", flush=True)
            return np.stack(tensors)
    for role in ("train", "validation"):
        manifest = BUNDLE / f"manifests/{role}_records.csv"
        rows = read_manifest(manifest)
        features = dict(np.load(BUNDLE / f"training/phase2_reference/{role}_features.npz", allow_pickle=False))
        packed = build_split(rows, features, waveforms(rows))
        output = args.output_dir / (role + ".npz")
        np.savez_compressed(output, **packed)
        shutil.copy2(manifest, args.output_dir / (role + ".csv"))
        receipts[role] = {"records": len(rows), "verified_tensor_hashes": len(rows), "npz_sha256": sha256(output)}
        print(f"Rebuilt {role}: {len(rows)} records", flush=True)
    rows = read_manifest(BUNDLE / "manifests/followup_records.csv")
    expected_rows = read_manifest(BUNDLE / "evaluation/followup_predictions.csv")
    expected = {(r["cohort"],r["dataset"],r["ecg_id"]):r for r in expected_rows}
    matched = [expected[(r["cohort"],r["dataset"],r["ecg_id"])] for r in rows]
    if any(r["reference_label"] != p["reference"] for r,p in zip(rows,matched,strict=True)):
        raise ValueError("Follow-up reference labels differ")
    packed = {"x": waveforms(rows), "y": np.asarray([PRIMARY9_CLASSES.index(r["reference_label"]) for r in rows],dtype=np.int64),
              "dataset": np.asarray([r["dataset"] for r in rows]), "cohort": np.asarray([r["cohort"] for r in rows]),
              "ecg_id": np.asarray([r["ecg_id"] for r in rows]),
              "gold_label": np.asarray([PRIMARY9_CLASSES.index(r["gold_prediction"]) for r in matched],dtype=np.int64),
              "phase2_label": np.asarray([PRIMARY9_CLASSES.index(r["phase2_prediction"]) for r in matched],dtype=np.int64),
              "previous_partial_label": np.asarray([PRIMARY9_CLASSES.index(r["previous_partial_prediction"]) for r in matched],dtype=np.int64),
              "tensor_sha256": np.asarray([r["tensor_sha256"] for r in rows])}
    np.savez_compressed(args.output_dir / "evaluation.npz", **packed)
    shutil.copy2(BUNDLE / "manifests/followup_records.csv", args.output_dir / "evaluation.csv")
    receipts["evaluation"] = {"records":len(rows), "verified_tensor_hashes":len(rows),
                              "npz_sha256":sha256(args.output_dir / "evaluation.npz")}
    for name in ("contract.py", "model_core.py", "titan_v4_net.py", "train_phase3.py", "test_phase3.py", "protocol.json", "job.sbatch"):
        shutil.copy2(BUNDLE / "training/phase3" / name, args.output_dir / name)
    shutil.copy2(ROOT / "models/gold_master/gold_master_primary9_model.pth", args.output_dir / "gold.pth")
    shutil.copy2(BUNDLE / "training/warm_start_phase2.pth", args.output_dir / "phase2_selected.pth")
    (args.output_dir / "rebuild_receipt.json").write_text(json.dumps(receipts,indent=2) + "\n",encoding="utf-8")
    manifest = {p.name:sha256(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()}
    (args.output_dir / "INPUT_SHA256.json").write_text(json.dumps(manifest,indent=2) + "\n",encoding="utf-8")
    print(json.dumps({"output":str(args.output_dir),"splits":receipts},indent=2))


if __name__ == "__main__":
    main()
