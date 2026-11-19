from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from titan_v4.metrics.external_validation import sha256_file
from titan_v4.reproducibility.inference import load_primary9_model, predict_primary9, primary9_metrics
from titan_v4.reproducibility.physionet import (
    FRONTAL_LEADS,
    PHYSIONET_RELEASE,
    PHYSIONET_RELEASE_BASE,
    TARGET_FS,
    TARGET_SAMPLES,
    build_record_source_rows,
    ensure_local_wfdb_record,
    preprocess_primary9_signal,
)
from titan_v4.data.primary9_registry import PRIMARY9_CLASSES


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "data/external_validation/final_external_validation_labels.csv"
DEFAULT_SOURCE_MANIFEST = ROOT / "data/external_validation/record_source_manifest.csv"
DEFAULT_CHECKPOINT = ROOT / "models/gold_master/gold_master_primary9_model.pth"
DEFAULT_DATA_ROOT = ROOT / "data/cache/physionet/challenge-2021/1.0.3"
DEFAULT_OUTPUT = ROOT / "outputs/reproduced/primary9"
PREDICTION_FIELDS = (
    "record_id",
    "source_label",
    "source_dataset",
    "source_path",
    "true_label",
    "predicted_label",
    *(f"prob_{name}" for name in PRIMARY9_CLASSES),
)
SOURCE_HASH_FIELDS = ("relative_path", "sha256", "bytes")
CODE_FILES = (
    "scripts/run_primary9_inference.py",
    "scripts/build_primary9_evaluation_labels.py",
    "src/titan_v4/reproducibility/inference.py",
    "src/titan_v4/reproducibility/labels.py",
    "src/titan_v4/reproducibility/physionet.py",
    "src/titan_v4/models/titan_v4_net.py",
    "src/titan_v4/data/primary9_registry.py",
    "src/titan_v4/data/label_registry.py",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _key(row: dict[str, str], source_field: str) -> tuple[str, str]:
    return row.get(source_field, "").strip().casefold(), row.get("record_id", "").strip().upper()


def _verify_input_manifests(
    labels: list[dict[str, str]], source_rows: list[dict[str, str]]
) -> None:
    label_keys = [_key(row, "source") for row in labels]
    source_keys = [_key(row, "source_label") for row in source_rows]
    if len(set(label_keys)) != len(labels):
        raise ValueError("Validation labels contain duplicate source/record_id pairs")
    if len(set(source_keys)) != len(source_rows):
        raise ValueError("Source manifest contains duplicate source/record_id pairs")
    if set(label_keys) != set(source_keys):
        missing = sorted(set(label_keys) - set(source_keys))
        extra = sorted(set(source_keys) - set(label_keys))
        raise ValueError(f"Label/source manifest mismatch; missing={missing[:5]}, extra={extra[:5]}")
    wrong_versions = sorted({row.get("physionet_release", "") for row in source_rows} - {PHYSIONET_RELEASE})
    if wrong_versions:
        raise ValueError(f"Source manifest uses unsupported PhysioNet releases: {wrong_versions}")


def _load_wfdb_record(record_base: Path):
    try:
        import wfdb
    except ImportError as exc:
        raise RuntimeError("WFDB is required; install the pinned requirements.txt") from exc
    return wfdb.rdrecord(str(record_base), physical=True)


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _jsonable_report(report: dict[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(report, allow_nan=False))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download/read the pinned ECG records and recompute Primary-9 predictions and metrics."
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--offline", action="store_true", help="Require the versioned records to exist under --data-root.")
    parser.add_argument("--limit", type=int, help="Run a smoke test on the first N records; not a full evaluation.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--download-workers", type=int, default=6)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.batch_size < 1 or args.download_workers < 1:
        parser.error("--batch-size and --download-workers must be positive")
    for path in (args.labels, args.source_manifest, args.checkpoint):
        if not path.is_file():
            parser.error(f"Required artifact is missing: {path}")

    try:
        label_rows = _read_csv(args.labels)
        source_rows = _read_csv(args.source_manifest)
        _verify_input_manifests(label_rows, source_rows)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.limit is None and len(label_rows) != 672:
        parser.error(f"Full-support Primary-9 evaluation requires exactly 672 labels, found {len(label_rows)}")
    if args.limit is not None:
        label_rows = label_rows[: args.limit]
    if len(label_rows) > len(source_rows):
        parser.error("--limit exceeds the available source manifest")

    source_by_key = {_key(row, "source_label"): row for row in source_rows}
    selected_source_rows = [source_by_key[_key(row, "source")] for row in label_rows]
    source_paths = [row["record_path"] for row in selected_source_rows]
    if len(set(source_paths)) != len(source_paths):
        parser.error("Two validation rows resolve to the same upstream record path")

    args.data_root.mkdir(parents=True, exist_ok=True)
    try:
        with ThreadPoolExecutor(max_workers=args.download_workers) as executor:
            download_results = list(
                executor.map(
                    lambda row: ensure_local_wfdb_record(
                        row["record_path"], args.data_root, download=not args.offline
                    ),
                    selected_source_rows,
                )
            )
    except Exception as exc:
        parser.error(f"Could not prepare source ECG records: {exc}")

    source_hashes: dict[str, str] = {}
    source_sizes: dict[str, int] = {}
    for _base, hashes in download_results:
        for relative_path, digest in hashes.items():
            previous = source_hashes.get(relative_path)
            if previous is not None and previous != digest:
                parser.error(f"Source file changed while evaluating: {relative_path}")
            source_hashes[relative_path] = digest
    for relative_path in source_hashes:
        source_sizes[relative_path] = (args.data_root / Path(relative_path)).stat().st_size

    try:
        import torch
        import wfdb
    except ImportError as exc:
        parser.error(f"Install project requirements before inference: {exc}")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = load_primary9_model(args.checkpoint, device="cpu")

    model_inputs: list[np.ndarray] = []
    for index, (label_row, source_row, (record_base, _hashes)) in enumerate(
        zip(label_rows, selected_source_rows, download_results, strict=True), start=1
    ):
        try:
            record = wfdb.rdrecord(str(record_base), physical=True)
            if record.p_signal is None:
                raise ValueError("WFDB returned no physical ECG samples")
            model_inputs.append(
                preprocess_primary9_signal(record.p_signal, record.sig_name, record.fs)
            )
        except Exception as exc:
            parser.error(
                f"Record {label_row['record_id']} ({source_row['record_path']}) failed: {exc}"
            )
        if index % 50 == 0 or index == len(label_rows):
            print(f"Prepared {index}/{len(label_rows)} records")

    probabilities = predict_primary9(model, np.stack(model_inputs), batch_size=args.batch_size)
    y_true = [row["rhythm_label"].strip() for row in label_rows]
    y_pred = [PRIMARY9_CLASSES[int(index)] for index in probabilities.argmax(axis=1)]
    report = primary9_metrics(y_true, y_pred)
    full_support = args.limit is None
    report.update(
        {
            "label_set": "frozen 672-record source-header single-label diagnostic set",
            "evaluation_status": "FULL_672_DIAGNOSTIC_ONLY" if full_support else "SMOKE_TEST_NOT_REPORTABLE",
            "evaluation_scope": (
                "Custom single-label argmax evaluation on 672 records resolving to PhysioNet Challenge 2021 v1.0.3 "
                "training paths; not the official Challenge metric or hidden test set. Checkpoint overlap is unknown."
            ),
            "label_form": "One project-selected class derived from multi-label WFDB Dx codes.",
            "official_challenge_metric": False,
            "external_validation_claim": False,
            "clinical_validation": False,
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "label_csv_sha256": sha256_file(args.labels),
            "source_manifest_sha256": sha256_file(args.source_manifest),
            "physionet_release": PHYSIONET_RELEASE,
            "physionet_release_url": PHYSIONET_RELEASE_BASE,
            "physionet_release_doi": "10.13026/34va-7q14",
            "upstream_partition": "training",
            "official_challenge_test_set": False,
            "checkpoint_training_overlap": "not established; exact checkpoint training manifest is unavailable",
            "label_selection_policy": (
                "first non-NSR Primary-9 diagnosis in WFDB Dx-code order; "
                "prolonged-PR code 164947007 maps to 1AVB only when no non-normal rhythm is present"
            ),
            "external_training_allowed": False,
            "external_threshold_tuning_allowed": False,
        }
    )

    prediction_rows: list[dict[str, object]] = []
    for label_row, source_row, prediction, score_row in zip(
        label_rows, selected_source_rows, y_pred, probabilities, strict=True
    ):
        item: dict[str, object] = {
            "record_id": label_row["record_id"],
            "source_label": label_row["source"],
            "source_dataset": source_row["dataset"],
            "source_path": source_row["record_path"],
            "true_label": label_row["rhythm_label"],
            "predicted_label": prediction,
        }
        item.update({f"prob_{name}": f"{float(score):.10f}" for name, score in zip(PRIMARY9_CLASSES, score_row, strict=True)})
        prediction_rows.append(item)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = args.out_dir / "primary9_record_predictions.csv"
    report_path = args.out_dir / "primary9_recomputed_report.json"
    source_hash_path = args.out_dir / "source_files_sha256.csv"
    run_manifest_path = args.out_dir / "run_manifest.json"
    _write_csv(prediction_path, PREDICTION_FIELDS, prediction_rows)
    report_path.write_text(json.dumps(_jsonable_report(report), indent=2) + "\n", encoding="utf-8")
    _write_csv(
        source_hash_path,
        SOURCE_HASH_FIELDS,
        [
            {"relative_path": path, "sha256": source_hashes[path], "bytes": source_sizes[path]}
            for path in sorted(source_hashes)
        ],
    )

    try:
        import scipy
        import sklearn
        import wfdb
    except ImportError as exc:
        parser.error(f"Could not record the inference environment: {exc}")
    run_manifest = {
        "schema_version": 1,
        "evaluation_status": report["evaluation_status"],
        "evaluation_scope": report["evaluation_scope"],
        "label_semantics": report["label_form"],
        "official_challenge_metric": report["official_challenge_metric"],
        "external_validation_claim": report["external_validation_claim"],
        "clinical_validation": report["clinical_validation"],
        "record_count": len(label_rows),
        "model": {
            "architecture": "TitanV4Lite",
            "checkpoint": args.checkpoint.name,
            "sha256": report["checkpoint_sha256"],
        },
        "label_file": {"name": args.labels.name, "sha256": report["label_csv_sha256"]},
        "source_manifest": {
            "name": args.source_manifest.name,
            "sha256": report["source_manifest_sha256"],
        },
        "source_dataset": {
            "name": "PhysioNet/Computing in Cardiology Challenge 2021",
            "version": PHYSIONET_RELEASE,
            "url": PHYSIONET_RELEASE_BASE,
            "license": "CC BY 4.0; see the source release terms and REFERENCES.md",
        },
        "preprocessing": {
            "input_leads": list(FRONTAL_LEADS),
            "target_sampling_rate_hz": TARGET_FS,
            "window_seconds": TARGET_SAMPLES / TARGET_FS,
            "window_start_seconds": 0,
            "bandpass_hz": [0.5, 45.0],
            "butterworth_order": 3,
            "filter": "causal scipy.signal.lfilter with two-second zero-padded context",
            "resampling": "scipy.signal.resample_poly",
            "normalization": "per-lead z-score over the 10-second window",
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "wfdb": wfdb.__version__,
            "device": "cpu",
            "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "code_files": {
            relative_path: sha256_file(ROOT / relative_path) for relative_path in CODE_FILES
        },
        "source_files": str(source_hash_path.name),
        "outputs": {
            prediction_path.name: sha256_file(prediction_path),
            report_path.name: sha256_file(report_path),
            source_hash_path.name: sha256_file(source_hash_path),
        },
        "source_file_count": len(source_hashes),
        "source_dataset_counts": dict(sorted(Counter(row["dataset"] for row in selected_source_rows).items())),
    }
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(_jsonable_report(report), indent=2))
    print(f"Evidence written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
