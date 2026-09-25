# NeuroGuardian TITAN V4 ECG

Reproducible inference and evidence package for the distributed TITAN V4 Primary-9 checkpoint, with ESP32-S3 firmware and archived engineering reports. This repository is research software and is not a medical device.

## Recomputed Primary-9 Result

The repository now includes a full CPU inference run over the frozen 672-record evaluation label set. ECG signals are fetched from the versioned PhysioNet/Computing in Cardiology Challenge 2021 release, and the per-record predictions and probabilities are included under `outputs/reproduced/primary9/`.

| Metric | Recomputed value |
|---|---:|
| Records | 672 |
| Correct | 466 |
| Accuracy | 0.693452 |
| Macro-F1 | 0.688841 |
| Weighted-F1 | 0.689590 |

These 672 records resolve to the public `training/` partition in Challenge 2021 v1.0.3. The local `split=test` field is a project label, not the Challenge's hidden test set. The exact training manifest for this checkpoint is not available, so independence from checkpoint training is not claimed. The older 605/672 aggregate has been superseded and is retained only as a marked legacy artifact.

Pathology Primary-5 and Cascade/OOD numbers in the historical report folder have not been recomputed from record-level predictions. They are not presented as current results. See [results and claim boundaries](reports/results/arrhythmia_primary9_external.md) and the [legacy evidence index](outputs/gold_master_external_validation/README.md).

## Run It

Requirements: Python 3.12, internet access for PhysioNet, and enough disk space for the selected ECG records. The runner downloads only the 672 referenced records, not the complete PhysioNet release.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python scripts/build_validation_record_index.py --check
python scripts/build_record_source_manifest.py --check
python scripts/build_primary9_evaluation_labels.py --check
python scripts/run_primary9_inference.py --download-workers 6 --batch-size 32
python scripts/run_primary9_external_validation.py
python scripts/run_combined_external_validation.py --write
python scripts/build_result_tables.py
python -m pytest -q
python scripts/verify_artifact_hashes.py
```

The first two source checks query the pinned PhysioNet 1.0.3 record indexes. The label check downloads only WFDB headers; inference downloads signal files as needed. `--offline` can be passed to the label/inference scripts after the files are cached. A smoke test with `--limit N` is not a reportable full evaluation.

The run writes `primary9_record_predictions.csv`, `primary9_recomputed_report.json`, `source_files_sha256.csv`, and `run_manifest.json`. The manifest records dependency versions, dataset/checkpoint/label/source hashes, preprocessing settings, and hashes for the code files used. Rerunning the command recomputes all metrics from the distributed checkpoint; it does not merely summarize the stored JSON.

## Evidence Boundaries

- Inference and the 672-record evaluation are reproducible from this checkout and the cited public dataset release.
- Training the checkpoint from scratch is not currently a reproducible claim: the exact training/validation manifests and offline teacher-sidecar arrays are not included.
- A read-only CEDIA comparison confirmed that the local 672 record IDs and labels match its external manifest. CEDIA's train/validation manifests and checkpoint files do not establish the lineage or training overlap of the checkpoint distributed here; see the [cross-check evidence](reports/evidence/cedia-readonly-crosscheck.md).
- The safe `esp32s3` firmware profile compiles with the pinned PlatformIO toolchain; the reproducible command and measured image sizes are in [firmware build evidence](reports/evidence/firmware-build.md). No physical-board flash, signal-chain equivalence, or on-device clinical performance is claimed.
- The current Arduino firmware still uses a pinned legacy TensorFlow Lite Micro library. Its upstream maintainer marks it outdated/not recommended; it is retained here only to reproduce this firmware revision, not recommended as a dependency choice for new projects.

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md), [DATA_PROVENANCE.md](DATA_PROVENANCE.md), [AUDIT_REPORT.md](AUDIT_REPORT.md), and [REFERENCES.md](REFERENCES.md).

## Repository Map

| Path | Contents |
|---|---|
| `src/`, `scripts/` | Model, signal processing, dataset resolution, inference, and reporting code |
| `data/external_validation/` | Minimal label table, 672-record index, and per-record source manifest; no ECG signals |
| `models/gold_master/` | Distributed Primary-9 checkpoint and training summary |
| `outputs/reproduced/` | Recomputed per-record predictions, metrics, input hashes, and run manifest |
| `outputs/gold_master_external_validation/` | Historical aggregate reports, explicitly marked legacy |
| `reports/` | Methods, results, and evidence traceability |
| `hardware/esp32/` | Firmware, conversion tools, and engineering artifacts |

## Citation and License

Dataset users must cite the versioned PhysioNet Challenge 2021 release and the relevant source-dataset papers. The release page specifies CC BY 4.0 for its files. TITAN source code is MIT-licensed; project-authored research artifacts and model weights have a separate CC BY 4.0 notice. Neither project license relicenses third-party datasets or recordings. See [license scope](LICENSE_SCOPE.md), [artifact license](LICENSE-ARTIFACTS.md), and [dataset provenance](DATA_PROVENANCE.md).

The result is for research use only. It must not be used to diagnose, treat, or rule out a medical condition.
