# NeuroGuardian TITAN V4 ECG

Reproducible inference and evidence package for the distributed TITAN V4 Primary-9 checkpoint, with ESP32-S3 firmware and archived engineering reports. This repository is research software and is not a medical device.

## Performance Evidence

The repository contains distinct performance figures from different checkpoints, units and protocols. They must not be combined or substituted for one another.

| Evidence | Metrics | Scope |
|---|---|---|
| CEDIA internal validation | Accuracy 89.84%; macro-F1 73.90%; weighted-F1 90.30% | 58,855 windows; separate CEDIA checkpoint |
| Distributed-checkpoint training summary | Best validation macro-F1 89.48% | 706 windows; model-selection metadata only |
| Recomputed repository-checkpoint diagnostic | Accuracy 69.35%; macro-F1 68.88%; weighted-F1 68.96% | 672 records; reproducible custom single-label diagnostic, not official Challenge score |
| Current checkpoint scored against the prior Primary-9 labels | 568/672 correct; accuracy 84.52%; macro-F1 82.35% | Same published checkpoint predictions, historical label table; does not reproduce the stored 605/672 aggregate |
| Historical Primary-9 aggregate | Accuracy 90.03%; macro-F1 87.82% | 672 rows; original predictions and checkpoint lineage unavailable; unverified, not disproven |
| Historical Pathology Primary-5 aggregate | Per-label accuracy 90.26%; macro-F1 66.87% | Unverified; no record-level prediction/target pair or checkpoint/threshold lineage |

See [metric reconciliation](reports/evidence/metric-reconciliation.md) for evidence boundaries. The 672 records resolve to the public `training/` partition in Challenge 2021 v1.0.3, not the hidden Challenge test set; checkpoint overlap is unknown. The 466/672 report is reproducible under its project-specific single-label rule, not proof of independent external performance. PhysioNet Challenge 2021 is multi-label and uses its own weighted scoring metric.

Pathology Primary-5 and Cascade/OOD outputs are archived historical aggregates, not reproduced from record-level predictions. CEDIA contains a separate pathology-supervised checkpoint and label map, but no matching Primary-5 prediction report; this is documented without attributing the historical metric to that model. See [Primary-9 diagnostic](reports/results/arrhythmia_primary9_external.md), [pathology evidence](reports/results/pathology_primary5_external.md), and the [legacy evidence index](outputs/gold_master_external_validation/README.md).

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
python scripts/compare_primary9_historical_labels.py
python scripts/audit_pathology_reproducibility.py
python scripts/run_primary9_external_validation.py
python scripts/run_combined_external_validation.py --write
python scripts/build_result_tables.py
python -m pytest -q
python scripts/verify_artifact_hashes.py
```

The first two source checks query the pinned PhysioNet 1.0.3 record indexes. The label check downloads only WFDB headers; inference downloads signal files as needed. `--offline` can be passed to the label/inference scripts after the files are cached. A smoke test with `--limit N` is not a reportable full evaluation.

The run writes `primary9_record_predictions.csv`, `primary9_recomputed_report.json`, `source_files_sha256.csv`, and `run_manifest.json`. The manifest records dependency versions, dataset/checkpoint/label/source hashes, preprocessing settings, and hashes for the code files used. Rerunning the command recomputes all metrics from the distributed checkpoint; it does not merely summarize the stored JSON.

The legacy-label comparator verifies those predictions against both the rebuilt and exact historical label tables, and writes the 672-row comparison plus metrics under `outputs/reviewer_verification/arrhythmia_primary9/`. The pathology audit writes a machine-readable readiness report; it deliberately does not compute a score without record-level targets and predictions. CEDIA evidence is a sanitized, read-only snapshot in `reports/evidence/cedia-pathology-crosscheck.json` and is not treated as a reproducible evaluation.

## Evidence Boundaries

- The 672-record single-label diagnostic is computationally reproducible from this checkout and the cited public dataset release; it is not an official Challenge metric or an independent external-validation claim.
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
