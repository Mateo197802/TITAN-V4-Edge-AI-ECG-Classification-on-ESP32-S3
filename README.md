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
| Fresh CEDIA pathology checkpoint evaluation | Calibrated mean per-label accuracy 71.58%; macro-F1 41.37%; fixed 0.50: 73.15% / 41.76%; fixed 0.65 sensitivity: 79.90% / 42.03% | 273 calibration and 406 patient-disjoint test records; separate checkpoint SHA-256 `e9a44e4e...498353b`; 0.65 was not verified as the historical decision threshold; not the historical 90.26%/66.87% result |
| Historical Primary-9 aggregate | Accuracy 90.03%; macro-F1 87.82% | 672 rows; original predictions and checkpoint lineage unavailable; unverified, not disproven |
| Historical Pathology Primary-5 aggregate | Per-label accuracy 90.26%; macro-F1 66.87% | Unverified; no record-level prediction/target pair or checkpoint/threshold lineage |

See [metric reconciliation](reports/evidence/metric-reconciliation.md) for evidence boundaries. The 672 records resolve to the public `training/` partition in Challenge 2021 v1.0.3, not the hidden Challenge test set; checkpoint overlap is unknown. The 466/672 report is reproducible under its project-specific single-label rule, not proof of independent external performance. PhysioNet Challenge 2021 is multi-label and uses its own weighted scoring metric.

The historical Pathology Primary-5 aggregate remains unreproduced. A separate CEDIA checkpoint has now been evaluated on a patient-disjoint PTB-XL subset; its per-record scores and input hashes are checked in and the historical metric is not attributed to it. The score table lets reviewers recompute all reported metrics, but full inference still requires access to the CEDIA checkpoint and source files, which are not redistributed. See [Primary-9 diagnostic](reports/results/arrhythmia_primary9_external.md), [pathology evidence](reports/results/pathology_primary5_external.md), and the [legacy evidence index](outputs/gold_master_external_validation/README.md).

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
python src/titan_v4/evaluation/pathology_primary5_protocol.py --predictions_csv outputs/reviewer_verification/pathology_primary5/cedia_ptbxl_v1_0_3_20260925/pathology_primary5_scores.csv --out_dir <temporary-directory> --fixed-threshold 0.50 --fixed-threshold 0.65
python scripts/run_primary9_external_validation.py
python scripts/run_combined_external_validation.py --write
python scripts/build_result_tables.py
python -m pytest -q
python scripts/verify_artifact_hashes.py
```

The first two source checks query the pinned PhysioNet 1.0.3 record indexes. The label check downloads only WFDB headers; inference downloads signal files as needed. `--offline` can be passed to the label/inference scripts after the files are cached. A smoke test with `--limit N` is not a reportable full evaluation.

The run writes `primary9_record_predictions.csv`, `primary9_recomputed_report.json`, `source_files_sha256.csv`, and `run_manifest.json`. The manifest records dependency versions, dataset/checkpoint/label/source hashes, preprocessing settings, and hashes for the code files used. Rerunning the command recomputes all metrics from the distributed checkpoint; it does not merely summarize the stored JSON.

The legacy-label comparator verifies those predictions against both the rebuilt and exact historical label tables, and writes the 672-row comparison plus metrics under `outputs/reviewer_verification/arrhythmia_primary9/`. The pathology audit keeps the old 90.26%/66.87% claim unverified; the new CEDIA result is a separate experiment with prediction-level evidence. The command above regenerates its calibrated and fixed-threshold metrics from the checked-in per-record score table. Full CEDIA inference needs the exact checkpoint hash and CEDIA source/runtime described in the [evaluation evidence](outputs/reviewer_verification/pathology_primary5/cedia_ptbxl_v1_0_3_20260925/README.md).

## Evidence Boundaries

- The 672-record single-label diagnostic is computationally reproducible from this checkout and the cited public dataset release; it is not an official Challenge metric or an independent external-validation claim.
- The CEDIA pathology score is recomputable from checked-in targets and probabilities. It is a patient-disjoint subset of PTB-XL v1.0.3, not a source-held-out estimate or a reproduction of the historical aggregate. End-to-end inference is not standalone until the checkpoint and the exact CEDIA source files are made available under an approved redistribution basis.
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
| `outputs/reviewer_verification/pathology_primary5/cedia_ptbxl_v1_0_3_20260925/` | CEDIA checkpoint pathology predictions, per-record scores, input hashes, and recomputable metrics |
| `reports/` | Methods, results, and evidence traceability |
| `hardware/esp32/` | Firmware, conversion tools, and engineering artifacts |

## Citation and License

Dataset users must cite the exact versioned PhysioNet releases used for each analysis, including PTB-XL v1.0.3 for the CEDIA pathology evaluation and Challenge 2021 v1.0.3 for the Primary-9 diagnostic. PhysioNet lists CC BY 4.0 for these dataset files. TITAN source code is MIT-licensed; project-authored research artifacts and model weights have a separate CC BY 4.0 notice. Neither project license relicenses third-party datasets or recordings. See [license scope](LICENSE_SCOPE.md), [artifact license](LICENSE-ARTIFACTS.md), [dataset provenance](DATA_PROVENANCE.md), and [references](REFERENCES.md).

The result is for research use only. It must not be used to diagnose, treat, or rule out a medical condition.
