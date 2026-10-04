# NeuroGuardian TITAN V4 ECG

Reproducible inference and evidence package for the distributed TITAN V4 Primary-9 checkpoint, with ESP32-S3 firmware and archived engineering reports. This repository is research software and is not a medical device.

## Primary Results

| Endpoint | Historical reported result | Evaluation unit |
|---|---|---|
| Arrhythmia Primary-9 | CEDIA report: 605/672; accuracy 90.03%; macro-F1 87.82%; weighted-F1 90.09% | 672 ECG records |
| Pathology Primary-5 | CEDIA summary: mean per-label accuracy 90.256%; macro-F1 66.874% | Five binary pathology labels |

The source aggregates, endpoint definitions, record-level evidence, and verification commands are indexed in [Primary Endpoint Results](reports/results/primary_endpoints.md). For arrhythmia, the CEDIA confusion matrix recalculates to 605/672; the packaged fresh inference with the repository checkpoint and final labels returns 568/672. The 129 CEDIA target updates are published as a sanitized row-level audit. For pathology, the 90.256% / 66.874% source summary and its provenance hashes are preserved separately from the CEDIA PTB-XL score table.

For full methods and evidence status, see [reproducibility](REPRODUCIBILITY.md), [metric reconciliation](reports/evidence/metric-reconciliation.md), [dataset provenance](DATA_PROVENANCE.md), and the [audit report](AUDIT_REPORT.md). The 672 ECG identifiers resolve to the public `training/` partition in PhysioNet Challenge 2021 v1.0.3; this project uses its documented single-label Primary-9 rule, not the hidden Challenge test set or official Challenge score.

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
python scripts/verify_primary9_gold_evidence.py
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

The legacy-label comparator writes the current model's 672-row predictions against the rebuilt and prior label tables. The Gold evidence verifier separately recalculates the historical Primary-9 aggregate from its saved confusion matrix and validates the target-update audit; its output also reports the distinct fresh inference result. The historical Pathology-5 values remain source-reported; the separate CEDIA PTB-XL score table is recomputable from its per-record probabilities and does not calculate those historical values.

## Evidence Boundaries

- The 672-record single-label diagnostic is computationally reproducible from this checkout and the cited public dataset release; it is not an official Challenge metric or an independent external-validation claim.
- The CEDIA pathology score is recomputable from checked-in targets and probabilities. It is a patient-disjoint subset of PTB-XL v1.0.3, not a source-held-out estimate or a reproduction of the historical aggregate. End-to-end inference is not standalone until the checkpoint and the exact CEDIA source files are made available under an approved redistribution basis.
- Training the checkpoint from scratch is not currently a reproducible claim: the exact training/validation manifests and offline teacher-sidecar arrays are not included.
- The historical CEDIA Primary-9 report includes 129 target updates; every final label in the update file equals its recorded model Top-1. The source report's confusion matrix recalculates to 605/672, while the fresh repository inference on the same final label set yields 568/672. Row-level artifacts and both calculations are kept distinct in `outputs/reviewer_verification/arrhythmia_primary9/cedia_gold_2026-05-26/`.
- The historical Pathology Primary-5 values are preserved in [the source record](reports/evidence/pathology-primary5-gold-source.json). Its source summary and promotion manifest declare the checkpoint lineage; the declared checkpoint and row-level target/prediction files were not present at their CEDIA paths during verification. The separate PTB-XL evaluation is a different experiment.
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
| `outputs/reviewer_verification/arrhythmia_primary9/cedia_gold_2026-05-26/` | Sanitized 672-row CEDIA label history, 129 target updates, historical matrix, and fresh row-level inference |
| `outputs/reviewer_verification/pathology_primary5/cedia_ptbxl_v1_0_3_20260925/` | CEDIA checkpoint pathology predictions, per-record scores, input hashes, and recomputable metrics |
| `reports/` | Methods, results, and evidence traceability |
| `hardware/esp32/` | Firmware, conversion tools, and engineering artifacts |

## Citation and License

Dataset users must cite the exact versioned PhysioNet releases used for each analysis, including PTB-XL v1.0.3 for the CEDIA pathology evaluation and Challenge 2021 v1.0.3 for the Primary-9 diagnostic. PhysioNet lists CC BY 4.0 for these dataset files. TITAN source code is MIT-licensed; project-authored research artifacts and model weights have a separate CC BY 4.0 notice. Neither project license relicenses third-party datasets or recordings. See [license scope](LICENSE_SCOPE.md), [artifact license](LICENSE-ARTIFACTS.md), [dataset provenance](DATA_PROVENANCE.md), and [references](REFERENCES.md).

The result is for research use only. It must not be used to diagnose, treat, or rule out a medical condition.
