# Reproducibility Protocol

## Reproducible Diagnostic Computation

This protocol recomputes CPU inference for the exact distributed Primary-9 checkpoint on a frozen 672-record project diagnostic set. It verifies source paths against PhysioNet's pinned Challenge 2021 v1.0.3 `RECORDS` indexes, reconstructs one project-selected rhythm label from multi-label WFDB headers, reads the ECG signals, applies the packaged preprocessing, runs the checkpoint, and computes ordinary accuracy/F1 from per-record predictions. This is a reproducible custom single-label diagnostic; it is not the official Challenge metric and does not establish independent external validation.

It does not retrain the checkpoint. `outputs/reproduced/primary9/run_manifest.json` records the Python and library versions, CPU/determinism settings, hashes of the checkpoint, labels, source manifest, inference-code files, downloaded WFDB files, predictions, and report. A complete run reports 672 records and identifies its diagnostic-only status. Internal validation values, the historical 605/672 aggregate, and the pathology aggregate are separate, non-comparable evidence; see [metric reconciliation](reports/evidence/metric-reconciliation.md).

## Clean-Checkout Commands

Tested with CPython 3.12 on Windows. Linux/macOS are expected to work with the same pinned Python packages, but are not the recorded reference environment.

```text
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

Internet access is needed for the PhysioNet source-index and WFDB downloads. The runner fetches only the 672 referenced records. The waveform cache is under ignored `data/cache/physionet/challenge-2021/1.0.3/`. To require a pre-populated cache, pass `--offline` to the label/inference scripts. `--limit N` is a smoke test and must not be reported as a full evaluation.

The hash verifier covers the checked-in artifact inventory. It is complementary to, not a replacement for, recomputing the inference. For an independent check, compare the regenerated prediction CSV and metrics with the checked-in files and inspect the run manifest and source-file hash inventory.

## CEDIA Pathology Checkpoint Evaluation

This is a separate, post hoc evaluation of the CEDIA checkpoint, not the distributed Primary-9 checkpoint and not a reproduction of the historical 90.256% / 66.874% aggregate. The checkpoint is loaded as `TitanV4Max` and strict state-dict loading is required. PTB-XL v1.0.3 metadata and SCP mappings are pinned by SHA-256; all 2,580 pathology-sidecar vectors were reconstructed exactly. The CEDIA split also contains PTB-XL `HR#####` aliases. These were mapped to the corresponding `ecg_id` before patient exclusion; seven train/validation record IDs overlap, and those records are not used for calibration or test. The final cohort has 273 calibration and 406 test records, with test patients excluded from both checkpoint splits. Two unreadable calibration signals are listed with hashes and error types in the input manifest.

The primary metric uses per-class F1 thresholds selected on calibration records only and then frozen for test scoring. Fixed 0.50 and 0.65 thresholds are reported as sensitivity checks. The measured result is 71.58% mean per-label accuracy and 41.37% macro-F1; fixed 0.65 gives 79.90% and 42.03%. The legacy JSON's `configured_macro_f1_reference: 0.65` is not evidence that 0.65 was the historical prediction threshold. The detailed per-record table is `outputs/reviewer_verification/pathology_primary5/cedia_ptbxl_v1_0_3_20260925/pathology_primary5_scores.csv`.

Anyone can recompute the saved metrics from the checked-in row-level scores without the checkpoint:

```powershell
python src/titan_v4/evaluation/pathology_primary5_protocol.py `
  --predictions_csv outputs/reviewer_verification/pathology_primary5/cedia_ptbxl_v1_0_3_20260925/pathology_primary5_scores.csv `
  --out_dir <temporary-directory> `
  --fixed-threshold 0.50 --fixed-threshold 0.65
```

The report, prediction rows, input-file hashes, and output checksums are in the same evidence directory. The aggregation command reproduces the reported values within floating-point tolerance. Full signal-to-prediction inference requires the CEDIA checkpoint with SHA-256 `e9a44e4eea8ebb8f89d5e32909ae4afcc96442d1fa8eacb1e57a73bfa498353b`, PTB-XL v1.0.3 signals, the CEDIA source files, and the runtime versions recorded in the report. The checkpoint and CEDIA source tree are not redistributed, so the inference run is not yet standalone for an outside reviewer; the model's redistribution rights must be confirmed separately. The run is exploratory and is not a prespecified or source-held-out clinical validation.

An authorized reviewer with those exact files can rerun inference with the committed evaluator (provide paths to the local CEDIA project; it reads PTB-XL from `<project-root>/DATA/ptb-xl/`):

```bash
python scripts/evaluate_cedia_pathology_validation.py \
  --project-root /data/V4_CEDIA \
  --weights /data/V4_CEDIA/03_OUTPUTS/titan_v4_lite_weights_best.pth \
  --training-summary /data/V4_CEDIA/03_OUTPUTS/training_summary.json \
  --split-metadata /data/V4_CEDIA/03_OUTPUTS/split_metadata.json \
  --cedia-code-dir /data/V4_CEDIA/01_CODIGO_FUENTE/AUDITORIA \
  --output-dir /data/titan-v4-pathology-results
```

The measured environment was Python 3.10.14, PyTorch 2.1.2, NumPy 1.26.4, scikit-learn 1.3.2, SciPy 1.13.1, and WFDB 4.1.2 on CPU. The machine-readable report also records exact input and code hashes.

## Frozen Signal and Diagnostic Label Contract

- Input: the six frontal leads I, II, III, aVR, aVL, aVF from the first ten seconds.
- Sampling: versioned WFDB physical signals, resampled to 125 Hz.
- Filter: third-order 0.5-45 Hz Butterworth, causal `scipy.signal.lfilter`, with two seconds of leading zero context.
- Normalization: per-lead z-score over the ten-second window.
- Output: argmax across AFIB, SB, STACH, NSR, PVC, RBBB, LBBB, PAC, and 1AVB.
- Labels: the first non-NSR Primary-9 class in WFDB `Dx` code order; prolonged-PR code `164947007` is a 1AVB fallback only when no non-normal rhythm is present.
- Evaluation: ordinary single-label accuracy, macro-F1, weighted-F1, class-level metrics, and a 9x9 confusion matrix, computed from all per-record predictions. No threshold tuning or training is performed on this set. The upstream Challenge labels are multi-label, and its official evaluator uses a distinct weighted metric; this report is not an official Challenge score.

## Current Evidence and Non-Claims

The result is a reproducible inference/evaluation run on records from the public Challenge training partition, not the hidden Challenge test set. The checkpoint's exact training manifest and offline teacher-sidecar arrays are not available. CEDIA's train/validation manifests and zero-overlap report belong to a different checkpoint lineage; they are not used to claim independence for this checkpoint. The test label is project-local and does not imply source-held-out evaluation.

The read-only CEDIA record/label cross-check and checkpoint-hash comparison are documented in [CEDIA cross-check evidence](reports/evidence/cedia-readonly-crosscheck.md).

The older 605/672 Primary-9 report is retained as an unverified historical aggregate, not a disproven result: its labels differ from the rebuilt labels on 129 rows, and its original predictions are unavailable. The comparator scores the current checkpoint against the old labels (568/672; accuracy 84.52%, macro-F1 82.35%, weighted-F1 84.64%) but does not reproduce 605/672. The per-record crosswalk and hashes are in `outputs/reviewer_verification/arrhythmia_primary9/`.

The 90.26% Pathology Primary-5 per-label accuracy and 66.87% macro-F1 remain unverified historical aggregates. The distributed checkpoint's training summary records zero pathology-labeled windows and loss weight 0; the checked 672-row candidate label table contains no Primary-5 targets. The separate CEDIA checkpoint's measured result is documented above and does not establish the historical checkpoint, thresholds, or cohort. The audit report is `outputs/reviewer_verification/pathology_primary5/pathology_reproducibility_audit.json`; read-only CEDIA provenance is summarized in `reports/evidence/cedia-pathology-crosscheck.json`. Cascade/OOD remains an archived safety annex, not a recomputed result.

## Firmware Build Check

The safe `esp32s3` profile compiled successfully on 2026-09-24 using the project-local ESP32-S3-DevKitC-1-N8R8 profile (8 MB flash and 8 MB octal PSRAM). The pinned environment is PlatformIO Core 6.2.0, Espressif32 platform 6.12.0, Arduino-ESP32 2.0.17, and TensorFlowLite_ESP32 commit `e88e0ebee0430ed716ff5b49854795db90066e59`. Reproduce it with:

```powershell
python -m pip install -r requirements-firmware.txt
python -m platformio run -d hardware/esp32/firmware -e esp32s3
```

The build used 137,232 / 327,680 bytes RAM (41.9%) and 5,517,309 / 6,553,600 bytes of the selected application partition (84.2%). Full evidence, third-party compiler warnings, and the firmware binary SHA-256 are in [the build record](reports/evidence/firmware-build.md). The legacy TFLite library is pinned to make this revision buildable; upstream says it is outdated and no longer recommended. This was a compile only: no board was flashed, no physical inference was run, and embedded preprocessing has not been proven numerically equivalent to the Python pipeline.

Firmware source-level and build-contract tests are part of `python -m pytest -q`. The fixed firmware interpreter lifetime prevents the wrapper from deleting a function-static interpreter it does not own.

## Artifact Inventory

`outputs/reproduced/primary9/` contains:

- `primary9_record_predictions.csv`: one row per record and one probability per class.
- `primary9_recomputed_report.json`: aggregate, class-level, and confusion-matrix metrics plus frozen-input hashes.
- `source_files_sha256.csv`: byte count and SHA-256 for every WFDB file used.
- `run_manifest.json`: environment, preprocessing contract, source/model/input/output hashes, and code-file hashes.

`data/external_validation/record_source_manifest.csv` maps each record to its exact versioned upstream path and release attribution. `data/external_validation/final_external_validation_labels.csv` contains only the minimal label fields needed to evaluate the model.
