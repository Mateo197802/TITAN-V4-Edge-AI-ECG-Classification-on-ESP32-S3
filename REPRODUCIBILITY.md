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

The 90.26% Pathology Primary-5 per-label accuracy and 66.87% macro-F1 remain unverified. The distributed checkpoint's training summary records zero pathology-labeled windows and loss weight 0; the checked 672-row candidate label table contains no Primary-5 targets. CEDIA has a separate supervised checkpoint and PTB-XL label map, but its checkpoint hash differs and no matching Primary-5 predictions or metrics were found. The audit report is `outputs/reviewer_verification/pathology_primary5/pathology_reproducibility_audit.json`; read-only CEDIA evidence is summarized in `reports/evidence/cedia-pathology-crosscheck.json`. Do not treat that CEDIA map's coverage as performance evidence. Cascade/OOD remains an archived safety annex, not a recomputed result.

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
