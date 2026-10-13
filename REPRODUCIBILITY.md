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

## Historical Gold Aggregate Checks

The sanitized CEDIA Primary-9 package contains the 672 final labels, the 129 changed-label rows, the source report's confusion matrix and per-class values, plus a fresh repository inference with row-level probabilities and input hashes. Verify the saved aggregate arithmetic and the separate fresh run with:

```powershell
python scripts/verify_primary9_gold_evidence.py
```

The verifier recalculates the saved CEDIA final-reference matrix to 605/672, 90.0298% accuracy, 87.8192% macro-F1, and 90.0853% weighted-F1. The report separately gives 476/672 on the original reference, without its original confusion matrix or historical row-level predictions. The 90.03% / 87.82% result is therefore tied to the final updated labels, not the original labels. The pre-update manifest is `DATASETS_CURADOS/RHYTHM_PRIMARY_V2/manifest_external_test.csv`, SHA-256 `8190EC19F4887D4618213CBF95B9DA3B68948DDDAB4F0233D975EBA5938E5B8C`; its 672 IDs and `rhythm_label_name` values were matched to the sanitized `original_rhythm_label` field. The raw manifest is not redistributed because it includes demographics and absolute paths.

Regenerate the paired local-checkpoint scores against both reference columns and the per-class/hash files with:

```powershell
python scripts/recompute_primary9_paired_reference.py
```

That fixed checkpoint yields 466/672 on original labels and 568/672 on final labels. The row-level files are in `outputs/reviewer_verification/arrhythmia_primary9/cedia_gold_2026-05-26/fresh_inference/`; this is a distinct signal-to-prediction result, not a reproduction of the historical CEDIA inference. The source report omits a checkpoint SHA.

The historical Pathology Primary-5 source record is `reports/evidence/pathology-primary5-gold-source.json`. It preserves the CEDIA summary and promotion-manifest hashes, declared checkpoint SHA, labels, and metric definitions. The 90.256% / 66.874% values are present in the source summary, but no row-level targets, predictions, scores, or effective thresholds were available to recalculate them. The separate PTB-XL score table above does not calculate those historical values.

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

The historical CEDIA Primary-9 report gives 476/672 on original labels and has a saved final-reference confusion matrix that recalculates to 605/672, 90.03% accuracy, and 87.82% macro-F1. Its 129-row update table records final labels equal to its model Top-1 values; this does not establish the label-decision process. The fixed repository inference gives 466/672 against original labels and 568/672 against final labels, with complete row-level predictions and hashes; it does not reproduce the CEDIA inference. See the [paired-reference evidence](reports/evidence/primary9-paired-reference-recomputation-2026-10-03.md), [run-reproduction record](reports/evidence/primary9-historical-report-reproduction-2026-10-03.md), and `scripts/verify_primary9_gold_evidence.py`.

The 90.256% / 66.874% Pathology Primary-5 figures are preserved from the CEDIA Gold summary. Its promotion manifest declares checkpoint SHA-256 `843ed16c2a4dd53a193c7f37b9ce75542762ac9606e2701ea7128283e953607d`, but that file and the manifest-linked source run summary were absent at their declared paths on 2026-10-03. No row-level pathology evaluation data were present for recalculation. The separate CEDIA checkpoint result is documented above and is not the historical experiment. See `reports/evidence/pathology-primary5-gold-source.json` and `reports/evidence/cedia-pathology-crosscheck.json`.

## Firmware Build Check

The safe `esp32s3` profile compiled successfully on 2026-09-24 using the project-local ESP32-S3-DevKitC-1-N8R8 profile (8 MB flash and 8 MB octal PSRAM). The pinned environment is PlatformIO Core 6.2.0, Espressif32 platform 6.12.0, Arduino-ESP32 2.0.17, and TensorFlowLite_ESP32 commit `e88e0ebee0430ed716ff5b49854795db90066e59`. Reproduce it with:

```powershell
python -m pip install -r requirements-firmware.txt
python -m platformio run -d hardware/esp32/firmware -e esp32s3
```

The build used 137,232 / 327,680 bytes RAM (41.9%) and 5,517,309 / 6,553,600 bytes of the selected application partition (84.2%). Full evidence, third-party compiler warnings, and the firmware binary SHA-256 are in [the build record](reports/evidence/firmware-build.md). The legacy TFLite library is pinned to make this revision buildable; upstream says it is outdated and no longer recommended. This was a compile only: no board was flashed, no physical inference was run, and embedded preprocessing has not been proven numerically equivalent to the Python pipeline.

Firmware source-level and build-contract tests are part of `python -m pytest -q`. The fixed firmware interpreter lifetime prevents the wrapper from deleting a function-static interpreter it does not own.

## CEDIA Same-Architecture Refinement

The separate 2026-10-12 research bundle includes the candidate PTH, warm-start PTH, exact training source/configuration, frozen teacher/features, source manifests, all 2,171 follow-up probabilities and CEDIA logs. Its six-lead 125 Hz input and 1,143,751-parameter four-head architecture are unchanged. The portable runner was verified from original public WFDB files, with identical Top-1 predictions for all 2,171 cases.

```bash
python scripts/verify_partial_backbone_evidence.py
python scripts/run_refinement_inference.py
python scripts/prepare_refinement_training_inputs.py --output-dir data/cache/refinement_training_inputs
```

The inference runner downloads pinned records and checks source/header/tensor hashes. The preparer reconstructs frozen arrays for a CEDIA GPU refit; no labels are adjusted to match predictions. The [bundle runbook](outputs/reviewer_verification/partial_backbone_refinement_2026-10-12/README.md) explains environments, outputs, class results and the ten validation-only weight-interpolation comparisons. Its `posthoc672` is a new PTB-XL cohort, not the historical Challenge cohort. This bundle does not make the original Gold training lineage reproducible.

## Training-Loss Ablation Replay

The separate [loss-ablation bundle](outputs/reviewer_verification/loss_anchor_ablation_2026-10-12/README.md) records CEDIA job 29344, six predeclared trials, the validation selector, bitwise selected-model refit and unchanged signal/parameter contracts. Its selected PTH improves the current six follow-up comparisons but not the historical final-reference result. Both reference versions and both research models remain available.

```bash
python scripts/verify_loss_ablation_evidence.py
python scripts/prepare_loss_ablation_training_inputs.py --output-dir data/cache/loss_ablation_training_inputs
python scripts/run_refinement_inference.py --checkpoint outputs/reviewer_verification/loss_anchor_ablation_2026-10-12/model/selected.pth --expected-predictions outputs/reviewer_verification/loss_anchor_ablation_2026-10-12/results/followup_predictions.csv --prediction-column phase5_prediction --output data/cache/loss_ablation_reproduced
```

The preparer rebuilt all three original CEDIA NPZ inputs byte-identically. The inference replay matched all 2,171 CEDIA predictions from original public signals on CPU. See the bundle for the measured GPU training runtime, CEDIA submission command, source-file hashes and per-reference historical diagnostics; these are research follow-up runs, not a new independent clinical test.

## Joint Rhythm/Pathology Replay

The [joint fine-tuning runbook](outputs/reviewer_verification/joint_refinement_2026-10-12/README.md) describes corrected job 29350, the 3,083-record training-only expansion, six predeclared comparisons, thirty validation checkpoints and both selection objectives, which returned the same learned state. The complete official rhythm taxonomy masks unsupported statements, including statements absent from the project mapping. Both PTHs preserve the original architecture and were refitted deterministically on CEDIA. No comparison records or reference labels were replaced.

```bash
python scripts/verify_joint_evidence.py
python scripts/verify_joint_sources.py
python scripts/run_joint_inference.py --output data/cache/joint_reproduced
python scripts/run_joint_inference.py --model rhythm_selected --output data/cache/joint_rhythm_reproduced
python scripts/run_joint_historical.py --model rhythm_selected --output data/cache/joint_history_reproduced
```

The source verifier checks the target vectors, SCP codes, complete expansion selection, all 6,166 additional official file hashes and patient exclusions against pinned upstream metadata. The signal replay checks both task outputs from the same checkpoint. Pathology thresholds are fitted only on the 386 PTB-XL validation records, then frozen for the 1,022 PTB-XL follow-up records. Pathology mean binary accuracy and macro-F1 must not be conflated with each other or with nine-class rhythm accuracy. The historical scorer infers on all 672 source signals and computes both unchanged reference versions; it does not reconstruct the unavailable original Gold run.

Training reconstruction commands and the exact GPU stack are provided in the bundle. Original teacher sidecars and the upstream initialization's lineage remain distinct from the fully recorded new fine-tuning run. Published quality/biometric preservation values are output consistency checks, not clinical-performance reports.

## Artifact Inventory

The 2026-10-13 continuation provides three additional sealed experiments:
[balanced joint fitting](outputs/reviewer_verification/balanced_joint_2026-10-13/README.md),
[sequential head fitting](outputs/reviewer_verification/sequential_heads_2026-10-13/README.md),
and [actual compatible-head reconstruction](outputs/reviewer_verification/compatible_heads_2026-10-13/README.md).
The first two retain the phase6 fallback after validation rejection. The third
is a real single checkpoint whose shared tensors match bytewise, not a merger
of two metric tables. Both tasks are inferred from that same PTH. Its rhythm
follow-up improves while pathology declines; neither target achievement nor
joint clinical validity is claimed. Each bundle works with the explicit
`--bundle` argument of the source, prediction, historical and evidence verifiers.

`outputs/reproduced/primary9/` contains:

- `primary9_record_predictions.csv`: one row per record and one probability per class.
- `primary9_recomputed_report.json`: aggregate, class-level, and confusion-matrix metrics plus frozen-input hashes.
- `source_files_sha256.csv`: byte count and SHA-256 for every WFDB file used.
- `run_manifest.json`: environment, preprocessing contract, source/model/input/output hashes, and code-file hashes.

`data/external_validation/record_source_manifest.csv` maps each record to its exact versioned upstream path and release attribution. `data/external_validation/final_external_validation_labels.csv` contains only the minimal label fields needed to evaluate the model.
