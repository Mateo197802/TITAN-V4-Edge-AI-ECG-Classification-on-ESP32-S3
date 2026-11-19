# NeuroGuardian TITAN V4 ECG

Reproducible inference and evidence package for the distributed TITAN V4 Primary-9 checkpoint, with ESP32-S3 firmware and archived engineering reports. This repository is research software and is not a medical device.

## Primary Results

| Endpoint | Historical reported result | Evaluation unit |
|---|---|---|
| Arrhythmia Primary-9, original labels | CEDIA report: 476/672; accuracy 70.83%; macro-F1 70.68%; weighted-F1 70.70% | 672 ECG records |
| Arrhythmia Primary-9, final updated labels | CEDIA report: 605/672; accuracy 90.03%; macro-F1 87.82%; weighted-F1 90.09% | Same 672 ECG records |
| Pathology Primary-5 | CEDIA summary: mean per-label accuracy 90.256%; macro-F1 66.874% | Five binary pathology labels |

The source aggregates, endpoint definitions, record-level evidence, and verification commands are indexed in [Primary Endpoint Results](reports/results/primary_endpoints.md). The [historical Primary-9 runbook](reports/evidence/primary9-historical-report-reproduction-2026-10-03.md) distinguishes aggregate reconstruction from an exact model rerun. The 90.03% / 87.82% arrhythmia result is against the final updated labels, not the original reference labels. The report matrix recalculates to 605/672; the repository checkpoint's packaged inference scores 466/672 against original labels and 568/672 against final labels. These are distinct prediction runs. The sanitized 672-row label history identifies both references, and the 129 source updates are published separately. The 90.256% / 66.874% pathology aggregate remains a source-reported result; its separate CEDIA PTB-XL score table is not a reproduction of that historical metric.

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
python scripts/recompute_primary9_paired_reference.py
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

The legacy-label comparator writes the current model's 672-row predictions against the rebuilt and prior label tables. The paired-reference script scores the same predictions against the CEDIA original and final reference columns and writes JSON, per-class CSV, and hashes. The Gold evidence verifier recalculates the archived 605/672 result from its saved final-label confusion matrix and validates the target-update audit. The CEDIA report gives only aggregate values for the original-reference result (476/672), without a matrix or row-level predictions. The historical Pathology-5 values remain source-reported; the separate CEDIA PTB-XL score table is recomputable from its per-record probabilities and does not calculate those historical values.

## Same-Architecture Research Update

The [pretrained transfer study](outputs/reviewer_verification/founder_transfer_2026-10-14/README.md) includes 30 ECGFounder teacher trials, nine student fits across three seeds and deterministic training repeats. The twelve-lead rhythm teacher reaches 88.80% mean accuracy / 87.15% mean macro-F1 on validation, but the continued six-lead student comparisons do not improve the fixed deployment control. The original 1,143,751-parameter student is retained. CEDIA source replay gives **82.96% rhythm accuracy / 82.41% macro-F1** on 2,171 follow-up ECGs and **94.58% mean label accuracy / 46.16% pathology macro-F1** on 1,022 native ECGs. The packet preserves all trials, the 672 historical case outputs, exact executed-code snapshots, allocation logs and hashes. Teacher scores are not attributed to the deployed student; Gold remains unchanged.

```bash
python scripts/verify_founder_transfer.py --output data/cache/founder_source_verification.json
```

The [2026-10-12 CEDIA refinement](outputs/reviewer_verification/partial_backbone_refinement_2026-10-12/README.md) preserves the original 1,143,751 parameters, four heads, six leads and 125 Hz input. It includes a separate candidate PTH, 12,040 pinned train/validation/follow-up record entries, frozen teacher sidecars, CEDIA source/logs, all 2,171 follow-up predictions, and a portable signal-to-prediction runner. CPU inference from the original WFDB files reproduced all 2,171 CEDIA predictions. Validation accuracy is 80.28%; follow-up accuracy is 78.86%. This is a new research experiment, not a reproduction of the historical 90.03% result, and Gold is unchanged.

```bash
python scripts/verify_partial_backbone_evidence.py
python scripts/run_refinement_inference.py
```

## Evidence Boundaries

The [plateau diagnosis](outputs/reviewer_verification/plateau_audit_2026-10-14/README.md), CEDIA job 29490, performs read-only train/validation inference with three retained checkpoints. It quantifies class/source mix, subtype support, shared errors and pathology ranking/decision metrics, and examines 140 prior candidate entries. Current native IRBBB recall is only 274/604 in training versus 17/41 in validation; the same model recognizes 201/212 and 14/15 CRBBB. The three models share 185 validation errors, and their retrospective target-knowing oracle reaches only 87.47%, not a deployable result. The report specifies separate representation and voltage-preserving input experiments without claiming they have run or changed Gold.

The [independent-head and supervision comparison](outputs/reviewer_verification/supervision_control_2026-10-14/README.md), completed CEDIA job 29488, executes six controlled continuations without changing the original architecture or any comparison reference. Patient-group threshold-cross-fit pathology validation macro-F1 rises from 39.74% to 43.23%. Its actual composed checkpoint preserves rhythm followup at **82.96% accuracy / 82.41% macro-F1** and increases pathology followup from **42.64% to 46.16% macro-F1**, with **94.58% mean label accuracy**. The earlier 48.51% pathology checkpoint remains stronger and preserved. The bundle includes all 24 learned states, source controls, complete predictions and exact selection/refit evidence; it does not attain the historical targets.

```bash
python scripts/verify_supervision_evidence.py --bundle outputs/reviewer_verification/supervision_control_2026-10-14
python scripts/run_joint_inference.py --bundle outputs/reviewer_verification/supervision_control_2026-10-14 --output data/cache/supervision_joint_replay
```

The [directed-source and PCGrad comparison](outputs/reviewer_verification/directed_pcgrad_2026-10-13/README.md), completed CEDIA job 29484, examines all 45,152 official ECG-arrhythmia headers and adds 190 non-overlapping difficult-class training records. It compares frozen control, last-block fitting, progressive fitting and shared PCGrad without changing the original architecture or comparison references. PCGrad raises pathology validation macro-F1 from 43.82% to 45.00% while rhythm declines; neither selector replaces the starting model. The retained joint model reproduces **82.77% rhythm accuracy / 82.23% macro-F1** and **94.56% mean pathology label accuracy / 48.40% macro-F1** on the fixed followup. Earlier stronger task-specific models and Gold remain preserved. Every candidate, full-row output and cancelled-attempt receipt is included; this comparison does not achieve 90% / 87%.

```bash
python scripts/verify_directed_evidence.py --bundle outputs/reviewer_verification/directed_pcgrad_2026-10-13
python scripts/run_joint_inference.py --bundle outputs/reviewer_verification/directed_pcgrad_2026-10-13 --output data/cache/directed_joint_replay
```

The [expanded-population continuation](outputs/reviewer_verification/expanded_population_2026-10-13/README.md), CEDIA job 29418, adds 10,901 native Primary-5-negative training records and expands pathology calibration to 2,027 patient-disjoint fold-9 ECGs. It keeps the original architecture, six leads, 125 Hz and all comparison references. The rhythm-selected PTH reaches **82.96% accuracy / 82.41% macro-F1** on the fixed 2,171-case follow-up (one net additional correct case). The separate joint-selected PTH reaches **94.56% mean label accuracy / 48.40% pathology macro-F1**, versus 93.87% / 46.45% for the equally recalibrated frozen control; its rhythm accuracy is 82.77%. These are different real checkpoints, not combined best-head metrics. Both candidates and all earlier models are retained; the experiment does not achieve 90% / 87% or replace Gold.

```bash
python scripts/verify_expanded_evidence.py --bundle outputs/reviewer_verification/expanded_population_2026-10-13
python scripts/run_joint_inference.py --bundle outputs/reviewer_verification/expanded_population_2026-10-13 --output data/cache/expanded_joint_reproduced
```

The [compatible-head checkpoint](outputs/reviewer_verification/compatible_heads_2026-10-13/README.md) is an actual single PTH reconstructed and evaluated in CEDIA job 29407 after twelve continued-fit controls in jobs 29368 and 29384. It preserves 1,143,751 parameters, four heads, six frontal leads and 125 Hz. Its fixed 2,171-record rhythm follow-up reaches **1,800/2,171, 82.91% accuracy / 82.10% macro-F1**, adding 16 correct predictions to phase6. Primary-5 follow-up instead declines to **93.13% mean label accuracy / 45.39% macro-F1**, versus phase6 94.44% / 48.51%. It is not promoted as an improvement of both tasks, and Gold/reference models remain unchanged. All rejected validation candidates, actual model predictions and source/protocol hashes are preserved.

```bash
python scripts/verify_joint_evidence.py --bundle outputs/reviewer_verification/compatible_heads_2026-10-13
python scripts/run_joint_inference.py --bundle outputs/reviewer_verification/compatible_heads_2026-10-13 --output data/cache/compatible_joint_reproduced
```

The [joint rhythm/pathology experiment](outputs/reviewer_verification/joint_refinement_2026-10-12/README.md) completed its corrected CEDIA run, job 29350, on 2026-10-13. It added 3,083 training-only PTB-XL records and compared six fine-tuning strategies, without resetting the network, changing comparison references, or changing the original 1,143,751 parameters, four heads, six leads and 125 Hz input. Full official rhythm-taxonomy checks corrected 56 ambiguous additional training targets before this rerun. Both validation objectives selected the same learned state: **84.28% accuracy / 83.37% macro-F1 in validation**, **82.17% / 81.53% on the 2,171 fixed follow-up records**, and **94.44% mean per-label accuracy / 48.51% macro-F1 for Primary-5 on 1,022 PTB-XL records**. This does not achieve the overall rhythm 90% / approximately 87% target or improve the historical final-reference score; no result is promoted over Gold.

```bash
python scripts/verify_joint_evidence.py
python scripts/run_joint_inference.py --model rhythm_selected --output data/cache/joint_rhythm_reproduced
```

The [CEDIA training-loss ablation](outputs/reviewer_verification/loss_anchor_ablation_2026-10-12/README.md) executed six trials on the same original architecture. Its validation-selected candidate reached 82.52% accuracy / 81.03% macro-F1 and added 53 correct predictions across the fixed 2,171-record follow-up. Against historical final references it scored 571/672, versus the prior candidate's 576/672; both PTHs are preserved. The bundle includes training/refit evidence, all candidate validation probabilities, public-signal replay commands and both historical reference scores.

The separate [historical-cohort target-gap audit](outputs/reviewer_verification/gold_target_gap_2026-10-12/README.md) reevaluates the research candidate on the actual 672 Challenge records: 576/672 (85.71% accuracy, 83.81% macro-F1) against the archived final references. It also identifies an exact CE/KL objective conflict in 1,007 training records and documents the unrun ablation needed to test its effect. This is not a reproduction of the historical 605/672 prediction run.

- The 672-record single-label diagnostic is computationally reproducible from this checkout and the cited public dataset release; it is not an official Challenge metric or an independent external-validation claim.
- The CEDIA pathology score is recomputable from checked-in targets and probabilities. It is a patient-disjoint subset of PTB-XL v1.0.3, not a source-held-out estimate or a reproduction of the historical aggregate. End-to-end inference is not standalone until the checkpoint and the exact CEDIA source files are made available under an approved redistribution basis.
- Training the checkpoint from scratch is not currently a reproducible claim: the exact training/validation manifests and offline teacher-sidecar arrays are not included.
- The historical CEDIA Primary-9 report records 476/672 against its original reference and 605/672 against its final updated reference. Only the latter has a saved confusion matrix. All 672 original labels were matched to the pre-update CEDIA manifest; the raw manifest is not redistributed because it contains demographic fields and absolute paths. The fixed repository checkpoint gives 466/672 on original labels and 568/672 on final labels. Source evidence and paired calculations are in `outputs/reviewer_verification/arrhythmia_primary9/`.
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
| `outputs/reviewer_verification/arrhythmia_primary9/paired_reference_recomputation_2026-10-03/` | Fixed predictions scored against both the original and final reference labels, with per-class metrics and checksums |
| `outputs/reviewer_verification/pathology_primary5/cedia_ptbxl_v1_0_3_20260925/` | CEDIA checkpoint pathology predictions, per-record scores, input hashes, and recomputable metrics |
| `reports/` | Methods, results, and evidence traceability |
| `hardware/esp32/` | Firmware, conversion tools, and engineering artifacts |

## Citation and License

Dataset users must cite the exact versioned PhysioNet releases used for each analysis, including PTB-XL v1.0.3 for the CEDIA pathology evaluation and Challenge 2021 v1.0.3 for the Primary-9 diagnostic. PhysioNet lists CC BY 4.0 for these dataset files. TITAN source code is MIT-licensed; project-authored research artifacts and model weights have a separate CC BY 4.0 notice. Neither project license relicenses third-party datasets or recordings. See [license scope](LICENSE_SCOPE.md), [artifact license](LICENSE-ARTIFACTS.md), [dataset provenance](DATA_PROVENANCE.md), and [references](REFERENCES.md).

The result is for research use only. It must not be used to diagnose, treat, or rule out a medical condition.
