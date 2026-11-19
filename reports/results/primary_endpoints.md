# Primary Endpoint Results

This page collects the historical Primary-9 arrhythmia and Primary-5 pathology results with links to the versioned evidence stored in this repository.

## Arrhythmia Primary-9

The CEDIA source report separates two references for the same 672 ECG records across AFIB, SB, STACH, NSR, PVC, RBBB, LBBB, PAC, and 1AVB:

| Reference | Correct | Accuracy | Macro-F1 | Weighted-F1 | Source evidence |
|---|---:|---:|---:|---:|---|
| Original labels | 476/672 | 70.83% | 70.68% | 70.70% | Aggregate reported by CEDIA; original confusion matrix and row-level historical predictions are not in the report. |
| Final updated labels | 605/672 | 90.03% | 87.82% | 90.09% | Recalculated from the CEDIA report's saved confusion matrix. |

The **90.03% / 87.82%** result is on the final updated labels, not the original reference and not an independently reproduced signal-to-prediction run. The exact pre-update source manifest is `DATASETS_CURADOS/RHYTHM_PRIMARY_V2/manifest_external_test.csv` (SHA-256 `8190ec19f4887d4618213cbf95b9da3b68948dddba4f0233d975eba5938e5b8c`); all 672 source IDs and `rhythm_label_name` values were matched to the sanitized `original_rhythm_label` column. The raw manifest is not distributed because it contains demographics and absolute paths. The 672-row sanitized history is included in the evidence bundle.

- [Historical aggregate JSON](../../outputs/gold_master_external_validation/arrhythmia_primary9/primary9_external_validation_report.json)
- [Historical per-class metrics](../../outputs/gold_master_external_validation/arrhythmia_primary9/primary9_per_class_metrics.csv)
- [672-row record-level comparison table](../../outputs/reviewer_verification/arrhythmia_primary9/primary9_current_vs_historical_label_predictions.csv)
- [Comparison metrics and provenance](../../outputs/reviewer_verification/arrhythmia_primary9/primary9_current_vs_historical_label_report.json)
- [CEDIA Gold evidence bundle: 672 labels, 129 target updates, and fresh inference](../../outputs/reviewer_verification/arrhythmia_primary9/cedia_gold_2026-05-26/README.md)
- [Paired-reference recalculation and source cross-check](../evidence/primary9-paired-reference-recomputation-2026-10-03.md)
- [Historical run recovery and reproduction requirements](../evidence/primary9-historical-report-reproduction-2026-10-03.md)
- [Generated paired metrics, row-level predictions, and checksums](../../outputs/reviewer_verification/arrhythmia_primary9/paired_reference_recomputation_2026-10-03/README.md)
- [Historical aggregate verifier](../../scripts/verify_primary9_gold_evidence.py)
- [Evaluation method and reproduction steps](arrhythmia_primary9_external.md)

The historical CEDIA report states that 129 records were updated. The sanitized update file shows all 129 final labels equal their recorded CEDIA model Top-1 as values; that equality alone does not establish how the changes were decided. The report matrix and per-class metrics are preserved with source SHA-256 values; `python scripts/verify_primary9_gold_evidence.py` recalculates 605/672 and macro-F1 0.878191679 from the final-reference matrix, verifies all 672 final-label supports, and checks the update rows. The original-reference aggregate is reported by CEDIA without a saved matrix.

A fresh inference using the repository's Primary-9 checkpoint (SHA-256 `BC7BA03D0D6D40E823FDB9BB261EBE29AE8B7A74C97D1C98E7140BCA4D2D305B`) gives **466/672** against original labels and **568/672** against final labels. Accuracy, macro-F1, per-class metrics, both confusion matrices, record-level predictions, input hashes, and the run manifest are in the paired evidence bundle. This checkpoint agrees with 110 of the 129 CEDIA Top-1 update values. The CEDIA source report does not record the checkpoint hash or historical prediction rows, so the fixed local run is distinct from the archived 605/672 evaluation.

To regenerate the current-checkpoint predictions from the pinned public records and compare them with both label tables, follow [REPRODUCIBILITY.md](../../REPRODUCIBILITY.md). The source records are from PhysioNet Challenge 2021 v1.0.3; dataset citation and licensing details are in [REFERENCES.md](../../REFERENCES.md) and [DATA_PROVENANCE.md](../../DATA_PROVENANCE.md).

## Pathology Primary-5

The CEDIA source summary reports **90.256% mean per-label accuracy** and **66.874% macro-F1** for IMI, ALMI, ILMI, LAE, and ISC_. Per-label accuracy is the unweighted mean of the five binary-label accuracies.

- [Historical aggregate JSON](../../outputs/gold_master_external_validation/pathology_primary5/pathology_primary5_external_summary.json)
- [CEDIA source and checkpoint provenance](../evidence/pathology-primary5-gold-source.json)
- [Pathology result and related evidence](pathology_primary5_external.md)
- [Evidence reconciliation](../evidence/metric-reconciliation.md)

The source summary and promotion manifest hashes are recorded in the linked provenance file. The manifest declares checkpoint SHA-256 `843ed16c2a4dd53a193c7f37b9ce75542762ac9606e2701ea7128283e953607d`; that checkpoint and the source run summary were absent from their declared CEDIA paths during verification. The 90.256% / 66.874% values are preserved as the CEDIA-reported result, not recalculated from row-level targets and predictions. Separately, the repository includes CEDIA pathology scores for 273 calibration and 406 test ECGs; the [evaluation package](../../outputs/reviewer_verification/pathology_primary5/cedia_ptbxl_v1_0_3_20260925/README.md) documents that distinct experiment and its recomputation command.

## Same-Architecture Refinement Study

A separate CEDIA run warm-started from the phase-2 checkpoint and updated only the original final residual block and rhythm head. It retained the six-lead, 125 Hz, `[6,1250]` input and 1,143,751-parameter four-head model. The candidate reached 1,185/1,476 validation accuracy (80.28%) and 78.93% macro-F1, compared with 1,182/1,476 (80.08%) and 78.79% for its phase-2 start. On the previously examined 2,171-record follow-up, it scored 1,712/2,171 (78.86%) versus 1,713/2,171 (78.90%) for phase 2. It is retained as a research candidate and does not replace Gold.

The [CEDIA refinement evidence bundle](../../outputs/reviewer_verification/partial_backbone_refinement_2026-10-12/README.md) contains the candidate checkpoint, per-record probabilities, confusion matrices, class recalls, record manifests, training configuration, execution logs, and hashes. `python scripts/verify_partial_backbone_evidence.py` verifies the bundle and recalculates metrics; `python scripts/run_refinement_inference.py` repeats inference from pinned public signals. Portable CPU inference reproduced all 2,171 CEDIA Top-1 predictions, and the training-input preparer rebuilt all three NPZ inputs byte-identically. Ten complementary weight mixtures were evaluated in validation without replacing the base candidate. This experiment is distinct from, and does not reproduce, the historical 605/672 aggregate.

### Training-loss ablation

CEDIA job 29344 selected a strict warm-start update of the two final residual blocks and rhythm head using validation only. It retained 1,143,751 parameters and the original six-lead 125 Hz input. Validation increased from 1,185/1,476 (80.28% accuracy; 78.93% macro-F1) to 1,218/1,476 (82.52%; 81.03%). The fixed follow-up comparisons increased from 1,712/2,171 to 1,765/2,171, adding 53 correct predictions. All six cohort scores improved relative to the preceding candidate.

On the separate historical Challenge cohort, original-reference correct predictions increased from 479 to 493, while final-reference correct predictions decreased from 576 to 571. This does not achieve or recreate the reported historical 605/672 result. The [loss-ablation runbook](../../outputs/reviewer_verification/loss_anchor_ablation_2026-10-12/README.md) includes both scores, full per-record outputs, all candidate validation arrays, the selected/initial PTHs, bitwise refit evidence and portable signal replay. The prior candidate and Gold remain unchanged.

## Expanded Native-Population Continuation

The [2026-10-13 expanded-population study](../../outputs/reviewer_verification/expanded_population_2026-10-13/README.md), CEDIA job 29418, preserves all original model dimensions and reference labels. Its rhythm-selected checkpoint scores 1,801/2,171 (82.96% accuracy, 82.41% macro-F1) in follow-up. Its distinct joint-selected checkpoint scores 1,797/2,171 (82.77%, 82.23%) and 94.56% mean label accuracy / 48.40% pathology macro-F1 on 1,022 native ECGs. The equally recalibrated control scores 46.45% pathology macro-F1; the prior 386-case-calibration score was 45.39%. Neither candidate is promoted over Gold or represented as a reproduction of the original historical aggregates. Original signals reproduce all final decisions, and the bundle contains the exact PTHs, expanded population, candidate validations, scripts, logs, both 672-reference comparisons and hashes.

## Independent-Head Supervision Controls

The [2026-10-14 six-arm study](../../outputs/reviewer_verification/supervision_control_2026-10-14/README.md), CEDIA job 29488, preserves the original architecture, patient/source populations and reference labels. Separate existing-head fits retain the phase10 rhythm state and select a new pathology state using patient-group threshold-cross-fit validation. The actual composed PTH reproduces **1,801/2,171, 82.96% rhythm accuracy / 82.41% macro-F1**, with **94.58% mean pathology label accuracy / 46.16% macro-F1** on the same 1,022 native ECGs. Its equally calibrated warm-start control gives 42.64% pathology macro-F1. This 3.52-point gain does not exceed the earlier phase6 pathology score of 48.51%, which remains preserved. Both new model roles score 494/672 against original historical references and 567/672 against final references; neither recreates the historical 605/672 run.

## Dataset and artifact citation

For the 672-record arrhythmia evaluation, cite PhysioNet/Computing in Cardiology Challenge 2021 v1.0.3, DOI `10.13026/34va-7q14`. For the separate CEDIA pathology evaluation, cite PTB-XL v1.0.3. Complete bibliographic entries and license scope are in [REFERENCES.md](../../REFERENCES.md) and [LICENSE-ARTIFACTS.md](../../LICENSE-ARTIFACTS.md).
