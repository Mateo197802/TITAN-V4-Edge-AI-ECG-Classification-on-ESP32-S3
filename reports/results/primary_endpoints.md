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

## Dataset and artifact citation

For the 672-record arrhythmia evaluation, cite PhysioNet/Computing in Cardiology Challenge 2021 v1.0.3, DOI `10.13026/34va-7q14`. For the separate CEDIA pathology evaluation, cite PTB-XL v1.0.3. Complete bibliographic entries and license scope are in [REFERENCES.md](../../REFERENCES.md) and [LICENSE-ARTIFACTS.md](../../LICENSE-ARTIFACTS.md).
