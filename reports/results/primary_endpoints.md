# Primary Endpoint Results

This page collects the historical Primary-9 arrhythmia and Primary-5 pathology results with links to the versioned evidence stored in this repository.

## Arrhythmia Primary-9

The archived result reports **605/672 correct**, **90.03% accuracy**, **87.82% macro-F1**, and **90.09% weighted-F1** across the nine project labels: AFIB, SB, STACH, NSR, PVC, RBBB, LBBB, PAC, and 1AVB.

- [Historical aggregate JSON](../../outputs/gold_master_external_validation/arrhythmia_primary9/primary9_external_validation_report.json)
- [Historical per-class metrics](../../outputs/gold_master_external_validation/arrhythmia_primary9/primary9_per_class_metrics.csv)
- [672-row record-level comparison table](../../outputs/reviewer_verification/arrhythmia_primary9/primary9_current_vs_historical_label_predictions.csv)
- [Comparison metrics and provenance](../../outputs/reviewer_verification/arrhythmia_primary9/primary9_current_vs_historical_label_report.json)
- [Evaluation method and reproduction steps](arrhythmia_primary9_external.md)

The 672-row table includes record IDs, the historical and rebuilt labels, predictions and probabilities from the distributed checkpoint, and correctness under each label table. Those predictions score 568/672 against the historical labels. The table is a current-checkpoint comparison, not the original prediction file behind the archived 605/672 aggregate. The archived aggregate is retained as reported; it cannot be recalculated from that comparison table.

To regenerate the current-checkpoint predictions from the pinned public records and compare them with both label tables, follow [REPRODUCIBILITY.md](../../REPRODUCIBILITY.md). The source records are from PhysioNet Challenge 2021 v1.0.3; dataset citation and licensing details are in [REFERENCES.md](../../REFERENCES.md) and [DATA_PROVENANCE.md](../../DATA_PROVENANCE.md).

## Pathology Primary-5

The archived result reports **90.256% mean per-label accuracy** and **66.874% macro-F1** for IMI, ALMI, ILMI, LAE, and ISC_. Per-label accuracy is the unweighted mean of the five binary-label accuracies.

- [Historical aggregate JSON](../../outputs/gold_master_external_validation/pathology_primary5/pathology_primary5_external_summary.json)
- [Pathology result and related evidence](pathology_primary5_external.md)
- [Evidence reconciliation](../evidence/metric-reconciliation.md)

The historical JSON preserves the aggregate values and label set. The checked-in repository does not contain a row-level prediction/target table for that historical result. Separately, the repository includes a CEDIA pathology score table for 273 calibration and 406 test ECGs; its protocol, record-level files, and recomputation command are documented in the [CEDIA evaluation package](../../outputs/reviewer_verification/pathology_primary5/cedia_ptbxl_v1_0_3_20260925/README.md). That is a distinct evaluation and is not used to calculate the historical figures above.

## Dataset and artifact citation

For the 672-record arrhythmia evaluation, cite PhysioNet/Computing in Cardiology Challenge 2021 v1.0.3, DOI `10.13026/34va-7q14`. For the separate CEDIA pathology evaluation, cite PTB-XL v1.0.3. Complete bibliographic entries and license scope are in [REFERENCES.md](../../REFERENCES.md) and [LICENSE-ARTIFACTS.md](../../LICENSE-ARTIFACTS.md).
