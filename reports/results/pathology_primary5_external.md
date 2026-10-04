# Pathology Primary-5: CEDIA Gold Result and Fresh Measurement

## Historical CEDIA Gold Result

The CEDIA Gold summary reports **90.256% mean per-label accuracy** and **66.874% macro-F1** for IMI, ALMI, ILMI, LAE, and ISC_. Mean per-label accuracy is the unweighted average of the five binary-label accuracies. The summary and promotion-manifest hashes, declared checkpoint identity, and metric definition are preserved in the [machine-readable provenance record](../evidence/pathology-primary5-gold-source.json).

## Separate PTB-XL Measurement

### Fresh checkpoint measurement

The preserved CEDIA checkpoint `e9a44e4eea8ebb8f89d5e32909ae4afcc96442d1fa8eacb1e57a73bfa498353b` was loaded strictly as `TitanV4Max` (1024-dimensional encoder, 9 rhythm outputs, 10 pathology outputs, no morphology fusion). Its training summary records pathology loss weight 0.2 and 1,811 pathology-labeled windows; the checkpoint was selected at epoch 25 by rhythm F1, not by pathology performance.

The evaluation used PTB-XL v1.0.3 metadata and available `records100` signals. The database and SCP mapping checksums are pinned in the [machine-readable report](../../outputs/reviewer_verification/pathology_primary5/cedia_ptbxl_v1_0_3_20260925/pathology_primary5_report.json). The frozen ten-class mapping reconstructed all 2,580 labeled sidecar records exactly. Primary-5 consists of IMI, ALMI, ILMI, LAE, and ISC_. Per-label accuracy is the unweighted mean of five binary-label accuracies, not ordinary record-level accuracy.

| Threshold policy | Calibration records | Test records | Mean per-label accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| Per-class thresholds selected for F1 on calibration only | 273 | 406 | 71.58% | 41.37% |
| Fixed threshold 0.50 | none | 406 | 73.15% | 41.76% |
| Fixed threshold 0.65 | none | 406 | 79.90% | 42.03% |

The calibration thresholds were fixed before scoring the test rows. They do not approach the historical pair of 90.26% per-label accuracy and 66.87% macro-F1. Raising a common threshold to 0.65 increases accuracy while leaving macro-F1 near 42%; no test-set threshold was searched to force the historical values.

The split metadata contained 118 training and 10 validation `HR#####` PTB-XL aliases in addition to numeric `records100` paths. Alias IDs were resolved to PTB-XL `ecg_id`s and included in overlap checks. Seven record IDs occurred in both the checkpoint's train and validation lists; those patients were excluded from calibration and test. Test patients are disjoint from both checkpoint splits. Two calibration records, 03690 and 16200, failed WFDB input reading with `TypeError`; their input hashes and status are in `pathology_input_records.csv`. No test signal failed preflight.

## Historical result

The CEDIA Gold summary reports **90.256% mean per-label accuracy** and **66.874% macro-F1** for IMI, ALMI, ILMI, LAE, and ISC_. The source summary and promotion-manifest SHA-256 values, the declared checkpoint SHA, and the source-reported metric definition are recorded in [the Primary-5 provenance record](../evidence/pathology-primary5-gold-source.json). The declared checkpoint and source run summary were not present at their declared paths during the read-only CEDIA check; no record-level targets, predictions, scores, cohort size, or actual thresholds were available for recalculation. These two figures are therefore preserved as a source-reported result, not as a fresh model evaluation.

The CEDIA checkpoint measured above is a separate model, cohort, and protocol and does not reproduce the historical figures. The distributed repository checkpoint is another model lineage; its training summary records zero pathology-labeled windows. Do not attribute either measured result to the historical aggregate.

## Recompute and evidence

The evidence directory contains record-level truth/scores, test predictions, per-class metrics, hashes for every input signal/header, a frozen label-selection file, the full JSON report, and `SHA256SUMS.csv`. Recompute the calibrated and fixed-threshold metrics from the checked-in score rows with:

```powershell
python src/titan_v4/evaluation/pathology_primary5_protocol.py `
  --predictions_csv outputs/reviewer_verification/pathology_primary5/cedia_ptbxl_v1_0_3_20260925/pathology_primary5_scores.csv `
  --out_dir <temporary-directory> `
  --fixed-threshold 0.50 --fixed-threshold 0.65
```

The aggregation reproduces the checked-in metrics within floating-point tolerance. Full signal-to-prediction inference is not standalone from this repository: the 87.7 MB CEDIA checkpoint, CEDIA source tree, and signal files are not redistributed. Their hashes and the reference runtime are recorded in the JSON report. The model's redistribution rights must be confirmed before publishing its weights.

This is an exploratory, post hoc, patient-disjoint subset evaluation within PTB-XL, not source-held-out or clinical validation. A confirmatory publication claim requires a prespecified untouched test cohort and authorized access to the exact checkpoint and source.

See [metric reconciliation](../evidence/metric-reconciliation.md), [dataset attribution](../../REFERENCES.md), and the [historical readiness audit](../../outputs/reviewer_verification/pathology_primary5/pathology_reproducibility_audit.json).
