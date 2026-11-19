# Metric Formulas And Validation Definitions

The current computed Primary-9 report is `outputs/reproduced/primary9/primary9_recomputed_report.json`. It is a reproduced 672-record project-specific diagnostic, not independent external validation. `scripts/compare_primary9_historical_labels.py` evaluates those exact predictions against the prior labels as a separate sensitivity-to-label-table check. Pathology Primary-5 and Cascade/OOD formulas remain relevant to archived reports, but the historical scores are not reproduced without matched record-level targets and predictions.

## Accuracy

For multiclass rhythm classification:

```text
accuracy = correct_predictions / total_predictions
```

For multilabel pathology classification, per-label accuracy is:

```text
accuracy_l = (TP_l + TN_l) / (TP_l + TN_l + FP_l + FN_l)
```

Primary pathology accuracy is the mean over the Primary-5 labels.

## F1

For a class or label:

```text
precision = TP / (TP + FP)
recall = TP / (TP + FN)
F1 = 2 * precision * recall / (precision + recall)
```

Macro-F1 is the arithmetic mean of class-level F1 values:

```text
macro_F1 = mean(F1_c for c in classes)
```

## Safety Annex

Cascade/OOD outputs are summarized separately:

```text
coverage = accepted_predictions / diagnostic_subset_records
quarantine_rate = quarantined_records / diagnostic_subset_records
```

These metrics describe triage behavior and are not used as the primary performance source.

