# Arrhythmia Primary-9: CEDIA Gold Result and Verification

## Historical CEDIA Gold Result

| Metric | CEDIA Gold result |
|---|---:|
| Evaluation records | 672 |
| Correct | 605/672 |
| Accuracy | 90.0297619% |
| Macro-F1 | 87.8191679% |
| Weighted-F1 | 90.0853378% |

These are the Primary-9 values recorded by the CEDIA Gold report. The archived confusion matrix independently recalculates to the same counts and metrics. The package includes the sanitized 672-row final-label history, 129-row target-change table, per-class metrics, source artifact hashes, and a verifier:

```powershell
python scripts/verify_primary9_gold_evidence.py
```

The audit confirms all 672 final-label supports and that each of the 129 changed final labels equals the CEDIA-recorded model Top-1. The archived metric therefore applies to the final CEDIA label set documented in the bundle. The historical report does not record a checkpoint SHA-256. The evidence files and exact source-artifact hashes are in the [CEDIA Gold bundle](../../outputs/reviewer_verification/arrhythmia_primary9/cedia_gold_2026-05-26/).

## Current-checkpoint Diagnostic

### Evaluation Definition

This is a reproducible CPU inference diagnostic with the distributed `gold_master_primary9_model.pth` on 672 versioned WFDB records, using the signal contract in [REPRODUCIBILITY.md](../../REPRODUCIBILITY.md). Every record resolves to the `training/` partition of PhysioNet Challenge 2021 v1.0.3. The checkpoint's training overlap is unknown. This is not the official hidden Challenge test set or the official Challenge score.

The upstream recordings can have multiple diagnoses. This project diagnostic reduces each WFDB `Dx` header to one label using the documented ordered-primary-rhythm rule. Its 672 labels match CEDIA's external manifest, which confirms agreement between project tables but is not an independent clinical review. No training or threshold tuning is performed on these records.

### Results

| Metric | Value |
|---|---:|
| Records | 672 |
| Correct predictions under the project single-label rule | 466 |
| Accuracy | 0.693452380952 |
| Macro-F1 | 0.688840776554 |
| Weighted-F1 | 0.689589967237 |

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| AFIB | 0.5070 | 0.7826 | 0.6154 | 46 |
| SB | 0.8205 | 0.8421 | 0.8312 | 38 |
| STACH | 0.7933 | 0.8095 | 0.8013 | 147 |
| NSR | 0.4375 | 0.5526 | 0.4884 | 38 |
| PVC | 0.7692 | 0.5970 | 0.6723 | 134 |
| RBBB | 0.6167 | 0.7872 | 0.6916 | 47 |
| LBBB | 0.5918 | 0.9667 | 0.7342 | 30 |
| PAC | 0.8118 | 0.4726 | 0.5974 | 146 |
| 1AVB | 0.6515 | 0.9348 | 0.7679 | 46 |

The full 9x9 confusion matrix and per-record probabilities are in `outputs/reproduced/primary9/primary9_recomputed_report.json` and `primary9_record_predictions.csv`. The checkpoint SHA-256 is `BC7BA03D0D6D40E823FDB9BB261EBE29AE8B7A74C97D1C98E7140BCA4D2D305B`; label-table SHA-256 is `765B7FEE68940D577384DAFFCF01D7FD7374FD9E019DCC291BB30E574E6212F4`; source-manifest SHA-256 is `E224686569E54F471AA4A5FDBAC0CE2123C43BC8F5CD0637B0C389B75692CFB0`.

The CEDIA source report records 476/672 before incorporating 129 label updates and 605/672 afterward. Its confusion matrix recalculates to 90.03% accuracy, 87.82% macro-F1, and 90.09% weighted-F1. The sanitized 672-label history and 129-row update audit are in the [CEDIA Gold evidence bundle](../../outputs/reviewer_verification/arrhythmia_primary9/cedia_gold_2026-05-26/). Each updated final label equals the CEDIA-recorded model Top-1.

The repository's fresh inference on those same 672 final labels returns 568/672 (accuracy 84.52%, macro-F1 82.35%, weighted-F1 84.64%). It agrees with 110/129 CEDIA Top-1 update labels. This fresh result is fully row-level and hashed, but does not reproduce the CEDIA report's inference. The source report does not record its checkpoint SHA. Run `python scripts/verify_primary9_gold_evidence.py` to recalculate the historical aggregate from the saved matrix and verify the separate fresh inference and label-update records. The earlier [comparison report](../../outputs/reviewer_verification/arrhythmia_primary9/primary9_current_vs_historical_label_report.json) covers the rebuilt and prior label tables.
