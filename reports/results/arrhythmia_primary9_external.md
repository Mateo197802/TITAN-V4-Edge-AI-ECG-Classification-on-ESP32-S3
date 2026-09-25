# Arrhythmia Primary-9 Recomputed Inference

## Evaluation Definition

This is CPU inference with the distributed `gold_master_primary9_model.pth` on the frozen 672-record label set, using the versioned WFDB signals and preprocessing contract in [REPRODUCIBILITY.md](../../REPRODUCIBILITY.md). Every record resolves to the `training/` partition of PhysioNet Challenge 2021 v1.0.3. This is not the official hidden Challenge test set, and the checkpoint training overlap is not established.

Labels are reconstructed from the WFDB header `Dx` codes using the documented ordered-primary-rhythm rule. The 672 labels match the read-only CEDIA external manifest. No training or threshold tuning is performed on this evaluation set.

## Results

| Metric | Value |
|---|---:|
| Records | 672 |
| Correct predictions | 466 |
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

The former 605/672 (0.90030 accuracy) result is superseded: its bundled labels did not agree with the source-header label policy for 129 records. It remains only in the legacy folder and is not a current result.
