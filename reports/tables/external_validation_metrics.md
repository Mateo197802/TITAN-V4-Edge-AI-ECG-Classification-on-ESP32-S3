# Metric Evidence Reconciliation

These figures use different checkpoints, units and protocols; they are not directly comparable. The 672-record single-label diagnostic is not the official PhysioNet Challenge score or verified independent external performance. See [the evidence reconciliation](../evidence/metric-reconciliation.md).

| Evidence | Unit | Accuracy | Macro-F1 | Weighted-F1 | Interpretation |
|---|---:|---:|---:|---:|---|
| CEDIA internal validation | 58,855 windows | 0.89836 | 0.73898 | 0.90301 | Different checkpoint; internal window-level aggregate |
| Distributed-checkpoint selection summary | 706 windows | n/a | 0.89478 | n/a | Stored best-epoch metric; validation predictions unavailable |
| Recomputed Primary-9 diagnostic | 672 records | 0.69345 | 0.68884 | 0.68959 | Reproducible project single-label diagnostic; not official Challenge score |
| Current checkpoint vs prior Primary-9 labels | 672 records | 0.84524 | 0.82352 | 0.84641 | 568/672 correct; same current predictions vs old labels; not a reproduction of the historical 605/672 aggregate |
| Historical Primary-9 aggregate | 672 rows | 0.90030 | 0.87819 | 0.90085 | Unverified; 129 label differences and no original prediction-level artifact |
| Pathology Primary-5 aggregate | 5 binary labels | 0.90256 | 0.66874 | n/a | Legacy per-label accuracy; target/prediction rows and checkpoint/threshold lineage unavailable |
| Cascade/OOD | 211 records | n/a | n/a | n/a | Historical safety annex only; coverage=0.5071 |
