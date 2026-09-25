# Historical Aggregate Outputs (Legacy)

These files are retained to preserve the repository's earlier evidence snapshot. They are not the current reproduced evaluation and must not be used by the current reporting commands.

- The old Primary-9 report records 605/672 correct (accuracy 0.90030, macro-F1 0.87819). Its previous label table disagrees with the rebuilt project labels on 129 of 672 rows. Scoring the current checkpoint's predictions against that old table gives 568/672, not 605/672. No original historical predictions or exact evaluation lineage are available, so the old result remains unverified and has not been proven false. See `outputs/reviewer_verification/arrhythmia_primary9/`.
- The old Pathology Primary-5 aggregate (reported per-label accuracy 0.90256, macro-F1 0.66874) has no complete record-level predictions, reference rows, checkpoint hash, or threshold lineage and has not been reproduced. CEDIA contains a separately supervised pathology model and target map, but they are not tied to this aggregate; see `reports/evidence/cedia-pathology-crosscheck.json`.
- The old Cascade/OOD report is a 211-record safety annex and has not been recomputed in the current workflow.
- The old validation CSV includes metadata beyond the minimal current label table. Use `data/external_validation/final_external_validation_labels.csv` for the current evaluation.

The legacy files are preserved for audit traceability, not as current reproduced results. CEDIA's 89.84% internal accuracy/90.30% weighted-F1 and the distributed checkpoint's 89.48% training-summary macro-F1 are separate window-level internal results, not substitutes for an independent 672-record test. See [metric reconciliation](../../reports/evidence/metric-reconciliation.md). Their hashes remain in the artifact inventory.
