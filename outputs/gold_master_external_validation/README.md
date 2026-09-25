# Historical Aggregate Outputs (Legacy)

These files are retained to preserve the repository's earlier evidence snapshot. They are not the current reproduced evaluation and must not be used by the current reporting commands.

- The old Primary-9 report records 605/672 correct (accuracy 0.90030, macro-F1 0.87819). Its previous label table disagreed with the deterministic source-header/CEDIA label mapping on 129 of 672 records. The result is superseded by the recomputed 466/672 result in `outputs/reproduced/primary9/`.
- The old Pathology Primary-5 aggregate (accuracy 0.90256, macro-F1 0.66874) has no complete record-level predictions in the current workflow and has not been reproduced.
- The old Cascade/OOD report is a 211-record safety annex and has not been recomputed in the current workflow.
- The old validation CSV includes metadata beyond the minimal current label table. Use `data/external_validation/final_external_validation_labels.csv` for the current evaluation.

The legacy files are preserved for audit traceability, not as current results. Their hashes remain in the artifact inventory.
