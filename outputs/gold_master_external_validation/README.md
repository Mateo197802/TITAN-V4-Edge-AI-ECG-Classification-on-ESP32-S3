# Historical Aggregate Outputs

These files preserve the earlier aggregate snapshot. Use them with the source records and verification package linked below; do not confuse the aggregates with a fresh model inference.

- The CEDIA Primary-9 source report records 605/672 correct (accuracy 0.90030, macro-F1 0.87819) and includes a confusion matrix that recalculates to those values. Its 129 target updates all set the final label to the recorded model Top-1. A fresh repository inference on those final labels gives 568/672. The sanitized 672-row labels, 129-row update audit, fresh predictions, hashes, and verifier are in `outputs/reviewer_verification/arrhythmia_primary9/cedia_gold_2026-05-26/`.
- The CEDIA Gold Pathology Primary-5 summary reports mean per-label accuracy 0.90256 and macro-F1 0.66874. Its source summary and promotion-manifest hashes, and the declared checkpoint hash, are recorded in `reports/evidence/pathology-primary5-gold-source.json`. The checkpoint and row-level evaluation files were absent from their declared paths during verification.
- The old Cascade/OOD report is a 211-record safety annex and has not been recomputed in the current workflow.
- The old validation CSV includes metadata beyond the minimal current label table. Use `data/external_validation/final_external_validation_labels.csv` for the current evaluation.

The reports preserve the values and sources; the linked verification package distinguishes aggregate recalculation from fresh signal-to-prediction inference. CEDIA's 89.84% window accuracy/90.30% weighted-F1 and the distributed checkpoint's 89.48% training-summary macro-F1 are separate results, not substitutes for the 672-record endpoint. See [metric reconciliation](../../reports/evidence/metric-reconciliation.md). Their hashes remain in the artifact inventory.
