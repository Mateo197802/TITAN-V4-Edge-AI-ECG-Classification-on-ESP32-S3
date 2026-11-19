# Primary-9 Paired-Reference Output

Generated 2026-10-03 from the packaged 672-record prediction file and label-history table. The model output is held fixed; only the reference-label column changes. The 90.03% accuracy / 87.82% macro-F1 result is for the final updated reference, not the original labels.

| File | Contents |
|---|---|
| `primary9_paired_reference_metrics.json` | Input hashes, checkpoint identity, historical CEDIA aggregates, paired metrics, per-class results, both local-run confusion matrices, and interpretation. |
| `primary9_per_class_paired_metrics.csv` | Support, precision, recall, and F1 by class for original and updated references. |
| `SHA256SUMS.csv` | SHA-256 and byte count for every file in this bundle except the checksum file itself. |

The original labels are `rhythm_label_name` from CEDIA's pre-update `DATASETS_CURADOS/RHYTHM_PRIMARY_V2/manifest_external_test.csv` (SHA-256 `8190EC19F4887D4618213CBF95B9DA3B68948DDDAB4F0233D975EBA5938E5B8C`). Its 672 IDs and labels were cross-checked against `original_rhythm_label` in the sanitized history. The final labels are the separate `final_rhythm_label` values. The raw original manifest is not redistributed because it includes demographics and absolute paths; the sanitized row-level history is included in the CEDIA Gold bundle. The historical CEDIA Top-1 field is a third, separate column and is not substituted for the original reference. Records are joined by `record_id`; duplicates or incomplete coverage fail generation.

The CEDIA report gives 476/672 for the original reference and 605/672 for the final updated reference. Its saved confusion matrix allows recalculation of the latter. For the original-reference result, it stores only aggregate values, with no matrix or historical row-level predictions. The source 129-row update table records a decision and timestamp for each changed row; free-text notes are not copied into the sanitized bundle. Matching updated labels to recorded Top-1 values does not establish how changes were decided.

The JSON preserves the archived CEDIA source aggregates separately from the fixed local checkpoint recalculation. The same checked-in predictions score 466/672 against original labels and 568/672 against final labels; this is not the archived CEDIA prediction run. See [the evidence interpretation](../../../../reports/evidence/primary9-paired-reference-recomputation-2026-10-03.md) and [historical-run reconciliation](../../../../reports/evidence/primary9-historical-report-reproduction-2026-10-03.md).

From the repository root, regenerate the paired metrics and hashes with `python scripts/recompute_primary9_paired_reference.py`; verify the historical final-reference matrix and CEDIA bundle with `python scripts/verify_primary9_gold_evidence.py`.
