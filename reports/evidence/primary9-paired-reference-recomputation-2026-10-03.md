# Primary-9 Paired-Reference Recalculation

Audit date: 2026-10-03
Evaluation unit: 672 record-level, single-label predictions
Classes: AFIB, SB, STACH, NSR, PVC, RBBB, LBBB, PAC, 1AVB

## Purpose

This audit scores one fixed set of 672 predictions against both label columns available in the CEDIA evidence bundle: the original labels and the updated labels. No model weights, predictions, thresholds, class definitions, or cohort membership were changed for this comparison.

The recalculation files, both 9-by-9 confusion matrices, input hashes, per-class scores, and run metadata are in [the paired-reference output bundle](../../outputs/reviewer_verification/arrhythmia_primary9/paired_reference_recomputation_2026-10-03/).

## Results

### Archived CEDIA report

| Reference used in archived report | Correct / total | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| Original labels | 476 / 672 | 70.8333% | 70.6844% | 70.6975% |
| Updated labels | 605 / 672 | 90.0298% | 87.8192% | 90.0853% |

The updated-label metrics recalculate from the archived report's confusion matrix. The archived report gives aggregate values for the original reference but no original confusion matrix or historical row-level predictions. The original and updated aggregates are separate reference results; the label changes are not a change in model performance.

The accuracy can be reconciled from the reported original correct count and the row-level update table if the table's recorded CEDIA Top-1 corresponds to the report's prediction run: all 129 changed labels differ from the original label and equal the recorded CEDIA Top-1. Under that linkage, `476 + 129 = 605` correct, or `605/672 = 90.0298%`. The report does not provide row-level historical predictions to independently verify that linkage. Macro-F1 is not additive; it recalculates from the saved final-reference confusion matrix. This is aggregate reconstruction, not a rerun of the historical model inference. See the [historical reproduction runbook](primary9-historical-report-reproduction-2026-10-03.md).

The archived report does not contain its 672 row-level predictions or record the checkpoint SHA-256. The CEDIA Gold checkpoint is available and hashed, but the accessible artifacts do not establish that this exact checkpoint generated the archived confusion matrix.

### Fixed local checkpoint, same predictions against both references

| Reference used for scoring | Correct / total | Accuracy | Macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| Original labels | 466 / 672 | 69.3452% | 68.8841% | 68.9590% |
| Updated labels | 568 / 672 | 84.5238% | 82.3519% | 84.6412% |

This paired comparison uses the record-level predictions in `fresh_inference/primary9_record_predictions.csv` and the same record IDs in `primary9_label_history_672.csv`. The prediction file's `true_label` matches the updated-label column. Rejoining on `record_id` and switching only the reference column yields 466/672 on original labels and 568/672 on updated labels.

The original reference is specifically `rhythm_label_name` in CEDIA's pre-update `DATASETS_CURADOS/RHYTHM_PRIMARY_V2/manifest_external_test.csv` (SHA-256 `8190ec19f4887d4618213cbf95b9da3b68948dddba4f0233d975eba5938e5b8c`); all 672 IDs and labels were cross-checked against `original_rhythm_label` in the sanitized history. The paired final reference is `final_rhythm_label`. A separate 129-row source table records each row's original label, CEDIA model Top-1, final label, change decision, confidence, timestamp, and notes. Top-1 is not the original reference.

| Class | Original support | Original F1 | Updated support | Updated F1 |
|---|---:|---:|---:|---:|
| AFIB | 46 | 0.6154 | 76 | 0.8844 |
| SB | 38 | 0.8312 | 38 | 0.8312 |
| STACH | 147 | 0.8013 | 150 | 0.9133 |
| NSR | 38 | 0.4884 | 62 | 0.7273 |
| PVC | 134 | 0.6723 | 112 | 0.8889 |
| RBBB | 47 | 0.6916 | 54 | 0.7368 |
| LBBB | 30 | 0.7342 | 30 | 0.7342 |
| PAC | 146 | 0.5974 | 86 | 0.8187 |
| 1AVB | 46 | 0.7679 | 64 | 0.8769 |

The full precision, recall, F1, support, and confusion matrices are in the JSON/CSV bundle. The updated reference improves this checkpoint's score, but the fixed inference remains below 90% accuracy: 568 correct versus 605 needed, a difference of 37 records.

## Metric Definitions

For the 9-class confusion matrix (C), rows are reference labels and columns are predicted labels:

\[
\mathrm{Accuracy}=\frac{\sum_{i=1}^{9} C_{ii}}{N},\qquad
P_i=\frac{C_{ii}}{\sum_j C_{ji}},\qquad
R_i=\frac{C_{ii}}{\sum_j C_{ij}},
\]

\[
F1_i=\frac{2P_iR_i}{P_i+R_i},\qquad
\mathrm{MacroF1}=\frac{1}{9}\sum_{i=1}^{9}F1_i,\qquad
\mathrm{WeightedF1}=\sum_{i=1}^{9}\frac{n_i}{N}F1_i.
\]

Here (N=672), and (n_i) is the reference support for class (i). Undefined precision or recall is treated as zero, matching the packaged inference report.

## Reference-Label History

The source table contains 129 changed rows with a recorded CEDIA Top-1, a final reference value, a decision field, and a timestamp. The sanitized local copy preserves record IDs, original and final labels, and the recorded Top-1, while omitting timestamps and free-text notes. This package validates the row counts and value relationships; it does not reconstruct the workflow that produced each reference value.

All 129 final labels in the source table equal the recorded historical CEDIA Top-1. The fixed local checkpoint's predictions equal that historical Top-1 on 110 of the 129 changed records. This confirms the archived report and local inference are distinct prediction outputs; it does not make the local run a reproduction of the archived 605/672 result.

## Source Artifacts and Checksums

The following source paths and SHA-256 values were checked read-only in CEDIA. Paths are relative to the CEDIA project root.

| Artifact | Role | SHA-256 |
|---|---|---|
| `DATASETS_CURADOS/RHYTHM_PRIMARY_V2/manifest_external_test.csv` | Pre-update original-label manifest; 672/672 IDs and `rhythm_label_name` values matched to the sanitized history | `8190ec19f4887d4618213cbf95b9da3b68948dddba4f0233d975eba5938e5b8c` |
| `03_OUTPUTS/2026-05-26_GOLD_MASTER_EXTERNAL_GATE/04_EXTERNAL_VALIDATION_SET/primary9_final_external_validation_manifest.csv` | Final post-update 672-record reference manifest | `b302d33f35bb056757e76bf89cc4b9fe7a8811074d2d1a7bcf3e94df451ba556` |
| CEDIA 129-row source change table | Original/final values, decision fields, timestamps, and notes field | `f66ce6d3122dfe92d7ef237b345da48ca79ac67156768ff4f7322a417ee68661` |
| `03_OUTPUTS/01_ARRHYTHMIA_RHYTHM_OUTPUTS/2026-05-26_FINAL_EXTERNAL_GATE/primary9_final_external_report.json` | Archived aggregate report and final confusion matrix | `582590fa1809b089ad6123f21e18a98cfe3352d00aeb8b3a2d9a015eead3692c` |
| `03_OUTPUTS/2026-05-26_GOLD_MASTER_EXTERNAL_GATE/01_GOLD_MASTER_PTH/gold_master_primary9_model.pth` | Available Gold checkpoint; source report does not record this hash | `bc7ba03d0d6d40e823fdb9bb261ebe29ae8b7a74c97d1c98e7140bca4d2d305b` |
| `00_LANZADORES_CEDIA/02_AUDITORIAS_EVALUACION/lanzar_primary9_student_external_eval.sh` | Candidate evaluator launcher; not linked by the archived report | `ed7de5858a21e84b8ca79a8d16f5b233582e585e05396637d5ebd283ddc1750e` |
| `00_LANZADORES_CEDIA/02_AUDITORIAS_EVALUACION/lanzar_primary9_student_v2_external_eval_full_support.sh` | V2 wrapper around the candidate launcher; not linked by the archived report | `eb1ecdf116b448abb9da3138d43bcaf8c01f754431a14e960c8f3c3de0255bdb` |
| `01_CODIGO_FUENTE/primary9_clean_protocol.py` | Available reportable-subset protocol; not the model inference runner | `d21adda9d0ff2f8edb4239b2ad6ca002d47f8c0e7122d62cc4c8aec6393faa9c` |

The candidate launcher calls `01_CODIGO_FUENTE/AUDITORIA/evaluate_primary9_student.py`, but that exact file is absent at the referenced path. Read-only Slurm accounting confirms jobs 12874 and 12887 completed, but does not link either job to the archived 605/672 report. Their available log summaries report different metrics. See the [historical reproduction runbook](primary9-historical-report-reproduction-2026-10-03.md).

## Scope and Limitations

- The 672-row endpoint is a project-specific single-label diagnostic, not the official PhysioNet Challenge metric, which permits multi-label diagnoses and uses its own scoring rules.
- The source manifest resolves these records to the Challenge 2021 `training` partition. The available checkpoint training manifest does not establish whether any records overlap model training.
- The archived 605/672 result is arithmetically verified from its saved confusion matrix for final updated labels. The report gives 476/672 on original labels but has no original-reference confusion matrix or historical row-level predictions. The 90.03% / 87.82% result is not the original-reference aggregate.
- The paired local recalculation is reproducible from the included predictions, label history, checkpoint, code/runtime manifest, and hashes; it does not reproduce the archived model output.
- The update CSV provides row-level decision fields and timestamps. The sanitized evidence allows the metric and label-value checks to be repeated, but does not include the full source workflow.

To regenerate the paired local-checkpoint metrics and hashes, run `python scripts/recompute_primary9_paired_reference.py`. To verify the CEDIA Gold bundle and saved final-reference matrix, run `python scripts/verify_primary9_gold_evidence.py` from the repository root. The paired metrics artifact records the exact inputs and both reference columns.
