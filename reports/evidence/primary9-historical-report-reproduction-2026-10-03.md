# Primary-9 Historical Report Reproduction

Audit date: 2026-10-03
Endpoint: 672-record Primary-9 external evaluation
Classes: AFIB, SB, STACH, NSR, PVC, RBBB, LBBB, PAC, 1AVB

## Conclusion

The archived aggregate can be reconstructed from its saved confusion matrix and row-level label-update table. The historical model inference itself is not reproducible from the artifacts currently linked to that report: its 672 predictions and the checkpoint identity used by the report are missing.

Keep these statements distinct:

| Reproduction target | Status | Evidence |
|---|---|---|
| Recalculate the archived 90.0298% accuracy and 87.8192% macro-F1 | Reproduced at aggregate level | Saved final confusion matrix and source metric summary |
| Reconcile why the correct count increases from 476 to 605 | Reproduced arithmetically | 129 update rows; every final label equals recorded CEDIA Top-1 and differs from its original label |
| Rerun the exact historical model and obtain the same 672 predictions | Not reproduced | Historical prediction table and report-to-checkpoint linkage are absent |
| Rerun the available Gold checkpoint | Reproduced, different result | 568/672 against the updated reference: 84.5238% accuracy, 82.3519% macro-F1 |

The score is against the updated reference labels. It must not be described as a change in model performance: the predictions were not shown to improve between the original and updated-reference calculations.

## Aggregate Reconstruction

Let `C_orig = 476` be the archived number correct against the original labels, and let `U = 129` be the audited label updates. For every update row, the archived table satisfies:

\[
y_{i,\mathrm{original}} \ne y_{i,\mathrm{final}}, \qquad
y_{i,\mathrm{final}} = \hat y_{i,\mathrm{CEDIA}}.
\]

Therefore, using the same recorded predictions and assuming the update table corresponds to that prediction run:

\[
C_{\mathrm{final}} = C_{\mathrm{orig}} + U = 476 + 129 = 605,
\qquad
\mathrm{Accuracy}_{\mathrm{final}} = \frac{605}{672}=0.900297619.
\]

This reconciles accuracy. Macro-F1 and weighted-F1 are not additive; they must be recomputed from the saved final confusion matrix `C`:

\[
F1_k=\frac{2TP_k}{2TP_k+FP_k+FN_k},\qquad
\mathrm{MacroF1}=\frac{1}{9}\sum_{k=1}^{9}F1_k,
\qquad
\mathrm{WeightedF1}=\sum_{k=1}^{9}\frac{n_k}{672}F1_k.
\]

The saved matrix yields macro-F1 `0.8781916793` and weighted-F1 `0.9008533778`. The aggregate reconstruction is only as traceable as the archived source report, its matrix, and the update table; it does not establish which executable, weights, or signal preprocessing generated the predictions.

## CEDIA Run-History Cross-Check

Read-only Slurm accounting for 2026-05-26 identified these completed evaluation jobs in the active `V4_CEDIA` project tree:

| Job | Slurm command | Result in job log | Why it is not the 605/672 source |
|---:|---|---|---|
| 12874 | `sbatch 00_LANZADORES_CEDIA/lanzar_eval_primary9.sh` | 672 records; accuracy `0.488095`; macro-F1 `0.408838` | Log says 50 labels were automatically injected; report and weights differ from the archived Gold report |
| 12887 | `sbatch 00_LANZADORES_CEDIA/lanzar_eval_fullunfreeze.sh` | 672 records; accuracy `0.505952`; macro-F1 `0.418456` | Separate full-unfreeze evaluation, not linked to the archived Gold report |

Job 12874 used `03_OUTPUTS/TEACHER_PRIMARY9/titan_v4_lite_weights_best.pth` (SHA-256 `a226acf2105789cb8d4c4d9419b33c71ff3472a92b3e7f83836b6b9a6704511d`) and the `RHYTHM_PRIMARY9_DISTILL/manifest_external_test.csv` input (SHA-256 `8e0855c7254bb024dd2d3eead1ad569ae33d7fb1588545fd22f3b1e463928e60`). Its launcher SHA-256 is `7387a0783a2120632179339d94a53957c99569878c729e4f4faaa6c5b36b075d`; its evaluator SHA-256 is `7d5e650d3e58414f0195402df19cc1a811a75823ca746b0bb1b285cf2d8657e5`.

These jobs explain earlier CEDIA evaluations but do not establish the provenance of 605/672. The bounded Slurm history did not identify a completed job whose command, checkpoint, manifest, and outputs jointly match the archived 605 report.

## Exact Rerun Requirements

To reproduce historical inference rather than only its aggregate, recover and freeze all of the following:

1. The exact 672-row prediction table from the run that generated the archived report, with `record_id`, top-1 class, and per-class scores or logits.
2. The exact checkpoint file used in that run, with a SHA-256 recorded in the report manifest.
3. The evaluator source revision and command, including signal resampling, lead handling, window selection, aggregation across windows, class order, and any calibration or post-processing.
4. The exact 672-record manifest and signal file identities/hashes, plus the software environment and relevant library versions.
5. The original and final reference labels, linked by immutable record ID, with the update decision log preserved separately from the prediction outputs.
6. A run manifest linking the Slurm job ID, checkpoint hash, code revision, input hashes, command/configuration, prediction-file hash, and output-report hash.

Then rerun inference without fitting or tuning on these 672 records, join predictions and both reference columns by `record_id`, assert exactly 672 unique matches, and recompute both confusion matrices and metric tables. A mismatch in any per-record prediction means the historical inference has not been reproduced, even if aggregate metrics happen to match.

If the historical prediction table or exact checkpoint cannot be recovered, retain the claim only as an aggregate reconstruction from the archived report and update audit. Do not call it an exact model rerun. The available Gold checkpoint rerun is a separate result and remains 568/672 against the updated labels.

## Local Verification Commands

From the audit repository root, verify the evidence bundle and regenerate the paired-reference calculations:

```powershell
python scripts/verify_primary9_gold_evidence.py
python scripts/recompute_primary9_paired_reference.py `
  --bundle outputs/reviewer_verification/arrhythmia_primary9/cedia_gold_2026-05-26 `
  --checkpoint models/gold_master/gold_master_primary9_model.pth `
  --out-dir outputs/reviewer_verification/arrhythmia_primary9/paired_reference_recomputation_2026-10-03
```

The generated JSON contains separate historical aggregate and fixed-checkpoint sections. The first validates saved CEDIA arithmetic; the second reports the available checkpoint's fresh inference. They are not interchangeable.

## Source Hashes

Archived CEDIA artifacts and the locally available checkpoint are identified in [the paired-reference report](primary9-paired-reference-recomputation-2026-10-03.md) and its JSON output. The two recovered CEDIA job logs have SHA-256 values:

| Job log | SHA-256 |
|---|---|
| `eval_primary9_log_12874.out` | `c91c4947b1b0bc5a347cafacd060edb6e1c3b58c38ccb039fd70ac599465ab97` |
| `eval_fullunfreeze_log_12887.out` | `e6e848576ea05259b37a1762c93c2037e588e95c75517ec16373d9f7cd3bf026` |
