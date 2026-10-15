# Pretrained Transfer Execution

This study adapts separate ECGFounder rhythm/pathology teachers before any
transfer into the existing TITAN student. The student remains six frontal
leads at 125 Hz, four heads and 1,143,751 parameters. Research targets are not
measured outcomes.

## Completed Checks

- `inputs/input_receipt.json`: 28,913 source/student pairs verified against
  original bytes, including all evaluation records. The new fitting allowlist
  contains 22,540 records; 27 source/tensor-identical historical aliases remain
  preserved with role `train_excluded` and do not enter the new fits.
- `model_inspection.json`: pinned 150-output teacher architecture and safe
  serialization checks. External weights stay outside Git.
- `pilot/status.json`: Slurm job 29619 completed with exit code 0:0 on an
  A100-SXM4-40GB. Sixty-four training pairs were loaded; a batch of eight
  produced finite teacher logits/features and adapted-head gradients.
  Peak allocated memory was 204,987,392 bytes. The student repeated identical
  four-head outputs; no optimizer step or input mutation occurred.
- The pilot seal binds its job, attempt, staged code inventory and frozen
  paired-input manifest. Runtime installation used an isolated Python 3.10.14,
  Torch 2.6.0+cu118 environment, leaving the previous audit runtime untouched.
- `snapshots/pilot-a2-code.tar.gz` preserves the exact code/input inventory
  used by that allocation, before subsequent execution-receipt hardening.
- `features/status.json`: job 29625 completed 0:0 in 2,306 GPU-seconds.
  Four preprocessing variants cover 25,657 unique development records.
  The retained task controls and joint warm-start were recomputed on the same
  rhythm/pathology validation cohorts; their complete predictions are included.

The failed first bootstrap consumed 58 GPU-seconds; successful installation
consumed 268, and the pilot consumed 39. All count toward the approved budget.
The installation receipt predates the stricter submission seal; it is retained
as an installation record, not a model-evaluation result.

## Execution Status

Teacher job 29627 completed 0:0 in 4,376 GPU-seconds. Its 125-file inventory
and validation references passed verification; all 30 trial decisions were
independently recalculated from the published predictions. One additional
head fit repeated with identical parameters and logits.

| Twelve-lead teacher | Three-seed mean accuracy | Three-seed mean macro-F1 | Transfer decision |
| --- | --- | --- | --- |
| Rhythm, canonical filtered, final stage | 88.7986% | 87.1488% | Eligible |
| Pathology, canonical unfiltered, final stage | 95.3626% Primary-5 per label | 47.8978% Primary-5 cross-fit | Not selected; class recall regressed |

The rhythm mean-logit ensemble scores 88.7534% accuracy and 87.0605% macro-F1
on 1,476 validation ECGs. Pathology uses 2,027 native validation ECGs; its
warm-start Primary-5 cross-fit macro-F1 is 43.2334%. The teacher comparisons
and every class/source decision are in `teachers/status.json` and
`teachers/teacher-fits/*/result.json`.

These are teacher results, not student deployment results. Job 29644 completed
0:0 in 520 GPU-seconds and exported rhythm ensemble targets for exactly 22,540
training keys. The target cache joins against the original references and
source/tensor hashes; no pathology teacher is included. Student comparisons
(job 29650) and source replay must also complete before the
whole-study verifier can pass. Input, feature and zero-step pilot checks do
not establish student improvement.

The fixed protocol is in
`docs/superpowers/plans/2026-10-14-founder-transfer.md`. Fitting and threshold
selection use development data only; previously examined tracking and
historical cohorts cannot select a teacher or student.

## Local Verification

```sh
python -m pytest tests/test_founder_transfer_inputs.py tests/test_founder_transfer_model.py tests/test_founder_transfer_staging.py -q
python scripts/verify_artifact_hashes.py
```

`scripts/verify_founder_transfer.py` is the strict whole-study verifier. It
requires the completed five-stage packet, allocation accounting and exact
stage-code snapshots. Its default mode reconstructs signals and performs CPU
student inference; `--metrics-only` explicitly does not certify source inference.
An incomplete study fails rather than returning a review-ready status.

Unit tests use synthetic model fixtures and the hash-pinned reference manifest;
they do not require the external pretrained weight or raw ECG cache. Full
source reconstruction is a separate, mandatory integration check: use
`scripts/prepare_founder_transfer.py --output <fresh-input-directory> --offline`
with the pinned source cache present, followed by the source-inference verifier.
Synthetic fixture success is not a claim about pretrained-model performance.

Manager commands require an explicit state directory and this bundle path.
The CEDIA study registry, not the local directory, accounts for submissions,
including failed and connection-interrupted attempts. Unresolved or nonterminal
allocations block another submission. Remote fitting additionally requires a
successful sealed pilot and a tested implementation of its stage.
