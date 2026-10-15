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

The failed first bootstrap consumed 58 GPU-seconds; successful installation
consumed 268, and the pilot consumed 39. All count toward the approved budget.
The installation receipt predates the stricter submission seal; it is retained
as an installation record, not a model-evaluation result.

## Execution Status

Teacher qualification, soft-target export, student comparisons and their
source replay are subsequent stages. No improvement in rhythm or pathology
metrics is established by input checks or this zero-step pilot.

The fixed protocol is in
`docs/superpowers/plans/2026-10-14-founder-transfer.md`. Fitting and threshold
selection use development data only; previously examined tracking and
historical cohorts cannot select a teacher or student.

## Local Verification

```sh
python -m pytest tests/test_founder_transfer_inputs.py tests/test_founder_transfer_model.py tests/test_founder_transfer_staging.py -q
python scripts/verify_artifact_hashes.py
```

Manager commands require an explicit state directory and this bundle path.
The CEDIA study registry, not the local directory, accounts for submissions,
including failed and connection-interrupted attempts. Unresolved or nonterminal
allocations block another submission. Remote fitting additionally requires a
successful sealed pilot and a tested implementation of its stage.
