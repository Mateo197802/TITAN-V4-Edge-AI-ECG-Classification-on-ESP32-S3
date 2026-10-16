# Balanced Joint Refinement Implementation Plan

## Scope

Continue the approved existing-network experiments for rhythm and Primary-5
pathology. Keep 1,143,751 parameters, four output heads, six frontal leads,
125 Hz, all source tensors and comparison references unchanged. All fitting
runs in a CEDIA Slurm GPU allocation. No model is selected using follow-up
or historical scores, and no target performance is guaranteed.

## Evidence Behind the Comparison

The corrected phase6 checkpoint is the warm start. Its pathology training
has 5,512 observed records, including 3,083 additional Primary-5-positive
records but only 1,973 Primary-5-negative records. Primary-5-positive
prevalence is 64.2% in training versus 24.1% in validation. Its validation
rhythm errors concentrate in NSR, RBBB and PAC; unconditional starting-model
KL still opposes some hard-reference corrections. These observations support
controlled experiments, not a uniquely established explanation of errors.

## Fixed Design

- Reuse the byte-pinned phase6 code, data, metadata and patient exclusions.
- Add training-only balanced pathology sampling: half Primary-5-negative
  mass, half positive mass with inverse label-support weighting.
- Compare the legacy loss, agreement-conditioned rhythm anchoring,
  balanced BCE and masked asymmetric focal loss. Include pathology-head-only
  controls and low-rate full-backbone fine-tuning; reset no layers.
- Keep batch-normalization buffers and quality/biometric parameters frozen;
  measure output preservation separately from clinical performance.
- Run seven predefined trials for 40 epochs, retaining epochs 4, 8, 16,
  24 and 40. Refit selected trials deterministically.
- Freeze the same validation-only threshold rule and preservation policy.
  Publish separate rhythm-best and joint-best checkpoints and score both
  heads of each; do not combine metrics from different checkpoints.
- Reproduce selected predictions from original signals and independently
  replay all candidate validation outputs before publication.

## Tasks

- [x] Add failing tests for masks, balanced sampling, agreement anchoring,
  immutable architecture, fallback and validation-only selection.
- [x] Implement the isolated trainer and generic bundle/run-directory flags.
- [x] Review the trainer before CEDIA execution; correct findings first.
- [x] Execute, refit, seal and retrieve the CEDIA run (29368, fallback).
- [x] Replay original-signal predictions and validation selection.
- [ ] Publish measured results, source/protocol hashes and evidence in PR #1
  only after checkout and clean Git export tests pass.

## Method Reference

Asymmetric loss is an experimental adaptation of Ridnik et al., ICCV 2021,
DOI 10.1109/ICCV48922.2021.00015. The repository's mask, per-label weighting
and mean reduction are explicit project choices. Improvement on ECG is to
be measured, not inferred from the original image-classification study.
