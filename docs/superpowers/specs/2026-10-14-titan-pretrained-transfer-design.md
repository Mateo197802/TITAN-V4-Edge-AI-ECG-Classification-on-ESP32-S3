# TITAN pretrained transfer design

Status: written design for review, 2026-10-14. No new neural training,
teacher inference, or student improvement is claimed by this document.

## Objective and fixed deployment contract

Improve rhythm discrimination and native pathology recognition using stronger
training supervision, while continuing existing TITAN weights. The user's
research targets are 90% rhythm accuracy and approximately 87% rhythm macro-F1,
and higher pathology macro-F1. These are targets, not acceptance evidence.

The deployed student retains 1,143,751 parameters, the four existing heads,
six leads in the order I, II, III, aVR, aVL, aVF, 125 Hz, and float32 input
`[B,6,1250]`. Rhythm has nine outputs; pathology has ten. Quality and the legacy
three-output biometrics head remain present. Their heuristic or legacy targets
are not clinical event supervision.

The primary warm-start is
`outputs/reviewer_verification/supervision_control_2026-10-14/model/joint_selected.pth`,
SHA-256 `72d3ece9b00e844d6a94f207f4ced71af0af69c59cef66b625031ff92b2f965b`.
This is weight initialization, not optimizer/scheduler resume: those states
are not available in this student checkpoint. Gold and previous candidates
remain immutable comparison controls. No student is initialized from scratch.

## Decision and alternatives

Start with ECGFounder as the first eligible teacher candidate, not as an
already proven winner. Its official 12-lead checkpoint is publicly downloadable,
MIT-declared, and has a supervised 150-output head with useful rhythm and
conduction labels. Task-specific adaptation must be evaluated before transfer.
The evidence receipt is
`reports/evidence/pretrained-teacher-inspection-2026-10-14.json`.

| Candidate | Evidence inspected | Decision for first pilot |
| --- | --- | --- |
| ECGFounder | Official code, model card, labels, license, downloaded checkpoint hash and safely loaded state shapes | First candidate; resolve preprocessing and task interfaces before GPU execution |
| ECG-FM, wanglab | Official README declares MIMIC-IV and Challenge 2021 pretraining | Not a clean external comparison on the current Challenge-derived populations |
| ECG-JEPA, kweimann | Official README distinguishes MIMIC-only and all-data variants; GitHub release list was empty | MIMIC-only remains an alternative if an official matching weight file and rights can be established; do not substitute all-data weights |
| ECG-LFM, biomed-AI | Official repository and linked Zenodo record; open 7,522,279,891-byte pretrain.pt, CC BY 4.0 weight metadata | Secondary candidate, not downloaded or executed; larger runtime integration and safe-load inspection required |

ECGFounder pretraining is declared to use HEEDB; the paper treats PTB-XL as
external evaluation. This does not establish record-level disjointness for
every TITAN source, especially Georgia. ECG-LFM also uses HEEDB and MIMIC-IV.
Do not infer independence from a dataset name or model card alone. Existing
Gold pretraining membership remains a separate provenance question.

The choice is between pretrained supervised transfer first, own training-only
SSL first, or another broad supervised sweep. Pretrained transfer is preferred
because the previous teachers used six-lead features and the recent sweeps
share many errors. SSL is a separate comparison, not silently combined with
transfer before its individual effect is measured.

## Program decomposition

| Subproject | Purpose | Dependency |
| --- | --- | --- |
| A: paired inputs and teacher qualification | Establish 12-lead teacher contracts and task-specific validation results | This written spec, then reviewed implementation plan |
| B: controlled distillation | Transfer useful outputs into the unchanged student | A demonstrates endpoint-specific benefit |
| C: continued student SSL | Improve the existing extractor on training-only ECGs | Frozen input allowlist; separate design and execution plan |
| D: event evidence | Build development-only event review material and test confirmed supervision | Recorded event decisions; separate design and execution plan |
| E: optional temporal-stride variant | Test information retention versus deployed cost | Explicit variant protocol and hardware profiling; separate design and execution plan |

A and B define the first implementable subsystem in this spec. C, D and E are
approved research directions, not implemented subsystems. Each needs its own
reviewed spec and plan before code or training starts. Their individual
effects must be known before a combined final candidate is tested.

## A: paired input boundary

Use the manifest-directed training union of 22,567 records: 8,393 original
development records, 13,984 native pathology additions and 190 directed rhythm
additions. There are 16,413 observed native pathology training records. Missing
rhythm or pathology labels stay masked, not converted to negative findings.

The existing cache has the referenced 12-lead, 500-Hz WFDB records. Header
contracts and file existence were checked for the union; raw byte hashes were
sampled by source in the local inspection, not exhaustively recomputed. The
implementation must verify every source header/signal hash before staging.

Build a separate teacher input contract from physical-unit WFDB signals;
pair the same record and ten-second interval with the original student input.
Teacher inputs are `[B,12,5000]`; student inputs remain `[B,6,1250]`. Reorder
leads by explicit header names. Do not invent precordial leads by interpolating
the six frontal leads. Preserve the exact existing student preprocessing as
the control: causal third-order 0.5-45 Hz filtering with leading zero context,
polyphase conversion to 125 Hz, first ten seconds and per-lead z-score.

The pinned ECGFounder code has material inconsistencies: HEED and MIMIC lead
orders differ, the PTB evaluator uses global z-score without the README's
explicit filtering, and an unrelated LVEF loader overwrites a resampled
signal. Therefore no upstream loader is adopted wholesale. Record the
candidate preprocessing contracts, reproduce their tensor transformations on
fixtures and resolve the checkpoint's expected lead order before interpreting
teacher performance. If the available evidence cannot settle the alternatives,
register them as validation comparisons before any tracking-cohort inference.
Never choose a preprocessing variant from tracking errors.

Split at the available patient/record boundary before creating windows.
Exclude rhythm validation (1,476), pathology validation (2,027), follow-up
(2,171) and the historical 672 from every teacher fit, SSL fit and supervised
student fit. The two validation sets share 386 PTB records: they are not two
independent samples for uncertainty estimates. Check available patient IDs,
source signal hashes and tensor hashes. Identical-waveform exclusion cannot
prove cross-source patient identity where metadata do not provide it.

## A: teacher task interfaces

The downloaded ECGFounder state has 509 keys, first convolution weight
`[64,12,16]` and dense head `[150,1024]`. The 30,885,443 state tensor elements
include buffers and are not a measured teacher parameter count.

Keep its original 150-output head only for a semantically explicit diagnostic
probe. Incomplete RBBB is represented upstream, but exact ILMI and ISC_ outputs
are not available under those task names. No probability union, approximate
label synonym, or 150-label argmax is a substitute for TITAN's task definition.

Adapt two separate teacher copies from the same pinned pretrained body:

- Rhythm: nine-class head, using only observed original rhythm references.
  Class order: AFIB, SB, STACH, NSR, PVC, RBBB, LBBB, PAC, 1AVB. Retain IRBBB
  versus complete RBBB metadata for diagnostic reporting without altering the
  deployed nine-class meaning.
- Pathology: ten-label sigmoid head with the existing observed-target mask.
  Class order: IMI, ASMI, LVH, ISC_, ISCAL, NST_, ILMI, AMI, ALMI, LAE.
  Report all ten labels and Primary-5 indices `[0,8,6,9,3]`. Zero diagnostic
  likelihood or unavailable labels do not become known negatives.

Start with a frozen-body head fit, followed by a controlled final-stage teacher
fit only if the head probe supports it. A teacher must outperform its relevant
student control on that endpoint before its predictions supervise transfer.
Use rhythm macro-F1/accuracy and pathology macro-F1/average precision, source
strata and per-class support. Pathology per-label accuracy is descriptive,
not the selection objective; the all-negative baseline already scores highly.

## B: distillation boundary

Export teacher outputs only for training allowlisted record/windows, with
source hashes, teacher checkpoint SHA, class order, preprocessing version,
window bounds and task masks. Validation predictions are selection evidence,
not targets for fitting the student. Assert stable row joins by source and
record key, never by CSV row position alone.

Fit from the pinned student warm-start. Compare supervised-only, rhythm-only
transfer, pathology-only transfer and joint transfer, with fixed initialization
and data. Rhythm uses softmax temperature KL; pathology uses independent
Bernoulli targets and masks. Label references remain the primary supervision.
An incorrect teacher must not outweigh a known reference automatically; the
transfer weight and agreement-conditioned rule are selected only in validation.
Any old-Gold preservation control must address the documented disagreement
problem rather than replaying an unconditional high-weight constraint.

No training-only projection module is saved into the deployed student. Output
transfer is the initial method. A later feature-alignment adapter may exist
only in the training system, must handle 1024-to-192 dimensions explicitly and
must not masquerade as an unchanged deployment graph.

Freeze quality and biometrics output heads. Their outputs can still change
when the shared extractor changes; compare their behavior and document it.
Check existing BatchNorm state explicitly and retain train-only balanced
recalibration as its own control, not an unlogged inference mutation.

## C: continued SSL direction

Start with the current training allowlist, including signals with incomplete
task labels. Additional unlabeled data require the same patient/source audit
and a new allowlist revision before use. Continue the student extractor,
with temporary training-only projection/prediction layers removed on export.

Use conservative same-record aligned views or masked-context objectives.
Do not equate arbitrary neighboring fragments with the same ectopic event,
assume all leads contain the same pathology, or enforce equal representations
for different dates merely because a patient ID matches. Avoid time warps,
beat removal and voltage scaling that destroy the desired targets. CLOCS is
a research precedent, not proof that its invariance assumptions suit TITAN.
Its upstream code has an NC/SA notice; do not copy it into this repository
under a different license. An independently implemented, cited objective is
distinct from adopting that code.

Compare the same supervised fit with and without SSL continuation. The
voltage-preserving input variant stays separate from the original z-score
control and requires unit checks and its own preprocessing manifest.

## D: event evidence direction

Prepare development-only PAC/PVC event positions and stratified conduction
review views. R-peak/delineation algorithms may propose times and uncertainty;
they cannot provide clinical beat types by themselves. Record-level diagnoses,
RR-selected windows and existing AI proposals are not event ground truth.

Preserve the user's communication that earlier low/moderate-confidence record
proposals were accepted, with its exact scope and source status. Do not create
new event timestamps, reviewer signatures or independent confirmations from
that communication. Existing forms inspected here have no completed event
decisions. Collect case-specific human decisions before supervised event loss.

Reapply existing rhythm head weights to temporal features for an auxiliary
training loss where confirmed event intervals support it. This reuses deployed
weights; export returns the unchanged original record-level computation.
AI-only weak supervision, if explored, is a separate named comparison.

## E: optional temporal-stride direction

Changing the last two existing stride-2 blocks to stride 1 keeps tensor shapes
and the parameter count but changes operations: total stride 32 becomes 8;
approximately 40 temporal positions become 157. This is a distinct architecture
variant, not the unchanged original TITAN. Profile complete-model operations,
activation memory, TFLite/int8 behavior and actual ESP32-S3 latency before
claiming deployability. Lack of a board measurement must not become a claimed
hardware result. Default A/B student trials keep the original strides.

## Execution, selection and evidence requirements

All neural fits and GPU probes run in a CEDIA Slurm allocation. The login node
is for access and file management. Use an isolated teacher environment rather
than mutating the archived audit environment. First measure safe-load behavior,
GPU memory and a small forward/backward pilot; then establish the resource
budget and exact three-seed comparison in the reviewed implementation plan.
Do not schedule an unlimited search until the desired metric appears.

Pin teacher code and weight revisions, license notices, runtime, student
warm-start, random seeds, fit schedule and selection criteria before fitting.
Restricted checkpoint loading is required; do not execute remote model code
or fall back to unrestricted pickle loading. The inspected Founder file loads
with `weights_only=True` plus a narrow NumPy serialization allowlist. Its
optimizer, scheduler and stored AUROC are not TITAN continuation states or
our validation results.

Choose task thresholds exclusively from validation. Use the existing
threshold cross-fit diagnostic and report full-calibration results separately;
this is not model-training cross-validation. Candidate selection considers
both tasks, minority classes and source regressions. Declare the exact
promotion margins before fitting rather than inventing them after results.
Protect the archived controls even when a candidate fails to improve.

Current validation and tracking cohorts support development comparisons.
An untouched final evaluation needs a separately frozen, previously unused
patient/record selection and a checkpoint-lineage audit before opening it.
Do not replace ambiguous or misclassified test records to improve a score.

Each completed trial must deliver raw predictions/probabilities, observed
reference masks, per-class/source tables, checkpoint SHA, paired manifests,
input/output hashes, exact commands, Slurm logs and environment information.
Require student state-shape/count assertions, split/pairing/mask tests,
preprocessing fixtures, repeated seed execution checks and inference replay
from original signals. Publish only completed, measured results in the PR;
this design does not replace the historical Gold evidence or the current PTH.

## Review state and primary references

This specification is ready for user review. The next stage is the written
implementation plan for A/B; no training entrypoint or new run exists yet.

- [ECGFounder official repository](https://github.com/PKUDigitalHealth/ECGFounder), code commit `04edac702b61c91face519774ddcc0cd712fef23`.
- [ECGFounder public model](https://huggingface.co/PKUDigitalHealth/ECGFounder), revision `d9b1793951b2342f5f7e84f1ac03cd37f8a08724`.
- [ECGFounder primary study](https://arxiv.org/html/2410.04133v4).
- [ECG-FM official repository](https://github.com/bowang-lab/ecg-fm).
- [ECG-JEPA official repository](https://github.com/kweimann/ECG-JEPA).
- [ECG-LFM official repository](https://github.com/biomed-AI/ECG-LFM), inspected commit `647369a012dd8d5d362e99e2bf238c91f3e1d03b`.
- [ECG-LFM weight record](https://zenodo.org/records/20388950), DOI `10.5281/zenodo.20388950`; this is the upstream model DOI, not a TITAN DOI.
- [CLOCS primary study](https://proceedings.mlr.press/v139/kiyasseh21a.html).
