# Directed Source Expansion and Shared PCGrad

## Contract

The four approved arms compare a frozen control with directed data, progressive
existing-block fitting, and shared-gradient PCGrad. Every arm starts from the
same actual phase10 joint checkpoint, SHA-256
`77a8c3749ad863256fe03e038c5c686bb62f6bb4cc9db7ad1f926801845d4112`.
No layer is reset. The student retains 1,143,751 parameters, four heads, six
frontal leads I, II, III, aVR, aVL and aVF, 125 Hz and `[6,1250]` input.
Batch-normalization statistics and quality/biometrics parameters stay frozen.
Output consistency of those two heads is not clinical age/sex performance.

The frozen control is not trained. The other three arms use identical data,
source/class rhythm weighting, masked pathology BCE, agreement-only anchors,
auxiliary preservation, seed, 20 epochs and checkpoints at epochs 4/8/12/20.
The first fits the last two existing blocks and both task heads. Progressive
arms fit those blocks through epoch 4, then allow all existing blocks with
a tenfold lower learning rate in the stem and first three blocks.

PCGrad is an explicit two-task adaptation of
[Yu et al., Gradient Surgery for Multi-Task Learning, 2020](https://arxiv.org/abs/2001.06782).
Each task's conflicting shared-backbone gradient is projected against the
other task's original gradient. The resulting gradients are summed, keeping
the unprojected loss scale. Head-own gradients and auxiliary preservation
gradients are not projected. Conflicts and cosine similarities are recorded;
the paper's non-ECG results are not evidence of TITAN target achievement.

## Source Selection

The complete **45,152-record** header census uses ECG-arrhythmia **v1.0.0**,
DOI [10.13026/wgex-er52](https://doi.org/10.13026/wgex-er52),
[official release](https://physionet.org/content/ecg-arrhythmia/1.0.0/).
Also cite Zheng et al.,
[Scientific Reports 2020](https://doi.org/10.1038/s41598-020-59821-7).
The upstream taxonomy and official file index are preserved under CC BY 4.0.
Source signals are not included in Git; the source index pins every file.

Selection uses only official codes, known record identities/waveform hashes,
and a fixed ID hash order with a predeclared cap of 500 per difficult class.
Unsupported rhythms, unrecognized codes and multiple non-sinus primary statements are excluded.
Sinus background plus exactly one in-scope statement follows the existing
singleton rule. Other morphology diagnoses do not automatically invalidate
an otherwise singleton rhythm reference. No prediction, confidence or error
status enters selection. All added signals are native 500 Hz/10 s and use
the unchanged preprocessing pipeline.

| Added class | Records |
|---|---:|
| RBBB | 74 |
| PAC | 48 |
| 1AVB | 35 |
| PVC | 27 |
| LBBB | 6 |
| Total | 190 |

The training stream is 8,393 original ECGs, 13,984 native additions and these
190 records: **22,567 entries**. All ten pathology targets of new-source
records are masked. Generic MI codes lose wall localization in this release
and are not converted into IMI/ALMI/ILMI references. The prior native pathology
stream and prevalence remain unchanged.

Original rhythm validation (1,476), pathology calibration (2,027), followup
(2,171, including 1,022 native pathology references), and both archived
historical-reference columns (672) remain unchanged. Selected fits are
repeated exactly before followup labels are opened. These repeatedly examined
comparisons are followup, not a new untouched test or source-independence claim.

## Results

CEDIA job **29484** completed with exit code `0:0` in **10m55s** on an A100.
This Slurm wall time includes startup and input validation. All twelve
candidate checkpoints and validation probability arrays are preserved.
Both validation selectors retain the starting checkpoint: no candidate
improves the required joint objectives. Consequently no selected fit needs
retraining, and no rejected candidate is chosen using followup results.

| Actual checkpoint | Rhythm validation accuracy / macro-F1, 1,476 | Primary-5 validation macro-F1, 2,027 | Mean source rhythm macro-F1 |
|---|---:|---:|---:|
| Frozen control | 85.16% / 84.60% | 43.82% | 66.18% |
| Directed last two blocks, epoch 20 | 84.42% / 83.84% | 44.82% | 65.45% |
| Directed progressive, epoch 20 | 85.09% / 84.57% | 44.55% | 65.90% |
| Directed PCGrad, epoch 20 | 85.09% / 84.57% | 45.00% | 65.90% |

PCGrad's validation pathology gain is **1.18 percentage points**, accompanied
by a rhythm decline. Global and mean-source macro-F1 are different metrics;
each source score uses the fixed nine-class order, including absent classes.
The gain is not presented as a demonstrated followup or clinical improvement.

| Retained starting model, reproduced followup | Accuracy | Macro-F1 |
|---|---:|---:|
| Rhythm, 2,171 ECGs | 82.77%, 1,797 correct | 82.23% |
| Primary-5, 1,022 native ECGs | 94.56% mean binary label accuracy | 48.40% |

Both final-role PTHs retain every initial parameter and buffer. CPU signal
replay reproduces all 2,171 rhythm decisions and pathology decision vectors
for both roles; maximum probability discrepancy is `2.45e-6`, tolerance `2e-5`.
The preceding rhythm-specific checkpoint (82.96% / 82.41%) and older phase6
pathology checkpoint (48.51% macro-F1) remain available; no best endpoints
from different models are combined or silently replaced.

The 672 historical source signals were also rerun for both retained roles:
original references give **498/672, 74.11% / 73.63%**; final references give
**564/672, 83.93% / 81.91%**. These unchanged-reference checks do not reproduce
the historical 605/672 run. No 90% / 87% target attainment is claimed.

PCGrad measures **1,711 conflicting batches out of 3,540**, mean epoch cosine
`0.002824`. The ordinary/progressive arms sample the first batch per epoch,
finding 7/20 and 6/20 conflicts respectively; these are different sampling
denominators. Gradient conflict is measurable, but its removal does not
by itself recover the historical metrics.

Preliminary job 29480 was cancelled before model selection when a source-code
coverage review identified unsupported statements absent from the published
taxonomy. The stricter parser rejects all unrecognized codes. The first
inputs, code, cancellation receipt and sanitized logs are preserved under
`attempts/29480/`; no checkpoint from that attempt enters reported selection.
The corrected job 29482 was subsequently cancelled during startup on
`compute-0-1` after process inspection confirmed an NFS wait. Its unchanged
inputs, source and startup evidence are under `attempts/29482/`. The numerical
protocol is retained for the retry, which excludes that node.
The input manifests of jobs 29482 and 29484 differ only in `job.sbatch`;
every data, checkpoint and numerical fitting-source hash is identical.

All 45,152 original header byte hashes and complete diagnosis sets are
authenticated by the verifier. One malformed upstream header identity is
excluded and never enters a train/validation/followup tensor. The 190-record
new signal NPZ, row manifest and complete census reproduce byte-identically.
All comparison labels, pathology masks and archived reference columns remain
unchanged. Upstream source files remain outside Git.

## Reproduction

Run from the repository root with the pinned environment and fresh output
directories. The source preparer reconstructs the complete census and the
selected signal tensors with official SHA-256 checks. A second offline
preparation reproduced the census, row manifest and tensor NPZ byte for byte.

```bash
python scripts/prepare_directed_rhythm.py --output data/cache/directed_rebuilt --workers 24 --cap 500
python scripts/verify_directed_evidence.py --bundle outputs/reviewer_verification/directed_pcgrad_2026-10-13
python scripts/replay_expanded_validation.py --bundle outputs/reviewer_verification/directed_pcgrad_2026-10-13 --output data/cache/directed_validation_replay
python scripts/run_joint_inference.py --bundle outputs/reviewer_verification/directed_pcgrad_2026-10-13 --output data/cache/directed_joint_replay
python scripts/run_joint_inference.py --bundle outputs/reviewer_verification/directed_pcgrad_2026-10-13 --model rhythm_selected --output data/cache/directed_rhythm_replay
python scripts/run_joint_historical.py --bundle outputs/reviewer_verification/directed_pcgrad_2026-10-13 --output data/cache/directed_history_replay
```

Use `--offline` after the official source files are cached. Rebuild native
expanded inputs with the phase10 runbook and original raw inputs with the
partial-backbone runbook. In an isolated copy of `training/`, place rebuilt
`path_validation.npz` and `rhythm_extra.npz`; the manager supplies and checks
the remaining raw/native NPZs against `INPUT_SHA256.json` before submission.

```bash
python scripts/manage_joint_cedia_run.py submit --training-dir YOUR_ISOLATED_TRAINING_COPY --state-dir YOUR_FRESH_STATE_DIR --path-inputs YOUR_REBUILT_NATIVE_INPUTS --raw-inputs YOUR_REBUILT_ORIGINAL_INPUTS --source-run YOUR_VERIFIED_PRIOR_CEDIA_RUN --run-name YOUR_FRESH_CEDIA_RUN
```

`training/job.sbatch` invokes archived code on a CEDIA Slurm GPU only.
No login-node or local CPU training fallback is permitted. Gold and all
earlier candidates remain preserved. Mean binary pathology accuracy and
macro-F1 are separate endpoints, not interchangeable measures.
