# One Checkpoint From Compatible Trained Heads

## Method and Chronology

This experiment constructs and evaluates one actual PTH, not a table made
by taking scores from two different models. The existing architecture stays
at 1,143,751 parameters, four heads, six frontal leads, 125 Hz and `[6,1250]`.
No layers are reset and no additional parameters or signal features are added.

Both heads were fitted in CEDIA job 29384 on exactly the same frozen
backbone. The rhythm source, `heads_sb3_e16`, has the highest source-mean
fixed-nine macro-F1 among eligible phase8 validation candidates. The
pathology source, `heads_sb3_low_lr_e08`, has the highest Primary-5 macro-F1
among that run's validation candidates. The [protocol](training/protocol.json)
pins their original SHA-256 values and this selection rule. All shared body,
batch-normalization buffer, quality and biometric tensors must match byte
for byte before recomposition is permitted.

The method was specified after phase8 validation analysis and before this
new checkpoint's evaluation. It is not retrospectively presented as a
predeclared phase8 trial or an untouched-test discovery. This step does not
fit gradients: it reuses learned tensors, verifies reconstruction, then
runs full-network inference. The actual composed network must reproduce
the selected rhythm head's and pathology head's validation outputs.
Thresholds and preservation checks are recalculated on those actual outputs.

The reference and fallback remain the preceding phase6 model. Both tasks
of each final PTH are scored. The PTH, thresholds and validation selection
are saved before follow-up references are opened. Every source tensor,
reference vector and evaluation record is unchanged.

## Measured Results

CEDIA job **29407** completed on 2026-10-13, exit `0:0`, elapsed
`00:01:27`. It used Python 3.10.14, PyTorch 2.1.2+cu121, NumPy 1.26.4,
CUDA 12.1 and an A100-SXM4-40GB. No gradient fitting occurred in this
composition run. Both selection objectives returned the same reconstructed
state; distinct serialized role filenames have separate byte hashes.
Actual full-network validation inference matched both source heads, and
the deterministic state reconstruction matched every tensor.

| Endpoint | Phase6 reference | Actual composed PTH |
|---|---|---|
| Rhythm validation, 1,476 | 84.28% accuracy / 83.37% macro-F1 | 84.62% / 84.08% |
| Primary-5 validation, 386 | 50.71% macro-F1 | 51.93% |
| Rhythm follow-up, 2,171 | 1,784/2,171: 82.17% / 81.53% | 1,800/2,171: 82.91% / 82.10% |
| Primary-5 follow-up, 1,022 | 94.44% mean label accuracy / 48.51% macro-F1 | 93.13% / 45.39% |
| Historical final-reference rhythm, 672 | 564/672: 83.93% / 82.03% | 564/672: 83.93% / 81.81% |

Rhythm gained 16 net correct follow-up predictions, but pathology declined
despite its validation gain. This is not an improvement of both tasks in
follow-up and does not replace the reference or Gold as a combined model.
The original phase6 model and all measured alternatives remain available.

The Primary-5 false-positive count increased from 145 to 230, while true
positives increased from 136 to 154. IMI accounts for the largest change:
72 to 97 true positives, but 82 to 172 false positives. Its frozen threshold
was chosen at 0.45 on validation, versus reference 0.70. These counts locate
the measured failure; they do not prove a unique cause. Additional pathology
calibration coverage should be investigated before another joint promotion.

## Reproduce

After installing the repository's pinned dependencies, from its root:

```bash
python scripts/verify_joint_evidence.py --bundle outputs/reviewer_verification/compatible_heads_2026-10-13
python scripts/verify_joint_sources.py --bundle outputs/reviewer_verification/compatible_heads_2026-10-13
python scripts/run_joint_inference.py --bundle outputs/reviewer_verification/compatible_heads_2026-10-13 --model joint_selected --output data/cache/compatible_joint_reproduced
python scripts/run_joint_inference.py --bundle outputs/reviewer_verification/compatible_heads_2026-10-13 --model rhythm_selected --output data/cache/compatible_rhythm_reproduced
python scripts/run_joint_historical.py --bundle outputs/reviewer_verification/compatible_heads_2026-10-13 --model rhythm_selected --output data/cache/compatible_history_reproduced
```

`training/start.pth` contains the rhythm source and `path_source.pth` the
pathology source. `reference.pth` is the phase6 reference. The state-tensor
operation is provided in `composition_metrics.py`. To repeat the recorded
CEDIA inference, rebuild the same raw/pathology NPZs as the
[joint experiment](../joint_refinement_2026-10-12/README.md), put them with
`training/` in a fresh directory, verify `INPUT_SHA256.json`, then submit
`sbatch job.sbatch` with the recorded runtime.

To independently replay candidate validation outputs and auxiliary consistency:

```bash
python scripts/replay_joint_validation_auxiliary.py --bundle outputs/reviewer_verification/compatible_heads_2026-10-13 --checkpoints-dir YOUR_REPLAY/results --output data/cache/compatible_validation_replayed
```

## Evidence Scope

Rhythm has nine fixed classes; pathology reports Primary-5 mean binary
accuracy and macro-F1 on observed PTB-XL targets, not nine-class accuracy.
Both endpoints are predictions from the same real checkpoint. Quality,
age, sex and BMI output consistency is not clinical performance.
The 2,171 follow-up records and 672 historical records have been examined
in development and are not claimed as new untouched clinical tests.
This is not a recreation of the historical Gold inference or a guarantee
of the desired 90% accuracy / approximately 87% macro-F1.

Upstream attribution and CC BY 4.0 data-derived artifact scope remain in
[REFERENCES.md](../../../REFERENCES.md) and
[LICENSE-ARTIFACTS.md](../../../LICENSE-ARTIFACTS.md). Project source code is
MIT-licensed; no source ECG waveform is included in Git.
