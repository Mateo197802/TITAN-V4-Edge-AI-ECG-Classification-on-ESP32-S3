# Sequential Rhythm and Pathology Head Repair

## Scope and Initialization

This is a continued fit of the existing TITAN model, not a new architecture
or a reset of its layers. It keeps 1,143,751 parameters, four heads, six
frontal leads, 125 Hz and the `[6,1250]` signal input. The previous phase6
checkpoint is the reference and fallback.

The intermediate initialization is the highest global rhythm macro-F1
checkpoint on phase7 validation, `full_balanced_asl_e40`. Its source SHA-256
is `5c3c4e8334a02907371eb074de8359d30c58c89aedea8b2ebeaf05a1668ec43a`.
It was not promoted by phase7: native PTB-XL SB validation recall declined
from 30/30 to 27/30 and pathology macro-F1 declined. Its exact candidate
report and validation-only initialization rule are in `data/initialization.json`.
No follow-up or historical scores were used to choose this initialization.

## Fixed Method

Every backbone parameter and buffer is frozen. Only the original rhythm
and pathology heads may fit, 38,899 parameters in total; the pathology-only
control fits 19,498. Quality and biometrics parameters remain frozen.
The network's existing `forward_features` caches its 192-dimensional
features on the CEDIA GPU. Cached-head outputs are compared against the
full network before fitting; this does not add an input feature or change
the deployed signal-to-output architecture.

The [protocol](training/protocol.json) predeclares five 32-epoch controls:
pathology-only natural masked BCE; joint heads with training SB focus
weights one, two and three; and a lower-rate SB3 control. Other focus
weights match the preceding phase6 method. Masks, original teacher
probabilities, source/class weights and positive weights use training
data only. No reference labels or evaluation records change. Native
PTB-XL pathology targets have the same ten output columns and Primary-5
subset as the previous experiment. Unknown-source pathology is masked,
not treated as negative.

Validation is measured against phase6, not the ineligible intermediate.
The same supported-class preservation checks and validation-only threshold
rule apply. Both heads of each selected PTH are evaluated. A selected
trial is refitted from its exact initialization and must match every
original state tensor. Only then are the follow-up targets opened.
Fallback means `reference.pth`, the eligible phase6 model, not `start.pth`.

## Completed Results

CEDIA job **29384** completed on 2026-10-13, exit `0:0`, elapsed
`00:03:18`, with the same Python 3.10.14 / PyTorch 2.1.2+cu121 /
NumPy 1.26.4 / CUDA 12.1 / A100-SXM4-40GB runtime as phase7.
The full-validation cached-head outputs matched the full network bitwise.
Twenty-five checkpoints were compared. No single candidate satisfied both
task selection conditions; both selected-role PTHs are unchanged phase6
fallbacks, scoring 82.17% accuracy / 81.53% macro-F1 in rhythm follow-up
and 94.44% mean binary accuracy / 48.51% Primary-5 macro-F1 in pathology.

`heads_sb3_e16` preserved supported-class recall and had the highest
eligible source-mean rhythm macro-F1. The best pathology validation head
was instead `heads_sb3_low_lr_e08`. Their metrics are not presented as
one phase8 checkpoint. Their compatible-state reconstruction is a separate,
actually executed experiment documented in
[compatible heads](../compatible_heads_2026-10-13/README.md).

All 25 validation candidates were independently replayed on CPU from the
original signals, as were both final-role outputs on 2,171 signals and
both reference versions of the 672 historical cases.

## Reproduce

From the repository root, after installing its pinned dependencies:

```bash
python scripts/verify_joint_evidence.py --bundle outputs/reviewer_verification/sequential_heads_2026-10-13
python scripts/verify_joint_sources.py --bundle outputs/reviewer_verification/sequential_heads_2026-10-13
python scripts/run_joint_inference.py --bundle outputs/reviewer_verification/sequential_heads_2026-10-13 --model joint_selected --output data/cache/sequential_joint_reproduced
python scripts/run_joint_inference.py --bundle outputs/reviewer_verification/sequential_heads_2026-10-13 --model rhythm_selected --output data/cache/sequential_rhythm_reproduced
python scripts/run_joint_historical.py --bundle outputs/reviewer_verification/sequential_heads_2026-10-13 --model rhythm_selected --output data/cache/sequential_history_reproduced
```

The raw and pathology NPZs rebuild with the same commands as the
[preceding joint experiment](../joint_refinement_2026-10-12/README.md).
Place them in a fresh CEDIA directory with `training/`, verify all files
against `INPUT_SHA256.json` and submit `sbatch job.sbatch`. The recorded
Python/CUDA runtime is required for a bitwise selected-trial refit.
The script refuses fitting outside a CEDIA Slurm GPU allocation.

Intermediate PTHs remain in the sealed run, with their original byte hashes
in `results/SHA256SUMS.json`. To replay candidate validation outputs:

```bash
python scripts/replay_joint_validation_auxiliary.py --bundle outputs/reviewer_verification/sequential_heads_2026-10-13 --checkpoints-dir YOUR_REPLAY/results --output data/cache/sequential_validation_replayed
```

## Interpretation and Rights

The 2,171 fixed comparison records and the 672 historical records remain
follow-up measurements, not a new untouched clinical test. Rhythm accuracy
and fixed-nine macro-F1 are separate from pathology mean binary accuracy
and Primary-5 macro-F1 on the 1,022 observed native PTB-XL records. Auxiliary
output consistency is not measured clinical quality, age, sex or BMI accuracy.
Neither a desired 90%/87% score nor the historical Gold inference is implied
by running this new experiment.

Data provenance, upstream CC BY 4.0 releases and file hashes are preserved
from [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) and the
source manifests. No source waveform is included in Git. Project source
code remains MIT; research artifacts retain the scope and attribution in
[LICENSE-ARTIFACTS.md](../../../LICENSE-ARTIFACTS.md) and
[REFERENCES.md](../../../REFERENCES.md).
