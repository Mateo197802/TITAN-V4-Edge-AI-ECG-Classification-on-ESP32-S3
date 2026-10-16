# Balanced Rhythm and Pathology Continuation

## Fixed Scientific Contract

This continues the corrected phase6 checkpoint. No layer is reset, no new
network is initialized, and Gold is not replaced. The network keeps
1,143,751 parameters, four heads, six frontal leads and a `[6,1250]` input
at 125 Hz. Unknown-source pathology targets remain masked, not negative.
All comparison signals and references remain unchanged.

The training tensors and reference vectors are reused byte-for-byte from
the [corrected joint experiment](../joint_refinement_2026-10-12/README.md).
This includes its complete official rhythm-taxonomy check and exclusions
of validation, follow-up and historical-cohort patients. The initial
checkpoint's earlier training lineage is not reconstructed by this run.

## Development Diagnosis and Design

The 5,512 observed pathology training records include 3,539 with at least
one Primary-5 statement and 1,973 without one. In validation, 93/386 have
at least one such statement. This prevalence difference is recorded in
`data/development_diagnosis.json`; it is an observed motivation for a
controlled sampling comparison, not proof of a unique error mechanism.

The [predeclared protocol](training/protocol.json) defines seven 40-epoch
trials with checkpoints at epochs 4, 8, 16, 24 and 40:

1. Continue the previous optimizer/loss as a control.
2. Condition starting-model rhythm KL on agreement with training references,
   remove manual class-focus multipliers, and use cosine learning rates.
3. Train the existing pathology head with balanced sampling and BCE.
4. Match that head-only control with asymmetric focal loss.
5. Train the existing last two blocks and both heads with balanced BCE.
6. Match the joint last-two-block trial with asymmetric focal loss.
7. Train all existing backbone blocks at a lower rate with asymmetric loss.

Balanced pathology sampling assigns half of its probability mass to
Primary-5-negative training records and half to positive records, with
positive weights derived from inverse per-label training support. It uses
no validation or follow-up prevalence to fit those probabilities.
Sampling indices use a dedicated seeded CPU generator inside the CEDIA
job, then transfer to CUDA; all neural-network fitting remains on the GPU.
This avoids PyTorch 2.1's nondeterministic CUDA replacement-sampling kernel
without disabling deterministic algorithms. Primary-5
output columns have loss weight two; the other existing columns have weight
one. Asymmetric loss uses positive exponent zero, negative exponent four,
negative probability clipping 0.05 and differentiable focusing weights.
Stable log-space arithmetic, masks and weighted-mean reduction are explicit project adaptations of
[Ridnik et al., ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/papers/Ridnik_Asymmetric_Loss_for_Multi-Label_Classification_ICCV_2021_paper.pdf),
DOI [10.1109/ICCV48922.2021.00015](https://doi.org/10.1109/ICCV48922.2021.00015).
The paper does not establish that this method improves ECG classification.

Batch-normalization buffers and quality/biometric head parameters are
frozen. Auxiliary output consistency is measured separately; it is not
clinical quality, age, sex or BMI performance. First-batch shared-gradient
cosines are recorded on training batches to investigate task interference.

Selection uses validation only. Supported-class recall and auxiliary
preservation are checked against the warm start. A rhythm-best candidate
must not trade away more than 0.005 pathology macro-F1; a joint-best
candidate must not trade away more than 0.002 source-mean rhythm macro-F1.
Fallback is the unchanged starting checkpoint. Both heads of each selected
checkpoint are evaluated; scores from different PTHs are never combined.

## Completed Comparison

CEDIA job **29368** completed on 2026-10-13, exit `0:0`, Slurm elapsed
`00:16:18`, with Python 3.10.14, PyTorch 2.1.2+cu121, NumPy 1.26.4,
CUDA 12.1 and an A100-SXM4-40GB. Earlier job 29364 stopped when CUDA
replacement sampling rejected strict determinism; its submitted code and
24 partial output/log files are preserved separately. The full comparison
was rerun with a dedicated CPU sampling generator, without changing data,
reference vectors or numerical hyperparameters. The correction receipt
is `data/deterministic_sampler_correction.json`.

None of the 35 candidates satisfied both the predeclared selection and
preservation conditions. Both published selected-role PTHs are the unchanged
phase6 fallback, not newly promoted candidates. The highest global rhythm
macro-F1 candidate was `full_balanced_asl_e40`: 1,257/1,476, 85.16% accuracy
and 84.67% macro-F1. Its pathology macro-F1 was 47.42%, versus reference
50.71%, and its PTB-XL validation SB recall was 27/30 versus 30/30.
Consequently that intermediate was not presented as an improved combined
model. It may be studied in a separately declared follow-up experiment.

| Endpoint of the retained fallback | Records | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Rhythm validation | 1,476 | 84.28% | 83.37% |
| Rhythm follow-up | 2,171 | 82.17% | 81.53% |
| Primary-5 follow-up | 1,022 | 94.44% mean per-label | 48.51% |
| Historical final-reference rhythm | 672 | 83.93% | 82.03% |

Shared first-batch gradient cosines were negative in 17/40 epochs of
balanced BCE, 19/40 of last-two-block asymmetric loss and 24/40 of
full-backbone asymmetric loss. These limited training-batch observations
motivate further investigation; they do not uniquely establish a cause.
All 35 candidate validation outputs were independently replayed on CPU
from the original 1,476 signals. Both fallback roles were also replayed
on all 2,171 signals, matching every rhythm and pathology decision.

## Replay Commands

After installing the repository dependencies, from the repository root:

```bash
python scripts/verify_joint_evidence.py --bundle outputs/reviewer_verification/balanced_joint_2026-10-13
python scripts/verify_joint_sources.py --bundle outputs/reviewer_verification/balanced_joint_2026-10-13
python scripts/run_joint_inference.py --bundle outputs/reviewer_verification/balanced_joint_2026-10-13 --model joint_selected --output data/cache/balanced_joint_reproduced
python scripts/run_joint_inference.py --bundle outputs/reviewer_verification/balanced_joint_2026-10-13 --model rhythm_selected --output data/cache/balanced_rhythm_reproduced
python scripts/run_joint_historical.py --bundle outputs/reviewer_verification/balanced_joint_2026-10-13 --model rhythm_selected --output data/cache/balanced_history_reproduced
```

Training input reconstruction is identical to the prior joint experiment.
Copy the three reconstructed raw NPZ files and the two pathology NPZ files
into a fresh directory with `training/`, verify `INPUT_SHA256.json`, and
submit `sbatch job.sbatch` on CEDIA with the recorded Python/CUDA stack.
Local or login-node gradient fitting is prohibited. The selected refit
uses the full 40-epoch cosine schedule even when stopped at an earlier
selected epoch. Every state tensor must match its original trial exactly.

To replay all validation checkpoint probabilities and auxiliary outputs:

```bash
python scripts/replay_joint_validation_auxiliary.py --bundle outputs/reviewer_verification/balanced_joint_2026-10-13 --checkpoints-dir YOUR_REPLAY/results --output data/cache/balanced_validation_replayed
```

Primary-9 rhythm accuracy and macro-F1 are single-label endpoints.
Primary-5 mean binary accuracy and macro-F1 are separate multi-label
endpoints on observed native PTB-XL records only. The fixed comparison
cohorts are development follow-up, not a new untouched clinical test.
The experiment does not reconstruct the original historical Gold run.

## Attribution

Additional-source targets and descriptions retain [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/)
attribution, DOI [10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45),
under CC BY 4.0. Source waveforms are obtained separately and never
included in Git. See [REFERENCES.md](../../../REFERENCES.md) and
[LICENSE-ARTIFACTS.md](../../../LICENSE-ARTIFACTS.md) for source releases
and the research-artifact/source-code license distinction.
