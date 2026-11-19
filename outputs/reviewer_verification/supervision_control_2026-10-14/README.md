# Independent Heads and Supervision Controls

## Contract

CEDIA Slurm job **29488** completed with exit code `0:0` in **4m15s**.
All six comparisons continue the existing phase10 rhythm-selected checkpoint,
SHA-256 `0941b585d62416ab176942a2b0a65c2c8824c2cffb87609a515e6f36de126d8c`.
No student is initialized from scratch. The network retains **1,143,751
parameters**, four heads, six frontal leads I/II/III/aVR/aVL/aVF, **125 Hz**
and `[6,1250]` input. The shared extractor, batch-normalization buffers,
quality head and biometric head are frozen. Their output preservation is
not a clinical quality/biometric-performance claim.

The 22,567-record training stream, 1,476 rhythm-validation records,
2,027 native pathology-validation records, 2,171 followup records and
672 historical signal identities/reference columns are inherited unchanged
from the preceding directed-source study. Non-native unknown pathology
targets are masked, never converted into negative labels. No prediction
or followup error determines record inclusion or supervision weights.

GPU execution uses Python 3.10.14, PyTorch 2.1.2+cu121, CUDA 12.1 and
NumPy 1.26.4 on an A100-SXM4-40GB. Read-only distribution metadata inspected
after the run lists scikit-learn 1.3.2, SciPy 1.13.1 and WFDB 4.1.2 in that
same CEDIA virtual environment. Portable inference uses the repository's
pinned Python 3.12 environment; this is a cross-runtime output replay,
not a claim of byte-identical CUDA training across different Torch versions.

## Sequential Audit

1. Adding five ECG-arrhythmia classes changed its share of the earlier
   equal-source/class cross-entropy coefficient mass from **13.79% to
   26.47%**, even though only 190 records were added. Two rhythm-head trials
   freeze the prior source budget while balancing classes within each source.
   This controls CE coefficients, not the total gradient contribution:
   the agreement-only KL anchors remain unweighted and separately normalized.
2. Of 16,413 observed native pathology-training ECGs, **5,698** have
   `validated_by_human=False`. That flag is not proof of an incorrect label.
   Four pathology-head trials compare natural BCE, human-flag weights,
   diagnostic-likelihood weights and human-flag/positive-class weights.
3. Thresholds are cross-fitted across five fixed patient groups in validation.
   Every held group uses thresholds fitted only on the other four groups;
   the verifier binds each fold to its canonical ECG ID and patient token.
4. Rhythm and pathology components are selected separately using only the
   recorded validation criteria. A real single checkpoint is composed only
   if every shared and auxiliary tensor matches exactly. The selected fit
   is repeated on CEDIA before followup outputs are generated.

The official [PTB-XL v1.0.3 documentation](https://physionet.org/content/ptb-xl/1.0.3/)
defines SCP likelihood zero as unknown. The 131 positive training `NST_`
statements with zero likelihood therefore keep likelihood factor 1,
not zero. Human-validation weighting is a separate factor. Source targets
are not changed. Cite Wagner et al.,
[Scientific Data 2020](https://doi.org/10.1038/s41597-020-0495-6), and the
[versioned release DOI](https://doi.org/10.13026/kfzx-aw45).
Additional-source provenance and citations remain in the
[directed-source runbook](../directed_pcgrad_2026-10-13/README.md).

## Fitting and Selection

All fits use 20 epochs, seed 20261014, AdamW weight decay 0.01,
gradient clipping 1 and checkpoints at epochs 4/8/12/20. Features are
computed on the CEDIA GPU; cached-feature inference is checked bitwise
against raw-input GPU inference. Existing head counts are 19,401 rhythm
and 19,498 pathology parameters; the full architecture does not change.

| Trial | Existing head | Learning rate | Weighting |
|---|---|---:|---|
| rhythm_fixed_030 | Rhythm | 0.0003 | Fixed train-only CE source budget |
| rhythm_fixed_010 | Rhythm | 0.0001 | Fixed train-only CE source budget |
| path_natural | Pathology | 0.001 | Natural masked BCE |
| path_human | Pathology | 0.001 | Human flag: True 1, False 0.5 |
| path_certainty | Pathology | 0.001 | Human factor times known positive likelihood factor |
| path_human_positive | Pathology | 0.001 | Human factor plus capped positive-class weight |

Known positive likelihood factor is `0.5 + 0.5 * likelihood/100`.
Unknown zero likelihood keeps factor 1. Positive-class weights are
`sqrt(weighted negatives / weighted positives)`, clipped to `[1,4]`,
from training references only. Primary-5 label coefficients are 2;
the other five pathology label coefficients are 1.

Rhythm selection requires at least 0.002 gain in both global macro-F1 and
mean-source macro-F1, with protected-class preservation. Pathology selection
requires at least 0.005 gain in patient-group threshold-cross-fit macro-F1,
no more than 0.005 drop in mean Primary-5 average precision, and no drop in
the full-calibration macro-F1. Otherwise the starting head is retained.

**Threshold-order clarification:** the archived source computes both the
cross-fit and full-validation thresholds for every candidate before model
selection. The full-calibration result participates in the non-degradation
screen. Thresholds are recalculated for the final composed model after
component selection. The shorthand in `protocol.json` about fitting full
thresholds after selection refers to this final composition, not all
candidate thresholds; the executed source bytes are preserved unchanged.
Only threshold fitting is cross-fitted, not neural training or the prior
development selections. This remains development validation, not nested
model cross-validation or an independent final test.

## Results

The selector retains the starting rhythm head and chooses
`path_human_positive_e12` for pathology. That selected fit reproduces every
state tensor on a second CEDIA fit. The composed checkpoint SHA-256 is
`72d3ece9b00e844d6a94f207f4ced71af0af69c59cef66b625031ff92b2f965b`.

| Same validation population | Starting control | Composed checkpoint |
|---|---:|---:|
| Rhythm accuracy / macro-F1, 1,476 | 85.43% / 84.82% | 85.43% / 84.82% |
| Pathology full-calibration macro-F1, 2,027 | 42.92% | 45.59% |
| Pathology patient-group threshold-cross-fit macro-F1 | 39.74% | 43.23% |
| Mean Primary-5 average precision | 38.34% | 41.58% |

| Fixed followup | Starting control | Composed checkpoint |
|---|---:|---:|
| Rhythm, 2,171, accuracy / macro-F1 | 82.96% / 82.41% | 82.96% / 82.41% |
| Primary-5, 1,022, mean label accuracy / macro-F1 | 94.58% / 42.64% | 94.58% / 46.16% |

Pathology gains **3.52 percentage points** macro-F1 over its equally
calibrated starting control. Rhythm predictions remain unchanged for all
2,171 records. This is not a best-overall pathology result: the earlier
phase6 checkpoint remains at 48.51% macro-F1, and phase10 joint at 48.40%.
Both stay available. No 90% / 87% rhythm or historical pathology target
achievement is claimed. Mean binary accuracy is not pathology macro-F1.

Both model roles were also inferred on all 672 historical public source
signals. Original references give **494/672, 73.51% accuracy / 73.83%
macro-F1**; final references give **567/672, 84.38% / 82.80%**. The two
models have the same rhythm state. These are new, complete recorded runs,
not reconstruction of the historical 605/672 predictions.

The two CE-budget rhythm-head trials fail validation preservation; all
eight rejected states are retained. The strongest late rhythm candidate
has 84.76% validation accuracy / 84.36% macro-F1, below the starting control.
Source-budget change is thus a measured objective confound, not a proven
sole explanation or a demonstrated remedy for the current rhythm errors.

## Next Sequential Decision

The recorded evidence supports moving beyond head-only fitting, not raising
probabilities or altering comparison references. Rhythm followup retains
147/243 RBBB, 100/143 PAC and 317/415 NSR correct. These are diagnostic
followup observations, not candidate-selection criteria. Primary-5 gains
are uneven: ALMI F1 rises from 34.78% to 62.07%, but LAE falls from 28.57%
to 16.67% and ILMI from 47.06% to 44.00%. The macro gain is not a claim that
every label improved.

The next experiment should compare continued fitting of the last existing
blocks with alternating task batches, using these audited training weights
and a preservation anchor from the relevant retained task model. It should
include an equally calibrated phase6 pathology control, so a head-only
gain over the rhythm-oriented initialization cannot be mistaken for a
best-overall pathology gain. No new layers/parameters are needed. Selection
must remain patient/source validation-only, with protected-label checks and
a subsequent new, unexamined test. **This next block-fitting comparison
has not run in job 29488.** All conclusions above describe the completed
six-arm experiment, not that proposed continuation.

## Evidence and Replay

`training/` preserves executed code, start checkpoint, protocol, job script,
small supervision-control arrays and the submitted input hash manifest.
`data/` contains official source metadata/indexes, inherited signal manifests,
training control rows and their audit. `results/` contains all 24 learned
states and validation arrays, the actual composed state, selected models,
training histories, selection/refit receipts and complete followup CSVs.
`logs/` preserves Slurm outputs with private home paths removed; a receipt
records both original and published log hashes. `historical672/` contains
both final models' complete predictions against the two unchanged references.
`replay/` records independent CPU signal replays.

Run from the repository root, using the pinned inference environment and
fresh output directories. Omit `--offline` on first use to download the
version-pinned public WFDB source files; source/header/tensor hashes are
checked. No waveform download is required to recompute the stored metrics.

```bash
python scripts/verify_supervision_evidence.py --bundle outputs/reviewer_verification/supervision_control_2026-10-14
python scripts/prepare_supervision_controls.py --bundle outputs/reviewer_verification/supervision_control_2026-10-14 --output data/cache/supervision_rebuilt
python scripts/replay_expanded_validation.py --bundle outputs/reviewer_verification/supervision_control_2026-10-14 --output data/cache/supervision_validation_replay
python scripts/run_joint_inference.py --bundle outputs/reviewer_verification/supervision_control_2026-10-14 --output data/cache/supervision_joint_replay
python scripts/run_joint_inference.py --bundle outputs/reviewer_verification/supervision_control_2026-10-14 --model rhythm_selected --output data/cache/supervision_rhythm_replay
python scripts/run_joint_historical.py --bundle outputs/reviewer_verification/supervision_control_2026-10-14 --model joint_selected --output data/cache/supervision_history_replay
```

For GPU refitting, first rebuild original, native-expanded and directed
inputs using the preceding bundles' runbooks. Place the seven rebuilt input
NPZs in an isolated copy of `training/`, not the immutable evidence tree;
compare every file with `INPUT_SHA256.json`. `job.sbatch` runs the archived
trainer in a Slurm GPU allocation. No local CPU/login-node fitting fallback
is permitted. Source signals and large tensor caches are not redistributed
in Git; the checked-in manifests and indexes identify them exactly.

These repeatedly examined followup cohorts are tracking comparisons, not
an untouched independent test. The historical aggregates remain separate
from these new fits. Gold and all earlier checkpoints stay available.
TITAN source code is MIT; project model/evidence artifacts are CC BY 4.0
under the repository license scope. Third-party PTB-XL and ECG-arrhythmia
files retain their own CC BY 4.0 attribution requirements.
