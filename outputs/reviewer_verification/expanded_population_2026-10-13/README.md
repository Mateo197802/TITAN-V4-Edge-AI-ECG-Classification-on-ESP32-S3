# Expanded Native Population: Same-Architecture Continuation

## Contract

CEDIA Slurm job 29418 continues the actual compatible-head checkpoint,
SHA-256 `4b20177029ee5388b76c336bda09b10bb962750b94feacb0b32b47d7e1fc6637`.
No layer is reset. The network retains 1,143,751 parameters, four heads,
six frontal leads (I, II, III, aVR, aVL, aVF), 125 Hz and `[6,1250]` input.
Batch-normalization buffers and quality/biometric parameters remain frozen.
Auxiliary consistency is not an evaluation of clinical age, sex or quality.

The protocol and submitted inputs were frozen before fitting. A frozen
control is calibrated on the same expanded validation population as the
candidates, so calibration effects and parameter-update effects are separate.
The three fits are pathology-head-only and two final-block updates with
pathology loss weights 0.25 and 1.0. Each runs 20 epochs, with validation
checkpoints at 4, 8, 12 and 20. Strict deterministic selected-trial refits
precede opening follow-up labels. All gradient fitting occurs on CEDIA GPU.

## Population and References

PTB-XL **v1.0.3**, DOI [10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45),
provides native 500 Hz recordings and official patient-level folds.
The [official release](https://physionet.org/content/ptb-xl/1.0.3/) recommends
folds 1-8 for training, fold 9 for validation and fold 10 for testing.
Upstream metadata, statements and file index are included unchanged with
their pinned hashes; upstream material remains under its CC BY 4.0 license.
Also cite Wagner et al., [Scientific Data 2020](https://doi.org/10.1038/s41597-020-0495-6)
and Strodthoff et al., [PTB-XL benchmark preprint, 2020](https://arxiv.org/abs/2004.13701v1).
General dataset citations and project license scope are in the root documents.

All eligible native fold 1-8 records not already in the original training
manifest are included, excluding known follow-up/historical native patients.
This gives **13,984 additional records / 12,114 patients**: the previous
3,083 added records and 10,901 new Primary-5-negative records. These negatives
can have other ECG diagnoses; they are not described as healthy controls.
The prior 3,083 tensors, targets and masks are preserved exactly.
Together with the original 8,393 rhythm records, the training stream has
22,377 ECG entries. There are **16,413 observed pathology references**, of
which 3,539 (21.56%) have at least one Primary-5 target, versus 64.21% before.
Unknown-source pathology references stay masked, never converted to negatives.
Additional native rhythm targets use the existing official singleton mapping;
2,361 uncertain/unsupported additional rhythm targets remain `-1` and masked.

Pathology calibration uses **2,027 eligible fold-9 ECGs / 1,826 patients**,
including the prior 386 cases. Primary-5 positive supports are IMI 234,
ALMI 21, ILMI 41, LAE 31 and ISC_ 109. No training, historical or follow-up
native patient enters this population. Selection uses metadata/folds, not
model probabilities, error status, confidence or desired metrics. Multiple
recordings of one patient stay in the same role. Independent native source
waveform hashes are checked against the official release index.

The original rhythm validation (1,476 ECGs), follow-up (2,171 ECGs, including
1,022 observed native pathology records) and both historical reference
columns (672 ECGs) remain unchanged. These already-examined comparisons are
follow-up, not a new untouched final test or evidence of source independence.
Original initialization lineage is not reconstructed by the new exclusions.

Primary rhythm endpoints use the same nine classes. Primary pathology
endpoints use IMI, ALMI, ILMI, LAE and ISC_. Pathology accuracy is the mean
of five binary-label accuracies, not a five-way classification accuracy.
Official SCP presence, including diagnostic likelihood zero, follows the
existing registry and does not establish independent clinical confirmation.

## Results

Job 29418 completed with exit code `0:0` in **6m44s**. The joint selector
chose `joint_natural_100_e08`; the rhythm selector chose
`joint_natural_100_e04`. Both selected fits reproduced every state tensor
exactly. All twelve original validation checkpoints, the frozen control,
and both final-role follow-up models were replayed from original signals on CPU.

| Model | Rhythm validation accuracy / macro-F1, 1,476 | Pathology validation macro-F1, 2,027 | Rhythm follow-up accuracy / macro-F1, 2,171 | Pathology follow-up mean label accuracy / macro-F1, 1,022 |
|---|---:|---:|---:|---:|
| Frozen start, expanded calibration | 84.62% / 84.08% | 40.50% | 82.91% / 82.10%, 1,800 correct | 93.87% / 46.45% |
| Joint-selected, epoch 8 | 85.16% / 84.60% | 43.82% | 82.77% / 82.23%, 1,797 correct | 94.56% / 48.40% |
| Rhythm-selected, epoch 4 | 85.43% / 84.82% | 42.92% | 82.96% / 82.41%, 1,801 correct | 94.58% / 42.64% |

The preceding compatible model, using its old 386-case calibration, scored
93.13% / 45.39% for pathology. Calibration alone now gives 93.87% / 46.45%;
the joint fit then gives 94.56% / 48.40%. Thus the measured fitting gain over
the equally recalibrated control is 1.94 macro-F1 percentage points, not the
entire change versus the previous calibration. IMI false positives decrease
92 to 80 while true positives increase 73 to 81. LAE instead declines from
10/30 to 5/30 true positives. The older phase6 pathology score, 48.51%, is
still slightly higher than this new joint follow-up score.

Rhythm-selected corrects 24 prior errors and introduces 23, giving one net
correct case. SB correct predictions fall 250 to 242, while 1AVB rises 136
to 144. NSR and RBBB remain concentrated failures. This is not a substantial
uniform improvement, statistical-significance claim, or achievement of
90% accuracy / approximately 87% macro-F1. No new model replaces Gold or
is promoted as an improvement of both tasks.

| Historical 672 reference | Joint-selected correct / accuracy / macro-F1 | Rhythm-selected correct / accuracy / macro-F1 |
|---|---:|---:|
| Original archived labels | 498 / 74.11% / 73.63% | 494 / 73.51% / 73.83% |
| Final archived labels | 564 / 83.93% / 81.91% | 567 / 84.38% / 82.80% |

Both historical reference versions remain unchanged. These source-inference
scores do not reconstruct the historical 605/672 run.

The PTHs, twelve candidate validation arrays/checkpoints, full follow-up CSVs,
both historical 672-row tables, original file inventories, actual executed
source/protocol, deterministic refit receipt and sanitized Slurm logs are
included. All 16,011 added/calibration source tensors and record manifests
were regenerated byte-identically; the original pathology target NPZ was
also regenerated from official metadata with the original SHA-256. Portable
follow-up replay reproduces every rhythm and pathology decision vector;
maximum probability discrepancy is `2.45e-6`, within the declared `2e-5`.

## Reproduction

Run from the repository root with the pinned environment. Source waveforms
and large training tensors stay outside Git in `data/cache/`.

```bash
python scripts/prepare_expanded_pathology.py --output data/cache/expanded_rebuilt --metadata outputs/reviewer_verification/expanded_population_2026-10-13/data/ptbxl_database.csv --scp-statements outputs/reviewer_verification/expanded_population_2026-10-13/data/scp_statements.csv --source-index outputs/reviewer_verification/expanded_population_2026-10-13/data/official_SHA256SUMS.txt --transport s3
python scripts/verify_expanded_evidence.py --bundle outputs/reviewer_verification/expanded_population_2026-10-13
python scripts/run_joint_inference.py --bundle outputs/reviewer_verification/expanded_population_2026-10-13 --output data/cache/expanded_joint_replay
python scripts/run_joint_historical.py --bundle outputs/reviewer_verification/expanded_population_2026-10-13 --output data/cache/expanded_historical_replay
python scripts/replay_expanded_validation.py --bundle outputs/reviewer_verification/expanded_population_2026-10-13 --output data/cache/expanded_validation_replay
```

Use `--offline` after official signals are cached; use fresh output directories.
Every downloaded/header/signal byte and prepared tensor is hash-checked.
The S3 transport is the public PhysioNet mirror listed by the official release;
the same official SHA-256 index is required with either transport.

For fitting, rebuild the original three NPZ arrays with the prior refinement
preparer/runbook; their exact hashes are in `training/INPUT_SHA256.json`.
Copy the rebuilt `path_validation.npz` into an isolated copy of `training/`.
Submit that copy with `scripts/manage_joint_cedia_run.py`, using the rebuilt
expanded directory for `--path-inputs` and original raw inputs for `--raw-inputs`.
The manager verifies the complete input inventory before `sbatch`.
`training/job.sbatch` runs the original archived source in the CEDIA environment.
Do not train on the login node or use a CPU training fallback.

This is a new continued-fit experiment, not recovery of the original historical
Gold run. Original/reference checkpoints, labels and historical source reports
remain preserved. Project-authored code is MIT; project research artifacts
follow the separate CC BY 4.0 notice. No clinical deployment claim is made.
