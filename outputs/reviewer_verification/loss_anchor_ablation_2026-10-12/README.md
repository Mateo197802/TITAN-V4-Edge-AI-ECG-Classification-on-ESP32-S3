# CEDIA Training-Only Loss Ablation

Slurm job **29344** completed successfully (`0:0`, elapsed `00:11:45`). Six predeclared trials evaluated three checkpoints each, selected only on the 1,476-record validation set. No evaluation labels or records were changed. Gold and the prior research candidate are retained.

## Model and Method

- Original architecture: 1,143,751 parameters, four output heads, six frontal leads I/II/III/aVR/aVL/aVF, 125 Hz, `[6,1250]` input.
- Strict warm start: the prior candidate, SHA-256 `6410de24a38a8021cb68fc2a51c275fac8fd16009d5bb7124b680e540e5eb5f4`. No layers were reset.
- Selected update: the existing final two residual blocks and rhythm head, 759,445 trainable parameters. All other parameters and all batch-normalization buffers stayed bitwise unchanged.
- References: the same frozen 8,393 training, 1,476 validation and 2,171 follow-up records. Public source manifests and frozen teacher/feature sidecars remain in the prior refinement bundle.
- Training: CEDIA GPU only, PyTorch 2.1.2+cu121, NumPy 1.26.4, CUDA 12.1, A100-SXM4-40GB. Inference was independently repeated on local CPU; no neural-network fitting was run locally.
- Loss comparisons: unmasked Gold coefficients 2/0/0.1/0.5, training-reference agreement masking, and one source/class-weight sensitivity trial. A same-settings control pair isolates Gold coefficient 2 versus 0. Agreement masks change only the strength of teacher regularization, never hard target labels.
- Selected trial: `anchor05_block2_e04`, learning rate `1e-4`, Gold coefficient `0.5`, four epochs. Selection used source-mean fixed-nine-class macro-F1 with predeclared class/output preservation criteria. Follow-up labels were loaded only after writing the selection.

Selected PTH SHA-256: `552372cf5141b9f86643661e7c5d18209fa22627de583ca801fb543342cd7c93`.

## Measured Results

| Comparison | Prior candidate | Selected loss-ablation candidate |
| --- | ---: | ---: |
| Validation correct | 1,185/1,476 | 1,218/1,476 |
| Validation accuracy | 80.28% | 82.52% |
| Validation macro-F1 | 78.93% | 81.03% |
| Summed follow-up correct | 1,712/2,171 | 1,765/2,171 |
| Summed follow-up accuracy | 78.86% | 81.30% |
| Historical 672, original references | 479/672; 71.28%; macro-F1 71.14% | 493/672; 73.36%; macro-F1 72.96% |
| Historical 672, final references | 576/672; 85.71%; macro-F1 83.81% | 571/672; 84.97%; macro-F1 82.87% |

The selected model adds 33 validation correct predictions and 53 summed follow-up correct predictions. It improves all six current follow-up comparisons:

| Cohort | N | Prior accuracy | New accuracy | New macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| external399 | 399 | 82.96% | 86.47% | 76.18% |
| new_js | 301 | 86.38% | 88.70% | 40.81% |
| new_ptb | 111 | 59.46% | 62.16% | 25.55% |
| new_source | 449 | 77.73% | 79.29% | 78.16% |
| posthoc239 | 239 | 73.64% | 75.31% | 75.18% |
| posthoc672 | 672 | 78.87% | 81.55% | 81.17% |

All these cohorts were already examined during development. Their summed accuracy is descriptive across these fixed comparisons, not a new untouched external-test claim. Macro-F1 uses all nine classes; `new_js` and `new_ptb` have incomplete class support. `posthoc672` is PTB-XL, not the historical Challenge cohort.

On the historical cohort, reference versions give different answers: the new model gains 14 original-reference correct predictions but loses five final-reference correct predictions. The prior PTH therefore remains available; this run does not achieve or reconstruct the historical 605/672 / 87.82% macro-F1 result. No training or candidate selection used these historical predictions.

The controlled twelve-epoch comparison improved validation accuracy from 80.62% at coefficient 2 to 81.03% at coefficient 0, with all other settings identical. The highest overall validation-accuracy trial reached 83.40% / 82.61% macro-F1 but did not meet the predeclared class-preservation criteria; it was not promoted. Both outcomes are retained in the candidate table.

## Reproduction

Install the repository's pinned requirements and package. From the root:

```bash
python scripts/verify_loss_ablation_evidence.py
python scripts/run_refinement_inference.py --checkpoint outputs/reviewer_verification/loss_anchor_ablation_2026-10-12/model/selected.pth --expected-predictions outputs/reviewer_verification/loss_anchor_ablation_2026-10-12/results/followup_predictions.csv --prediction-column phase5_prediction --output data/cache/loss_ablation_reproduced
python scripts/prepare_loss_ablation_training_inputs.py --output-dir data/cache/loss_ablation_training_inputs
```

The inference command obtains only pinned public WFDB records and verifies every header, waveform and preprocessed-tensor hash. Use `--offline` after caching and a fresh output directory for each run. CPU re-inference matched all 2,171 CEDIA predictions, maximum absolute probability difference `1.0431e-6` (tolerance `2e-5`). No source waveforms are redistributed.

The preparer rebuilds the same train/validation/evaluation NPZ files and copies the exact training core, configuration and initial PTH into a fresh directory. Its verified NPZ files are byte-identical to the original CEDIA inputs. Transfer that directory to CEDIA and submit `sbatch job.sbatch` using the measured runtime. The job's Python interpreter is `$HOME/V4_CEDIA_LINK/.venv_audit/bin/python`; adapt the interpreter/resource paths for another account and record any changes. Gradient execution requires a CEDIA Slurm GPU allocation. The preparer retains inherited phase-3 source files as provenance; only `train_phase5.py` is the entrypoint for this replay.

To repeat the historical original-reference diagnostic:

```bash
python scripts/run_primary9_inference.py --checkpoint outputs/reviewer_verification/loss_anchor_ablation_2026-10-12/model/selected.pth --out-dir data/cache/loss_ablation_historical672
```

Score the unchanged predictions against both frozen columns as in `scripts/verify_loss_ablation_evidence.py`. The generic historical runner's initial-Gold-lineage metadata is not a fresh split audit of this fine-tuning run. The inherited Gold training provenance is separate from the published added fine-tuning manifests.

## Evidence

- `model/selected.pth`, `training/start.pth`: selected and initial checkpoints, unchanged architecture.
- `results/candidate_metrics.csv`, `selection.json`, and 18 validation NPZ files: every candidate's predictions, reference targets, source/class metrics and selection criteria.
- `results/followup_predictions.csv`, `followup_metrics.json`: all 2,171 predictions, probabilities and source-tensor hashes.
- `historical672/paired_predictions.csv`: all 672 original/final references, old/new predictions and probabilities.
- `historical672/source_files_sha256.csv`: 1,344 source-file hashes, identical to the archived Gold signal run.
- `results/portable_inference_receipt.json`: independent CPU reproduction from original WFDB signals.
- `results/auxiliary_output_crosscheck.json`: independently recalculated validation output-preservation errors. These are not clinical pathology accuracy measurements.
- `results/refit.json`: selected-trial repetition produced bitwise identical state tensors.
- `training/`: exact core/configuration, original CEDIA inventories, Slurm accounting/logs and portable input-rebuild receipt. Account-specific home prefixes in warning logs are redacted with before/after hashes. Runtime diagnostics found an NFS wait during startup; the original job completed without a restart or changed model protocol.
- `results/verification_summary.json`, `SHA256SUMS.json`: independently recalculated metrics, architecture/state checks and byte-level inventory.

Intermediate candidate PTHs remain preserved in the original CEDIA run and local study archive; their original hashes are listed in `training/original_cedia_result_inventory.json`. The public bundle distributes the selected/initial PTHs and all candidate validation arrays. All six trials can be repeated using the frozen inputs and archived code. Supplemental contract tests are separate from the original submitted test file.

Dataset citations, upstream CC BY 4.0 terms and project source/artifact license scope remain in the repository's `REFERENCES.md`, `DATA_PROVENANCE.md` and `LICENSE-ARTIFACTS.md`. This is research software, not a clinical diagnostic device.
