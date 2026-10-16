# CEDIA Partial-Backbone Refinement

**Run date:** 2026-10-12

**Slurm job:** 29336
**Status:** research candidate; the Gold checkpoint was not replaced.

## Model and Run

This run warm-started from the phase-2 checkpoint (`763ca7c30a17cbe69d56b21204869bf1fdd6444fc8b9b6d5bf80d0b7440e11d1`), itself initialized from Gold. It did not initialize a new network. The original CNN, four output heads, 1,143,751 total parameters, six frontal leads (I, II, III, aVR, aVL, aVF), 125 Hz sampling, and `[6, 1250]` input were retained. Training updated only the existing final residual block and rhythm head (460,065 parameters); labels were not changed.

CEDIA used 8,393 training records and 1,476 validation records. The 2,171 follow-up records were not used for fitting or candidate selection. The selected checkpoint is `model/partial_backbone_candidate.pth` (SHA-256 `6410de24a38a8021cb68fc2a51c275fac8fd16009d5bb7124b680e540e5eb5f4`). The exact source and configuration files used for the CEDIA run are in `training/phase3/`.

## Measured Results

| Evaluation | Phase 2 start | Partial-backbone candidate |
| --- | ---: | ---: |
| Validation accuracy | 1,182/1,476 (80.08%) | 1,185/1,476 (80.28%) |
| Validation macro-F1 | 78.79% | 78.93% |
| Validation mean source macro-F1 | 60.71% | 61.00% |
| Follow-up accuracy | 1,713/2,171 (78.90%) | 1,712/2,171 (78.86%) |

Across the paired follow-up predictions, 23 labels changed: 5 formerly incorrect cases became correct and 6 formerly correct cases became incorrect. The net change was one fewer correct prediction. This candidate therefore remains an experimental checkpoint, not a performance replacement.

| Follow-up cohort | N | Candidate accuracy | Candidate macro-F1 | Phase-2 accuracy |
| --- | ---: | ---: | ---: | ---: |
| external399 | 399 | 82.96% | 72.79% | 83.46% |
| new_js | 301 | 86.38% | 39.92% | 86.38% |
| new_ptb | 111 | 59.46% | 24.39% | 59.46% |
| new_source | 449 | 77.73% | 76.93% | 77.51% |
| posthoc239 | 239 | 73.64% | 73.53% | 73.64% |
| posthoc672 | 672 | 78.87% | 78.30% | 78.87% |

Macro-F1 uses the fixed nine-class set. Two cohorts do not contain all nine reference classes, so their macro-F1 is shown with class support in `evaluation/followup_metrics.json` and must not be read as a stand-alone aggregate score.

The remaining errors in `posthoc672` concentrate in RBBB (50.7% recall), PAC (60.8%), NSR (65.3%), and PVC (78.7%). No evaluation label was edited based on model predictions. Any clinical reference revision requires a blinded, attributable clinician review record.

## Reproduction and Evidence

- `evaluation/followup_predictions.csv`: per-record reference, predictions from the candidate and comparison checkpoints, and candidate probabilities for all 2,171 records.
- `evaluation/followup_metrics.json`, `cohort_metrics.csv`, and `per_class_recall.csv`: aggregate, confusion-matrix, and class-level evidence.
- `validation_candidates/`: per-case validation probabilities for each evaluated candidate and `candidate_metrics.csv`.
- `training/phase3/`: exact CEDIA training source, protocol, input-hash manifest, source-data receipt, split-overlap report, Slurm logs, and the CEDIA result hash inventory.
- `manifests/`: minimal record manifests with source version, target label, and source/header/preprocessed-tensor hashes. Waveform files are not included; they must be obtained from the versioned PhysioNet sources cited in the repository's `REFERENCES.md`.
- `SHA256SUMS.json`: package-level checksums.

### Repeat Inference From Public Signals

From the repository root, install `requirements.txt` in a Python 3.12 environment and install the package with `python -m pip install --no-deps -e .`. Then run:

```bash
python scripts/verify_partial_backbone_evidence.py
python scripts/run_refinement_inference.py
```

The second command downloads only the 2,171 manifest records from the pinned releases. It verifies every header, signal-file and preprocessed-tensor SHA-256 before applying the candidate PTH. It writes new probabilities, per-cohort metrics and a comparison receipt in `data/cache/refinement_reproduced/`. A nonzero exit status indicates a prediction or probability mismatch. Use a fresh `--output` directory for each run; `--offline` reuses an already verified source cache. `--device cuda` is optional; the recorded portable verification used CPU.

The `posthoc672` cohort here is PTB-XL and is **not** the historical 672-record Challenge cohort. All six cohorts were examined during earlier development and are follow-up comparisons, not an untouched final test. Validation selection never used their targets or predictions.

### Reconstruct the CEDIA Training Inputs

```bash
python scripts/prepare_refinement_training_inputs.py --output-dir data/cache/refinement_training_inputs
```

This command downloads the training, validation and follow-up signals, enforces their frozen tensor hashes, and reconstructs `train.npz`, `validation.npz` and `evaluation.npz`. The training inputs contain the original labels, Gold features/probabilities, and training-only frozen teacher probabilities. The teacher sidecar combines 75% expanded attention-MIL and 25% prior partial-backbone probabilities; it does not replace reference labels. The sidecar arrays, their preparation protocol and hashes are in `training/phase2_reference/`. The verified portable reconstruction produced all three NPZ files byte-identically to CEDIA's original inputs; see `training/portable_input_rebuild_receipt.json`. No source ECG waveforms are checked in.

Transfer the reconstructed directory to CEDIA, retain its generated `INPUT_SHA256.json`, and run `sbatch job.sbatch` from that directory using the measured Python 3.10 / PyTorch 2.1.2+cu121 / NumPy 1.26.4 / CUDA 12.1 runtime. The archived job uses `$HOME/V4_CEDIA_LINK/.venv_audit/bin/python`; update that interpreter path for another account while recording the change. Gradient training must run in a Slurm GPU allocation, not on the login node. The deterministic CEDIA refit receipt is in `training/refit.json`. Byte-identical inputs do not guarantee identical training outputs on an untested GPU/software stack.

`training/phase3/prepare_raw.py` is an archived original-project-layout preparer. The portable root script above replaces its workstation-relative paths without changing arrays or reference labels. Original execution inventories are retained for provenance; account-specific home paths in error logs are redacted, with original and public hashes in `training/log_redaction_receipt.json`.

The portable CPU re-inference from the original WFDB source files matched all 2,171 CEDIA Top-1 predictions; the maximum absolute probability difference was `1.0133e-6`. The full file/tensor-check and environment receipt is in `evaluation/portable_inference_receipt.json`.

## Complementary Weight Interpolation

Slurm job 29342 evaluated ten convex mixtures with two previously trained checkpoints, without gradient updates or extra inference parameters. The exact code, donor PTHs, coefficient grid, validation probabilities, execution receipt, class metrics, auxiliary-output comparisons and follow-up outputs are in `training/weight_interpolation/`.

The strongest mixture reached 83.54% accuracy and 82.71% macro-F1 **on validation**, but its normalized biometric/pathology output errors were 0.341/0.426 relative to the base. It did not satisfy the predeclared preservation protocol. No mixture meeting that protocol exceeded its minimum selection improvement, so the base candidate was retained. This validation finding is not reported as follow-up performance or a replacement of the four-head model.

To prepare this comparison after reconstructing the training inputs, run:

```bash
python scripts/prepare_refinement_interpolation_inputs.py --training-inputs data/cache/refinement_training_inputs --output-dir data/cache/refinement_interpolation_inputs
```

Transfer that fresh directory to CEDIA and submit its archived `job.sbatch` using the measured GPU runtime. The preparer verifies the reconstructed validation/evaluation checksums and writes a new `INPUT_SHA256.json` covering all replay inputs. `cedia_result_inventory.json` records the original output hashes; the selected base PTH is already distributed under `model/`.

This experiment is separate from the historical CEDIA 605/672 arrhythmia aggregate and does not reproduce or revise the reported 90.03% accuracy / 87.82% macro-F1 result.
