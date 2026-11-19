# Supervised Pathology and Rhythm Fine-Tuning

This experiment continues the existing phase5 model. It does not initialize a new network, reset layers, change reference labels, or replace the distributed Gold checkpoint. The architecture remains 1,143,751 parameters, four heads, six frontal leads, 125 Hz, and an input of `[6,1250]`.

## Data and Endpoints

- Rhythm development: the existing 8,393 training and 1,476 validation records, with the original nine-class references and fixed teacher probabilities.
- Additional pathology development: 3,083 PTB-XL v1.0.3 records with at least one Primary-5 statement, from folds 1-8 only. Validation, follow-up and historical-cohort patients are excluded. The first expansion audit identified 108 historical-overlap records; those records are not used.
- Pathology supervision: all ten existing output columns are preserved. Primary-5 results use IMI, ALMI, ILMI, LAE and ISC_. The official `LAO/LAE` statement maps to LAE. Statement presence, including likelihood zero, follows the repository's existing registry. Other-source pathology targets are unknown and masked, not negative.
- Additional rhythm supervision: use only official SCP statements with exactly one non-NSR Primary-9 class, or SR alone. The full twelve-statement official rhythm domain is checked, including statements absent from the project mapping. Multiple rhythms, absent statements and statements outside Primary-9 are masked, not assigned an invented target: 2,510 additional records have usable targets and 573 are masked. A matched rhythm-expansion control measures the contribution separately from supervised pathology.
- Follow-up: the unchanged 2,171 records from the earlier six comparisons. Pathology references are observed for the 1,022 native PTB-XL records only; other sources do not enter that endpoint.

Rhythm accuracy is single-label accuracy; macro-F1 always uses the fixed nine classes. Pathology accuracy is the mean binary accuracy across five labels, not exact-match accuracy. Its macro-F1, positive/negative supports, recall, specificity and confusion counts are reported separately. A large number of negative labels can make binary accuracy high without making F1 high.

The upstream source is [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/), DOI [10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45), under CC BY 4.0. The database SHA-256 is `7600de9c1b27d181d850b3c6038a35d7c3ddb6bb33b702e3a20252a6859d216b`. Cite its original [Scientific Data paper](https://doi.org/10.1038/s41597-020-0495-6) and the source references in [REFERENCES.md](../../../REFERENCES.md).

## Predeclared Comparisons

The [protocol](training/protocol.json) fixes six 32-epoch trials before execution: rhythm continuation, rhythm expansion, pathology-head-only control, two joint last-two-block trials, and a joint full-backbone trial at a low learning rate. Every trial starts from the exact same phase5 state. Batch-normalization statistics stay frozen; the quality and biometric head parameters stay frozen. Joint trials train the existing pathology head with masked, training-weighted BCE. They do not preserve its old logits as if those were pathology reference labels.

The validation selector preserves supported-class recall and both rhythm macro-F1 criteria relative to the starting candidate. It chooses a joint checkpoint using source-mean rhythm F1 plus half the Primary-5 F1, and retains a separate rhythm-best checkpoint for comparison. Thresholds use validation records only; follow-up targets are opened only after checkpoints and thresholds are saved. Both heads of each PTH are evaluated, so scores from different checkpoints are not combined into a single claimed model result.

## Measured Results

The public result set is the corrected rerun, not superseded job 29349. Independent review found 56 extra rhythm targets that had ignored unsupported official statements (`SARRH`, `SVARR`, `SVTAC`, `PSVT`, `BIGU`, `TRIGU`). Those training targets were changed to unknown, with all source tensors, pathology targets, original development/comparison labels, baseline, seed and optimization settings unchanged. The exact correction is recorded in `data/taxonomy_correction.json`; the original executed snapshot is retained separately rather than modified after execution.

Development started on 2026-10-12; corrected execution completed on 2026-10-13 in CEDIA job **29350**, exit `0:0`, elapsed `00:07:48`. The measured stack was Python 3.10.14, PyTorch 2.1.2+cu121, NumPy 1.26.4, CUDA 12.1, and an A100-SXM4-40GB. Thirty checkpoints were compared on validation. Both selection objectives chose `joint_100_e08`; their PTH files contain bitwise-identical learned states. Both selected-trial refits reproduced every state tensor bitwise. Distinct serialized filenames have distinct file hashes, which are recorded separately.

| Model | Validation rhythm accuracy / macro-F1, 1,476 | Follow-up rhythm accuracy / macro-F1, 2,171 | Follow-up Primary-5 mean label accuracy / macro-F1, 1,022 |
|---|---|---|---|
| Starting phase5 | 82.52% / 81.03% | 81.30% / 80.41% | 93.33% / 46.80% |
| Joint-selected, epoch 8 | 84.28% / 83.37% | 82.17% / 81.53% | 94.44% / 48.51% |
| Rhythm-selected, epoch 8 | 84.28% / 83.37% | 82.17% / 81.53% | 94.44% / 48.51% |

The selected model adds **19 correct predictions** on the identical 2,171 records, from 1,765 to 1,784. Its pathology result reduces Primary-5 false positives from 201 to 145 and increases true positives from 135 to 136. This is a measured but modest F1 gain; binary accuracy alone is not evidence of strong pathology detection.

| Follow-up cohort | N | Starting correct | Joint-selected correct | Rhythm-selected correct | Rhythm-selected accuracy / fixed-nine macro-F1 |
|---|---|---|---|---|---|
| external399 | 399 | 345 | 348 | 348 | 87.22% / 76.79% |
| new_js | 301 | 267 | 272 | 272 | 90.37% / 41.48% |
| new_ptb | 111 | 69 | 68 | 68 | 61.26% / 24.01% |
| new_source | 449 | 356 | 360 | 360 | 80.18% / 79.28% |
| posthoc239 | 239 | 180 | 183 | 183 | 76.57% / 76.71% |
| posthoc672 | 672 | 548 | 553 | 553 | 82.29% / 81.97% |

Some cohorts do not contain all nine classes. Their fixed-nine F1 includes zero for unsupported classes; the 90.37% figure on `new_js` is not the overall nine-class result and is not paired with 87% macro-F1. `posthoc672` contains native PTB-XL records, not the historical Challenge cohort.

| Model | Historical original references, correct / 672 | Historical final references, correct / 672 | Historical final accuracy / macro-F1 |
|---|---|---|---|
| Starting phase5 | 493 | 571 | 84.97% / 82.87% |
| Joint-selected | 498 | 564 | 83.93% / 82.03% |
| Rhythm-selected | 498 | 564 | 83.93% / 82.03% |

The new models do not improve the historical final-reference score. Gold and all earlier research models remain unchanged, including the earlier candidate scoring 576/672 on that reference version. The historical 605/672 aggregate is not attributed to this new execution. Both original and final references are scored from exactly the same newly inferred probabilities, without altering labels or replacing records.

The remaining rhythm errors concentrate in RBBB (149/243 recognized), NSR (310/415), and PAC (97/143) in follow-up. Pathology recognition remains weakest for LAE (10/30 positives detected) and ISC_ (34/77). These counts describe observed errors; they do not establish one unique cause or justify changing reference labels.

## Reproduce

From the repository root, after installing the pinned dependencies:

```bash
python scripts/verify_joint_evidence.py
python scripts/verify_joint_sources.py
python scripts/run_joint_inference.py --output data/cache/joint_reproduced
python scripts/run_joint_inference.py --model rhythm_selected --output data/cache/joint_rhythm_reproduced
python scripts/run_joint_historical.py --model rhythm_selected --output data/cache/joint_history_reproduced
```

The replay downloads the original release files, verifies header, signal and preprocessed tensor hashes, and checks both probability arrays and thresholded decisions against CEDIA. `--offline` uses a populated cache. It runs inference on CPU; training is prohibited outside a CEDIA GPU allocation.

Reconstruct the training inputs, including the additional signals:

```bash
python scripts/prepare_refinement_training_inputs.py --output-dir data/cache/joint_raw_rebuilt
python scripts/prepare_joint_pathology_inputs.py --output data/cache/joint_path_rebuilt
```

The additional-data builder downloads pinned metadata and the official release checksum index, verifies both hashes, reconstructs targets from SCP statements, excludes the comparison patients, and writes `path_extra.npz`, `path_targets.npz`, record manifests and a preparation receipt. Source recordings remain outside Git.

Historical alias `HR00141` is a deleted duplicate of retained native record `2509`, as recorded in the [official v1.0.3 change log](https://physionet.org/files/ptb-xl/1.0.3/ptbxl_v103_changelog.txt). Its retained patient is excluded too; all 378 historical PTB-XL aliases are resolved for the patient check.

On CEDIA, put the three reconstructed raw NPZ files, the two pathology NPZ files, and the unchanged files in `training/` in a fresh run directory. Verify them against `training/INPUT_SHA256.json`, then submit `sbatch job.sbatch`. The Python runtime must match the recorded execution stack and be available at `$HOME/V4_CEDIA_LINK/.venv_audit/bin/python`. The trainer checks the full input inventory before fitting and enforces a CEDIA Slurm GPU allocation. The selected-trial refit must reproduce all state tensors exactly on that stack.

The repository also supplies `scripts/manage_joint_cedia_run.py` for the configured SSH alias, isolated staging, status and sealed-result retrieval. It uses strict host-key checking and never overwrites an existing remote run or local result directory.

Intermediate PTHs are retained in the sealed CEDIA/local research run rather than duplicating all thirty in Git. Their original hashes are in `results/SHA256SUMS.json`. To replay their validation probabilities and auxiliary outputs after refitting or retrieving that directory:

```bash
python scripts/replay_joint_validation_auxiliary.py --checkpoints-dir YOUR_REPLAY/results --output data/cache/joint_validation_replayed
```

The quality and biometric outputs are consistency measurements only, not measured clinical quality, age, sex or BMI accuracy.

## Evidence and Interpretation

`results/selection.json` contains every validation candidate and the saved thresholds. Validation NPZ files permit recalculation of the selector. Per-record CSVs contain unchanged rhythm references, all nine rhythm probabilities, ten pathology scores, observed pathology references, and thresholded outputs. `data/` identifies source metadata, release versions, file/tensor hashes, targets and patient exclusion checks. `model/` holds the separate research PTHs; `training/` holds the warm start and exact execution source. The bundle inventory hashes every published artifact.

These fixed comparison cohorts have already been examined during development. Their scores are follow-up measurements, not a new untouched clinical test and not a reproduction of the historical Gold run. The 90% rhythm-accuracy / approximately 87% macro-F1 figures are targets for the research, not guaranteed or inferred results. This research experiment does not establish clinical diagnostic validity.
