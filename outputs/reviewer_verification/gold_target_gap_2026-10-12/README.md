# Historical-Cohort Target Gap Audit

This is a fresh diagnostic comparison on the historical 672-record Challenge cohort, not the PTB-XL `posthoc672` cohort and not a reconstruction of the historical inference run. No target labels or waveform records were changed. All 1,344 source-file hashes match the archived repository Gold inference.

## Measured Comparison

| Checkpoint | Original-reference correct / accuracy / macro-F1 | Final-reference correct / accuracy / macro-F1 |
| --- | --- | --- |
| Distributed Gold | 466/672 / 69.35% / 68.88% | 568/672 / 84.52% / 82.35% |
| Partial-backbone candidate | 479/672 / 71.28% / 71.14% | 576/672 / 85.71% / 83.81% |
| Validation-ranked exploratory mixture | 495/672 / 73.66% / 73.32% | 569/672 / 84.67% / 82.81% |

The candidate adds eight final-reference correct predictions versus the distributed Gold checkpoint. Reaching at least 90% requires 605/672: **29 additional net correct predictions**. That count alone does not ensure the historical 87.82% macro-F1, which also depends on class-level false positives and false negatives.

The original historical CEDIA report is a distinct source aggregate: 476/672 on original labels and 605/672 on final labels. Neither is replaced by these fresh diagnostic runs. The final-reference labels are archived source values, not newly certified clinical reference readings.

The exploratory mixture was chosen for this diagnostic by its previously recorded validation accuracy, not by historical-cohort predictions. It remains unpromoted: its ECG-arrhythmia validation SB recall fell from 95% to 88%, and its biometric/pathology output drift exceeded the predefined preservation protocol. This output drift is not a measured clinical pathology error rate.

## Training Objective Finding

The recorded rhythm loss is `CE(y,q) + a*KL(Gold||q) + 0.15*KL(teacher||q)`. For unrestricted independent per-record probabilities, its exact minimizer is:

`q* = (one_hot(y) + a*Gold + 0.15*teacher) / (1 + a + 0.15)`.

With `a=2`, that minimizer's highest-probability class differs from the training reference in **1,007/8,393** cases: NSR 317, RBBB 247, PAC 108, PVC 104, STACH 63, SB 55, 1AVB 53, AFIB 48 and LBBB 12. With `a=5`, disagreements rise to 1,645. The audited `a=0`, `0.1` and `0.5` alternatives have zero such disagreements in these frozen arrays.

This identifies a conflict in the isolated rhythm objective, not a measured improvement from changing it. The shared CNN, auxiliary losses, optimization and data quality still matter. It is not a proof of a network accuracy ceiling. Source/class weighting multiplies the per-record rhythm objective and does not change this isolated minimizer.

The training teacher matches 7,134/8,393 references (85.00%) versus Gold's 6,304/8,393 (75.11%). On RBBB, the teacher matches 764/1,087 (70.29%); on NSR, 1,643/2,120 (77.50%). Copying teacher outputs more strongly is therefore not itself a demonstrated path to 90%.

## What To Test Next

1. Ablate Gold anchoring using `0`, `0.1` and `0.5`, alongside an agreement-gated anchor computed only from training references. Keep hard reference labels unchanged. Select exclusively on validation, not the 29 historical-cohort errors.
2. Adapt the existing final two blocks and rhythm head rather than reset the model. Compare frozen and training-only updated normalization statistics. Preserve total parameters, six leads, 125 Hz and four outputs; monitor existing auxiliary outputs and SB/LBBB per source.
3. Improve training support and teacher reliability for PAC/PVC and NSR/RBBB, not merely the overall record count. Current training has 597 PAC, 638 PVC and 467 PTB-XL RBBB examples, of which 362 are IRBBB and 105 CRBBB. Subtype support must be audited by patient and source before adding records.
4. Report final-reference and original-reference results separately and reserve an untouched patient-disjoint evaluation for any new generalization claim.

On original historical references, PAC accounts for 74 errors and PVC for 54 of the candidate's 193 errors. On final references, the largest remaining counts are NSR 23, PVC 18, PAC 12, RBBB 11 and STACH 11. These are different error distributions and must not be combined to select targets or modify labels.

The six frontal leads omit V1/V2 morphology used in clinical RBBB definitions. This motivates testing reduced-lead representations; it does not prove 90% is unattainable. See the [AHA/ACCF/HRS conduction statement](https://www.jacc.org/doi/10.1016/j.jacc.2008.12.013). Patient-level evaluation follows the separation principles documented by [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/). Weight interpolation follows the experimental idea in [Wortsman et al.](https://arxiv.org/abs/2203.05482), whose image-model findings do not establish an ECG performance gain.

## Evidence and Replay

- `historical672_paired_predictions.csv`: unchanged original/final references, all three predictions and 27 probability columns for each of the 672 records.
- `source_files_sha256.csv`: all 1,344 WFDB source-file hashes; no waveforms included.
- `inference_provenance.json`: checkpoint/code hashes and measured runtimes for these prediction runs.
- `gap_audit.json`: full metrics, matrices, class errors, training-objective counts and exact/alias record checks.
- `model/validation_strong_mixture.pth`: unpromoted exploratory mixture; candidate and Gold PTHs remain at their existing repository paths.

```bash
python scripts/audit_gold_target_gap.py
python scripts/run_primary9_inference.py --checkpoint outputs/reviewer_verification/partial_backbone_refinement_2026-10-12/model/partial_backbone_candidate.pth --out-dir data/cache/phase3_historical_diagnostic
python scripts/run_primary9_inference.py --checkpoint outputs/reviewer_verification/gold_target_gap_2026-10-12/model/validation_strong_mixture.pth --out-dir data/cache/mixture_historical_diagnostic
```

The existing inference runner generates predictions against original references; score its unchanged predictions against both archived reference columns as in `scripts/audit_gold_target_gap.py`. The generic runner's unknown-lineage metadata must not be interpreted as a fresh split audit of these research candidates. Their added train/validation manifests show zero exact or numeric PTB alias matches with these historical IDs; this does not establish patient identity across releases or independence from initial Gold training.

No additional gradient training was performed for this audit. The proposed loss/backbone ablations have not yet been run and are not reported as improvements.
