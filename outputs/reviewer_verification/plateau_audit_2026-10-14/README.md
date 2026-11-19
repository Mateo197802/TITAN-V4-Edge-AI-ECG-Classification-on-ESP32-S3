# Plateau Diagnosis and Defensible Next Experiments

## Executed Scope

CEDIA job **29490**, read-only GPU inference on an A100, completed in
**3m02s**, exit `0:0`. Three actual retained checkpoints are scored on the
same 22,567 training entries (20,206 known rhythm references and 16,413
observed pathology references), 1,476 rhythm-validation and 2,027 native
pathology-validation records. **No optimizer, fitting or model selection
ran; no followup labels were loaded.** Every model parameter and buffer
remained bit-identical. The original 1,143,751 parameters, four heads,
six frontal leads and 125 Hz are unchanged.

Executed code, pinned input hashes, actual checkpoints, complete training
and validation probabilities/references/IDs, Slurm logs and private-path
redaction receipts are included. `audit.json` recomputes the diagnostic and
140 candidate entries from seven preceding studies. Entry count is not
140 independent trials: it includes multiple epochs and compositions.
This is a focused computational/methodological diagnosis, not a claim of
an exhaustive security, clinical or publication-readiness audit.

## What Explains the Plateau

### Rhythm: mix, generalization and shared errors

| Actual checkpoint | Raw train accuracy | Train accuracy at validation source/class mix | Validation accuracy / macro-F1 |
|---|---:|---:|---:|
| Phase6 | 75.05% | 84.41% | 84.28% / 83.37% |
| Phase10 joint | 78.07% | 89.70% | 85.16% / 84.60% |
| Phase12 composed | 78.57% | 89.31% | 85.43% / 84.82% |

The raw train/validation comparison confounds class and source mix. The
standardized train value reweights source/class recalls to the validation
counts, without changing predictions, labels or the reported validation
metric. The current **3.88-point** residual gap supports a generalization
problem as a working hypothesis; it does not prove a unique causal mechanism.
There is no evidence here that simply continuing to lower training loss
will produce the requested test result.

Current validation errors are NSR **62**, RBBB **53**, PAC **32**, PVC **22**,
1AVB **14**, AFIB **12**, SB **11**, STACH **5**, LBBB **4**. NSR/RBBB/PAC
account for **147/215, 68.37%** of errors. RBBB-to-NSR contributes 39.
These counts identify where to investigate, not records to exclude.

All three checkpoints are wrong on **185/1,476** validation records.
A target-knowing retrospective oracle that picks a correct answer whenever
any of those models has one obtains only **1,291/1,476, 87.47%**. That is
not a deployable ensemble, a clinical ceiling, or a bound on future models.
It demonstrates that combining just these three existing outputs cannot
reach 90% on this fixed validation population. The current model has 1,261
correct; 90% requires at least 1,329, or 68 net additional correct cases.

### RBBB: the hard subtype is not learned even in training

| Native subtype, current checkpoint | Training correct / support | Validation correct / support |
|---|---:|---:|
| IRBBB only | 274/604, 45.36% | 17/41, 41.46% |
| CRBBB only | 201/212, 94.81% | 14/15, 93.33% |

One validation record containing both statements is correct and is counted
separately. Subtype counts come from pinned native SCP statements, not
model-derived reference edits. This rules out explaining the whole IRBBB
problem as test-only overfitting: it already occurs among training inputs.
It does not determine whether insufficient observable information, label
heterogeneity or learned representation is dominant.

The deployed input omits V1-V6. Standard RBBB morphology includes right
precordial leads; see the primary
[AHA/ACCF/HRS statement, Part III](https://doi.org/10.1016/j.jacc.2008.12.013).
Applying labels originating from a 12-lead ECG to frontal-only inputs can
therefore create an observability mismatch. This is a hypothesis about this
project, not a diagnosis of any record or proof that all frontal-only RBBB
classification is impossible. More copies of the same six-lead evidence
cannot directly supply a missing precordial waveform.

### Pathology: ranking improves, decisions remain weak

| Actual checkpoint | Train / validation Primary-5 AP | Train / validation AUROC | Train / validation macro-F1 |
|---|---:|---:|---:|
| Phase6 | 39.13% / 35.70% | 90.84% / 89.59% | 43.40% / 41.05% |
| Phase10 joint | 47.31% / 39.12% | 93.38% / 90.65% | 44.97% / 43.82% |
| Phase12 composed | 52.00% / 41.58% | 94.39% / 91.29% | 47.59% / 45.59% |

All checkpoints use thresholds fitted on the same validation population
and frozen when scoring training. These are full-calibration diagnostic
values, not independent threshold-cross-fit or followup results. AUROC,
AP and F1 are different quantities: high AUROC does not establish high
precision or F1 for rare labels. Prevalence and every class's confusion
counts are preserved in the raw diagnostic. The rare-label imbalance and
remaining false positives/negatives warrant per-label selection checks,
not reporting mean binary accuracy alone.

The existing preprocessing independently z-normalizes each input lead.
A reproducible synthetic positive-gain control changes the final tensor by
only **1.31e-6**, showing that absolute signal scale is removed by this
pipeline. The synthetic control is not a patient recording, training sample
or performance experiment. Relative morphology remains; it would be wrong
to say normalization destroys every diagnostic feature. Voltage-dependent
criteria motivate a separate amplitude-preserving input ablation after
verifying WFDB units; see the primary
[AHA/ACCF/HRS statement, Part V](https://doi.org/10.1016/j.jacc.2008.12.015).
This is not proof that a new normalizer alone will raise F1.

### Selection is not the sole obstruction

Among 140 archived candidate entries, the highest rhythm validation
macro-F1 is **84.82%**. Removing the preservation screen would not turn any
of those measured entries into 90% accuracy / 87% macro-F1. A pathology
candidate reaches **52.27% validation macro-F1** but fails rhythm
preservation. This is an actual task tradeoff, not a reason to discard the
screen or combine endpoints from different checkpoints as one result.
The last independent-head round could only change the readout of its frozen
192-dimensional embedding; it could not relearn the extractor.

## Recommended Sequence

1. **Freeze the next comparison before fitting.** Keep the existing
   patient/source manifests and all nine rhythm / five pathology endpoints.
   Use the current rhythm checkpoint and an equally calibrated stronger
   pathology checkpoint as explicit separate controls. No test replacement,
   retrospective reference changes or oracle-based selection.
2. **Test representation adaptation, one factor at a time.** Compare a
   control with continued fitting of the last existing blocks using
   alternating task batches and subtype-aware training support. Preserve
   the network dimensions and original start state. Monitor IRBBB, PAC and
   NSR separately; never use those examined followup cases for selection.
   A train-only source-balanced BatchNorm recalibration control can isolate
   running-statistics mismatch before an expensive fit. Changes to those
   buffers must be declared, not called byte-identical preservation.
3. **Run a separate voltage-preserving preprocessing ablation.** First
   authenticate source units, then use a fixed/train-derived scale instead
   of per-record z-scoring. Keep six channels, 125 Hz and parameter count,
   but explicitly version this changed input contract and its warm-start
   continuation. Do not silently feed it into an old checkpoint or claim
   that its tensors match the original loader. Missing precordial signals
   remain missing; a 12-lead teacher during training is another distinct
   hypothesis, not an already available verified remedy.
4. **Require evidence beyond one validation peak.** Compare three fixed
   seeds, report per-class and per-source effects and patient-level
   uncertainty, retain rejected candidates and require protected-label
   preservation. Threshold cross-fitting is not model cross-validation.
   Select the configuration on development data, then evaluate a new
   patient-disjoint unexamined test once. Previously examined cohorts
   remain tracking cohorts, not that test.

These are proposed next experiments; **job 29490 did not train them**.
No physical information ceiling, causal root cause or 90% result has been
established. The measured evidence favors targeted representation and input
investigation over another broad epoch/LR search or more head-only fitting.
PTB-XL's primary benchmark methods and patient-preserving splits are in
[Strodthoff et al.](https://arxiv.org/abs/2004.13701) and the
[versioned official release](https://physionet.org/content/ptb-xl/1.0.3/).

## Verification

```bash
python scripts/build_plateau_audit.py
python -m pytest tests/test_plateau_audit.py -q
python scripts/verify_artifact_hashes.py
```

The first command recalculates all metrics from the saved complete
probabilities and authenticates ordered source references, unchanged
checkpoint receipts, executed input hashes and artifact bytes. It does
not train. For a new CEDIA diagnostic, reconstruct the seven signal input
NPZs using the preceding runbooks, compare their hashes, and run the archived
script in a Slurm GPU allocation. Signals and large raw tensor caches are
not distributed in Git. Research use only; no clinical decision is implied.
