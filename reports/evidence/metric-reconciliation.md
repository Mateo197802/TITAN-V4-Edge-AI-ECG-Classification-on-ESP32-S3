# Metric Reconciliation

Audit update: 2026-09-25. The figures around 90% refer to different artifacts and evaluation units. They must not be combined or substituted for one another.

## Results and Their Evidence

| Artifact | Evaluation unit and checkpoint | Reported metrics | Evidence status |
|---|---|---|---|
| CEDIA internal validation | 58,855 windows; CEDIA checkpoint SHA-256 `e9a44e4eea8ebb8f89d5e32909ae4afcc96442d1fa8eacb1e57a73bfa498353b` | Accuracy 89.84%; macro-F1 73.90%; weighted-F1 90.30% | Aggregate report and training summary inspected read-only. This checkpoint differs from the distributed repository checkpoint; these are not 672-record results. See [captured summary](cedia-validation-summary.json). |
| Distributed-checkpoint selection summary | 706 validation windows; repository checkpoint SHA-256 `BC7BA03D0D6D40E823FDB9BB261EBE29AE8B7A74C97D1C98E7140BCA4D2D305B` | Best validation macro-F1 89.48% | Training-summary metadata only. The record manifest and per-window predictions are unavailable; this is not an independently reproduced validation result. |
| Historical Primary-9 aggregate | 672 rows; old label table | Accuracy 90.03%; macro-F1 87.82%; weighted-F1 90.09% | Stored aggregate only. Its labels differ from the rebuilt/CEDIA manifest on 129 records; per-record predictions and exact evaluation lineage are absent. The score is unverified and not comparable to the diagnostic below. It has not been proven false. |
| Recomputed Primary-9 diagnostic | 672 ECG records; distributed checkpoint SHA-256 `BC7BA03D0D6D40E823FDB9BB261EBE29AE8B7A74C97D1C98E7140BCA4D2D305B` | Accuracy 69.35%; macro-F1 68.88%; weighted-F1 68.96% | All signals, inputs, source files, per-record probabilities, code and environment are hashed. Reproducible as the documented project-specific single-label diagnostic, not as the official Challenge metric or independent external validation. |
| Historical Pathology Primary-5 aggregate | Five binary labels; original prediction/label rows unavailable | Reported per-label accuracy 90.26%; macro-F1 66.87% | Unverified aggregate. The accuracy is not ordinary record-level accuracy. The source checkpoint and threshold provenance are not available. The distributed checkpoint's training summary records pathology loss weight 0 and zero pathology-labeled windows. |

## Interpretation

The 89.84% CEDIA accuracy and 90.30% weighted-F1 substantiate the user's recollection of high internal validation figures, but they describe a different checkpoint and 58,855 windows. The repository training summary also records an 89.48% best validation macro-F1 on 706 windows, but lacks prediction-level evidence. Neither figure is a score on the frozen 672-record cohort.

The 466/672 result was not a correction of the earlier 605/672 experiment under an otherwise identical protocol. It used rebuilt single-label targets and a newly documented deterministic inference path. Applying its stored predictions to the exact old label table gives 568/672, not 605/672. That comparison shows the old aggregate cannot be regenerated from the current prediction artifact; without the original predictions, it does not establish why the old score differs.

PhysioNet Challenge 2021 describes recordings with one or more labels and supplies its own weighted Challenge metric. TITAN's 672-record report collapses each header's diagnoses to one class and computes ordinary accuracy/F1. Therefore, its values are project-specific diagnostics, not official Challenge scores. The cohort resolves to the Challenge's public `training/` partition, and checkpoint overlap is unknown. See the [official Challenge release](https://physionet.org/content/challenge-2021/1.0.3/) and [official evaluation code](https://github.com/physionetchallenges/evaluation-2021).

## Publication Boundary

Report the internal window results, 672-record diagnostic and pathology aggregate separately, with their checkpoint, sample unit and evidence status. Do not state that the model has independently verified 90% external accuracy, that the 69% result supersedes the 90% result, or that the historical pathology aggregate is reproduced. A publication-grade external estimate still requires a checkpoint with documented training records, a prespecified independent cohort, prediction-level output and a label/metric protocol matched to the claimed task.
