# Claim Boundaries

## Current Evidence

The repository reproducibly computes Primary-9 CPU inference on 672 records with the distributed checkpoint: 466/672 under a project-specific single-label Dx rule. Record-level probabilities and full metric outputs are published under `outputs/reproduced/primary9/`. This is a diagnostic computation, not the official PhysioNet Challenge metric or verified independent external performance.

High internal values also exist: CEDIA reports 89.84% accuracy and 90.30% weighted-F1 on 58,855 windows with a different checkpoint; the distributed checkpoint's training summary reports 89.48% best validation macro-F1 on 706 windows. These are internal model-selection/validation metrics and have no record-level prediction artifact in the repository. The historical 605/672 aggregate is unverified and not comparable; the pathology aggregate is also unverified. See [metric reconciliation](../evidence/metric-reconciliation.md).

The 672-record cohort resolves to the PhysioNet Challenge 2021 v1.0.3 `training/` partition, not the official hidden Challenge test set. The exact checkpoint training manifest is unavailable; model/evaluation overlap and source-held-out independence are not established.

## Historical and Engineering Evidence

- The former 605/672 Primary-9 result used labels that disagree with rebuilt project labels on 129 records; without prediction-level evidence it is neither validated nor disproven by the new diagnostic.
- Pathology Primary-5 and Cascade/OOD remain archived legacy aggregates and were not recomputed from record-level predictions in this workflow.
- The safe ESP32-S3 firmware profile compiles with the pinned toolchain. Existing recordings document engineering acquisition/serialization only; hardware-in-the-loop tests and numerical equivalence with offline preprocessing remain unestablished.

TITAN V4 is research software, not a cleared medical device. Outputs must not be used to diagnose, treat, or rule out a medical condition.
