# Claim Boundaries

## Current Evidence

The repository reproducibly computes Primary-9 CPU inference on 672 records with the distributed checkpoint: 466/672 under a project-specific single-label Dx rule. Record-level probabilities and full metric outputs are published under `outputs/reproduced/primary9/`. This is a diagnostic computation, not the official PhysioNet Challenge metric or verified independent external performance.

High internal values also exist: CEDIA reports 89.84% accuracy and 90.30% weighted-F1 on 58,855 windows with a different checkpoint; the distributed checkpoint's training summary reports 89.48% best validation macro-F1 on 706 windows. These are internal model-selection/validation metrics and do not independently validate the 672-record claim. The historical 605/672 aggregate is unverified; current predictions score 568/672 against that old label table. A separate CEDIA pathology-supervised checkpoint was evaluated on patient-disjoint PTB-XL v1.0.3 records. At fixed threshold 0.65 it scores 79.90% mean per-label accuracy and 42.03% macro-F1 on 406 records; thresholds selected using 273 calibration records score 71.58% and 41.37%. The legacy `configured_macro_f1_reference: 0.65` does not identify the historical prediction threshold, so the fixed-0.65 value is a sensitivity check only. Neither result reproduces the historical 90.26%/66.87% aggregate, whose checkpoint, per-record labels/predictions, and threshold lineage are unavailable. The CEDIA result is a distinct post hoc evaluation in the same dataset family, not independent external validation. See [metric reconciliation](../evidence/metric-reconciliation.md) and [CEDIA pathology evidence](../evidence/cedia-pathology-crosscheck.json).

The 672-record cohort resolves to the PhysioNet Challenge 2021 v1.0.3 `training/` partition, not the official hidden Challenge test set. The exact checkpoint training manifest is unavailable; model/evaluation overlap and source-held-out independence are not established.

## Historical and Engineering Evidence

- The former 605/672 Primary-9 result used labels that disagree with rebuilt project labels on 129 records; the current checkpoint scores 568/672 on those old labels, but that does not recreate or disprove the historical run.
- The historical Pathology Primary-5 aggregate remains unverified; its source predictions, targets, and checkpoint/threshold lineage are missing. The distinct CEDIA checkpoint now has a checked per-record evaluation package, but it is not evidence for the old aggregate. Cascade/OOD remains an archived legacy aggregate without matching record-level prediction/target artifacts.
- The safe ESP32-S3 firmware profile compiles with the pinned toolchain. Existing recordings document engineering acquisition/serialization only; hardware-in-the-loop tests and numerical equivalence with offline preprocessing remain unestablished.

TITAN V4 is research software, not a cleared medical device. Outputs must not be used to diagnose, treat, or rule out a medical condition.
