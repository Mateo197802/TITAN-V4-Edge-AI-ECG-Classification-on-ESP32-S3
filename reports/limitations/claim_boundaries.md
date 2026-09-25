# Claim Boundaries

## Current Result

The only current reproduced performance result is Primary-9 CPU inference on 672 records with the distributed checkpoint: 466/672 correct, accuracy 0.693452, macro-F1 0.688841, weighted-F1 0.689590. Record-level probabilities and full metric outputs are published under `outputs/reproduced/primary9/`.

The cohort is resolved to the PhysioNet Challenge 2021 v1.0.3 `training/` partition, not the official hidden Challenge test set. The exact checkpoint training manifest is unavailable; model/evaluation overlap and source-held-out independence are not established. Do not describe this as independent external test performance.

## Historical and Engineering Evidence

- The former 605/672 Primary-9 result is superseded because its bundled labels disagreed with source-header labels for 129 records.
- Pathology Primary-5 and Cascade/OOD remain archived legacy aggregates and were not recomputed from record-level predictions in this workflow.
- The safe ESP32-S3 firmware profile compiles with the pinned toolchain. Existing recordings document engineering acquisition/serialization only; hardware-in-the-loop tests and numerical equivalence with offline preprocessing remain unestablished.

TITAN V4 is research software, not a cleared medical device. Outputs must not be used to diagnose, treat, or rule out a medical condition.
