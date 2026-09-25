# External Validation Inputs

This directory contains only a frozen 672-record index, a versioned per-record source manifest, and a minimal label table. It contains no ECG waveform files. Downloaded `.hea`/signal files are stored under ignored `data/cache/`.

The source release is PhysioNet/Computing in Cardiology Challenge 2021 v1.0.3, DOI `10.13026/34va-7q14`, licensed CC BY 4.0 for its files. Each manifest path is verified against the release's official `RECORDS` index. `split=test` is a project-local label; all resolved files are in the release's `training/` directory. The set is not the Challenge hidden test set.

The labels are reconstructed deterministically from WFDB `Dx` codes and checked against all source headers. See [DATA_PROVENANCE.md](../../DATA_PROVENANCE.md) and [REFERENCES.md](../../REFERENCES.md) for the attribution, label rule, and license scope. The label table and project-authored derived artifacts are covered by the separate [CC BY 4.0 notice](../../LICENSE-ARTIFACTS.md), in addition to required upstream attribution.
