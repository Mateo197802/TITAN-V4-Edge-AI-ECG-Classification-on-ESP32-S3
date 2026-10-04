# External Validation Inputs

This directory contains only a frozen 672-record index, a versioned per-record source manifest, and a minimal label table. It contains no ECG waveform files. Downloaded `.hea`/signal files are stored under ignored `data/cache/`.

The source release is PhysioNet/Computing in Cardiology Challenge 2021 v1.0.3, DOI `10.13026/34va-7q14`, licensed CC BY 4.0 for its files. Each manifest path is verified against the release's official `RECORDS` index. `split=test` is a project-local label; all resolved files are in the release's `training/` directory. The set is not the Challenge hidden test set.

The upstream WFDB headers may contain multiple diagnoses. The checked-in `rhythm_label` is one project-selected Primary-9 label reconstructed from the ordered `Dx` codes; it is not the full multi-label target or the official Challenge scoring protocol. The reproducible 672-record output is a project-specific diagnostic, not an independent external test. See [DATA_PROVENANCE.md](../../DATA_PROVENANCE.md), [metric reconciliation](../../reports/evidence/metric-reconciliation.md), and [REFERENCES.md](../../REFERENCES.md) for the label rule, evidence boundary, attribution, and license scope. The label table and project-authored derived artifacts are covered by the separate [CC BY 4.0 notice](../../LICENSE-ARTIFACTS.md), in addition to required upstream attribution.
