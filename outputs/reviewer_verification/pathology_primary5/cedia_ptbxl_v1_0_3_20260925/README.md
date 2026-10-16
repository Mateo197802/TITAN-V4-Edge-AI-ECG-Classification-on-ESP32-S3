# CEDIA Pathology Primary-5 Evaluation

This package contains a separate CPU evaluation of the CEDIA `TitanV4Max` pathology checkpoint on a patient-disjoint PTB-XL v1.0.3 cohort. It does not reproduce the historical 90.256% mean per-label accuracy / 66.874% macro-F1 aggregate; the historical checkpoint, per-record targets/predictions, and threshold lineage were not available.

## Results

On the 406-record test cohort, this checkpoint obtains 79.901% mean per-label accuracy and 42.027% macro-F1 at a fixed 0.65 sensitivity threshold. Thresholds selected on a separate 273-record calibration cohort obtain 71.576% and 41.370%; fixed-threshold 0.50 sensitivity results are 73.153% and 41.763%. The legacy JSON's `configured_macro_f1_reference: 0.65` does not establish that 0.65 was the threshold applied historically. These fixed-threshold checks were not optimized against test outcomes.

The test cohort contains 406 records from 387 patients; calibration contains 273 records from 271 patients. Two unreadable calibration signals are recorded in the input manifest and excluded before the final split. Test patients are excluded from both checkpoint train and validation sets. Seven PTB-XL record IDs overlapped between the checkpoint train and validation manifests; their patients were excluded from calibration and test.

## Files

- `pathology_primary5_scores.csv`: per-record true labels and unthresholded scores for the calibration and test cohorts. This is the input for metric re-aggregation.
- `pathology_primary5_predictions.csv`: predictions and thresholds used by the calibrated protocol.
- `pathology_primary5_per_class.csv` and `pathology_primary5_report.json`: aggregate and per-class metrics, cohort details, checkpoint identity, and runtime.
- `pathology_input_records.csv`: record inclusion status and SHA-256 hashes of WFDB header and signal inputs. It contains no patient identifiers or local filesystem paths.
- `reportable_pathology_selection.json`: reportable class and threshold-selection details.
- `SHA256SUMS.csv`: SHA-256 and byte size for every other file in this package.

## Re-aggregate the checked-in scores

From the repository root, with the documented Python dependencies installed:

```bash
python src/titan_v4/evaluation/pathology_primary5_protocol.py \
  --predictions_csv outputs/reviewer_verification/pathology_primary5/cedia_ptbxl_v1_0_3_20260925/pathology_primary5_scores.csv \
  --out_dir /tmp/titan-v4-pathology-recomputed \
  --fixed-threshold 0.50 \
  --fixed-threshold 0.65
```

The resulting calibrated and fixed-threshold metrics should match `pathology_primary5_report.json` within floating-point tolerance. This verifies metric aggregation from saved scores; it is not a standalone rerun of model inference.

## Provenance and scope

The PTB-XL metadata and SCP statements were pinned to PhysioNet v1.0.3 by SHA-256, and all 2,580 pathology vectors in the CEDIA sidecar were reconstructed exactly from official `scp_codes`. Record aliases from the PhysioNet Challenge split were resolved before patient-overlap exclusion. See the repository's [evaluation report](../../../../reports/results/pathology_primary5_external.md), [reproducibility guide](../../../../REPRODUCIBILITY.md), and [data provenance](../../../../DATA_PROVENANCE.md).

PTB-XL is available from [PhysioNet v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) under its stated CC BY 4.0 terms; cite the [original PTB-XL paper](https://doi.org/10.1038/s41597-020-0495-6) and dataset DOI `10.13026/kfzx-aw45`. This repository does not redistribute PTB-XL waveforms, the CEDIA checkpoint, or CEDIA source files. Full inference requires those exact inputs and their upstream access rights. The CEDIA checkpoint is SHA-256 `e9a44e4eea8ebb8f89d5e32909ae4afcc96442d1fa8eacb1e57a73bfa498353b`; the training summary is SHA-256 `01629aca2287d2ae58c71f51e9e80a19a3fa77c48438fe93d59995d10b8ab81d`. See `pathology_primary5_report.json` for the remaining code, data, and environment hashes.

The repository's CC BY 4.0 artifact notice applies only to rights held by the project authors; it does not grant rights in the CEDIA checkpoint or upstream data. Confirm the applicable author/institution permissions before republishing checkpoint-derived artifacts or making them part of a journal supplement.
