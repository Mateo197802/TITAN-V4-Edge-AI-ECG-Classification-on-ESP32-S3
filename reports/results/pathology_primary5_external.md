# Pathology Primary-5: Unverified Historical Aggregate

The archived report lists 0.90256 per-label accuracy and 0.66874 macro-F1 for IMI, ALMI, ILMI, LAE, and ISC_. The reported accuracy is an average over binary-label accuracies, not ordinary record-level accuracy. Its per-record predictions, reference rows, checkpoint hash, and threshold provenance are unavailable; it is not an independently reproduced result.

The distributed repository checkpoint's included training summary records pathology loss weight 0.0 and zero pathology-labeled windows. Therefore, the historical aggregate cannot be attributed to that distributed checkpoint from the available evidence. Do not cite these values as verified performance for the current checkpoint. The stored summary is `outputs/gold_master_external_validation/pathology_primary5/pathology_primary5_external_summary.json`; see [metric reconciliation](../evidence/metric-reconciliation.md).
