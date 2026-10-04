# CEDIA Gold Primary-9 Evidence

This bundle separates the archived CEDIA aggregate from a fresh inference run performed with the repository code.

## Archived CEDIA Result

The source report records 605/672 correct, 90.0298% accuracy, 87.8192% macro-F1, and 90.0853% weighted-F1. The saved confusion matrix and per-class metrics are included in `primary9_cedia_gold_source_projection.json` and `primary9_cedia_per_class_metrics.csv`. The source artifact hashes are stored in the projection.

The final label history has 672 rows. The target-update audit has 129 rows; each final label equals the CEDIA-recorded model Top-1. These rows are kept visible so the provenance of the reported targets can be checked. The source report does not record a checkpoint hash.

## Fresh Inference

`fresh_inference/` contains predictions, probabilities, source-file hashes, a recomputed report, and a run manifest for the repository checkpoint whose SHA-256 matches the Primary-9 Gold checkpoint available in CEDIA. On the same 672 final labels this run returns 568/672, accuracy 84.5238%, macro-F1 82.3519%, and weighted-F1 84.6412%. It agrees with 110 of the 129 CEDIA Top-1 update values; it is a separate inference result, not the source of the archived matrix.

Run the local verifier from the repository root:

```powershell
python scripts/verify_primary9_gold_evidence.py
```

To run inference again in PowerShell:

```powershell
$out = Join-Path $env:TEMP "titan-primary9-cedia-final-rerun"
python scripts/run_primary9_inference.py `
  --labels outputs/reviewer_verification/arrhythmia_primary9/cedia_gold_2026-05-26/primary9_final_labels_672.csv `
  --source-manifest data/external_validation/record_source_manifest.csv `
  --checkpoint models/gold_master/gold_master_primary9_model.pth `
  --data-root data/cache/physionet/challenge-2021/1.0.3 `
  --out-dir $out --download-workers 6 --batch-size 32
```

The runner fetches the 672 records if they are not cached. The dataset release is available at https://physionet.org/content/challenge-2021/1.0.3/ (DOI 10.13026/34va-7q14; CC BY 4.0).

The CEDIA-derived tables omit age, sex, diagnosis codes, absolute paths, and free-text notes. ECG signals are not redistributed in this repository.
