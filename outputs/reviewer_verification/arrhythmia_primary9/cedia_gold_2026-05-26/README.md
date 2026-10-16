# CEDIA Gold Primary-9 Evidence

This bundle separates the archived CEDIA aggregate from a fresh inference run performed with the repository code.

## Archived CEDIA Result

The source report records two references for the same 672 records. Against the original labels it reports 476/672 correct, 70.8333% accuracy, 70.6844% macro-F1, and 70.6975% weighted-F1. Against the final updated labels it reports 605/672 correct, 90.0298% accuracy, 87.8192% macro-F1, and 90.0853% weighted-F1. Only the final-reference confusion matrix is saved in the report and can be recalculated from that matrix.

The pre-update original labels are from `DATASETS_CURADOS/RHYTHM_PRIMARY_V2/manifest_external_test.csv` (SHA-256 `8190EC19F4887D4618213CBF95B9DA3B68948DDDAB4F0233D975EBA5938E5B8C`); all 672 record IDs and labels were cross-checked against `original_rhythm_label` in `primary9_label_history_672.csv`. The raw source manifest is not redistributed because it contains demographic fields and absolute paths. The target-update audit has 129 rows and records CEDIA Top-1 separately from original and final labels. Each final value equals its recorded Top-1, but that value match does not establish how the update was decided. The source report does not record a checkpoint hash or historical row-level predictions.

## Fresh Inference

`fresh_inference/` contains predictions, probabilities, source-file hashes, a recomputed report, and a run manifest for the repository checkpoint whose SHA-256 matches the Primary-9 Gold checkpoint available in CEDIA. The same predictions return 466/672 against original labels and 568/672 against final labels; on final labels this is 84.5238% accuracy, 82.3519% macro-F1, and 84.6412% weighted-F1. It agrees with 110 of the 129 CEDIA Top-1 update values; it is a separate inference result, not the source of the archived matrix.

Run the local verifier from the repository root:

```powershell
python scripts/verify_primary9_gold_evidence.py
python scripts/recompute_primary9_paired_reference.py
```

The paired-reference script writes the local run's two confusion matrices, per-class CSV, input hashes, and `SHA256SUMS.csv` under `outputs/reviewer_verification/arrhythmia_primary9/paired_reference_recomputation_2026-10-03/`.

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
