# Data Provenance and Redistribution

This file separates what can be read from the public checkout from what still needs source-owner evidence. The repository contains a record-level label/metadata table and aggregate summaries; it does not contain source ECG waveforms.

## Included Validation Labels

`data/external_validation/final_external_validation_labels.csv` contains 672 rows, all marked `split=test`. The current `source` column groups are:

| Source value | Rows |
|---|---:|
| `data_test` | 312 |
| `training/chapman_shaoxing` | 70 |
| `training/cpsc_2018` | 41 |
| `training/cpsc_2018_extra` | 42 |
| `training/georgia` | 49 |
| `training/ningbo` | 70 |
| `training/ptb-xl` | 88 |
| **Total** | **672** |

These are literal source labels in this repository, not independently verified upstream release identifiers. In particular, the local `split=test` field does not prove that records are independent of the datasets used to train the checkpoint. Source labels beginning `training/` make that distinction especially important.

The CSV includes record identifiers, rhythm/pathology labels, age/sex fields, and diagnosis-related fields. No direct names or contact fields were found in the checked-out CSV, but that is not a privacy or redistribution clearance. The `data_test` subset has no accompanying source manifest, dataset citation, permission record, or version identifier. Its provenance and redistribution rights remain unresolved.

## Dataset References and Limits

The `training/` source names correspond to public ECG dataset families described in [REFERENCES.md](REFERENCES.md): Chapman-Shaoxing, CPSC 2018/CPSC-Extra, Georgia, Ningbo, and PTB-XL. Their presence in the CSV does not establish which exact upstream release, files, or license terms produced each row. The repository does not record versioned source archives or per-record upstream checksums.

The current PhysioNet dataset pages identify newer releases (including PTB-XL 1.0.3 and Challenge 2021 1.0.3), but this repository does not establish that those are the versions used to train or evaluate this checkpoint. Do not silently substitute a current download and call the old scores reproduced.

## Required Before an Unqualified Public Data Release

1. Identify the source and authorized license for all 312 `data_test` records.
2. Record the upstream dataset name, release/version, source record ID mapping, license, and redistribution basis for every label-table row.
3. Confirm permission and privacy review for redistribution of row-level health labels and age/sex metadata.
4. Provide the frozen training and validation manifests or a reproducible method to reconstruct them, then run overlap checks using record IDs, normalized path stems, and waveform hashes.

Until those checks are complete, do not claim the table is fully cleared for redistribution, that validation is source-held-out, or that the reported metrics are independently reproducible. The MIT license covers only original software source code; it does not grant rights to third-party data.

## Repository License Status

The previous one-line `LICENSE` text was not a recognized open-source license. The current MIT license is scoped to original software source code as described in `LICENSE_SCOPE.md`; it does not cover datasets, labels, model weights, recordings, results, or evidence files. Third-party datasets and recordings remain subject to their own terms. The `data_test` rows have no established source/license basis in this checkout and are not granted a redistribution license here.
