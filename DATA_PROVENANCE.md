# Data Provenance and License

## Evaluation Records

The project evaluation set contains 672 unique ECG record IDs. Every record has been resolved to a specific record path in the `training/` partition of PhysioNet/Computing in Cardiology Challenge 2021, release 1.0.3 (DOI `10.13026/34va-7q14`). The release describes its constituent data sources and lists CC BY 4.0 for its files. The project does not distribute the ECG waveform files; the inference runner downloads only the selected records from the versioned release.

| Resolved Challenge source | Records |
|---|---:|
| CPSC 2018 | 56 |
| CPSC 2018 Extra | 42 |
| Georgia 12-lead ECG | 56 |
| PTB-XL | 378 |
| Chapman-Shaoxing | 70 |
| Ningbo | 70 |
| **Total** | **672** |

`record_source_manifest.csv` contains each ID, original project source field, resolved source family, release/version, upstream partition, DOI, license, exact `training/<dataset>/<group>/<record>` path, and resolution method. For the 312 rows originally grouped under the generic `data_test` source label, the resolver uses the record-ID prefix to select a candidate family and then requires an exact match in that family's official PhysioNet `RECORDS` index. This yields zero unresolved records; IDs are never resolved by guessing a group number.

`validation_record_index.csv` preserves the frozen set/order of evaluation IDs. Its `split=test` value is a project-local split label only. All resolved files are in PhysioNet's public `training/` partition; this is not the official hidden Challenge test set. The exact training manifest for the distributed checkpoint is unavailable, so overlap between checkpoint training and this evaluation set has not been established.

## Label Construction

`final_external_validation_labels.csv` is a minimal table containing the record ID, original project source label, local split label, one Primary-9 rhythm label, WFDB header `Dx` codes, and mapped rhythm names. Age, sex, waveform samples, local filesystem paths, and direct identifiers are excluded.

The deterministic label rule follows the ordered SNOMED `Dx` codes in the WFDB header: choose the first non-NSR Primary-9 rhythm. If no non-normal Primary-9 rhythm is present, use 1AVB for prolonged-PR code `164947007`; otherwise use NSR when present. The label builder fails when no Primary-9 class can be derived. This rule reproduces all 672 rows and was independently cross-checked against the read-only CEDIA external manifest with zero label disagreements.

Run `python scripts/build_validation_record_index.py --check`, `python scripts/build_record_source_manifest.py --check`, and `python scripts/build_primary9_evaluation_labels.py --check` to verify the index, official source mappings, and labels. The last command downloads only WFDB headers unless `--offline` is specified. The inference run separately records SHA-256 hashes and byte sizes for the waveform/header files it actually reads.

## Attribution and Redistribution

The evaluation labels and the source-derived validation artifacts are provided under CC BY 4.0 with attribution to the PhysioNet Challenge 2021 versioned release and its cited source papers. See [LICENSE-ARTIFACTS.md](LICENSE-ARTIFACTS.md) and [REFERENCES.md](REFERENCES.md). The upstream dataset retains its own license and attribution requirements. This notice does not relicense any upstream ECG waveform, dataset, publication, or third-party asset.

The repository contains no downloaded PhysioNet waveform files. `data/cache/` is ignored by Git. The checked-in row-level labels contain public record IDs and derived diagnosis labels; do not interpret this as a privacy clearance for any additional data.

## Checkpoint Training Provenance

The checked-in training summary reports 3,785 train windows, 706 validation windows, and three epochs, but its exact training/validation record manifests and offline teacher-sidecar arrays are not included. CEDIA's `RHYTHM_PRIMARY9_DISTILL` manifests corroborate a separate 1,918/338-record train/validation split and 672 external IDs. The associated May 21 student summary reports 3,042/543 windows and 32 epochs, and its student checkpoint hashes do not match the Primary-9 checkpoint distributed here. CEDIA's zero-overlap result therefore cannot be attributed to this checkpoint. The distributed checkpoint's training lineage and train/evaluation overlap remain unverified; no source-held-out claim is made. See the [read-only CEDIA comparison](reports/evidence/cedia-readonly-crosscheck.md).
