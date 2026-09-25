# TITAN V4 Repository and Reproducibility Audit

Audit date: 2026-09-25

Repository: `Mateo197802/TITAN-V4-Edge-AI-ECG-Classification-on-ESP32-S3`
Working branch: `codex/audit-fixes`

## Corrections Made

- Implemented an end-to-end Primary-9 inference runner that loads the distributed checkpoint, downloads the exact 672 PhysioNet records, verifies source IDs and headers, applies documented signal preprocessing, and writes prediction-level probabilities, metrics, waveform hashes, and a run manifest.
- Replaced the former label table with a minimal 672-row table regenerated from official Challenge 2021 v1.0.3 WFDB headers. All 312 records formerly marked `data_test` now resolve to exact upstream record paths via source prefix plus official `RECORDS` indexes.
- Compared the rebuilt 672 project labels with CEDIA's external manifest in a read-only session: same record IDs and no table disagreements. This is agreement between project tables, not an independent clinical review. The previous labels differ on 129 rows.
- Reconciled internal validation, the historical 605/672 aggregate, the 466/672 single-label diagnostic, and the pathology summary as separate evidence. The current checkpoint yields 568/672 against the exact historical Primary-9 labels, not 605/672; absent historical predictions mean 605 remains unverified, not disproven. Pathology and Cascade/OOD remain unverified historical aggregates.
- Updated the 672-record inference report's status to a reproducible project-specific single-label diagnostic. It is not the official PhysioNet Challenge score or a verified independent external test result.
- Added versioned PhysioNet dataset attribution and separated software MIT terms from CC BY 4.0 research artifacts. Third-party data/models remain under their own rights and terms.
- Pinned the firmware's previously missing TensorFlow Lite dependency to an immutable upstream commit, removed incompatible/permissive compiler flags, fixed deletion of a non-owned static interpreter, and compiled the safe `esp32s3` profile against a project-local ESP32-S3-DevKitC-1-N8R8 definition with 8 MB octal PSRAM. The profile matches the 4 MB tensor-arena requirement; the physical board was not connected. Build warnings from bundled third-party LCD drivers are disclosed in `reports/evidence/firmware-build.md`.
- Preserved the earlier security fixes: sensitive HTTP routes are disabled by default, and CEDIA SSH rejects unknown host keys. Historical local path disclosures remain in public Git history; that history was not rewritten.

## Metric Reconciliation

The distributed checkpoint (`BC7BA03D0D6D40E823FDB9BB261EBE29AE8B7A74C97D1C98E7140BCA4D2D305B`) was run on all 672 records under a project-specific single-label rule:

| Metric | Result |
|---|---:|
| Correct | 466 / 672 |
| Accuracy | 0.693452380952 |
| Macro-F1 | 0.688840776554 |
| Weighted-F1 | 0.689589967237 |

Per-record predictions, confusion matrix, source waveform hashes, input hashes, environment, preprocessing, and inference-code hashes are stored under `outputs/reproduced/primary9/`.

This 69.35% result is computationally reproducible under that rule, but it does not establish the original 90.03% historical aggregate was false. The old per-record predictions and exact lineage are missing, and the two label tables differ on 129 rows. Applying the current predictions against the exact prior label table yields 568/672, not the historical 605/672.

CEDIA's separate internal report records 89.84% accuracy, 73.90% macro-F1 and 90.30% weighted-F1 on 58,855 windows using a different checkpoint. The distributed checkpoint's training summary records 89.48% best validation macro-F1 on 706 windows. Neither is a 672-record independent test. CEDIA also has a separate pathology-supervised checkpoint (loss weight 0.2, 1,811 labeled windows), but its hash differs and no Primary-5 evaluation outputs were found. The historical 90.26% per-label accuracy and 66.87% macro-F1 remain unverified. See [metric reconciliation](reports/evidence/metric-reconciliation.md) and the [CEDIA pathology cross-check](reports/evidence/cedia-pathology-crosscheck.json).

## CEDIA Cross-Check

CEDIA was accessed read-only. Its external record-ID set matches the local 672-row evaluation set, and label disagreement is zero after rebuilding from official source headers. CEDIA's train/validation manifests and overlap report belong to a separate pipeline. Its student checkpoint hashes do not match the checkpoint evaluated here, so its train/test zero-overlap finding is not used to claim independence for this repository model. A CEDIA teacher/boundary artifact matches the repository's separate protected baseline, not the evaluated Primary-9 checkpoint. No remote files were changed and no SLURM job or hardware run was launched. See [cross-check evidence](reports/evidence/cedia-readonly-crosscheck.md).

The CEDIA root validation report and checkpoint hashes were also read-only inspected and summarized without user paths in `reports/evidence/cedia-validation-summary.json`.

## Dataset and Citation Review

The exact inference source is PhysioNet/Computing in Cardiology Challenge 2021 v1.0.3 (DOI `10.13026/34va-7q14`), under CC BY 4.0 for its files. Challenge labels may be multi-label and the official evaluator uses a weighted Challenge metric; TITAN's current 672-row report reduces labels to one project-selected class and reports ordinary accuracy/F1. The original 2020-2021 database/Challenge papers remain appropriate dataset/method citations; they are supplemented by the exact version DOI and the current recommended PhysioNet citation. See `REFERENCES.md`.

The project-local `split=test` is not the official hidden Challenge test set. The exact training manifest and offline teacher sidecars for the distributed checkpoint are unavailable, so from-scratch training and checkpoint train/evaluation independence are not established. No source-held-out claim is made.

## Licensing and Security

Original software source is under MIT. Project-authored model weights, results, and derived label/prediction artifacts have a separate CC BY 4.0 notice. That grant applies only to project authors' rights and excludes third-party models and data; it is not an institutional or legal opinion. The downloaded ECG waveforms are not checked into Git. A scan of the working tree and Git history found no matches for common API-token/private-key formats, and the current working tree contains no local absolute user paths. Historical public Git commits still contain personal filesystem-path disclosures; no full-history rewrite or credential revocation was performed.

## Verification

Verification on 2026-09-25:

- Official release record index, source manifest, and label checks passed for all 672 records; WFDB labels were checked offline.
- Full offline inference regenerated all 672 prediction rows and reproduced the project-specific single-label diagnostic above; against the old labels it yields 568/672. This is not the official Challenge metric.
- The historical-label comparator and pathology evidence audit ran from their documented default commands. The former verifies checkpoint, label, source-manifest and prediction hashes; the latter confirms that the historical pathology claim has no scoreable record-level target/prediction pair.
- `python -m pytest -q`: 73 passed. `compileall` passed for project code, scripts, and tests.
- Artifact hash verifier passed for 153 repository files. The safe ESP32-S3 firmware compile passed; measured resources and binary hash are in `reports/evidence/firmware-build.md`.

Physical ESP32-S3 validation, firmware/offline signal equivalence, checkpoint training reconstruction, and source-held-out independence were not established.
