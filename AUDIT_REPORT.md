# TITAN V4 Repository and Reproducibility Audit

Audit date: 2026-09-24

Repository: `Mateo197802/TITAN-V4-Edge-AI-ECG-Classification-on-ESP32-S3`
Working branch: `codex/audit-fixes`

## Corrections Made

- Implemented an end-to-end Primary-9 inference runner that loads the distributed checkpoint, downloads the exact 672 PhysioNet records, verifies source IDs and headers, applies documented signal preprocessing, and writes prediction-level probabilities, metrics, waveform hashes, and a run manifest.
- Replaced the former label table with a minimal 672-row table regenerated from official Challenge 2021 v1.0.3 WFDB headers. All 312 records formerly marked `data_test` now resolve to exact upstream record paths via source prefix plus official `RECORDS` indexes.
- Compared the rebuilt 672 record labels with CEDIA's external manifest in a read-only session: same record IDs and zero label disagreements. The earlier labels disagreed on 129 records; the previous 605/672 score is superseded.
- Updated current result documents and scripts to use recomputed Primary-9 inference. Pathology Primary-5 and Cascade/OOD summaries remain preserved but are explicitly marked legacy and are excluded from current results.
- Added versioned PhysioNet dataset attribution and separated software MIT terms from CC BY 4.0 research artifacts. Third-party data/models remain under their own rights and terms.
- Pinned the firmware's previously missing TensorFlow Lite dependency to an immutable upstream commit, removed incompatible/permissive compiler flags, fixed deletion of a non-owned static interpreter, and compiled the safe `esp32s3` profile against a project-local ESP32-S3-DevKitC-1-N8R8 definition with 8 MB octal PSRAM. The profile matches the 4 MB tensor-arena requirement; the physical board was not connected. Build warnings from bundled third-party LCD drivers are disclosed in `reports/evidence/firmware-build.md`.
- Preserved the earlier security fixes: sensitive HTTP routes are disabled by default, and CEDIA SSH rejects unknown host keys. Historical local path disclosures remain in public Git history; that history was not rewritten.

## Recomputed Primary-9 Result

The distributed checkpoint (`BC7BA03D0D6D40E823FDB9BB261EBE29AE8B7A74C97D1C98E7140BCA4D2D305B`) was evaluated on all 672 records:

| Metric | Result |
|---|---:|
| Correct | 466 / 672 |
| Accuracy | 0.693452380952 |
| Macro-F1 | 0.688840776554 |
| Weighted-F1 | 0.689589967237 |

Per-record predictions, confusion matrix, source waveform hashes, input hashes, environment, preprocessing, and inference-code hashes are stored under `outputs/reproduced/primary9/`.

## CEDIA Cross-Check

CEDIA was accessed read-only. Its external record-ID set matches the local 672-row evaluation set, and label disagreement is zero after rebuilding from official source headers. CEDIA's train/validation manifests and overlap report belong to a separate pipeline. Its student checkpoint hashes do not match the checkpoint evaluated here, so its train/test zero-overlap finding is not used to claim independence for this repository model. A CEDIA teacher/boundary artifact matches the repository's separate protected baseline, not the evaluated Primary-9 checkpoint. No remote files were changed and no SLURM job or hardware run was launched. See [cross-check evidence](reports/evidence/cedia-readonly-crosscheck.md).

## Dataset and Citation Review

The exact inference source is PhysioNet/Computing in Cardiology Challenge 2021 v1.0.3 (DOI `10.13026/34va-7q14`), under CC BY 4.0 for its files. Its training package contains the resolved source families recorded in the manifest. The original 2020-2021 database/Challenge papers remain appropriate dataset/method citations; they are supplemented by the exact version DOI and the current recommended PhysioNet citation. See `REFERENCES.md`.

The project-local `split=test` is not the official hidden Challenge test set. The exact training manifest and offline teacher sidecars for the distributed checkpoint are unavailable, so from-scratch training and checkpoint train/evaluation independence are not established. No source-held-out claim is made.

## Licensing and Security

Original software source is under MIT. Project-authored model weights, results, and derived label/prediction artifacts have a separate CC BY 4.0 notice. That grant applies only to project authors' rights and excludes third-party models and data; it is not an institutional or legal opinion. The downloaded ECG waveforms are not checked into Git. A scan of the working tree and Git history found no matches for common API-token/private-key formats, and the current working tree contains no local absolute user paths. Historical public Git commits still contain personal filesystem-path disclosures; no full-history rewrite or credential revocation was performed.

## Verification

Verification on 2026-09-24:

- Official release record index, source manifest, and label checks passed for all 672 records; WFDB labels were checked offline.
- Full offline inference regenerated all 672 prediction rows and reproduced the metrics above.
- `python -m pytest -q`: 64 passed. `compileall` passed for CEDIA scripts, project code, tests, and hardware tooling.
- Artifact hash verifier passed for 140 repository files. The safe ESP32-S3 firmware compile passed; measured resources and binary hash are in `reports/evidence/firmware-build.md`.

Physical ESP32-S3 validation, firmware/offline signal equivalence, checkpoint training reconstruction, and source-held-out independence were not established.
