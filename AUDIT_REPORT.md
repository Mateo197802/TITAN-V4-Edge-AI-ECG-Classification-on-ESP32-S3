# Repository Audit Report

Audit date: 2026-09-24
Repository: `Mateo197802/TITAN-V4-Edge-AI-ECG-Classification-on-ESP32-S3`
Audited baseline: `3a2ef71b35e0d9d05dca511120f976fc27f7fab9`
Audit changes are prepared on branch `codex/audit-fixes` against this baseline.

## Findings and Changes

- The default firmware profile exposed unauthenticated routes for raw ECG windows and model results. Those routes now default to disabled; only explicitly named private-hotspot profiles enable them. They still have no authentication and must remain on an isolated network.
- CEDIA SSH scripts accepted unknown host keys. They now load system/configured known hosts and reject unknown keys. No remote login or SLURM job was run.
- Personal workstation paths and account identifiers were present in checked-in summaries/scripts. They were removed from the current working tree. They remain in the repository's existing public Git history; that history was not rewritten.
- The artifact hash manifest failed on Windows line endings and contained stale entries. Text hashing is now LF-normalized, binary hashing remains byte-exact, and the checked-in manifest covers 114 repository files.
- A recorder test used a stale path, pytest had duplicate module-name collection, and the model-header generator did not search the tracked TFLite model location. These were corrected and covered by the test suite.
- The previous `LICENSE` text was not a recognized open-source license. The repository now uses the standard MIT license for original software source code only; `LICENSE_SCOPE.md` excludes datasets, labels, models, recordings, results, and evidence documents.
- The validation-label table contains 312 `data_test` rows without source/version/license evidence. The remaining source groups also lack per-record upstream mappings and redistribution clearance. These are documented in `DATA_PROVENANCE.md` and remain release blockers for redistributing row-level data.

## Verification

- `python -m pytest -q`: 35 passed on CPython 3.12.10.
- `python scripts/build_artifact_hashes.py`: manifest matches 114 files.
- `python scripts/verify_artifact_hashes.py`: passed, no failures.
- Primary-9, Pathology Primary-5, and combined aggregate scripts ran successfully. They reproduce packaged JSON summaries; they do not run inference or retrain the checkpoint.
- `python -m compileall -q cedia scripts src tests hardware/esp32`: passed.
- Firmware compilation, physical-device tests, signal-chain equivalence, and CEDIA connectivity were not verified.

## Security Scope

The source review found three baseline issues: two medium-severity findings (unauthenticated sensitive HTTP routes enabled by some profiles; permissive SSH host-key handling) and one low-severity finding (local account/path identifiers). Their current-tree changes and remaining boundaries are described above.

Tracked text in the complete three-commit Git history was checked for private-key PEM headers and common GitHub, AWS access-key-ID, Slack, Google API-key, and OpenAI token prefixes; no matches were found. The source scanner also found no actual password/private-key material in the checked-out source files. Binary blobs and unrecognized secret formats are not covered by that regex pass. This is not a guarantee against every secret format. The local path disclosure remains in old public commits, and this audit did not rewrite remote history or revoke credentials.

## Readiness

Code tests and packaged aggregate checks pass, and the original software source has an MIT license. This is not a fully reproducible scientific release or a rights-cleared public data release: resolve the `data_test` provenance and row-level data permissions, document exact training/evaluation source versions and manifests, and decide whether to remove historical personal-path disclosures through a coordinated history rewrite.
