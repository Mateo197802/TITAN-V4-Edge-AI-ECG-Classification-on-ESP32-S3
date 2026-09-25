# CEDIA Read-Only Cross-Check

Audit date: 2026-09-25. The TITAN artifacts under the Mateo Gavilanes CEDIA directory were inspected read-only. No remote files were changed and no SLURM job was submitted.

## Label/ID Comparison

| Check | Result |
|---|---:|
| Local evaluation records | 672 |
| CEDIA external-manifest records | 672 |
| Record-ID mismatches | 0 |
| Current source-header label mismatches | 0 |
| Previous bundled labels that disagreed with the current/CEDIA labels | 129 |

The CEDIA external manifest reviewed was `DATA/DATASETS_CURADOS/RHYTHM_PRIMARY9_DISTILL/manifest_external_test.csv`, SHA-256 `8E0855C7254BB024DD2D3EEAD1AD569AE33D7FB1588545FD22F3B1E463928E60`. The CEDIA train/validation manifests have 1,918/338 record IDs and are not redistributed here. The current label table can independently be rebuilt from the official PhysioNet WFDB headers using `scripts/build_primary9_evaluation_labels.py`.

## Checkpoint and Split Boundary

CEDIA's train and validation manifests contain 1,918 and 338 record IDs, respectively, and its overlap report records zero overlap with its 672-record external set. This is useful corroboration about CEDIA's own run, but it is not proof of the training split for the checkpoint in this repository.

The distributed checkpoint hash is `BC7BA03D0D6D40E823FDB9BB261EBE29AE8B7A74C97D1C98E7140BCA4D2D305B`. CEDIA's student-best (`0C008B2231757D4865A23324FEA4FFEB05FAABCAD0AC69ACA6CB7FBEB5BC7CC4`) and student-final (`D3C37613F9A91C0534F9B2DB1A59F473EDB3F22F3C66BE9283BF00BED1B703C0`) hashes do not match it. The CEDIA teacher/boundary checkpoint hash matches this repository's separate protected baseline (`BBA87435C514EDCCB043C7485037753823C5FD19A816A16872B7CD9430BDD1EB`), not the evaluated model. Therefore, CEDIA's zero-overlap finding is not transferred to the evaluated checkpoint.

A read-only inventory of `~/V4_CEDIA` found no file with the evaluated checkpoint's exact byte size (4,636,444 bytes). The directory contains additional archived checkpoints and training summaries, but they belong to other runs and do not establish lineage for this checkpoint. A differently serialized file could contain equivalent tensors, but no evidence of that equivalence or a matching checkpoint hash was found.

## Training-Run Comparison

The repository's distributed-checkpoint summary reports 3,785 train windows, 706 validation windows, three epochs, and best epoch 2. The corresponding exact training and validation record manifests and offline teacher-sidecar arrays are not present in this repository; their original paths were removed from the summary.

CEDIA's `03_OUTPUTS/20260521_PRIMARY9_DISTILLED_STUDENT_V1/training_summary.json` (SHA-256 `3B32526A45D1FE8DBAE1F63CCE78636622F900726F4AB25516FA1C9BA6D3BC14`) reports 3,042 train windows, 543 validation windows, 32 epochs, and best epoch 22. Its student-best/student-final weights have the hashes above. These counts, run settings, and checkpoint hashes are different from this repository's distributed checkpoint. The CEDIA training files therefore do not fill the missing lineage for this checkpoint; they must not be substituted as if they were the same experiment.

## CEDIA Internal Validation Metric

The separate CEDIA root run at `03_OUTPUTS/validation_report.json` reports 89.836% accuracy, 73.898% macro-F1, and 90.301% weighted-F1 over 58,855 validation windows. Its `training_summary.json` reports 455,756 training windows, 33 epochs, and best epoch 25. The checkpoint SHA-256 is `e9a44e4eea8ebb8f89d5e32909ae4afcc96442d1fa8eacb1e57a73bfa498353b`; it is a different file from the distributed repository checkpoint (SHA-256 `BC7BA03D0D6D40E823FDB9BB261EBE29AE8B7A74C97D1C98E7140BCA4D2D305B`). The remote report is an internal window-level aggregate, not a prediction-level reproduction of the public 672-record checkpoint. Sanitized values and source hashes are preserved in [cedia-validation-summary.json](cedia-validation-summary.json).

The repository checkpoint's own training summary records 3,785 train windows, 706 validation windows, and best validation macro-F1 0.89478. This is a checkpoint-selection metric; prediction-level validation artifacts and the exact record manifests are not included. These internal validation figures are separate from both the 672-record diagnostic and the unverified historical 605/672 aggregate. The label match with CEDIA confirms agreement between two project label tables, not an independent clinical review.
