# Distributed Checkpoints

`gold_master_primary9_model.pth` is the checkpoint used by the recomputed Primary-9 evaluation. Its SHA-256 is `BC7BA03D0D6D40E823FDB9BB261EBE29AE8B7A74C97D1C98E7140BCA4D2D305B` (4,636,444 bytes). `protected_baseline_primary9_model.pth` is a separate protected artifact and is not the checkpoint evaluated in the current result.

The training summary is included as historical metadata, but the exact training/validation record manifests and offline teacher-sidecar arrays are not included. It therefore does not enable a from-scratch reconstruction of the checkpoint or establish its overlap with the 672-record evaluation set. See [REPRODUCIBILITY.md](../../REPRODUCIBILITY.md).

Project-authored checkpoint weights are covered by the separate [CC BY 4.0 artifact notice](../../LICENSE-ARTIFACTS.md), limited to rights held by the project authors and excluding any third-party rights. The root MIT software license does not cover model weights.
