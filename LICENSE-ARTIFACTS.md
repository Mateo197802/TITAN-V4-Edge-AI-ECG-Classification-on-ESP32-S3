# CC BY 4.0 for Project Research Artifacts

Except for material identified as third-party or otherwise excluded below, the project-authored research artifacts in this repository are licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0): https://creativecommons.org/licenses/by/4.0/.

This notice covers the project-authored model checkpoints in `models/gold_master/`, the CEDIA research checkpoints and derived feature/probability sidecars in `outputs/reviewer_verification/partial_backbone_refinement_2026-10-12/`, derived model parameters in `hardware/esp32/firmware/src/model_data.h`, and project-authored reports, tables, validation labels, predictions, and manifests in `data/external_validation/`, `outputs/`, and `reports/`. Source-code files in these directories retain the repository's MIT source-code license.

The refinement bundle also derives labels and feature/probability arrays from PTB-XL v1.0.3 (DOI `10.13026/kfzx-aw45`) and ECG-arrhythmia v1.0.0 (DOI `10.13026/wgex-er52`). Retain their upstream attribution listed in `REFERENCES.md` and the per-record source manifests. These arrays contain model features and probabilities, not ECG waveforms.

Attribution: Isaac Mateo Gavilanes Chavez, NeuroGuardian TITAN V4 ECG, repository `https://github.com/Mateo197802/TITAN-V4-Edge-AI-ECG-Classification-on-ESP32-S3`.

The same CC BY 4.0 research-artifact notice covers the project-authored exploratory checkpoint and evidence in `outputs/reviewer_verification/gold_target_gap_2026-10-12/`; this does not replace or relicense upstream data.

It also covers the project-authored initial/selected checkpoints and derived evidence in `outputs/reviewer_verification/loss_anchor_ablation_2026-10-12/`. Its training source remains MIT-licensed, and its input data retain the exact upstream attribution and release terms listed in the linked manifests.

The evaluation labels and predictions are derived from the PhysioNet/Computing in Cardiology Challenge 2021 v1.0.3 release (DOI `10.13026/34va-7q14`); users must also retain upstream attribution and comply with the source release terms. This notice does not license or redistribute the source ECG recordings, third-party software, or third-party model assets. No source ECG waveforms are included in this repository.

The grant applies only to rights held by the project authors. It does not grant rights held by another author, institution, dataset provider, or model provider. See [LICENSE_SCOPE.md](LICENSE_SCOPE.md) and [DATA_PROVENANCE.md](DATA_PROVENANCE.md).
