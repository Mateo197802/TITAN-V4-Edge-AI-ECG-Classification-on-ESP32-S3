# License Scope

This repository contains material under separate licenses. Do not infer a license for one category from the license of another.

## Software Source

The root [MIT License](LICENSE) applies to original software source code, tests, and build/configuration files in `src/`, `scripts/`, `cedia/`, `hardware/esp32/`, and `tests/`, plus the root `run` script, `Makefile`, `pyproject.toml`, and `requirements*.txt`. It does not apply to embedded model parameters in `hardware/esp32/firmware/src/model_data.h`.

## Research Artifacts

The separate [CC BY 4.0 artifact notice](LICENSE-ARTIFACTS.md) applies to project-authored reports, tables, validation labels/predictions, and model weights, subject to the third-party exclusions in that notice. The upstream Challenge 2021 release and source datasets retain their own attribution and license terms. No downloaded ECG waveform files are included in the Git repository.

## Exclusions

Neither the MIT license nor the CC BY 4.0 project-artifact notice grants rights to:

- Third-party ECG recordings, dataset content, source publications, or source-dataset marks.
- Third-party models, logits, code, documentation, or assets that may have influenced the project artifacts.
- Engineering recordings whose source/consent basis is not documented in this repository.
- Personal information or medical data beyond the specific checked-in artifact and source release terms.

The model grant is limited to rights held by the TITAN project authors; it cannot sublicense third-party material. The repository owner must ensure all co-authors and institutional rights holders have approved the applicable grant before treating it as a final legal clearance. See [data provenance](DATA_PROVENANCE.md) and the [audit report](AUDIT_REPORT.md).
