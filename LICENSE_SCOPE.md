# License Scope

## Original Software

The MIT License in `LICENSE` applies only to original software source code,
tests, and build/configuration files in `src/`, `scripts/`, `cedia/`,
`hardware/esp32/`, and `tests/`, plus the root `run` script, `Makefile`,
`pyproject.toml`, and `requirements*.txt`. It excludes the third-party and
generated materials listed below. The copyright notice names the software
author identified in `CITATION.cff`. The MIT grant does not relicense files
owned by another party.

## Excluded Materials

The MIT License does not apply to:

- ECG datasets, row-level labels, metadata, source record identifiers, or
  files under `data/`.
- Model checkpoints and derived weights, including `.pth`, `.tflite`, and the
  generated `hardware/esp32/firmware/src/model_data.h`.
- ECG or other recordings under `hardware/esp32/recordings/`.
- Aggregate results, validation reports, evidence packages, and other research
  artifacts under `outputs/`, `reports/`, and `models/`.
- Third-party software, documentation, or assets, which remain under their
  respective notices and license terms.

The source-specific licenses and citations listed in `REFERENCES.md` apply
where they match the exact included material. A dataset citation alone does
not establish that a local record came from that dataset or authorize
redistribution. In particular, the rights and provenance of the 312 `data_test`
rows are unresolved. Do not infer a data, model, or recording license from the
MIT license for software. See `DATA_PROVENANCE.md` before using or redistributing
those materials.
