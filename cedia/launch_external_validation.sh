#!/usr/bin/env bash
set -euo pipefail

python scripts/run_primary9_external_validation.py
python scripts/run_pathology_primary5_external_validation.py
python scripts/run_combined_external_validation.py

