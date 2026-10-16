.PHONY: test verify primary9 pathology combined tables reproduce

test:
	python -m pytest -q

verify:
	python scripts/verify_artifact_hashes.py

primary9:
	python scripts/run_primary9_inference.py
	python scripts/run_primary9_external_validation.py

pathology:
	python scripts/run_pathology_primary5_external_validation.py

combined:
	python scripts/run_combined_external_validation.py --write

tables:
	python scripts/build_result_tables.py

reproduce: primary9 combined tables test verify

