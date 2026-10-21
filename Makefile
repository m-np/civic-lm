PY ?= python
CFG ?= configs/scope.yaml

.PHONY: ingest normalize versions data rebuild check-manifest lock spotcheck test lint

ingest:          ## stage 1: download raw snapshots (never overwrites)
	$(PY) -m civic_data.ingest.ecfr --config $(CFG)
	$(PY) -m civic_data.ingest.fedreg --config $(CFG)
	$(PY) -m civic_data.ingest.hf_sets --config $(CFG)

normalize:       ## stage 2: raw XML -> sections.parquet + provenance.parquet
	$(PY) -m civic_data.normalize --config $(CFG)

versions:        ## stage 3: section_history.parquet + diffs.parquet
	$(PY) -m civic_data.versions --config $(CFG)

data: normalize versions

rebuild: data check-manifest   ## gate: rebuild from raw/ reproduces the recorded hashes

check-manifest:
	$(PY) -m civic_data.manifest --check

lock:            ## record current processed hashes as the expected rebuild result
	cp data/processed/MANIFEST.json configs/data_manifest.lock.json

spotcheck:       ## gate 2: 50 stratified diffs for hand review
	$(PY) -m civic_data.spotcheck -n 50

test:
	$(PY) -m pytest -q

lint:
	ruff check .
	lint-imports
