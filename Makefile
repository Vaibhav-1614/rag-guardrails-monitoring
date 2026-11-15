PY ?= python
export PYTHONPATH := src:.

.PHONY: sync-upstream install install-ml test lint eval eval-v1 drift-validate serve dashboard docker

install:
	git submodule update --init
	$(PY) -m pip install -r requirements-dev.txt

install-ml:
	$(PY) -m pip install -r requirements-ml.txt && $(PY) -m spacy download en_core_web_sm

lint:
	ruff check .

test:
	pytest

eval:            ## guard eval + CI gates on the promoted config
	$(PY) eval/run_guard_eval.py --config configs/guards/v2.yaml --gates eval/gates.yaml

eval-v1:         ## same suite on the initial shadow-mode config (fails gates by design)
	$(PY) eval/run_guard_eval.py --config configs/guards/v1.yaml --gates eval/gates.yaml --no-plots

drift-validate:  ## simulate traffic scenarios and check every detector fires on the right one
	$(PY) -m monitor.validate_detectors

serve:
	uvicorn rag.api:app --reload --port 8000

dashboard:
	streamlit run dashboard/app.py

docker:
	docker build -t rag-guardrails .

sync-upstream:   ## move to the latest rag-eval-platform commit, then re-check everything against it
	git submodule update --remote upstream/rag-eval-platform
	$(PY) scripts/build_pii_allowlist.py
	$(PY) eval/build_synthetic_cases.py
	$(MAKE) test eval
