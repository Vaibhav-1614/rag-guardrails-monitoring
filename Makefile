PY ?= python
export PYTHONPATH := src:.

.PHONY: install install-ml test lint eval eval-v1 drift-validate serve dashboard docker

install:
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
