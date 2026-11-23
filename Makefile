.PHONY: install test app brief evals evals-offline
install:
	pip install -e ".[dev]"
test:
	pytest -q
app:
	streamlit run app/chat.py
brief:
	python -m pia.brief
evals:
	python -m pia.evals.run_evals
evals-offline:
	python -m pia.evals.run_evals --offline
