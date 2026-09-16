PYTHON ?= python3

.PHONY: report external test validate notebook check app

report:
	$(PYTHON) scripts/build_report.py

external:
	$(PYTHON) scripts/build_pbp_features.py --download

test:
	$(PYTHON) -m pytest -q

validate:
	$(PYTHON) scripts/validate_repo.py

notebook:
	$(PYTHON) scripts/execute_notebook.py

check: test report validate notebook

app:
	$(PYTHON) -m streamlit run app/역전점_앱.py
