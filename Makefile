VENV=.venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

setup:
	python -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

check:
	$(PYTHON) -c "import sklearn, xgboost; print('Environment OK')"
	
run:
	$(PYTHON) -m scripts.run_pipeline

clean:
	rm -rf outputs/*