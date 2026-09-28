.PHONY: setup clean_venv

venv:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip
#	. .venv/bin/activate && pip install -r requirements.txt
#	#. .venv/bin/activate && pip install -e .
	@echo "To activate the virtual environment, run: source .venv/bin/activate"

clean_venv:
	rm -rf .venv