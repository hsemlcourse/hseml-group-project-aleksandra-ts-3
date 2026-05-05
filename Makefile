.PHONY: lint lint-fix test install

install:
	python -m pip install -r requirements.txt

lint:
	ruff check src/ tests/ --line-length 120
	flake8 src/ tests/ --max-line-length=120 --extend-ignore=E203,W503

lint-fix:
	ruff check src/ tests/ --line-length 120 --fix

test:
	python -m pytest tests/ -q
