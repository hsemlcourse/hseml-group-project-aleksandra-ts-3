.PHONY: lint lint-fix test install export-deploy run-api run-ui report-plot report-pdf

install:
	python -m pip install -r requirements.txt

export-deploy:
	python -m src.scripts.export_deploy

run-api:
	uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000

run-ui:
	streamlit run app/streamlit_app.py --server.port 8501

report-plot:
	python -m src.scripts.plot_feature_importance

lint:
	ruff check src/ tests/ --line-length 120
	flake8 src/ tests/ --max-line-length=120 --extend-ignore=E203,W503

lint-fix:
	ruff check src/ tests/ --line-length 120 --fix

test:
	python -m pytest tests/ -q
