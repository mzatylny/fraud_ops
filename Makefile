.PHONY: install test coverage lint security quality train run docker

install:
	python -m pip install -r requirements-dev.txt

test:
	python -m pytest

coverage:
	python -m pytest --cov=src --cov=app --cov-report=term-missing --cov-fail-under=72

lint:
	ruff check .

security:
	bandit -r src train_models.py -q
	pip-audit --local

quality: lint coverage security

train:
	python train_models.py --customers 300 --terminals 600 --days 25 --max-transactions 25000

run:
	streamlit run app.py

docker:
	docker compose up --build
