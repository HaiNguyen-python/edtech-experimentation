.PHONY: install test lint typecheck format check generate report serve

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy src

format:
	ruff format .
	ruff check --fix .

# The same gate as CI
check: lint typecheck
	pytest --cov-fail-under=80

# Seeded synthetic logs in data/ (users, exposures, events as Parquet)
generate:
	xp generate

# All five experiment reports, simulations, docs/results.md and charts
report:
	xp report

serve:
	xp serve
