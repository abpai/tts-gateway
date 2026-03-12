.PHONY: help install install-dev setup pre-commit-install pre-commit-run lint format typecheck test clean lock-check run
VENV_DIR = .venv
PROVIDER ?= kokoro

help:
	@echo 'Available commands:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	uv sync

install-dev: ## Install development dependencies
	uv sync --group dev

setup: ## Initialize development environment
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo 'Creating virtual environment in $(VENV_DIR)...'; \
		uv venv; \
	fi
	@echo 'Installing dependencies...'
	@uv sync --group dev
	@echo 'Installing pre-commit hooks...'
	@uv run pre-commit install
	@echo '\n✅ Setup complete. To activate the environment, run:\nsource .venv/bin/activate'

pre-commit-install: ## Install pre-commit hooks
	uv run pre-commit install

pre-commit-run: ## Run pre-commit hooks on all files
	uv run pre-commit run --all-files

lint: ## Run linter (ruff)
	uv run ruff check . --fix

format: ## Run formatter (ruff)
	uv run ruff format .

typecheck: ## Run ty type checker
	uv run ty check

test: ## Run Pytest
	uv run pytest

clean: ## Remove caches & pyc files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage

lock-check: ## Ensure uv.lock is up-to-date
	uv sync --locked --group dev

run: ## Run the TTS server
	uv run tts serve --provider $(PROVIDER)
