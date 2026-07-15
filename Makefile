.PHONY: help setup lint test docker-build docker-run clean

IMAGE := credit-risk
ARGS ?= run --stage all
ARTEFACT_DIRS := models reports out

# Raw data stays read-only; derived data, models and reports are written back to the host.
DOCKER_RUN = docker run --rm \
	-v "$(PWD)/data:/app/data" \
	-v "$(PWD)/data/raw:/app/data/raw:ro" \
	-v "$(PWD)/models:/app/models" \
	-v "$(PWD)/reports:/app/reports" \
	-v "$(PWD)/out:/app/out" \
	$(IMAGE)

help:  ## Show the available targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*## /\t/'

setup:  ## Install the locked environment and the git hooks
	uv sync --all-groups
	uv run pre-commit install

lint:  ## Check style and formatting
	uv run ruff check .
	uv run ruff format --check .

test:  ## Run the test suite with coverage
	uv run pytest

docker-build:  ## Build the container image
	docker build -t $(IMAGE) .

docker-run: | $(ARTEFACT_DIRS)  ## Run the CLI in the container: make docker-run ARGS="run --stage train --tune"
	$(DOCKER_RUN) $(ARGS)

$(ARTEFACT_DIRS):
	mkdir -p $@

clean:  ## Remove generated artefacts
	rm -rf $(ARTEFACT_DIRS) .pytest_cache .ruff_cache .coverage
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
