.PHONY: help setup lint test prepare train eval predict \
	docker-build docker-prepare docker-train docker-eval docker-predict clean

IMAGE := credit-risk
APPLICANTS := data/raw/loan_risk_prediction_dataset.csv
MODEL ?= logistic_regression
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

prepare:  ## Repair and convert the raw CSV to a typed parquet dataset
	uv run credit-risk prepare

train:  ## Train a model: make train MODEL=random_forest
	uv run credit-risk train --model-name $(MODEL)

eval:  ## Evaluate the saved model on the holdout split
	uv run credit-risk evaluate

predict:  ## Score applicants from APPLICANTS=path/to.csv
	uv run credit-risk predict --input-path $(APPLICANTS)

docker-build:  ## Build the container image
	docker build -t $(IMAGE) .

docker-prepare: | $(ARTEFACT_DIRS)  ## Build the typed dataset inside the container
	$(DOCKER_RUN) prepare

docker-train: | $(ARTEFACT_DIRS)  ## Train in the container: make docker-train MODEL=random_forest
	$(DOCKER_RUN) train --model-name $(MODEL)

docker-eval: | $(ARTEFACT_DIRS)  ## Evaluate the saved model inside the container
	$(DOCKER_RUN) evaluate

docker-predict: | $(ARTEFACT_DIRS)  ## Score APPLICANTS in the container into out/
	$(DOCKER_RUN) predict --input-path $(APPLICANTS)

$(ARTEFACT_DIRS):
	mkdir -p $@

clean:  ## Remove generated artefacts
	rm -rf $(ARTEFACT_DIRS) .pytest_cache .ruff_cache .coverage
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
