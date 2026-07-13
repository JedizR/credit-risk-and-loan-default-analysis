.PHONY: help setup lint test prepare train eval predict docker-build docker-train clean

IMAGE := credit-risk
APPLICANTS := data/raw/loan_risk_prediction_dataset.csv

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

train:  ## Train the model and write metrics
	uv run credit-risk train

eval:  ## Evaluate the saved model on the holdout split
	uv run credit-risk evaluate

predict:  ## Score applicants from APPLICANTS=path/to.csv
	uv run credit-risk predict --input-path $(APPLICANTS)

docker-build:  ## Build the container image
	docker build -t $(IMAGE) .

docker-train:  ## Train inside the container, writing results to the host
	docker run --rm \
		-v "$(PWD)/data:/app/data:ro" \
		-v "$(PWD)/models:/app/models" \
		-v "$(PWD)/reports:/app/reports" \
		$(IMAGE) train

clean:  ## Remove generated artefacts
	rm -rf models reports out .pytest_cache .ruff_cache .coverage
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
