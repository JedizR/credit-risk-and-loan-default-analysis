.PHONY: help setup lint test prepare preprocess train eval predict explain \
	docker-build docker-prepare docker-preprocess docker-train docker-eval docker-predict \
	docker-explain clean

IMAGE := credit-risk
APPLICANTS := data/raw/loan_risk_prediction_dataset.csv
ARTEFACT_DIRS := models reports out

# Training knobs. Override on the command line, e.g. make docker-train MODEL=random_forest PLOT=true
MODEL ?= lightgbm
PLOT ?= false
TUNE ?= true
SELECT ?= true
REMOVE_OUTLIERS ?= true
TRIALS ?=

TRAIN_FLAGS := --model-name $(MODEL) \
	$(if $(filter true,$(PLOT)),--plots) \
	$(if $(filter true,$(TUNE)),--tune) \
	$(if $(filter true,$(SELECT)),--select-features) \
	$(if $(filter true,$(REMOVE_OUTLIERS)),,--keep-outliers) \
	$(if $(TRIALS),--trials $(TRIALS))

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

preprocess:  ## Engineer the features and save the model-ready parquet
	uv run credit-risk preprocess

train: | $(ARTEFACT_DIRS)  ## Train: make train MODEL=lightgbm PLOT=true TUNE=true
	uv run credit-risk train $(TRAIN_FLAGS)

eval:  ## Evaluate the saved model on the holdout split
	uv run credit-risk evaluate

predict:  ## Score applicants from APPLICANTS=path/to.csv
	uv run credit-risk predict --input-path $(APPLICANTS)

explain:  ## Score applicants with a readable reason per decision
	uv run credit-risk explain --input-path $(APPLICANTS)

docker-build:  ## Build the container image
	docker build -t $(IMAGE) .

docker-prepare: | $(ARTEFACT_DIRS)  ## Build the typed dataset inside the container
	$(DOCKER_RUN) prepare

docker-preprocess: | $(ARTEFACT_DIRS)  ## Save the model-ready dataset inside the container
	$(DOCKER_RUN) preprocess

docker-train: | $(ARTEFACT_DIRS)  ## Train in the container: make docker-train MODEL=lightgbm PLOT=true
	$(DOCKER_RUN) train $(TRAIN_FLAGS)

docker-eval: | $(ARTEFACT_DIRS)  ## Evaluate the saved model inside the container
	$(DOCKER_RUN) evaluate

docker-predict: | $(ARTEFACT_DIRS)  ## Score APPLICANTS in the container into out/
	$(DOCKER_RUN) predict --input-path $(APPLICANTS)

docker-explain: | $(ARTEFACT_DIRS)  ## Explain decisions in the container into out/
	$(DOCKER_RUN) explain --input-path $(APPLICANTS)

$(ARTEFACT_DIRS):
	mkdir -p $@

clean:  ## Remove generated artefacts
	rm -rf $(ARTEFACT_DIRS) .pytest_cache .ruff_cache .coverage
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
