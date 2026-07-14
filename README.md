# Credit Risk and Loan Default Analysis

Final project for the Practical Machine Learning class. Predicts whether a loan applicant is approved from their financial, demographic, and employment profile.

## Dataset

`data/raw/loan_risk_prediction_dataset.csv` — 5,000 applicants, 9 features, binary target `LoanApproved`. See [data/raw/datacard.md](data/raw/datacard.md). `make prepare` repairs impossible values and writes a typed `data/processed/applicants.parquet` alongside a `data/registry.json` provenance record.

Three properties drive the modelling choices:

- **Imbalanced target.** 23% of applicants are approved, so accuracy is misleading — a model that rejects everyone scores 77%. Average precision (PR-AUC) is the headline metric, and every model trains with balanced class weights.
- **Missing values.** `Income`, `CreditScore`, and `Education` are each about 4% null. Numeric gaps are median-imputed; missing `Education` becomes its own category rather than being dropped.
- **Mixed types.** Numeric features are scaled and categorical features are one-hot encoded inside the fitted pipeline, so scoring cannot drift from training.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Python itself is installed by uv.

```bash
make setup
```

## Usage

```bash
make prepare                        # repair impossible values, write the typed parquet
make train                          # train, save models/model.joblib, write reports/metrics.json
make train MODEL=random_forest      # train a different model
make eval                           # score the saved model on the held-out 20%
make predict APPLICANTS=new.csv     # score new applicants into out/predictions.csv
make lint                           # ruff check + format check
make test                           # pytest with coverage
```

The CLI takes the same work directly:

```bash
uv run credit-risk train --model-name gradient_boosting
uv run credit-risk predict --input-path new.csv --output-path out/scores.csv
```

Available models: `logistic_regression` (default), `random_forest`, `gradient_boosting`.

## Docker

The image ships the CLI as its entry point, so anything after the image name is passed
straight to `credit-risk`. Data and artefacts are mounted rather than baked in.

```bash
make docker-build                        # build the image
make docker-prepare                      # build the typed dataset in the container
make docker-train MODEL=random_forest    # train in the container
make docker-eval                         # evaluate the saved model
make docker-predict                      # score APPLICANTS into out/
```

The whole pipeline runs in the container, and results land on the host:

| Mount | Mode | Contents |
| --- | --- | --- |
| `data/raw` | read-only | the immutable source dataset |
| `data/` | read-write | derived parquet and provenance |
| `models/`, `reports/`, `out/` | read-write | model, metrics, predictions |

A bare `docker run --rm credit-risk` prints the CLI help — the smoke test CI runs on every
pull request. Tagging a release on `main` publishes the image to
`ghcr.io/jedizr/credit-risk-and-loan-default-analysis`.

## Layout

```
src/credit_risk/
  data/         schema, parquet/duckdb io, quality rules and provenance
  pipeline.py   preprocessing and the model registry
  train.py      split, fit, score, persist
  cli.py        prepare / train / evaluate / predict
tests/          pytest suite behind an 85% coverage gate
notebooks/      exploratory analysis
```

## Contributing

Gitflow: branch `feature/*` off `develop`, rebase onto `develop`, open a pull request. `main` holds releases only, tagged `vMAJOR.MINOR.PATCH`.

`make lint && make test` must pass before pushing; CI runs the same checks on every pull request.
