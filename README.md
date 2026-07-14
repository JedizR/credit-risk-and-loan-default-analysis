# Credit Risk and Loan Default Analysis

Final project for the Practical Machine Learning class. Predicts whether a loan applicant is approved from their financial, demographic, and employment profile.

## Dataset

`data/raw/loan_risk_prediction_dataset.csv` — 5,000 applicants, 9 features, binary target `LoanApproved`. See [data/raw/datacard.md](data/raw/datacard.md).

The data moves through three stages, each stored as a typed parquet and hashed into
`data/registry.json`:

| Stage | Path | What it holds |
| --- | --- | --- |
| raw | `data/raw/*.csv` | the original file, never modified |
| processed | `data/processed/applicants.parquet` | `make prepare` — impossible values repaired, dtypes fixed |
| preprocessed | `data/preprocessed/applicants.parquet` | `make preprocess` — the model-ready frame, with the engineered features |

The preprocessed parquet is the exact input the model pipeline is fitted on. Only the
*stateless* half of preprocessing is stored there: imputation, scaling and encoding are
deliberately left out, because they are fitted per training fold — a globally fitted version
saved to disk would leak the holdout into the training data. Those steps stay inside the
sklearn pipeline.

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
make preprocess                     # engineer the features, write the model-ready parquet
make train                          # full pipeline: outliers, selection, tuning, threshold
make train MODEL=random_forest      # train a different model
make train PLOT=true                # also write every figure to reports/figures/
make eval                           # score the saved model on the held-out 20%
make predict APPLICANTS=new.csv     # score new applicants into out/predictions.csv
make explain APPLICANTS=new.csv     # score them with a readable reason per decision
make lint                           # ruff check + format check
make test                           # pytest with coverage
```

`make train` runs the same pipeline the modelling notebook does: engineer features, drop
consensus outliers from the training rows, select a feature set, search hyperparameters with
Optuna, choose the decision threshold by expected cost, and score once on the holdout. Each
stage can be switched off:

```bash
make train TUNE=false SELECT=false REMOVE_OUTLIERS=false   # plain fit, no pipeline steps
make train TRIALS=100                                      # a longer Optuna search
```

The CLI takes the same work directly:

```bash
uv run credit-risk train --model-name lightgbm --tune --select-features --plots
uv run credit-risk explain --input-path new.csv --output-path out/decisions.csv
```

Available models: `lightgbm` (default), `logistic_regression`, `random_forest`,
`gradient_boosting`.

## Configuration

`src/credit_risk/config.py` is the single source of truth for paths, the random seed, the
domain thresholds measured in the exploratory analysis, and the training knobs. Override any of
it with a `config.toml` in the project root — no code change needed:

```toml
seed = 7

[training]
model_name = "random_forest"
optuna_trials = 100
false_approval_cost = 10.0     # a bad approval costs ten times a missed good applicant
```

## Docker

The image ships the CLI as its entry point, so anything after the image name is passed
straight to `credit-risk`. Data and artefacts are mounted rather than baked in.

```bash
make docker-build                            # build the image
make docker-prepare                          # build the typed dataset in the container
make docker-train MODEL=lightgbm PLOT=true   # full pipeline, figures written to the host
make docker-eval                             # evaluate the saved model
make docker-predict                          # score APPLICANTS into out/
make docker-explain                          # decisions with reasons into out/
```

The whole pipeline runs in the container, and results land on the host:

| Mount | Mode | Contents |
| --- | --- | --- |
| `data/raw` | read-only | the immutable source dataset |
| `data/` | read-write | derived parquet and provenance |
| `models/`, `out/` | read-write | model artefact, predictions and decisions |
| `reports/` | read-write | `metrics.json`, and `figures/` when `PLOT=true` |

With `PLOT=true` the container writes the model dynamics (learning curve, ROC/PR, calibration),
the evaluation (confusion matrix, threshold cost), the tuning history, the SHAP explanations
(beeswarm, importance, dependence) and the error analysis into `reports/figures/`.

A bare `docker run --rm credit-risk` prints the CLI help — the smoke test CI runs on every
pull request. Tagging a release on `main` publishes the image to
`ghcr.io/jedizr/credit-risk-and-loan-default-analysis`.

## Layout

```
src/credit_risk/
  config.py         paths, seed, domain thresholds and training knobs
  data/             schema, parquet/duckdb io, quality rules and provenance
  eda/              profiling and exploratory figures
  anomaly/          isolation forest + one-class svm, outlier handling
  features/         engineering transformers and consensus selection
  pipeline.py       preprocessing and the model registry
  tuning.py         optuna search with cross-validation inside the objective
  train.py          split, fit, calibrate, score, persist
  evaluation.py     cross-validation, cost-based threshold, model dynamics
  explain.py        shap explanations and human-readable decision reasons
  error_analysis.py where the model is wrong, how wrong, and on whom
  workflow.py       the whole pipeline, shared by the notebook and the CLI
  cli.py            prepare / train / evaluate / predict / explain
tests/              pytest suite behind an 85% coverage gate
notebooks/          01 exploratory analysis, 02 modelling
```

Notebooks contain no function definitions: they import from `credit_risk` and narrate, so the
notebook, the CLI and the container all run the same code.

## Contributing

Gitflow: branch `feature/*` off `develop`, rebase onto `develop`, open a pull request. `main` holds releases only, tagged `vMAJOR.MINOR.PATCH`.

`make lint && make test` must pass before pushing; CI runs the same checks on every pull request.
