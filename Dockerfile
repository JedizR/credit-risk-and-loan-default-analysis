FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project --no-editable

COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable


FROM python:3.12-slim-bookworm

RUN useradd --create-home --uid 1000 analyst

WORKDIR /app

COPY --from=builder --chown=analyst:analyst /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

USER analyst

ENTRYPOINT ["credit-risk"]
CMD ["--help"]
