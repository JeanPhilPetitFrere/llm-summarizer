FROM python:3.11-slim-bookworm
ARG USER_UID
ARG USER_GID

WORKDIR /app

ENV PATH="$PATH:/opt/.venv/bin"

RUN mkdir -p /opt/.venv && chown -vR ${USER_UID}:${USER_GID} /opt/.venv
RUN apt update -y && apt install -y gcc g++
RUN pip install --upgrade pip && pip install --no-cache-dir poetry

# Configure Poetry to use the virtual environment at /opt/.venv
ENV POETRY_VIRTUALENVS_PATH="/opt/.venv"
ENV POETRY_VIRTUALENVS_IN_PROJECT=false

COPY pyproject.toml poetry.lock* /app/

# Install the dependencies
RUN poetry install
CMD ["poetry","run","streamlit","run","app.py"]