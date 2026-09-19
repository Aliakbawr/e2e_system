# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

ARG APP_UID=1000
ARG APP_GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    NUMBA_CACHE_DIR=/tmp/persian_assistant_cache/numba \
    MPLCONFIGDIR=/tmp/persian_assistant_cache/matplotlib

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        alsa-utils \
        ca-certificates \
        libasound2 \
        libasound2-plugins \
        libespeak-ng1 \
        libgomp1 \
        libportaudio2 \
        libsndfile1 \
        pulseaudio-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
RUN python -m pip install --upgrade pip \
    && python -m pip install torch --index-url "${PYTORCH_INDEX_URL}" \
    && python -m pip install -r requirements.txt

ENV TORCH_DISABLE_NATIVE_JIT=1

RUN groupadd --gid "${APP_GID}" assistant \
    && useradd --uid "${APP_UID}" --gid "${APP_GID}" --create-home assistant \
    && install -d -o assistant -g assistant \
        /app/data/audio_in \
        /app/data/audio_out \
        /app/data/logs \
        /tmp/persian_assistant_cache/numba \
        /tmp/persian_assistant_cache/matplotlib

COPY --chown=assistant:assistant config ./config
COPY --chown=assistant:assistant scripts ./scripts
COPY --chown=assistant:assistant src ./src
COPY --chown=assistant:assistant main.py README.md LICENSE ./

USER assistant

ENTRYPOINT ["python", "-m", "scripts.container_entrypoint"]
CMD ["python", "main.py"]

FROM runtime AS test

USER root
RUN python -m pip install "pandas<3" pytest
COPY --chown=assistant:assistant tests ./tests
COPY --chown=assistant:assistant benchmark/__init__.py ./benchmark/__init__.py
COPY --chown=assistant:assistant benchmark/asr_parameter_selection/__init__.py benchmark/asr_parameter_selection/common.py ./benchmark/asr_parameter_selection/
COPY --chown=assistant:assistant benchmark/asr_llm_error_propagation/__init__.py benchmark/asr_llm_error_propagation/prepare_downstream_eval.py ./benchmark/asr_llm_error_propagation/
COPY --chown=assistant:assistant benchmark/asr_llm_error_propagation/asr/__init__.py benchmark/asr_llm_error_propagation/asr/run_fleurs_fa_ir_asr.py ./benchmark/asr_llm_error_propagation/asr/
USER assistant

ENTRYPOINT []
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests"]
