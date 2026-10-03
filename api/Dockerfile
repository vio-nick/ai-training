FROM python:3.12-slim

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_RELEASE_DIR=/app/model_release

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY api ./api
RUN pip install --no-cache-dir --index-url "$PIP_INDEX_URL" '.[api]' \
    && useradd --system --uid 10001 --create-home app

USER app
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
