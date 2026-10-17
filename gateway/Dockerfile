# Gateway image: a small CPU-only FastAPI service. The GPU is used by vLLM only.
FROM python:3.12.15-slim-trixie

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

# Dependencies first: this layer stays cached as long as requirements.txt
# does not change, so editing the code does not trigger a full reinstall.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

# Run as an unprivileged user: a compromised process does not get root.
RUN useradd --system --uid 10001 --no-create-home gateway
USER gateway

EXPOSE 8080

# `--no-access-log`: the app emits its own structured access log line.
CMD ["uvicorn", "app.main:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8080", "--no-access-log"]
