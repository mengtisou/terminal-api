# Slim base: the app is pandas + FastAPI, nothing that needs a full build image.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Requirements first so Docker caches the dependency layer across code changes.
COPY requirements.txt .

# MetaTrader5 is Windows-only and will not install here. The provider chain
# detects that and falls through to Twelve Data / Yahoo automatically.
RUN grep -v -i "^MetaTrader5" requirements.txt > /tmp/req.txt \
    && pip install --no-cache-dir -r /tmp/req.txt

COPY app/ ./app/
COPY static/ ./static/

# Cloud hosts inject $PORT; default to 8000 for local runs.
ENV PORT=8000
EXPOSE 8000

# Single worker on purpose: the news store, candle cache and routing overrides
# live in process memory, so multiple workers would each hold a different copy.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
