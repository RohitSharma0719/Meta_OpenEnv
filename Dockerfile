FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

COPY . .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps .

# HF Spaces injects $PORT; default to 7860 locally.
EXPOSE 7860

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860}"]
