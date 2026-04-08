FROM python:3.11-slim

WORKDIR /app

ENV PYTHONIOENCODING=utf-8
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY run_standalone.py .
COPY app.py .
COPY inference.py .
COPY openenv.yaml .

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
