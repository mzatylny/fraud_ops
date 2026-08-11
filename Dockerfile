FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/tmp

WORKDIR /app
RUN addgroup --system fraudops \
    && adduser --system --ingroup fraudops --home /home/fraudops fraudops

COPY requirements.txt ./
RUN python -m pip install --upgrade pip \
    && pip install -r requirements.txt

COPY app.py train_models.py ./
COPY src ./src
RUN python train_models.py \
      --customers 300 \
      --terminals 600 \
      --days 25 \
      --max-transactions 25000 \
    && chown -R fraudops:fraudops /app

USER fraudops
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
