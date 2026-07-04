# HW3 — CPU-only serving image for the phishing-obviousness scorer.
FROM python:3.11-slim

ENV WANDB_CACHE_DIR=/app/.cache/wandb \
    HF_HOME=/app/.cache/hf \
    ONNX_DIR=/app/models/onnx \
    MODEL_DIR=/app/models/serve \
    LOG_DIR=/app/logs \
    BACKEND=onnx \
    PYTHONUNBUFFERED=1
WORKDIR /app
RUN mkdir -p /app/.cache/wandb /app/.cache/hf /app/models /app/logs

# App code (preprocess.py is the SAME contract used at training time)
COPY requirements.txt requirements.txt
COPY preprocess.py scoring.py infer.py onnx_infer.py explain.py \
     download_model.py export_onnx.py main.py metrics.py predlog.py \
     load_gen.py start.sh sample_emails.json ./
RUN chmod +x start.sh

# CPU torch first (much smaller than the default CUDA build), then the rest.
# torch is needed for /explain (Integrated Gradients) and for the ONNX export step.
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

EXPOSE 8080
ENTRYPOINT ["./start.sh"]
