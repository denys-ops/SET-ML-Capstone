#!/bin/sh
# Entry point: validate the W&B key, then serve. The app pulls the right model from
# the registry at startup (ONNX-INT8 by default; PyTorch if BACKEND=torch).
set -e

if [ -z "$WANDB_API_KEY" ]; then
    echo "ERROR: WANDB_API_KEY is not set."
    echo "Run with: docker run -e WANDB_API_KEY=your_key -p 8080:8080 phishing-scorer-serve"
    exit 1
fi

echo "Starting FastAPI server on :8080 (backend=${BACKEND:-onnx})..."
exec uvicorn main:app --host 0.0.0.0 --port 8080
