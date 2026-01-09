@echo off
setlocal

set IMAGE_NAME=bacbench-trainer:dev

echo Building Docker image %IMAGE_NAME% ...
docker build -t %IMAGE_NAME% .
if errorlevel 1 (
  echo Docker build failed.
  exit /b 1
)

echo.
echo Starting interactive shell in container...
echo - Project mounted at /workspace
echo - Run: python train.py

docker run --rm -it --gpus all ^
  -v "%cd%":/workspace ^
  -v bacbench-trainer-cache:/workspace/.cache ^
  -e XDG_CACHE_HOME=/workspace/.cache ^
  -e TORCH_HOME=/workspace/.cache/torch ^
  -e HF_HOME=/workspace/.cache/huggingface ^
  -e HF_HUB_CACHE=/workspace/.cache/huggingface/hub ^
  -e HF_DATASETS_CACHE=/workspace/.cache/huggingface/datasets ^
  -w /workspace ^
  %IMAGE_NAME% bash

endlocal
