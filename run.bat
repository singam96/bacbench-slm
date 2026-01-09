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
  -w /workspace ^
  %IMAGE_NAME% bash

endlocal
