@echo off
setlocal
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
  echo uv is required. Install it from https://docs.astral.sh/uv/getting-started/installation/
  exit /b 127
)

uv run --frozen streamlit run %* --server.address 127.0.0.1 src\ai_vocab_video_generator\webui.py
