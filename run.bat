@echo off
rem Starts the FastAPI backend in its own window, then the dashboard. Double-click to run.
cd /d %~dp0
if not exist .venv (
  python -m venv .venv
  call .venv\Scripts\activate
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate
)
start "CRF backend" cmd /k "call .venv\Scripts\activate && uvicorn crf.api:app --app-dir src --port 8000"
timeout /t 3 >nul
start "" http://localhost:8501
streamlit run dashboard/app.py --server.port 8501
