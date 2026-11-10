"""FastAPI backend for the Conjunction Risk Filter.

Run:   uvicorn crf.api:app --app-dir src --reload      (docs at http://127.0.0.1:8000/docs)

What it does:
  - downloads the latest CelesTrak SOCRATES file on startup and every CRF_REFRESH_HOURS (default 6),
    keeps the last good copy if a download fails, falls back to the bundled sample if none exists
  - serves ranked, flagged alerts and the validation results, so the dashboard and any other
    tool read the same data

Settings (environment variables):
  CRF_SOCRATES_URL     where to download from (default: CelesTrak sort-minRange.csv, unverified)
  CRF_REFRESH_HOURS    refresh interval, default 6
  CRF_AUTO_REFRESH     1 (default) or 0 to turn background downloads off
  CRF_SOCRATES_CSV     use this local file only, no downloads

Endpoints: GET /health, GET /alerts, GET /models, POST /refresh
"""
import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query

from crf.service import DEFAULT_URL, Service

ROOT = Path(__file__).resolve().parents[2]
REFRESH_HOURS = float(os.environ.get("CRF_REFRESH_HOURS", "6"))
AUTO = os.environ.get("CRF_AUTO_REFRESH", "1") == "1"
_override = os.environ.get("CRF_SOCRATES_CSV")

svc = Service(url=os.environ.get("CRF_SOCRATES_URL", DEFAULT_URL),
              cache=ROOT / "cache" / "socrates_latest.csv",
              sample=ROOT / "sample" / "socrates_sample.csv",
              results=ROOT / "results",
              refresh_hours=REFRESH_HOURS,
              override=Path(_override) if _override else None)


async def _refresh_loop():
    while True:
        await asyncio.to_thread(svc.refresh)
        await asyncio.sleep(REFRESH_HOURS * 3600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_refresh_loop()) if AUTO and not _override else None
    yield
    if task:
        task.cancel()


app = FastAPI(title="Conjunction Risk Filter", version="0.2", lifespan=lifespan)


@app.get("/health")
def health():
    """Service status: data source, when it was fetched, last download error."""
    return svc.health()


@app.get("/alerts")
def alerts(max_miss_km: float = Query(0.5, ge=0, description="pairs closer than this count as close"),
           stale_days: float = Query(3.0, ge=0, description="orbit file older than this = stale (assumption)"),
           active_only: bool = True,
           hide_same_family: bool = False,
           limit: int = Query(50, ge=1, le=1000)):
    """Ranked alerts, each labelled act / check data first / probably noise."""
    return svc.alerts(max_miss_km=max_miss_km, stale_days=stale_days, active_only=active_only,
                      hide_same_family=hide_same_family, limit=limit)


@app.get("/models")
def models():
    """Validation results: independent ESA test and cross-validation, per model."""
    return svc.models()


@app.post("/refresh")
async def refresh():
    """Download the latest SOCRATES file now."""
    return await asyncio.to_thread(svc.refresh)
