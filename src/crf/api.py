"""FastAPI backend for the Conjunction Risk Filter.

Run:   uvicorn crf.api:app --app-dir src --reload      (then open http://127.0.0.1:8000/docs)
Data:  set CRF_SOCRATES_CSV to a CelesTrak SOCRATES CSV; otherwise the small sample in sample/ is used.

Endpoints:
  GET /health   service status and which file is loaded
  GET /alerts   ranked alerts with flags; query: max_miss_km, stale_days, active_only,
                hide_same_family, limit
"""
import os
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, Query

from crf.socrates import load_socrates, triage

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "sample" / "socrates_sample.csv"

app = FastAPI(title="Conjunction Risk Filter", version="0.1")


def _path() -> Path:
    return Path(os.environ.get("CRF_SOCRATES_CSV", SAMPLE))


@lru_cache(maxsize=4)
def _load(path: str, mtime: float):
    return load_socrates(path)


def _data():
    p = _path()
    return _load(str(p), p.stat().st_mtime)


@app.get("/health")
def health():
    p = _path()
    return {"status": "ok", "file": p.name, "rows": len(_data())}


@app.get("/alerts")
def alerts(max_miss_km: float = Query(0.5, ge=0, description="alerts closer than this are 'close'"),
           stale_days: float = Query(3.0, ge=0, description="older orbit file >= this many days = stale (assumption)"),
           active_only: bool = True,
           hide_same_family: bool = False,
           limit: int = Query(50, ge=1, le=1000)):
    t = triage(_data(), max_miss_km=max_miss_km, stale_days=stale_days,
               active_only=active_only, hide_same_family=hide_same_family)
    counts = t["label"].value_counts().to_dict()
    cols = ["rank", "label", "name1", "name2", "norad1", "norad2", "tca", "miss_km", "speed_kms",
            "max_dse", "max_prob", "dilution", "same_family"]
    rows = t[cols].head(limit).copy()
    rows["tca"] = rows["tca"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return {"counts": counts, "total": int(len(t)), "alerts": rows.to_dict(orient="records")}
