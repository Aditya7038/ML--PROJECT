"""Backend core (no web framework): fetch SOCRATES on a schedule, cache it, serve it.

api.py is a thin FastAPI wrapper around this. Keeping the logic here means it can be tested
without FastAPI installed.

Why a backend at all: CelesTrak publishes a new SOCRATES file about three times a day. The
service downloads it, keeps the last good copy if a download fails, and lets the dashboard
(and any other tool) read the same ranked alerts through one API.
"""
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from crf.socrates import load_socrates, triage

# Best guess from the file name CelesTrak uses; verify by opening it in a browser,
# or override with the CRF_SOCRATES_URL environment variable.
DEFAULT_URL = "https://celestrak.org/SOCRATES/sort-minRange.csv"
EXPECTED_START = "NORAD_CAT_ID_1"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def fetch_socrates(url: str, dest: Path, timeout: float = 120.0) -> int:
    """Download a SOCRATES CSV to dest. Writes atomically; refuses files that are not SOCRATES CSVs."""
    req = urllib.request.Request(url, headers={"User-Agent": "crf-student-project/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    if not data.lstrip().startswith(EXPECTED_START.encode()):
        raise ValueError("downloaded file does not look like a SOCRATES CSV")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return len(data)


class Service:
    def __init__(self, url: str, cache: Path, sample: Path, results: Path, refresh_hours: float = 6.0,
                 override: Path | None = None, timeout: float = 120.0):
        self.url, self.cache, self.sample, self.results = url, cache, sample, results
        self.refresh_hours = refresh_hours
        self.timeout = timeout
        self.override = override
        self.meta_path = cache.with_suffix(".json")
        self.last_attempt: str | None = None
        self.last_error: str | None = None
        self.fetched_at: str | None = None
        self._frame: tuple[str, float, pd.DataFrame] | None = None
        if self.meta_path.exists():
            try:
                self.fetched_at = json.loads(self.meta_path.read_text()).get("fetched_at")
            except (OSError, ValueError):
                pass

    # ---- data -------------------------------------------------------------------------------
    def path(self) -> Path:
        if self.override:
            return self.override
        return self.cache if self.cache.exists() else self.sample

    def source(self) -> str:
        if self.override:
            return "file override"
        return "celestrak" if self.cache.exists() else "bundled sample"

    def refresh(self) -> dict:
        """Blocking download. On failure keep the last good file and record the error."""
        self.last_attempt = _now().isoformat()
        try:
            size = fetch_socrates(self.url, self.cache, self.timeout)
            self.fetched_at = _now().isoformat()
            self.last_error = None
            self.meta_path.write_text(json.dumps({"fetched_at": self.fetched_at, "bytes": size, "url": self.url}))
        except Exception as e:  # network, HTTP, or bad content: never crash the service
            self.last_error = f"{type(e).__name__}: {e}"
        return self.health()

    def ensure_fresh(self, retry_hours: float = 0.25) -> None:
        """Download if the cached file is missing or older than refresh_hours.

        Used when the dashboard runs without the FastAPI server: it still gets live data.
        After a failed attempt it waits retry_hours before trying again, so a page reload
        never hammers CelesTrak or blocks on a dead connection every time.
        """
        if self.override:
            return
        age = None
        if self.fetched_at and self.cache.exists():
            age = (_now() - datetime.fromisoformat(self.fetched_at)).total_seconds() / 3600
        if age is not None and age < self.refresh_hours:
            return
        if self.last_attempt:
            since = (_now() - datetime.fromisoformat(self.last_attempt)).total_seconds() / 3600
            if since < retry_hours:
                return
        self.refresh()

    def data(self) -> pd.DataFrame:
        p = self.path()
        key = (str(p), p.stat().st_mtime)
        if self._frame is None or self._frame[:2] != key:
            self._frame = (*key, load_socrates(p))
        return self._frame[2]

    def alerts(self, *, max_miss_km: float, stale_days: float, active_only: bool,
               hide_same_family: bool, limit: int) -> dict:
        t = triage(self.data(), max_miss_km=max_miss_km, stale_days=stale_days,
                   active_only=active_only, hide_same_family=hide_same_family)
        cols = ["rank", "label", "name1", "name2", "norad1", "norad2", "tca", "miss_km", "speed_kms",
                "max_dse", "max_prob", "dilution", "same_family"]
        rows = t[cols].head(limit).copy()
        rows["tca"] = rows["tca"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return {"counts": t["label"].value_counts().to_dict(), "total": int(len(t)),
                "alerts": rows.to_dict(orient="records")}

    # ---- status and results ------------------------------------------------------------------
    def health(self) -> dict:
        age = None
        if self.fetched_at:
            age = round((_now() - datetime.fromisoformat(self.fetched_at)).total_seconds() / 3600, 2)
        nxt = None
        if self.last_attempt:
            since = (_now() - datetime.fromisoformat(self.last_attempt)).total_seconds() / 3600
            nxt = round(max(0.0, self.refresh_hours - since), 2)
        return {"status": "ok", "source": self.source(), "file": self.path().name,
                "rows": int(len(self.data())), "fetched_at": self.fetched_at, "age_hours": age,
                "refresh_every_hours": self.refresh_hours, "next_refresh_in_hours": nxt,
                "last_error": self.last_error}

    def models(self) -> dict:
        return read_results(self.results)


def read_results(results_dir: Path) -> dict:
    out = {}
    for key, name in (("independent_test", "independent_test.json"), ("cross_validation", "cross_validation.json")):
        p = Path(results_dir) / name
        out[key] = json.loads(p.read_text()) if p.exists() else None
    return out
