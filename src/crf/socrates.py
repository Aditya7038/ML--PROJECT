"""Load a CelesTrak SOCRATES CSV and turn it into a ranked, flagged alert table.

What the ranking is, and is not (see README, "Validation"):
  - Alerts are ranked by miss distance. On ESA's independent test set, miss distance alone
    removed about 60% of false alarms at 90% recall, and ML models built on the same free
    fields did not beat it. So we do not pretend to add ML here.
  - Flags add context a human needs: stale orbit data, SOCRATES' own MAX_PROB and DILUTION,
    inactive objects, and same-constellation pairs. These flags are NOT validated predictors.
    They are triage hints. The stale-data cut-off is an adjustable assumption.
"""
import re

import pandas as pd

STATUS_RE = re.compile(r"\s*\[([+\-?])\]\s*$")
STATUS_MEANING = {"+": "active", "-": "inactive", "?": "unknown"}


def _split_name(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    status = series.str.extract(STATUS_RE)[0].map(STATUS_MEANING).fillna("unknown")
    name = series.str.replace(STATUS_RE, "", regex=True).str.strip()
    return name, status


def _family(name: str) -> str:
    """Rough constellation key: first word of the name (STARLINK-123 -> STARLINK)."""
    return re.split(r"[\s\-]", name.strip(), maxsplit=1)[0].upper()


def load_socrates(path_or_buffer) -> pd.DataFrame:
    d = pd.read_csv(path_or_buffer)
    d["name1"], d["status1"] = _split_name(d["OBJECT_NAME_1"])
    d["name2"], d["status2"] = _split_name(d["OBJECT_NAME_2"])
    out = pd.DataFrame({
        "norad1": d["NORAD_CAT_ID_1"], "name1": d["name1"], "status1": d["status1"], "dse1": d["DSE_1"],
        "norad2": d["NORAD_CAT_ID_2"], "name2": d["name2"], "status2": d["status2"], "dse2": d["DSE_2"],
        "tca": pd.to_datetime(d["TCA"], utc=True),
        "miss_km": d["TCA_RANGE"],           # SOCRATES reports km
        "speed_kms": d["TCA_RELATIVE_SPEED"],  # and km/s
        "max_prob": d["MAX_PROB"],
        "dilution": d["DILUTION"],
    })
    out["max_dse"] = out[["dse1", "dse2"]].max(axis=1)
    out["same_family"] = out["name1"].map(_family) == out["name2"].map(_family)
    out["any_active"] = (out["status1"] == "active") | (out["status2"] == "active")
    return out


def triage(df: pd.DataFrame, *, max_miss_km: float = 1.0, stale_days: float = 3.0,
           active_only: bool = True, hide_same_family: bool = False,
           as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    """Rank by miss distance, add flags, and label each alert 'act' or 'probably noise'.

    'act'  = miss distance under max_miss_km AND orbit data not stale.
    'check data first' = close, but the older orbit file is stale (>= stale_days), so the
                         miss distance itself is unreliable; get a fresh orbit before acting.
    'probably noise' = everything else.
    """
    d = df.copy()
    if active_only:
        d = d[d["any_active"]]
    if hide_same_family:
        d = d[~d["same_family"]]
    d["stale"] = d["max_dse"] >= stale_days
    close = d["miss_km"] <= max_miss_km
    d["label"] = "probably noise"
    d.loc[close & ~d["stale"], "label"] = "act"
    d.loc[close & d["stale"], "label"] = "check data first"
    if as_of is not None:
        d["days_to_tca"] = (d["tca"] - pd.Timestamp(as_of, tz="UTC")).dt.total_seconds() / 86400.0
    return d.sort_values(["miss_km", "tca"]).reset_index(drop=True).assign(rank=lambda x: x.index + 1)
