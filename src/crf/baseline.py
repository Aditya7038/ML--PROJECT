"""Step 1: distance-only baseline vs a 3-feature model on the ESA CDM data.

Usage:
    python baseline.py path/to/train_data.csv

One row per EVENT is built from the CDM table:
  - snapshot = the last CDM that is still at least MIN_DAYS before closest approach
               (what an operator would see early)
  - label    = 1 if the event's FINAL risk (last CDM) is >= HIGH_RISK, else 0
Because there is one row per event, the train/test split is automatically by event.

Only features that also exist in SOCRATES are used: miss distance, relative speed,
time to closest approach.
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import train_test_split

HIGH_RISK = -6.0      # final risk >= this = high risk. CONFIRM on the Kelvins page.
MIN_DAYS = 2.0        # snapshot must be at least this many days before TCA. CONFIRM too.
TARGET_RECALL = 0.90  # we want to keep at least this share of the real risks
SEED = 42
FEATURES = ["time_to_tca", "miss_km", "speed_kms"]


def build_event_table(df: pd.DataFrame) -> pd.DataFrame:
    need = ["event_id", "time_to_tca", "risk", "miss_distance", "relative_speed"]
    df = df[need].dropna()
    # earliest CDM first, final CDM (smallest time_to_tca) last within each event
    df = df.sort_values(["event_id", "time_to_tca"], ascending=[True, False])
    final = df.groupby("event_id").tail(1).set_index("event_id")
    early = df[df["time_to_tca"] >= MIN_DAYS]
    snap = early.groupby("event_id").tail(1).set_index("event_id")
    ev = snap[["time_to_tca"]].copy()
    ev["miss_km"] = snap["miss_distance"] / 1000.0       # ESA metres -> km (check!)
    ev["speed_kms"] = snap["relative_speed"] / 1000.0    # ESA m/s -> km/s (check!)
    ev["high_risk"] = (final.loc[ev.index, "risk"] >= HIGH_RISK).astype(int)
    return ev


def report(name: str, y: np.ndarray, score: np.ndarray) -> None:
    ap = average_precision_score(y, score)
    auc = roc_auc_score(y, score)
    prec, rec, thr = precision_recall_curve(y, score)
    ok = np.where(rec[:-1] >= TARGET_RECALL)[0]
    print(f"\n{name}")
    print(f"  PR-AUC {ap:.3f}   ROC-AUC {auc:.3f}")
    if len(ok) == 0:
        print(f"  never reaches recall {TARGET_RECALL:.0%}")
        return
    t = thr[ok[-1]]                      # highest threshold that still keeps the recall
    flagged = score >= t
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    print(f"  at recall >= {TARGET_RECALL:.0%}: flagged {int(flagged.sum())} alerts, "
          f"caught {tp}/{n_pos} real risks, {fp} false alarms kept")
    print(f"  false alarms removed vs alerting on everything: {1 - fp / n_neg:.1%}")


def main(path: str) -> None:
    df = pd.read_csv(path)
    ev = build_event_table(df)
    print(f"events with a usable snapshot: {len(ev)}   high-risk: {int(ev.high_risk.sum())} "
          f"({ev.high_risk.mean():.2%})")
    print(f"median miss distance {ev.miss_km.median():.1f} km, "
          f"median speed {ev.speed_kms.median():.1f} km/s  <- sanity-check the units")

    train, test = train_test_split(ev, test_size=0.25, stratify=ev["high_risk"], random_state=SEED)
    y_test = test["high_risk"].to_numpy()

    # Baseline: smaller miss distance = higher risk
    report("Distance-only baseline", y_test, -test["miss_km"].to_numpy())

    # 3-feature model (features that also exist in SOCRATES)
    model = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)
    model.fit(train[FEATURES], train["high_risk"])
    report("3-feature model", y_test, model.predict_proba(test[FEATURES])[:, 1])


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python baseline.py path/to/train_data.csv")
    main(sys.argv[1])
