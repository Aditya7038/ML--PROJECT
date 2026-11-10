"""Step 2: cross-validated comparison with confidence intervals.

Usage:
    python src/crf/evaluate.py data/train_data.csv

Compares, with 5-fold cross-validation (one row per event, so the split is by event):
  - distance-only baseline (a rule, not trained)
  - logistic regression, random forest, gradient boosting on the 2 shared features (miss distance, speed)
  - a REFERENCE gradient boosting model on all ESA features (includes covariance).
    It is not usable on SOCRATES; it only shows what covariance adds.
    risk, max_risk_estimate and max_risk_scaling are dropped so it cannot copy the label.

Metrics come from the pooled out-of-fold scores. Confidence intervals are 95% bootstrap
intervals over events. Reported on all events, and on events with miss distance < 5 km
(the range SOCRATES covers).
"""
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HIGH_RISK = -6.0      # final risk >= this = high risk (confirm on the Kelvins page)
MIN_DAYS = 2.0        # snapshot = last CDM at least this many days before TCA (confirm too)
TARGET_RECALL = 0.90
SEED = 42
N_FOLDS = 5
N_BOOT = 200
# time_to_tca is deliberately NOT a feature: in ESA train data it is a collection artifact
# (early alerts are riskier there) that does not hold in ESA test data. See README.
SHARED = ["miss_km", "speed_kms"]
NO_TIMING = ["time_to_tca"]
# "Catalog" tier: things a public catalog (CelesTrak GP + SATCAT) can supply for any object:
# object type, size (RCS), orbit shape. No covariance, no tracking quality.
CATALOG = SHARED + ["c_object_type", "t_rcs_estimate", "c_rcs_estimate",
                    "t_j2k_sma", "t_j2k_ecc", "t_j2k_inc", "c_j2k_sma", "c_j2k_ecc", "c_j2k_inc",
                    "t_h_apo", "t_h_per", "c_h_apo", "c_h_per"]
OBJ_TYPES = ["DEBRIS", "PAYLOAD", "ROCKET BODY", "TBA", "UNKNOWN"]
LEAKY = ["risk", "max_risk_estimate", "max_risk_scaling"]


def build_events(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["event_id", "time_to_tca", "risk", "miss_distance", "relative_speed"]).copy()
    df = df.sort_values(["event_id", "time_to_tca"], ascending=[True, False])
    final_risk = df.groupby("event_id")["risk"].last()          # last CDM of each event
    snap = df[df["time_to_tca"] >= MIN_DAYS].groupby("event_id").tail(1).set_index("event_id")
    snap["high_risk"] = (final_risk.loc[snap.index] >= HIGH_RISK).astype(int)
    snap["miss_km"] = snap["miss_distance"] / 1000.0
    snap["speed_kms"] = snap["relative_speed"] / 1000.0
    if "c_object_type" in snap:
        snap["c_object_type"] = pd.Categorical(snap["c_object_type"], categories=OBJ_TYPES).codes
    return snap


def removal_at_recall(y: np.ndarray, s: np.ndarray) -> float:
    """Share of false alarms removed when the threshold keeps recall >= TARGET_RECALL."""
    order = np.argsort(-s, kind="stable")
    recall = np.cumsum(y[order]) / y.sum()
    k = min(int(np.searchsorted(recall, TARGET_RECALL)), len(s) - 1)
    flagged = s >= s[order][k]
    fp = int((flagged & (y == 0)).sum())
    return 1.0 - fp / int((y == 0).sum())


def metrics(y, s):
    return np.array([average_precision_score(y, s), roc_auc_score(y, s), removal_at_recall(y, s)])


def with_ci(y, s, rng):
    point = metrics(y, s)
    n = len(y)
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if y[idx].sum() == 0:
            continue
        boots.append(metrics(y[idx], s[idx]))
    lo, hi = np.percentile(np.array(boots), [2.5, 97.5], axis=0)
    return point, lo, hi


def oof_scores(make_model, X, y):
    oof = np.zeros(len(y))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for tr, te in skf.split(X, y):
        model = make_model()
        model.fit(X.iloc[tr], y[tr])
        oof[te] = model.predict_proba(X.iloc[te])[:, 1]
    return oof


def main(path: str) -> None:
    ev = build_events(pd.read_csv(path))
    y = ev["high_risk"].to_numpy()
    print(f"events: {len(ev)}  high-risk: {int(y.sum())} ({y.mean():.2%})")

    full_cols = [c for c in ev.select_dtypes("number").columns
                 if c not in LEAKY + NO_TIMING + ["high_risk"]]
    X3 = ev[SHARED]
    Xfull = ev[full_cols]
    print(f"reference model uses {len(full_cols)} features (covariance included)")

    scores = {"Distance-only baseline": -ev["miss_km"].to_numpy()}
    scores["Logistic regression (2 feat)"] = oof_scores(
        lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)), X3, y)
    scores["Random forest (2 feat)"] = oof_scores(
        lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                       random_state=SEED, n_jobs=-1), X3, y)
    scores["Gradient boosting (2 feat)"] = oof_scores(
        lambda: HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED), X3, y)
    scores["Gradient boosting (catalog feat)"] = oof_scores(
        lambda: HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED), ev[CATALOG], y)
    scores["REFERENCE: gradient boosting (all feat)"] = oof_scores(
        lambda: HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED), Xfull, y)

    rng = np.random.default_rng(SEED)
    subsets = {"ALL EVENTS": np.ones(len(y), bool), "MISS DISTANCE < 5 km": (ev["miss_km"] < 5).to_numpy()}
    for title, mask in subsets.items():
        ys = y[mask]
        print(f"\n=== {title}: {int(mask.sum())} events, {int(ys.sum())} high-risk ===")
        print(f"{'model':42s} {'PR-AUC':>20s} {'ROC-AUC':>20s} {'false alarms removed @90% recall':>36s}")
        for name, s in scores.items():
            p, lo, hi = with_ci(ys, s[mask], rng)
            print(f"{name:42s} {p[0]:.3f} [{lo[0]:.3f},{hi[0]:.3f}]  {p[1]:.3f} [{lo[1]:.3f},{hi[1]:.3f}]"
                  f"   {p[2]:6.1%} [{lo[2]:.1%},{hi[2]:.1%}]")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python src/crf/evaluate.py path/to/train_data.csv")
    main(sys.argv[1])
