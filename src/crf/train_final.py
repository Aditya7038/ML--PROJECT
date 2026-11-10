"""Step 3: train the final models on ALL of train_data.csv, check them once on the
independent ESA test files, and save the demo model.

Usage:
    python src/crf/train_final.py data/train_data.csv data/test_data.csv data/test_data_private.csv

ESA test layout (checked): test_data.csv holds only CDMs at least 2 days before closest
approach, and test_data_private.csv holds each event's final CDM with `true_risk`.
So the test label is true_risk >= HIGH_RISK, and the test snapshot is the last CDM in test_data.csv.

Prints metrics only. No model is saved: on free-data features no model beat the distance rule.
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import (CATALOG, HIGH_RISK, LEAKY, NO_TIMING, OBJ_TYPES, SEED, SHARED, build_events,  # noqa: E402
                      record, save_results, with_ci)


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))


def build_test_events(test_path: str, private_path: str) -> pd.DataFrame:
    t = pd.read_csv(test_path)
    t = t.dropna(subset=["event_id", "time_to_tca", "miss_distance", "relative_speed"])
    t = t.sort_values(["event_id", "time_to_tca"], ascending=[True, False])
    snap = t.groupby("event_id").tail(1).set_index("event_id")
    snap["miss_km"] = snap["miss_distance"] / 1000.0
    snap["speed_kms"] = snap["relative_speed"] / 1000.0
    if "c_object_type" in snap:
        snap["c_object_type"] = pd.Categorical(snap["c_object_type"], categories=OBJ_TYPES).codes
    priv = pd.read_csv(private_path).set_index("event_id")
    snap = snap.loc[snap.index.intersection(priv.index)]
    snap["high_risk"] = (priv.loc[snap.index, "true_risk"] >= HIGH_RISK).astype(int)
    return snap


def main(train_path: str, test_path: str, private_path: str) -> None:
    tr = build_events(pd.read_csv(train_path))
    te = build_test_events(test_path, private_path)
    ytr, yte = tr["high_risk"].to_numpy(), te["high_risk"].to_numpy()
    print(f"train events {len(tr)} ({int(ytr.sum())} high-risk)   test events {len(te)} ({int(yte.sum())} high-risk)")

    full_cols = [c for c in tr.select_dtypes("number").columns if c not in LEAKY + NO_TIMING + ["high_risk"]]
    full_cols = [c for c in full_cols if c in te.columns]

    # Sensitivity check: train only on 2-3 day alerts, the mix the ESA test set has.
    near = (tr["time_to_tca"] < 3.0).to_numpy()
    models = {
        "Logistic regression (2 feat)": (lr(), SHARED),
        "Random forest (2 feat)": (RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                   class_weight="balanced", random_state=SEED, n_jobs=-1), SHARED),
        "Gradient boosting (2 feat)": (HistGradientBoostingClassifier(class_weight="balanced",
                                       random_state=SEED), SHARED),
        "Gradient boosting (catalog feat)": (HistGradientBoostingClassifier(class_weight="balanced",
                                       random_state=SEED), CATALOG),
        "REFERENCE: gradient boosting (all feat)": (HistGradientBoostingClassifier(
                                       class_weight="balanced", random_state=SEED), full_cols),
    }

    rng = np.random.default_rng(SEED)
    print("\n=== INDEPENDENT ESA TEST SET (models trained on all of train_data) ===")
    print(f"{'model':50s} {'PR-AUC':>20s} {'ROC-AUC':>20s} {'false alarms removed @90% recall':>36s}")
    rows = [("Distance-only baseline", -te["miss_km"].to_numpy())]
    for name, (m, cols) in models.items():
        m.fit(tr[cols], ytr)
        rows.append((name, m.predict_proba(te[cols])[:, 1]))
    m = lr().fit(tr.loc[near, SHARED], ytr[near])
    rows.append(("Logistic regression (2 feat, 2-3 day alerts only)", m.predict_proba(te[SHARED])[:, 1]))
    m = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED).fit(tr.loc[near, SHARED], ytr[near])
    rows.append(("Gradient boosting (2 feat, 2-3 day alerts only)", m.predict_proba(te[SHARED])[:, 1]))
    saved = []
    for name, s in rows:
        p, lo, hi = with_ci(yte, s, rng)
        saved.append(record(name, p, lo, hi))
        print(f"{name:50s} {p[0]:.3f} [{lo[0]:.3f},{hi[0]:.3f}]  {p[1]:.3f} [{lo[1]:.3f},{hi[1]:.3f}]"
              f"   {p[2]:6.1%} [{lo[2]:.1%},{hi[2]:.1%}]")

    save_results("results/independent_test.json",
                 {"note": "Models trained on all ESA train events, scored on ESA's own test files",
                  "events": int(len(yte)), "high_risk": int(yte.sum()), "models": saved})


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: python src/crf/train_final.py train.csv test_data.csv test_data_private.csv")
    main(*sys.argv[1:])
