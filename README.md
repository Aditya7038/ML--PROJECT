# Conjunction Risk Filter

ML endsem project, IIT Patna. A low-cost triage tool that ranks satellite collision alerts as "act", "check data first" or "probably noise", aimed at small operators.

Team: Aditya Rayatuwar, Arvind Prajapat, Kushagra Singh. Guide: Dr. Sandeep Aashish.

## Data (not committed, download yourself)

- **ESA Collision Avoidance Challenge** (Kelvins / Zenodo): real CDMs with risk outcomes. Used for validation. Needs `train_data.csv`, `test_data.csv`, `test_data_private.csv`.
- **CelesTrak SOCRATES CSV** (celestrak.org/SOCRATES): used for the live demo only. It has no outcome labels, so it is not validation. Open it in Notepad, not Excel, which truncates timestamps. A small sample is in `sample/`.

## Validation (src/crf/evaluate.py, src/crf/train_final.py)

One row per ESA event: last CDM at least 2 days before closest approach; label = final risk >= -6 (confirm both on the Kelvins page). Metrics: PR-AUC and share of false alarms removed at 90% recall, with 95% bootstrap intervals. Cross-validation is by event; the independent test uses ESA's own test files (2,167 events, 150 high-risk).

Independent ESA test result:

| Model | PR-AUC | False alarms removed @90% recall |
|---|---|---|
| Distance-only rule | 0.31 | 60% |
| Logistic regression, miss distance + speed | 0.25 | 61% |
| Gradient boosting, free catalog features | 0.24 | 45% |
| **Gradient boosting, all ESA features (covariance, tracking quality)** | **0.47** | **73%** |

What this says, honestly:

- On the free fields SOCRATES has, ML did **not** beat the plain miss-distance rule. So the demo ranks by miss distance and does not claim ML.
- Covariance and tracking-quality data give a clear gain. That is the case for operators using CDMs (which carry covariance) instead of TLE screening.
- `time_to_tca` is excluded on purpose. In ESA train data, earlier alerts look riskier, but ESA's test data has almost only 2-3 day alerts and the pattern disappears. Including it made models worse on the test set.
- Mismatch between train (2.7% high-risk) and test (6.9%) means absolute numbers differ between cross-validation and the test; compare models within one table.
- We looked at test results before dropping `time_to_tca`. The decision rests on the train/test mismatch above, but the test is no longer fully untouched.

## Software layer

```
CelesTrak SOCRATES --(every 6 h)--> FastAPI backend --/alerts--> Streamlit dashboard
ESA data --> evaluate.py, train_final.py --> results/*.json --/models--> backend
```

- **Backend** (`src/crf/api.py`, logic in `service.py`): downloads the latest SOCRATES file on startup and every `CRF_REFRESH_HOURS` (default 6; CelesTrak publishes about 3 times a day), keeps the last good copy if a download fails, ranks and flags alerts, and serves validation results. Endpoints: `GET /alerts`, `GET /health`, `GET /models`, `POST /refresh`. Swagger at `/docs`.
- **Ranking** (`src/crf/socrates.py`): **act** (close and orbit data fresh), **check data first** (close but one orbit file is stale), **probably noise**. Ranked by miss distance, since nothing beat it on free fields. The close and stale cut-offs are slider assumptions. MAX_PROB and DILUTION are shown for comparison.
- **Dashboard** (`dashboard/app.py`): Alerts tab, Models and accuracy tab (reads `results/`), How it works tab. Falls back to the bundled sample if the backend is off.

Settings (environment variables): `CRF_SOCRATES_URL` (default is a best guess, verify by opening it in a browser), `CRF_REFRESH_HOURS`, `CRF_AUTO_REFRESH=0` to stop downloads, `CRF_SOCRATES_CSV` to pin a local file, `CRF_API_URL` for the dashboard.

```
pip install -r requirements.txt
uvicorn crf.api:app --app-dir src --reload     # terminal 1, from the project folder
streamlit run dashboard\app.py                 # terminal 2
python src/crf/evaluate.py data/train_data.csv  # writes results/cross_validation.json
python src/crf/train_final.py data/train_data.csv data/test_data.csv data/test_data_private.csv  # writes results/independent_test.json
```

Tested: the scoring, download, caching and failure handling (in `service.py`) on a real SOCRATES file. The web wrappers `api.py` and `dashboard/app.py` could not be run by the author's tooling (no package access), so check them on your machine. The live CelesTrak download URL has not been tested from here.

## Limits

- TLE-based data has no covariance, so we do not compute a true collision probability.
- SOCRATES demo output is illustrative, not validated.
- The stale-data flag is a heuristic.
