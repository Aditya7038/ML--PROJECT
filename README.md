# Conjunction Risk Filter

ML endsem project, IIT Patna. A low-cost filter that ranks satellite collision alerts as "act on this" or "probably noise", aimed at small operators.

Team: Aditya Rayatuwar, Arvind Prajapat, Kushagra Singh. Guide: Dr. Sandeep Aashish.

## Data (not committed, download yourself)

- **ESA Collision Avoidance Challenge dataset** (Kelvins / Zenodo): real CDMs with risk outcomes. Used for training and testing. Put `train_data.csv` in `data/`.
- **CelesTrak SOCRATES raw CSV** (celestrak.org/SOCRATES): used only for the live demo. It has no outcome labels, so it is not validation. Open it in Notepad, not Excel, because Excel truncates the TCA timestamps.

## Method

1. Build one row per ESA event: the last CDM at least 2 days before closest approach, with the event's final risk as the label (high risk if risk >= -6; confirm both settings on the Kelvins page).
2. Use only features that SOCRATES also has: miss distance, relative speed, time to closest approach (converted to km, km/s).
3. Split by event, compare against a distance-only baseline, report PR-AUC and false alarms removed at 90% recall.

## Run

```
pip install -r requirements.txt
python src/crf/baseline.py data/train_data.csv
```

## Status and limits

- First result: the 3-feature model beats the distance-only baseline modestly. Needs cross-validation and confidence intervals before any claim.
- TLE-based data has no covariance, so we do not compute a true collision probability.
- SOCRATES demo output is illustrative, not validated.
