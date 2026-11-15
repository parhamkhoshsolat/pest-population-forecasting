# Pest Population Forecasting

A university team project that tried to predict insect catches from weather and trap data. A later re-check found that the headline result depended on a feature that leaked the answer; without it, no model beat simple baselines on the data available. Both the original pipeline and the re-analysis are in this repository.

<p>
  <a href="https://www.parhamkhoshsolat.com/projects/pest-forecasting"><img src="https://img.shields.io/badge/Write--up-portfolio-5dc6ff?style=flat-square&labelColor=0a1422" alt="Write-up: portfolio"></a>
  <img src="https://img.shields.io/badge/2025-team_of_3-5dc6ff?style=flat-square&labelColor=0a1422" alt="2025: team of 3">
</p>

## Overview

A dual-task pipeline that predicts insect catch counts (regression) and whether a catch happens at all (classification) from weather and trap data, with a Streamlit dashboard.

**Data.** This was the only data available for the course project: 5 traps over 49 days (6 July to 23 August 2024), 245 trap-days in total, of which 21 had a catch. The catches came in two bursts, mid to late July and 17 to 23 August, so the chronological split leaves the model one outbreak to learn from (12 catch days) and another to predict (9 catch days).

## What was built

- Features: lags, 3-day rolling means, recency and calendar features
- Regression (6 models): ARIMAX, SARIMAX, Prophet, Random Forest, XGBoost, LightGBM
- Classification (5 models): Random Forest, XGBoost, LightGBM, LSTM, GRU
- Validation: TimeSeriesSplit cross-validation and a chronological hold-out
- A Streamlit dashboard with the project overview and both model comparisons

## Results, corrected (November 2026)

The original notebooks report a Random Forest classifier with recall 1.00, F1 0.667 and AUC 0.92 on the hold-out. A re-check found that the feature `Recent_Activity` includes the same day's catch: it equals 1 on all 21 catch days, including 7 days with no catch at that trap in the week before. The models were reading the answer from it.

Re-running the same pipeline (same split, same search grid, same threshold procedure) without that feature:

| Classifier (49 test trap-days, 9 with a catch) | F1 | Recall | AUC |
|---|---|---|---|
| Original features (reproduces the notebook exactly) | 0.667 | 1.00 | 0.919 |
| `Recent_Activity` removed | 0.364 | 0.89 | 0.488 |
| `Recent_Activity` rebuilt from the previous 7 days only | 0.391 | 1.00 | 0.465 |

An AUC near 0.5 means the model cannot rank catch days above other days.

| Regression (50 test trap-days) | MAE |
|---|---|
| Random Forest, original features | 0.355 |
| Random Forest, `Recent_Activity` removed | 0.496 |
| Baseline: predict yesterday's count | 0.340 |
| Baseline: predict 0 | 0.520 |

Adding synthetic catch days to the training data (SMOTE, or copying catch days) made the classifier worse on the real test days, not better (Random Forest AUC 0.64 without them, 0.53 with SMOTE).

Other evaluation details worth knowing:
- The validation window used to choose the decision threshold (6 to 14 August) contains no catch days, so the threshold search returned its first value, 0.10.
- The split is by row count, so 14 August appears in both the training and the test set.
- The LSTM and GRU models used the test window for early stopping and threshold choice.

**Conclusion.** With 21 catch days from one summer, there is not enough signal to predict catches from weather. More traps or more seasons would be needed. The dashboard and notebooks still show the original numbers and should be read with this correction.

Scripts and full outputs: [`reanalysis/`](reanalysis/).

## Run it

```bash
git clone https://github.com/parhamkhoshsolat/pest-population-forecasting.git
cd pest-population-forecasting
pip install -r requirements.txt
streamlit run app.py
```

Re-analysis (Python 3.11):

```bash
pip install scikit-learn==1.3.2 imbalanced-learn==0.11.0 pandas==2.2.2 numpy==1.26.4
python reanalysis/rerun_classifier.py cleaned_engineered_data.csv
python reanalysis/rerun_regression.py cleaned_engineered_data.csv
python reanalysis/synthetic_data_check.py cleaned_engineered_data.csv
```

## Project layout

```
├── notebooks/
│   ├── Notebook_1_Data_Preprocessing_&_EDA.ipynb
│   ├── Notebook_2_Regression_Modeling.ipynb
│   └── Notebook_3_Classification_Modeling.ipynb
├── reanalysis/             # November 2026 re-check: scripts and results
├── models/                 # trained models and tournament results
├── utils_eda.py · utils_regression.py · utils_classification.py
├── app.py                  # Streamlit dashboard
├── cleaned_engineered_data.csv
└── requirements.txt
```

## Team & course

Information Systems and Business Intelligence, University of Naples Federico II.
- Parham Khosh Solat: model comparison design, model training and evaluation, the re-analysis
- Farshad Farahtaj: data preprocessing, exploratory analysis, feature engineering
- Giovanni Montanile: Power BI dashboards and reporting

## License

MIT. See [`LICENSE`](LICENSE).

---

Part of my portfolio: [parhamkhoshsolat.com](https://www.parhamkhoshsolat.com)
