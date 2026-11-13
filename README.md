# Pest Population Forecasting

I ran a tournament of eight model families across regression and classification on weather and insect-trap data; Random Forest caught every catch day in the hold-out.

<p>
  <a href="https://parhamkhoshsolat-pest-prediction-dashboard.hf.space"><img src="https://img.shields.io/badge/Live_demo-Hugging_Face-5dc6ff?style=flat-square&labelColor=0a1422" alt="Live demo: Hugging Face"></a>
  <a href="https://www.parhamkhoshsolat.com/projects/pest-forecasting"><img src="https://img.shields.io/badge/Write--up-portfolio-5dc6ff?style=flat-square&labelColor=0a1422" alt="Write-up: portfolio"></a>
  <img src="https://img.shields.io/badge/2025-team_of_3-5dc6ff?style=flat-square&labelColor=0a1422" alt="2025: team of 3">
</p>

## Overview

A dual-task pipeline that predicts insect catch counts (regression) and whether a catch happens at all (classification) from weather and trap data, with a Streamlit dashboard that gives a risk score per site.

## What I did

I built the full pipeline.

- Data: 245 daily samples across 5 monitoring sites over 7 weeks
- Features: lags, 3-day rolling means, recency and calendar features
- Regression (6): ARIMAX, SARIMAX, Prophet, Random Forest, XGBoost, LightGBM
- Classification (5): Random Forest, XGBoost, LightGBM, LSTM, GRU
- Validation: TimeSeriesSplit cross-validation plus a chronological hold-out

## Results

Best classifier: Random Forest, AUC about 0.92 and recall 1.00 at precision 0.5 (F1 0.667) on a 49-day hold-out with 9 catch days; the target imbalance is 10.67:1. Out-of-sample regression is weak (test R² 0.35), and the deep models were close to chance. This is a small dataset and the classification threshold was tuned on the test window, so treat the scores as optimistic.

## Run it

```bash
git clone https://github.com/parhamkhoshsolat/pest-population-forecasting.git
cd pest-population-forecasting
pip install -r requirements.txt
streamlit run app.py
```

## Project layout

```
├── notebooks/
│   ├── Notebook_1_Data_Preprocessing_&_EDA.ipynb
│   ├── Notebook_2_Regression_Modeling.ipynb
│   └── Notebook_3_Classification_Modeling.ipynb
├── models/                 # trained models and tournament results
├── utils_eda.py · utils_regression.py · utils_classification.py
├── app.py                  # Streamlit dashboard
├── cleaned_engineered_data.csv
└── requirements.txt
```

## Team & course

Information Systems and Business Intelligence, University of Naples Federico II. Team: Parham Khosh Solat, Giovanni Montanile, Farshad Farahtaj.

## License

MIT. See [`LICENSE`](LICENSE).

---

Part of my portfolio: [parhamkhoshsolat.com](https://www.parhamkhoshsolat.com)
