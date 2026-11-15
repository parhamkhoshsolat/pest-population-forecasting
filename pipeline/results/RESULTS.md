# Corrected results

Produced by `pipeline/evaluate.py` (Python 3.11, scikit-learn 1.3.2, pandas 2.2.2, numpy 1.26.4).

Data: 245 trap-days (5 traps x 49 dates), 21 with a new catch.
Train: 2024-07-06 to 2024-08-13, 195 trap-days, 12 with a catch. Test: 2024-08-14 to 2024-08-23, 50 trap-days, 9 with a catch. No date is in both.

## Will a trap record a new catch today?

| model or baseline | threshold | F1 | precision | recall | AUC (95% CI) | TN FP FN TP |
|---|---|---|---|---|---|---|
| Logistic regression | 0.69 | 0.222 | 0.222 | 0.222 | 0.472 (0.28 to 0.66) | 34 7 7 2 |
| Random Forest | 0.36 | 0.000 | 0.000 | 0.000 | 0.593 (0.38 to 0.82) | 41 0 9 0 |
| baseline: Catch yesterday at the same trap | | 0.267 | 0.333 | 0.222 | 0.562 | 37 4 7 2 |
| baseline: Catch at the same trap in the previous 7 days | | 0.435 | 0.357 | 0.556 | 0.668 | 32 9 4 5 |
| baseline: Predict a catch every day | | 0.305 | 0.180 | 1.000 | 0.500 | 0 41 0 9 |

Thresholds:
- Logistic regression: 0.69 (maximises out-of-fold F1 on the training dates; 120 out-of-fold rows, 7 with a catch; out-of-fold AUC 0.636).
- Random Forest: 0.36 (maximises out-of-fold F1 on the training dates; 120 out-of-fold rows, 7 with a catch; out-of-fold AUC 0.915).
- Random Forest test AUC over 10 seeds: 0.545 to 0.629.

## How many insects are in the trap today?

| model or baseline | MAE (95% CI) | RMSE | R² |
|---|---|---|---|
| Ridge regression | 0.506 (0.27 to 0.77) | 0.875 | 0.424 |
| Random Forest | 0.525 (0.26 to 0.82) | 0.988 | 0.266 |
| baseline: Yesterday's count at the same trap | 0.340 (0.16 to 0.54) | 0.812 | 0.504 |
| baseline: Predict 0 | 0.520 | 1.265 | -0.203 |
| baseline: Training mean | 0.611 | 1.204 | -0.090 |

## Conclusion

Without the leaking feature, no model is better than the simple baselines on this data (10 test dates, 9 catch trap-days).

- Random Forest classifier F1 above the best baseline F1: no.
- Random Forest classifier AUC interval above 0.5: no.
- Random Forest regressor MAE below yesterday's count: no.

Rebuilt features against the committed `cleaned_engineered_data.csv` (share of rows equal):

- Temp_Range: 1.0
- Temp_Avg_3d: 1.0
- Humidity_Avg_3d: 1.0
- Insects_Lag1: 1.0
- Insects_Lag3: 1.0
- Recent_Activity: 0.918
- Days_Since_Cleaning: 1.0
- original_Recent_Activity_on_catch_days: 21 of 21
