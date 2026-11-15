# Re-analysis results (15 November 2026)

Environment: Python 3.11, scikit-learn 1.3.2, imbalanced-learn 0.11.0, pandas 2.2.2, numpy 1.26.4, CPU.
Input: `cleaned_engineered_data.csv` (245 trap-days, 21 with a catch).

## Leak check
- `Recent_Activity` = 1 on all 21 catch rows; on 7 of them there was no catch at that trap in the 7 days before, so the flag includes the same day's catch.
- `Insects_Lag1` equals the previous day's count at the same trap on 98% of rows: a genuine lag, not a leak.

## Classifier (`rerun_classifier.py`; Notebook 3 cells 9 and 12: same chronological 80/20 split, same grid, same threshold search)
| variant | threshold | F1 | precision | recall | AUC | TN FP FN TP |
|---|---|---|---|---|---|---|
| original features | 0.10 | 0.667 | 0.500 | 1.000 | 0.919 | 31 9 0 9 |
| `Recent_Activity` removed | 0.10 | 0.364 | 0.229 | 0.889 | 0.488 | 13 27 1 8 |
| `Recent_Activity` from the previous 7 days only | 0.10 | 0.391 | 0.243 | 1.000 | 0.465 | 12 28 0 9 |

Validation F1 was 0.0 at every threshold in all three runs (the validation window has no catch days). For reference, predicting "catch" on every test day gives F1 0.31.

## Regression (`rerun_regression.py`; Notebook 2 cells 29 and 31: train up to 13 August, test 14 to 23 August)
| variant | test MAE | test R² |
|---|---|---|
| Random Forest, original features | 0.355 | 0.391 |
| Random Forest, `Recent_Activity` removed | 0.496 | 0.300 |
| baseline: yesterday's count | 0.340 | 0.504 |
| baseline: predict 0 | 0.520 | |
| baseline: training mean | 0.611 | |

The notebook printed MAE 0.34 for the original features; the re-run gives 0.355 (the notebook's exact preprocessing for this table is not fully committed).

## Synthetic data (`synthetic_data_check.py`; synthetic rows added to training only, real test set, 10 seeds, `Recent_Activity` removed)
| training data | Random Forest AUC (mean, range) | Logistic regression AUC |
|---|---|---|
| real only (12 catch days) | 0.64 (0.60 to 0.69) | 0.44 |
| plus SMOTE synthetic catch days | 0.53 (0.49 to 0.55) | 0.40 |
| plus copied catch days | 0.42 (0.40 to 0.45) | 0.37 |

With 9 catch days in the test set, the standard error of an AUC is about 0.1, so 0.64 is not clearly better than chance.
