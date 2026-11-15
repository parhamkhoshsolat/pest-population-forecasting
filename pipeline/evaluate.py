"""Corrected evaluation: leak-free features, a strict date split, training-only thresholds, baselines.

Run from the repository root:
    uv run --no-project --python 3.11 --with-requirements pipeline/requirements.txt \
        python pipeline/evaluate.py

Writes pipeline/results/results.json and pipeline/results/RESULTS.md. CPU only, a few seconds.

Design:
  * Features come from pipeline/features.py (see its docstring for each definition).
  * Split by calendar date: the last 10 dates (14 to 23 August 2024) are the test set and
    every earlier date is training. No date appears in both.
  * Fixed, conventional model settings. Nothing is tuned on the test set, and with 12
    training catch days there is too little data for a meaningful hyperparameter search.
  * Decision threshold: chosen from out-of-fold predictions on the training dates only
    (expanding-window folds by date). If the out-of-fold predictions contain no catch
    days, or no threshold gives an F1 above 0, the threshold is 0.5 and the output says so.
  * Baselines on the same test rows: yesterday's catch or count at the same trap, a catch at
    the same trap in the previous 7 days, predicting a catch every day, predicting 0, and
    the training mean.
  * Uncertainty: 95% intervals from resampling the 10 test dates with replacement.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEATURES, build_features, compare_with_original  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TEST_DAYS = 10
N_FOLDS = 4
MIN_TRAIN_DAYS = 15
THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.01), 2)
N_BOOT = 2000
SEEDS = list(range(10))


def classifiers(seed: int = 0) -> dict:
    return {
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000)),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced_subsample", random_state=seed, n_jobs=-1),
    }


def regressors(seed: int = 0) -> dict:
    return {
        "Ridge regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest": RandomForestRegressor(
            n_estimators=500, min_samples_leaf=2, max_features="sqrt", random_state=seed, n_jobs=-1),
    }


def date_folds(train_dates: np.ndarray):
    """Expanding-window folds over the training dates: train on earlier dates, predict the next block."""
    usable = train_dates[MIN_TRAIN_DAYS:]
    for block in np.array_split(usable, N_FOLDS):
        yield train_dates[train_dates < block[0]], block


def choose_threshold(make_model, train: pd.DataFrame) -> dict:
    dates = np.sort(train["Date"].unique())
    oof_y, oof_p = [], []
    for fit_dates, val_dates in date_folds(dates):
        fit, val = train[train["Date"].isin(fit_dates)], train[train["Date"].isin(val_dates)]
        if fit["catch"].nunique() < 2:
            continue
        m = make_model().fit(fit[FEATURES], fit["catch"])
        oof_y.extend(val["catch"].tolist())
        oof_p.extend(m.predict_proba(val[FEATURES])[:, 1].tolist())
    oof_y, oof_p = np.array(oof_y), np.array(oof_p)
    info = {"oof_rows": int(len(oof_y)), "oof_catch_rows": int(oof_y.sum())}
    if oof_y.sum() == 0:
        info.update(threshold=0.5, rule="no catch days in the out-of-fold predictions; default 0.5")
        return info
    f1s = np.array([f1_score(oof_y, (oof_p >= t).astype(int), zero_division=0) for t in THRESHOLD_GRID])
    if f1s.max() <= 0:
        info.update(threshold=0.5, rule="no threshold gave an out-of-fold F1 above 0; default 0.5")
        return info
    best = float(THRESHOLD_GRID[int(np.argmax(f1s))])
    info.update(threshold=best, rule="maximises out-of-fold F1 on the training dates",
                oof_f1_at_threshold=round(float(f1s.max()), 3),
                oof_auc=round(float(roc_auc_score(oof_y, oof_p)), 3) if oof_y.min() != oof_y.max() else None)
    return info


def boot_ci(test: pd.DataFrame, fn, rng: np.random.Generator) -> list | None:
    groups = [idx.to_numpy() for _, idx in test.groupby("Date").groups.items()]
    vals = []
    for _ in range(N_BOOT):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(groups), size=len(groups))])
        v = fn(test.loc[idx])
        if v is not None and np.isfinite(v):
            vals.append(v)
    if len(vals) < N_BOOT / 2:
        return None
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def _auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else None


def clf_metrics(y, pred, score=None) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {"f1": round(f1_score(y, pred, zero_division=0), 3),
           "precision": round(precision_score(y, pred, zero_division=0), 3),
           "recall": round(recall_score(y, pred, zero_division=0), 3),
           "tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)]}
    if score is not None and len(np.unique(score)) > 1:
        out["auc"] = round(float(roc_auc_score(y, score)), 3)
        out["average_precision"] = round(float(average_precision_score(y, score)), 3)
    else:
        out["auc"] = 0.5
        out["average_precision"] = round(float(np.mean(y)), 3)
    return out


def reg_metrics(y, pred) -> dict:
    return {"mae": round(float(mean_absolute_error(y, pred)), 3),
            "rmse": round(float(np.sqrt(mean_squared_error(y, pred))), 3),
            "r2": round(float(r2_score(y, pred)), 3)}


def find_data(name: str) -> str:
    """The CSVs sit at the repository root (GitHub) or in data/cleaned/ (Hugging Face Space)."""
    for folder in (ROOT, os.path.join(ROOT, "data", "cleaned")):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(name)


def main() -> None:
    merged = find_data("cleaned_merged_data.csv")
    engineered = find_data("cleaned_engineered_data.csv")
    df = build_features(merged)
    rng = np.random.default_rng(0)

    dates = np.sort(df["Date"].unique())
    test_dates, train_dates = dates[-TEST_DAYS:], dates[:-TEST_DAYS]
    train = df[df["Date"].isin(train_dates)].reset_index(drop=True)
    test = df[df["Date"].isin(test_dates)].reset_index(drop=True)
    assert set(train["Date"]).isdisjoint(set(test["Date"])), "a date is in both train and test"

    res = {
        "description": ("Corrected evaluation of the pest forecasting models: leak-free features, "
                        "strict date split, thresholds from training data only, simple baselines."),
        "data": {
            "rows": int(len(df)), "traps": int(df["Location"].nunique()), "dates": int(len(dates)),
            "catch_rows": int(df["catch"].sum()),
            "train": {"first_date": str(pd.Timestamp(train_dates[0]).date()),
                      "last_date": str(pd.Timestamp(train_dates[-1]).date()),
                      "dates": int(len(train_dates)), "rows": int(len(train)), "catch_rows": int(train["catch"].sum())},
            "test": {"first_date": str(pd.Timestamp(test_dates[0]).date()),
                     "last_date": str(pd.Timestamp(test_dates[-1]).date()),
                     "dates": int(len(test_dates)), "rows": int(len(test)), "catch_rows": int(test["catch"].sum())},
            "dates_in_both_train_and_test": 0,
        },
        "features": FEATURES,
        "rebuilt_feature_agreement_with_committed_csv": compare_with_original(df, engineered),
        "classification": {"models": {}, "baselines": {}},
        "regression": {"models": {}, "baselines": {}},
    }

    test_out = test[["Date", "Location", "catch", "count", "Insects_Lag1"]].copy()
    test_out["Date"] = test_out["Date"].dt.strftime("%Y-%m-%d")

    # ---------- classification: is there a new catch at this trap today? ----------
    ytr, yte = train["catch"], test["catch"]
    for name in classifiers():
        thr = choose_threshold(lambda: classifiers(0)[name], train)
        model = classifiers(0)[name].fit(train[FEATURES], ytr)
        proba = model.predict_proba(test[FEATURES])[:, 1]
        pred = (proba >= thr["threshold"]).astype(int)
        entry = {"threshold": thr, **clf_metrics(yte, pred, proba)}
        t = test.assign(_p=proba, _y=pred)
        entry["auc_95ci"] = boot_ci(t, lambda s: _auc(s["catch"], s["_p"]), rng)
        entry["f1_95ci"] = boot_ci(t, lambda s: f1_score(s["catch"], s["_y"], zero_division=0), rng)
        if name == "Random Forest":
            aucs = []
            for seed in SEEDS:
                m = classifiers(seed)[name].fit(train[FEATURES], ytr)
                aucs.append(roc_auc_score(yte, m.predict_proba(test[FEATURES])[:, 1]))
            entry["auc_over_10_seeds"] = [round(float(min(aucs)), 3), round(float(max(aucs)), 3)]
            imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda z: -z[1])[:5]
            entry["top_features"] = [[f, round(float(v), 3)] for f, v in imp]
        res["classification"]["models"][name] = entry
        test_out[f"{name} catch probability"] = np.round(proba, 4)

    b = res["classification"]["baselines"]
    b["Catch yesterday at the same trap"] = clf_metrics(yte, test["catch_yesterday"], test["catch_yesterday"])
    b["Catch at the same trap in the previous 7 days"] = clf_metrics(yte, test["Recent_Activity"], test["Recent_Activity"])
    b["Predict a catch every day"] = clf_metrics(yte, np.ones(len(yte), dtype=int))

    # ---------- regression: number of insects in the trap today ----------
    ytr_r, yte_r = train["count"], test["count"]
    for name in regressors():
        model = regressors(0)[name].fit(train[FEATURES], ytr_r)
        pred = np.clip(model.predict(test[FEATURES]), 0, None)
        entry = reg_metrics(yte_r, pred)
        t = test.assign(_p=pred)
        entry["mae_95ci"] = boot_ci(t, lambda s: mean_absolute_error(s["count"], s["_p"]), rng)
        res["regression"]["models"][name] = entry
        test_out[f"{name} predicted count"] = np.round(pred, 4)
    rb = res["regression"]["baselines"]
    rb["Yesterday's count at the same trap"] = reg_metrics(yte_r, test["Insects_Lag1"])
    rb["Yesterday's count at the same trap"]["mae_95ci"] = boot_ci(
        test, lambda s: mean_absolute_error(s["count"], s["Insects_Lag1"]), rng)
    rb["Predict 0"] = reg_metrics(yte_r, np.zeros(len(yte_r)))
    rb["Training mean"] = reg_metrics(yte_r, np.full(len(yte_r), ytr_r.mean()))

    # ---------- plain-language conclusion, computed rather than written by hand ----------
    rf_c = res["classification"]["models"]["Random Forest"]
    best_clf_base_f1 = max(v["f1"] for v in b.values())
    rf_r = res["regression"]["models"]["Random Forest"]
    lag_mae = rb["Yesterday's count at the same trap"]["mae"]
    checks = {
        "classifier_beats_best_baseline_f1": bool(rf_c["f1"] > best_clf_base_f1),
        "classifier_auc_ci_excludes_0_5": bool(rf_c["auc_95ci"] is not None and rf_c["auc_95ci"][0] > 0.5),
        "regressor_beats_yesterdays_count_mae": bool(rf_r["mae"] < lag_mae),
    }
    n_pos = res["data"]["test"]["catch_rows"]
    if not any(checks.values()):
        summary = ("Without the leaking feature, no model is better than the simple baselines on this data "
                   f"({TEST_DAYS} test dates, {n_pos} catch trap-days).")
    else:
        summary = ("Without the leaking feature, the Random Forest passes some checks against the baselines but "
                   f"not others; with {TEST_DAYS} test dates and {n_pos} catch trap-days the differences are small.")
    res["conclusion"] = {**checks, "summary": summary}

    res["test_predictions"] = test_out.rename(columns={"Insects_Lag1": "yesterday count"}).to_dict(orient="records")

    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(res, f, indent=2)
    with open(os.path.join(out_dir, "RESULTS.md"), "w") as f:
        f.write(markdown(res))
    print(markdown(res))


def markdown(res: dict) -> str:
    d = res["data"]
    lines = [
        "# Corrected results",
        "",
        "Produced by `pipeline/evaluate.py` (Python 3.11, scikit-learn 1.3.2, pandas 2.2.2, numpy 1.26.4).",
        "",
        f"Data: {d['rows']} trap-days ({d['traps']} traps x {d['dates']} dates), {d['catch_rows']} with a new catch.",
        f"Train: {d['train']['first_date']} to {d['train']['last_date']}, {d['train']['rows']} trap-days, "
        f"{d['train']['catch_rows']} with a catch. Test: {d['test']['first_date']} to {d['test']['last_date']}, "
        f"{d['test']['rows']} trap-days, {d['test']['catch_rows']} with a catch. No date is in both.",
        "",
        "## Will a trap record a new catch today?",
        "",
        "| model or baseline | threshold | F1 | precision | recall | AUC (95% CI) | TN FP FN TP |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in res["classification"]["models"].items():
        ci = m.get("auc_95ci")
        ci_s = f" ({ci[0]:.2f} to {ci[1]:.2f})" if ci else ""
        lines.append(f"| {name} | {m['threshold']['threshold']:.2f} | {m['f1']:.3f} | {m['precision']:.3f} | "
                     f"{m['recall']:.3f} | {m['auc']:.3f}{ci_s} | {' '.join(map(str, m['tn_fp_fn_tp']))} |")
    for name, m in res["classification"]["baselines"].items():
        lines.append(f"| baseline: {name} | | {m['f1']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
                     f"{m['auc']:.3f} | {' '.join(map(str, m['tn_fp_fn_tp']))} |")
    lines += ["", "Thresholds:"]
    for name, m in res["classification"]["models"].items():
        t = m["threshold"]
        lines.append(f"- {name}: {t['threshold']:.2f} ({t['rule']}; {t['oof_rows']} out-of-fold rows, "
                     f"{t['oof_catch_rows']} with a catch"
                     + (f"; out-of-fold AUC {t['oof_auc']:.3f}" if t.get("oof_auc") is not None else "") + ").")
    rf = res["classification"]["models"]["Random Forest"]
    lines.append(f"- Random Forest test AUC over 10 seeds: {rf['auc_over_10_seeds'][0]:.3f} to {rf['auc_over_10_seeds'][1]:.3f}.")
    lines += ["", "## How many insects are in the trap today?", "",
              "| model or baseline | MAE (95% CI) | RMSE | R² |", "|---|---|---|---|"]
    for group, prefix in [("models", ""), ("baselines", "baseline: ")]:
        for name, m in res["regression"][group].items():
            ci = m.get("mae_95ci")
            ci_s = f" ({ci[0]:.2f} to {ci[1]:.2f})" if ci else ""
            lines.append(f"| {prefix}{name} | {m['mae']:.3f}{ci_s} | {m['rmse']:.3f} | {m['r2']:.3f} |")
    c = res["conclusion"]
    lines += ["", "## Conclusion", "", c["summary"], "",
              f"- Random Forest classifier F1 above the best baseline F1: {'yes' if c['classifier_beats_best_baseline_f1'] else 'no'}.",
              f"- Random Forest classifier AUC interval above 0.5: {'yes' if c['classifier_auc_ci_excludes_0_5'] else 'no'}.",
              f"- Random Forest regressor MAE below yesterday's count: {'yes' if c['regressor_beats_yesterdays_count_mae'] else 'no'}.",
              "", "Rebuilt features against the committed `cleaned_engineered_data.csv` (share of rows equal):", ""]
    for k, v in res["rebuilt_feature_agreement_with_committed_csv"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
