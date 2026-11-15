# corrected_results.py
"""Load and show the corrected results written by pipeline/evaluate.py.

The table builders are plain pandas so they can be checked without Streamlit.
"""
import json
import os

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
# pipeline/ sits next to this file (GitHub) or one level up from src/ (Hugging Face Space).
_CANDIDATES = [os.path.join(_HERE, "pipeline", "results", "results.json"),
               os.path.join(os.path.dirname(_HERE), "pipeline", "results", "results.json")]


def load_results(path=None):
    if path is None:
        path = next((p for p in _CANDIDATES if os.path.exists(p)), _CANDIDATES[0])
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _ci(ci):
    return f"{ci[0]:.2f} to {ci[1]:.2f}" if ci else ""


def classification_table(res):
    rows = []
    for name, m in res["classification"]["models"].items():
        rows.append({"Model or baseline": name, "Threshold": f"{m['threshold']['threshold']:.2f}",
                     "F1": m["f1"], "Precision": m["precision"], "Recall": m["recall"],
                     "AUC": m["auc"], "AUC 95% interval": _ci(m.get("auc_95ci")),
                     "TN FP FN TP": " ".join(map(str, m["tn_fp_fn_tp"]))})
    for name, m in res["classification"]["baselines"].items():
        rows.append({"Model or baseline": f"Baseline: {name}", "Threshold": "",
                     "F1": m["f1"], "Precision": m["precision"], "Recall": m["recall"],
                     "AUC": m["auc"], "AUC 95% interval": "",
                     "TN FP FN TP": " ".join(map(str, m["tn_fp_fn_tp"]))})
    return pd.DataFrame(rows)


def regression_table(res):
    rows = []
    for group, prefix in [("models", ""), ("baselines", "Baseline: ")]:
        for name, m in res["regression"][group].items():
            rows.append({"Model or baseline": prefix + name, "MAE": m["mae"],
                         "MAE 95% interval": _ci(m.get("mae_95ci")), "RMSE": m["rmse"], "R²": m["r2"]})
    return pd.DataFrame(rows)


def test_predictions(res):
    return pd.DataFrame(res["test_predictions"])


def headline(res):
    """Short facts for the overview, all read from the results file."""
    d = res["data"]
    rf_c = res["classification"]["models"]["Random Forest"]
    rf_r = res["regression"]["models"]["Random Forest"]
    clf_base = res["classification"]["baselines"]
    best_base_name = max(clf_base, key=lambda k: clf_base[k]["f1"])
    lag = res["regression"]["baselines"]["Yesterday's count at the same trap"]
    return {
        "test_window": f"{d['test']['first_date']} to {d['test']['last_date']}",
        "test_rows": d["test"]["rows"], "test_catch_rows": d["test"]["catch_rows"],
        "train_catch_rows": d["train"]["catch_rows"], "test_dates": d["test"]["dates"],
        "rf_auc": rf_c["auc"], "rf_auc_ci": _ci(rf_c.get("auc_95ci")), "rf_f1": rf_c["f1"],
        "rf_threshold": rf_c["threshold"]["threshold"],
        "best_baseline_name": best_base_name, "best_baseline_f1": clf_base[best_base_name]["f1"],
        "rf_mae": rf_r["mae"], "lag_mae": lag["mae"], "summary": res["conclusion"]["summary"],
    }


# ---------------------------------------------------------------- Streamlit views
def render_story(st, res):
    h = headline(res)
    st.markdown(f"""
**What we tried.** Predict, for each of five pheromone traps, whether a new insect will be caught on a given day
(classification) and how many insects the trap holds (regression), from local weather and the trap's own history.

**What we found.** {h['summary']} The headline scores in the original notebooks (for example a Random Forest
with recall 1.00 and AUC 0.92) are withdrawn: the feature `Recent_Activity` included the same day's catch, so the
models were reading the answer.

**The corrected numbers** (test window {h['test_window']}, {h['test_rows']} trap-days, {h['test_catch_rows']} with a catch):
- Random Forest classifier: AUC {h['rf_auc']:.3f} (95% interval {h['rf_auc_ci']}), F1 {h['rf_f1']:.3f} at a threshold
  of {h['rf_threshold']:.2f} chosen on training data only.
- Best simple baseline, "{h['best_baseline_name'].lower()}": F1 {h['best_baseline_f1']:.3f}.
- Random Forest regressor: mean absolute error {h['rf_mae']:.3f} insects per trap-day, against
  {h['lag_mae']:.3f} for simply predicting yesterday's count.

With {h['train_catch_rows']} catch days to learn from and one summer of data, there is not enough signal to
forecast catches. More traps or more seasons would be needed.
""")


def render_method(st, res):
    d = res["data"]
    st.markdown(f"""
- **Features** are rebuilt from `cleaned_merged_data.csv` by `pipeline/features.py`. Anything derived from the traps
  uses only earlier days: yesterday's and three-days-ago counts, a catch at the same trap in the previous 7 days,
  and days since the last trap service. Same-day weather is treated as known.
- **Split by date**: training {d['train']['first_date']} to {d['train']['last_date']} ({d['train']['rows']} trap-days,
  {d['train']['catch_rows']} with a catch), test {d['test']['first_date']} to {d['test']['last_date']}
  ({d['test']['rows']} trap-days, {d['test']['catch_rows']} with a catch). No date is in both.
- **Thresholds** come from out-of-fold predictions on the training dates; the test window is used once, at the end.
- **Baselines** are shown next to every model. Intervals come from resampling the {d['test']['dates']} test dates.
""")


def render_classification(st, px, res):
    st.dataframe(classification_table(res), hide_index=True, use_container_width=True)
    rf = res["classification"]["models"]["Random Forest"]
    thr = rf["threshold"]
    st.markdown(
        f"The Random Forest looked strong in out-of-fold checks on the training dates (AUC {thr.get('oof_auc', float('nan')):.2f}), "
        f"but on the later test window its AUC is {rf['auc']:.2f}, and across 10 random seeds it ranges from "
        f"{rf['auc_over_10_seeds'][0]:.2f} to {rf['auc_over_10_seeds'][1]:.2f}. An AUC of 0.5 means no ability to rank "
        "catch days above other days.")
    preds = test_predictions(res)
    preds["Outcome"] = preds["catch"].map({1: "New catch", 0: "No catch"})
    fig = px.strip(preds, x="Random Forest catch probability", y="Outcome", color="Outcome",
                   hover_data=["Date", "Location"],
                   title="Random Forest catch probability on each test trap-day")
    fig.add_vline(x=thr["threshold"], line_dash="dash",
                  annotation_text=f"threshold {thr['threshold']:.2f} (from training data)")
    fig.update_layout(showlegend=False, xaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)


def render_regression(st, px, res):
    st.dataframe(regression_table(res), hide_index=True, use_container_width=True)
    preds = test_predictions(res)
    daily = preds.groupby("Date")[["count", "Random Forest predicted count", "yesterday count"]].sum().reset_index()
    daily = daily.rename(columns={"count": "Actual", "Random Forest predicted count": "Random Forest",
                                  "yesterday count": "Baseline: yesterday's count"})
    fig = px.line(daily.melt(id_vars="Date", var_name="Series", value_name="Insects (sum of 5 traps)"),
                  x="Date", y="Insects (sum of 5 traps)", color="Series", markers=True,
                  title="Test window: actual counts against the Random Forest and the yesterday's-count baseline")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Errors in the table are per trap-day. The chart sums the five traps per day only for readability.")


def render_withdrawn(st):
    st.markdown("""
- **`Recent_Activity` leaked the target.** In `cleaned_engineered_data.csv` it is 1 on all 21 catch days, including
  catch days with no catch at that trap in the week before. It was the most important feature, and on its own it
  predicted the original test window better than the Random Forest.
- **The 0.10 threshold was not tuned.** The validation slice used to choose it had no catch days, so every threshold
  scored F1 0.0 and the search returned its first value.
- **Model choice and the LSTM and GRU settings used the test window**, which makes their test scores optimistic.
- **The original split was by row count**, so one date (14 August) was in both training and test.
- **ARIMAX, SARIMAX and Prophet were scored on the five-trap daily total** and cannot be compared with per-trap errors.
- **An earlier version of this dashboard showed a regression R² measured on the training rows and a classifier
  threshold tuned on the test set.** Both are removed.

The original notebooks are kept for the record, with a note at the top of each.
""")


def render_prediction_note(st):
    st.info(
        "The live prediction form has been removed. The models behind it were trained with the leaking "
        "`Recent_Activity` feature, and the corrected models do not beat simple baselines, so a prediction "
        "from them would not be useful advice for a grower.")
