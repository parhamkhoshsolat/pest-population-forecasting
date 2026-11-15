# Faithful re-run of Notebook 3 cell 9 + 12 (Random Forest), with three feature variants.
import sys, json, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, confusion_matrix, precision_score, recall_score
np.random.seed(42)
data = sys.argv[1]
raw = pd.read_csv(data)

# Leak check on the lag columns: Insects_Lag1 should be the previous day's count at the same trap.
r = raw.copy(); r['Date'] = pd.to_datetime(r['Date']); r = r.sort_values(['Location','Date'])
r['prev_count'] = r.groupby('Location')['Number of insects'].shift(1)
lag_ok = (r['Insects_Lag1'].fillna(-1) == r['prev_count'].fillna(-1)).mean()
same_day = (r['Insects_Lag1'] == r['Number of insects']).mean()
# Leak-free version of Recent_Activity: any new catch at the same trap in the 7 days BEFORE today.
def prior7(g):
    g = g.sort_values('Date'); out = []
    for i, row in g.iterrows():
        m = (g['Date'] < row['Date']) & (g['Date'] >= row['Date'] - pd.Timedelta(days=7)) & (g['New catches'] > 0)
        out.append((i, int(m.any())))
    return out
pairs = [p for _, g in r.groupby('Location') for p in prior7(g)]
fix = pd.Series(dict(pairs))
print(json.dumps({"lag1_equals_prev_day_count_share": round(float(lag_ok),3), "lag1_equals_same_day_count_share": round(float(same_day),3)}))

def run(variant):
    df = raw.copy()
    df['New catches'] = (df['New catches'] > 0).astype(int)
    df['Date'] = pd.to_datetime(df['Date'])
    if variant == 'prior7':
        df['Recent_Activity'] = fix.reindex(df.index).values
    df_sorted = df.sort_values('Date').reset_index(drop=True)
    exclude = ['Date', 'New catches', 'Number of insects'] + (['Recent_Activity'] if variant == 'dropped' else [])
    feats = [c for c in df_sorted.columns if c not in exclude]
    X, y = df_sorted[feats], df_sorted['New catches']
    s = int(len(X) * 0.8)
    Xtr, Xte, ytr, yte = X.iloc[:s], X.iloc[s:], y.iloc[:s], y.iloc[s:]
    cats = Xtr.select_dtypes(include=['object']).columns.tolist()
    Xtr = pd.get_dummies(Xtr, columns=cats, drop_first=True); Xte = pd.get_dummies(Xte, columns=cats, drop_first=True)
    for c in set(Xtr.columns) - set(Xte.columns): Xte[c] = 0
    for c in set(Xte.columns) - set(Xtr.columns): Xtr[c] = 0
    Xte = Xte[Xtr.columns]
    grid = {'n_estimators': [100, 200, 300], 'max_depth': [10, 15, 20, None], 'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [2, 4, 6], 'max_features': ['sqrt', 'log2']}
    gs = GridSearchCV(RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1), grid,
                      cv=TimeSeriesSplit(n_splits=3), scoring='f1', n_jobs=-1).fit(Xtr, ytr)
    sp = int(len(Xtr) * 0.8)
    tm = RandomForestClassifier(**gs.best_params_, class_weight='balanced', random_state=42, n_jobs=-1).fit(Xtr.iloc[:sp], ytr.iloc[:sp])
    pv = tm.predict_proba(Xtr.iloc[sp:])[:, 1]
    ths = np.arange(0.1, 0.9, 0.01); f1s = []
    for t in ths:
        p = (pv >= t).astype(int)
        f1s.append(f1_score(ytr.iloc[sp:], p) if len(np.unique(p)) > 1 else 0.0)
    th = ths[int(np.argmax(f1s))]
    fm = RandomForestClassifier(**gs.best_params_, class_weight='balanced', random_state=42, n_jobs=-1).fit(Xtr, ytr)
    pt = fm.predict_proba(Xte)[:, 1]; pr = (pt >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pr, labels=[0, 1]).ravel()
    imp = sorted(zip(Xtr.columns, fm.feature_importances_), key=lambda z: -z[1])[:3]
    # simple baselines on the same test window
    base_ra = None
    if 'Recent_Activity' in Xte.columns:
        b = Xte['Recent_Activity'].astype(int).values
        base_ra = round(f1_score(yte, b), 3)
    return {"variant": variant, "n_features": Xtr.shape[1], "best_params": gs.best_params_, "threshold": round(float(th), 2),
            "val_f1_at_threshold": round(float(max(f1s)), 3), "test_f1": round(f1_score(yte, pr), 3),
            "test_precision": round(precision_score(yte, pr, zero_division=0), 3), "test_recall": round(recall_score(yte, pr), 3),
            "test_auc": round(roc_auc_score(yte, pt), 3), "test_accuracy": round(accuracy_score(yte, pr), 3),
            "tn_fp_fn_tp": [int(tn), int(fp), int(fn), int(tp)], "top_features": [(a, round(float(b), 3)) for a, b in imp],
            "f1_of_rule_RecentActivity_eq_1": base_ra, "test_positives": int(yte.sum()), "test_n": int(len(yte))}

for v in ['original', 'dropped', 'prior7']:
    print(json.dumps(run(v)))
