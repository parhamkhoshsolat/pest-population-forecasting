# Does synthetic oversampling help? Synthetic rows are added to TRAINING only; the test stays the real 49 trap-days.
import sys, json, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
df = pd.read_csv(sys.argv[1]); df['New catches'] = (df['New catches'] > 0).astype(int)
df['Date'] = pd.to_datetime(df['Date']); df = df.sort_values('Date').reset_index(drop=True)
feats = [c for c in df.columns if c not in ['Date', 'New catches', 'Number of insects', 'Recent_Activity']]
X = pd.get_dummies(df[feats], columns=['Location', 'Season'], drop_first=True).astype(float); y = df['New catches']
s = int(len(X) * 0.8); Xtr, Xte, ytr, yte = X.iloc[:s], X.iloc[s:], y.iloc[:s], y.iloc[s:]
res = []
for sname, sampler in [('no synthetic data', None), ('SMOTE (synthetic catch days)', SMOTE(random_state=0, k_neighbors=3)),
                       ('copying catch days', RandomOverSampler(random_state=0))]:
    for mname, mk in [('Random Forest', lambda: RandomForestClassifier(n_estimators=300, max_depth=10, min_samples_leaf=2, class_weight=None, random_state=42)),
                      ('Logistic regression', lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)))]:
        aucs = []
        for seed in range(10):
            Xs, ys = (Xtr, ytr) if sampler is None else type(sampler)(**{**sampler.get_params(), 'random_state': seed}).fit_resample(Xtr, ytr)
            m = mk();
            try: m.set_params(random_state=seed)
            except Exception: pass
            m.fit(Xs, ys); aucs.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
        res.append({"training data": sname, "model": mname, "real-test AUC mean": round(float(np.mean(aucs)), 3),
                    "min": round(float(np.min(aucs)), 3), "max": round(float(np.max(aucs)), 3), "train catch days (real)": int(ytr.sum())})
for r in res: print(json.dumps(r))
