# Re-run of Notebook 2 cells 29 + 31 (Random Forest regression), with and without Recent_Activity, plus a naive baseline.
import sys, json, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
df = pd.read_csv(sys.argv[1]); df['Date'] = pd.to_datetime(df['Date']); df = df.sort_values('Date').reset_index(drop=True)
tr, te = df[df['Date'] <= '2024-08-13'], df[df['Date'] >= '2024-08-14']
base = ['Location_Code','Average Temperature','Average Humidity','Temp_Range','Temp_Avg_3d','Humidity_Avg_3d',
        'Insects_Lag1','Insects_Lag3','Recent_Activity','Days_Since_Cleaning','Month','Day']
def run(cols, name):
    sc = StandardScaler().fit(tr[cols].values)
    gs = GridSearchCV(RandomForestRegressor(random_state=42), {'n_estimators':[100,200,300],'max_depth':[5,10,15,None],
         'min_samples_split':[2,5,10],'min_samples_leaf':[1,2,4]}, cv=TimeSeriesSplit(n_splits=3),
         scoring='neg_mean_absolute_error', n_jobs=-1).fit(sc.transform(tr[cols].values), tr['Number of insects'].values)
    p = gs.best_estimator_.predict(sc.transform(te[cols].values)); y = te['Number of insects'].values
    return {"variant": name, "n_train": len(tr), "n_test": len(te), "best_params": gs.best_params_,
            "test_mae": round(mean_absolute_error(y, p), 3), "test_r2": round(r2_score(y, p), 3)}
print(json.dumps(run(base, 'original')))
print(json.dumps(run([c for c in base if c != 'Recent_Activity'], 'dropped_Recent_Activity')))
y = te['Number of insects'].values
print(json.dumps({"variant": "naive: predict yesterday's count (Insects_Lag1)", "test_mae": round(mean_absolute_error(y, te['Insects_Lag1'].fillna(0).values), 3),
                  "test_r2": round(r2_score(y, te['Insects_Lag1'].fillna(0).values), 3)}))
print(json.dumps({"variant": "naive: predict 0", "test_mae": round(mean_absolute_error(y, np.zeros_like(y)), 3)}))
print(json.dumps({"variant": "naive: predict train mean", "test_mae": round(mean_absolute_error(y, np.full_like(y, tr['Number of insects'].mean(), dtype=float)), 3)}))
