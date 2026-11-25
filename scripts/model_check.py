"""Can scheduling variables predict patient wait time?

Run once (python scripts/model_check.py) to regenerate data/model_check.json,
which the dashboard displays. Kept offline so the web app starts quickly.
Uses a time-based split (train Jan-Sep, test Oct-Dec) to avoid leakage.
"""
import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = pathlib.Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "clinical_analytics.csv.gz")
df["t"] = pd.to_datetime(df["Check-In Time"], format="%Y-%m-%d %I:%M:%S %p")
df["Admit Source"] = df["Admit Source"].fillna("Not Identified")

X = pd.DataFrame(
    {
        "hour": df["t"].dt.hour,
        "day_of_week": df["t"].dt.day_of_week,
        "month": df["t"].dt.month,
        "clinic": df["Clinic Name"].astype("category"),
        "department": df["Department"].astype("category"),
        "admit_source": df["Admit Source"].astype("category"),
    }
)
y = df["Wait Time Min"]
train = df["t"].dt.month <= 9

model = HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=0)
model.fit(X[train], y[train])
pred = model.predict(X[~train])
actual = y[~train]

mae_model = float(np.abs(pred - actual).mean())
mae_base = float(np.abs(y[train].mean() - actual).mean())
ss_res = float(((actual - pred) ** 2).sum())
ss_tot = float(((actual - actual.mean()) ** 2).sum())

out = {
    "train_rows": int(train.sum()),
    "test_rows": int((~train).sum()),
    "mae_model": round(mae_model, 2),
    "mae_baseline": round(mae_base, 2),
    "r2_test": round(1 - ss_res / ss_tot, 4),
    "corr_wait_vs_care_score": round(float(df["Wait Time Min"].corr(df["Care Score"])), 2),
    "negative_wait_rows": int((df["Wait Time Min"] < 0).sum()),
}
(ROOT / "data" / "model_check.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
