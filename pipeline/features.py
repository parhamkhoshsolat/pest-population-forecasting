"""Rebuild the engineered features from cleaned_merged_data.csv without leakage.

Every feature for trap-day (trap, t) uses only:
  * the weather recorded for that trap on day t and earlier days, and
  * trap counts, catches and service events from days strictly before t.

Same-day weather is treated as known when the prediction is made (an observed
or forecast value). It carries no information about the trap itself.

Feature by feature:
  Average Temperature, Average Humidity   same-day weather, as recorded.
  Temp_Range                              same-day high minus low.
  Temp_Avg_3d, Humidity_Avg_3d            mean of days t-2, t-1, t (weather only).
  Insects_Lag1, Insects_Lag3              trap count on day t-1 and t-3 at the same
                                          trap. Before the first recorded day there
                                          is no history, so the value is 0.
  Recent_Activity                         1 if the same trap had a new catch on any
                                          of days t-7 .. t-1, else 0. The original
                                          column in cleaned_engineered_data.csv also
                                          counted day t itself, which leaked the
                                          target (it was 1 on every catch day).
  Days_Since_Cleaning                     days since the most recent service event
                                          ("Event" = 1 in cleaned_merged_data.csv)
                                          strictly before day t; 0 before the first
                                          event. This reproduces the original column
                                          exactly and uses no future information. The
                                          original code that defined it is not in the
                                          repository; reading "Event" as a trap
                                          service follows the original column name.
  Month, Day, Season                      calendar. Season is Mid_Summer up to
                                          30 July 2024 and Late_Summer from 31 July,
                                          matching the original column.
  Location_*                              one-hot trap identity.

Targets (never used as inputs for the same row):
  catch      1 if "New catches" > 0 on day t
  count      "Number of insects" on day t
"""
from __future__ import annotations

import pandas as pd

LATE_SUMMER_START = pd.Timestamp("2024-07-31")

FEATURES = [
    "Average Temperature", "Average Humidity", "Temp_Range",
    "Temp_Avg_3d", "Humidity_Avg_3d",
    "Insects_Lag1", "Insects_Lag3", "Recent_Activity", "Days_Since_Cleaning",
    "Month", "Day", "Season_Late_Summer",
    "Location_Cicalino 2", "Location_Imola 1", "Location_Imola 2", "Location_Imola 3",
]


def _days_since_event(dates: pd.Series, events: pd.Series) -> list[int]:
    out, last = [], None
    for d, e in zip(dates, events):
        out.append(0 if last is None else int((d - last).days))
        if e > 0:
            last = d  # takes effect from the next day on
    return out


def _recent_activity(dates: pd.Series, catches: pd.Series, window_days: int = 7) -> list[int]:
    catch_days = [d for d, c in zip(dates, catches) if c > 0]
    out = []
    for d in dates:
        lo = d - pd.Timedelta(days=window_days)
        out.append(int(any(lo <= c < d for c in catch_days)))
    return out


def build_features(merged_csv: str) -> pd.DataFrame:
    raw = pd.read_csv(merged_csv, parse_dates=["Date"])
    parts = []
    for loc, g in raw.sort_values(["Location", "Date"]).groupby("Location", sort=True):
        g = g.copy()
        g["Temp_Range"] = g["Temperature Range (High)"] - g["Temperature Range (Low)"]
        g["Temp_Avg_3d"] = g["Average Temperature"].rolling(3, min_periods=1).mean()
        g["Humidity_Avg_3d"] = g["Average Humidity"].rolling(3, min_periods=1).mean()
        g["Insects_Lag1"] = g["Number of insects"].shift(1).fillna(0)
        g["Insects_Lag3"] = g["Number of insects"].shift(3).fillna(0)
        g["Recent_Activity"] = _recent_activity(g["Date"], g["New catches"])
        g["Days_Since_Cleaning"] = _days_since_event(g["Date"], g["Event"])
        parts.append(g)
    df = pd.concat(parts)
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Season_Late_Summer"] = (df["Date"] >= LATE_SUMMER_START).astype(int)
    for loc in ["Cicalino 2", "Imola 1", "Imola 2", "Imola 3"]:
        df[f"Location_{loc}"] = (df["Location"] == loc).astype(int)
    df["catch"] = (df["New catches"] > 0).astype(int)
    df["count"] = df["Number of insects"]
    df["catch_yesterday"] = df.groupby("Location")["catch"].shift(1).fillna(0).astype(int)
    return df.sort_values(["Date", "Location"]).reset_index(drop=True)


def compare_with_original(df: pd.DataFrame, engineered_csv: str) -> dict:
    """Share of rows where each rebuilt feature equals the committed column."""
    orig = pd.read_csv(engineered_csv, parse_dates=["Date"])
    m = df.merge(orig, on=["Date", "Location"], suffixes=("", "_orig"))
    out = {}
    for c in ["Temp_Range", "Temp_Avg_3d", "Humidity_Avg_3d", "Insects_Lag1", "Insects_Lag3",
              "Recent_Activity", "Days_Since_Cleaning"]:
        out[c] = round(float(((m[c] - m[f"{c}_orig"]).abs() < 1e-6).mean()), 3)
    ra_on_catch = m.loc[m["catch"] == 1, "Recent_Activity_orig"]
    out["original_Recent_Activity_on_catch_days"] = f"{int(ra_on_catch.sum())} of {len(ra_on_catch)}"
    return out
