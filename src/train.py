"""
Subway incident triage: predict, at the moment an incident is logged, whether it
will become a major delay (15+ minutes), and choose an alert threshold by cost.

Usage: python src/train.py <clean_delays_csv> <out_dir>
"""
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             precision_score, recall_score)
from sklearn.calibration import calibration_curve

SRC, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
MAJOR = 15          # minutes: threshold for a "major" delay
SEED = 7

# ------------------------------------------------------------------ data
df = pd.read_csv(SRC, parse_dates=["date"])
df = df[df["min_delay"] > 0].copy()                 # incidents that delayed a train
df["y"] = (df["min_delay"] >= MAJOR).astype(int)
df["dow"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["weekend"] = (df["dow"] >= 5).astype(int)
df["bound"] = df["bound"].fillna("NA").where(df["bound"].isin(["N", "S", "E", "W"]), "Other")
df["peak"] = df["hour"].isin([7, 8, 9, 16, 17, 18]).astype(int)

# Only information known when the incident is first logged. min_gap is excluded:
# it is measured after the fact and would leak the outcome.
CATS = ["code", "category", "station_clean", "line_clean", "bound"]
NUMS = ["hour", "dow", "month", "weekend", "peak"]

# Time-based split: never train on the future.
train = df[df["date"] < "2024-07-01"].copy()
valid = df[(df["date"] >= "2024-07-01") & (df["date"] < "2025-01-01")].copy()
test = df[df["date"] >= "2025-01-01"].copy()

# Rare categories -> "Other", learned from training data only.
def collapse(col, min_n):
    keep = set(train[col].value_counts()[lambda s: s >= min_n].index)
    for part in (train, valid, test):
        part[col] = part[col].where(part[col].isin(keep), "Other")
    return sorted(keep | {"Other"})

levels = {"code": collapse("code", 30), "station_clean": collapse("station_clean", 40),
          "category": collapse("category", 1), "line_clean": collapse("line_clean", 1),
          "bound": collapse("bound", 1)}
for part in (train, valid, test):
    for c in CATS:
        part[c] = pd.Categorical(part[c], categories=levels[c])

X = lambda d: d[CATS + NUMS]
ytr, yva, yte = train["y"], valid["y"], test["y"]

# ------------------------------------------------------------------ baselines
# 1) Code base rate: "how often has this delay code become major before?"
rate = train.groupby("code", observed=False)["y"].mean()
prior = ytr.mean()
base_code = lambda d: d["code"].map(rate).astype(float).fillna(prior).values

# 2) Logistic regression on one-hot features (small enough to run in a browser).
def onehot(d):
    o = pd.get_dummies(d[CATS], prefix=CATS, prefix_sep="=", dtype=float)
    o = pd.concat([o, pd.get_dummies(d["hour"].astype(str), prefix="hour", prefix_sep="=", dtype=float),
                   pd.get_dummies(d["dow"].astype(str), prefix="dow", prefix_sep="=", dtype=float)], axis=1)
    return o
oh_tr = onehot(train); cols = oh_tr.columns
oh = lambda d: onehot(d).reindex(columns=cols, fill_value=0.0)
logit = LogisticRegression(C=0.5, max_iter=3000)
logit.fit(oh_tr, ytr)

# ------------------------------------------------------------------ LightGBM
params = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=40,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              cat_smooth=20, verbose=-1, seed=SEED)
dtr = lgb.Dataset(X(train), ytr, categorical_feature=CATS)
dva = lgb.Dataset(X(valid), yva, categorical_feature=CATS, reference=dtr)
gbm = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                callbacks=[lgb.early_stopping(150, verbose=False)])

# ------------------------------------------------------------------ evaluation
def scores(y, p):
    return {"pr_auc": round(float(average_precision_score(y, p)), 3),
            "roc_auc": round(float(roc_auc_score(y, p)), 3),
            "brier": round(float(brier_score_loss(y, p)), 4)}

preds_test = {"Code base rate": base_code(test),
              "Logistic regression": logit.predict_proba(oh(test))[:, 1],
              "LightGBM": gbm.predict(X(test), num_iteration=gbm.best_iteration)}
results = {name: scores(yte, p) for name, p in preds_test.items()}
results["_prevalence_test"] = round(float(yte.mean()), 4)

# ------------------------------------------------------------------ decision layer
# Each alert triggers early recovery (e.g. stage shuttle buses / supervisors).
# Assumed relative costs: a false alarm costs 1 unit; a missed major incident costs 8
# (passenger-minutes lost while recovery starts late). Pick the threshold on the
# validation set, then report it on the untouched 2025 test set.
C_FP, C_FN = 1.0, 8.0
p_va = gbm.predict(X(valid), num_iteration=gbm.best_iteration)
grid = np.round(np.arange(0.02, 0.81, 0.01), 2)
cost = lambda y, p, t: float(((p >= t) & (y == 0)).sum() * C_FP + ((p < t) & (y == 1)).sum() * C_FN)
t_best = float(grid[np.argmin([cost(yva.values, p_va, t) for t in grid])])
p_te = preds_test["LightGBM"]
alerts = p_te >= t_best
days = test["date"].dt.normalize().nunique()
decision = {
    "cost_false_alarm": C_FP, "cost_missed_major": C_FN, "threshold": t_best,
    "precision": round(float(precision_score(yte, alerts)), 3),
    "recall": round(float(recall_score(yte, alerts)), 3),
    "alerts_per_day": round(float(alerts.sum() / days), 2),
    "majors_per_day": round(float(yte.sum() / days), 2),
    "cost_model": cost(yte.values, p_te, t_best),
    "cost_alert_never": cost(yte.values, p_te, 1.01),
    "cost_alert_always": cost(yte.values, p_te, -1),
    "cost_code_rule": cost(yte.values, preds_test["Code base rate"], t_best),
}
decision["saving_vs_best_simple_policy_pct"] = round(
    100 * (1 - decision["cost_model"] / min(decision["cost_alert_never"], decision["cost_alert_always"])), 1)

# Threshold sweep on test, for the interactive chart on the portfolio.
sweep = [{"t": float(t), "precision": round(float(precision_score(yte, p_te >= t, zero_division=0)), 3),
          "recall": round(float(recall_score(yte, p_te >= t)), 3),
          "alerts_per_day": round(float((p_te >= t).sum() / days), 2),
          "cost": cost(yte.values, p_te, t)} for t in grid]

frac_pos, mean_pred = calibration_curve(yte, p_te, n_bins=10, strategy="quantile")
calib = [{"pred": round(float(a), 3), "obs": round(float(b), 3)} for a, b in zip(mean_pred, frac_pos)]

# ------------------------------------------------------------------ explainability
sample = X(test).sample(4000, random_state=SEED)
sv = shap.TreeExplainer(gbm).shap_values(sample)
sv = sv[1] if isinstance(sv, list) else sv
imp = pd.Series(np.abs(sv).mean(0), index=sample.columns).sort_values(ascending=False)
importance = [{"feature": k, "mean_abs_shap": round(float(v), 4)} for k, v in imp.items()]

# ------------------------------------------------------------------ export browser model
coef = dict(zip(cols, logit.coef_[0].round(4)))
browser_model = {"intercept": round(float(logit.intercept_[0]), 4),
                 "coef": {k: float(v) for k, v in coef.items() if abs(v) >= 0.01},
                 "levels": {k: [str(x) for x in v] for k, v in levels.items()},
                 "metrics_test": results["Logistic regression"], "major_minutes": MAJOR}
json.dump(browser_model, open(OUT / "browser_model.json", "w"), separators=(",", ":"))
gbm.save_model(str(OUT / "lightgbm_model.txt"), num_iteration=gbm.best_iteration)

summary = {"rows": {"train": len(train), "valid": len(valid), "test": len(test)},
           "split": {"train": "2022-01 to 2024-06", "valid": "2024-07 to 2024-12", "test": "2025-01 to 2025-10"},
           "best_iteration": gbm.best_iteration, "models": results, "decision": decision,
           "calibration": calib, "importance": importance, "sweep": sweep}
json.dump(summary, open(OUT / "metrics.json", "w"), indent=1)
print(json.dumps({k: summary[k] for k in ["rows", "best_iteration", "models", "decision"]}, indent=1))
print(imp.round(4).to_string())
