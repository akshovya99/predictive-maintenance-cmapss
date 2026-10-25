"""Explainable and Drift-Aware Predictive Maintenance on NASA C-MAPSS.

Run from the repository root:  python src/cmapss_experiment.py
This is the plain-Python version of notebooks/CMAPSS_Predictive_Maintenance.ipynb.
Charts are saved to figures/.
"""
# %% [markdown]
# # Explainable and Drift-Aware Predictive Maintenance for IoT-Monitored Industrial Equipment
# **Dataset:** NASA C-MAPSS, a public benchmark of simulated turbofan engines run until failure. Each engine has 21 sensors recorded every operating cycle.
#
# **Why this dataset:** public transformer data with real failures does not exist, and C-MAPSS is the standard benchmark for predictive maintenance. The method (sensor features, early-warning model, explanations, drift checks) applies to any sensor-monitored asset, including smart grid equipment.
#
# **Questions**
# 1. Can we warn that an engine will fail within **30 cycles**, better than a simple rule or age-based maintenance?
# 2. Which sensors drive each alert (explainability)?
# 3. What happens when operating conditions change (drift), and how do we respond?

# %%
import matplotlib
matplotlib.use("Agg")   # save charts without opening windows
import numpy as np, pandas as pd, matplotlib.pyplot as plt, warnings, os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from scipy.stats import ks_2samp
from xgboost import XGBClassifier
import shap
warnings.filterwarnings("ignore")
FIG = "../figures" if os.path.isdir("../figures") else "figures"   # where charts are saved
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

BASE = "https://raw.githubusercontent.com/ericlrf/rul/main/CMAPSSData"
cols = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

def load(name, header_names=cols):
    # look for the file in data/, ../data/ or the current folder; otherwise download it from GitHub
    for folder in ["data", "../data", "."]:
        path = f"{folder}/{name}.txt"
        if os.path.exists(path):
            return pd.read_csv(path, sep=r"\s+", header=None, names=header_names)
    return pd.read_csv(f"{BASE}/{name}.txt", sep=r"\s+", header=None, names=header_names)

tr1, te1 = load("train_FD001"), load("test_FD001")
rul1 = load("RUL_FD001", ["RUL"])["RUL"].values
print("FD001 train:", tr1.shape, "engines:", tr1.unit.nunique(), "| test:", te1.shape, "engines:", te1.unit.nunique())
tr1.head()

# %% [markdown]
# ## 1. Data understanding
# Each row is one operating cycle of one engine: 3 operating settings (op1 to op3) and 21 sensors (s1 to s21). In the training set every engine runs until failure, so we can compute the **remaining useful life (RUL)** = cycles left before failure.

# %%
tr1["RUL"] = tr1.groupby("unit").cycle.transform("max") - tr1.cycle
last = te1.groupby("unit").cycle.transform("max")
te1["RUL"] = (last - te1.cycle) + te1.unit.map(dict(zip(sorted(te1.unit.unique()), rul1)))
life = tr1.groupby("unit").cycle.max()
print(f"Engine lifetime: min {life.min()}, median {int(life.median())}, max {life.max()} cycles")
sensor_cols = [f"s{i}" for i in range(1, 22)]
sensors = [s for s in sensor_cols if tr1[s].nunique() > 5]
print("Sensors that actually change (kept):", sensors)
print("Constant or near-constant sensors (dropped):", [s for s in sensor_cols if s not in sensors])

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharex=True)
for ax, s in zip(axes, ["s2", "s11", "s12"]):
    for u in [1, 2, 3, 4, 5]:
        d = tr1[tr1.unit == u]
        ax.plot(d.RUL, d[s], lw=0.8)
    ax.set_title(f"Sensor {s}"); ax.set_xlabel("Cycles before failure"); ax.invert_xaxis()
plt.tight_layout(); plt.savefig(f"{FIG}/01_sensor_degradation.png"); plt.show()

# %% [markdown]
# Sensors drift steadily as an engine approaches failure, which is the signal a predictive model can use.

# %% [markdown]
# ## 2. Target and features
# **Target:** `fail_30` = 1 if the engine will fail within the next 30 cycles. This is the early-warning window.
#
# **Features** (per engine, using only past cycles): normalised sensor value, 10-cycle rolling mean, 10-cycle change, and the cycle count.

# %%
H = 30
def fit_norm(df, regime):
    g = df[sensors].groupby(regime)
    return g.mean(), g.std().replace(0, 1)

def make_features(df, mu, sd, regime):
    z = (df[sensors].values - mu.loc[regime].values) / sd.loc[regime].values
    Z = pd.DataFrame(z, columns=sensors, index=df.index)
    Z["unit"] = df.unit.values
    g = Z.groupby("unit")[sensors]
    cur = Z[sensors].add_suffix("_now")
    m10 = g.rolling(10, min_periods=1).mean().reset_index(level=0, drop=True).reindex(Z.index).add_suffix("_m10")
    d10 = (Z[sensors] - g.shift(10)).fillna(0).add_suffix("_d10")
    out = pd.concat([cur, m10, d10], axis=1)
    out["cycle"] = df.cycle.values
    return out

reg1_tr = np.zeros(len(tr1), dtype=int); reg1_te = np.zeros(len(te1), dtype=int)
mu1, sd1 = fit_norm(tr1, reg1_tr)
X1 = make_features(tr1, mu1, sd1, reg1_tr)
y1 = (tr1.RUL <= H).astype(int)
print("Feature matrix:", X1.shape, "| positive rate:", f"{y1.mean():.1%}")

# %% [markdown]
# ## 3. Engine-wise split (no engine appears in two sets)
# Train 70 engines, validation 15 (to pick the alert threshold), test 15. Later we also check the official NASA test set.

# %%
rng = np.random.RandomState(0)
units = rng.permutation(tr1.unit.unique())
u_tr, u_va, u_te = units[:70], units[70:85], units[85:]
m_tr, m_va, m_te = tr1.unit.isin(u_tr).values, tr1.unit.isin(u_va).values, tr1.unit.isin(u_te).values
Xtr, ytr, Xva, yva, Xte, yte = X1[m_tr], y1[m_tr], X1[m_va], y1[m_va], X1[m_te], y1[m_te]
print(f"train {len(Xtr)} rows (alerts {ytr.sum()}) | val {len(Xva)} ({yva.sum()}) | test {len(Xte)} ({yte.sum()})")

# %% [markdown]
# ## 4. Models and baselines
# - **Age rule (scheduled maintenance):** alert purely by cycle count. This mimics fixed-schedule maintenance.
# - **Simple sensor rule:** one sensor (the one most correlated with failure on the training engines), smoothed over 10 cycles.
# - **Logistic Regression, Random Forest, XGBoost** with class weighting, since alerts are rare.

# %%
corr = {c: abs(np.corrcoef(Xtr[c], ytr)[0, 1]) for c in Xtr.columns if c.endswith("_m10")}
rule_col = max(corr, key=corr.get)
rule_sign = np.sign(np.corrcoef(Xtr[rule_col], ytr)[0, 1])
print("Simple rule uses:", rule_col, "(sign", int(rule_sign), ")")

pos_w = (ytr == 0).sum() / (ytr == 1).sum()
def new_models():
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000)),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5,
                                                class_weight="balanced_subsample", n_jobs=-1, random_state=0),
        "XGBoost": XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, scale_pos_weight=pos_w, eval_metric="aucpr",
                                 random_state=0, n_jobs=-1),
    }
models = new_models()
for m in models.values():
    m.fit(Xtr, ytr)

def score(name, X):
    if name == "Age rule (cycle count)": return X["cycle"].values
    if name == "Simple sensor rule": return rule_sign * X[rule_col].values
    return models[name].predict_proba(X)[:, 1]

names = ["Age rule (cycle count)", "Simple sensor rule"] + list(models)
rows = [["Random guess (alert rate)", yte.mean(), 0.5]]
for n in names:
    p = score(n, Xte)
    rows.append([n, average_precision_score(yte, p), roc_auc_score(yte, p)])
res = pd.DataFrame(rows, columns=["Model", "PR-AUC", "ROC-AUC"]).set_index("Model").round(3)
res

# %% [markdown]
# Only 15 test engines is a small sample, so we also run **5-fold cross-validation by engine** over all 100 engines for a more reliable comparison.

# %%
gkf = GroupKFold(n_splits=5)
cv = {n: [] for n in names}
for a, b in gkf.split(X1, y1, groups=tr1.unit):
    Xa, ya, Xb, yb = X1.iloc[a], y1.iloc[a], X1.iloc[b], y1.iloc[b]
    pw = (ya == 0).sum() / (ya == 1).sum()
    ms = new_models(); ms["XGBoost"].set_params(scale_pos_weight=pw)
    for m in ms.values(): m.fit(Xa, ya)
    for n in names:
        if n == "Age rule (cycle count)": p = Xb["cycle"].values
        elif n == "Simple sensor rule": p = rule_sign * Xb[rule_col].values
        else: p = ms[n].predict_proba(Xb)[:, 1]
        cv[n].append(average_precision_score(yb, p))
cvt = pd.DataFrame({n: [np.mean(v), np.std(v)] for n, v in cv.items()}, index=["PR-AUC mean", "PR-AUC std"]).T.round(3)
cvt

# %%
fig, ax = plt.subplots(figsize=(6, 4.2))
for n in names:
    p, r, _ = precision_recall_curve(yte, score(n, Xte))
    ax.plot(r, p, label=f"{n} ({average_precision_score(yte, score(n, Xte)):.2f})")
ax.axhline(yte.mean(), ls="--", color="grey", lw=1, label="Random guess")
ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision-recall, held-out engines")
ax.legend(fontsize=7); plt.tight_layout(); plt.savefig(f"{FIG}/02_precision_recall.png"); plt.show()

# %% [markdown]
# ## 5. Alerts and the cost of false alarms
# The alert threshold is chosen on the **validation engines** (best F2, which favours catching failures). Then we count alerts on the test engines.
#
# **Assumption (adjustable):** a missed warning costs 10 units, a false alarm costs 1 unit.

# %%
def pick_threshold(name):
    p, r, t = precision_recall_curve(yva, score(name, Xva))
    f2 = 5 * p[:-1] * r[:-1] / (4 * p[:-1] + r[:-1] + 1e-9)
    return t[np.nanargmax(f2)]

COST_FN, COST_FP = 10, 1
thr = {n: pick_threshold(n) for n in names}
out = []
for n in names:
    pred = (score(n, Xte) >= thr[n]).astype(int)
    tp = int(((pred == 1) & (yte == 1)).sum()); fp = int(((pred == 1) & (yte == 0)).sum()); fn = int(((pred == 0) & (yte == 1)).sum())
    out.append([n, round(tp / max(tp + fp, 1), 2), round(tp / max(tp + fn, 1), 2), fp, fn, COST_FN * fn + COST_FP * fp])
out.append(["No model (never alert)", np.nan, 0.0, 0, int(yte.sum()), COST_FN * int(yte.sum())])
alert = pd.DataFrame(out, columns=["Model", "Precision", "Recall", "False alarms", "Missed", "Cost"]).set_index("Model")
alert

# %% [markdown]
# **Engine view:** for each held-out engine, how many cycles before failure does the XGBoost model first raise an alert (3 cycles in a row above threshold)?

# %%
p_te = score("XGBoost", Xte)
d = tr1[m_te].assign(p=p_te)
lead = []
for u, g in d.groupby("unit"):
    flag = (g.p >= thr["XGBoost"]).rolling(3).sum().fillna(0) >= 3
    lead.append(int(g.RUL[flag].iloc[0]) if flag.any() else np.nan)
lead = pd.Series(lead, index=sorted(d.unit.unique()), name="RUL at first alert")
print("Cycles before failure at first alert (per engine):", lead.dropna().astype(int).tolist())
print(f"Median lead time: {lead.median():.0f} cycles | engines never alerted: {int(lead.isna().sum())} of {len(lead)}")

# %% [markdown]
# **Official NASA test set check:** the last recorded cycle of each of the 100 test engines, with the true RUL from NASA's answer file.

# %%
X1_te = make_features(te1, mu1, sd1, reg1_te)
last_idx = te1.groupby("unit").cycle.idxmax()
Xo, yo = X1_te.loc[last_idx], (te1.loc[last_idx, "RUL"] <= H).astype(int)
rows = [["Random guess (alert rate)", yo.mean(), 0.5, np.nan, np.nan]]
for n in names:
    p = score(n, Xo); pred = (p >= thr[n]).astype(int)
    tp = int(((pred == 1) & (yo == 1)).sum())
    rows.append([n, average_precision_score(yo, p), roc_auc_score(yo, p), tp / max(pred.sum(), 1), tp / yo.sum()])
pd.DataFrame(rows, columns=["Model", "PR-AUC", "ROC-AUC", "Precision", "Recall"]).set_index("Model").round(2)

# %% [markdown]
# ## 6. Explainability with SHAP (XGBoost)

# %%
explainer = shap.TreeExplainer(models["XGBoost"])
Xs = Xte.sample(1500, random_state=0)
sv = explainer.shap_values(Xs)
plt.figure()
shap.summary_plot(sv, Xs, plot_type="bar", max_display=10, show=False)
plt.title("Top features driving failure alerts"); plt.tight_layout()
plt.savefig(f"{FIG}/03_shap_importance.png", bbox_inches="tight"); plt.show()

# %%
plt.figure()
shap.summary_plot(sv, Xs, max_display=10, show=False)
plt.tight_layout(); plt.savefig(f"{FIG}/04_shap_beeswarm.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **One alert explained:** the highest-risk cycle among held-out engines.

# %%
i = int(np.argmax(p_te)); row = Xte.iloc[[i]]
sv1 = pd.Series(explainer.shap_values(row)[0], index=Xte.columns)
top = sv1.reindex(sv1.abs().sort_values(ascending=False).index).head(5)
print(f"Engine {int(tr1[m_te].unit.iloc[i])}, cycle {int(tr1[m_te].cycle.iloc[i])} | risk {p_te[i]:.2f} | true cycles to failure: {int(tr1[m_te].RUL.iloc[i])}")
pd.DataFrame({"value (normalised)": row[top.index].iloc[0].round(2), "pushes risk": top.round(2)})

# %% [markdown]
# ## 7. Drift awareness: new operating conditions (FD002)
# FD001 has one operating condition. **FD002 has six**, so sensor values shift with the flight regime even for a healthy engine. This is a realistic drift scenario: the model was trained in one setting and is deployed in another.
#
# **Step 1: detect drift.** Compare each sensor's distribution in FD001 and FD002 with the Kolmogorov-Smirnov statistic (0 = identical, 1 = completely different).

# %%
tr2 = load("train_FD002")
tr2["RUL"] = tr2.groupby("unit").cycle.transform("max") - tr2.cycle
y2 = (tr2.RUL <= H).astype(int)
ops = StandardScaler().fit_transform(tr2[["op1", "op2", "op3"]])
km = KMeans(n_clusters=6, n_init=10, random_state=0).fit(ops)
reg2 = km.labels_
print("FD002 engines:", tr2.unit.nunique(), "| rows:", len(tr2), "| operating regimes found:", np.bincount(reg2).tolist())

ks_raw = pd.Series({s: ks_2samp(tr1[s], tr2[s]).statistic for s in sensors}).sort_values(ascending=False)
print(f"Mean KS (raw sensors, FD001 vs FD002): {ks_raw.mean():.2f}")
ks_raw.head(8).round(2).to_frame("KS statistic")

# %% [markdown]
# **Step 2: measure the damage.** Apply the FD001-trained XGBoost model to FD002 engines it has never seen.
# - **Naive:** normalise with FD001 statistics (what a deployed model would do by default).
# - **Drift-aware:** detect the operating regime and normalise each regime separately.

# %%
reg0 = np.zeros(len(tr2), dtype=int)
X2_naive = make_features(tr2, mu1, sd1, reg0)
mu2, sd2 = fit_norm(tr2, reg2)
X2_aware = make_features(tr2, mu2, sd2, reg2)

ks_norm = pd.Series({s: ks_2samp(X1[f"{s}_now"], X2_aware[f"{s}_now"]).statistic for s in sensors})
print(f"Mean KS after regime normalisation: {ks_norm.mean():.2f} (was {ks_raw.mean():.2f})")

def evaluate(Xe, ye, name="XGBoost"):
    p = models[name].predict_proba(Xe)[:, 1]; pred = (p >= thr[name]).astype(int)
    tp = int(((pred == 1) & (ye == 1)).sum())
    return [average_precision_score(ye, p), roc_auc_score(ye, p), tp / max(pred.sum(), 1), tp / ye.sum()]

drift_tbl = pd.DataFrame(
    [["FD001 held-out engines (no drift)"] + evaluate(Xte, yte),
     ["FD002, naive (FD001 normalisation)"] + evaluate(X2_naive, y2),
     ["FD002, drift-aware (per-regime normalisation)"] + evaluate(X2_aware, y2)],
    columns=["Scenario", "PR-AUC", "ROC-AUC", "Precision", "Recall"]).set_index("Scenario").round(2)
drift_tbl

# %% [markdown]
# **Step 3: retrain on the new conditions.** Train a fresh model on 70% of FD002 engines (with regime normalisation) and compare it with the transferred FD001 model on the same held-out FD002 engines.

# %%
u2 = np.random.RandomState(1).permutation(tr2.unit.unique())
a2, b2 = tr2.unit.isin(u2[:182]).values, tr2.unit.isin(u2[182:]).values
retrained = XGBClassifier(n_estimators=250, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=(y2[a2] == 0).sum() / (y2[a2] == 1).sum(), eval_metric="aucpr",
                          random_state=0, n_jobs=-1).fit(X2_aware[a2], y2[a2])
cmp_rows = []
for label, mdl, Xe in [("FD001 model, naive", models["XGBoost"], X2_naive[b2]),
                       ("FD001 model, drift-aware features", models["XGBoost"], X2_aware[b2]),
                       ("Retrained on FD002, drift-aware features", retrained, X2_aware[b2])]:
    p = mdl.predict_proba(Xe)[:, 1]
    cmp_rows.append([label, average_precision_score(y2[b2], p), roc_auc_score(y2[b2], p)])
cmp = pd.DataFrame(cmp_rows, columns=["Approach", "PR-AUC", "ROC-AUC"]).set_index("Approach").round(2)
print(f"Held-out FD002 engines: {len(u2) - 182} | alert rate {y2[b2].mean():.1%}")
cmp

# %%
fig, ax = plt.subplots(figsize=(7, 3.4))
vals = [drift_tbl.iloc[0, 0], drift_tbl.iloc[1, 0], drift_tbl.iloc[2, 0]]
ax.bar(["No drift\n(FD001)", "Drift, naive\n(FD002)", "Drift, regime-\nnormalised"], vals, color=["#1F3A5F", "#C0392B", "#2A9D8F"])
for i, v in enumerate(vals): ax.text(i, v + 0.01, f"{v:.2f}", ha="center")
ax.set_ylabel("PR-AUC"); ax.set_ylim(0, 1); ax.set_title("Effect of changed operating conditions")
plt.tight_layout(); plt.savefig(f"{FIG}/05_drift_effect.png"); plt.show()

# %% [markdown]
# ## 8. Findings
# Write these up from the tables above (numbers change slightly if you change seeds or splits):
# 1. **Comparison:** how do the models compare with the age rule and the single-sensor rule (held-out engines and cross-validation)?
# 2. **Alerts:** precision, recall, false alarms and cost at the validation-chosen threshold, and the median warning lead time.
# 3. **Explainability:** which sensors dominate the SHAP ranking.
# 4. **Drift:** how much performance falls under new operating conditions, and how much per-regime normalisation or retraining recovers.
#
# **Limitations:** C-MAPSS is simulated turbofan data, not smart grid equipment; the 30-cycle window and the 10:1 cost ratio are assumptions; FD001 has only 100 engines; no LSTM was trained here.
