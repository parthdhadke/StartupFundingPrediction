import gc, math
import numpy as np, pandas as pd, tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# basic setup
PATH = "/Users/anuj/Desktop/duck pond/ML/output/cleaned_master.csv"
SEED = 42
BATCH = 64
EPOCHS = 250
LR = 0.001
STOP_PATIENCE = 15
MAX_LOG = 18.0

# helper functions
def fold_rare(s, n=10, other="__OTHER__"):
    v = s.value_counts()
    keep = v[v >= n].index
    return s.where(s.isin(keep), other)

def stage_score(x):
    d = {
        "pre seed":0,"seed":1,"series a":2,"series b":3,"series c":4,"series d":5,
        "series e":6,"series f":7,"series g":8,"series h":9,"series j":10,
        "private equity":11,"angel":1,"venture":4,"equity":2,"undisclosed":2
    }
    x = str(x).lower()
    return float(d[x] if x in d else 2.0)

def sector_type(x):
    x = str(x).lower()
    if "fin" in x: return "FinTech"
    if "health" in x or "med" in x or "bio" in x: return "HealthTech"
    if "ed" in x or "learn" in x or "edu" in x: return "EdTech"
    if "commerce" in x or "retail" in x: return "Ecommerce"
    if "ai" in x or "ml" in x or "data" in x or "analytics" in x: return "AI/Data"
    return "Other"

def top_city(x):
    hubs = ["bengaluru","mumbai","delhi","gurgaon","hyderabad","pune","noida"]
    c = str(x).lower()
    return 1.0 if any(h in c for h in hubs) else 0.0

def batch_gen(Xn, Xc, y, b):
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    for i in range(0, len(y), b):
        s = idx[i:i+b]
        yield Xn[s], [x[s] for x in Xc], y[s]

# load and clean
df = pd.read_csv(PATH)
df = df[df["amount_usd"].notna() & (df["amount_usd"] > 0)]
df = df[~df["amount_outlier_iqr"] & ~df["amount_outlier_logz"]].copy()
df["industry_vertical"].fillna("Unknown", inplace=True)
df["city_norm"].fillna("Unknown", inplace=True)
df["investment_stage_norm"].fillna("Unknown", inplace=True)
df["investor_count"].fillna(1, inplace=True)
df["year"].fillna(df["year"].median(), inplace=True)
df["month"].fillna(df["month"].median(), inplace=True)

# feature creation
df["TopHub"] = df["city_norm"].apply(top_city)
df["StageScore"] = df["investment_stage_norm"].apply(stage_score)
df["SectorType"] = df["industry_vertical"].apply(sector_type)

# fold rare categories
cats = ["industry_vertical","city_norm","investment_stage_norm","SectorType"]
for c in cats: df[c] = fold_rare(df[c])

# numeric block
nums = ["investor_count","year","month","TopHub","StageScore"]
num_data = df[nums].astype(np.float32)
meanv, stdv = num_data.mean(), num_data.std().replace(0,0.000001)
num_data = (num_data - meanv) / stdv

# target setup
y_usd = df["amount_usd"].astype(np.float32).values
y_log = np.log1p(y_usd)
y_mean, y_std = y_log.mean(), y_log.std() if y_log.std()>0.000001 else 1.0
y_stdv = (y_log - y_mean) / y_std

# vocab maps
def make_vocab(s):
    c = sorted(s.unique())
    to_i = {k:i for i,k in enumerate(c)}
    return to_i, c

vocab = {}
for c in cats:
    to_i, keys = make_vocab(df[c])
    vocab[c] = {"map":to_i, "size":len(keys)}

def to_ids(s, d): return s.map(d).astype(np.int32).values
cat_data = [to_ids(df[c], vocab[c]["map"]) for c in cats]

# split
N = len(df)
ids = np.arange(N)
rng = np.random.default_rng(SEED)
rng.shuffle(ids)
tsz = int(0.2 * N)
te, tr = ids[:tsz], ids[tsz:]

Xn_tr, Xn_te = num_data.iloc[tr].to_numpy(), num_data.iloc[te].to_numpy()
Xc_tr, Xc_te = [a[tr] for a in cat_data], [a[te] for a in cat_data]
y_tr, y_te = y_stdv[tr], y_stdv[te]
ylog_te, yusd_te = y_log[te], y_usd[te]

print("Train:", len(tr), "Test:", len(te))
print("Num dim:", Xn_tr.shape[1])
print("Cat sizes:", [v["size"] for v in vocab.values()])

# model parts
class CatBlock(tf.keras.layers.Layer):
    def __init__(self, vsizes, edims):
        super().__init__()
        self.layers_ = [tf.keras.layers.Embedding(v, e) for v, e in zip(vsizes, edims)]
    def call(self, x):
        out = [l(ids) for ids,l in zip(x, self.layers_)]
        return tf.concat(out, axis=-1)

class ResBlock(tf.keras.layers.Layer):
    def __init__(self, h=64, drop=0.2):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(h, activation="relu")
        self.fc2 = tf.keras.layers.Dense(h)
        self.proj = None
        self.act = tf.keras.layers.ReLU()
        self.drop = tf.keras.layers.Dropout(drop)
        self.h = h
    def build(self, shape):
        d = int(shape[-1])
        if d != self.h: self.proj = tf.keras.layers.Dense(self.h)
    def call(self, x, training=False):
        h = self.fc1(x)
        h = self.fc2(h)
        if self.proj is not None: x = self.proj(x)
        h = x + h
        h = self.act(h)
        return self.drop(h, training=training)

class FundingNet(tf.keras.Model):
    def __init__(self, n_dim, meta):
        super().__init__()
        self.num1 = tf.keras.layers.Dense(64, activation="relu")
        self.num2 = tf.keras.layers.Dense(32, activation="relu")
        vs = [m["size"] for m in meta]
        ed = [max(4, min(16, int(round((v**0.25)*2)))) for v in vs]
        self.cat = CatBlock(vs, ed)
        self.fuse = tf.keras.layers.Dense(128, activation="relu")
        self.res1 = ResBlock(64, 0.25)
        self.res2 = ResBlock(64, 0.25)
        self.out1 = tf.keras.layers.Dense(32, activation="relu")
        self.outf = tf.keras.layers.Dense(1)
    def call(self, x, training=False):
        xn, xc = x
        n = self.num1(xn); n = self.num2(n)
        c = self.cat(xc)
        z = tf.concat([n,c], axis=-1)
        z = self.fuse(z)
        z = self.res1(z, training=training)
        z = self.res2(z, training=training)
        z = self.out1(z)
        return self.outf(z)

# train setup
net = FundingNet(Xn_tr.shape[1], [vocab[c] for c in cats])
opt = tf.keras.optimizers.Adam(LR)
loss_fn = tf.keras.losses.MeanSquaredError()

cut = int(0.9 * len(Xn_tr))
Xn_tr_, Xn_va = Xn_tr[:cut], Xn_tr[cut:]
Xc_tr_, Xc_va = [a[:cut] for a in Xc_tr], [a[cut:] for a in Xc_tr]
y_tr_, y_va = y_tr[:cut], y_tr[cut:]

best, pat, best_w = math.inf, 0, net.get_weights()

# lists to store loss values for plotting
train_loss_list, val_loss_list = [], []

def val_loss():
    Xn = tf.convert_to_tensor(Xn_va, tf.float32)
    Xc = [tf.convert_to_tensor(a, tf.int32) for a in Xc_va]
    yv = tf.convert_to_tensor(y_va.reshape(-1,1), tf.float32)
    p = net((Xn,Xc), training=False)
    return loss_fn(yv,p).numpy()

for e in range(EPOCHS):
    bl = []
    for xn, xc, yb in batch_gen(Xn_tr_, Xc_tr_, y_tr_, BATCH):
        xn = tf.convert_to_tensor(xn, tf.float32)
        xc = [tf.convert_to_tensor(a, tf.int32) for a in xc]
        yb = tf.convert_to_tensor(yb.reshape(-1,1), tf.float32)
        with tf.GradientTape() as t:
            p = net((xn,xc), training=True)
            loss = loss_fn(yb,p)
        g = t.gradient(loss, net.trainable_variables)
        opt.apply_gradients(zip(g, net.trainable_variables))
        bl.append(loss.numpy())
    
    tr_loss = float(np.mean(bl))
    v_loss = val_loss()

    # store losses for graphs
    train_loss_list.append(tr_loss)
    val_loss_list.append(v_loss)

    print(f"Epoch {e:03d} | Train={tr_loss:.4f} | Val={v_loss:.4f}")
    if v_loss < best - 0.0001:
        best, pat, best_w = v_loss, 0, net.get_weights()
    else:
        pat += 1
        if pat >= STOP_PATIENCE:
            print("Stopped early.")
            break


net.set_weights(best_w)
gc.collect()

# evaluate
Xn_te_tf = tf.convert_to_tensor(Xn_te, tf.float32)
Xc_te_tf = [tf.convert_to_tensor(a, tf.int32) for a in Xc_te]
pred_std = net((Xn_te_tf, Xc_te_tf), training=False).numpy().reshape(-1)
pred_log = np.clip(pred_std * y_std + y_mean, 0, MAX_LOG)
pred_usd = np.expm1(pred_log)

r2l = r2_score(ylog_te, pred_log)
r2u = r2_score(yusd_te, pred_usd)
rml = mean_squared_error(ylog_te, pred_log)**0.5
rmu = mean_squared_error(yusd_te, pred_usd)**0.5
mal = mean_absolute_error(ylog_te, pred_log)
mau = mean_absolute_error(yusd_te, pred_usd)
print("R2 log:", round(r2l,4),"R2 usd:", round(r2u,4))
print("RMSE log:", round(rml,4),"RMSE usd:", round(rmu,2))
print("MAE log:", round(mal,4),"MAE usd:", round(mau,2))

# sector summary
df["Sector_folded"] = fold_rare(df["SectorType"])
sect_te = df["Sector_folded"].values[te]
rows=[]
for s in np.unique(sect_te):
    m = sect_te==s
    if m.sum()<5: continue
    r2s = r2_score(yusd_te[m], pred_usd[m])
    rms = mean_squared_error(yusd_te[m], pred_usd[m])**0.5
    mas = mean_absolute_error(yusd_te[m], pred_usd[m])
    rows.append([s,int(m.sum()),r2s,rms,mas])
res = pd.DataFrame(rows,columns=["Sector","N","R2_USD","RMSE_USD","MAE_USD"])
print(res)



# =========================================================
# 9. FINAL PUBLICATION-GRADE VISUALIZATIONS
# =========================================================
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline         
from sklearn.inspection import permutation_importance

# ---------- Build feature matrix (num + dummies) ----------
X_dum = pd.get_dummies(df[cats], prefix=cats, drop_first=False)
X_full_df = pd.concat([num_data.reset_index(drop=True), X_dum.reset_index(drop=True)], axis=1)
X_full = X_full_df.to_numpy()

Xtr_full, Xte_full = X_full[tr], X_full[te]
ytr_log, yte_log = y_log[tr], y_log[te]
ytr_usd, yte_usd = y_usd[tr], y_usd[te]

# ---------- Baseline 1: XGB-style (HistGradientBoostingRegressor) ----------
xgb = HistGradientBoostingRegressor(
    loss="squared_error",
    max_iter=400,
    learning_rate=0.05,
    max_depth=None,
    min_samples_leaf=20,
    random_state=SEED
)
xgb.fit(Xtr_full, ytr_log)
xgb_pred_log = np.clip(xgb.predict(Xte_full), 0, MAX_LOG)

# ---------- Baseline 2: Linear Regression ----------
lin = Pipeline([
    ("scaler", StandardScaler(with_mean=True, with_std=True)),
    ("lr", LinearRegression())
])
lin.fit(Xtr_full, ytr_log)
lin_pred_log = np.clip(lin.predict(Xte_full), 0, MAX_LOG)

# ---------- Metrics (R² log) ----------
r2_hybrid = r2_score(ylog_te, pred_log)
r2_xgb = r2_score(yte_log, xgb_pred_log)
r2_lin = r2_score(yte_log, lin_pred_log)

metrics_log = pd.DataFrame({
    "Model": ["HybridNN (Ours)", "XGB", "Linear Regression"],
    "R2_log": [r2_hybrid, r2_xgb, r2_lin]
})

# =========================================================
# 1️⃣ COMPARATIVE BAR CHART (R² log space)
# =========================================================
plt.figure(figsize=(6.5,5))
bars = plt.bar(
    metrics_log["Model"],
    metrics_log["R2_log"],
    color=["#1f77b4", "#2ca02c", "#ff7f0e"],  # Blue, Green, Orange
    alpha=0.85
)

for bar in bars:
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.005,
        f"{bar.get_height():.3f}",
        ha="center", va="bottom", fontsize=10, fontweight="bold"
    )

plt.ylabel("R² (Log Space)")
plt.xlabel("Model")
plt.title("Comparative R² (Log Space) Across Models")
plt.grid(True, axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("/Users/anuj/Desktop/duck pond/comp_logr2.png", dpi=300, bbox_inches="tight")

# =========================================================
# 2️⃣ SECTOR-WISE BAR CHART (HybridNN only, R² in log space)
# =========================================================
df["Sector_folded"] = fold_rare(df["SectorType"])
sect_te = df["Sector_folded"].values[te]
rows=[]
for s in np.unique(sect_te):
    m = sect_te==s
    if m.sum()<5: continue
    r2s = r2_score(ylog_te[m], pred_log[m])
    rows.append([s,int(m.sum()),r2s])
res_sect_log = pd.DataFrame(rows,columns=["Sector","N","R2_log"]).sort_values("R2_log", ascending=False)

plt.figure(figsize=(8,5))
plt.bar(res_sect_log["Sector"], res_sect_log["R2_log"], color="teal", alpha=0.8)
for i,v in enumerate(res_sect_log["R2_log"]):
    plt.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

plt.xticks(rotation=25, ha="right")
plt.xlabel("Sector")
plt.ylabel("R² (Log Space)")
plt.title("Sector-wise R² (Log Space) — HybridNN")
plt.grid(True, axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("/Users/anuj/Desktop/duck pond/sector_logr2.png", dpi=300, bbox_inches="tight")

# =========================================================
# 3️⃣ FEATURE IMPORTANCE (Permutation on XGB, log target)
# =========================================================
print("\nComputing permutation importances for XGB baseline (proxy for HybridNN interpretability)...")
pi = permutation_importance(
    xgb, Xte_full, yte_log,
    n_repeats=20,
    random_state=SEED,
    scoring="r2"
)
importances = pd.Series(pi.importances_mean, index=X_full_df.columns).sort_values(ascending=False)
topk = importances.head(15)

plt.figure(figsize=(8,6))
plt.barh(topk.index[::-1], topk.values[::-1], color="steelblue", alpha=0.85)
plt.xlabel("Permutation Importance (ΔR² on Log Target)")
plt.title("Top 15 Important Features — Proxy from XGB Model")
plt.grid(True, axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("/Users/anuj/Desktop/duck pond/feature_importance.png", dpi=300, bbox_inches="tight")

print("\n✅ Saved graphs:")
print("• Comparative R²: comp_logr2.png")
print("• Sector-wise (HybridNN): sector_logr2.png")
print("• Feature Importance: feature_importance.png")
