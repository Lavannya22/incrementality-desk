"""T-learner uplift models (gradient boosting) for `visit` and `conversion`.
Train on the train split only; score the held-out test split; build a surrogate rule tree."""
import json, time, os
import duckdb, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeRegressor
from config import DB_PATH, FEATURES, TRAIN_TREATED_N, SEED

t0 = time.time()
con = duckdb.connect(str(DB_PATH))
cols = ", ".join(f"u.{f}" for f in FEATURES)

def fetch(treat, sample=None):
    q = f"""SELECT {cols}, u.visit, u.conversion FROM uplift u JOIN split s USING(user_id)
            WHERE s.part='train' AND u.treatment={treat}"""
    if sample:
        q += f" USING SAMPLE reservoir({sample} ROWS) REPEATABLE ({SEED})"
    return con.execute(q).df()

tr_c = fetch(0)
tr_t = fetch(1, TRAIN_TREATED_N)
print(f"train control {len(tr_c):,}  treated {len(tr_t):,}  ({time.time()-t0:.0f}s)", flush=True)

def make():
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=24, min_samples_leaf=400,
        l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=SEED)

models = {}
for y in ["visit", "conversion"]:
    for name, d in [("t", tr_t), ("c", tr_c)]:
        m = make().fit(d[FEATURES].to_numpy(np.float32), d[y].to_numpy())
        models[(y, name)] = m
        print(f"fit {y}/{name}: {m.n_iter_} iters ({time.time()-t0:.0f}s)", flush=True)

te = con.execute(f"""SELECT u.user_id, u.treatment, u.exposure, u.visit, u.conversion, {cols}
                     FROM uplift u JOIN split s USING(user_id) WHERE s.part='test'
                     ORDER BY u.user_id""").df()
X = te[FEATURES].to_numpy(np.float32)
for y, short in [("visit", "visit"), ("conversion", "conv")]:
    pt = models[(y, "t")].predict_proba(X)[:, 1]
    pc = models[(y, "c")].predict_proba(X)[:, 1]
    te[f"pt_{short}"] = pt.astype(np.float32)
    te[f"pc_{short}"] = pc.astype(np.float32)
    te[f"u_{short}"] = (pt - pc).astype(np.float32)
print(f"scored {len(te):,} test users ({time.time()-t0:.0f}s)", flush=True)

# --- surrogate rule trees: approximate the model's uplift score with readable rules ---
rules = {}
for short in ["visit", "conv"]:
    tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=100_000, random_state=SEED)
    tree.fit(X, te[f"u_{short}"].to_numpy())
    te[f"leaf_{short}"] = tree.apply(X).astype(np.int32)
    tr = tree.tree_
    def walk(node, conds):
        if tr.children_left[node] == -1:
            return {int(node): conds}
        f = FEATURES[tr.feature[node]]; thr = float(tr.threshold[node])
        out = {}
        out.update(walk(tr.children_left[node], conds + [f"{f} ≤ {thr:.3f}"]))
        out.update(walk(tr.children_right[node], conds + [f"{f} > {thr:.3f}"]))
        return out
    rules[short] = {str(k): v for k, v in walk(0, []).items()}
    fi = pd.Series(
        __import__("sklearn.inspection", fromlist=["x"]).permutation_importance(
            models[("visit" if short == "visit" else "conversion", "t")],
            X[:200_000], te["visit" if short == "visit" else "conversion"].to_numpy()[:200_000],
            n_repeats=2, random_state=SEED, scoring="neg_log_loss", n_jobs=-1).importances_mean,
        index=FEATURES)
    rules[f"importance_{short}"] = fi.round(6).to_dict()

con.register("te_df", te)
con.execute("CREATE OR REPLACE TABLE scored AS SELECT * FROM te_df")
con.execute("CREATE OR REPLACE TABLE meta (k VARCHAR, v VARCHAR)")
con.execute("INSERT INTO meta VALUES ('rules', ?)", [json.dumps(rules)])
print(f"done in {time.time()-t0:.0f}s", flush=True)
