"""DuckDB access layer. Every number in the dashboard comes from a SQL query over the 14M-row table."""
import json
import sys
from pathlib import Path

import os

import duckdb
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from config import APP_DB_PATH, DB_PATH, FEATURES  # noqa: E402

# FULL: the 750 MB DuckDB with all 14M rows (local). Otherwise the dashboard runs on data/app.duckdb, a few-MB database of
# aggregates DuckDB computed over those 14M rows (this is what gets deployed). Force it locally: INCREMENTALITY_DB=slim

FULL = DB_PATH.exists() and os.environ.get("INCREMENTALITY_DB") != "slim"
SHORT = {"visit": "visit", "conversion": "conv"}


@st.cache_resource(show_spinner=False)
def _con():
    return duckdb.connect(str(DB_PATH if FULL else APP_DB_PATH), read_only=True)


@st.cache_data(show_spinner=False, max_entries=256)
def query(sql: str, params: tuple = ()) -> pd.DataFrame:
    return _con().cursor().execute(sql, list(params)).df()


# ---------------------------------------------------------------- full experiment (14M users)
@st.cache_data(show_spinner=False)
def arm_cells() -> pd.DataFrame:
    """treatment x exposure cells over the entire randomized experiment."""
    if not FULL:
        return query("SELECT treatment, exposure, n, visits, conv FROM arm_cells ORDER BY treatment, exposure")
    return query("""SELECT treatment, exposure, count(*) AS n, sum(visit) AS visits, sum(conversion) AS conv
                    FROM uplift GROUP BY ALL ORDER BY ALL""")


@st.cache_data(show_spinner=False)
def arms() -> dict:
    c = arm_cells()
    t, k = c[c.treatment == 1], c[c.treatment == 0]
    e, u = t[t.exposure == 1], t[t.exposure == 0]
    return dict(
        nt=int(t.n.sum()), nc=int(k.n.sum()),
        visit_t=int(t.visits.sum()), visit_c=int(k.visits.sum()),
        conversion_t=int(t.conv.sum()), conversion_c=int(k.conv.sum()),
        n_exposed=int(e.n.sum()), visit_exposed=int(e.visits.sum()), conversion_exposed=int(e.conv.sum()),
        n_unexposed=int(u.n.sum()), visit_unexposed=int(u.visits.sum()), conversion_unexposed=int(u.conv.sum()),
    )


@st.cache_data(show_spinner=False)
def balance() -> pd.DataFrame:
    return query("SELECT * FROM balance")


@st.cache_data(show_spinner=False)
def convergence() -> pd.DataFrame:
    return query("SELECT * FROM convergence ORDER BY sample_n")


@st.cache_data(show_spinner=False)
def feature_summary() -> pd.DataFrame:
    if not FULL:
        return query('SELECT feature, min, mean, max, distinct_values AS "distinct" FROM feature_summary')
    sel = ", ".join(f"min({f}) AS min_{f}, avg({f}) AS avg_{f}, max({f}) AS max_{f}, count(DISTINCT {f}) AS nd_{f}"
                    for f in FEATURES)
    r = query(f"SELECT {sel} FROM uplift").iloc[0]
    return pd.DataFrame([dict(feature=f, min=r[f"min_{f}"], mean=r[f"avg_{f}"], max=r[f"max_{f}"],
                              distinct=int(r[f"nd_{f}"])) for f in FEATURES])


# ---------------------------------------------------------------- hold-out scored users
@st.cache_data(show_spinner=False)
def uplift_bins(outcome: str) -> pd.DataFrame:
    """100 score-ranked bins (1 = highest predicted uplift) over the hold-out set, split into two random
    halves (used to choose a cut-off on one half and validate it on the other). Returns sums, not averages."""
    if not FULL:
        return query("SELECT pct, half, n, n_t, n_c, y_t, y_c, exp_t, sum_u, sum_pc FROM bins WHERE outcome = ? ORDER BY pct, half",
                     (outcome,))
    s = SHORT[outcome]
    return query(f"""
        WITH r AS (SELECT *, ntile(100) OVER (ORDER BY u_{s} DESC, user_id) AS pct,
                          hash(user_id) % 2 AS half FROM scored)
        SELECT pct, half, count(*) AS n,
               count(*) FILTER (WHERE treatment=1) AS n_t, count(*) FILTER (WHERE treatment=0) AS n_c,
               sum({outcome}) FILTER (WHERE treatment=1) AS y_t, sum({outcome}) FILTER (WHERE treatment=0) AS y_c,
               sum(exposure) FILTER (WHERE treatment=1) AS exp_t,
               sum(u_{s}) AS sum_u, sum(pc_{s}) AS sum_pc
        FROM r GROUP BY pct, half ORDER BY pct, half""")


_SEG_CTE = """
    thr AS (SELECT quantile_cont(u_{s}, {q}) AS hi FROM scored),
    base AS (SELECT quantile_cont(pc_{s}, 0.5) AS med FROM scored, thr WHERE u_{s} >= 0 AND u_{s} < thr.hi),
    seg AS (SELECT sc.*, CASE WHEN u_{s} < 0 THEN 'Predicted backfire'
                              WHEN u_{s} >= thr.hi THEN 'Persuadables'
                              WHEN pc_{s} >= base.med THEN 'Sure things'
                              ELSE 'Lost causes' END AS segment
            FROM scored sc, thr, base)"""


@st.cache_data(show_spinner=False)
def segments(outcome: str, top_pct: int) -> pd.DataFrame:
    """Four-way persona split from the model: persuadables / sure things / lost causes / sleeping dogs."""
    if not FULL:
        return query("""SELECT segment, n, n_t, n_c, y_t, y_c, exp_t, pred_uplift, pred_base
                        FROM seg_stats WHERE outcome = ? AND top_pct = ?""", (outcome, int(top_pct)))
    s = SHORT[outcome]
    cte = _SEG_CTE.format(s=s, q=1 - top_pct / 100)
    return query(f"""
        WITH {cte}
        SELECT segment, count(*) AS n,
               count(*) FILTER (WHERE treatment=1) AS n_t, count(*) FILTER (WHERE treatment=0) AS n_c,
               sum({outcome}) FILTER (WHERE treatment=1) AS y_t, sum({outcome}) FILTER (WHERE treatment=0) AS y_c,
               sum(exposure) FILTER (WHERE treatment=1) AS exp_t,
               avg(u_{s}) AS pred_uplift, avg(pc_{s}) AS pred_base
        FROM seg GROUP BY segment""")


@st.cache_data(show_spinner=False)
def scatter_sample(outcome: str, top_pct: int, n: int = 12000) -> pd.DataFrame:
    if not FULL:
        s_ = SHORT[outcome]
        pool = query("SELECT * FROM scatter_pool")
        thr = query("SELECT hi, med FROM seg_thr WHERE outcome = ? AND top_pct = ?", (outcome, int(top_pct))).iloc[0]
        u, pc = pool[f"u_{s_}"], pool[f"pc_{s_}"]
        seg = np.where(u < 0, "Predicted backfire", np.where(u >= thr.hi, "Persuadables",
                       np.where(pc >= thr.med, "Sure things", "Lost causes")))
        return pd.DataFrame({"base_rate": pc, "uplift": u, "segment": seg})
    s = SHORT[outcome]
    cte = _SEG_CTE.format(s=s, q=1 - top_pct / 100)
    return query(f"""
        WITH {cte}
        SELECT pc_{s} AS base_rate, u_{s} AS uplift, segment
        FROM (SELECT * FROM seg USING SAMPLE reservoir({n} ROWS) REPEATABLE (7))""")


@st.cache_data(show_spinner=False)
def rules_meta() -> dict:
    return json.loads(query("SELECT v FROM meta WHERE k='rules'").iloc[0, 0])


@st.cache_data(show_spinner=False)
def leaves(outcome: str) -> pd.DataFrame:
    """Rule-tree cohorts, split by random half so rules can be chosen on one half and validated on the other."""
    if not FULL:
        return query("SELECT leaf, half, n, n_t, n_c, y_t, y_c, exp_t, sum_u FROM leaves WHERE outcome = ? ORDER BY leaf, half",
                     (outcome,))
    s = SHORT[outcome]
    return query(f"""
        SELECT leaf_{s} AS leaf, hash(user_id) % 2 AS half,
               count(*) AS n, count(*) FILTER (WHERE treatment=1) AS n_t, count(*) FILTER (WHERE treatment=0) AS n_c,
               sum({outcome}) FILTER (WHERE treatment=1) AS y_t, sum({outcome}) FILTER (WHERE treatment=0) AS y_c,
               sum(exposure) FILTER (WHERE treatment=1) AS exp_t, sum(u_{s}) AS sum_u
        FROM scored GROUP BY 1, 2 ORDER BY 1, 2""")


@st.cache_data(show_spinner=False)
def profile_top_bottom(outcome: str, pct: int = 20) -> pd.DataFrame:
    """Standardized average feature value of the top vs bottom `pct`% by predicted uplift."""
    if not FULL:
        return query("SELECT feature, top, bottom FROM profile WHERE outcome = ? ORDER BY CAST(substr(feature, 2) AS INT)", (outcome,))
    s = SHORT[outcome]
    z = ", ".join(f"avg({f}) FILTER (WHERE grp='top') AS top_{f}, avg({f}) FILTER (WHERE grp='bottom') AS bot_{f}, "
                  f"avg({f}) AS all_{f}, stddev({f}) AS sd_{f}" for f in FEATURES)
    r = query(f"""
        WITH q AS (SELECT quantile_cont(u_{s}, {pct / 100}) lo, quantile_cont(u_{s}, {1 - pct / 100}) hi FROM scored),
        g AS (SELECT sc.*, CASE WHEN u_{s} >= q.hi THEN 'top' WHEN u_{s} <= q.lo THEN 'bottom' END AS grp
              FROM scored sc, q)
        SELECT {z} FROM g""").iloc[0]
    return pd.DataFrame([dict(feature=f, top=(r[f"top_{f}"] - r[f"all_{f}"]) / r[f"sd_{f}"],
                              bottom=(r[f"bot_{f}"] - r[f"all_{f}"]) / r[f"sd_{f}"]) for f in FEATURES])


def run_sql(sql: str, limit: int = 1000) -> pd.DataFrame:
    s = sql.strip().rstrip(";")
    low = s.lower()
    if not low.startswith(("select", "with", "describe", "show", "summarize")):
        raise ValueError("Only read-only SELECT / WITH / DESCRIBE statements are allowed.")
    q = f"SELECT * FROM ({s}) LIMIT {limit}" if low.startswith(("select", "with")) else s
    return _con().cursor().execute(q).df()
