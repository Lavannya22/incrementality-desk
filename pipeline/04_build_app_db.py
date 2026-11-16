"""Build data/app.duckdb: a few-MB database of pre-aggregated tables so the dashboard can be deployed
(GitHub limits files to 100 MB; the full DuckDB file is ~750 MB). All aggregates are computed by DuckDB over the full data."""
import duckdb
from config import APP_DB_PATH, DB_PATH, FEATURES

TOP_PCTS = [5, 10, 15, 20, 25, 30, 35, 40]
OUTCOMES = {"visit": "visit", "conversion": "conv"}

if APP_DB_PATH.exists():
    APP_DB_PATH.unlink()
con = duckdb.connect(str(APP_DB_PATH))
con.execute(f"ATTACH '{DB_PATH.as_posix()}' AS src (READ_ONLY)")

con.execute("""CREATE TABLE arm_cells AS SELECT treatment, exposure, count(*) AS n, sum(visit) AS visits, sum(conversion) AS conv
               FROM src.uplift GROUP BY ALL ORDER BY ALL""")
sel = ", ".join(f"min({f}) AS min_{f}, avg({f}) AS avg_{f}, max({f}) AS max_{f}, count(DISTINCT {f}) AS nd_{f}" for f in FEATURES)
r = con.execute(f"SELECT {sel} FROM src.uplift").fetchone()
con.execute("CREATE TABLE feature_summary (feature VARCHAR, min DOUBLE, mean DOUBLE, max DOUBLE, distinct_values BIGINT)")
for i, f in enumerate(FEATURES):
    con.execute("INSERT INTO feature_summary VALUES (?,?,?,?,?)", [f, r[4 * i], r[4 * i + 1], r[4 * i + 2], r[4 * i + 3]])
for t in ["balance", "convergence", "meta"]:
    con.execute(f"CREATE TABLE {t} AS SELECT * FROM src.{t}")

SEG = """thr AS (SELECT quantile_cont(u_{s}, {q}) AS hi FROM src.scored),
 base AS (SELECT quantile_cont(pc_{s}, 0.5) AS med FROM src.scored, thr WHERE u_{s} >= 0 AND u_{s} < thr.hi),
 seg AS (SELECT sc.*, CASE WHEN u_{s} < 0 THEN 'Predicted backfire' WHEN u_{s} >= thr.hi THEN 'Persuadables'
                           WHEN pc_{s} >= base.med THEN 'Sure things' ELSE 'Lost causes' END AS segment
         FROM src.scored sc, thr, base)"""
created = set()


def ins(t, sql):
    con.execute(f"INSERT INTO {t} {sql}" if t in created else f"CREATE TABLE {t} AS {sql}")
    created.add(t)


first = True
for o, s in OUTCOMES.items():
    ins("bins", f"""
        WITH r AS (SELECT *, ntile(100) OVER (ORDER BY u_{s} DESC, user_id) AS pct, hash(user_id) % 2 AS half FROM src.scored)
        SELECT '{o}' AS outcome, pct, half, count(*) AS n,
               count(*) FILTER (WHERE treatment=1) AS n_t, count(*) FILTER (WHERE treatment=0) AS n_c,
               sum({o}) FILTER (WHERE treatment=1) AS y_t, sum({o}) FILTER (WHERE treatment=0) AS y_c,
               sum(exposure) FILTER (WHERE treatment=1) AS exp_t, sum(u_{s}) AS sum_u, sum(pc_{s}) AS sum_pc
        FROM r GROUP BY pct, half ORDER BY pct, half""")
    ins("leaves", f"""
        SELECT '{o}' AS outcome, leaf_{s} AS leaf, hash(user_id) % 2 AS half, count(*) AS n,
               count(*) FILTER (WHERE treatment=1) AS n_t, count(*) FILTER (WHERE treatment=0) AS n_c,
               sum({o}) FILTER (WHERE treatment=1) AS y_t, sum({o}) FILTER (WHERE treatment=0) AS y_c,
               sum(exposure) FILTER (WHERE treatment=1) AS exp_t, sum(u_{s}) AS sum_u
        FROM src.scored GROUP BY leaf_{s}, hash(user_id) % 2 ORDER BY 2, 3""")
    z = ", ".join(f"avg({f}) FILTER (WHERE grp='top') AS top_{f}, avg({f}) FILTER (WHERE grp='bottom') AS bot_{f}, "
                  f"avg({f}) AS all_{f}, stddev({f}) AS sd_{f}" for f in FEATURES)
    pr = con.execute(f"""
        WITH q AS (SELECT quantile_cont(u_{s}, 0.2) lo, quantile_cont(u_{s}, 0.8) hi FROM src.scored),
        g AS (SELECT sc.*, CASE WHEN u_{s} >= q.hi THEN 'top' WHEN u_{s} <= q.lo THEN 'bottom' END AS grp FROM src.scored sc, q)
        SELECT {z} FROM g""").fetchone()
    if first:
        con.execute("CREATE TABLE profile (outcome VARCHAR, feature VARCHAR, top DOUBLE, bottom DOUBLE)")
    for i, f in enumerate(FEATURES):
        t, b, a, sd = pr[4 * i], pr[4 * i + 1], pr[4 * i + 2], pr[4 * i + 3]
        con.execute("INSERT INTO profile VALUES (?,?,?,?)", [o, f, (t - a) / sd, (b - a) / sd])
    for tp in TOP_PCTS:
        cte = SEG.format(s=s, q=1 - tp / 100)
        ins("seg_stats", f"""
            WITH {cte}
            SELECT '{o}' AS outcome, {tp} AS top_pct, segment, count(*) AS n,
                   count(*) FILTER (WHERE treatment=1) AS n_t, count(*) FILTER (WHERE treatment=0) AS n_c,
                   sum({o}) FILTER (WHERE treatment=1) AS y_t, sum({o}) FILTER (WHERE treatment=0) AS y_c,
                   sum(exposure) FILTER (WHERE treatment=1) AS exp_t, avg(u_{s}) AS pred_uplift, avg(pc_{s}) AS pred_base
            FROM seg GROUP BY segment""")
        ins("seg_thr", f"""
            WITH thr AS (SELECT quantile_cont(u_{s}, {1 - tp / 100}) AS hi FROM src.scored),
            base AS (SELECT quantile_cont(pc_{s}, 0.5) AS med FROM src.scored, thr WHERE u_{s} >= 0 AND u_{s} < thr.hi)
            SELECT '{o}' AS outcome, {tp} AS top_pct, hi, med FROM thr, base""")
    first = False

con.execute("""CREATE TABLE scatter_pool AS SELECT pc_visit, u_visit, pc_conv, u_conv
               FROM src.scored USING SAMPLE reservoir(12000 ROWS) REPEATABLE (7)""")
con.execute(f"""CREATE TABLE scored_sample AS SELECT user_id, treatment, exposure, visit, conversion, {', '.join(FEATURES)},
                pt_visit, pc_visit, u_visit, pt_conv, pc_conv, u_conv
                FROM src.scored USING SAMPLE reservoir(100000 ROWS) REPEATABLE (11)""")
con.execute("DETACH src")
con.close()
print("built", APP_DB_PATH, round(APP_DB_PATH.stat().st_size / 1e6, 1), "MB")
