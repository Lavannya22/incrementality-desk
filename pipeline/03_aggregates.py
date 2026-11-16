"""Pre-aggregate small tables the dashboard reads instantly:
 - balance      : covariate means by arm (randomization check)
 - convergence  : nested random subsamples -> how the estimate & CI tighten with sample size"""
import duckdb
from config import DB_PATH, FEATURES

con = duckdb.connect(str(DB_PATH))

aggs = ", ".join(f"avg({f}) AS m_{f}, var_samp({f}) AS v_{f}" for f in FEATURES)
df = con.execute(f"SELECT treatment, count(*) n, {aggs} FROM uplift GROUP BY 1 ORDER BY 1").df()
rows = []
for f in FEATURES:
    m0, m1 = df.loc[0, f"m_{f}"], df.loc[1, f"m_{f}"]
    v0, v1 = df.loc[0, f"v_{f}"], df.loc[1, f"v_{f}"]
    rows.append((f, float(m0), float(m1), float((m1 - m0) / ((v0 + v1) / 2) ** 0.5)))
con.execute("CREATE OR REPLACE TABLE balance (feature VARCHAR, mean_control DOUBLE, mean_treated DOUBLE, smd DOUBLE)")
con.executemany("INSERT INTO balance VALUES (?,?,?,?)", rows)

N = con.execute("SELECT count(*) FROM uplift").fetchone()[0]
con.execute("""CREATE OR REPLACE TABLE convergence (sample_n BIGINT, nt BIGINT, nc BIGINT,
               visit_t BIGINT, visit_c BIGINT, conv_t BIGINT, conv_c BIGINT)""")
for n in [10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000, N]:
    k = round(n / N * 1_000_000)
    r = con.execute(f"""SELECT count(*), sum(treatment), sum(treatment*visit), sum((1-treatment)*visit),
                               sum(treatment*conversion), sum((1-treatment)*conversion)
                        FROM uplift WHERE hash(user_id) % 1000000 < {k}""").fetchone()
    tot, nt, vt, vc, ct, cc = [int(x) for x in r]
    con.execute("INSERT INTO convergence VALUES (?,?,?,?,?,?,?)", [tot, nt, tot - nt, vt, vc, ct, cc])
print(con.sql("SELECT * FROM convergence").fetchall())
print(con.sql("SELECT * FROM balance").fetchall())
