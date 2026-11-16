"""Load the raw Criteo uplift CSV into DuckDB (14M rows) and create the train/test split."""
import time
import duckdb
from config import DB_PATH, RAW_CSV, TEST_PCT

if not RAW_CSV.exists():
    raise SystemExit(f"Missing {RAW_CSV}. Download with pipeline/00_download.py first.")

t = time.time()
con = duckdb.connect(str(DB_PATH))
con.execute(f"""
CREATE OR REPLACE TABLE uplift AS
SELECT CAST(row_number() OVER () - 1 AS BIGINT) AS user_id,
       CAST(f0 AS FLOAT) AS "f0", CAST(f1 AS FLOAT) AS "f1", CAST(f2 AS FLOAT) AS "f2", CAST(f3 AS FLOAT) AS "f3",
       CAST(f4 AS FLOAT) AS "f4", CAST(f5 AS FLOAT) AS "f5", CAST(f6 AS FLOAT) AS "f6", CAST(f7 AS FLOAT) AS "f7",
       CAST(f8 AS FLOAT) AS "f8", CAST(f9 AS FLOAT) AS "f9", CAST(f10 AS FLOAT) AS "f10", CAST(f11 AS FLOAT) AS "f11",
       CAST(treatment AS TINYINT) AS "treatment", CAST(exposure AS TINYINT) AS "exposure",
       CAST(visit AS TINYINT) AS "visit", CAST(conversion AS TINYINT) AS "conversion"
FROM read_csv('{RAW_CSV.as_posix()}', header=true, compression='gzip')
""")
con.execute(f"""
CREATE OR REPLACE TABLE split AS
SELECT user_id, CASE WHEN hash(user_id) % 100 < {TEST_PCT} THEN 'test' ELSE 'train' END AS part
FROM uplift
""")
print(con.sql("SELECT part, count(*) n FROM split GROUP BY 1").fetchall())
print(f"ingest done in {time.time()-t:.0f}s")
