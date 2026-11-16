from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "criteo-research-uplift-v2.1.csv.gz"
DB_PATH = ROOT / "data" / "criteo.duckdb"
FEATURES = [f"f{i}" for i in range(12)]
TEST_PCT = 30          # % of users held out (hash split on user_id)
TRAIN_TREATED_N = 2_500_000
SEED = 42
APP_DB_PATH = ROOT / "data" / "app.duckdb"   # slim, deployable (a few MB): pre-aggregated from the full DB
