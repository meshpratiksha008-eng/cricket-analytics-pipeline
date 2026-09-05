from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

DB_USER = "postgres"
DB_PASS = "megha@23"
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "ipl_analytics"

connection_url = URL.create(
    drivername="postgresql+psycopg2",
    username=DB_USER,
    password=DB_PASS,
    host=DB_HOST,
    port=DB_PORT,
    database=DB_NAME,
)

engine = create_engine(connection_url)

sql_script = """
-- 1. Add match_number column to deliveries if missing
ALTER TABLE deliveries ADD COLUMN IF NOT EXISTS match_number INT;

-- 2. Extract the integer from the match_id text (e.g. 'match_15_...' -> 15)
UPDATE deliveries 
SET match_number = CAST(SPLIT_PART(match_id, '_', 2) AS INT)
WHERE match_number IS NULL;

-- 3. Drop existing views
DROP VIEW IF EXISTS view_batter_phase_stats CASCADE;
DROP VIEW IF EXISTS view_bowler_phase_stats CASCADE;

-- 4. Recreate batter view using match_number
CREATE VIEW view_batter_phase_stats AS
SELECT
    match_number,
    batter,
    phase,
    COUNT(ball) AS balls_faced,
    SUM(batter_runs) AS total_runs,
    COUNT(CASE WHEN batter_runs = 0 AND COALESCE(wides, 0) = 0 THEN 1 END) AS dot_balls,
    COUNT(CASE WHEN batter_runs = 4 THEN 1 END) AS fours,
    COUNT(CASE WHEN batter_runs = 6 THEN 1 END) AS sixes,
    ROUND(
        (SUM(batter_runs)::numeric / NULLIF(COUNT(CASE WHEN COALESCE(wides, 0) = 0 THEN 1 END), 0)) * 100, 
        2
    ) AS strike_rate
FROM deliveries
GROUP BY match_number, batter, phase;

-- 5. Recreate bowler view using match_number
CREATE VIEW view_bowler_phase_stats AS
SELECT
    match_number,
    bowler,
    phase,
    COUNT(CASE WHEN COALESCE(wides, 0) = 0 AND COALESCE(noballs, 0) = 0 THEN 1 END) AS legal_deliveries,
    SUM(batter_runs + COALESCE(wides, 0) + COALESCE(noballs, 0)) AS runs_conceded,
    COUNT(CASE WHEN is_wicket IS TRUE AND wicket_kind NOT IN ('run out', 'retired hurt', 'obstructing the field') THEN 1 END) AS wickets,
    COUNT(CASE WHEN total_runs = 0 THEN 1 END) AS dot_balls,
    ROUND(
        (SUM(batter_runs + COALESCE(wides, 0) + COALESCE(noballs, 0))::numeric / 
        NULLIF(COUNT(CASE WHEN COALESCE(wides, 0) = 0 AND COALESCE(noballs, 0) = 0 THEN 1 END) / 6.0, 0)), 
        2
    ) AS economy_rate
FROM deliveries
GROUP BY match_number, bowler, phase;
"""

def run_migration():
    with engine.begin() as conn:
        print("Extracting match_number and recreating analytical views...")
        conn.execute(text(sql_script))
        print("✓ All views updated with match_number successfully!")

if __name__ == "__main__":
    run_migration()