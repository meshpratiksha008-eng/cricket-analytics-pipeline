import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

# Database Connection Configuration
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

def test_views():
    print("\n--- TOP DEATH-OVER BATTERS (Min 30 Runs) ---")
    query_batters = """
        SELECT batter, balls_faced, total_runs, strike_rate, sixes
        FROM view_batter_phase_stats
        WHERE phase = 'death' AND total_runs >= 30
        ORDER BY strike_rate DESC
        LIMIT 5;
    """
    print(pd.read_sql(query_batters, engine))

    print("\n--- TOP POWERPLAY BOWLERS (Min 3 Wickets) ---")
    query_bowlers = """
        SELECT bowler, legal_deliveries, runs_conceded, wickets, economy_rate
        FROM view_bowler_phase_stats
        WHERE phase = 'powerplay' AND wickets >= 3
        ORDER BY economy_rate ASC
        LIMIT 5;
    """
    print(pd.read_sql(query_bowlers, engine))

if __name__ == "__main__":
    test_views()