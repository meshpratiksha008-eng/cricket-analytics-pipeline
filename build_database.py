import os
import glob
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

# --- Database Connection Configuration ---
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

CLEAN_DIR = Path("cleaned_data")

def load_cleaned_tables():
    print("Loading cleaned datasets into PostgreSQL...\n")
    parquet_files = list(CLEAN_DIR.glob("*_clean.parquet"))

    if not parquet_files:
        print("No cleaned parquet files found in 'cleaned_data'.")
        return

    for p_file in parquet_files:
        table_name = p_file.stem.replace("_clean", "")
        df = pd.read_parquet(p_file)
        
        # Write to PostgreSQL
        df.to_sql(table_name, engine, if_exists="replace", index=False, method="multi", chunksize=1000)
        print(f"✓ Created table '{table_name}' with {len(df)} rows.")

def build_deliveries_table():
    print("\nConsolidating ball-by-ball delivery files...")
    all_files = glob.glob("raw_data/matches/*/ball_by_ball.csv")
    
    if not all_files:
        print("No ball_by_ball.csv files located.")
        return

    records = []
    for f in all_files:
        try:
            match_name = os.path.basename(os.path.dirname(f))
            df = pd.read_csv(f)
            df['match_id'] = match_name
            records.append(df)
        except Exception:
            continue

    if records:
        deliveries_df = pd.concat(records, ignore_index=True)
        print(f"Total delivery records compiled: {len(deliveries_df):,}. Writing to Postgres...")
        
        # chunksize prevents high memory spikes in 4GB RAM
        deliveries_df.to_sql(
            "deliveries", 
            engine, 
            if_exists="replace", 
            index=False, 
            chunksize=2000, 
            method="multi"
        )
        print(f"✓ Loaded {len(deliveries_df):,} total deliveries into 'deliveries' table.")

if __name__ == "__main__":
    # Test connection and run ingestion
    try:
        with engine.connect() as conn:
            print("Connected to PostgreSQL successfully.\n")
        load_cleaned_tables()
        build_deliveries_table()
        print("\nAll data successfully migrated to PostgreSQL!")
    except Exception as e:
        print(f"\nConnection or ingestion error: {e}")