import os
import glob
import sqlite3
from pathlib import Path
import pandas as pd

CLEAN_DIR = Path("cleaned_data")
DB_PATH = Path("cleaned_data/cricket_database.sqlite")

# Connect to (or create) local SQLite database
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

print("Loading cleaned datasets into SQLite database...\n")

# Find all cleaned parquet files
parquet_files = list(CLEAN_DIR.glob("*_clean.parquet"))

for p_file in parquet_files:
    # Table name derived from filename (e.g., matches_clean -> matches)
    table_name = p_file.stem.replace("_clean", "")
    
    df = pd.read_parquet(p_file)
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    print(f"✓ Created table '{table_name}' with {len(df)} rows.")

conn.close()
print(f"\nDatabase ready at: {DB_PATH}")

import glob

def build_deliveries_table():
    print("Consolidating ball-by-ball delivery files...")
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
        except Exception as e:
            continue
            
    if records:
        deliveries_df = pd.concat(records, ignore_index=True)
        conn = sqlite3.connect("cleaned_data/cricket_database.sqlite")
        deliveries_df.to_sql("deliveries", conn, if_exists="replace", index=False)
        conn.close()
        print(f"Loaded {len(deliveries_df):,} total deliveries into 'deliveries' table.")

if __name__ == "__main__":
    build_deliveries_table()