import sqlite3
import pandas as pd
from pathlib import Path

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