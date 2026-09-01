import pandas as pd
from pathlib import Path

RAW_DIR = Path("raw_data")
OUTPUT_DIR = Path("cleaned_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find all CSV files directly in raw_data
csv_files = list(RAW_DIR.glob("*.csv"))

print(f"Found {len(csv_files)} root CSV files to process:\n")

for file in csv_files:
    print(f"Processing: {file.name}")
    df = pd.read_csv(file)
    
    # 1. Print quick snapshot
    print(f"  • Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # 2. Standardize column names (lowercase + snake_case)
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    
    # 3. Deduplicate
    before_rows = len(df)
    df = df.drop_duplicates()
    dups_removed = before_rows - len(df)
    if dups_removed > 0:
        print(f"  • Removed {dups_removed} duplicate rows")
        
    # 4. Clean text strings (strip whitespace)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()
    
    # 5. Export cleaned version as Parquet (optimized) and CSV
    parquet_out = OUTPUT_DIR / f"{file.stem}_clean.parquet"
    csv_out = OUTPUT_DIR / f"{file.stem}_clean.csv"
    
    df.to_parquet(parquet_out, index=False)
    df.to_csv(csv_out, index=False)
    print(f"  ✓ Saved to {parquet_out.name} & {csv_out.name}\n")

print("All root CSV files cleaned and staged successfully!")