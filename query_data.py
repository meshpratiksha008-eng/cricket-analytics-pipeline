import sqlite3
import pandas as pd

conn = sqlite3.connect("cleaned_data/cricket_database.sqlite")

# 1. Inspect available tables
tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
print("Available Tables:")
print(tables, "\n")

# 2. Sample query: view top rows of points table or matches
sample_query = """
SELECT * FROM points_table LIMIT 5;
"""

result = pd.read_sql(sample_query, conn)
print("Points Table Sample:")
print(result)

conn.close()