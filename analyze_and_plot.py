import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup output folder for plots
OUTPUT_DIR = Path("analytics_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Connect to the SQLite database
conn = sqlite3.connect("cleaned_data/cricket_database.sqlite")

# 1. Inspect Team Standings & Win Distribution
print("Generating Team Standings Analysis...")
points_df = pd.read_sql("SELECT * FROM points_table", conn)

if not points_df.empty:
    plt.figure(figsize=(10, 6))
    # Adjust column name to match your schema (e.g., 'team', 'pts' or 'points')
    team_col = [c for c in points_df.columns if "team" in c][0]
    pts_col = [c for c in points_df.columns if "pt" in c or "point" in c][0]
    
    sns.barplot(data=points_df.sort_values(by=pts_col, ascending=False), x=team_col, y=pts_col, palette="viridis")
    plt.title("Team Points Distribution", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "team_points.png")
    plt.close()
    print("✓ Saved 'team_points.png'")

# 2. Inspect Matches Outcome Summary
print("Generating Match Outcome Summary...")
matches_df = pd.read_sql("SELECT * FROM matches", conn)

if not matches_df.empty:
    # Identify winner column
    winner_cols = [c for c in matches_df.columns if "win" in c or "result" in c]
    if winner_cols:
        winner_col = winner_cols[0]
        top_winners = matches_df[winner_col].value_counts().head(10)
        
        plt.figure(figsize=(10, 5))
        top_winners.plot(kind="bar", color="skyblue", edgecolor="black")
        plt.title("Total Matches Won by Team", fontsize=14)
        plt.xlabel("Team")
        plt.ylabel("Wins")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "match_winners.png")
        plt.close()
        print("✓ Saved 'match_winners.png'")

conn.close()
print(f"\nAll reports and charts generated in '{OUTPUT_DIR}/'!")