import sqlite3
import joblib
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Load Data
conn = sqlite3.connect("cleaned_data/cricket_database.sqlite")
df = pd.read_sql("SELECT * FROM matches", conn)
conn.close()

# Dynamically locate key columns
team1_col = next((c for c in df.columns if "team1" in c or "team_1" in c), None)
team2_col = next((c for c in df.columns if "team2" in c or "team_2" in c), None)
winner_col = next((c for c in df.columns if "win" in c or "result" in c), None)
toss_win_col = next((c for c in df.columns if "toss_win" in c or "toss_winner" in c), None)
venue_col = next((c for c in df.columns if "venue" in c or "city" in c or "ground" in c), None)

# Drop missing target rows
df = df.dropna(subset=[team1_col, team2_col, winner_col])

# Filter out matches with no result / ties if needed
df = df[df[winner_col].isin(df[team1_col].unique()) | df[winner_col].isin(df[team2_col].unique())]

# 2. Select Features & Target
feature_cols = [col for col in [team1_col, team2_col, toss_win_col, venue_col] if col is not None]
X = df[feature_cols]
y = df[winner_col]

# 3. Build Preprocessing & Modeling Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), feature_cols)
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)),
    ]
)

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Train Model
print("Training match prediction model...")
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")
print("--- Classification Report ---")
print(classification_report(y_test, y_pred, zero_division=0))

# 7. Save Model Artifact
model_dir = Path("models")
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / "match_winner_model.joblib"

joblib.dump(model, model_path)
print(f"Model saved successfully to: {model_path}")