import os
import joblib
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

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

def fetch_and_prep_data():
    print("Extracting second innings chase data from PostgreSQL...")
    query = """
    WITH first_innings AS (
        SELECT 
            match_number,
            SUM(total_runs) AS target_score
        FROM deliveries
        WHERE innings = 1
        GROUP BY match_number
    )
    SELECT 
        d.match_number,
        d.innings,
        d.team AS batting_team,
        m.team_1,
        m.team_2,
        m.winner,
        m.venue,
        (fi.target_score + 1) AS target,
        d.over,
        d.ball,
        d.total_runs,
        d.is_wicket
    FROM deliveries d
    JOIN first_innings fi ON d.match_number = fi.match_number
    JOIN matches m ON d.match_number = m.match_number
    WHERE d.innings = 2
    ORDER BY d.match_number, d.over, d.ball;
    """
    df = pd.read_sql(query, engine)

    for col in ['target', 'total_runs', 'over', 'ball', 'match_number']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)

    df['is_wicket'] = df['is_wicket'].astype(bool).astype(int)

    for str_col in ['batting_team', 'team_1', 'team_2', 'winner', 'venue']:
        df[str_col] = df[str_col].astype(str).str.strip()

    df['bowling_team'] = df.apply(
        lambda r: r['team_2'] if r['batting_team'] == r['team_1'] else r['team_1'],
        axis=1
    )

    df['current_score'] = df.groupby('match_number')['total_runs'].cumsum()
    df['balls_bowled'] = (df['over'] * 6.0) + df['ball']
    df['balls_left'] = 120.0 - df['balls_bowled']
    df['runs_left'] = df['target'] - df['current_score']

    df['wickets_lost'] = df.groupby('match_number')['is_wicket'].cumsum()
    df['wickets_left'] = 10.0 - df['wickets_lost']

    df['crr'] = (df['current_score'] * 6.0) / df['balls_bowled']
    df['rrr'] = (df['runs_left'] * 6.0) / df['balls_left']

    df['result'] = (df['batting_team'] == df['winner']).astype(int)

    clean_df = df[(df['balls_left'] > 0) & (df['runs_left'] >= 0)].copy()
    clean_df = clean_df.replace([float('inf'), -float('inf')], pd.NA).dropna()

    return clean_df

def train():
    df = fetch_and_prep_data()

    features = [
        'batting_team', 
        'bowling_team', 
        'venue', 
        'runs_left', 
        'balls_left', 
        'wickets_left', 
        'target', 
        'crr', 
        'rrr'
    ]

    X = df[features]
    y = df['result']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), ['batting_team', 'bowling_team', 'venue']),
            ('num', StandardScaler(), ['runs_left', 'balls_left', 'wickets_left', 'target', 'crr', 'rrr'])
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(solver='lbfgs', max_iter=1000))
    ])

    print("Training Win-Probability Logistic Regression...")
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)

    print(f"\nModel Evaluation:")
    print(f"• Accuracy : {accuracy_score(y_test, preds) * 100:.2f}%")
    print(f"• Log Loss : {log_loss(y_test, probs):.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, "models/win_predictor.joblib")
    print("\n✓ Model saved to: models/win_predictor.joblib")

if __name__ == "__main__":
    train()