# 🏏 Cricket Analytics & AI Match Prediction Engine

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cricket-analytics-pipeline-7ocyyvqfd7ukomzgmhqnmu.streamlit.app)

> **Live Demo:** [Open Interactive Cricket Hub](https://pratiksha-cricket-analytics.streamlit.app/)

An end-to-end data analytics and machine learning application built with Python, SQLite, Scikit-Learn, and Streamlit.

## 🚀 Key Features
* **Automated ETL Pipeline:** Ingests, standardizes, deduplicates, and structures raw cricket match/player data into Parquet and SQLite.
* **Interactive Web Hub:** Streamlit-powered dashboard featuring head-to-head records and dynamic player search.
* **AI Match Outcome Predictor:** Supervised machine learning classification pipeline (`RandomForestClassifier`) predicting match outcomes based on toss, teams, and venue conditions.
### 🎯 Advanced Decision Intelligence Modules

* **Tactical Batter vs. Bowler Matchup Matrix:** Granular head-to-head tactical query engine analyzing 17,500+ ball-by-ball deliveries. Computes situational metrics (strike rate suppression, dot-ball percentage, boundary frequency, and dismissal vulnerability) to drive pre-match scouting and bowling rotation strategies.
* **Phase-of-Play Impact Engine:** Contextual feature engineering isolating player performance across tournament phases: **Powerplay (Overs 1–6)**, **Middle Overs (Overs 7–15)**, and **Death Overs (Overs 16–20)**. Replaces flat aggregate metrics with situational efficiency indices (Economy vs. Dot Ball % containment quadrants, Strike Rate vs. Boundary % output).

## 🛠️ Tech Stack
* **Language:** Python
* **Data Processing & Storage:** Pandas, PyArrow, SQLite3
* **Machine Learning:** Scikit-Learn, Joblib
* **Visualization & App:** Streamlit, Plotly Express

## ⚡ Quick Start

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/meshpratiksha008-eng/cricket-analytics-pipeline.git](https://github.com/meshpratiksha008-eng/cricket-analytics-pipeline.git)
   cd cricket-analytics-pipeline