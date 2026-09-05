# 🏏 IPL Tactical Intelligence & Match Prediction Hub

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pratiksha-cricket-analytics.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Database PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-336791.svg?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Machine Learning Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Visuals Plotly](https://img.shields.io/badge/Visuals-Plotly-3F4F75.svg?logo=plotly&logoColor=white)](https://plotly.com/)

An end-to-end sports analytics application and decision-intelligence platform built to transform granular ball-by-ball tournament data into actionable tactical scouting insights, historical ground intelligence, and live second-innings match outcome predictions.

🔗 **Live Deployment:** [Interactive Analytics Hub](https://pratiksha-cricket-analytics.streamlit.app/)

---

## 📌 Executive Summary & Key Features

* **Real-Time Chase Predictor:** Calculates dynamic live win probabilities based on target, wickets in hand, overs completed, and required vs. current run rates.
* **Ground & Venue Intelligence:** Direct SQL queries extract historical ground benchmarks (average 1st-innings totals, overall chase win rates, and target feasibility) from PostgreSQL.
* **Next-Ball Sensitivity Analysis (WPA):** Simulates the next delivery across 6 standard outcomes (Dot, Single, Double, Boundary, Six, Wicket) to measure game volatility and Win Probability Added.
* **Franchise Branding & Visuals:** Dual Plotly circular gauges rendered with franchise colors and official crests.
* **Tactical Matchup & Phase Analytics:** Batter vs. bowler matchup matrices and phase-of-play breakdowns across Powerplay, Middle, and Death overs.

---

## 🧭 The Engineering Journey: What Was Done & Why

| Project Stage | Implementation ("What I Did") | Architectural Justification ("Why I Did It") |
| :--- | :--- | :--- |
| **1. Ingestion & Storage Evolution** | Migrated from an embedded SQLite flat-file prototype (`cricket_database.sqlite`) to an enterprise-grade PostgreSQL relational database (`ipl_analytics`) using SQLAlchemy and `psycopg2`. | SQLite is suitable for rapid single-user prototypes, but an external PostgreSQL service provides standard ACID transactions, connection pooling, and multi-user scaling necessary for production-grade pipelines. |
| **2. Data Cleaning & Feature Engineering** | Cleaned multi-season match records in `process_data.py`, engineering critical chase variables: `runs_left`, `balls_left`, `wickets_left`, `crr`, and `rrr`. | Raw ball counts lack chase context; relative rate metrics (`crr`, `rrr`) allow the model to interpret scoreboard pressure consistently across varying targets. |
| **3. Modeling & Pipeline Serialization** | Trained a classification pipeline (Logistic Regression / Random Forest with One-Hot Encoding and Scalers) in `train_model.py` and serialized it to `.joblib`. | Encapsulating encoders and scalers into an end-to-end pipeline prevents data leakage and ensures seamless inference during live Streamlit execution. |
| **4. Ground Intelligence Layer** | Developed SQL views (`create_views.py`, `query_data.py`) querying venue-specific chase baselines and win rates. | Standalone ML predictions lack environmental nuance; querying real venue records grounds model forecasts in historical reality. |
| **5. Interface Evolution (Broadcasting UI)** | Replaced standard numeric sliders and basic progress bars with branded franchise assets and side-by-side Plotly circular probability dials in `app.py`. | Shifts the application from a bare ML prototype to a client-facing, broadcast-ready analytics product. |
| **6. Sensitivity Analysis (WPA Matrix)** | Implemented a real-time Win Probability Added (WPA) matrix simulating game leverage across 6 standard delivery outcomes. | Transforms the dashboard from a static calculator into an actionable in-game leverage tool by quantifying match volatility. |
| **7. Workspace Hygiene & Deployment** | Configured environment variables, excluded `.vscode/` and virtual environments via `.gitignore`, and deployed to Streamlit Community Cloud. | Prevents committing local IDE state or sensitive database credentials while maintaining reproducibility across environments. |

---

## 🛠️ Technology Stack & Engineering Ecosystem

| Domain | Technology / Library | Purpose & Architectural Role |
| :--- | :--- | :--- |
| **Core Runtime** | Python 3.10+ | Core pipeline processing, feature engineering, and inference engine. |
| **Data Warehouse** | PostgreSQL, SQLAlchemy, psycopg2 | Relational storage and analytical querying for match and delivery data. |
| **Machine Learning** | Scikit-Learn, Joblib | Classification modeling, categorical preprocessing, and artifact serialization. |
| **Interactive UI** | Streamlit | Web framework utilizing `@st.cache_data` and `@st.cache_resource` for low-latency re-renders. |
| **Data Visualization** | Plotly Express / Graph Objects | Circular probability gauges, situational scatter charts, and outcome distributions. |
| **DevOps & Hosting** | Git, GitHub, Streamlit Cloud | Version control and cloud hosting. |

---

## 🏗️ High-Level System Architecture

```text
Raw Match CSVs (Cricsheet Ball-by-Ball)
       │
       ▼
[ Data Ingestion Engine: build_database.py ]
       ├── Schema normalization
       ├── Relational keys & SQL indexing
       └── Ball-by-ball delivery aggregation
       │
       ▼
[ Enterprise Relational Store: PostgreSQL (ipl_analytics) ]
       │
       ├───► [ ML Training Pipeline: train_model.py ]
       │          ├── Categorical encoding & chase rate feature engineering
       │          └── Serialized Model Artifacts (models/win_predictor.joblib)
       │
       └───► [ Presentation Engine: app.py (Streamlit) ]
                  ├── Live Win Probability Predictor (Dual Gauges + Team Logos)
                  ├── Ground Intelligence Benchmarks (SQL Venue Aggregations)
                  ├── Next-Ball WPA Sensitivity Matrix (6 Outcome Scenarios)
                  └── Phase-of-Play Impact Engine (Powerplay, Middle, Death)