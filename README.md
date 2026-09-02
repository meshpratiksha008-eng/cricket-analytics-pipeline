# 🏏 IPL Tactical Intelligence & Match Prediction Hub

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pratiksha-cricket-analytics.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Database SQLite](https://img.shields.io/badge/Database-SQLite3-003B57.svg?logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Machine Learning Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Visuals Plotly](https://img.shields.io/badge/Visuals-Plotly-3F4F75.svg?logo=plotly&logoColor=white)](https://plotly.com/)

An end-to-end sports decision intelligence platform and machine learning engine built to transform granular ball-by-ball tournament data into actionable tactical scouting insights and live match outcome predictions.

🔗 **Live Deployment:** [Interactive Analytics Hub](https://pratiksha-cricket-analytics.streamlit.app/)

---

## 🛠️ Technology Stack & Engineering Ecosystem

| Domain | Technology / Library | Purpose & Architectural Role |
| :--- | :--- | :--- |
| **Core Runtime** | Python 3.10+ | Core pipeline processing, feature engineering, and inference engine. |
| **Data Warehouse** | SQLite3, Pandas | Serverless ACID-compliant relational storage; 17,500+ indexed delivery records with sub-millisecond query windows. |
| **Machine Learning** | Scikit-Learn | Calibrated Random Forest multi-class classification pipeline with categorical one-hot encoding. |
| **Interactive UI** | Streamlit | Responsive multi-tab interactive dashboard optimized with `@st.cache_data` in-memory serialization. |
| **Data Visualization** | Plotly Express | Interactive situational scatter quadrant charts, match outcome win-shares, and scoring distribution graphs. |
| **Reporting & Docs** | WeasyPrint, FPDF2 | Automated generation of downloadable executive PDF dossiers and scout summaries. |
| **DevOps & CI/CD** | Git, GitHub, Streamlit Cloud | Version control and automated zero-downtime deployment pipeline. |

---

## 🏗️ High-Level System Architecture

```text
Raw Match CSVs (74 Matches)
        │
        ▼
[ Data Ingestion Engine: build_database.py ]
        │  ├── Schema normalization
        │  ├── Relational keys & SQL indexing
        │  └── Ball-by-ball delivery aggregation (17,523 rows)
        ▼
[ Relational Store: cricket_database.sqlite ]
        │
        ├───► [ ML Training Pipeline: train_model.py ]
        │         ├── Categorical encoding & pre-match feature gating
        │         └── Random Forest Model Artifact (joblib/pickle)
        │
        └───► [ Presentation Engine: app.py (Streamlit) ]
                  ├── Live Win Probability Predictor (Random Forest)
                  ├── Head-to-Head Record Explorer (with Dynamic Badges)
                  ├── Tactical Matchup Matrix (Batter vs. Bowler)
                  └── Phase-of-Play Impact Engine (Powerplay, Middle, Death)