# 🏏 IPL Tactical Intelligence & Match Prediction Hub

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://pratiksha-cricket-analytics.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Database PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-336791.svg?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Machine Learning Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Visuals Plotly](https://img.shields.io/badge/Visuals-Plotly-3F4F75.svg?logo=plotly&logoColor=white)](https://plotly.com/)

An end-to-end sports decision intelligence platform and machine learning system built to transform granular ball-by-ball tournament data into tactical scouting insights, historical venue benchmarks, and live second-innings match outcome predictions.

🔗 **Live Application:** [Interactive Analytics Hub](https://pratiksha-cricket-analytics.streamlit.app/)

---

## 📌 Executive Overview

Second-innings chase dynamics in T20 cricket depend heavily on resource preservation, venue-specific scoring trends, and scoreboard pressure. This application provides a unified analytical engine that evaluates chase leverage in real time by pairing historical database aggregations with predictive modeling.

### Core Capabilities
* **Dynamic Win Probability Engine:** Real-time ML calculation estimating second-innings win equity based on target, wickets in hand, balls remaining, and required vs. current run rates.
* **Historical Venue Benchmarking:** Direct SQL integration querying ground-level metrics, including average first-innings scores, historical chase success rates, and target feasibility baselines.
* **Win Probability Added (WPA) Matrix:** Live sensitivity matrix simulating game equity shifts across standard next-delivery outcomes (Dot, 1, 2, 4, 6, Wicket).
* **Broadcast-Style Interface:** Custom circular Plotly dials dynamically styled with official franchise colors, badges, and real-time match KPI differentials.
* **Tactical Scouting Module:** Batter vs. bowler head-to-head match-up matrices and phase-of-play impact summaries (Powerplay, Middle, Death).

---

## 🛠️ Technology Stack

| Domain | Technology / Library | Architectural Role |
| :--- | :--- | :--- |
| **Runtime & Pipeline** | Python 3.10+ | Feature engineering, data extraction, and model inference. |
| **Relational Storage** | PostgreSQL, SQLAlchemy, psycopg2 | Normalized relational schema for matches, deliveries, and venue indexes. |
| **Machine Learning** | Scikit-Learn, Joblib | Classification pipeline with categorical encoding, rate scaling, and serialization. |
| **Web Dashboard** | Streamlit | In-memory cached UI optimized with `@st.cache_data` and `@st.cache_resource`. |
| **Data Visualization** | Plotly Express & Graph Objects | Dual probability gauges, match leverage spreads, and situational scatter plots. |
| **Version Control & CI/CD** | Git, GitHub, Streamlit Cloud | Source control and cloud deployment. |

---

## 🏗️ System Architecture

```text
Raw Match & Delivery Data (Cricsheet Records)
                     │
                     ▼
       [ Ingestion & Normalization Engine ]
       ├── Schema validation & primary key indexing
       └── Ball-by-ball event aggregation
                     │
                     ▼
       [ PostgreSQL Data Warehouse (ipl_analytics) ]
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
[ SQL View Aggregations ]  [ Scikit-Learn ML Pipeline ]
├── Venue chase history    ├── Feature transforms (CRR, RRR, resources)
└── Phase-of-play stats    └── Serialized inference model (.joblib)
         │                       │
         └───────────┬───────────┘
                     ▼
         [ Streamlit Presentation Layer ]
         ├── Real-time probability dials (Plotly)
         ├── Next-ball WPA sensitivity matrix
         └── Historical venue intelligence cards