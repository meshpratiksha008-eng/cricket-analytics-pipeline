import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="wide")

@st.cache_resource
def load_pipeline():
    return joblib.load("models/win_predictor.joblib")

pipeline = load_pipeline()

# Franchise Branding & Colors
TEAM_COLORS = {
    "Chennai Super Kings": "#F9CD05",
    "Delhi Capitals": "#004C93",
    "Gujarat Titans": "#1B2133",
    "Kolkata Knight Riders": "#3A225D",
    "Lucknow Super Giants": "#A72056",
    "Mumbai Indians": "#004BA0",
    "Punjab Kings": "#DD1F2D",
    "Rajasthan Royals": "#EA1A85",
    "Royal Challengers Bengaluru": "#DA1818",
    "Sunrisers Hyderabad": "#FF822A"
}

TEAMS = sorted(list(TEAM_COLORS.keys()))

VENUES = [
    "Wankhede Stadium", "Eden Gardens", "M Chinnaswamy Stadium",
    "Narendra Modi Stadium", "MA Chidambaram Stadium", 
    "Arun Jaitley Stadium", "Rajiv Gandhi International Stadium"
]

st.title("🏏 IPL Second-Innings Win Probability Predictor")
st.markdown("Dynamic match momentum forecasting powered by Logistic Regression & Historical Cricsheet data.")

# --- Match Configuration Section ---
col_teams, col_venue = st.columns([2, 1])

with col_teams:
    t_col1, t_col2 = st.columns(2)
    with t_col1:
        batting_team = st.selectbox("Batting Team (Chasing)", TEAMS, index=TEAMS.index("Royal Challengers Bengaluru"))
    with t_col2:
        available_bowlers = [t for t in TEAMS if t != batting_team]
        bowling_team = st.selectbox("Bowling Team (Defending)", available_bowlers, index=available_bowlers.index("Chennai Super Kings"))

with col_venue:
    venue = st.selectbox("Match Venue", VENUES, index=2)

st.divider()

# --- Match State Sliders & Inputs ---
c1, c2, c3, c4 = st.columns(4)

with c1:
    target = st.number_input("Target Score", min_value=50, max_value=320, value=195, step=1)
with c2:
    score = st.number_input("Current Score", min_value=0, max_value=int(target), value=142, step=1)
with c3:
    wickets_out = st.number_input("Wickets Down", min_value=0, max_value=9, value=4, step=1)
with c4:
    overs_completed = st.number_input(
        "Overs Bowled (e.g., 15.2)", 
        min_value=0.1, 
        max_value=19.5, 
        value=15.0, 
        step=0.1,
        format="%.1f"
    )

# --- Derived Ball & Rate Calculations ---
full_overs = int(overs_completed)
balls_in_over = int(round((overs_completed - full_overs) * 10))

if balls_in_over > 5:
    st.error("⚠️ Invalid over notation: Decimal must be between .0 and .5 (e.g., 15.5)")
    st.stop()

balls_bowled = (full_overs * 6) + balls_in_over
balls_left = 120 - balls_bowled
runs_left = target - score
wickets_left = 10 - wickets_out

crr = (score * 6.0) / balls_bowled if balls_bowled > 0 else 0.0
rrr = (runs_left * 6.0) / balls_left if balls_left > 0 else 0.0

# Dynamic Equation Badges
m1, m2, m3, m4 = st.columns(4)
m1.metric("Runs Needed", f"{runs_left} runs")
m2.metric("Balls Remaining", f"{balls_left} balls")
m3.metric("Current Run Rate (CRR)", f"{crr:.2f}")
m4.metric("Required Run Rate (RRR)", f"{rrr:.2f}", delta=f"{crr - rrr:.2f}", delta_color="normal")

st.write("")

if st.button("Calculate Live Odds", type="primary", use_container_width=True):
    input_data = pd.DataFrame([{
        'batting_team': batting_team,
        'bowling_team': bowling_team,
        'venue': venue,
        'runs_left': float(runs_left),
        'balls_left': float(balls_left),
        'wickets_left': float(wickets_left),
        'target': float(target),
        'crr': float(crr),
        'rrr': float(rrr)
    }])

    probs = pipeline.predict_proba(input_data)[0]
    bat_prob = round(probs[1] * 100, 1)
    bowl_prob = round(probs[0] * 100, 1)

    # --- Circular Gauge Meters ---
    g_col1, g_col2 = st.columns(2)

    def create_gauge(team_name, prob, color):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title={'text': team_name, 'font': {'size': 20}},
            number={'suffix': "%"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': "#f0f2f6"},
                    {'range': [50, 100], 'color': "#e1e4ea"}
                ]
            }
        ))
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
        return fig

    with g_col1:
        st.plotly_chart(create_gauge(batting_team, bat_prob, TEAM_COLORS.get(batting_team, "#2b5c8f")), use_container_width=True)
    with g_col2:
        st.plotly_chart(create_gauge(bowling_team, bowl_prob, TEAM_COLORS.get(bowling_team, "#e65c00")), use_container_width=True)