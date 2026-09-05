import os
import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

st.set_page_config(page_title="IPL Win Predictor", page_icon="🏏", layout="wide")

@st.cache_resource
def load_pipeline():
    return joblib.load("models/win_predictor.joblib")

pipeline = load_pipeline()

TEAM_CONFIG = {
    "Chennai Super Kings": {
        "color": "#F9CD05",
        "logo": "https://upload.wikimedia.org/wikipedia/en/2/2b/Chennai_Super_Kings_Logo.svg",
        "file": "logos/csk.png"
    },
    "Delhi Capitals": {
        "color": "#004C93",
        "logo": "https://upload.wikimedia.org/wikipedia/en/2/2f/Delhi_Capitals.svg",
        "file": "logos/dc.png"
    },
    "Gujarat Titans": {
        "color": "#1B2133",
        "logo": "https://upload.wikimedia.org/wikipedia/en/0/09/Gujarat_Titans_Logo.svg",
        "file": "logos/gt.png"
    },
    "Kolkata Knight Riders": {
        "color": "#3A225D",
        "logo": "https://upload.wikimedia.org/wikipedia/en/4/4c/Kolkata_Knight_Riders_Logo.svg",
        "file": "logos/kkr.png"
    },
    "Lucknow Super Giants": {
        "color": "#A72056",
        "logo": "https://upload.wikimedia.org/wikipedia/en/a/a9/Lucknow_Super_Giants_IPL_Logo.svg",
        "file": "logos/lsg.png"
    },
    "Mumbai Indians": {
        "color": "#004BA0",
        "logo": "https://upload.wikimedia.org/wikipedia/en/c/cd/Mumbai_Indians_Logo.svg",
        "file": "logos/mi.png"
    },
    "Punjab Kings": {
        "color": "#DD1F2D",
        "logo": "https://upload.wikimedia.org/wikipedia/en/d/d4/Punjab_Kings_Logo.svg",
        "file": "logos/pbks.png"
    },
    "Rajasthan Royals": {
        "color": "#EA1A85",
        "logo": "https://upload.wikimedia.org/wikipedia/en/6/60/Rajasthan_Royals_Logo.svg",
        "file": "logos/rr.png"
    },
    "Royal Challengers Bengaluru": {
        "color": "#DA1818",
        "logo": "https://upload.wikimedia.org/wikipedia/en/2/2a/Royal_Challengers_Bangalore_2020.svg",
        "file": "logos/rcb.png"
    },
    "Sunrisers Hyderabad": {
        "color": "#FF822A",
        "logo": "https://upload.wikimedia.org/wikipedia/en/8/81/Sunrisers_Hyderabad.svg",
        "file": "logos/srh.png"
    }
}

TEAMS = sorted(list(TEAM_CONFIG.keys()))

VENUES = [
    "Wankhede Stadium", "Eden Gardens", "M Chinnaswamy Stadium",
    "Narendra Modi Stadium", "MA Chidambaram Stadium", 
    "Arun Jaitley Stadium", "Rajiv Gandhi International Stadium"
]

def render_team_logo(team_name):
    cfg = TEAM_CONFIG.get(team_name, {})
    local_file = cfg.get("file")
    if local_file and os.path.exists(local_file):
        st.image(local_file, width=110)
    else:
        st.image(cfg.get("logo"), width=110)

st.title("🏏 IPL Live Win Probability Predictor")
st.markdown("Machine Learning Chase Forecaster backed by PostgreSQL ball-by-ball records.")

# --- Match Selectors ---
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

# --- Match Situation Inputs ---
c1, c2, c3, c4 = st.columns(4)

with c1:
    target = st.number_input("Target Score", min_value=50, max_value=320, value=195, step=1)
with c2:
    score = st.number_input("Current Score", min_value=0, max_value=int(target), value=142, step=1)
with c3:
    wickets_out = st.number_input("Wickets Down", min_value=0, max_value=9, value=4, step=1)
with c4:
    overs_completed = st.number_input(
        "Overs Bowled (e.g. 15.2)", 
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

# Metric Banner
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

    def create_gauge(team_name, prob, color):
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob,
            title={'text': f"<b>{team_name}</b>", 'font': {'size': 18}},
            number={'suffix': "%", 'font': {'size': 32}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': "#f4f4f7"},
                    {'range': [50, 100], 'color': "#e2e4e9"}
                ]
            }
        ))
        fig.update_layout(height=230, margin=dict(l=10, r=10, t=30, b=10))
        return fig

    # Two balanced presentation columns
    g_col1, g_col2 = st.columns(2)

    with g_col1:
        st.write("### Batting Side")
        render_team_logo(batting_team)
        st.plotly_chart(
            create_gauge(batting_team, bat_prob, TEAM_CONFIG[batting_team]["color"]), 
            use_container_width=True
        )

    with g_col2:
        st.write("### Defending Side")
        render_team_logo(bowling_team)
        st.plotly_chart(
            create_gauge(bowling_team, bowl_prob, TEAM_CONFIG[bowling_team]["color"]), 
            use_container_width=True
        )

        # ==========================================
    # DATA ANALYTICS: NEXT-BALL WPA SENSITIVITY
    # ==========================================
    st.divider()
    st.subheader("⚡ Next-Ball Leverage Analysis (Win Probability Added)")
    st.caption("Sensitivity analysis quantifying match volatility across potential next-ball events.")

    def calc_future_prob(r_inc, w_inc):
        sim_b_left = max(balls_left - 1, 1)
        sim_r_left = max(runs_left - r_inc, 0)
        sim_w_left = max(wickets_left - w_inc, 0)
        sim_b_bowled = balls_bowled + 1
        sim_score = score + r_inc

        sim_crr = (sim_score * 6.0) / sim_b_bowled
        sim_rrr = (sim_r_left * 6.0) / sim_b_left

        sim_df = pd.DataFrame([{
            'batting_team': batting_team,
            'bowling_team': bowling_team,
            'venue': venue,
            'runs_left': float(sim_r_left),
            'balls_left': float(sim_b_left),
            'wickets_left': float(sim_w_left),
            'target': float(target),
            'crr': float(sim_crr),
            'rrr': float(sim_rrr)
        }])
        sim_probs = pipeline.predict_proba(sim_df)[0]
        return round(sim_probs[1] * 100, 1), sim_rrr

    scenarios = [
        ("Dot Ball (0 runs)", 0, 0),
        ("Single (1 run)", 1, 0),
        ("Two Runs (2 runs)", 2, 0),
        ("Boundary (4 runs)", 4, 0),
        ("Maximum (6 runs)", 6, 0),
        ("Wicket Falls", 0, 1)
    ]

    analytics_table = []
    for event_name, runs_added, wickets_added in scenarios:
        new_prob, new_rrr = calc_future_prob(runs_added, wickets_added)
        wpa = round(new_prob - bat_prob, 1)

        analytics_table.append({
            "Next Ball Scenario": event_name,
            "Updated RRR": round(new_rrr, 2),
            f"{batting_team} New Win %": f"{new_prob}%",
            "Win Probability Added (WPA)": f"{'+' if wpa > 0 else ''}{wpa}%"
        })

    st.dataframe(pd.DataFrame(analytics_table), hide_index=True, use_container_width=True)