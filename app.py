import sqlite3
from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# -------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------
st.set_page_config(
    page_title="Cricket Analytics & AI Predictor Hub",
    page_icon="🏏",
    layout="wide",
)

st.title("🏏 Cricket Analytics & AI Match Predictor Hub")


# -------------------------------------------------------------
# DATA & MODEL LOADERS
# -------------------------------------------------------------
@st.cache_data
def load_table(table_name):
    conn = sqlite3.connect("cleaned_data/cricket_database.sqlite")
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df


@st.cache_resource
def load_ml_model():
    model_path = Path("models/match_winner_model.joblib")
    if model_path.exists():
        return joblib.load(model_path)
    return None


matches_df = load_table("matches")
points_df = load_table("points_table")
players_df = load_table("players")
model = load_ml_model()

# Dynamically identify core columns
team1_col = next(
    (c for c in matches_df.columns if "team1" in c or "team_1" in c), None
)
team2_col = next(
    (c for c in matches_df.columns if "team2" in c or "team_2" in c), None
)
winner_col = next(
    (c for c in matches_df.columns if "win" in c or "result" in c), None
)
toss_win_col = next(
    (c for c in matches_df.columns if "toss_win" in c or "toss_winner" in c),
    None,
)
venue_col = next(
    (c for c in matches_df.columns if "venue" in c or "city" in c or "ground" in c),
    None,
)

# Extract unique team names
if team1_col and team2_col:
    unique_teams = sorted(
        list(
            set(
                matches_df[team1_col].dropna().unique().tolist()
                + matches_df[team2_col].dropna().unique().tolist()
            )
        )
    )
else:
    unique_teams = []

# -------------------------------------------------------------
# MAIN NAVIGATION TABS
# -------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🤖 AI Match Predictor",
        "⚔️ Head-to-Head",
        "🔍 Player Search",
        "📊 Standings & Raw Data",
    ]
)

# -------------------------------------------------------------
# TAB 1: AI MATCH OUTCOME PREDICTOR
# -------------------------------------------------------------
with tab1:
    st.subheader("🎯 Live Win Probability Predictor")
    st.caption(
        "Powered by Random Forest Machine Learning trained on historical match features."
    )

    if model is None:
        st.error(
            "⚠️ Model artifact not found! Run `python train_model.py` in your terminal first to train and save the model."
        )
    elif not unique_teams:
        st.warning("⚠️ Team columns could not be identified in the database.")
    else:
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            team_a = st.selectbox(
                "Select Team 1", unique_teams, index=0, key="ml_team_a"
            )
            available_opponents_ml = [t for t in unique_teams if t != team_a]
            team_b = st.selectbox(
                "Select Team 2",
                available_opponents_ml,
                index=0,
                key="ml_team_b",
            )

        with col_m2:
            toss_winner = st.selectbox(
                "Select Toss Winner", [team_a, team_b], key="ml_toss"
            )

            # Venue selection (fallback if venue column missing)
            if venue_col:
                unique_venues = sorted(
                    matches_df[venue_col].dropna().unique().tolist()
                )
                selected_venue = st.selectbox(
                    "Select Venue / Ground", unique_venues, key="ml_venue"
                )
            else:
                selected_venue = "Unknown"

        st.write("")
        if st.button("🚀 Calculate Win Probability", use_container_width=True):
            # Construct row matching model training schema
            input_dict = {
                team1_col: team_a,
                team2_col: team_b,
            }
            if toss_win_col:
                input_dict[toss_win_col] = toss_winner
            if venue_col:
                input_dict[venue_col] = selected_venue

            input_df = pd.DataFrame([input_dict])

            try:
                # Predict probabilities
                probabilities = model.predict_proba(input_df)[0]
                classes = list(model.classes_)

                prob_a = (
                    probabilities[classes.index(team_a)]
                    if team_a in classes
                    else 0.0
                )
                prob_b = (
                    probabilities[classes.index(team_b)]
                    if team_b in classes
                    else 0.0
                )

                # Normalize probabilities between the two selected teams
                total_prob = prob_a + prob_b
                if total_prob > 0:
                    norm_prob_a = (prob_a / total_prob) * 100
                    norm_prob_b = (prob_b / total_prob) * 100
                else:
                    norm_prob_a, norm_prob_b = 50.0, 50.0

                st.divider()

                # Visual probability cards
                res_col1, res_col2 = st.columns(2)
                res_col1.metric(f"🏏 {team_a} Win Chance", f"{norm_prob_a:.1f}%")
                res_col2.metric(f"🏏 {team_b} Win Chance", f"{norm_prob_b:.1f}%")

                st.progress(int(norm_prob_a))

                # Highlight Favorite
                if norm_prob_a > norm_prob_b:
                    st.success(
                        f"🏆 **Prediction:** {team_a} is favored to win ({norm_prob_a:.1f}% probability)."
                    )
                elif norm_prob_b > norm_prob_a:
                    st.success(
                        f"🏆 **Prediction:** {team_b} is favored to win ({norm_prob_b:.1f}% probability)."
                    )
                else:
                    st.info("⚖️ **Prediction:** Exact 50/50 toss-up match.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# -------------------------------------------------------------
# TAB 2: HEAD-TO-HEAD ANALYSIS
# -------------------------------------------------------------
with tab2:
    st.subheader("Team vs Team Head-to-Head Record")

    if team1_col and team2_col:
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            h2h_team_a = st.selectbox(
                "Select Team A", unique_teams, index=0, key="h2h_a"
            )
        with col_h2:
            opponents_h2h = [t for t in unique_teams if t != h2h_team_a]
            h2h_team_b = st.selectbox(
                "Select Team B", opponents_h2h, index=0, key="h2h_b"
            )

        h2h_matches = matches_df[
            (
                (matches_df[team1_col] == h2h_team_a)
                & (matches_df[team2_col] == h2h_team_b)
            )
            | (
                (matches_df[team1_col] == h2h_team_b)
                & (matches_df[team2_col] == h2h_team_a)
            )
        ]

        total_played = len(h2h_matches)
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Matches", total_played)

        if winner_col and total_played > 0:
            wins_a = (h2h_matches[winner_col] == h2h_team_a).sum()
            wins_b = (h2h_matches[winner_col] == h2h_team_b).sum()
            m2.metric(f"{h2h_team_a} Wins", wins_a)
            m3.metric(f"{h2h_team_b} Wins", wins_b)

            summary = h2h_matches[winner_col].value_counts().reset_index()
            summary.columns = ["Winner", "Count"]
            fig_pie = px.pie(
                summary,
                names="Winner",
                values="Count",
                title=f"Win Share: {h2h_team_a} vs {h2h_team_b}",
                hole=0.4,
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.dataframe(h2h_matches, use_container_width=True)

# -------------------------------------------------------------
# TAB 3: PLAYER SEARCH
# -------------------------------------------------------------
with tab3:
    st.subheader("Player Profile & Stats Lookup")

    name_col = next(
        (c for c in players_df.columns if "name" in c or "player" in c),
        players_df.columns[0],
    )
    team_player_col = next((c for c in players_df.columns if "team" in c), None)

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        query = st.text_input(
            "Search Player Name", placeholder="e.g. Virat, Rohit, Stokes"
        )
    with col_s2:
        team_opts = (
            sorted(players_df[team_player_col].dropna().unique())
            if team_player_col
            else []
        )
        selected_t_filter = (
            st.multiselect("Filter by Team", team_opts) if team_opts else []
        )

    filtered_p = players_df.copy()
    if query:
        filtered_p = filtered_p[
            filtered_p[name_col]
            .astype(str)
            .str.contains(query, case=False, na=False)
        ]
    if selected_t_filter and team_player_col:
        filtered_p = filtered_p[
            filtered_p[team_player_col].isin(selected_t_filter)
        ]

    st.write(f"Showing **{len(filtered_p)}** player(s)")
    st.dataframe(filtered_p, use_container_width=True)

# -------------------------------------------------------------
# TAB 4: STANDINGS & RAW DATA
# -------------------------------------------------------------
with tab4:
    st.subheader("Points Table & Standings")
    st.dataframe(points_df, use_container_width=True)

    st.divider()
    st.subheader("All Matches Raw Table")
    st.dataframe(matches_df, use_container_width=True)