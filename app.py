import sqlite3
from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
import os

TEAM_LOGOS = {
    "Chennai Super Kings": "logos/CSK.png",
    "Delhi Capitals": "logos/DC.png",
    "Gujarat Titans": "logos/GT.png",
    "Kolkata Knight Riders": "logos/KKR.png",
    "Lucknow Super Giants": "logos/LSG.png",
    "Mumbai Indians": "logos/MI.png",
    "Punjab Kings": "logos/PBKS.png",
    "Rajasthan Royals": "logos/RR.png",
    "Royal Challengers Bengaluru": "logos/RCB.png",
    "Sunrisers Hyderabad": "logos/SRH.png"
}

def get_team_logo(team_name):
    path = TEAM_LOGOS.get(team_name)
    if path and os.path.exists(path):
        return path
    # Fallback if extensions differ
    for ext in [".jpg", ".jpeg", ".webp"]:
        alt_path = path.rsplit(".", 1)[0] + ext if path else None
        if alt_path and os.path.exists(alt_path):
            return alt_path
    return None

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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "🤖 AI Match Predictor",
        "⚔️ Head-to-Head",
        "🔍 Player Search",
        "📊 Standings & Raw Data",
        "🎯 Tactical Matchup Matrix",
        "⚡ Phase-of-Play Impact",
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

        # Display Team Logos and Matchup Header
        st.write("")
        logo_col1, vs_col, logo_col2 = st.columns([2, 1, 2])

        with logo_col1:
            img_a = get_team_logo(h2h_team_a)
            if img_a:
                st.image(img_a, width=110)
            st.subheader(h2h_team_a)

        with vs_col:
            st.markdown("<h2 style='text-align: center; margin-top: 25px;'>VS</h2>", unsafe_allow_html=True)

        with logo_col2:
            img_b = get_team_logo(h2h_team_b)
            if img_b:
                st.image(img_b, width=110)
            st.subheader(h2h_team_b)

        st.divider()

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

    # -----------------------------------------------------------------------------
# TAB 5: BATTER VS. BOWLER TACTICAL MATCHUP MATRIX
# -----------------------------------------------------------------------------
with tab5:
    st.header("🎯 Batter vs. Bowler Tactical Matchup Matrix")
    st.caption("Granular head-to-head scouting analytics derived from 17,500+ ball-by-ball deliveries.")
    
    deliveries_df = load_table("deliveries")
    
    if deliveries_df.empty:
        st.warning("Deliveries data not found. Please verify SQLite database build.")
    else:
        batters = sorted(deliveries_df["batter"].dropna().unique().tolist())
        bowlers = sorted(deliveries_df["bowler"].dropna().unique().tolist())
        
        c1, c2 = st.columns(2)
        selected_batter = c1.selectbox("Select Batter", batters, index=0)
        
        # Filter bowlers who have actually bowled to this batter
        faced_bowlers = deliveries_df[deliveries_df["batter"] == selected_batter]["bowler"].unique()
        available_bowlers = [b for b in bowlers if b in faced_bowlers]
        
        if not available_bowlers:
            available_bowlers = bowlers
            
        selected_bowler = c2.selectbox("Select Bowler", available_bowlers, index=0)
        
        # Aggregate head-to-head
        h2h = deliveries_df[(deliveries_df["batter"] == selected_batter) & (deliveries_df["bowler"] == selected_bowler)]
        
        if h2h.empty:
            st.info(f"No registered ball-by-ball encounters between **{selected_batter}** and **{selected_bowler}**.")
        else:
            balls = len(h2h)
            runs = int(h2h["batter_runs"].sum())
            wickets = int(h2h["is_wicket"].sum())
            dots = int((h2h["batter_runs"] == 0).sum())
            fours = int((h2h["batter_runs"] == 4).sum())
            sixes = int((h2h["batter_runs"] == 6).sum())
            
            strike_rate = (runs / balls * 100) if balls > 0 else 0.0
            dot_pct = (dots / balls * 100) if balls > 0 else 0.0
            boundary_pct = ((fours + sixes) / balls * 100) if balls > 0 else 0.0
            
            # Tactical Executive Summary
            st.markdown("### Matchup Summary")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Balls Faced", balls)
            m2.metric("Runs Scored", runs)
            m3.metric("Strike Rate", f"{strike_rate:.1f}")
            m4.metric("Dismissals", wickets)
            m5.metric("Dot Ball %", f"{dot_pct:.1f}%")
            
            # Tactical Insight Banner
            if wickets >= 2 and strike_rate < 120:
                st.error(f"⚠️ **Tactical Edge: {selected_bowler}**. High dismissal rate with sustained scoring suppression against {selected_batter}.")
            elif strike_rate > 150 and wickets == 0:
                st.success(f"🔥 **Tactical Edge: {selected_batter}**. High boundary rate with complete wicket preservation against {selected_bowler}.")
            else:
                st.info(f"⚖️ **Neutral / Balanced Contest**: Sample shows standard competitive rotation.")
            
            # Ball-by-ball breakdown chart
            st.markdown("#### Scoring Distribution")
            score_counts = h2h["batter_runs"].value_counts().reset_index()
            score_counts.columns = ["Runs on Ball", "Count"]
            score_counts["Runs on Ball"] = score_counts["Runs on Ball"].astype(str) + " Run(s)"
            
            fig_h2h = px.bar(
                score_counts,
                x="Runs on Ball",
                y="Count",
                color="Runs on Ball",
                title=f"{selected_batter} vs. {selected_bowler} Scoring Breakdown",
                text="Count"
            )
            fig_h2h.update_layout(showlegend=False, template="plotly_white")
            st.plotly_chart(fig_h2h, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 6: PHASE-OF-PLAY IMPACT ENGINE
# -----------------------------------------------------------------------------
with tab6:
    st.header("⚡ Phase-of-Play Impact Engine")
    st.caption("Situational valuation across tournament phases: Powerplay (1–6), Middle (7–15), and Death (16–20).")
    
    deliveries_df = load_table("deliveries")
    
    if deliveries_df.empty or "phase" not in deliveries_df.columns:
        st.warning("Deliveries or phase data not available.")
    else:
        role_type = st.radio("Analyze Role:", ["Batting Phase Efficiency", "Bowling Phase Containment"], horizontal=True)
        selected_phase = st.selectbox("Select Game Phase", ["powerplay", "middle", "death"])
        
        phase_subset = deliveries_df[deliveries_df["phase"].str.lower() == selected_phase.lower()]
        
        if role_type == "Batting Phase Efficiency":
            phase_agg = phase_subset.groupby("batter").agg(
                balls=("ball", "count"),
                runs=("batter_runs", "sum"),
                fours=("batter_runs", lambda x: (x == 4).sum()),
                sixes=("batter_runs", lambda x: (x == 6).sum()),
                dots=("batter_runs", lambda x: (x == 0).sum())
            ).reset_index()
            
            # Filter for meaningful sample size
            phase_agg = phase_agg[phase_agg["balls"] >= 30].copy()
            phase_agg["Strike Rate"] = (phase_agg["runs"] / phase_agg["balls"]) * 100
            phase_agg["Boundary %"] = ((phase_agg["fours"] + phase_agg["sixes"]) / phase_agg["balls"]) * 100
            phase_agg["Dot %"] = (phase_agg["dots"] / phase_agg["balls"]) * 100
            
            st.markdown(f"#### Top Impact Batters: **{selected_phase.upper()}** (Min. 30 Balls)")
            
            fig_phase = px.scatter(
                phase_agg,
                x="Strike Rate",
                y="Boundary %",
                size="runs",
                hover_name="batter",
                color="Dot %",
                color_continuous_scale="Viridis",
                title=f"{selected_phase.capitalize()} Phase: Strike Rate vs Boundary % (Size = Runs Scored)"
            )
            fig_phase.update_layout(template="plotly_white")
            st.plotly_chart(fig_phase, use_container_width=True)
            
            st.dataframe(
                phase_agg.sort_values(by="Strike Rate", ascending=False)[["batter", "runs", "balls", "Strike Rate", "Boundary %", "Dot %"]].head(15),
                use_container_width=True,
                hide_index=True
            )
            
        else:
            # Bowling Phase Containment
            phase_bowler = phase_subset.groupby("bowler").agg(
                balls=("ball", "count"),
                runs_conceded=("total_runs", "sum"),
                wickets=("is_wicket", "sum"),
                dots=("total_runs", lambda x: (x == 0).sum()),
                boundaries=("total_runs", lambda x: (x >= 4).sum())
            ).reset_index()
            
            phase_bowler = phase_bowler[phase_bowler["balls"] >= 30].copy()
            phase_bowler["Economy"] = (phase_bowler["runs_conceded"] / phase_bowler["balls"]) * 6.0
            phase_bowler["Dot %"] = (phase_bowler["dots"] / phase_bowler["balls"]) * 100
            phase_bowler["Boundary % Conceded"] = (phase_bowler["boundaries"] / phase_bowler["balls"]) * 100
            
            st.markdown(f"#### Bowling Containment Matrix: **{selected_phase.upper()}** (Min. 30 Balls)")
            
            fig_bowl = px.scatter(
                phase_bowler,
                x="Economy",
                y="Dot %",
                size="wickets",
                hover_name="bowler",
                color="Boundary % Conceded",
                color_continuous_scale="Reds_r",
                title=f"{selected_phase.capitalize()} Bowling: Economy vs Dot Ball % (Higher Dot % & Lower Economy = Elite)"
            )
            fig_bowl.update_layout(template="plotly_white")
            st.plotly_chart(fig_bowl, use_container_width=True)
            
            st.dataframe(
                phase_bowler.sort_values(by="Economy", ascending=True)[["bowler", "balls", "runs_conceded", "wickets", "Economy", "Dot %", "Boundary % Conceded"]].head(15),
                use_container_width=True,
                hide_index=True
            )