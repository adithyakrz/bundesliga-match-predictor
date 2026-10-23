from pathlib import Path
from src.features import computeteamstats
import pickle
import numpy as np
import streamlit as st



st.set_page_config(
    page_title="Bundesliga Match Predictor", page_icon="⚽", layout="centered"
)

@st.cache_resource
def load_artifacts():
  model_path = Path("models/bundesliga_model.pkl")
  with open(model_path, "rb") as f:
    return pickle.load(f)

artifacts = load_artifacts()
model = artifacts["model"]
scaler = artifacts["scaler"]
team_hist = artifacts["team_hist"]
home_hist = artifacts["home_hist"]
away_hist = artifacts["away_hist"]
teams = artifacts["teams"]


st.title("⚽ Bundesliga Match Outcome Predictor")
st.markdown(
    "Predict 3-way match outcomes using rolling form, venue-specific splits, and"
    " shot differentials."
)
st.write("---")

col1, col2 = st.columns(2)
with col1:
  default_home = (
      teams.index("Bayern Munich") if "Bayern Munich" in teams else 0
  )
  home_team = st.selectbox("🏠 Home Team", teams, index=default_home, key="home")

with col2:
  default_away = (
      teams.index("Dortmund")
      if "Dortmund" in teams
      else (1 if len(teams) > 1 else 0)
  )
  away_team = st.selectbox("✈️ Away Team", teams, index=default_away, key="away")

if home_team == away_team:
  st.warning("Please choose two different clubs to simulate a match.")
else:
  # Extract current pre-match differentials
  h_s3 = computeteamstats(team_hist[home_team], n=3)
  a_s3 = computeteamstats(team_hist[away_team], n=3)
  h_s8 = computeteamstats(team_hist[home_team], n=8)
  a_s8 = computeteamstats(team_hist[away_team], n=8)
  h_venue = computeteamstats(home_hist[home_team], n=5)
  a_venue = computeteamstats(away_hist[away_team], n=5)

  raw_features = np.array([[
      h_s3["avg_points"] - a_s3["avg_points"],
      h_s3["avg_goal_diff"] - a_s3["avg_goal_diff"],
      h_s8["avg_points"] - a_s8["avg_points"],
      h_s8["avg_goal_diff"] - a_s8["avg_goal_diff"],
      h_s8["avg_shots_target"] - a_s8["avg_shots_target"],
      h_venue["avg_points"] - a_venue["avg_points"],
      h_venue["avg_goal_diff"] - a_venue["avg_goal_diff"],
  ]])

  scaled_features = scaler.transform(raw_features)
  probs = model.predict_proba(scaled_features)[0]
  prob_map = dict(zip(model.classes_, probs))

  if st.button("🚀 Calculate Match Probabilities", use_container_width=True):
    st.subheader(f"📊 Prediction: {home_team} vs {away_team}")

    m1, m2, m3 = st.columns(3)
    m1.metric(f"{home_team} Win", f"{prob_map['H']:.1%}")
    m2.metric("Draw", f"{prob_map['D']:.1%}")
    m3.metric(f"{away_team} Win", f"{prob_map['A']:.1%}")

    prob_diff = abs(prob_map["H"] - prob_map["A"])

    if prob_diff < 0.12 or prob_map["D"] > 0.26:
        st.warning(
            "⚠️ **High Draw / Upset Alert:** Tactical differentials indicate a"
            " tightly contested fixture. Consider Draw or Double Chance (1X / X2)."
        )

    st.write(f"**{home_team} Win Probability**")
    st.progress(float(prob_map["H"]))

    st.write("**Draw Probability**")
    st.progress(float(prob_map["D"]))

    st.write(f"**{away_team} Win Probability**")
    st.progress(float(prob_map["A"]))

    st.write("---")
    st.markdown("### 🔍 Key Tactical Differentials")
    st.table({
        "Metric": [
            "Baseline Goal Diff (Last 8)",
            "Baseline Shots on Target Diff",
            "Venue Points Diff (Home vs Away)",
            "Short-Term Form Diff (Last 3)",
        ],
        "Advantage (Home - Away)": [
            f"{raw_features[0][3]:+.2f} goals/game",
            f"{raw_features[0][4]:+.2f} SOT/game",
            f"{raw_features[0][5]:+.2f} pts/game",
            f"{raw_features[0][0]:+.2f} pts/game",
        ],
    })