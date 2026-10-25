from pathlib import Path
import pickle
import numpy as np
import streamlit as st

from src.features import computeteamstats, compute_h2h

# Page configuration
st.set_page_config(
    page_title="Bundesliga Match Predictor | Opta-Style Analytics",
    page_icon="⚽",
    layout="centered",
)

st.markdown(
    """
<style>
    /* Global Charcoal Background & Typography */
    .stApp {
        background-color: #121417;
        color: #e5e9f0;
    }
    
    /* Headers & Labels */
    h1, h2, h3, h4, p, span, label {
        color: #e5e9f0 !important;
    }
    
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8 !important;
    }
    
    /* Selectboxes and Interactive Elements */
    div[data-baseweb="select"] > div {
        background-color: #1a1d21 !important;
        border-color: #2e3440 !important;
        color: #e5e9f0 !important;
    }
    
    /* Primary Action Button (Forest Green with subtle hover) */
    div.stButton > button:first-child {
        background-color: #1e4630 !important;
        border: 1px solid #275c3f !important;
        color: #f8fafc !important;
        font-weight: 600;
        border-radius: 6px;
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:first-child:hover {
        background-color: #275c3f !important;
        border-color: #2ecc71 !important;
        color: #ffffff !important;
    }

    /* 3-Segment Match Probability Bar */
    .prob-container {
        display: flex;
        height: 24px;
        width: 100%;
        border-radius: 6px;
        overflow: hidden;
        margin: 12px 0 24px 0;
        background-color: #1a1d21;
        border: 1px solid #2e3440;
        font-size: 0.8rem;
        font-weight: 600;
        line-height: 24px;
        color: #ffffff;
        text-align: center;
    }
    .bar-home { background-color: #1e4630; }  /* Forest Green */
    .bar-draw { background-color: #2e3440; }  /* Charcoal Slate */
    .bar-away { background-color: #8a1824; }  /* Crimson Red Accent */

    /* Crimson Accent Warning Banner */
    .alert-banner {
        background-color: rgba(138, 24, 36, 0.2);
        border: 1px solid #8a1824;
        border-left: 5px solid #a82331;
        border-radius: 6px;
        padding: 12px 16px;
        margin: 16px 0;
        color: #fecdd3;
        font-size: 0.88rem;
    }

    /* Tactical Differential Table */
    .diff-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
        font-size: 0.9rem;
        background-color: #1a1d21;
        border-radius: 6px;
        overflow: hidden;
    }
    .diff-table th {
        text-align: left;
        padding: 10px 12px;
        background-color: #15181c;
        border-bottom: 2px solid #2e3440;
        color: #94a3b8;
        font-weight: 600;
    }
    .diff-table td {
        padding: 10px 12px;
        border-bottom: 1px solid #262a30;
        color: #cbd5e1;
    }

    /* Advantage Badges */
    .badge-adv {
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .adv-home { 
        background-color: rgba(30, 70, 48, 0.35); 
        color: #55d688; 
        border: 1px solid rgba(46, 204, 113, 0.3);
    }
    .adv-away { 
        background-color: rgba(138, 24, 36, 0.3); 
        color: #ff6b81; 
        border: 1px solid rgba(168, 35, 49, 0.4);
    }
    .adv-neutral { 
        background-color: rgba(60, 68, 80, 0.3); 
        color: #94a3b8; 
        border: 1px solid #334155;
    }
</style>
""",
    unsafe_allow_html=True,
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
h2h_hist = artifacts.get("h2h_hist", {})
teams = artifacts["teams"]

# Modern 2-Column Header with Official Logo
head_col1, head_col2 = st.columns([1, 5], vertical_alignment="center")

with head_col1:
  st.image(
      "https://upload.wikimedia.org/wikipedia/en/d/df/Bundesliga_logo_%282017%29.svg",
      width=75,
  )

with head_col2:
  st.markdown(
      "<h2 style='margin:0; padding:0; line-height: 1.1;'>Bundesliga Match"
      " Outcome Engine</h2>",
      unsafe_allow_html=True,
  )
  st.caption(
      "Class-balanced logistic regression using rolling metrics, venue splits,"
      " and head-to-head dynamics. Data current through October 2026."
  )

st.write("---")
st.write("")

# Team Selectors
col1, col2 = st.columns(2)
with col1:
  default_home = (
      teams.index("Bayern Munich") if "Bayern Munich" in teams else 0
  )
  home_team = st.selectbox("Home Team", teams, index=default_home, key="home")

with col2:
  default_away = (
      teams.index("Dortmund")
      if "Dortmund" in teams
      else (1 if len(teams) > 1 else 0)
  )
  away_team = st.selectbox("Away Team", teams, index=default_away, key="away")

if home_team == away_team:
  st.warning("Please choose two different clubs to generate a matchup.")
else:
  # Metric extraction
  h_s3 = computeteamstats(team_hist[home_team], n=3)
  a_s3 = computeteamstats(team_hist[away_team], n=3)
  h_s8 = computeteamstats(team_hist[home_team], n=8)
  a_s8 = computeteamstats(team_hist[away_team], n=8)
  h_venue = computeteamstats(home_hist[home_team], n=5)
  a_venue = computeteamstats(away_hist[away_team], n=5)

  h2h_pts = compute_h2h(h2h_hist, home_team, away_team, n=4)

  # Construct the full 8-element feature array
  raw_features = np.array([[
      h_s3["avg_points"] - a_s3["avg_points"],
      h_s3["avg_goal_diff"] - a_s3["avg_goal_diff"],
      h_s8["avg_points"] - a_s8["avg_points"],
      h_s8["avg_goal_diff"] - a_s8["avg_goal_diff"],
      h_s8["avg_shots_target"] - a_s8["avg_shots_target"],
      h_venue["avg_points"] - a_venue["avg_points"],
      h_venue["avg_goal_diff"] - a_venue["avg_goal_diff"],
      h2h_pts,
  ]])

  scaled_features = scaler.transform(raw_features)
  probs = model.predict_proba(scaled_features)[0]
  prob_map = dict(zip(model.classes_, probs))

  p_home = float(prob_map["H"])
  p_draw = float(prob_map["D"])
  p_away = float(prob_map["A"])

  if st.button("Generate Matchup Forecast", use_container_width=True):
    st.write("")
    st.subheader(f"{home_team} vs {away_team}")

    # Metrics summary cards
    m1, m2, m3 = st.columns(3)
    m1.metric(f"{home_team} Win", f"{p_home:.1%}")
    m2.metric("Draw", f"{p_draw:.1%}")
    m3.metric(f"{away_team} Win", f"{p_away:.1%}")

    # Custom 3-Segment Probability Bar
    st.markdown(
        f"""
        <div class="prob-container">
            <div class="bar-home" style="width: {p_home*100:.1f}%;">{p_home*100:.0f}%</div>
            <div class="bar-draw" style="width: {p_draw*100:.1f}%;">{p_draw*100:.0f}%</div>
            <div class="bar-away" style="width: {p_away*100:.1f}%;">{p_away*100:.0f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Crimson Alert Box for Tightly Contested Games
    prob_diff = abs(p_home - p_away)
    if prob_diff < 0.12 or p_draw > 0.26:
      st.markdown(
          """
          <div class="alert-banner">
              <strong>Evenly Contested Fixture:</strong> Historical chance generation and venue performance indicate near parity. Elevated probability of a draw or narrow 1-goal margin.
          </div>
          """,
          unsafe_allow_html=True,
      )

    # Tactical Differentials Table
    st.write("")
    st.markdown("#### Tactical Differentials (Pre-Match)")

    def get_adv_badge(val):
      if abs(val) < 0.05:
        return '<span class="badge-adv adv-neutral">Even</span>'
      elif val > 0:
        return f'<span class="badge-adv adv-home">+{val:.2f} {home_team}</span>'
      else:
        return (
            f'<span class="badge-adv adv-away">+{abs(val):.2f}'
            f" {away_team}</span>"
        )

    def get_h2h_badge(pts):
      if abs(pts - 1.35) < 0.25:
        return '<span class="badge-adv adv-neutral">Historical Parity</span>'
      elif pts > 1.35:
        return (
            f'<span class="badge-adv adv-home">{pts:.2f} pts/gm'
            f" ({home_team})</span>"
        )
      else:
        return (
            f'<span class="badge-adv adv-away">{3.0 - pts:.2f} pts/gm'
            f" ({away_team})</span>"
        )

    table_html = f"""
    <table class="diff-table">
        <thead>
            <tr>
                <th>Factor</th>
                <th>Delta / Metric</th>
                <th>Advantage</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Baseline Shots on Target (Last 8)</td>
                <td>{raw_features[0][4]:+.2f} / match</td>
                <td>{get_adv_badge(raw_features[0][4])}</td>
            </tr>
            <tr>
                <td>Baseline Goal Difference (Last 8)</td>
                <td>{raw_features[0][3]:+.2f} / match</td>
                <td>{get_adv_badge(raw_features[0][3])}</td>
            </tr>
            <tr>
                <td>Venue Form Delta (Home vs Away)</td>
                <td>{raw_features[0][5]:+.2f} pts / match</td>
                <td>{get_adv_badge(raw_features[0][5])}</td>
            </tr>
            <tr>
                <td>Recent Momentum (Last 3)</td>
                <td>{raw_features[0][0]:+.2f} pts / match</td>
                <td>{get_adv_badge(raw_features[0][0])}</td>
            </tr>
            <tr>
                <td>Head-to-Head Point Rate (Last 4 Mtgs)</td>
                <td>{raw_features[0][7]:.2f} pts earned by Home</td>
                <td>{get_h2h_badge(raw_features[0][7])}</td>
            </tr>
        </tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)

# Footer
st.write("---")
st.markdown(
    "<small style='color: #64748b;'>Balanced logistic regression (C=0.1).</small><br>"
    "<small style='color: #64748b;'>Trained using the help of 8 features</small>",
    unsafe_allow_html=True,
)