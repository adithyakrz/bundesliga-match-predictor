from collections import defaultdict

def computeteamstats(history : list[dict],n: int = 5):
    if not history:
        return {
            "avg_points": 0.0,
            "avg_goals_scored": 0.0,
            "avg_goals_conceded": 0.0,
            "avg_goal_diff": 0.0,
            "avg_shots_target": 0.0,
        }
    recent = history[-n:]
    count = len(recent)

    avg_points = sum(m["points"] for m in recent) / count
    avg_goals_scored = sum(m["goals_scored"] for m in recent) / count
    avg_goals_conceded = sum(m["goals_conceded"] for m in recent) / count
    avg_shots_target = sum(m["shots_target"] for m in recent) / count

    return {
      "avg_points": avg_points,
      "avg_goals_scored": avg_goals_scored,
      "avg_goals_conceded": avg_goals_conceded,
      "avg_goal_diff": avg_goals_scored - avg_goals_conceded,
      "avg_shots_target": avg_shots_target,
  }

def extract_team_performance(match,team):
    ishome = match["home_team"] == team
    result = match["result"]

    if result == "D":
        points = 1
    elif (result == "H" and ishome) or (result == "A" and not ishome):
        points = 3
    else:
        points = 0

    goals_scored = match["home_goals"] if ishome else match["away_goals"]
    goals_conceded = match["away_goals"] if ishome else match["home_goals"]
    shots_target = (
      match["home_shots_target"] if ishome else match["away_shots_target"]
    )

    return {
      "points": points,
      "goals_scored": goals_scored,
      "goals_conceded": goals_conceded,
      "shots_target": shots_target,
    }

def build_dataset(matches,n=5):
    team_hist = defaultdict(list)
    home_hist = defaultdict(list)
    away_hist = defaultdict(list)
    X = []
    y = []

    feature_names = [
      "short_term_points_diff",
      "short_term_gd_diff",
      "baseline_points_diff",
      "baseline_gd_diff",
      "baseline_sot_diff",
      "venue_points_diff",
      "venue_gd_diff",
    ]
    for match in matches:
        home = match["home_team"]
        away = match["away_team"]
        h_s3 = computeteamstats(team_hist[home], n=3)
        a_s3 = computeteamstats(team_hist[away], n=3)

        h_s8 = computeteamstats(team_hist[home], n=8)
        a_s8 = computeteamstats(team_hist[away], n=8)

        h_venue = computeteamstats(home_hist[home], n=5)
        a_venue = computeteamstats(away_hist[away], n=5)

        features = [
        h_s3["avg_points"] - a_s3["avg_points"],
        h_s3["avg_goal_diff"] - a_s3["avg_goal_diff"],
        h_s8["avg_points"] - a_s8["avg_points"],
        h_s8["avg_goal_diff"] - a_s8["avg_goal_diff"],
        h_s8["avg_shots_target"] - a_s8["avg_shots_target"],
        h_venue["avg_points"] - a_venue["avg_points"],
        h_venue["avg_goal_diff"] - a_venue["avg_goal_diff"],
    ]

        X.append(features)
        y.append(match["result"])

        perf_h = extract_team_performance(match, home)
        perf_a = extract_team_performance(match, away)

        team_hist[home].append(perf_h)
        team_hist[away].append(perf_a)
        home_hist[home].append(perf_h)
        away_hist[away].append(perf_a)

    return X, y

if __name__ == "__main__":
    from pathlib import Path
    from dataloader import loadmatches

    testfile = "./dataset/2324.csv"

    matches = loadmatches(testfile)
    X, y = build_dataset(matches)

    print(f"Dataset generated! Total samples: {len(X)}")
    print("Sample 0 (Matchday 1 - no history):", X[0], "--> Result:", y[0])
    print("Sample 50 (Mid-season form):", X[50], "--> Result:", y[50])