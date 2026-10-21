from collections import defaultdict

def computeteamstats(history : list[dict],n: int = 5):
    if not history:
        return {
            "avg_points": 0.0,
            "avg_goals_scored": 0.0,
            "avg_goals_conceded": 0.0,
            "avg_shots_target": 0.0,
        }
    recent = history[-n:]
    count = len(history)

    avg_points = sum(m["points"] for m in recent) / count
    avg_goals_scored = sum(m["goals_scored"] for m in recent) / count
    avg_goals_conceded = sum(m["goals_conceded"] for m in recent) / count
    avg_shots_target = sum(m["shots_target"] for m in recent) / count

    return {
      "avg_points": avg_points,
      "avg_goals_scored": avg_goals_scored,
      "avg_goals_conceded": avg_goals_conceded,
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
    team_histories = defaultdict(list)
    X = []
    y = []

    for match in matches:
        home = match["home_team"]
        away = match["away_team"]
        home_stats = computeteamstats(team_histories[home], n=n)
        away_stats = computeteamstats(team_histories[away], n=n)

        features = [
            home_stats["avg_points"] - away_stats["avg_points"],
            home_stats["avg_goals_scored"] - away_stats["avg_goals_scored"],
            home_stats["avg_goals_conceded"] - away_stats["avg_goals_conceded"],
            home_stats["avg_shots_target"] - away_stats["avg_shots_target"],
        ]

        X.append(features)
        y.append(match["result"])

        team_histories[home].append(extract_team_performance(match, home))
        team_histories[away].append(extract_team_performance(match, away))

    return X, y


from pathlib import Path
from dataloader import loadmatches
\
testfile = "./dataset/2324.csv"

matches = loadmatches(testfile)
X, y = build_dataset(matches)

print(f"Dataset generated! Total samples: {len(X)}")
print("Sample 0 (Matchday 1 - no history):", X[0], "--> Result:", y[0])
print("Sample 50 (Mid-season form):", X[50], "--> Result:", y[50])