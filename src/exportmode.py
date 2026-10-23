from collections import defaultdict
from pathlib import Path
import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from dataloader import loadmatches
from features import build_dataset, computeteamstats, extract_team_performance

def export_production_details():
    datasetdir = Path("./dataset")
    models_dir = Path("./models")
    models_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(datasetdir.glob("*.csv"))
    all_matches = []
    for f in csv_files:
        all_matches.extend(loadmatches(f))
    print(f"Loaded {len(all_matches)} matches...")

    X_raw, y_raw = build_dataset(all_matches)
    X = np.array(X_raw)
    y = np.array(y_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(C=0.1, max_iter=1000)
    model.fit(X_scaled, y)

    team_hist = defaultdict(list)
    home_hist = defaultdict(list)
    away_hist = defaultdict(list)

    for match in all_matches:
        h, a = match["home_team"], match["away_team"]
        perf_h = extract_team_performance(match, h)
        perf_a = extract_team_performance(match, a)
        team_hist[h].append(perf_h)
        team_hist[a].append(perf_a)
        home_hist[h].append(perf_h)
        away_hist[a].append(perf_a)

    teams = sorted(list(team_hist.keys()))

    artifacts = {
        "model": model,
        "scaler": scaler,
        "team_hist": dict(team_hist),
        "home_hist": dict(home_hist),
        "away_hist": dict(away_hist),
        "teams": teams,
    }
    artifact_path = models_dir / "bundesliga_model.pkl"
    with open(artifact_path, "wb") as f:
        pickle.dump(artifacts, f)

    print(f"Production artifact saved successfully to: {artifact_path}")
    print(f"Total active Bundesliga clubs packaged: {len(teams)}")


if __name__ == "__main__":
  export_production_details()