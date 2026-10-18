def computeteamstats(history : list[dict],n : int):
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