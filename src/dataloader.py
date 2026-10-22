import csv
from pathlib import Path
import os


def safe_int(value, default = 0) -> int:
  if not value or value.strip() == "":
    return default
  try:
    return int(value)
  except ValueError:
    return default
  
def loadmatches(csv_path):
    matches = []
    path = Path(csv_path)

    with open(path,mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)

        for row in reader:
            if not row.get("FTR"):
                continue

            else:
                cleandata = {
                    "date": row["Date"],
                    "home_team": row["HomeTeam"],
                    "away_team": row["AwayTeam"],
                    "home_goals": safe_int(row["FTHG"]),
                    "away_goals": safe_int(row["FTAG"]),
                    "result": row["FTR"],
                    "home_shots": safe_int(row["HS"]),
                    "away_shots": safe_int(row["AS"]),
                    "home_shots_target": safe_int(row["HST"]),
                    "away_shots_target": safe_int(row["AST"]),
                }
                matches.append(cleandata)
        return matches

if __name__ == "__main__":
    print(os.getcwd())
    testfile = Path("dataset/2627.csv")
    data = loadmatches(testfile)

    print(f"Total matches loaded: {len(data)}")
    print("\nFirst match sample:")
    print(data[0])

