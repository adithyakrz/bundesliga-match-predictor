import csv
from pathlib import Path
import os

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
                    "home_goals": int(row["FTHG"]),
                    "away_goals": int(row["FTAG"]),
                    "result": row["FTR"],
                    "home_shots": int(row["HS"]),
                    "away_shots": int(row["AS"]),
                    "home_shots_target": int(row["HST"]),
                    "away_shots_target": int(row["AST"]),
                }
                matches.append(cleandata)
        return matches
print(os.getcwd())
testfile = Path("dataset/2627.csv")
data = loadmatches(testfile)

print(f"Total matches loaded: {len(data)}")
print("\nFirst match sample:")
print(data[0])

