from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from dataloader import loadmatches
from features import build_dataset

def train_baseline_model():
  datasetdir = Path("./dataset")

  csv_files = sorted(datasetdir.glob("*.csv"))
  print(f"Found {len(csv_files)} season files.")
  print(csv_files)

  all_matches = []

  for file_path in csv_files:
    all_matches.extend(loadmatches(file_path))

  print(f"Total No:of matches loaded: {len(all_matches)}")

  X,y = build_dataset(all_matches,n=5)

  X = np.array(X)
  y = np.array(y)

  split_idx = int(len(X) * 0.8)

  X_train, X_test = X[:split_idx], X[split_idx:]
  y_train, y_test = y[:split_idx], y[split_idx:]

  print(f"Training on: {len(X_train)} matches")
  print(f"Testing on:  {len(X_test)} matches")
        
  model = LogisticRegression(max_iter=1000)
  model.fit(X_train, y_train)

  predictions = model.predict(X_test)
  acc = accuracy_score(y_test, predictions)

  print(f"\nModel Accuracy: {acc:.2%}")
  print("\nDetailed Performance Breakdown:")
  print(classification_report(y_test, predictions, zero_division=0))



if __name__ == "__main__":
  train_baseline_model()