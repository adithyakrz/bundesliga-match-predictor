from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

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

  X,y = build_dataset(all_matches)

  X = np.array(X)
  y = np.array(y)

  split_idx = int(len(X) * 0.8)

  X_train, X_test = X[:split_idx], X[split_idx:]
  y_train, y_test = y[:split_idx], y[split_idx:]

  print(f"Training on: {len(X_train)} matches")
  print(f"Testing on:  {len(X_test)} matches")

  scaler = StandardScaler()

  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)


  tree_model = HistGradientBoostingClassifier(
    max_iter=100,
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=20,
    class_weight="balanced",
    random_state=42,
  )
  tree_model.fit(X_train_scaled, y_train)

  tree_preds = tree_model.predict(X_test_scaled)

  print(f"Tree Model Accuracy: {accuracy_score(y_test, tree_preds):.2%}\n")
  print(classification_report(y_test, tree_preds))

  
  #LOGISTIC REGRESSION
  model = LogisticRegression(C=0.1, max_iter=1000,class_weight="balanced")
  model.fit(X_train_scaled, y_train)

  predictions = model.predict(X_test_scaled)
  acc = accuracy_score(y_test, predictions)

  print(f"\nModel Accuracy: {acc:.2%}")
  print("\nDetailed Performance Breakdown:")
  print(classification_report(y_test, predictions, zero_division=0))

  #RANDOM FOREST

  rf = RandomForestClassifier(n_estimators=150, max_depth=4, random_state=42)
  rf.fit(X_train, y_train)
  rf_preds = rf.predict(X_test)
  rf_acc = accuracy_score(y_test, rf_preds)

  print("=" * 50)
  print(f"Random Forest (Depth 4) Accuracy: {rf_acc:.2%}")
  print("=" * 50)
  print(classification_report(y_test, rf_preds, zero_division=0))



if __name__ == "__main__":
  train_baseline_model()