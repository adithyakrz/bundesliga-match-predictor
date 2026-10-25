# ⚽ Bundesliga Match Outcome Engine

A machine-learning engine and interactive analytics dashboard for estimating Bundesliga match outcomes (Home Win / Draw / Away Win) from rolling team form, venue-specific performance, and head-to-head history.

🔗 **Live application:** [bundesliga-match-predictor.streamlit.app](https://bundesliga-match-predictor.streamlit.app/)

---

## Project Overview

This project builds pre-match Bundesliga outcome estimates from historical match data and serves them through an interactive Streamlit dashboard. The application combines a trained classifier with a tactical view of the selected matchup.

## Problems I Faced

- **Severe draw-class deficit:** The unweighted Random Forest reached 52.3% test accuracy but had 0.00 draw recall, effectively missing draws. I addressed the class imbalance with `class_weight="balanced"` in the logistic-regression model, which achieved 0.33 draw recall while reaching 0.56 recall for both home and away wins.
- **Building useful pre-match features:** A single form window did not capture the different time scales and contexts needed to compare teams. I engineered eight features from prior results only: points and goal-difference differentials over each team's last 3 matches; points, goal-difference, and shots-on-target differentials over the last 8; points and goal-difference differentials comparing the home team's last 5 home matches with the away team's last 5 away matches; and the home team's points per match across the last 4 head-to-head meetings.
- **Choosing a realistic evaluation target:** The unweighted model's 52.3% accuracy looked better as a headline number but concealed its failure to identify draws. On the chronological held-out set of 193 matches (from 963 total), class-balanced logistic regression achieved 50.8% accuracy, with recall of 0.56 for away wins, 0.33 for draws, and 0.56 for home wins. In a three-outcome sport, that more even recognition is more useful than accuracy from a model that ignores one outcome. These are classification results, not a formal probability-calibration study.
- **Making the forecast interpretable:** Probabilities alone did not show why a matchup might favor either club or be close. The dashboard adds a live tactical differentials table and an evenly contested fixture alert, alongside the home/draw/away probability display.
- **Stale Streamlit Cloud cache:** After code and serialized model updates were pushed, the deployed app continued to use legacy model state. The app's `@st.cache_resource` cache retained previously loaded model and scaler objects in memory. I cleared the Streamlit cache and rebooted the app so it loaded the updated artifact from disk.
- **Python and scikit-learn serialization differences:** Across local Python environments, I encountered `AttributeError: 'LogisticRegression' object has no attribute 'multi_class'` and `ModuleNotFoundError: No module named 'sklearn'`. The environments differed in their installed scikit-learn versions and dependencies. I rebuilt the serialized model using the same virtual environment and dependency versions as the deployment runtime, keeping model serialization and inference environments aligned.

## Tech Stack

- **Language:** Python 3.10+
- **Machine learning:** scikit-learn (`LogisticRegression`, `StandardScaler`, `HistGradientBoostingClassifier`, `RandomForestClassifier`)
- **Data processing:** NumPy and Python's standard library
- **Application:** Streamlit with custom CSS and interactive analytics
- **Deployment:** Streamlit Community Cloud

## Project Structure

```text
.
├── app.py                     # Streamlit application and dashboard
├── requirements.txt           # Python dependencies
├── dataset/                   # Historical Bundesliga match data
│   ├── 2324.csv
│   ├── 2425.csv
│   ├── 2526.csv
│   └── 2627.csv
├── models/
│   └── bundesliga_model.pkl   # Model, scaler, and historical feature data
├── src/
│   ├── dataloader.py          # Match CSV ingestion
│   ├── exportmode.py          # Model training and artifact export
│   ├── features.py            # Rolling form, venue, and H2H feature engineering
│   └── model.py               # Chronological evaluation and model comparisons
└── README.md
```

## Getting Started

### 1. Clone and set up the environment

```bash
git clone https://github.com/adithyakrz/bundesliga-match-predictor.git
cd bundesliga-match-predictor

python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run the app locally

```bash
streamlit run app.py
```

### 3. Evaluate or rebuild the model

Run the chronological model comparison:

```bash
python src/model.py
```

Rebuild the production model artifact after updating the data or feature pipeline:

```bash
python src/exportmode.py
```

The export script reads the CSV files in `dataset/` and writes the fitted model, scaler, and historical team data to `models/bundesliga_model.pkl`.

## Disclaimer

This project is intended for statistical analysis, machine-learning demonstration, and football exploration. Predictions are estimates, not certainties or financial or wagering advice.
