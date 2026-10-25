# Bundesliga Match Predictor

A simple machine learning project for predicting Bundesliga match outcomes using historical team form, goal difference, shot quality, and venue performance.

## Overview

This project uses a logistic regression model to estimate the probability of a home win, draw, or away win for a selected Bundesliga fixture. The model is trained on historical match data and packaged into a production artifact used by the Streamlit app.

## Features

- Predicts Home Win / Draw / Away Win probabilities
- Uses recent team form and rolling performance trends
- Includes venue-based context and tactical differentials
- Provides a lightweight web interface with Streamlit

## Tech Stack

- Python
- Streamlit
- scikit-learn
- NumPy
- pandas

## Project Structure

```text
.
├── app.py                     # Streamlit web application
├── requirements.txt          # Python dependencies
├── dataset/                  # Historical Bundesliga match data
│   ├── 2324.csv
│   ├── 2425.csv
│   ├── 2526.csv
│   └── 2627.csv
├── models/
│   └── bundesliga_model.pkl  # Trained model artifact used by the app
├── src/
│   ├── dataloader.py         # Loads and cleans match CSV data
│   ├── exportmode.py         # Builds and exports the trained model
│   ├── features.py           # Feature engineering for team form metrics
│   └── model.py              # Training logic for baseline model evaluation
└── README.md
```

## Getting Started

1. Clone the repository.
2. Create and activate a virtual environment if needed.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

Start the Streamlit application:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Model Training and Export

If you want to rebuild the packaged model artifact, run:

```bash
python src/exportmode.py
```

This reads all CSV files in `dataset/`, builds the feature set, trains the classifier, and saves the output to `models/bundesliga_model.pkl`.

## Data

The application relies on historical Bundesliga match statistics stored in the `dataset/` folder. The feature pipeline uses short-term and longer-term rolling form metrics, along with home/away venue strength, to generate predictions.

## Notes

This project is intended for learning, analysis, and experimentation in football match prediction. It is not a guaranteed forecasting system for real betting or wagering decisions.
