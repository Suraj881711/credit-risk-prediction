# 💳 Credit Risk Prediction — Approval Prediction App

A Flask web app that predicts whether a bank customer is likely to be
**approved** (target `approval` = 1) using a K-Nearest Neighbours (KNN)
classifier trained on customer/campaign data.

## What it does

You enter the eight customer features the model was trained on — job
category code, marital status code, education code, housing-loan code,
age, account balance, last-contact duration and number of campaign
contacts — and the app predicts **Approved / Not Approved**, shows the
model's confidence (share of the 5 nearest neighbours with that outcome),
and echoes your inputs back for checking.

## ⚠️ Important fix in this version

The original `app.py` collected *unrelated* fields (income, loan amount,
credit score, employment years, dependents…) and fed them into the model.
Because the field count happened to match, the app did not crash — but
every prediction was meaningless. The form and the app now use the exact
features, in the exact order, that `ml_credit.ipynb` trained the model on.

## Verified model performance

- **Test accuracy: 87.39%** (KNN, 80/20 train–test split,
  `random_state=42`, reproduced from `ml_credit.ipynb`)
- Note: the dataset is imbalanced — about 88.5% of rows are "not
  approved", so the model mostly learns the majority pattern. Treat the
  output as a demo of the ML pipeline, not financial advice.

## Tech stack

- Python, Flask
- scikit-learn (KNeighborsClassifier), pandas, NumPy, joblib

## How to run

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000 and submit the form.

## Files

- `app.py` — Flask routes (`/`, `/about`, `/predict`, `/health`)
- `ml_credit.lb` — trained KNN model
- `ml_credit.ipynb` — KNN training notebook
- `basic.ipynb` — Decision Tree comparison notebook (same dataset)
- `new_Cleaned_Credit.csv` — cleaned dataset (4,516 rows)

## Dataset note

Bank-marketing-style customer data with encoded categorical codes
(job 0–12, marital 0–3, education 0–4, housing 0–2), account balance and
campaign contact details, labelled with an `approval` outcome.
