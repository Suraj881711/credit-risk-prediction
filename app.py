import os

import joblib
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, 'ml_credit.lb'))

# IMPORTANT: these are the exact features (and order) the saved model was
# trained on in ml_credit.ipynb:
#   job, marital, education, housing, age, balance, duration, campaign
# The previous version of this app sent unrelated fields (income,
# loan_amount, credit_score, ...) into the model, so its predictions
# were meaningless. The form now collects the real trained features.
FIELDS = {
    # name: (min, max, is_int)
    'job': (0, 12, True),          # job category code in the dataset
    'marital': (0, 3, True),       # marital status code in the dataset
    'education': (0, 4, True),     # education level code in the dataset
    'housing': (0, 2, True),       # housing-loan code in the dataset
    'age': (18, 100, True),
    'balance': (-10000, 200000, False),   # account balance
    'duration': (0, 10000, True),         # last contact duration (seconds)
    'campaign': (1, 60, True),            # number of contacts in campaign
}


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/health')
def health():
    return {'status': 'ok'}


@app.route('/predict', methods=['POST'])
def predict():
    values = {}
    for name, (lo, hi, is_int) in FIELDS.items():
        raw = request.form.get(name, '')
        try:
            value = int(raw) if is_int else float(raw)
        except (TypeError, ValueError):
            return render_template(
                'index.html',
                error=f'⚠️ "{name}" must be a number. Please check your entries.',
            )
        if not (lo <= value <= hi):
            return render_template(
                'index.html',
                error=f'⚠️ "{name}" must be between {lo} and {hi}.',
            )
        values[name] = value

    features = np.array([[values[name] for name in FIELDS]])
    prediction = int(model.predict(features)[0])

    confidence = None
    if hasattr(model, 'predict_proba'):
        confidence = round(
            float(model.predict_proba(features)[0][prediction]) * 100, 1)

    result = 'Approved ✅ — likely to be approved' if prediction == 1 \
        else 'Not Approved ❌ — unlikely to be approved'
    return render_template(
        'result.html', prediction=result, confidence=confidence, inputs=values)


if __name__ == '__main__':
    app.run(debug=True)
