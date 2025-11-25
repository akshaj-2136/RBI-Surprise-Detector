import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score

# Set seed for reproducibility
np.random.seed(42)
n_samples = 500

# --- 1. Synthetic Data Generation ---
data = {
    'India_VIX': np.random.uniform(10, 30, n_samples),
    'Nifty_Vol_Change': np.random.uniform(-2, 2, n_samples),
    'Gov_Speech_Sentiment': np.random.uniform(-1, 1, n_samples),
    'Event_Timestamp': pd.to_datetime('2015-01-01') + pd.to_timedelta(np.arange(n_samples), unit='D')
}
df = pd.DataFrame(data)
df = df.sort_values('Event_Timestamp').reset_index(drop=True)

# --- 2. Synthetic Labeling Logic (The Core Hypothesis) ---
def label_surprise(row):
    base_condition = (row['India_VIX'] > 18) and (row['Gov_Speech_Sentiment'] < -0.3)
    if base_condition:
        return 1 if np.random.rand() > 0.1 else 0 
    else:
        return 0 if np.random.rand() > 0.05 else 1
df['Surprise_Event'] = df.apply(label_surprise, axis=1)

X = df[['India_VIX', 'Nifty_Vol_Change', 'Gov_Speech_Sentiment']]
y = df['Surprise_Event']

# --- 3. Time-Series Split for Validation ---
train_size = int(len(df) * 0.8)
X_train_raw, X_calib = X[:train_size], X[train_size:]
y_train_raw, y_calib = y[:train_size], y[train_size:]

# --- 4. Model Training (XGBoost) ---
model = xgb.XGBClassifier(
    use_label_encoder=False, 
    eval_metric='logloss',
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train_raw, y_train_raw)

# --- 5. Probability Calibration (Isotonic Regression) ---
raw_probs = model.predict_proba(X_calib)[:, 1]
calibrator = IsotonicRegression(out_of_bounds='clip')
calibrator.fit(raw_probs, y_calib)

# --- 6. Save Models ---
model_filename = 'rbi_surprise_model.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model, f)

calibrator_filename = 'rbi_calibrator.pkl'
with open(calibrator_filename, 'wb') as f:
    pickle.dump(calibrator, f)

print(f"Model saved to {model_filename}")
print(f"Calibrator saved to {calibrator_filename}")