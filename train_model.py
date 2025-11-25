import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==========================================
# 1. Synthetic Data Generation
# ==========================================
# INTERVIEWER NOTE: 
# In a production environment, we would scrape RBI Minutes (PDFs) 
# and map them to historical Nifty reaction dates.
# For this prototype, we simulate the logical correlation between 
# Market Fear (VIX) + Negative Sentiment -> Rate Hike Shocks.

np.random.seed(42)
n_samples = 500 # Simulating 10 years of weekly data/meetings

data = {
    # India VIX usually ranges from 10 (calm) to 30+ (high stress)
    'India_VIX': np.random.uniform(10, 30, n_samples),
    
    # Volume change in Nifty 50 (Proxy for market activity), range -2% to +2%
    'Nifty_Vol_Change': np.random.uniform(-2, 2, n_samples),
    
    # Sentiment score from -1 (Very Dovish/Negative) to 1 (Hawkish/Positive)
    'Gov_Speech_Sentiment': np.random.uniform(-1, 1, n_samples)
}

df = pd.DataFrame(data)

# ==========================================
# 2. Define Target Logic (The "Surprise" Event)
# ==========================================
# Logic: If the Market is fearful (VIX > 18) AND the Governor sounds negative/hawkish (< -0.3),
# there is a high probability of a Repo Rate Surprise.

def label_surprise(row):
    # We add some randomness so the model learns patterns, not just a hardcoded IF statement
    base_condition = (row['India_VIX'] > 18) and (row['Gov_Speech_Sentiment'] < -0.3)
    
    # 90% chance of surprise if condition met (Signal), 10% random noise
    if base_condition:
        return 1 if np.random.rand() > 0.1 else 0
    else:
        return 0 if np.random.rand() > 0.05 else 1 # Small chance of false alarm

df['Surprise_Event'] = df.apply(label_surprise, axis=1)

print("Data Generated. Sample Head:")
print(df.head())

# ==========================================
# 3. Train XGBoost Model
# ==========================================
# INTERVIEWER NOTE:
# We use XGBoost because it handles non-linear relationships better than Logistic Regression,
# especially for financial thresholds (e.g., VIX spiking).

X = df[['India_VIX', 'Nifty_Vol_Change', 'Gov_Speech_Sentiment']]
y = df['Surprise_Event']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    use_label_encoder=False, 
    eval_metric='logloss',
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1
)

model.fit(X_train, y_train)

# Validate
y_pred = model.predict(X_test)
print(f"\nModel Accuracy on Test Set: {accuracy_score(y_test, y_pred) * 100:.2f}%")

# ==========================================
# 4. Save the Model
# ==========================================
filename = 'rbi_surprise_model.pkl'
pickle.dump(model, open(filename, 'wb'))
print(f"Model saved as {filename}")