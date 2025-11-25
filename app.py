import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import plotly.graph_objects as go
from transformers import pipeline
import nltk
import os
import re

# Download NLTK dependency (Kept robust to prevent your specific crash)
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except Exception:
    pass

st.set_page_config(layout="wide", page_title="RBI Surprise Detector")

# --- 1. MODEL LOADING & CACHING ---
@st.cache_resource
def load_assets():
    """Load the XGBoost model and Calibrator safely."""
    model_path = 'rbi_surprise_model.pkl'
    calibrator_path = 'rbi_calibrator.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(calibrator_path):
        return None, None
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(calibrator_path, 'rb') as f:
            calibrator = pickle.load(f)
        return model, calibrator
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None

@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", framework="pt", device=-1)

# --- 2. DATA FETCHING ---
def get_live_market_data(ticker_symbol):
    """Fetches VIX and calculates Nifty Volume change safely."""
    try:
        vix_ticker = yf.Ticker(ticker_symbol)
        vix_hist = vix_ticker.history(period="5d")
        current_vix = float(vix_hist['Close'].iloc[-1]) if not vix_hist.empty else 15.0

        nifty = yf.Ticker("^NSEI")
        nifty_hist = nifty.history(period="5d")
        
        if len(nifty_hist) >= 2:
            vol_today = float(nifty_hist['Volume'].iloc[-1])
            vol_yesterday = float(nifty_hist['Volume'].iloc[-2])
            vol_change = ((vol_today - vol_yesterday) / max(vol_yesterday, 1)) * 100
            # Tanh scaling to fit training range [-2, 2]
            vol_change_scaled = np.tanh(vol_change / 50.0) * 2
        else:
            vol_change_scaled = 0.0

        return current_vix, vol_change_scaled
    except Exception as e:
        st.warning(f"Market data fetch problem: {e}. Using default VIX=15.0.")
        return 15.0, 0.0

# --- 3. ROBUST SENTIMENT LOGIC ---
def safe_sent_tokenize(text):
    """Fallback tokenizer: Uses NLTK first, falls back to Regex if NLTK data is missing."""
    try:
        sents = nltk.tokenize.sent_tokenize(text)
        if len(sents) == 0: raise ValueError
        return sents
    except Exception:
        # Fallback: Split on punctuation followed by whitespace
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def sentence_sentiment_aggregate(text, nlp_pipeline, neutral_bias=-0.1, debug_mode=False):
    """Runs FinBERT per sentence and aggregates with critical inversion logic."""
    sents = safe_sent_tokenize(text)
    if not sents:
        return 0.0, []

    results = []
    debug_data = [] # Store raw data for UI
    
    batch_size = 8
    for i in range(0, len(sents), batch_size):
        batch = sents[i:i+batch_size]
        try:
            outs = nlp_pipeline(batch)
        except Exception:
            outs = [{"label": "NEUTRAL", "score": 0.5} for _ in batch]

        for sent, out in zip(batch, outs):
            label = out.get("label", "").lower()
            score = float(out.get("score", 0.5))
            
            # Debug collection
            if debug_mode:
                debug_data.append({"sentence": sent[:50]+"...", "raw_label": label, "raw_conf": score})

            if "pos" in label:
                canonical = "positive"
            elif "neg" in label:
                canonical = "negative"
            else:
                canonical = "neutral"
            results.append((canonical, score))

    total_weight = sum([s for _, s in results]) or 1.0
    agg = 0.0
    for label, score in results:
        if label == "positive":
            # CRITICAL INVERSION: Hawkish signal (Positive FinBERT) -> Negative score input
            agg += (-score) * score 
        elif label == "negative":
            # Market Panic (Negative FinBERT) -> Small Positive score input
            agg += (0.5 * score) * score
        else:
            agg += neutral_bias * score 

    agg = float(np.clip(agg / total_weight, -1.0, 1.0))
    return agg, debug_data

def simulate_ewma_baseline():
    """Simulates an EWMA Baseline (Placeholder for real historical data)."""
    return 0.0 

# --- 4. HARDENED PREDICTION WRAPPER ---
def safe_predict_proba(model, calibrator, X_df):
    """Generates uncalibrated proba, then applies Isotonic Regression calibration."""
    try:
        # 1. Get raw probability from XGBoost
        if hasattr(model, "predict_proba"):
            raw_prob = model.predict_proba(X_df)[:, 1]
        else:
            dmat = xgb.DMatrix(X_df)
            raw = model.predict(dmat)
            raw_prob = raw if raw.ndim == 1 else raw[:, -1]

        # 2. Apply Calibration
        calibrated_arr = calibrator.predict(raw_prob)
        return float(np.clip(calibrated_arr[0], 0.0, 1.0))
            
    except Exception as e:
        st.error(f"Model prediction error: {e}")
        return 0.0

# --- MAIN APP EXECUTION ---
st.title("🚨 RBI Surprise Detector")
st.markdown("A prototype using Alt-Data (VIX and FinBERT Sentiment) to predict unexpected Repo Rate shocks.")

# Load Assets
model, calibrator = load_assets()
nlp_pipeline = load_sentiment_pipeline()

if model is None:
    st.warning("⚠️ Model not loaded. Please run `python train_model.py` to generate the model files.")
    st.stop()

current_vix, current_vol_change = get_live_market_data("^INDIAVIX")

# --- RBI Logo & Header ---
col_logo, col_title = st.columns([1, 5])
with col_logo:
    # Use official RBI Logo from Wikimedia
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/1/17/Reserve_Bank_of_India_logo.svg/500px-Reserve_Bank_of_India_logo.svg.png", width=80)
with col_title:
    st.header("Predictive Dashboard")

# Metrics Row
col1, col2 = st.columns(2)
with col1:
    st.metric(label="1. India VIX (Fear Index)", value=f"{current_vix:.2f}", help="Live measure of market expected volatility. Threshold for high fear is >18 in training.")
with col2:
    st.metric(label="2. Nifty Vol Proxy (Scaled Score)", value=f"{current_vol_change:.2f}", help="Scaled volume change indicator (Tanh scale approx -2 to 2).")

# Input Area
st.markdown("---")
st.subheader("3. Governor's Speech Input")
user_text = st.text_area(
    "Paste the full text of the RBI Governor's recent statement or minutes:",
    "Inflation control remains the paramount priority for the committee. We are prepared to use all available tools to significantly tighten liquidity immediately, ensuring long-term financial stability.",
    height=200
)

# Sidebar
st.sidebar.title("Configuration")
# AUTOMATED EXPECTATION BASELINE
expected_sentiment_baseline = simulate_ewma_baseline() 
st.sidebar.markdown(f"**Baseline Expectation:** `{expected_sentiment_baseline:.2f}`")
st.sidebar.caption("Simulated EWMA of past speeches.")

# NEW: Debug Checkbox
show_debug = st.sidebar.checkbox("Show FinBERT Debug Info")

st.sidebar.markdown("---")
st.sidebar.subheader("Methodology")
st.sidebar.info("""
**Delta-First Logic:** Prediction based on (Current Sentiment - Baseline).

**Inverted Sentiment:** Positive FinBERT (Hawkish) -> Negative Model Input.

**Calibrated:** Output probabilities scaled via Isotonic Regression.
""")


# --- PREDICTION EXECUTION ---
if st.button("Analyze & Predict Surprise", type="primary"):
    if not user_text:
        st.warning("Please paste the Governor's speech text to analyze.")
        st.stop()
        
    st.divider()
    
    # 1. Calculate Current Sentiment
    with st.spinner("Analyzing text sentiment (FinBERT)..."):
        # Pass the debug flag
        sentiment_score, debug_data = sentence_sentiment_aggregate(user_text, nlp_pipeline, debug_mode=show_debug)
        
    # 2. Calculate Surprise Magnitude
    surprise_magnitude = sentiment_score - expected_sentiment_baseline
    
    # 3. Prepare Data for XGBoost
    input_data = pd.DataFrame(
        [[current_vix, current_vol_change, surprise_magnitude]],
        columns=['India_VIX', 'Nifty_Vol_Change', 'Gov_Speech_Sentiment']
    )
    
    # 4. Predict Probability (Calibrated)
    prediction_prob = safe_predict_proba(model, calibrator, input_data)
    
    # --- OUTPUT DISPLAY ---
    st.subheader("Prediction Result")
    
    col_raw_a, col_raw_b = st.columns(2)
    with col_raw_a:
        st.markdown(f"**Current Sentiment:** `{sentiment_score:.2f}`")
    with col_raw_b:
        st.markdown(f"**Surprise Magnitude:** `{surprise_magnitude:.2f}`")
        
    # NEW: Debug Expandable Section
    if show_debug and debug_data:
        with st.expander("🛠️ Debug: FinBERT Sentence Analysis"):
            st.write("Raw labels before inversion:")
            st.dataframe(debug_data)

    # Alert Box
    threshold = 0.60
    is_surprise = prediction_prob >= threshold

    if is_surprise:
        st.error(f"🚨 HIGH PROBABILITY OF SHOCK ({prediction_prob*100:.1f}%)")
        st.markdown("**Why?** High VIX + Significant Hawkish Surprise.")
    else:
        st.success(f"✅ NO SURPRISE EXPECTED ({prediction_prob*100:.1f}%)")
        st.markdown("**Why?** Market calm OR Speech matched expectations.")

    # Gauge Chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction_prob * 100,
        title={'text': "Calibrated Probability (%)"},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "#333"},
               'steps': [
                   {'range': [0, 30], 'color': "lightgreen"},
                   {'range': [30, 60], 'color': "yellow"},
                   {'range': [60, 100], 'color': "red"}
               ],
               'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': threshold*100}}
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)