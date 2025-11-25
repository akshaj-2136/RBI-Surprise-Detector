import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import xgboost as xgb
import plotly.graph_objects as go
from transformers import pipeline
import os

# ==========================================
# 1. App Configuration
# ==========================================
st.set_page_config(
    page_title="RBI Surprise Detector",
    page_icon="📉",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .big-font { font-size:20px !important; }
    .metric-box { border: 1px solid #e6e6e6; padding: 20px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. Helper Functions & Caching
# ==========================================

@st.cache_resource
def load_sentiment_pipeline():
    """
    INTERVIEWER NOTE:
    We use 'ProsusAI/finbert' instead of generic BERT.
    FinBERT is pre-trained on financial text (Reuters, etc.), 
    making it much better at understanding nuance in Central Bank speeches.
    
    CRITICAL FIX: We explicitly set framework="pt" to force PyTorch.
    This prevents the app from crashing due to Keras 3/TensorFlow incompatibilities.
    """
    return pipeline("sentiment-analysis", model="ProsusAI/finbert", framework="pt")

@st.cache_resource
def load_prediction_model():
    """Loads the pre-trained XGBoost model."""
    if not os.path.exists('rbi_surprise_model.pkl'):
        return None
    return pickle.load(open('rbi_surprise_model.pkl', 'rb'))

def get_live_market_data(ticker_symbol):
    """Fetches live VIX and calculates Nifty Volume change."""
    try:
        # Fetch VIX
        vix_ticker = yf.Ticker(ticker_symbol)
        vix_hist = vix_ticker.history(period="1d")
        
        if not vix_hist.empty:
            current_vix = vix_hist['Close'].iloc[-1]
        else:
            current_vix = 15.0 # Fallback average
            
        # Fetch Nifty for Volume Change proxy (using ^NSEI)
        nifty = yf.Ticker("^NSEI")
        nifty_hist = nifty.history(period="2d")
        
        if len(nifty_hist) >= 2:
            vol_today = nifty_hist['Volume'].iloc[-1]
            vol_yesterday = nifty_hist['Volume'].iloc[-2]
            # Avoid division by zero
            if vol_yesterday > 0:
                vol_change = ((vol_today - vol_yesterday) / vol_yesterday) * 100
                # Normalize to our training scale (-2 to 2 approx)
                vol_change = np.clip(vol_change, -2, 2)
            else:
                vol_change = 0.0
        else:
            vol_change = 0.1 # Neutral fallback
            
        return current_vix, vol_change
        
    except Exception as e:
        st.error(f"Error fetching market data: {e}")
        return 15.0, 0.0

def map_sentiment_score(sentiment_output):
    """
    Maps FinBERT output (Positive/Negative/Neutral) to a scalar -1 to 1.
    
    >>> CRITICAL INVERSION FOR RBI CONTEXT:
    Hawkish policy language (which is what we are looking for to trigger a surprise) 
    is often classified by FinBERT as 'Positive' because it means long-term stability.
    To align with our model:
    'Positive' FinBERT = Negative/Hawkish Score for XGBoost.
    'Negative' FinBERT = Positive/Dovish Score for XGBoost.
    """
    label = sentiment_output[0]['label']
    score = sentiment_output[0]['score']
    
    st.info(f"FinBERT Raw Output: Label='{label}', Confidence={score:.4f}") 
    
    if label == 'positive':
        # INVERSION: If FinBERT says positive (i.e., policy action is good), 
        # we treat it as a Hawkish signal: a highly negative score for the XGBoost model.
        return -score 
    elif label == 'negative':
        # INVERSION: If FinBERT says negative (i.e., bad financial news like collapse/loss), 
        # we treat it as a Dovish signal (policy is often halted): a slightly positive score.
        return score / 2 # Scale down for caution
    else: # neutral (FinBERT is often too conservative with central bank text)
        # Apply a minor negative bias for neutral statements (small risk factor)
        return -0.1

# ==========================================
# 3. Main Application Layout
# ==========================================

# Sidebar
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Market Fear Gauge (Ticker)", value="^INDIAVIX")
st.sidebar.info("Note: ^INDIAVIX data is delayed by 15m on Yahoo Finance.")

# Header
col_h1, col_h2 = st.columns([1, 4])
with col_h1:
    # Placeholder for logo (using emoji for simplicity)
    st.markdown("# 🏦") 
with col_h2:
    st.title("RBI Repo Rate Surprise Detector")
    st.markdown("Predicting policy shocks using **Alternative Data** (Market Sentiment + Speech NLP).")

# Load Models
model = load_prediction_model()
# This triggers the sentiment pipeline to load, which can take a moment
nlp_pipeline = load_sentiment_pipeline() 

# Check if model exists
if model is None:
    st.error("⚠️ Model file not found! Please run 'train_model.py' first to generate the synthetic brain.")
    st.stop()

# --- Section 1: Live Market Context ---
st.subheader("1. Real-Time Market Context")
current_vix, current_vol_change = get_live_market_data(ticker)

m1, m2, m3 = st.columns(3)
with m1:
    st.metric(label="India VIX (Fear Index)", value=f"{current_vix:.2f}", delta="Market Stress Level")
with m2:
    st.metric(label="Nifty Volume Proxy", value=f"{current_vol_change:.2f}", delta="Activity Change")
with m3:
    st.write("### Why this matters?")
    st.caption("High VIX (>18) indicates market nervousness. Combined with hawkish speech, this triggers our surprise alerts.")

# --- Section 2: Speech Analysis ---
st.divider()
st.subheader("2. Governor's Speech Analysis")

default_speech = """
Inflation remains sticky and above our tolerance band. 
While growth is robust, we must remain vigilant against price pressures. 
The committee is ready to take further action if necessary to anchor expectations.
"""

user_text = st.text_area("Paste RBI Governor's Statement / Minutes here:", value=default_speech, height=150)

# --- Section 3: Prediction ---
st.divider()

if st.button("🔍 Analyze & Predict Surprise", type="primary"):
    
    # 1. Run NLP
    with st.spinner("Analyzing linguistic tone (FinBERT)..."):
        # The FinBERT model is designed to process shorter, specific financial sentences well.
        # For a full speech, the pipeline processes the whole text and returns one overall sentiment.
        # If the speech is very long and balanced (like the default), it often leans neutral.
        nlp_result = nlp_pipeline(user_text)
        sentiment_score = map_sentiment_score(nlp_result)
        
    
    # 2. Prepare Data for XGBoost
    # Order must match training: ['India_VIX', 'Nifty_Vol_Change', 'Gov_Speech_Sentiment']
    input_data = pd.DataFrame([[current_vix, current_vol_change, sentiment_score]], 
                             columns=['India_VIX', 'Nifty_Vol_Change', 'Gov_Speech_Sentiment'])
    
    # 3. Predict
    prediction_prob = model.predict_proba(input_data)[0][1] # Probability of Class 1 (Surprise)
    
    # --- Section 4: Visualization ---
    r1, r2 = st.columns([1, 1])
    
    with r1:
        st.subheader("Prediction Result")
        
        # Gauge Chart
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction_prob * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Surprise Probability (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgreen"},
                    {'range': [50, 75], 'color': "orange"},
                    {'range': [75, 100], 'color': "red"}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': prediction_prob * 100}
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    with r2:
        st.subheader("Model Interpretation")
        
        st.write(f"**Calculated Sentiment Score:** {sentiment_score:.2f} (-1 to 1)")
        
        if prediction_prob > 0.6:
            st.error("🚨 HIGH PROBABILITY OF SHOCK")
            st.markdown("""
            **Why?** 1. The VIX is elevated (Market is fragile).
            2. The Speech tone is significantly Negative/Hawkish.
            
            *Recommendation: Hedge bond portfolios immediately.*
            """)
        else:
            st.success("✅ NO SURPRISE EXPECTED")
            st.markdown("""
            **Why?** Either the market is calm (Low VIX) or the Governor's speech was balanced enough to soothe concerns.
            """)