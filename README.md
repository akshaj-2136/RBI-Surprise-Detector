📉 **RBI Surprise Detector**
Predicting Monetary Policy Shocks Using Market Fear + NLP on RBI Speeches
A Quant Research Prototype by Akshaj Kadamba
📌 **Overview**
Central bank surprises move markets violently.
This prototype attempts to predict unexpected RBI Repo Rate shocks using a combination of:
1.	Market Fear → India VIX
2.	Speech Tone Dynamics → FinBERT-based sentiment analysis of RBI Governor speeches
3.	Surprise Delta → Actual tone vs. expected tone (EWMA baseline)
4.	Calibrated ML Model → XGBoost + Isotonic Regression calibration
This dashboard is not a trading system but a quant-style research prototype built to demonstrate skill in:
•	Natural Language Processing (NLP)
•	Financial feature engineering
•	Event-based prediction
•	Probability calibration
•	Production-grade Streamlit engineering
🚀 **Live Demo**
https://rbi-surprise-detector-kgveulcyksfguphzaqkfxc.streamlit.app/
🧠** Model Architecture**
graph TD
    A[RBI Governor Speech] -->|Tokenization| B(Split into Sentences)
    B -->|FinBERT Model| C{Sentiment Analysis}
    C -->|Inverted Logic| D[Raw Sentiment Score]
    D -->|Subtract Baseline| E[Surprise Magnitude 'Delta']
    F[Live Market Data] -->|India VIX| G[Market Fear Input]
    E & G --> H{XGBoost Classifier}
    H -->|Raw Probability| I[Isotonic Calibration]
    I --> J[Final Shock Probability %]

🎯** Project Motivation**
Markets do not react to the absolute tone of central bank speeches — they react to the surprise.
Example:
If inflation is already high and markets expect hawkishness, a hawkish speech is not a shock. Traditional sentiment analysis fails here because it looks at absolute tone.
This project solves that with:
✔️ Sentence-Level Sentiment
Analyzing each sentence individually captures hawkish "spikes" that get diluted in full-text averages.
✔️ Delta-First Shock Logic
A shock is only predicted if:
(Current Sentiment – Expected Sentiment) << 0
AND
India VIX is elevated
✔️ Probability Calibration
Raw ML outputs are unreliable. We use Isotonic Regression to convert them into properly calibrated probabilities (e.g., a 70% confidence actually means the event happens 70% of the time).
🧩 **Key Innovations**
1. **The "Delta Trap" Fix**
Models fail if they only analyze the new speech. We compute expected sentiment (simulated via EWMA) to establish a baseline. The model predicts based on the Change vs. Expected, not the speech itself.
2. **Inverted Sentiment for Monetary Polic**y
FinBERT often classifies "inflation control actions" as Positive (because they are good for stability).
•	Reality: Markets interpret inflation action as Hawkish (Risk).
•	Our Logic: We apply domain inversion:
o	FinBERT Positive → Interpreted as Hawkish (Negative Input)
o	FinBERT Negative → Interpreted as Dovish/Panic (Positive Input)
🛠️ Tech Stack
•	Python 3.10+
•	Streamlit (UI/Dashboard)
•	XGBoost (Classification Model)
•	FinBERT via HuggingFace (Financial NLP)
•	Isotonic Regression (Probability Calibration)
•	NLTK (Robust Tokenization)
•	Yahoo Finance API (Live Market Data)
📂 **Project Structure**
rbi-surprise-detector/
│
├── app.py                     # Main Streamlit dashboard
├── train_model.py             # Model training + calibration pipeline
├── requirements.txt           # Python dependencies
├── rbi_surprise_model.pkl     # Trained XGBoost model
├── rbi_calibrator.pkl         # Isotonic regression calibrator
└── README.md                  # Documentation

🖥️ **How to Run Locally**
1.	Clone the repo
2.	git clone [https://github.com/yourusername/rbi-surprise-detector.git](https://github.com/yourusername/rbi-surprise-detector.git)
3.	cd rbi-surprise-detector

4.	Install dependencies
5.	pip install -r requirements.txt

6.	Train the model
(This generates the binary model files needed for the app)
python train_model.py

7.	Launch dashboard
8.	streamlit run app.py

⚠️ **Limitations** **(Read Carefully)**
1.	Synthetic Training Data: Since RBI MPC meetings occur only ~6 times a year, the sample size is too small for deep learning. This prototype uses synthetic data to demonstrate the logic pipeline, not to make real-world predictions.
2.	Simulated Baseline: The expected sentiment baseline is currently simulated using an EWMA function for demonstration purposes. A production version would use OIS curve implied expectations.
3.	FinBERT is Global: The model is trained on Reuters/SEC filings, not specific Indian central bank vernacular.
📬 **Contact**
Built by Akshaj Kadamba
Feel free to reach out for questions, collaboration, or improvements
