import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from flask import Flask, render_template
from sklearn.ensemble import GradientBoostingRegressor
from textblob import TextBlob
from datetime import datetime

app = Flask(__name__)

# Simulating Global News Sentiment (High-performance mock for production)
def get_market_sentiment():
    headlines = [
        "FII flows into Indian markets remain steady",
        "Global inflation cooling provides relief to Nifty",
        "Bank Nifty faces resistance at key psychological levels",
        "IT sector outlook improves on AI demand"
    ]
    scores = [TextBlob(h).sentiment.polarity for h in headlines]
    return round(np.mean(scores), 2)

def calculate_intelligence(ticker, vix_data):
    # Fetch 1 year of data
    df = yf.download(ticker, period='1y', interval='1d', multi_level_index=False)
    df.columns = [str(col) for col in df.columns]
    
    # Merge with VIX (Volatility)
    df['VIX'] = vix_data['Close']
    
    # Technical Engine
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df['Sentiment'] = get_market_sentiment()
    df.dropna(inplace=True)

    # Machine Learning Model (Gradient Boosting)
    features = ['Close', 'VIX', 'RSI', 'EMA_20', 'ATR', 'Sentiment']
    X = df[features][:-1]
    y = df['Close'].shift(-1).dropna()
    
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.07, max_depth=5)
    model.fit(X, y)

    # Live Prediction
    latest = df[features].tail(1)
    pred = model.predict(latest)[0]
    curr = df['Close'].iloc[-1]
    
    # Probability Logic
    prob_up = 50
    if pred > curr: prob_up += 20
    if df['RSI'].iloc[-1] < 40: prob_up += 10 # Oversold bounce probability
    prob_up = min(max(prob_up, 15), 85) # Cap between 15%-85%

    return {
        "name": "NIFTY 50" if "^NSEI" in ticker else "BANK NIFTY",
        "current": f"{curr:,.2f}",
        "predicted": f"{pred:,.2f}",
        "up": prob_up,
        "down": 100 - prob_up,
        "rsi": int(df['RSI'].iloc[-1]),
        "vix": round(df['VIX'].iloc[-1], 2),
        "sentiment": "BULLISH" if df['Sentiment'].iloc[0] > 0 else "BEARISH"
    }

@app.route('/')
def index():
    try:
        vix_raw = yf.download('^INDIAVIX', period='1y', interval='1d', multi_level_index=False)
        nifty = calculate_intelligence('^NSEI', vix_raw)
        bank = calculate_intelligence('^NSEBANK', vix_raw)
        return render_template('index.html', nifty=nifty, bank=bank, ts=datetime.now().strftime('%H:%M:%S'))
    except Exception as e:
        return f"Terminal Offline: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
