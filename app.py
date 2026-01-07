import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from flask import Flask, render_template
from sklearn.ensemble import GradientBoostingRegressor
from datetime import datetime

app = Flask(__name__)

def predict_ticker(ticker_symbol):
    # Fetch data (Multi-level index fix included)
    df = yf.download(ticker_symbol, period='1y', interval='1d', multi_level_index=False)
    df.columns = [str(col) for col in df.columns]
    
    # Technical Indicators
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    df.dropna(inplace=True)

    # Features for Prediction
    features = ['Close', 'RSI', 'EMA_20', 'ATR']
    X = df[features][:-1]
    y = df['Close'].shift(-1).dropna()
    
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X, y)

    # Logic for Percentages
    latest_data = df[features].tail(1)
    prediction = model.predict(latest_data)[0]
    current = df['Close'].iloc[-1]
    
    # Calculate Probability (Simplified Bayesian logic)
    # If RSI is neutral and price > EMA, Up-side probability increases
    rsi_val = df['RSI'].iloc[-1]
    base_prob = 50
    if prediction > current: base_prob += 15
    if rsi_val < 70: base_prob += 5
    if rsi_val < 30: base_prob += 10 # Oversold bounce
    
    up_prob = min(max(base_prob, 10), 90) # Cap between 10-90%
    down_prob = 100 - up_prob

    return {
        "symbol": "BANK NIFTY" if "BANK" in ticker_symbol else "NIFTY 50",
        "current": round(float(current), 2),
        "predicted": round(float(prediction), 2),
        "up_chance": up_prob,
        "down_chance": down_prob,
        "rsi": int(rsi_val)
    }

@app.route('/')
def home():
    try:
        nifty = predict_ticker('^NSEI')
        bank_nifty = predict_ticker('^NSEBANK')
        return render_template('index.html', nifty=nifty, bn=bank_nifty, time=datetime.now().strftime('%H:%M'))
    except Exception as e:
        return f"Market Data Sync Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
