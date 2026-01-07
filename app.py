import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from flask import Flask, render_template
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

app = Flask(__name__)

def get_advanced_prediction():
    # FIX 1: Use multi_level_index=False to prevent indexing errors
    df = yf.download('^NSEI', period='2y', interval='1d', multi_level_index=False)
    
    # Check if data actually exists
    if df.empty or len(df) < 50:
        raise ValueError("Not enough market data found. Check your internet connection.")

    # FIX 2: Ensure column names are standard
    df.columns = [str(col).capitalize() for col in df.columns]

    # Technical Indicators
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    
    # MACD Fix: Dynamically find the histogram column
    macd = ta.macd(df['Close'])
    df = pd.concat([df, macd], axis=1)
    macd_h_col = [col for col in df.columns if 'MACDh' in col or 'MACDH' in col][0]
    
    df.dropna(inplace=True)
    
    # Model Training
    features = ['Close', 'RSI', 'EMA_20', 'EMA_50', macd_h_col]
    X = df[features][:-1]
    y = df['Close'].shift(-1).dropna()
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Prediction
    latest_row = df[features].tail(1)
    prediction = model.predict(latest_row)[0]
    current_price = df['Close'].iloc[-1]
    
    return {
        "current": round(float(current_price), 2),
        "predicted": round(float(prediction), 2),
        "trend": "UP" if prediction > current_price else "DOWN",
        "time": datetime.now().strftime('%H:%M:%S')
    }

@app.route('/')
def index():
    try:
        data = get_advanced_prediction()
        return render_template('index.html', data=data)
    except Exception as e:
        return f"System Error: {str(e)}. Tip: Try updating yfinance with 'pip install -U yfinance'"

if __name__ == '__main__':
    app.run(debug=True)
