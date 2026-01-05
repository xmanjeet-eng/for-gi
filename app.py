import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
from flask import Flask, render_template
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

app = Flask(__name__)

def get_advanced_prediction():
    # 1. Fetch Nifty 50 Live Data
    df = yf.download('^NSEI', period='2y', interval='1d')
    
    # 2. Feature Engineering (Technical Indicators)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    # Average True Range for volatility
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    # MACD
    macd = ta.macd(df['Close'])
    df = pd.concat([df, macd], axis=1)
    
    # Drop rows with NaN values created by indicators
    df.dropna(inplace=True)
    
    # 3. Prepare Features (X) and Target (y)
    # We use indicators to predict the NEXT day's Close
    features = ['Close', 'RSI', 'EMA_20', 'EMA_50', 'ATR', 'MACDH_12_26_9']
    X = df[features]
    y = df['Close'].shift(-1).dropna()
    X = X[:-1] # Match length with y
    
    # 4. Train Ensemble Model (Random Forest)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # 5. Predict for Tomorrow
    latest_data_row = df[features].tail(1)
    prediction = model.predict(latest_data_row)[0]
    
    current_price = df['Close'].iloc[-1]
    change_pct = ((prediction - current_price) / current_price) * 100
    
    return {
        "current": round(float(current_price), 2),
        "predicted": round(float(prediction), 2),
        "change": round(float(change_pct), 2),
        "trend": "BULLISH" if change_pct > 0.1 else "BEARISH" if change_pct < -0.1 else "SIDEWAYS",
        "rsi": round(float(df['RSI'].iloc[-1]), 2),
        "time": datetime.now().strftime('%H:%M:%S')
    }

@app.route('/')
def index():
    try:
        data = get_advanced_prediction()
        return render_template('index.html', data=data)
    except Exception as e:
        return f"Market Data Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
