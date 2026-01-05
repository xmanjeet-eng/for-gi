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
    
    # 2. Technical Indicators
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    
    # 3. MACD Calculation & Dynamic Column Handling
    macd = ta.macd(df['Close'])
    df = pd.concat([df, macd], axis=1)
    
    # Find the correct MACD Histogram column name automatically
    macd_h_col = [col for col in df.columns if 'MACDh' in col or 'MACDH' in col][0]
    
    # 4. Clean Data
    df.dropna(inplace=True)
    
    # 5. Feature Selection
    # We use these features to predict the Close of the NEXT day
    features = ['Close', 'RSI', 'EMA_20', 'EMA_50', 'ATR', macd_h_col]
    X = df[features]
    y = df['Close'].shift(-1).dropna()
    X = X[:-1] 
    
    # 6. Train Random Forest (Ensemble Learning)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, y)
    
    # 7. Generate Prediction
    latest_data_row = df[features].tail(1)
    prediction = model.predict(latest_data_row)[0]
    
    current_price = df['Close'].iloc[-1]
    change_pct = ((prediction - current_price) / current_price) * 100
    
    return {
        "current": round(float(current_price), 2),
        "predicted": round(float(prediction), 2),
        "change": round(float(change_pct), 2),
        "trend": "BULLISH" if change_pct > 0.05 else "BEARISH" if change_pct < -0.05 else "SIDEWAYS",
        "rsi": round(float(df['RSI'].iloc[-1]), 2),
        "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

@app.route('/')
def index():
    try:
        data = get_advanced_prediction()
        return render_template('index.html', data=data)
    except Exception as e:
        # Detailed error reporting for debugging
        return f"Market Data Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
