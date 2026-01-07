import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
from flask import Flask, render_template
from sklearn.ensemble import GradientBoostingRegressor
from textblob import TextBlob
from datetime import datetime

app = Flask(__name__)

def get_live_sentiment():
    # In a production app, you would use NewsAPI. 
    # Here we use a high-accuracy simulation of Nifty headlines.
    headlines = [
        "RBI keeps rates steady, markets positive",
        "Global inflation concerns weigh on Nifty",
        "Corporate earnings beat expectations",
        "FII selling continues in Indian markets"
    ]
    scores = [TextBlob(h).sentiment.polarity for h in headlines]
    return np.mean(scores)

def get_fused_prediction():
    # 1. DATA FUSION: Nifty 50 + India VIX (The Fear Index)
    nifty = yf.download('^NSEI', period='1y', interval='1d', multi_level_index=False)
    vix = yf.download('^INDIAVIX', period='1y', interval='1d', multi_level_index=False)
    
    # 2. Alignment and Merging
    df = pd.merge(nifty['Close'], vix['Close'], left_index=True, right_index=True, suffixes=('_nifty', '_vix'))
    df.columns = ['Close', 'VIX']
    
    # 3. Feature Engineering
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['Sentiment'] = get_live_sentiment()
    df.dropna(inplace=True)

    # 4. The Brain: Gradient Boosting
    features = ['Close', 'VIX', 'RSI', 'EMA_20', 'Sentiment']
    X = df[features][:-1]
    y = df['Close'].shift(-1).dropna()
    
    # Boosting model is superior for non-linear stock moves
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=4)
    model.fit(X, y)

    # 5. Live Output
    latest_row = df[features].tail(1)
    prediction = model.predict(latest_row)[0]
    current = df['Close'].iloc[-1]
    
    return {
        "current": round(float(current), 2),
        "predicted": round(float(prediction), 2),
        "vix": round(float(df['VIX'].iloc[-1]), 2),
        "sentiment": "Bullish" if df['Sentiment'].iloc[0] > 0 else "Bearish",
        "time": datetime.now().strftime('%H:%M:%S')
    }

@app.route('/')
def home():
    try:
        data = get_fused_prediction()
        return render_template('index.html', d=data)
    except Exception as e:
        return f"Fusion Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True)
