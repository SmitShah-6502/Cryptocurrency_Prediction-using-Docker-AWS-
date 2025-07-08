from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import requests
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
import numpy as np

app = Flask(__name__)

# Load the trained model
model = load_model("crypto_model.h5")

# Function to get real-time data from Binance API
def get_crypto_data(symbol="BTCUSDT", interval="1h", limit=100):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url).json()
    df = pd.DataFrame(response, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                         'close_time', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore'])
    df = df[['timestamp', 'open', 'high', 'low', 'close']].astype(float)
    return df

# Calculate trading recommendation based on predicted price and momentum
def get_trading_recommendation(df, predicted_price):
    current_price = df['close'].iloc[-1]
    price_change_percent = ((predicted_price - current_price) / current_price) * 100

    # Calculate short-term (5-period) and long-term (20-period) SMAs for momentum
    sma_short = df['close'].rolling(window=5).mean().iloc[-1]
    sma_long = df['close'].rolling(window=20).mean().iloc[-1]

    # Recommendation logic
    if price_change_percent > 2 and sma_short > sma_long:
        return "Buy"
    elif price_change_percent < -2 and sma_short < sma_long:
        return "Sell"
    else:
        return "Hold"

# Calculate price change percentage
def get_price_change_percent(df, predicted_price):
    current_price = df['close'].iloc[-1]
    return ((predicted_price - current_price) / current_price) * 100

# Calculate historical volatility
def get_volatility(df, periods=14):
    returns = df['close'].pct_change().dropna()
    volatility = returns.rolling(window=periods).std().iloc[-1] * np.sqrt(365) * 100
    return volatility

# Determine market trend
def get_market_trend(df):
    sma_short = df['close'].rolling(window=5).mean().iloc[-1]
    sma_long = df['close'].rolling(window=20).mean().iloc[-1]
    if sma_short > sma_long * 1.01:
        return "Bullish"
    elif sma_short < sma_long * 0.99:
        return "Bearish"
    else:
        return "Neutral"

# Fetch and analyze sentiment from X posts (simulated)
def get_sentiment(symbol):
    try:
        # Placeholder for X API call (simulated due to lack of real API access)
        sample_posts = [
            f"{symbol} is going to the moon! 🚀 #crypto",
            f"Not sure about {symbol}, market looks bearish 😔",
            f"Just bought some {symbol}! Excited! 😊"
        ]
        sentiments = [TextBlob(post).sentiment.polarity for post in sample_posts]
        avg_sentiment = np.mean(sentiments)
        if avg_sentiment > 0.1:
            return "Positive"
        elif avg_sentiment < -0.1:
            return "Negative"
        else:
            return "Neutral"
    except Exception:
        return "Unknown"

# Simulate portfolio returns
def simulate_portfolio(df, predicted_price, investment_amount=1000):
    current_price = df['close'].iloc[-1]
    units = investment_amount / current_price
    predicted_value = units * predicted_price
    return (predicted_value - investment_amount) / investment_amount * 100  # Return as percentage

# Calculate price alert thresholds
def get_price_alerts(df, predicted_price):
    volatility = get_volatility(df)
    current_price = df['close'].iloc[-1]
    volatility_factor = volatility / 100  # Convert to decimal
    buy_alert = current_price * (1 - volatility_factor * 0.5)  # 50% of volatility below current price
    sell_alert = current_price * (1 + volatility_factor * 0.5)  # 50% of volatility above current price
    return {"buy_alert": buy_alert, "sell_alert": sell_alert}

# Preprocess data for LSTM model
def preprocess_data(df):
    scaler = MinMaxScaler(feature_range=(0,1))
    df_scaled = scaler.fit_transform(df[['close']])
    X_input = df_scaled[-60:].reshape(1, 60, 1)
    return X_input, scaler

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["GET"])
def predict():
    try:
        symbol = request.args.get("symbol", "BTCUSDT").upper()
        crypto_data = get_crypto_data(symbol)
        X_input, scaler = preprocess_data(crypto_data)

        predicted_price = model.predict(X_input)
        predicted_price = float(scaler.inverse_transform(predicted_price)[0][0])

        # Get existing features
        recommendation = get_trading_recommendation(crypto_data, predicted_price)
        price_change_percent = get_price_change_percent(crypto_data, predicted_price)
        volatility = get_volatility(crypto_data)
        trend = get_market_trend(crypto_data)

        # Get new features
        sentiment = get_sentiment(symbol)
        portfolio_return = simulate_portfolio(crypto_data, predicted_price)
        alerts = get_price_alerts(crypto_data, predicted_price)

        return jsonify({
            "symbol": symbol,
            "predicted_price": predicted_price,
            "recommendation": recommendation,
            "price_change_percent": float(price_change_percent),
            "volatility": float(volatility),
            "trend": trend,
            "sentiment": sentiment,
            "portfolio_return": float(portfolio_return),
            "buy_alert": float(alerts["buy_alert"]),
            "sell_alert": float(alerts["sell_alert"])
        })
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)









