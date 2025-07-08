import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import requests

# Fetching historical crypto data
def get_historical_data(symbol="BTCUSDT", interval="1h", limit=500):
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
    response = requests.get(url).json()
    df = pd.DataFrame(response, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 
                                         'close_time', 'qav', 'num_trades', 'taker_base', 'taker_quote', 'ignore'])
    df = df[['timestamp', 'open', 'high', 'low', 'close']].astype(float)
    return df

# Load and preprocess data
df = get_historical_data()
scaler = MinMaxScaler(feature_range=(0,1))
df_scaled = scaler.fit_transform(df[['close']])

# Create sequences for LSTM
X, y = [], []
time_step = 60  # Using last 60 prices for prediction
for i in range(len(df_scaled) - time_step):
    X.append(df_scaled[i:i+time_step])
    y.append(df_scaled[i+time_step])

X, y = np.array(X), np.array(y)

# Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X, y, epochs=20, batch_size=16)

# Save model
model.save("crypto_model.h5")