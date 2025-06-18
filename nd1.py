st.set_page_config(page_title="AI Stock Predictor", layout="wide")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# 🚨 Must be before any Streamlit output!
st.set_page_config(page_title="AI Stock Predictor", layout="wide")

st.title("📊 Stock Movement Prediction Dashboard (NSE)")

# --- Sidebar: Select Stock ---
st.sidebar.header("📌 Settings")
ticker_input = st.sidebar.text_input("Enter NSE Stock Symbol", value="^NSEI", help="Use 'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', or '^NSEI' for Nifty")
rsi_window = st.sidebar.slider("RSI Period", 5, 30, 14)
train_model = st.sidebar.checkbox("Retrain Model", value=False)

# --- Function: Calculate RSI ---
def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Load Data ---
try:
    df = yf.download(ticker_input, period="5y", interval="1d", progress=False)
    if df.empty:
        st.error("⚠️ Could not fetch data. Please check the ticker symbol.")
        st.stop()
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
df['RSI'] = compute_rsi(df, window=rsi_window)
df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
df.dropna(inplace=True)

# --- Scale Data ---
features = ['Close', 'RSI', 'VWAP']
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])
df_scaled = pd.DataFrame(scaled, columns=features, index=df.index)

# --- Create Sequences ---
sequence_length = 60
X = []
y = []
for i in range(sequence_length, len(df_scaled)):
    X.append(df_scaled.iloc[i-sequence_length:i].values)
    y.append(1 if df_scaled['Close'].iloc[i] > df_scaled['Close'].iloc[i-1] else 0)
X = np.array(X)
y = np.array(y)

# --- Load or Train Model ---
MODEL_PATH = f"{ticker_input.replace('^', '').replace('.', '_')}_model.h5"

if os.path.exists(MODEL_PATH) and not train_model:
    model = load_model(MODEL_PATH)
else:
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    model.save(MODEL_PATH)

# --- Predict Movement ---
latest_sequence = df_scaled[-60:].values.reshape(1, 60, 3)
prediction = model.predict(latest_sequence)[0][0]
predicted_movement = "🔼 Up" if prediction > 0.5 else "🔽 Down"

# --- Accuracy Metric ---
loss, acc = model.evaluate(X[-100:], y[-100:], verbose=0)

# --- Display Metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("📌 Stock", ticker_input.upper())
col2.metric("🔮 Predicted Movement", predicted_movement)
col3.metric("📊 Latest RSI", f"{df['RSI'].iloc[-1]:.2f}")
col4.metric("📈 Model Accuracy (last 100)", f"{acc * 100:.2f}%")

# --- Chart: Close, VWAP, RSI ---
st.subheader(f"📈 {ticker_input.upper()} Chart (Price + VWAP + RSI)")

fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(df.index[-120:], df['Close'].iloc[-120:], label='Close Price', color='blue')
ax1.plot(df.index[-120:], df['VWAP'].iloc[-120:], label='VWAP', color='orange', linestyle='--')
ax1.set_ylabel('Price')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(df.index[-120:], df['RSI'].iloc[-120:], label='RSI', color='green', alpha=0.5)
ax2.axhline(70, color='red', linestyle='--', linewidth=0.7)
ax2.axhline(30, color='red', linestyle='--', linewidth=0.7)
ax2.set_ylabel('RSI')
ax2.legend(loc='upper right')

st.pyplot(fig)

st.caption("📊 Powered by Yahoo Finance + TensorFlow LSTM. For education only.")
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

st.set_page_config(page_title="NIFTY 50 AI Prediction Dashboard", layout="wide")

# --- Function: Calculate RSI ---
def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Load Data ---
ticker = "^NSEI"
df = yf.download(ticker, period="5y", interval="1d")
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
df['RSI'] = compute_rsi(df)
df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
df.dropna(inplace=True)

# --- Scale Features ---
features = ['Close', 'RSI', 'VWAP']
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])
df_scaled = pd.DataFrame(scaled, columns=features, index=df.index)

# --- Create sequences ---
sequence_length = 60
X = []
y = []
for i in range(sequence_length, len(df_scaled)):
    X.append(df_scaled.iloc[i-sequence_length:i].values)
    y.append(1 if df_scaled['Close'].iloc[i] > df_scaled['Close'].iloc[i-1] else 0)
X = np.array(X)
y = np.array(y)

# --- Build Model ---
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=5, batch_size=32, verbose=0)

# --- Predict Today ---
latest_sequence = df_scaled[-60:].values.reshape(1, 60, 3)
prediction = model.predict(latest_sequence)[0][0]
predicted_movement = "🔼 Up" if prediction > 0.5 else "🔽 Down"

# --- Dashboard Layout ---
st.title("📈 NIFTY 50 Daily AI Prediction Dashboard")
st.markdown("---")

col1, col2, col3 = st.columns(3)
col1.metric("🔮 Predicted Movement", predicted_movement)
col2.metric("📊 Latest RSI", f"{df['RSI'].iloc[-1]:.2f}")
col3.metric("📉 Latest VWAP", f"{df['VWAP'].iloc[-1]:.2f}")

# --- Chart: Price, VWAP, RSI ---
st.subheader("📉 NIFTY Price Chart with VWAP and RSI")

fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(df.index[-120:], df['Close'].iloc[-120:], label='Close Price', color='blue')
ax1.plot(df.index[-120:], df['VWAP'].iloc[-120:], label='VWAP', color='orange', linestyle='--')
ax1.set_ylabel('Price')
ax1.legend(loc='upper left')

ax2 = ax1.twinx()
ax2.plot(df.index[-120:], df['RSI'].iloc[-120:], label='RSI', color='green', alpha=0.4)
ax2.axhline(70, color='red', linestyle='--', linewidth=0.7)
ax2.axhline(30, color='red', linestyle='--', linewidth=0.7)
ax2.set_ylabel('RSI')
ax2.legend(loc='upper right')

st.pyplot(fig)

st.caption("🔁 Auto-updated daily with latest data from Yahoo Finance.")
