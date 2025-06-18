import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# Streamlit config
st.set_page_config(page_title="AI Stock Dashboard", layout="wide")
st.title("\U0001F4C8 AI Stock Movement Prediction & Pattern Analysis")

# Sidebar
st.sidebar.header("\u2699\ufe0f Settings")
ticker_input = st.sidebar.text_input("Enter NSE Symbol (e.g., ^NSEI, RELIANCE.NS)", value="^NSEI")
rsi_window = st.sidebar.slider("RSI Window", 5, 30, 14)
train_model = st.sidebar.checkbox("Retrain AI Model", value=False)
sequence_length = 60

# RSI Calculation
def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Pattern Detection
def detect_patterns(df):
    patterns = []
    recent = df['Close'].values[-10:]
    prev = df['Close'].values[-20:-10]
    if len(prev) < 10 or len(recent) < 10:
        return ["Not enough data"]
    if (prev[0] > prev[4] < prev[8]) and (recent[0] > recent[4] < recent[8]):
        patterns.append("\U0001F7E2 Double Bottom")
    if (prev[0] < prev[4] > prev[8]) and (recent[0] < recent[4] > recent[8]):
        patterns.append("\U0001F534 Double Top")
    if recent[-1] > max(prev):
        patterns.append("\U0001F680 Breakout")
    ma10 = df['Close'].rolling(window=10).mean()
    if df['Close'].iloc[-1] < ma10.iloc[-1] and df['Close'].iloc[-2] > ma10.iloc[-2]:
        patterns.append("\u26A0\ufe0f Bearish Reversal")
    return patterns if patterns else ["No obvious pattern"]

# Pattern Backtest
def pattern_backtest(df):
    hits = 0
    total = 0
    for i in range(70, len(df) - 10):
        sub = df['Close'].iloc[i-20:i].values
        now = df['Close'].iloc[i:i+10].values
        if len(sub) < 9 or len(now) < 9:
            continue
        if sub[0] > sub[4] < sub[8] and now[0] > now[4] < now[8]:
            future = df['Close'].iloc[i+10]
            if future > now[-1]:
                hits += 1
            total += 1
    return (hits / total * 100) if total > 0 else None

# Load data
df = yf.download(ticker_input, period="5y", interval="1d", progress=False)
if df.empty:
    st.error("No data found for symbol.")
    st.stop()

df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
df['RSI'] = compute_rsi(df, rsi_window)
df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
df.dropna(inplace=True)

# Preprocess
target_features = ['Close', 'RSI', 'VWAP']
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[target_features])
df_scaled = pd.DataFrame(scaled, columns=target_features, index=df.index)

X, y = [], []
for i in range(sequence_length, len(df_scaled)):
    X.append(df_scaled.iloc[i-sequence_length:i].values)
    y.append(1 if df_scaled['Close'].iloc[i] > df_scaled['Close'].iloc[i-1] else 0)
X = np.array(X)
y = np.array(y)

if len(X) == 0:
    st.warning("Not enough data to train.")
    st.stop()

# Train or load model
model_id = ticker_input.replace('^', 'INDEX_').replace('.', '_')
model_file = f"{model_id}_model.h5"
if os.path.exists(model_file) and not train_model:
    model = load_model(model_file)
else:
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    model.save(model_file)

# Predict
latest_input = df_scaled[-sequence_length:].values.reshape(1, sequence_length, 3)
prediction = model.predict(latest_input)[0][0]
movement = "\U0001F53C Up" if prediction > 0.5 else "\U0001F53D Down"

# Pattern accuracy
pattern_accuracy = pattern_backtest(df)

# Metrics
t1, t2, t3, t4 = st.columns(4)
t1.metric("Symbol", ticker_input.upper())
t2.metric("AI Prediction", movement)
t3.metric("Latest RSI", f"{df['RSI'].iloc[-1]:.2f}")
t4.metric("Pattern Accuracy", f"{pattern_accuracy:.2f}%" if pattern_accuracy else "N/A")

# Patterns
t5 = st.container()
t5.subheader("\U0001F9E0 Pattern Detection")
for pattern in detect_patterns(df):
    t5.write(f"- {pattern}")

# Chart
st.subheader("\U0001F4C9 Candlestick Chart with VWAP")
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index[-120:],
    open=df['Open'].iloc[-120:],
    high=df['High'].iloc[-120:],
    low=df['Low'].iloc[-120:],
    close=df['Close'].iloc[-120:],
    name='Candlestick'
))
fig.add_trace(go.Scatter(
    x=df.index[-120:],
    y=df['VWAP'].iloc[-120:],
    line=dict(color='orange', width=1, dash='dot'),
    name='VWAP'
))
fig.update_layout(
    xaxis_rangeslider_visible=False,
    height=500,
    margin=dict(l=10, r=10, t=30, b=10),
    template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

st.caption("\u26A1 Powered by LSTM · Pattern screening · VWAP & RSI indicators")
