import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# 🚨 Required at top
st.set_page_config(page_title="AI Stock Dashboard", layout="wide")
st.title("📊 AI Stock Movement Prediction & Pattern Analysis")

# --- Sidebar ---
st.sidebar.header("⚙️ Settings")
ticker_input = st.sidebar.text_input("Enter NSE Stock Symbol", value="^NSEI")
rsi_window = st.sidebar.slider("RSI Window", 5, 30, 14)
train_model = st.sidebar.checkbox("Retrain Model", value=False)
sequence_length = 60

# --- RSI Calculation ---
def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Pattern Detection ---
def detect_patterns(df):
    patterns = []
    recent = df['Close'].values[-10:]
    prev = df['Close'].values[-20:-10]

    if len(prev) < 10 or len(recent) < 10:
        return ["Not enough data"]

    if (prev[0] > prev[4] < prev[8]) and (recent[0] > recent[4] < recent[8]):
        patterns.append("🟢 Double Bottom")

    if (prev[0] < prev[4] > prev[8]) and (recent[0] < recent[4] > recent[8]):
        patterns.append("🔴 Double Top")

    if recent[-1] > max(prev):
        patterns.append("🚀 Breakout")

    ma10 = df['Close'].rolling(window=10).mean()
    if df['Close'].iloc[-1] < ma10.iloc[-1] and df['Close'].iloc[-2] > ma10.iloc[-2]:
        patterns.append("⚠️ Bearish Reversal")

    return patterns if patterns else ["No obvious pattern"]

# --- Pattern Backtest ---
def pattern_backtest(df):
    hits = 0
    total = 0
    for i in range(70, len(df) - 10):
        sub = df['Close'].iloc[i-20:i]
        now = df['Close'].iloc[i:i+10]
        # Double bottom check
        if sub[0] > sub[4] < sub[8] and now[0] > now[4] < now[8]:
            future = df['Close'].iloc[i+10]
            if future > now[-1]:
                hits += 1
            total += 1
    return (hits / total * 100) if total > 0 else None

# --- Download Data ---
try:
    df = yf.download(ticker_input, period="5y", interval="1d", progress=False)
    if df.empty:
        st.error("No data found for given symbol.")
        st.stop()
except Exception as e:
    st.error(f"Data fetch error: {e}")
    st.stop()

df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
df['RSI'] = compute_rsi(df, rsi_window)
df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
df.dropna(inplace=True)

# --- Prepare Data ---
features = ['Close', 'RSI', 'VWAP']
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])
df_scaled = pd.DataFrame(scaled, columns=features, index=df.index)

X, y = [], []
for i in range(sequence_length, len(df_scaled)):
    X.append(df_scaled.iloc[i-sequence_length:i].values)
    y.append(1 if df_scaled['Close'].iloc[i] > df_scaled['Close'].iloc[i-1] else 0)
X = np.array(X)
y = np.array(y)

if len(X) == 0:
    st.warning("Not enough data to train.")
    st.stop()

# --- Model Load/Train ---
model_name = ticker_input.replace('^', 'INDEX_').replace('.', '_')
model_path = f"{model_name}_model.h5"

if os.path.exists(model_path) and not train_model:
    model = load_model(model_path)
else:
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    model.save(model_path)

# --- Predict Today ---
latest_sequence = df_scaled[-sequence_length:].values.reshape(1, sequence_length, 3)
prediction = model.predict(latest_sequence)[0][0]
predicted_movement = "🔼 Up" if prediction > 0.5 else "🔽 Down"

# --- Backtest Pattern Accuracy ---
pattern_accuracy = pattern_backtest(df)

# --- Metrics ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Symbol", ticker_input.upper())
col2.metric("Prediction", predicted_movement)
col3.metric("Latest RSI", f"{df['RSI'].iloc[-1]:.2f}")
col4.metric("Pattern Hit Rate", f"{pattern_accuracy:.2f}%" if pattern_accuracy else "N/A")

# --- Pattern Output ---
st.subheader("📉 Pattern Screening")
for p in detect_patterns(df):
    st.write(f"- {p}")

# --- Candlestick Chart ---
st.subheader("📈 Candlestick Chart with VWAP")

fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df.index[-120:],
    open=df['Open'].iloc[-120:],
    high=df['High'].iloc[-120:],
    low=df['Low'].iloc[-120:],
    close=df['Close'].iloc[-120:],
    name="Candlestick"
))
fig.add_trace(go.Scatter(
    x=df.index[-120:],
    y=df['VWAP'].iloc[-120:],
    mode='lines',
    name='VWAP',
    line=dict(color='orange', dash='dot')
))
fig.update_layout(
    xaxis_rangeslider_visible=False,
    height=500,
    margin=dict(l=0, r=0, t=30, b=0)
)
st.plotly_chart(fig, use_container_width=True)

st.caption("📌 AI-driven predictions · VWAP & RSI analysis · Pattern recognition & backtesting")
