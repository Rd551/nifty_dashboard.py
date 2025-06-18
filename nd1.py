import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os

# Streamlit UI settings
st.set_page_config(page_title="AI Stock Dashboard", layout="wide")
st.title("📈 AI Stock Movement Prediction & Pattern Analysis")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
ticker_input = st.sidebar.text_input("Enter NSE Symbol (e.g., ^NSEI, RELIANCE.NS)", value="^NSEI")
rsi_window = st.sidebar.slider("RSI Window", 5, 30, 14)
train_model = st.sidebar.checkbox("Retrain AI Model", value=False)
sequence_length = 60

# RSI Calculation
def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Pattern Detection
def detect_patterns(close_values):
    patterns = []
    recent = close_values[-10:]
    prev = close_values[-20:-10]
    if len(prev) < 10 or len(recent) < 10:
        return ["Not enough data"]
    if (prev[0] > prev[4] < prev[8]) and (recent[0] > recent[4] < recent[8]):
        patterns.append("🟢 Double Bottom")
    if (prev[0] < prev[4] > prev[8]) and (recent[0] < recent[4] > recent[8]):
        patterns.append("🔴 Double Top")
    if recent[-1] > max(prev):
        patterns.append("🚀 Breakout")
    return patterns if patterns else ["No obvious pattern"]

# Pattern Backtest
def pattern_backtest(close_values):
    hits, total = 0, 0
    for i in range(70, len(close_values) - 10):
        sub = close_values[i-20:i]
        now = close_values[i:i+10]
        if len(sub) < 9 or len(now) < 9:
            continue
        if sub[0] > sub[4] < sub[8] and now[0] > now[4] < now[8]:
            future = close_values[i+9]
            if future > now[-1]:
                hits += 1
            total += 1
    return (hits / total * 100) if total > 0 else None

# Load and prepare data
df = yf.download(ticker_input, period="5y", interval="1d", progress=False)
if df.empty:
    st.error("❌ No data found for symbol.")
    st.stop()

df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
df['RSI'] = compute_rsi(df, rsi_window)
df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
df.dropna(inplace=True)

# Prepare features
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
    st.warning("⚠️ Not enough data to train the model.")
    st.stop()

# Model handling
model_id = ticker_input.replace('^', 'INDEX_').replace('.', '_')
model_path = f"{model_id}_model.h5"

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

# Prediction
latest_input = df_scaled[-sequence_length:].values.reshape(1, sequence_length, 3)
prediction = model.predict(latest_input)[0][0]
movement = "🔼 Up" if prediction > 0.5 else "🔽 Down"

# Pattern accuracy
pattern_accuracy = pattern_backtest(df['Close'].values)

# Show Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Symbol", ticker_input.upper())
col2.metric("AI Prediction", movement)
col3.metric("Latest RSI", f"{df['RSI'].iloc[-1]:.2f}")
col4.metric("Pattern Accuracy", f"{pattern_accuracy:.2f}%" if pattern_accuracy else "N/A")

# Pattern Analysis
st.subheader("🧠 Pattern Detection")
for pattern in detect_patterns(df['Close'].values):
    st.write(f"- {pattern}")

# Candlestick Chart
st.subheader("🕯️ Candlestick Chart with VWAP")
df_recent = df[-120:].copy()
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df_recent.index,
    open=df_recent['Open'],
    high=df_recent['High'],
    low=df_recent['Low'],
    close=df_recent['Close'],
    name="Candlestick"
))
fig.add_trace(go.Scatter(
    x=df_recent.index,
    y=df_recent['VWAP'],
    line=dict(color='orange', dash='dot'),
    name='VWAP'
))
fig.update_layout(
    height=500,
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis_rangeslider_visible=False,
    template='plotly_white'
)
st.plotly_chart(fig, use_container_width=True)

st.caption("⚡ Powered by LSTM · Pattern Detection · VWAP · RSI · AI Insights")
