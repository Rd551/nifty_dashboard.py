import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Streamlit config
st.set_page_config(page_title="Pattern Sentiment Dashboard", layout="wide")
st.title("📊 Stock Pattern Sentiment Analysis")

# Sidebar
st.sidebar.header("⚙️ Settings")
ticker_input = st.sidebar.text_input("Enter NSE Symbol (e.g., ^NSEI, RELIANCE.NS)", value="^NSEI")
rsi_window = st.sidebar.slider("RSI Window", 5, 30, 14)

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

# Pattern Detection - Bullish or Bearish sentiment
def detect_sentiment_pattern(close_values):
    recent = close_values[-10:]
    prev = close_values[-20:-10]
    if len(prev) < 10 or len(recent) < 10:
        return "⚠️ Not enough data"

    # Bullish Double Bottom
    if (prev[0] > prev[4] < prev[8]) and (recent[0] > recent[4] < recent[8]):
        return "🟢 Bullish: Double Bottom"
    # Bearish Double Top
    if (prev[0] < prev[4] > prev[8]) and (recent[0] < recent[4] > recent[8]):
        return "🔴 Bearish: Double Top"
    # Breakout (Bullish)
    if recent[-1] > max(prev):
        return "🚀 Bullish: Breakout"

    return "🔍 No clear bullish/bearish pattern"

# Load data
df = yf.download(ticker_input, period="5y", interval="1d", progress=False)
if df.empty:
    st.error("No data found for symbol.")
    st.stop()

df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
df['RSI'] = compute_rsi(df, rsi_window)
df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
df.dropna(inplace=True)

# Display pattern sentiment
st.subheader("🧠 Pattern Sentiment")
sentiment = detect_sentiment_pattern(df['Close'].values)
st.write(sentiment)

# Chart
st.subheader("🕯️ Candlestick Chart with VWAP")
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

st.caption("🔎 Powered by Pattern Detection • VWAP • RSI")
