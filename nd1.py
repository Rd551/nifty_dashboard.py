import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime

# Streamlit config
st.set_page_config(page_title="Option Selling Entry Dashboard", layout="wide")
st.title("📊 Historical Validation for Option Selling Strategy")

# Sidebar
st.sidebar.header("⚙️ Settings")
ticker_input = st.sidebar.text_input("Enter NSE Symbol (e.g., ^NSEI, RELIANCE.NS)", value="^NSEI")
rsi_window = st.sidebar.slider("RSI Window", 5, 30, 14)

# RSI Calculation
@st.cache_data
def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Load 5-year daily data
df = yf.download(ticker_input, start=(datetime.date.today() - datetime.timedelta(days=1825)).isoformat(), interval="1d", progress=False)
if df.empty:
    st.error("No data found for symbol.")
    st.stop()

# Clean and calculate indicators
df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
df['RSI'] = compute_rsi(df, rsi_window)
df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
df['Next_Close'] = df['Close'].shift(-1)
df.dropna(inplace=True)

# Entry Logic with Profit/Loss Simulation
entry_signals = []
signal_count = {"SELL CALL": 0, "SELL PUT": 0, "STRANGLE ZONE": 0, "TOTAL": 0, "SUCCESS": 0}
profits = []

for i in range(1, len(df) - 1):
    price = float(df['Close'].iloc[i])
    next_price = float(df['Next_Close'].iloc[i])
    rsi = float(df['RSI'].iloc[i])
    vwap = float(df['VWAP'].iloc[i])
    signal = None
    success = False
    profit = 0

    if price > vwap and rsi > 70:
        signal = "SELL CALL"
        success = next_price <= price
        profit = price - next_price
    elif price < vwap and rsi < 30:
        signal = "SELL PUT"
        success = next_price >= price
        profit = next_price - price
    elif abs(price - vwap) / vwap < 0.005:
        signal = "STRANGLE ZONE"
        success = abs(next_price - price) < 0.01 * price
        profit = 0.01 * price - abs(next_price - price)

    if signal:
        entry_signals.append((df.index[i].strftime('%Y-%m-%d'), signal, success, round(profit, 2)))
        signal_count[signal] += 1
        signal_count["TOTAL"] += 1
        if success:
            signal_count["SUCCESS"] += 1
        profits.append(profit)

# Summary Display
st.subheader("📈 Strategy Success Analysis (5-Year Backtest)")
st.write(f"Total Entry Signals: {signal_count['TOTAL']}")
st.write(f"Successful Outcomes: {signal_count['SUCCESS']}")
if signal_count['TOTAL'] > 0:
    success_rate = signal_count['SUCCESS'] / signal_count['TOTAL'] * 100
    avg_profit = np.mean(profits)
    st.metric("📊 Strategy Success Rate", f"{success_rate:.2f}%")
    st.metric("💰 Avg. Profit per Trade", f"₹{avg_profit:.2f}")
    if success_rate >= 70:
        st.success("✅ Strategy validated with over 70% success rate on past 5 years data.")
    else:
        st.warning("⚠️ Strategy success rate is below 70%. Consider adjusting RSI/VWAP thresholds.")

# Show recent signals in list format
if entry_signals:
    st.subheader("📝 All Entry Positions (Date & Type)")
    for date, signal, success, profit in entry_signals[-50:][::-1]:
        st.markdown(f"- **{date}** → `{signal}` | {'✅' if success else '❌'} | Profit: ₹{profit}")

# Profit over time plot
st.subheader("📈 Cumulative Profit Over Time")
df_signals = pd.DataFrame(entry_signals, columns=['Date', 'Signal', 'Successful', 'Profit'])
cumulative = df_signals['Profit'].cumsum()
fig_profit = go.Figure()
fig_profit.add_trace(go.Scatter(x=df_signals['Date'], y=cumulative, mode='lines', name='Cumulative Profit'))
fig_profit.update_layout(height=300, template="plotly_white")
st.plotly_chart(fig_profit, use_container_width=True)

# Latest snapshot
st.subheader("🔍 Latest Market Snapshot")
latest = df.iloc[-1]
rsi = latest['RSI']
price = latest['Close']
vwap = latest['VWAP']
col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"₹{price:.2f}")
col2.metric("RSI", f"{rsi:.2f}")
col3.metric("VWAP", f"₹{vwap:.2f}")

# Chart Display
st.subheader("📉 Price Chart with VWAP")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index[-120:], y=df['Close'].iloc[-120:], name='Close'))
fig.add_trace(go.Scatter(x=df.index[-120:], y=df['VWAP'].iloc[-120:], name='VWAP', line=dict(dash='dot')))
fig.update_layout(height=400, template="plotly_white")
st.plotly_chart(fig, use_container_width=True)

st.caption("🔁 Strategy validated using RSI & VWAP based logic on 5 years NIFTY data with simulated profit/loss per entry.")
