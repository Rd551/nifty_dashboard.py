import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

def fetch_nse_option_chain(symbol='NIFTY'):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    session = requests.Session()
    session.get("https://www.nseindia.com", headers=headers)
    response = session.get(url, headers=headers)
    data = response.json()
    return data['records']['data']

def process_option_data(option_data, spot_price):
    calls = []
    puts = []
    for entry in option_data:
        strike_price = entry['strikePrice']
        ce = entry.get('CE', {})
        pe = entry.get('PE', {})
        if ce and pe:
            calls.append({
                'Strike': strike_price,
                'Volume': ce.get('totalTradedVolume', 0),
                'OI Change': ce.get('changeinOpenInterest', 0)
            })
            puts.append({
                'Strike': strike_price,
                'Volume': pe.get('totalTradedVolume', 0),
                'OI Change': pe.get('changeinOpenInterest', 0)
            })
    df_calls = pd.DataFrame(calls)
    df_puts = pd.DataFrame(puts)
    return df_calls, df_puts

def plot_option_pressure(df_calls, df_puts, symbol, spot_price):
    plt.figure(figsize=(14, 7))

    width = 0.35
    strikes = df_calls['Strike']
    x = range(len(strikes))

    plt.bar(x, df_calls['Volume'], width=width, label='Call Volume', color='skyblue')
    plt.bar([i + width for i in x], df_puts['Volume'], width=width, label='Put Volume', color='salmon')
    plt.xticks([i + width / 2 for i in x], strikes, rotation=45)
    plt.axvline(x=next((i for i, s in enumerate(strikes) if s > spot_price), len(strikes)//2), color='green', linestyle='--', label='Spot Price')
    plt.title(f"{symbol} Option Volume Pressure at {datetime.datetime.now().strftime('%d-%b %H:%M')}")
    plt.xlabel("Strike Price")
    plt.ylabel("Volume")
    plt.legend()
    plt.tight_layout()
    plt.show()

def main(symbol='NIFTY'):
    print("Fetching option data...")
    data = fetch_nse_option_chain(symbol)
    spot_price = float(data[0]['CE']['underlyingValue'])
    df_calls, df_puts = process_option_data(data, spot_price)
    plot_option_pressure(df_calls, df_puts, symbol, spot_price)

# Run for NIFTY or BANKNIFTY
main('NIFTY')
