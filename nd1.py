import requests
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import time

def fetch_nse_option_chain(symbol='NIFTY'):
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json",
        "Referer": "https://www.nseindia.com/option-chain",
    }

    session = requests.Session()

    try:
        # Step 1: Visit homepage to get cookies
        session.get("https://www.nseindia.com", headers=headers, timeout=5)
        time.sleep(1)  # Small delay to mimic browser
        # Step 2: Fetch option chain data
        response = session.get(url, headers=headers, timeout=10)

        if 'application/json' not in response.headers.get('Content-Type', ''):
            print("⚠️ Non-JSON response received. Check cookies or headers.")
            print(response.text[:300])  # print a snippet of HTML
            return None

        data = response.json()
        return data['records']['data']

    except Exception as e:
        print("❌ Failed to fetch NSE data:", e)
        return None

def process_option_data(option_data):
    calls = []
    puts = []
    spot_price = 0

    for entry in option_data:
        strike = entry.get('strikePrice')
        ce = entry.get('CE', {})
        pe = entry.get('PE', {})

        if 'underlyingValue' in ce:
            spot_price = ce['underlyingValue']

        if ce and pe:
            calls.append({
                'Strike': strike,
                'Volume': ce.get('totalTradedVolume', 0),
                'OI Change': ce.get('changeinOpenInterest', 0)
            })
            puts.append({
                'Strike': strike,
                'Volume': pe.get('totalTradedVolume', 0),
                'OI Change': pe.get('changeinOpenInterest', 0)
            })

    df_calls = pd.DataFrame(calls)
    df_puts = pd.DataFrame(puts)
    return df_calls, df_puts, spot_price

def plot_option_pressure(df_calls, df_puts, symbol, spot_price):
    plt.figure(figsize=(14, 7))

    width = 0.4
    x = range(len(df_calls))

    plt.bar(x, df_calls['Volume'], width=width, label='Call Volume', color='skyblue')
    plt.bar([i + width for i in x], df_puts['Volume'], width=width, label='Put Volume', color='salmon')
    plt.xticks([i + width / 2 for i in x], df_calls['Strike'], rotation=45)
    plt.axvline(x=next((i for i, s in enumerate(df_calls['Strike']) if s >= spot_price), len(df_calls)//2),
                color='green', linestyle='--', label=f'Spot Price: {spot_price}')
    plt.title(f"{symbol} Options Volume Pressure\n{datetime.datetime.now().strftime('%d-%b %H:%M')}")
    plt.xlabel("Strike Price")
    plt.ylabel("Volume Traded")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main(symbol='NIFTY'):
    print(f"Fetching Option Chain Data for {symbol}...")
    option_data = fetch_nse_option_chain(symbol)

    if not option_data:
        print("⚠️ Could not fetch option data.")
        return

    df_calls, df_puts, spot_price = process_option_data(option_data)
    print(f"✅ Spot Price: {spot_price}")
    plot_option_pressure(df_calls, df_puts, symbol, spot_price)

# Run the program
if __name__ == "__main__":
    main("NIFTY")  # Change to "BANKNIFTY" or others as needed
