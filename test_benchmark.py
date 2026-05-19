from data.fetcher import fetch_ohlcv
df = fetch_ohlcv('BTC/USDT', '1h', days=365)
start_price = df.iloc[0]['close']
end_price = df.iloc[-1]['close']
print(f"Buy and Hold Return: {(end_price - start_price) / start_price * 100:.2f}%")
