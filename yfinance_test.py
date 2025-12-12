import matplotlib.pyplot as plt
import yfinance as yf

def display_chart(ticker: str, period: str = '1mo', interval: str = '1d'):
    df = yf.download(ticker, period=period, interval=interval)

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['Close'])
    plt.title(f"{ticker} Closing Prices")
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.tight_layout()
    plt.show()

def get_samples(ticker: str, period='5y', lookback=120, forecast_days=20, cache_dir="stock_data"):
    pass