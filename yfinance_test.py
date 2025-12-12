import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os

def display_chart(ticker: str, period: str = '1mo', interval: str = '1d'):
    df = yf.download(ticker, period=period, interval=interval)

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['Close'])
    plt.title(f"{ticker} Closing Prices")
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.tight_layout()
    plt.show()

def get_daily_returns(ticker: str, period: str = '1mo', progress=False):
    df = yf.download(ticker, period=period, interval='1d', progress=progress, auto_adjust=True)['Close']
    close_prices = df.to_numpy().ravel()
    # Handle potential zeros to avoid inf
    returns = np.diff(close_prices) / (close_prices[:-1] + 1e-8)
    return returns

def get_samples(ticker: str, period='2y', lookback=10, forecast_days=5, cache_dir="stock_data"):
    """
    Fetches stock data and prepares samples of percent changes (returns).

    Returns:
        samples: np.array of shape (N, lookback+forecast, features)
        base_prices: np.array of shape (N,) - The Close price at the end of the lookback window (t=0 for forecast).
        sample_dates: np.array of shape (N, lookback+forecast)
        sample_tickers: np.array of shape (N,)
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}_{period}_{lookback}_{forecast_days}.csv")

    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file)

            if 'base_prices' in df.columns:
                print(f'loading saved data from {cache_file}')
                samples = np.array(
                    [np.fromstring(s, sep=' ').reshape(lookback + forecast_days, -1) for s in df['samples']])
                base_prices = np.array([float(b) for b in df['base_prices']])
                sample_dates = np.array([d.split('|') for d in df['sample_dates']])
                sample_tickers = np.array(df['sample_tickers'])
                return samples, base_prices, sample_dates, sample_tickers
        except Exception as e:
            print(f"Error loading cache, re-downloading: {e}")

    print(f'downloading data for {ticker}')

    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)

    data = data.ffill().bfill()

    raw_prices = data.values  # Shape: (T, Features)
    dates = data.index.strftime('%Y-%m-%d').tolist()

    # Calculate Returns for all features: (P_t - P_{t-1}) / P_{t-1}
    returns = np.diff(raw_prices, axis=0) / (raw_prices[:-1] + 1e-8)

    aligned_dates = dates[1:]
    aligned_prices = raw_prices[1:]

    sample_len = lookback + forecast_days

    samples, base_prices, sample_dates, sample_tickers = [], [], [], []

    for i in range(len(returns) - sample_len + 1):
        sample_window = returns[i: i + sample_len]
        date_window = aligned_dates[i: i + sample_len]

        anchor_price = aligned_prices[i + lookback - 1, 3]  # Index 3 is Close (Open,High,Low,Close,Vol)

        samples.append(sample_window)
        base_prices.append(anchor_price)
        sample_dates.append(date_window)
        sample_tickers.append(ticker)

    df = pd.DataFrame({
        "samples": [' '.join(map(str, s.flatten())) for s in samples],
        "base_prices": base_prices,
        "sample_dates": ['|'.join(d) for d in sample_dates],
        "sample_tickers": sample_tickers
    })
    df.to_csv(cache_file, index=False)

    return np.array(samples), np.array(base_prices), np.array(sample_dates), np.array(sample_tickers)