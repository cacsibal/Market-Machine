import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import yfinance as yf

def display_chart(ticker: str, period: str = '1mo', interval: str = '1d'):
    df = yf.download(ticker, period=period, interval=interval)

    plt.figure(figsize=(10,5))
    plt.plot(df.index, df['Close'])
    plt.title(f"{ticker} Closing Prices")
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.tight_layout()
    plt.show()

def get_daily_returns(ticker: str, period: str='1mo', progress=False):
    df = yf.download(ticker, period=period, interval='1d', progress=progress, auto_adjust=True)['Close']
    close_prices = df.to_numpy().ravel()
    returns = np.diff(close_prices) / close_prices[:-1] # returns = (P_t - P_{t-1}) / P_{t-1}

    return returns

def get_samples(ticker: str, period='2y', lookback=10, forecast_days=5):
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    prices = data.values
    dates = data.index.strftime('%Y-%m-%d').tolist()

    sample_len = lookback + forecast_days

    samples, means, stds, sample_dates, sample_tickers = [], [], [], [], []
    for i in range(len(prices) - sample_len + 1):
        sample_window = prices[i: i + sample_len]
        date_window = dates[i: i + sample_len]

        input_portion = sample_window[:lookback]

        mean = input_portion.mean(axis=0)
        std = input_portion.std(axis=0, ddof=0)

        std[std == 0] = 1.0

        norm_sample = (sample_window - mean) / std

        samples.append(norm_sample)
        means.append(mean)
        stds.append(std)
        sample_dates.append(date_window)
        sample_tickers.append(ticker)

    return np.array(samples), np.array(means), np.array(stds), np.array(sample_dates), np.array(sample_tickers)