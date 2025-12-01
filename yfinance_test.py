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
    df = yf.download(ticker, period=period, interval='1d', progress=progress)['Close']
    close_prices = df.to_numpy().ravel()
    returns = np.diff(close_prices) / close_prices[:-1] # returns = (P_t - P_{t-1}) / P_{t-1}

    return returns

print(get_daily_returns('AAPL'))