import os
import pandas as pd
import yfinance as yf


def get_samples(file_name: str, tickers: list[str], period='5y', cache_dir="stock_data"):
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    all_data = []

    print(f"Downloading data for {len(tickers)} tickers...")

    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, progress=False, auto_adjust=True)

            if df.empty:
                print(f"Warning: No data found for {ticker}")
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.reset_index()

            if 'Date' not in df.columns and 'index' in df.columns:
                df.rename(columns={'index': 'Date'}, inplace=True)

            df['ticker_name'] = ticker

            df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            }, inplace=True)

            required_columns = ['ticker_name', 'date', 'open', 'high', 'low', 'close', 'volume']

            df = df[[c for c in required_columns if c in df.columns]]

            all_data.append(df)

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)

        output_path = os.path.join(cache_dir, file_name)
        combined_df.to_csv(output_path, index=False)
        print(f"Saved {len(combined_df)} rows to {output_path}")
    else:
        print("No data was collected.")