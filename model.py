from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yfinance as yf
from torch.utils.data import ConcatDataset
from datetime import timedelta

from StockChartDataset import StockChartDataset
from StockLSTM import StockLSTM
from visualization import (
    plot_feature_chart,
    plot_training_history,
    print_training_results,
    plot_prediction_results,
    print_prediction_results,
    plot_future_prediction,
    print_future_prediction
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_stock_dataset(ticker: str, period: str = '7300d', interval='1d'):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)

    if df.empty:
        print(f"No data for {ticker}")
        return None

    features = df[['Open', 'High', 'Low', 'Close', 'Volume']].values

    if len(features) == 0:
        print(f"No valid data for {ticker}")
        return None

    daily_charts = features.astype(np.float32)
    original_close = daily_charts[:, 3].copy()
    normalized = normalize_features(daily_charts)

    return {
        'normalized': normalized,
        'close_prices': original_close,
        'dates': df.index.tolist()
    }

def normalize_features(daily_data):
    normalized = daily_data.copy()

    for i in range(len(daily_data)):
        open_price = daily_data[i, 0]
        normalized[i, :4] = (daily_data[i, :4] / open_price) - 1

        if daily_data[i, 4] > 0:
            normalized[i, 4] = np.log(daily_data[i, 4] + 1)
        else:
            normalized[i, 4] = 0

    volume_col = normalized[:, 4]
    vol_mean = volume_col.mean()
    vol_std = volume_col.std()
    if vol_std > 0:
        normalized[:, 4] = (volume_col - vol_mean) / vol_std

    return normalized

def train(dataset: ConcatDataset, forecast_days=5):
    from torch.utils.data import DataLoader, random_split

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = StockLSTM(input_size=5, hidden_size=64, num_layers=2, forecast_days=forecast_days).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                val_loss += criterion(predictions, y_batch).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    model.load_state_dict(best_model_state)

    model.eval()
    test_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            test_loss += criterion(predictions, y_batch).item()

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    print(f"\nTest Loss: {avg_test_loss:.6f}")

    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': avg_test_loss,
        'test_predictions': np.concatenate(all_predictions, axis=0),
        'test_targets': np.concatenate(all_targets, axis=0),
        'best_val_loss': best_val_loss
    }


def predict(model, data_dict, lookback=10, forecast_days=5, ticker_name="Stock"):
    """Make predictions on historical data and return results."""
    model.eval()

    normalized = data_dict['normalized']
    close_prices = data_dict['close_prices']
    dates = data_dict.get('dates', None)

    if len(normalized) < lookback + forecast_days:
        raise ValueError(f"Not enough data. Need at least {lookback + forecast_days} days")

    X = normalized[-lookback - forecast_days:-forecast_days]
    last_known_close = close_prices[-forecast_days - 1]
    actual_future_closes = close_prices[-forecast_days:]

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(X_tensor)
        prediction = prediction.squeeze(0).cpu().numpy()

    predicted_closes = last_known_close * (1 + prediction)
    actual_normalized = (actual_future_closes - last_known_close) / last_known_close

    if dates is not None:
        last_date = dates[-forecast_days - 1]
        forecast_dates = []
        current_date = last_date
        for _ in range(forecast_days):
            current_date += timedelta(days=1)
            while current_date.weekday() >= 5:
                current_date += timedelta(days=1)
            forecast_dates.append(current_date)
    else:
        forecast_dates = list(range(1, forecast_days + 1))

    # Use visualization functions
    plot_prediction_results(predicted_closes, actual_future_closes, forecast_dates,
                           last_known_close, ticker_name, dates is not None)
    mae, mape, rmse = print_prediction_results(predicted_closes, actual_future_closes,
                                               forecast_dates, last_known_close,
                                               ticker_name, dates is not None)

    return {
        'predicted_closes': predicted_closes,
        'actual_closes': actual_future_closes,
        'predicted_returns': prediction,
        'actual_returns': actual_normalized,
        'forecast_dates': forecast_dates,
        'last_known_close': last_known_close,
        'mae': mae,
        'mape': mape,
        'rmse': rmse
    }


def predict_future(model, data_dict, lookback=10, forecast_days=5, ticker_name="Stock"):
    """Predict future prices (when we don't have actual values yet)."""
    model.eval()

    normalized = data_dict['normalized']
    close_prices = data_dict['close_prices']
    dates = data_dict.get('dates', None)

    if len(normalized) < lookback:
        raise ValueError(f"Not enough data. Need at least {lookback} days")

    X = normalized[-lookback:]
    last_known_close = close_prices[-1]

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(X_tensor)
        prediction = prediction.squeeze(0).cpu().numpy()

    predicted_closes = last_known_close * (1 + prediction)

    if dates is not None:
        last_date = dates[-1]
        forecast_dates = []
        current_date = last_date
        for _ in range(forecast_days):
            current_date += timedelta(days=1)
            while current_date.weekday() >= 5:
                current_date += timedelta(days=1)
            forecast_dates.append(current_date)
    else:
        forecast_dates = list(range(1, forecast_days + 1))

    # Use visualization functions
    plot_future_prediction(predicted_closes, forecast_dates, close_prices, dates, ticker_name)
    print_future_prediction(predicted_closes, forecast_dates, last_known_close,
                          ticker_name, dates is not None)

    return {
        'predicted_closes': predicted_closes,
        'predicted_returns': prediction,
        'forecast_dates': forecast_dates,
        'last_known_close': last_known_close
    }


if __name__ == "__main__":
    print('hello, world!')

    tickers = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'VOO',
        'NVDA', 'META', 'NFLX', 'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'CSCO', 'AVGO',
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'V', 'MA',
        'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'LLY', 'BMY', 'AMGN',
        'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'COST', 'DIS', 'PG', 'KO', 'PEP',
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC',
        'BA', 'CAT', 'GE', 'HON', 'UPS', 'UNP', 'MMM', 'LMT', 'RTX',
        'T', 'VZ', 'CMCSA', 'TMUS',
        'AMT', 'PLD', 'SPG',
        'SPY', 'QQQ', 'IWM', 'DIA',
        'MU'
    ]

    all_stock_data = {}

    for ticker in tickers:
        print(f"Processing {ticker}...")
        all_stock_data[ticker] = build_stock_dataset(ticker)

    datasets = []
    for ticker, data in all_stock_data.items():
        if data is not None:
            dataset = StockChartDataset(data, lookback=10, forecast_days=5)
            datasets.append(dataset)

    combined_dataset = ConcatDataset(datasets)

    Path('models').mkdir(exist_ok=True)
    checkpoint_path = 'models/stock_lstm_checkpoint.pt'

    if Path(checkpoint_path).exists():
        print(f"\nLoading existing model from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model = StockLSTM(**checkpoint['model_config']).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        results = {
            'model': model,
            'train_losses': checkpoint['train_losses'],
            'val_losses': checkpoint['val_losses'],
            'test_loss': checkpoint['test_loss'],
            'best_val_loss': checkpoint['best_val_loss'],
            'test_predictions': None,
            'test_targets': None
        }

        print(f"Model loaded successfully!")
        print(f"Previous Best Validation Loss: {results['best_val_loss']:.6f}")
        print(f"Previous Test Loss: {results['test_loss']:.6f}")

    else:
        print("\nNo existing model found. Training new model...")
        print(f"\nTotal samples: {len(combined_dataset)}")

        results = train(combined_dataset)

        model_path = 'models/stock_lstm.pt'
        torch.save(results['model'].state_dict(), model_path)
        print(f"\nModel saved to {model_path}")

        torch.save({
            'model_state_dict': results['model'].state_dict(),
            'train_losses': results['train_losses'],
            'val_losses': results['val_losses'],
            'test_loss': results['test_loss'],
            'best_val_loss': results['best_val_loss'],
            'model_config': {
                'input_size': 5,
                'hidden_size': 64,
                'num_layers': 2,
                'forecast_days': 5
            }
        }, checkpoint_path)
        print(f"Full checkpoint saved to {checkpoint_path}")

        print_training_results(results)
        plot_training_history(results['train_losses'], results['val_losses'])

    # plot_feature_chart(all_stock_data['AAPL'], ticker_name='AAPL')

    for ticker in ['NVDA', 'TSLA', 'AVGO', 'MU']:
        # print("\n=== Testing Predictions on Historical Data ===")
        # prediction_results = predict(
        #     model=results['model'],
        #     data_dict=all_stock_data[ticker],
        #     lookback=10,
        #     forecast_days=5,
        #     ticker_name=ticker
        # )

        print("\n=== Predicting Future Prices ===")
        future_results = predict_future(
            model=results['model'],
            data_dict=all_stock_data[ticker],
            lookback=10,
            forecast_days=5,
            ticker_name=ticker
        )