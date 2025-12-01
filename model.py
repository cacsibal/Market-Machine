from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yfinance as yf
from torch.utils.data import ConcatDataset

from StockChartDataset import StockChartDataset
from StockLSTM import StockLSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_stock_dataset(ticker: str, period: str='7300d', interval='1d'):
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

    normalized = normalize_features(daily_charts)

    return normalized

def normalize_features(daily_data):
    normalized = daily_data.copy()

    for i in range(len(daily_data)):
        open_price = daily_data[i, 0]
        normalized[i, :4] = (daily_data[i, :4] / open_price) - 1

        if i > 0:
            recent_volumes = daily_data[max(0, i - 20):i + 1, 4]  # Last 20 days
            vol_mean = recent_volumes.mean()
            vol_std = recent_volumes.std()
            if vol_std > 0:
                normalized[i, 4] = (daily_data[i, 4] - vol_mean) / vol_std
            else:
                normalized[i, 4] = 0
        else:
            normalized[i, 4] = 0

    return normalized


def train(dataset: ConcatDataset):
    from torch.utils.data import DataLoader, random_split

    # Split data
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.manual_seed(42)
    )

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Model
    model = StockLSTM(input_size=5, hidden_size=64, num_layers=2).to(device)

    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Track history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
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

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()

        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    # Load best model
    model.load_state_dict(best_model_state)

    # Test evaluation
    model.eval()
    test_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            predictions = model(X_batch)
            test_loss += criterion(predictions, y_batch).item()

            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    print(f"\nTest Loss: {avg_test_loss:.6f}")

    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': avg_test_loss,
        'test_predictions': np.array(all_predictions),
        'test_targets': np.array(all_targets),
        'best_val_loss': best_val_loss
    }

if __name__ == "__main__":
    print('hello, world!')

    tickers = [
        # Original
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'VOO',

        # Tech (additional)
        'NVDA', 'META', 'NFLX', 'ADBE', 'CRM', 'ORCL', 'INTC', 'AMD', 'CSCO', 'AVGO',

        # Finance
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW', 'AXP', 'V', 'MA',

        # Healthcare
        'UNH', 'JNJ', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'LLY', 'BMY', 'AMGN',

        # Consumer
        'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'COST', 'DIS', 'PG', 'KO', 'PEP',

        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC',

        # Industrial
        'BA', 'CAT', 'GE', 'HON', 'UPS', 'UNP', 'MMM', 'LMT', 'RTX',

        # Telecom/Media
        'T', 'VZ', 'CMCSA', 'TMUS',

        # Real Estate/REITs
        'AMT', 'PLD', 'SPG',

        # ETFs (for diversity)
        'SPY', 'QQQ', 'IWM', 'DIA'
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

    print(f"\nTotal samples: {len(combined_dataset)}")

    # Train the model
    results = train(combined_dataset)

    # Create models directory if it doesn't exist
    Path('models').mkdir(exist_ok=True)

    # Save model weights
    model_path = 'models/stock_lstm.pt'
    torch.save(results['model'].state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    # Save complete checkpoint with metadata
    checkpoint_path = 'models/stock_lstm_checkpoint.pt'
    torch.save({
        'model_state_dict': results['model'].state_dict(),
        'train_losses': results['train_losses'],
        'val_losses': results['val_losses'],
        'test_loss': results['test_loss'],
        'best_val_loss': results['best_val_loss'],
        'model_config': {
            'input_size': 5,
            'hidden_size': 64,
            'num_layers': 2
        }
    }, checkpoint_path)
    print(f"Full checkpoint saved to {checkpoint_path}")

    # Print final metrics
    print(f"\nFinal Results:")
    print(f"Best Validation Loss: {results['best_val_loss']:.6f}")
    print(f"Test Loss: {results['test_loss']:.6f}")

    # Calculate additional metrics
    from sklearn.metrics import mean_absolute_error, r2_score
    mae = mean_absolute_error(results['test_targets'], results['test_predictions'])
    r2 = r2_score(results['test_targets'], results['test_predictions'])
    print(f"Test MAE: {mae:.6f}")
    print(f"Test R^2: {r2:.6f}")

    # todo:
    # currently, the model trains only on the stock to be predicted.
    # i want to have a single model that can predict any stock and take segments of market data as training
    # i also want to experiment with other models than mlps