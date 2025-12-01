import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import yfinance as yf

from StockReturnsDataset import StockReturnsDataset
from yfinance_test import get_daily_returns

def prepare_data(ticker: str, lookback=10, forecast_days=5, batch_size=32):
    returns = get_daily_returns(ticker, period='2y')

    split_idx = int(len(returns) * 0.8)
    train_returns = returns[:split_idx]
    test_returns = returns[split_idx:]

    train_dataset = StockReturnsDataset(train_returns, lookback, forecast_days)
    test_dataset = StockReturnsDataset(test_returns, lookback, forecast_days)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Feature shape: {train_dataset.X.shape}")

    return train_loader, test_loader

class StockMLP(nn.Module):
    def __init__(self, input_size=10, hidden_sizes=[64, 32, 16], dropout=0.2):
        super(StockMLP, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)

def train_model(model, train_loader, test_loader, epochs=10, lr=0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                test_loss += loss.item()

        test_loss /= len(test_loader)
        test_losses.append(test_loss)

        if(epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    return train_losses, test_losses

def predict(model, ticker, lookback=10, forecast_days=5, from_date=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    df = yf.download(ticker, period='5y', interval='1d')

    if from_date is not None:
        df = df[df.index <= from_date]
        if len(df) < lookback + 1:
            raise ValueError(f"Not enough data for prediction. Need at least {lookback + 1} days.")

    close_prices = df['Close'].to_numpy().ravel()
    prediction_date = df.index[-1].strftime('%Y-%m-%d')

    current_price = close_prices[-1]

    recent_returns = np.diff(close_prices[-lookback - 1:]) / close_prices[-lookback - 1:-1]

    with torch.no_grad():
        input_tensor = torch.tensor(recent_returns, dtype=torch.float32).unsqueeze(0).to(device)
        predicted_return = model(input_tensor).item()

    predicted_price = current_price * (1 + predicted_return)

    from datetime import datetime, timedelta
    current_date = datetime.strptime(prediction_date, '%Y-%m-%d')
    target_date = current_date + timedelta(days=forecast_days)
    target_date_str = target_date.strftime('%Y-%m-%d')

    print(f"\n{'=' * 50}")
    print(f"Ticker: {ticker}")
    print(f"Price on {prediction_date}: ${current_price:.2f}")
    print(f"Predicted Return ({forecast_days} days): {predicted_return:.4f} ({predicted_return * 100:.2f}%)")
    print(f"Predicted Price on ~{target_date_str}: ${predicted_price:.2f}")
    print(f"Expected Change: ${predicted_price - current_price:.2f}")
    print(f"{'=' * 50}\n")

    return current_price, predicted_price, predicted_return

if __name__ == "__main__":
    train_loader, test_loader = prepare_data(ticker='AAPL', lookback=10, forecast_days=5)

    model = StockMLP(input_size=10, hidden_sizes=[64, 32, 16], dropout=0.2)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    train_losses, test_losses = train_model(model, train_loader, test_loader, epochs=100, lr=0.001)

    torch.save(model.state_dict(), 'stock_mlp.pth')

    def plot_mse_loss():
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.show()

    predict(model, 'AAPL', from_date='2024-11-01')