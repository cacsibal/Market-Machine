import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import yfinance as yf
import os

from StockReturnsDataset import StockReturnsDataset
from yfinance_test import get_daily_returns
from visualization import plot_mse_loss, print_predictions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

if __name__ == "__main__":
    def predict_stock(ticker, lookback=10, forecast_days=5):
        model_path = f'stock_mlp_{ticker}.pth'
        model = StockMLP(input_size=lookback, hidden_sizes=[64, 32, 16], dropout=0.2)

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
        else:
            train_loader, test_loader = prepare_data(ticker, lookback, forecast_days)
            train_losses, test_losses = train_model(model, train_loader, test_loader, epochs=100, lr=0.001)
            torch.save(model.state_dict(), model_path)

        df = yf.download(ticker, period='1mo', interval='1d', progress=False)['Close']
        close_prices = df.to_numpy().ravel()

        returns = np.diff(close_prices) / close_prices[:-1]

        recent_returns = returns[-lookback:]
        current_price = close_prices[-1]

        predictions = {
            'dates' : [],
            'predicted_prices' : [],
            'predicted_returns' : [],
            'current_price' : current_price,
        }

        input_sequence = recent_returns.copy()
        predicted_price = current_price

        for i in range(forecast_days):
            X = torch.tensor(input_sequence[-lookback:], dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                predicted_cumulative_return = model(X).item()

                single_day_return = (1 + predicted_cumulative_return) ** (1 / forecast_days) - 1

                predicted_price *= (1 + single_day_return)

                predictions['predicted_prices'].append(predicted_price)
                predictions['predicted_returns'].append(single_day_return)

                input_sequence = np.append(input_sequence, single_day_return)

        return predictions

    nvda_predictions = predict_stock('NVDA')
    print_predictions('NVDA', nvda_predictions, 5)

    voo_predictions = predict_stock('VOO')
    print_predictions('VOO', voo_predictions, 5)