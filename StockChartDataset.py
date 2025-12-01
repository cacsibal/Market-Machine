import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class StockChartDataset(Dataset):
    def __init__(self, returns, lookback=10, forecast_days=5):
        self.returns = returns
        self.lookback = lookback
        self.forecast_days = forecast_days

        self.X, self.y = self._create_sequences()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

    def _create_sequences(self):
        X, y = [], []

        for i in range(0, len(self.returns) - self.lookback - self.forecast_days + 1, self.lookback + self.forecast_days):
            features = self.returns[i:i + self.lookback]

            future_returns = self.returns[i + self.lookback:i + self.lookback + self.forecast_days]
            cumulative_return = np.prod(1 + future_returns) - 1

            X.append(features)
            y.append(cumulative_return)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)