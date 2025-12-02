import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class StockChartDataset(Dataset):
    def __init__(self, data_dict, lookback=10, forecast_days=5):
        self.normalized_data = data_dict['normalized']
        self.close_prices = data_dict['close_prices']
        self.lookback = lookback
        self.forecast_days = forecast_days

        self.X, self.y = self._create_sequences()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

    def _create_sequences(self):
        X, y = [], []

        for i in range(0, len(self.normalized_data) - self.lookback - self.forecast_days + 1,
                       self.lookback + self.forecast_days):
            features = self.normalized_data[i:i + self.lookback]

            future_closes = self.close_prices[i + self.lookback:i + self.lookback + self.forecast_days]

            last_known_close = self.close_prices[i + self.lookback - 1]
            normalized_future = (future_closes - last_known_close) / last_known_close

            X.append(features)
            y.append(normalized_future)

        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)