import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    def __init__(self, dataset, lookback=10, forecast_days=5):
        """
        dataset: Tuple containing the data components.
            Expected structure: (X, BasePrices, Dates, Tickers)

            X: np.array of shape (num_samples, lookback+forecast_days, num_features)
               Contains daily percent changes (returns).
            BasePrices: np.array of shape (num_samples,)
               Contains the closing price at the last step of the lookback period
               (the anchor price for reconstructing future prices).
            Dates: np.array of shape (num_samples,)
               Date strings or timestamps.
            Tickers: np.array of shape (num_samples,)
               Ticker symbols.
        """
        self.X, self.BasePrices, self.D, self.T = dataset
        self.lookback = lookback
        self.forecast_days = forecast_days

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        sample = self.X[idx]

        x = sample[:self.lookback]

        y = sample[self.lookback:]

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y[:, 3], dtype=torch.float32)

        base_price = torch.tensor(self.BasePrices[idx], dtype=torch.float32)

        return x, y, base_price