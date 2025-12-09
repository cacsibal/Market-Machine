import torch
from torch.utils.data import Dataset

class StockDataset(Dataset):
    def __init__(self, dataset, lookback=10, forecast_days=5):
        """
        dataset: 3-tuple [X, M, S]
            X: np.array of shape (num_samples, lookback+forecast_days, 5)
            M: np.array of shape (num_samples,)
            S: np.array of shape (num_samples,)
        """
        self.X, self.M, self.S = dataset
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

        mean = torch.tensor(self.M[idx], dtype=torch.float32)
        std = torch.tensor(self.S[idx], dtype=torch.float32)

        return x, y, mean, std