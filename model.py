import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import time
from datetime import datetime

from StockDataset import StockDataset
from StockLSTM import StockLSTM
from yfinance_test import get_samples

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODELS_DIR = 'saved_models'

def save_model(model, filepath):
    pass

def load_model(model, filepath):
    pass

def train_model(model, train_loader, test_loader, epochs=10, epsilon=0.005, lambda_reg=0.0001):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=epsilon)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        epoch_start_time = time.time()

        model.train()
        train_loss = 0.0
        for x, y, base_price in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(x)

            loss = criterion(preds, y)

            l2_reg = sum(torch.sum(p ** 2) for p in model.parameters())
            loss = loss + lambda_reg * l2_reg

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for x, y, base_price in test_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)

                loss = criterion(preds, y)
                test_loss += loss.item() * x.size(0)

            test_loss /= len(test_loader.dataset)

        epoch_time = time.time() - epoch_start_time

        train_rmse = np.sqrt(train_loss)
        test_rmse = np.sqrt(test_loss)

        print(f"Epoch {epoch + 1}/{epochs} | Train RMSE: {train_rmse:.4f} | Test RMSE: {test_rmse:.4f} | Time: {epoch_time:.2f}s")

    return model

def tune_model(model, tickers: list[str], training_proportion=0.8, period='1y', lookback=10, forecast_days=5,
               epochs=2, epsilon=0.01, lambda_reg=0.0001, save_path=None, model_name=None):
    # tuned_model = train_model(
    #     model,
    #     train_loader,
    #     test_loader,
    #     epochs=epochs,
    #     epsilon=epsilon,
    #     lambda_reg=lambda_reg
    # )
    #
    # return tuned_model
    pass


def predict(model, forecast_days=5):
    pass

def get_loaders(data_file_name, training_proportion=0.8, period='1y', lookback=120, forecast_days=30):
    pass

if __name__ == "__main__":
    print('number of cores:', os.cpu_count())

    # hyperparameters
    lookback = 120
    forecast_days = 15
    hidden_size = 128
    batch_size = 64
    epochs = 5
    epsilon = 0.01
    lambda_reg = 1e-7
    num_layers = 3
    data_collection_period = '5y'
    training_proportion = 0.8

    base_etf_set = [
        'SPY', 'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLRE', 'XLU', 'XLC', 'SOXX', 'QTUM',
        'EFA', 'EEM', 'IWM', 'GLD', 'SLV'
    ]

    tech = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL', 'PLTR']

    BASE_MODEL_PATH = os.path.join(MODELS_DIR, 'base_model.pt')

    get_samples("base_etf_set_data.csv", base_etf_set)