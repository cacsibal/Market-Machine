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

def train_model(model, tickers, training_proportion=0.8, lookback=120, forecast_days=20, epochs=10, epsilon=0.005, lambda_reg=0.0001):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=epsilon)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        epoch_start_time = time.time()

        model.train()
        train_loss = 0.

def tune_model(base_model, tickers: list[str], training_proportion=0.2, lookback=120, forecast_days=20, epochs=2, epsilon=0.01, lambda_reg=0.0001):
    return train_model(base_model, tickers, training_proportion, lookback, forecast_days, epochs, epsilon, lambda_reg)

def predict(model, forecast_days=5):
    pass

def get_loaders(file_name, training_proportion=0.8, lookback=120, forecast_days=30):
    pass

if __name__ == "__main__":
    print('number of cores:', os.cpu_count())

    # hyperparameters
    input_size = 5
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

    model = StockLSTM()