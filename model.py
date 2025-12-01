import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import yfinance as yf
import os

from StockReturnsDataset import StockReturnsDataset
from yfinance_test import get_daily_returns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_data():
    pass

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

if __name__ == "__main__":
    print('hello, world!')
    # todo:
    # currently, the model trains only on the stock to be predicted.
    # i want to have a single model that can predict any stock and take segments of market data as training
    # i also want to experiment with other models than mlps