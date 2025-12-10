import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
from scipy.optimize import minimize, differential_evolution

from StockDataset import StockDataset
from StockLSTM import StockLSTM
from visualization import visualize_test, visualize_future
from yfinance_test import get_samples

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(model, train_loader, test_loader, epochs=10, epsilon=0.01, lambda_reg=0.0001):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=epsilon)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for x, y, mean, std in train_loader:
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
            for x, y, mean, std in test_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x)

                loss = criterion(preds, y)
                test_loss += loss.item() * x.size(0)

            test_loss /= len(test_loader.dataset)

        train_dollar_loss = np.sqrt(train_loss)
        test_dollar_loss = np.sqrt(test_loss)
        print(f"Epoch {epoch+1}/{epochs} | Train Dollar Loss: ${train_dollar_loss:.2f} | Test Dollar Loss: ${test_dollar_loss:.2f} | Difference: ${train_dollar_loss - test_dollar_loss:.2f}")

        # print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} | Difference: {train_loss - test_loss:.6f}")

    return model

def optimize_hyperparameters(model, train_loader, test_loader):
    pass

def predict(model, sample, mean, std, lookback=10, forecast_days=5):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(sample[:lookback], dtype=torch.float32).unsqueeze(0).to(device)

        preds = model(x)
        preds = preds.squeeze(0).cpu().numpy()

        preds_denorm = preds * std[3] + mean[3]

    return preds_denorm


def predict_future(model, ticker: str, lookback=10, forecast_days=5, period='1y'):
    samples, means, stds, dates, sample_tickers = get_samples(
        ticker,
        period=period,
        lookback=lookback,
        forecast_days=forecast_days
    )

    latest_sample = samples[-1]
    latest_mean = means[-1]
    latest_std = stds[-1]

    model.eval()
    model = model.to(device)

    with torch.no_grad():
        x = torch.tensor(latest_sample[:lookback], dtype=torch.float32).unsqueeze(0).to(device)

        preds = model(x)
        preds = preds.squeeze(0).cpu().numpy()

        preds_denorm = preds * latest_std[3] + latest_mean[3]

    return preds_denorm

def get_loaders(tickers: list[str], training_proportion, period='1y', lookback=10, forecast_days=5):
    print(f"Number of stocks: {len(tickers)}")

    all_samples, all_means, all_stds, all_dates, all_tickers = {}, {}, {}, {}, {}

    for ticker in tickers:
        samples, means, stds, dates, sample_tickers = get_samples(
            ticker,
            period=period,
            lookback=lookback,
            forecast_days=forecast_days
        )

        all_samples[ticker] = samples
        all_means[ticker] = means
        all_stds[ticker] = stds
        all_dates[ticker] = dates
        all_tickers[ticker] = sample_tickers

    X = np.concatenate(list(all_samples.values()), axis=0)
    M = np.concatenate(list(all_means.values()), axis=0)
    S = np.concatenate(list(all_stds.values()), axis=0)
    D = np.concatenate(list(all_dates.values()), axis=0)
    T = np.concatenate(list(all_tickers.values()), axis=0)

    perm = np.random.permutation(len(X))

    X_shuffled = X[perm]
    M_shuffled = M[perm]
    S_shuffled = S[perm]
    D_shuffled = D[perm]
    T_shuffled = T[perm]

    num_training_samples = int(training_proportion * len(X))

    training_samples = [X_shuffled[:num_training_samples], M_shuffled[:num_training_samples],
                        S_shuffled[:num_training_samples], D_shuffled[:num_training_samples],
                        T_shuffled[:num_training_samples]]
    testing_samples = [X_shuffled[num_training_samples:], M_shuffled[num_training_samples:],
                       S_shuffled[num_training_samples:], D_shuffled[num_training_samples:],
                       T_shuffled[num_training_samples:]]

    d_train = StockDataset(training_samples, lookback=lookback, forecast_days=forecast_days)
    d_test = StockDataset(testing_samples, lookback=lookback, forecast_days=forecast_days)

    print(f"Total training examples: {len(d_train)}")
    print(f"Total test examples: {len(d_test)}")

    train_loader = DataLoader(d_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(d_test, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, testing_samples

if __name__ == "__main__":
    print('hello, world!')
    print('number of cores: ', os.cpu_count())

    # hyperparameters
    lookback = 60
    forecast_days = 5
    hidden_size = 128
    batch_size = 32
    epochs = 10
    epsilon = 0.01
    lambda_reg = 1e-7

    data_collection_period = '1y'
    training_proportion = 0.8

    tickers = [
        # sector etfs
        'SPY', 'XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLRE', 'XLU', 'XLC', 'SOXX', 'QTUM'
    ]

    tech = [
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL',
    ]

    quantum = [
        'IONQ', 'RGTI', 'QBTS', 'ARQQ', 'QUBT'
    ]

    model = StockLSTM(input_size=5, hidden_size=hidden_size, forecast_days=forecast_days)

    train_loader, test_loader, testing_samples = get_loaders(tickers, training_proportion, period=data_collection_period, lookback=lookback, forecast_days=forecast_days)
    trained_model = train_model(model, train_loader, test_loader, epochs=epochs, epsilon=epsilon, lambda_reg=lambda_reg)

    future_preds = predict_future(trained_model, 'AAPL', lookback=lookback, forecast_days=forecast_days)
    visualize_future(future_preds, ticker='AAPL', lookback=lookback)

    # for i in range(5):
    #     sample = testing_samples[0][i]
    #     mean = testing_samples[1][i]
    #     std = testing_samples[2][i]
    #     dates = testing_samples[3][i]
    #     viz_ticker = testing_samples[4][i]
    #
    #     preds = predict(trained_model, sample, mean, std, lookback=lookback, forecast_days=forecast_days)
    #
    #     actuals_norm = sample[lookback:, 3]
    #     actuals = actuals_norm * std[3] + mean[3]
    #
    #     historical_norm = sample[max(0, lookback - 20):lookback, 3]
    #     historical_prices = historical_norm * std[3] + mean[3]
    #
    #     historical_dates = dates[max(0, lookback - 20):lookback].tolist()
    #     prediction_dates = dates[lookback:].tolist()
    #     plot_dates = historical_dates + prediction_dates
    #
    #     print(f"--- Visualization Sample {i+1} ({viz_ticker}) ---")
    #     print("Predicted next 5 days:", [float(f"{p:.2f}") for p in preds])
    #     print("Actual next 5 days:", [float(f"{a:.2f}") for a in actuals])
    #
    #     visualize_test(preds, actuals, historical_prices, dates=plot_dates, ticker=viz_ticker)
    #
    #     print(f"prediction MSE: {np.mean(np.square(preds - actuals)):.4f}")
    #     print("\n")