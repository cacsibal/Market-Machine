import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
import time
from datetime import datetime

from StockDataset import StockDataset
from StockLSTM import StockLSTM
from visualization import visualize_test, visualize_future, visualize_pca
from yfinance_test import get_samples

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MODELS_DIR = 'saved_models'

def save_model(model, filepath, hyperparams=None):

    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else MODELS_DIR, exist_ok=True)

    save_dict = {
        'model_state_dict': model.state_dict(),
        'timestamp': datetime.now().isoformat(),
    }

    if hyperparams:
        save_dict['hyperparams'] = hyperparams

    torch.save(save_dict, filepath)
    print(f"Model saved to: {filepath}")

def load_model(filepath, input_size=5, hidden_size=128, forecast_days=5, num_layers=3, dropout=0.2):

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No saved model found at: {filepath}")

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)

    # Use saved hyperparams if available
    if 'hyperparams' in checkpoint:
        hp = checkpoint['hyperparams']
        input_size = hp.get('input_size', input_size)
        hidden_size = hp.get('hidden_size', hidden_size)
        forecast_days = hp.get('forecast_days', forecast_days)
        num_layers = hp.get('num_layers', num_layers)
        dropout = hp.get('dropout', dropout)

    model = StockLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        forecast_days=forecast_days,
        num_layers=num_layers,
        dropout=dropout
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    timestamp = checkpoint.get('timestamp', 'unknown')
    print(f"Model loaded from: {filepath} (saved at: {timestamp})")

    return model

def train_model(model, train_loader, test_loader, epochs=10, epsilon=0.005, lambda_reg=0.0001):
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=epsilon)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        epoch_start_time = time.time()

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

        epoch_time = time.time() - epoch_start_time

        train_dollar_loss = np.sqrt(train_loss)
        test_dollar_loss = np.sqrt(test_loss)
        print(f"Epoch {epoch + 1}/{epochs} | Train Dollar Loss: ${train_dollar_loss:.2f} | Test Dollar Loss: ${test_dollar_loss:.2f} | Difference: ${train_dollar_loss - test_dollar_loss:.2f} | Time: {epoch_time:.2f}s")

    return model

def tune_model(model, tickers: list[str], training_proportion=0.8, period='1y', lookback=10, forecast_days=5,
               epochs=2, epsilon=0.01, lambda_reg=0.0001, save_path=None, model_name=None):
    train_loader, test_loader, _ = get_loaders(
        tickers,
        training_proportion=training_proportion,
        period=period, lookback=lookback,
        forecast_days=forecast_days
    )

    tuned_model = train_model(
        model,
        train_loader,
        test_loader,
        epochs=epochs,
        epsilon=epsilon,
        lambda_reg=lambda_reg
    )

    if model_name is None:
        ticker_str = '_'.join(tickers[:3])  # use first 3 tickers in name
        if len(tickers) > 3:
            ticker_str += f'_+{len(tickers) - 3}more'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"tuned_{ticker_str}_{timestamp}"

    if save_path is None:
        save_path = os.path.join(MODELS_DIR, f"{model_name}.pt")

    hyperparams = {
        'input_size': 5,
        'hidden_size': model.lstm.hidden_size,
        'forecast_days': forecast_days,
        'num_layers': model.lstm.num_layers,
        'dropout': model.lstm.dropout,
        'lookback': lookback,
        'tickers': tickers,
        'period': period,
        'epochs': epochs,
        'epsilon': epsilon,
        'lambda_reg': lambda_reg,
    }

    save_model(tuned_model, save_path, hyperparams)

    return tuned_model

def predict(model, sample, mean, std, lookback=10):
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
    print(f"Training on the following {len(tickers)} stocks/ETFs: {tickers}")

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
    print('number of cores:', os.cpu_count())

    # hyperparameters
    lookback = 60
    forecast_days = 5
    hidden_size = 128
    batch_size = 64
    epochs = 15
    epsilon = 0.01
    lambda_reg = 1e-7
    num_layers = 3

    data_collection_period = '5y'
    training_proportion = 0.8

    tickers = [
        # core etfs (broad market & sectors)
        'SPY',  # s&p 500
        'XLK',  # tech
        'XLF',  # financial
        'XLV',  # healthcare
        'XLE',  # energy
        'XLI',  # industrials
        'XLY',  # consumer
        'XLRE',  # real estate
        'XLU',  # utilities
        'XLC',  # communication
        'SOXX',  # semiconductors
        'QTUM',  # quantum

        # satellite etfs (global & style exposure)
        'EFA',  # developed international equities
        'EEM',  # emerging markets
        'IWM',  # small-cap u.s. equities

        'GLD',  # gold
        'SLV',  # silver
    ]

    tech = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'AVGO', 'ORCL', 'PLTR']
    quantum = ['IONQ', 'RGTI', 'QBTS', 'ARQQ', 'QUBT']

    # Model save paths
    BASE_MODEL_PATH = os.path.join(MODELS_DIR, 'base_model.pt')

    base_hyperparams = {
        'input_size': 5,
        'hidden_size': hidden_size,
        'forecast_days': forecast_days,
        'num_layers': num_layers,
        'dropout': 0.2,
        'lookback': lookback,
        'tickers': tickers,
        'period': data_collection_period,
        'epochs': epochs,
        'epsilon': epsilon,
        'lambda_reg': lambda_reg,
    }

    retrain = True

    if retrain:
        model = StockLSTM(input_size=5, hidden_size=hidden_size, num_layers=num_layers, forecast_days=forecast_days)

        train_loader, test_loader, testing_samples = get_loaders(
            tickers, training_proportion,
            period=data_collection_period, lookback=lookback,
            forecast_days=forecast_days
        )

        trained_model = train_model(
            model, train_loader, test_loader,
            epochs=epochs, epsilon=epsilon, lambda_reg=lambda_reg
        )

        save_model(trained_model, BASE_MODEL_PATH, base_hyperparams)
    else:
        print("Loading saved model...")
        trained_model = load_model(
            BASE_MODEL_PATH,
            input_size=5,
            hidden_size=hidden_size,
            forecast_days=forecast_days
        )

        _, _, testing_samples = get_loaders(
            tickers, training_proportion,
            period=data_collection_period, lookback=lookback,
            forecast_days=forecast_days
        )

    future_aapl_preds = predict_future(trained_model, 'NVDA', lookback=lookback, forecast_days=forecast_days)
    visualize_future(future_aapl_preds, ticker='NVDA', lookback=lookback)

    # tune on tech stocks
    tech_model = tune_model(
        trained_model,
        tech,
        training_proportion=training_proportion,
        period=data_collection_period,
        lookback=lookback,
        forecast_days=forecast_days,
        model_name='tech_tuned'
    ) if retrain else load_model(
        MODELS_DIR + '/tech_tuned.pt',
        input_size=5,
        hidden_size=hidden_size,
        forecast_days=forecast_days
    )

    tuned_future_aapl_preds = predict_future(tech_model, 'NVDA', lookback=lookback, forecast_days=forecast_days)
    visualize_future(tuned_future_aapl_preds, ticker='NVDA', lookback=lookback)

    # tune on quantum
    # quantum_model = tune_model(
    #     trained_model,
    #     quantum,
    #     training_proportion=training_proportion,
    #     period=data_collection_period,
    #     lookback=lookback,
    #     forecast_days=forecast_days,
    #     model_name='quantum_tuned'
    # ) if retrain else load_model(
    #     MODELS_DIR + '/quantum_tuned.pt',
    #     input_size=5,
    #     hidden_size=hidden_size,
    #     forecast_days=forecast_days
    # )
    #
    # future_ionq_preds = predict_future(quantum_model, 'IONQ', lookback=lookback, forecast_days=forecast_days)
    # visualize_future(future_ionq_preds, ticker='IONQ', lookback=lookback)

    train_loader, test_loader, _ = get_loaders(
        tickers, training_proportion,
        period=data_collection_period, lookback=lookback,
        forecast_days=forecast_days
    )

    visualize_pca(trained_model, test_loader, lookback=lookback, input_size=5)

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