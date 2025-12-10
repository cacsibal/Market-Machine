import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import yfinance as yf

from StockDataset import StockDataset
from visualization import visualize_test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class StockLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=3, forecast_days=5, dropout=0.2):
        super(StockLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )

        self.fc = nn.Linear(hidden_size, forecast_days)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]

        out = self.fc(last_hidden)

        return out


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

def get_samples(ticker: str, period='2y', lookback=10, forecast_days=5):
    data = yf.download(ticker, period=period, auto_adjust=True)
    prices = data.values
    dates = data.index.strftime('%Y-%m-%d').tolist()

    sample_len = lookback + forecast_days

    samples, means, stds, sample_dates, sample_tickers = [], [], [], [], []
    for i in range(len(prices) - sample_len + 1):
        sample_window = prices[i: i + sample_len]
        date_window = dates[i: i + sample_len]

        input_portion = sample_window[:lookback]

        mean = input_portion.mean(axis=0)
        std = input_portion.std(axis=0, ddof=0)

        std[std == 0] = 1.0

        norm_sample = (sample_window - mean) / std

        samples.append(norm_sample)
        means.append(mean)
        stds.append(std)
        sample_dates.append(date_window)
        sample_tickers.append(ticker)

    return np.array(samples), np.array(means), np.array(stds), np.array(sample_dates), np.array(sample_tickers) # Return sample_tickers

def predict(model, sample, mean, std, lookback=10, forecast_days=5):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(sample[:lookback], dtype=torch.float32).unsqueeze(0).to(device)

        preds = model(x)
        preds = preds.squeeze(0).cpu().numpy()

        preds_denorm = preds * std[3] + mean[3]

    return preds_denorm

if __name__ == "__main__":
    print('hello, world!')

    # hyperparameters
    lookback = 60
    forecast_days = 5
    hidden_size = 128
    batch_size = 32
    epochs = 10
    epsilon = 0.01
    lambda_reg = 1e-7

    data_collection_period = '5y'
    proportion_training = 0.8

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

    print(f"Number of stocks: {len(tickers)}")

    all_samples, all_means, all_stds, all_dates, all_tickers = {}, {}, {}, {}, {}

    for ticker in tickers:
        samples, means, stds, dates, sample_tickers = get_samples(
            ticker,
            period=data_collection_period,
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

    print(np.shape(X), np.shape(M), np.shape(S), np.shape(D), np.shape(T))

    num_training_samples = int(proportion_training*len(X))

    training_samples = [X_shuffled[:num_training_samples], M_shuffled[:num_training_samples], S_shuffled[:num_training_samples], D_shuffled[:num_training_samples], T_shuffled[:num_training_samples]]
    testing_samples = [X_shuffled[num_training_samples:], M_shuffled[num_training_samples:], S_shuffled[num_training_samples:], D_shuffled[num_training_samples:], T_shuffled[num_training_samples:]]

    print(np.shape(training_samples[0]), np.shape(testing_samples[0]))

    d_train = StockDataset(training_samples, lookback=lookback, forecast_days=forecast_days)
    d_test = StockDataset(testing_samples, lookback=lookback, forecast_days=forecast_days)

    print(f"Total training examples: {len(d_train)}")
    print(f"Total test examples: {len(d_test)}")

    train_loader = DataLoader(d_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(d_test, batch_size=batch_size, shuffle=False)

    model = StockLSTM(input_size=5, hidden_size=hidden_size, forecast_days=forecast_days)

    trained_model = train_model(model, train_loader, test_loader, epochs=epochs, epsilon=epsilon, lambda_reg=lambda_reg)

    for i in range(5):
        sample = testing_samples[0][i]
        mean = testing_samples[1][i]
        std = testing_samples[2][i]
        dates = testing_samples[3][i]
        viz_ticker = testing_samples[4][i]

        preds = predict(trained_model, sample, mean, std, lookback=lookback, forecast_days=forecast_days)

        actuals_norm = sample[lookback:, 3]
        actuals = actuals_norm * std[3] + mean[3]

        historical_norm = sample[max(0, lookback - 20):lookback, 3]
        historical_prices = historical_norm * std[3] + mean[3]

        historical_dates = dates[max(0, lookback - 20):lookback].tolist()
        prediction_dates = dates[lookback:].tolist()
        plot_dates = historical_dates + prediction_dates


        print(f"--- Visualization Sample {i+1} ({viz_ticker}) ---")
        print("Predicted next 5 days:", preds)
        print("Actual next 5 days:", actuals)
        print("Dates:", prediction_dates)

        visualize_test(preds, actuals, historical_prices, dates=plot_dates, ticker=viz_ticker)