from torch import nn


class StockLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.2, forecast_days=5):
        super(StockLSTM, self).__init__()

        self.forecast_days = forecast_days

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, forecast_days)
        )

    def forward(self, x):
        # x shape: (batch, 10, 5)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        last_hidden = h_n[-1]  # Shape: (batch, hidden_size)
        out = self.fc(last_hidden)  # Shape: (batch, forecast_days)
        return out