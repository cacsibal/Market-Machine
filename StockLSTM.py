import torch.nn as nn

class StockLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=4, forecast_days=5, dropout=0.2):
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