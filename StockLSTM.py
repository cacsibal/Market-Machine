from torch import nn

class StockLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout=0.2):
        super(StockLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,  # 5 features per timestep
            hidden_size=hidden_size,  # LSTM hidden dimension
            num_layers=num_layers,  # Stack 2 LSTM layers
            dropout=dropout,
            batch_first=True  # Input shape: (batch, seq_len, features)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch, 10, 5)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        last_hidden = h_n[-1]  # Shape: (batch, hidden_size)
        out = self.fc(last_hidden)
        return out.squeeze(-1)