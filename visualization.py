from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

def plot_mse_loss(train_losses, test_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.show()


def plot_feature_chart(data_dict, start_idx=0, num_days=30, ticker_name="Stock"):
    """
    Plot OHLCV features from the dataset.

    Args:
        data_dict: Dictionary with 'normalized' and 'close_prices' keys
        start_idx: Starting index for the plot
        num_days: Number of days to plot
        ticker_name: Name of the stock for the title
    """
    normalized = data_dict['normalized']
    close_prices = data_dict['close_prices']

    # Get the slice to plot
    end_idx = min(start_idx + num_days, len(normalized))
    plot_data = normalized[start_idx:end_idx]
    plot_closes = close_prices[start_idx:end_idx]

    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle(f'{ticker_name} - Features (Days {start_idx} to {end_idx})', fontsize=16)

    days = np.arange(len(plot_data))

    # Plot 1: Normalized OHLC
    ax1 = axes[0]
    ax1.plot(days, plot_data[:, 0], label='Open (normalized)', alpha=0.7)
    ax1.plot(days, plot_data[:, 1], label='High (normalized)', alpha=0.7)
    ax1.plot(days, plot_data[:, 2], label='Low (normalized)', alpha=0.7)
    ax1.plot(days, plot_data[:, 3], label='Close (normalized)', alpha=0.7, linewidth=2)
    ax1.set_ylabel('Normalized Price')
    ax1.set_title('Normalized OHLC (relative to daily open)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    # Plot 2: Actual Close Prices
    ax2 = axes[1]
    ax2.plot(days, plot_closes, color='blue', linewidth=2)
    ax2.set_ylabel('Close Price ($)')
    ax2.set_title('Actual Close Prices')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Normalized Volume
    ax3 = axes[2]
    ax3.bar(days, plot_data[:, 4], alpha=0.6, color='purple')
    ax3.set_xlabel('Days')
    ax3.set_ylabel('Normalized Volume')
    ax3.set_title('Normalized Volume (z-score)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()