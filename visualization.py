from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import torch


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
        data_dict: Dictionary with 'normalized', 'close_prices', and 'dates' keys
        start_idx: Starting index for the plot
        num_days: Number of days to plot
        ticker_name: Name of the stock for the title
    """
    normalized = data_dict['normalized']
    close_prices = data_dict['close_prices']
    dates = data_dict.get('dates', None)

    end_idx = min(start_idx + num_days, len(normalized))
    plot_data = normalized[start_idx:end_idx]
    plot_closes = close_prices[start_idx:end_idx]

    if dates is not None:
        plot_dates = dates[start_idx:end_idx]
    else:
        plot_dates = np.arange(len(plot_data))

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    if dates is not None:
        fig.suptitle(
            f'{ticker_name} - Features ({plot_dates[0].strftime("%Y-%m-%d")} to {plot_dates[-1].strftime("%Y-%m-%d")})',
            fontsize=16)
    else:
        fig.suptitle(f'{ticker_name} - Features (Days {start_idx} to {end_idx})', fontsize=16)

    # Plot 1: Normalized OHLC
    ax1 = axes[0]
    ax1.plot(plot_dates, plot_data[:, 0], label='Open (normalized)', alpha=0.7)
    ax1.plot(plot_dates, plot_data[:, 1], label='High (normalized)', alpha=0.7)
    ax1.plot(plot_dates, plot_data[:, 2], label='Low (normalized)', alpha=0.7)
    ax1.plot(plot_dates, plot_data[:, 3], label='Close (normalized)', alpha=0.7, linewidth=2)
    ax1.set_ylabel('Normalized Price')
    ax1.set_title('Normalized OHLC (relative to daily open)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    if dates is not None:
        fig.autofmt_xdate()

    # Plot 2: Actual Close Prices
    ax2 = axes[1]
    ax2.plot(plot_dates, plot_closes, color='blue', linewidth=2)
    ax2.set_ylabel('Close Price ($)')
    ax2.set_title('Actual Close Prices')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Normalized Volume
    ax3 = axes[2]
    ax3.bar(plot_dates, plot_data[:, 4], alpha=0.6, color='purple')
    ax3.set_xlabel('Date' if dates else 'Days')
    ax3.set_ylabel('Normalized Volume')
    ax3.set_title('Normalized Volume (z-score)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_training_history(train_losses, val_losses):
    """Plot training and validation loss over epochs."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_training_results(results):
    """Print training results and metrics."""
    print(f"\nFinal Results:")
    print(f"Best Validation Loss: {results['best_val_loss']:.6f}")
    print(f"Test Loss: {results['test_loss']:.6f}")

    if results['test_predictions'] is not None and results['test_targets'] is not None:
        from sklearn.metrics import mean_absolute_error, r2_score

        print(f"\nPer-day metrics:")
        for day in range(results['test_predictions'].shape[1]):
            mae = mean_absolute_error(results['test_targets'][:, day], results['test_predictions'][:, day])
            r2 = r2_score(results['test_targets'][:, day], results['test_predictions'][:, day])
            print(f"  Day {day + 1}: MAE = {mae:.6f} ({mae * 100:.2f}%), R² = {r2:.6f}")

        # Overall metrics
        mae_overall = mean_absolute_error(results['test_targets'].flatten(), results['test_predictions'].flatten())
        r2_overall = r2_score(results['test_targets'].flatten(), results['test_predictions'].flatten())
        print(f"\nOverall: MAE = {mae_overall:.6f} ({mae_overall * 100:.2f}%), R² = {r2_overall:.6f}")


def plot_prediction_results(predicted_closes, actual_closes, forecast_dates, last_known_close, ticker_name,
                            dates_available=True):
    """Plot predicted vs actual close prices with error analysis."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'{ticker_name} - {len(predicted_closes)} Day Price Prediction', fontsize=16)

    # Plot 1: Predicted vs Actual Close Prices
    ax1 = axes[0]
    ax1.plot(forecast_dates, actual_closes, 'o-', label='Actual', linewidth=2, markersize=8, color='blue')
    ax1.plot(forecast_dates, predicted_closes, 's--', label='Predicted', linewidth=2, markersize=8, color='red')
    ax1.axhline(y=last_known_close, color='green', linestyle=':', linewidth=2,
                label=f'Last Known Close (${last_known_close:.2f})')
    ax1.set_ylabel('Close Price ($)')
    ax1.set_title('Predicted vs Actual Close Prices')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    if dates_available:
        ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Prediction Errors
    ax2 = axes[1]
    errors = predicted_closes - actual_closes
    error_pct = (errors / actual_closes) * 100
    colors = ['red' if e > 0 else 'green' for e in errors]
    ax2.bar(forecast_dates, error_pct, color=colors, alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Date' if dates_available else 'Day')
    ax2.set_ylabel('Prediction Error (%)')
    ax2.set_title('Prediction Errors (Predicted - Actual)')
    ax2.grid(True, alpha=0.3)
    if dates_available:
        ax2.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def print_prediction_results(predicted_closes, actual_closes, forecast_dates, last_known_close, ticker_name,
                             dates_available=True):
    """Print prediction results in a formatted table."""
    errors = predicted_closes - actual_closes
    error_pct = (errors / actual_closes) * 100

    print(f"\n{ticker_name} Predictions:")
    print(f"Last Known Close: ${last_known_close:.2f}")
    print(f"\n{'Day':<5} {'Date':<12} {'Actual':<10} {'Predicted':<10} {'Error':<10} {'Error %':<10}")
    print("-" * 70)

    for i in range(len(predicted_closes)):
        if dates_available:
            date_str = forecast_dates[i].strftime('%Y-%m-%d')
        else:
            date_str = str(i + 1)
        print(
            f"{i + 1:<5} {date_str:<12} ${actual_closes[i]:<9.2f} ${predicted_closes[i]:<9.2f} ${errors[i]:<9.2f} {error_pct[i]:<9.2f}%")

    # Calculate metrics
    mae = np.mean(np.abs(errors))
    mape = np.mean(np.abs(error_pct))
    rmse = np.sqrt(np.mean(errors ** 2))

    print(f"\nMetrics:")
    print(f"  MAE:  ${mae:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  RMSE: ${rmse:.2f}")

    return mae, mape, rmse


def plot_future_prediction(predicted_closes, forecast_dates, close_prices, dates, ticker_name):
    """Plot future predictions along with historical data."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot historical data (last 30 days)
    if dates is not None:
        hist_dates = dates[-30:]
        hist_closes = close_prices[-30:]
        dates_available = True
    else:
        hist_dates = list(range(-29, 1))
        hist_closes = close_prices[-30:]
        dates_available = False

    ax.plot(hist_dates, hist_closes, 'o-', label='Historical', linewidth=2, markersize=6, color='blue')
    ax.plot(forecast_dates, predicted_closes, 's--', label='Predicted', linewidth=2, markersize=8, color='red')
    ax.axvline(x=hist_dates[-1], color='gray', linestyle=':', linewidth=2, alpha=0.5)

    ax.set_xlabel('Date' if dates_available else 'Day')
    ax.set_ylabel('Close Price ($)')
    ax.set_title(f'{ticker_name} - Future Price Prediction ({len(predicted_closes)} Days)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if dates_available:
        fig.autofmt_xdate()

    plt.tight_layout()
    plt.show()


def print_future_prediction(predicted_closes, forecast_dates, last_known_close, ticker_name, dates_available=True):
    """Print future prediction results."""
    print(f"\n{ticker_name} Future Predictions:")
    print(f"Current Close: ${last_known_close:.2f}")
    print(f"\n{'Day':<5} {'Date':<12} {'Predicted Price':<15} {'Change':<10} {'Change %':<10}")
    print("-" * 65)

    for i in range(len(predicted_closes)):
        if dates_available:
            date_str = forecast_dates[i].strftime('%Y-%m-%d')
        else:
            date_str = str(i + 1)
        change = predicted_closes[i] - last_known_close
        change_pct = (change / last_known_close) * 100
        print(f"{i + 1:<5} {date_str:<12} ${predicted_closes[i]:<14.2f} ${change:<9.2f} {change_pct:<9.2f}%")