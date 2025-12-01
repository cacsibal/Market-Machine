from matplotlib import pyplot as plt
from datetime import datetime, timedelta

def plot_mse_loss(train_losses, test_losses):
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.show()


def print_predictions(ticker, predictions, n_days):
    """Print formatted prediction results."""
    print(f"\n{'=' * 60}")
    print(f"Predictions for {ticker}")
    print(f"{'=' * 60}")
    print(f"Current Price: ${predictions['current_price']:.2f}")
    print(f"\nPredicted prices for the next {n_days} days:")
    print(f"{'Day':<6} {'Date':<12} {'Predicted Price':<18} {'Daily Return':<15}")
    print(f"{'-' * 60}")

    # Generate dates starting from tomorrow
    start_date = datetime.now() + timedelta(days=1)

    for i, (price, ret) in enumerate(zip(predictions['predicted_prices'],
                                         predictions['predicted_returns']), 1):
        pred_date = start_date + timedelta(days=i - 1)
        date_str = pred_date.strftime('%Y-%m-%d')
        print(f"{i:<6} {date_str:<12} ${price:<17.2f} {ret * 100:>6.2f}%")

    total_change = predictions['predicted_prices'][-1] - predictions['current_price']
    total_change_pct = (total_change / predictions['current_price']) * 100

    print(f"\n{'-' * 60}")
    print(f"Total predicted change: ${total_change:+.2f} ({total_change_pct:+.2f}%)")
    print(f"{'=' * 60}\n")