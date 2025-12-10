import matplotlib.pyplot as plt

def visualize_test(predictions, actuals, historical_prices=None, dates=None, ticker="Stock"):
    plt.figure(figsize=(12, 6))

    if historical_prices is not None:
        if dates is not None and len(dates) == len(historical_prices) + len(predictions):
            historical_dates = dates[:len(historical_prices)]
            prediction_dates = dates[len(historical_prices):]
            x_values = historical_dates + prediction_dates
            historical_x = historical_dates
            pred_x = prediction_dates
        else:
            historical_x = range(-len(historical_prices), 0)
            pred_x = range(len(predictions))
            x_values = list(historical_x) + list(pred_x)
    else:
        if dates is not None and len(dates) == len(predictions):
            pred_x = dates
            x_values = dates
        else:
            pred_x = range(len(predictions))
            x_values = list(pred_x)


    if historical_prices is not None:
        plt.plot(historical_x, historical_prices, label="Historical", marker="o", color="blue", alpha=0.7)

    plt.plot(pred_x, predictions, label="Predictions", marker="o", color="orange", linewidth=2)
    plt.plot(pred_x, actuals, label="Actuals", marker="s", color="green", linewidth=2)

    if historical_prices is not None:
        plt.axvline(x=historical_x[-1] if historical_x else -0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label="Forecast Start")
    else:
        plt.axvline(x=-0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.5, label="Forecast Start")

    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.title(f"Stock Price Predictions vs Actuals for {ticker}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if dates is not None and len(dates) == len(historical_prices) + len(predictions):
        tick_indices = range(0, len(x_values), max(1, len(x_values) // 8))
        plt.xticks([x_values[i] for i in tick_indices], [dates[i] for i in tick_indices], rotation=45, ha='right')
    elif dates is not None and len(dates) == len(predictions):
        tick_indices = range(0, len(dates), max(1, len(dates) // 5))
        plt.xticks([dates[i] for i in tick_indices], [dates[i] for i in tick_indices], rotation=45, ha='right')
    else:
        pass

    plt.tight_layout()
    plt.show()