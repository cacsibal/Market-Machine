import matplotlib.pyplot as plt
import numpy as np

def visualize_test(predictions, actuals, historical_prices=None, dates=None, ticker='SPY'):
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

def visualize_future(predictions, ticker='SPY', lookback=60, period='1y'):
    from yfinance_test import get_samples
    import numpy as np

    samples, means, stds, dates, _ = get_samples(
        ticker,
        period=period,
        lookback=lookback,
        forecast_days=len(predictions)
    )

    latest_sample = samples[-1]
    latest_mean = means[-1]
    latest_std = stds[-1]
    latest_dates = dates[-1]

    historical_norm = latest_sample[:lookback, 3]
    historical_prices = historical_norm * latest_std[3] + latest_mean[3]

    historical_dates = latest_dates[:lookback].tolist()

    prediction_indices = list(range(lookback, lookback + len(predictions)))

    plt.figure(figsize=(14, 7))

    plt.plot(range(len(historical_prices)), historical_prices,
             label="Historical", marker="o", color="blue", alpha=0.7, markersize=3)

    plt.plot(prediction_indices, predictions,
             label="Predictions", marker="o", color="orange", linewidth=2.5, markersize=6)

    plt.axvline(x=len(historical_prices) - 0.5, color='red', linestyle='--',
                linewidth=2, alpha=0.6, label="Forecast Start")

    plt.plot([len(historical_prices) - 1, lookback],
             [historical_prices[-1], predictions[0]],
             color='gray', linestyle=':', alpha=0.5, linewidth=1.5)

    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Price ($)", fontsize=12)
    plt.title(f"Future Price Predictions for {ticker} (Next {len(predictions)} Days)", fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    num_ticks = 10
    tick_indices = np.linspace(0, len(historical_prices) - 1, num_ticks // 2, dtype=int)
    tick_labels = [historical_dates[i] for i in tick_indices]
    tick_positions = list(tick_indices)

    for i, pred_idx in enumerate(prediction_indices):
        tick_positions.append(pred_idx)
        tick_labels.append(f"Day +{i + 1}")

    plt.xticks(tick_positions, tick_labels, rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

def visualize_pca(model, data_loader, lookback=60, input_size=5):
    import torch
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_inputs = []
    all_preds = []

    model.eval()
    with torch.no_grad():
        for x, _, _, _ in data_loader:
            x = x.to(device)
            preds = model(x)

            pred_scalar = preds.mean(dim=1).cpu().numpy()

            flat_input = x.cpu().numpy().reshape(x.shape[0], -1)

            all_inputs.append(flat_input)
            all_preds.append(pred_scalar)

    X_flat = np.concatenate(all_inputs, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_flat)

    x_min, x_max = X_pca[:, 0].min(), X_pca[:, 0].max()
    y_min, y_max = X_pca[:, 1].min(), X_pca[:, 1].max()

    padding = 1.0
    x_range = np.linspace(x_min - padding, x_max + padding, 50)
    y_range = np.linspace(y_min - padding, y_max + padding, 50)
    xx, yy = np.meshgrid(x_range, y_range)

    grid_points = np.c_[xx.ravel(), yy.ravel()]

    synthetic_flat = pca.inverse_transform(grid_points)
    synthetic_inputs = synthetic_flat.reshape(-1, lookback, input_size)
    synthetic_tensor = torch.tensor(synthetic_inputs, dtype=torch.float32).to(device)

    with torch.no_grad():
        grid_preds = model(synthetic_tensor)
        grid_z = grid_preds.mean(dim=1).cpu().numpy()

    Z = grid_z.reshape(xx.shape)

    fig = go.Figure()

    fig.add_trace(go.Surface(
        z=Z, x=xx, y=yy,
        colorscale='Viridis',
        opacity=0.8,
        name='Model Landscape',
        colorbar=dict(title='Predicted Price (Norm)')
    ))

    fig.add_trace(go.Scatter3d(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        z=y_pred,
        mode='markers',
        marker=dict(
            size=3,
            color=y_pred,
            colorscale='Plasma',
            opacity=0.9
        ),
        name='Actual Samples',
        hovertemplate='PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>Pred: %{z:.2f}<extra></extra>'
    ))

    fig.update_layout(
        title='PCA for g(x)',
        scene=dict(
            xaxis_title='Principal Component 1',
            yaxis_title='Principal Component 2',
            zaxis_title='Model Prediction (yhat)',
            aspectmode='cube'
        ),
        width=1000,
        height=800,
        margin=dict(r=20, l=10, b=10, t=40)
    )

    print(f'Variance: {pca.explained_variance_ratio_}')
    print(f'Sum of variance across first 2 dimensions: {sum(pca.explained_variance_ratio_)}')
    fig.show()