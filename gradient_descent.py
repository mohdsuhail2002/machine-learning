import pandas as pd
import numpy as np

# Expected answer. m = 0.05168176, b=18.0465

def gradient_descent(x, y, lr=0.1, epochs=3000):
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()

    x_scaled = (x - x_min) / (x_max - x_min)
    y_scaled = (y - y_min) / (y_max - y_min)

    b, m = 0.0, 0.0

    for epoch in range(epochs):
        y_pred = b + m * x_scaled
        error = y_scaled - y_pred

        db = -2 * np.mean(error)
        dm = -2 * np.mean(error * x_scaled)

        b -= lr * db
        m -= lr * dm

    # Scale back
    m_original = m * (y_max - y_min) / (x_max - x_min)
    b_original = y_min + (y_max - y_min) * (b - m * x_min / (x_max - x_min))

    return m_original, b_original


if __name__ == "__main__":
    df = pd.read_csv("home_prices.csv")

    x = df["area_sqr_ft"].to_numpy()
    y = df["price_lakhs"].to_numpy()

    m, b = gradient_descent(x, y)

    print(f"Final Results: m={m}, b={b}")


