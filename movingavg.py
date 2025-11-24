import numpy as np

class ExponentialMovingAverageForecaster:
    """
    Exponential Moving Average (EMA) Forecaster.

    Uses smoothing factor alpha (0 < alpha <= 1).
    Higher alpha → more weight on recent values.
    """

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.last_ema = None

    def fit(self, time_series):
        ts = np.array(time_series, dtype=float)

        # Initialize EMA with the first value
        self.last_ema = ts[0]

        # Compute EMA over the training window
        for x in ts[1:]:
            self.last_ema = self.alpha * x + (1 - self.alpha) * self.last_ema

    def forecast(self, steps=1):
        if self.last_ema is None:
            raise ValueError("Call fit() before forecast().")

        # All future values are the same for EMA (flat forecast)
        return np.full(steps, self.last_ema, dtype=float)
