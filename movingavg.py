import numpy as np

class MovingAverageForecaster:
    """
    Simple Moving Average forecaster.
    
    Given a window size k:
        forecast = average of last k observed values
    """

    def __init__(self, window=5):
        self.window = window
        self.history = None

    def fit(self, time_series):
        self.history = np.array(time_series, dtype=float)

    def forecast(self, steps=3):
        if self.history is None:
            raise ValueError("Call fit() before forecast().")

        forecasts = []
        hist = list(self.history)

        for _ in range(steps):
            if len(hist) < self.window:
                window_vals = hist[:]
            else:
                window_vals = hist[-self.window:]

            next_val = np.mean(window_vals)
            forecasts.append(next_val)

            hist.append(next_val)

        return np.array(forecasts, dtype=float)
