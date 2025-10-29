import numpy as np
import pandas as pd
from typing import Optional, Tuple

from statsmodels.tsa.stattools import adfuller, pacf
from statsmodels.tsa.ar_model import AutoReg


class AdaptiveAutoRegForecaster:
    """
    Forecasts a 1-D numpy series using an AutoReg model with optional differencing.

    Logic:
    - Run ADF on the provided training series; if p-value > significance, apply first differencing.
    - If lags not provided, choose using PACF up to max_lags using a Bartlett bound threshold.
    - Fit statsmodels AutoReg; provide predict/forecast on fitted scale and inversion utility.

    Parameters
    ----------
    significance : float
        ADF p-value threshold to decide differencing (default 0.05).
    max_lags : int
        Upper bound for lag search when lags is not specified.
    lags : Optional[int]
        Explicit lag order; if None, the model will choose via PACF.
    """

    def __init__(self, significance: float = 0.05, max_lags: int = 24, lags: Optional[int] = None) -> None:
        if significance <= 0 or significance >= 1:
            raise ValueError("significance must be between 0 and 1")
        if max_lags <= 0:
            raise ValueError("max_lags must be > 0")
        if lags is not None and lags <= 0:
            raise ValueError("lags must be > 0 when provided")

        self._significance = float(significance)
        self._max_lags = int(max_lags)

        self._lags = int(lags) if lags is not None else None

        self._original_series: Optional[np.ndarray] = None
        self._prepared_series: Optional[pd.Series] = None
        self._adf_pvalue: Optional[float] = None
        self._differenced: Optional[bool] = None

        self._model: Optional[AutoReg] = None
        self._results = None

    def fit(self, series: np.ndarray, train_range: Optional[Tuple[int, int]] = None) -> None:
        if series is None or series.ndim != 1 or len(series) == 0:
            raise ValueError("series must be a non-empty 1-D numpy array")
        series = np.asarray(series, dtype=float)
        self._original_series = series

        if train_range is not None:
            start_idx, end_idx = train_range
            if start_idx < 0 or end_idx >= len(series) or start_idx > end_idx:
                raise IndexError("Invalid train_range indices")
            train_series = series[start_idx : end_idx + 1]
        else:
            train_series = series

        # ADF test on the raw training series
        adf_stat = adfuller(train_series)
        self._adf_pvalue = float(adf_stat[1])
        self._differenced = self._adf_pvalue > self._significance

        if self._differenced:
            prepped = np.diff(train_series)
        else:
            prepped = train_series

        if prepped.size <= 2:
            raise ValueError("Training series too short after preparation for AR modeling")

        self._prepared_series = pd.Series(prepped)
        self._max_lags = min(self._max_lags, (len(self._prepared_series)-1)//2)

        lag_order = self._lags if self._lags is not None else self._select_lags_via_pacf(prepped, self._max_lags)
        # Ensure lag order does not exceed series length - 1
        lag_order = int(max(1, min(lag_order, len(prepped) - 1)))

        self._model = AutoReg(self._prepared_series, lags=lag_order, old_names=False)
        self._results = self._model.fit()

        # Persist chosen lag order
        self._lags = lag_order

    def predict(self, start: int, end: int, dynamic: bool = False) -> np.ndarray:
        self._ensure_fitted()
        preds = self._results.predict(start=start, end=end, dynamic=dynamic)
        return np.asarray(preds, dtype=float)

    def forecast(self, steps: int, dynamic: bool = False) -> np.ndarray:
        self._ensure_fitted()
        if steps <= 0:
            raise ValueError("steps must be > 0")
        start = len(self._prepared_series)
        end = start + steps - 1
        preds = self._results.predict(start=start, end=end, dynamic=dynamic)
        return np.asarray(preds, dtype=float)

    def invert_difference(self, diffs: np.ndarray, last_observed_value: float) -> np.ndarray:
        if not self._differenced:
            return np.asarray(diffs, dtype=float)
        cumsum = np.cumsum(np.asarray(diffs, dtype=float))
        return cumsum + float(last_observed_value)

    def get_adf_pvalue(self) -> float:
        self._ensure_fitted()
        return float(self._adf_pvalue)

    def is_differenced(self) -> bool:
        self._ensure_fitted()
        return bool(self._differenced)

    def get_lags(self) -> int:
        self._ensure_fitted()
        return int(self._lags)

    def _select_lags_via_pacf(self, series_vals: np.ndarray, max_lags: int) -> int:
        # Use PACF to select lag where coefficients fall within significance bounds.
        n = len(series_vals)
        # Bartlett (approx) 95% confidence interval for PACF: ±1.96/sqrt(n)
        threshold = 1.96 / np.sqrt(n)
        pacf_vals = pacf(series_vals, nlags=max_lags, method="yw")
        # pacf_vals[0] corresponds to lag 0 (always 1). Consider lags 1..max_lags
        significant_lags = [lag for lag in range(1, len(pacf_vals)) if abs(pacf_vals[lag]) > threshold]
        if not significant_lags:
            return 1
        return max(significant_lags)

    def _ensure_fitted(self) -> None:
        if self._results is None or self._prepared_series is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")


