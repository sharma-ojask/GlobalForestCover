import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict
from prophet import Prophet


class AdaptiveProphetForecaster:
    """
    Forecasts a 1-D numpy series using Facebook Prophet with optional configuration.

    Prophet is designed for time series with strong seasonal patterns and works well
    with missing data and outliers. It's particularly good for longer time series.

    Parameters
    ----------
    seasonality_mode : str
        Either 'additive' (default) or 'multiplicative'.
    yearly_seasonality : bool or int
        Fit yearly seasonality. Can be True/False or an integer for Fourier order.
    weekly_seasonality : bool or int
        Fit weekly seasonality. Can be True/False or an integer for Fourier order.
    daily_seasonality : bool or int
        Fit daily seasonality. Can be True/False or an integer for Fourier order.
    changepoint_prior_scale : float
        Controls flexibility of trend (default 0.05). Higher = more flexible.
    seasonality_prior_scale : float
        Controls flexibility of seasonality (default 10.0).
    """

    def __init__(
        self,
        seasonality_mode: str = 'additive',
        yearly_seasonality: bool = True,
        weekly_seasonality: bool = False,
        daily_seasonality: bool = False,
        changepoint_prior_scale: float = 0.05,
        seasonality_prior_scale: float = 10.0
    ) -> None:
        if seasonality_mode not in ['additive', 'multiplicative']:
            raise ValueError("seasonality_mode must be 'additive' or 'multiplicative'")
        if changepoint_prior_scale <= 0:
            raise ValueError("changepoint_prior_scale must be > 0")
        if seasonality_prior_scale <= 0:
            raise ValueError("seasonality_prior_scale must be > 0")

        self._seasonality_mode = seasonality_mode
        self._yearly_seasonality = yearly_seasonality
        self._weekly_seasonality = weekly_seasonality
        self._daily_seasonality = daily_seasonality
        self._changepoint_prior_scale = float(changepoint_prior_scale)
        self._seasonality_prior_scale = float(seasonality_prior_scale)

        self._original_series: Optional[np.ndarray] = None
        self._dates: Optional[pd.DatetimeIndex] = None
        self._model: Optional[Prophet] = None
        self._fitted: bool = False

    def fit(
        self,
        series: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        train_range: Optional[Tuple[int, int]] = None
    ) -> None:
        """
        Fit the Prophet model on the provided series.

        Parameters
        ----------
        series : np.ndarray
            1-D array of values to forecast.
        dates : Optional[pd.DatetimeIndex]
            Datetime index for the series. If None, assumes annual data starting from 2000.
        train_range : Optional[Tuple[int, int]]
            (start_idx, end_idx) to use only a subset for training.
        """
        if series is None or series.ndim != 1 or len(series) == 0:
            raise ValueError("series must be a non-empty 1-D numpy array")
        
        series = np.asarray(series, dtype=float)
        self._original_series = series

        # Handle dates
        if dates is None:
            # Assume annual data starting from year 2000
            dates = pd.date_range(start='2000-01-01', periods=len(series), freq='YS')
        else:
            if len(dates) != len(series):
                raise ValueError(f"dates length ({len(dates)}) must match series length ({len(series)})")
            dates = pd.DatetimeIndex(dates)
        
        self._dates = dates

        # Apply train range if specified
        if train_range is not None:
            start_idx, end_idx = train_range
            if start_idx < 0 or end_idx >= len(series) or start_idx > end_idx:
                raise IndexError("Invalid train_range indices")
            series = series[start_idx : end_idx + 1]
            dates = dates[start_idx : end_idx + 1]

        if len(series) < 2:
            raise ValueError("Training series too short (need at least 2 observations)")

        # Prepare DataFrame for Prophet (requires 'ds' and 'y' columns)
        df = pd.DataFrame({
            'ds': dates,
            'y': series
        })

        # Initialize Prophet model
        self._model = Prophet(
            seasonality_mode=self._seasonality_mode,
            yearly_seasonality=self._yearly_seasonality,
            weekly_seasonality=self._weekly_seasonality,
            daily_seasonality=self._daily_seasonality,
            changepoint_prior_scale=self._changepoint_prior_scale,
            seasonality_prior_scale=self._seasonality_prior_scale
        )

        # Fit the model (suppress Prophet's verbose output)
        import logging
        logging.getLogger('prophet').setLevel(logging.WARNING)
        self._model.fit(df)
        self._fitted = True

    def predict(self, start: int, end: int) -> np.ndarray:
        """
        Make predictions for indices in the training data range.

        Parameters
        ----------
        start : int
            Start index in the original series.
        end : int
            End index in the original series (inclusive).

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        self._ensure_fitted()
        
        if start < 0 or end >= len(self._dates) or start > end:
            raise IndexError(f"Invalid prediction range [{start}, {end}]")

        future_dates = pd.DataFrame({'ds': self._dates[start:end+1]})
        forecast = self._model.predict(future_dates)
        return np.asarray(forecast['yhat'].values, dtype=float)

    def forecast(self, steps: int, freq: str = 'YS') -> np.ndarray:
        """
        Forecast future values beyond the training data.

        Parameters
        ----------
        steps : int
            Number of future time steps to forecast.
        freq : str
            Frequency string for pd.date_range (default 'YS' for yearly).
            Use 'MS' for monthly, 'D' for daily, etc.

        Returns
        -------
        np.ndarray
            Forecasted values.
        """
        self._ensure_fitted()
        
        if steps <= 0:
            raise ValueError("steps must be > 0")

        # Generate future dates
        last_date = self._dates[-1]
        future_dates = pd.date_range(start=last_date, periods=steps+1, freq=freq)[1:]
        
        future_df = pd.DataFrame({'ds': future_dates})
        forecast = self._model.predict(future_df)
        return np.asarray(forecast['yhat'].values, dtype=float)

    def get_forecast_components(self, steps: int, freq: str = 'YS') -> pd.DataFrame:
        """
        Get detailed forecast with trend, seasonality components.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: ds, yhat, yhat_lower, yhat_upper, trend, etc.
        """
        self._ensure_fitted()
        
        if steps <= 0:
            raise ValueError("steps must be > 0")

        last_date = self._dates[-1]
        future_dates = pd.date_range(start=last_date, periods=steps+1, freq=freq)[1:]
        future_df = pd.DataFrame({'ds': future_dates})
        
        return self._model.predict(future_df)

    def get_parameters(self) -> Dict:
        """
        Return the configuration parameters of the model.
        """
        return {
            'seasonality_mode': self._seasonality_mode,
            'yearly_seasonality': self._yearly_seasonality,
            'weekly_seasonality': self._weekly_seasonality,
            'daily_seasonality': self._daily_seasonality,
            'changepoint_prior_scale': self._changepoint_prior_scale,
            'seasonality_prior_scale': self._seasonality_prior_scale
        }

    def is_fitted(self) -> bool:
        """Check if the model has been fitted."""
        return self._fitted

    def _ensure_fitted(self) -> None:
        if not self._fitted or self._model is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
