# import numpy as np
# from MoD44BLoader import MoD44BLoader
# from AdaptiveAutoRegForecaster import AdaptiveAutoRegForecaster


# loader = MoD44BLoader(
#     data_dir="./data",
#     sds_name="Percent_Tree_Cover"
# )

# print(f"Loaded {loader.get_num_time_steps()} time steps")
# print(f"Dimensions: {loader.get_dimensions()}")
# print(f"Years: {loader.get_time_years()}")

# # 2. Search for pixels with actual temporal variation
# print("\nSearching for pixels with changing tree cover...")
# found_good_pixel = False

# for i in range(0, 4800, 100):  # Sample every 100 pixels
#     for j in range(0, 4800, 100):
#         pixel_series = loader.get_pixel_time_series(i=i, j=j)
        
#         # Check if there's actual variation (not constant)
#         if np.std(pixel_series) > 0.1:  # Some real variation
#             print(f"\nFound changing pixel at ({i},{j}): {pixel_series}")
#             print(f"  Change: {pixel_series[-1] - pixel_series[0]}")
#             found_good_pixel = True
#             break
#     if found_good_pixel:
#         break

# if not found_good_pixel:
#     print("\nNo pixels with temporal variation found.")
#     print("This could mean:")
#     print("  1. The area is very stable (no deforestation/reforestation)")
#     print("  2. This tile is mostly ocean/water")
#     print("  3. Need more time steps to see change")
    
#     # Show some sample pixels anyway
#     print("\nSample pixel values:")
#     for i in range(500, 1500, 500):
#         for j in range(500, 1500, 500):
#             pixel = loader.get_pixel_time_series(i=i, j=j)
#             print(f"  Pixel ({i},{j}): {pixel}")
#     exit()

# # 3. Fit model if we found variation
# forecaster = AdaptiveAutoRegForecaster(
#     significance=0.05,
#     max_lags=2,
#     lags=1  # Explicitly use 1 lag with only 3 data points
# )

# forecaster.fit(pixel_series)

# print(f"\nModel fitted!")
# print(f"ADF p-value: {forecaster.get_adf_pvalue():.4f}")
# print(f"Differenced: {forecaster.is_differenced()}")
# print(f"Selected lags: {forecaster.get_lags()}")

# # 4. Make forecast
# future_steps = 1
# forecast_vals = forecaster.forecast(steps=future_steps)

# if forecaster.is_differenced():
#     last_value = pixel_series[-1]
#     forecast_vals = forecaster.invert_difference(forecast_vals, last_value)

# print(f"\nForecast for next {future_steps} time step: {forecast_vals}")
# print(f"Current value (2024): {pixel_series[-1]}")
# print(f"Predicted value (2025): {forecast_vals[0]:.2f}")

import numpy as np
import pandas as pd
from MoD44BLoader import MoD44BLoader
from prophet_forecaster import AdaptiveProphetForecaster

# 1. Load your MOD44B data
loader = MoD44BLoader(
    data_dir="./data",
    sds_name="Percent_Tree_Cover"
)

print(f"Loaded {loader.get_num_time_steps()} time steps")
print(f"Years: {loader.get_time_years()}")

# 2. Get time series for a pixel
i, j = 0, 3400  # Use the pixel that worked before
pixel_series = loader.get_pixel_time_series(i=i, j=j)
print(f"\nPixel ({i},{j}) time series: {pixel_series}")

# 3. Create dates for the series
dates = pd.DatetimeIndex(loader.get_time_datetimes())

# 4. Fit Prophet model
prophet_forecaster = AdaptiveProphetForecaster(
    seasonality_mode='additive',
    yearly_seasonality=False,  # Disable for annual data
    changepoint_prior_scale=0.05
)

prophet_forecaster.fit(pixel_series, dates=dates)

print(f"\nModel fitted!")
print(f"Parameters: {prophet_forecaster.get_parameters()}")

# 5. Forecast future values
future_steps = 3
forecast_vals = prophet_forecaster.forecast(steps=future_steps, freq='YS')

print(f"\nForecast for next {future_steps} years:")
for i, val in enumerate(forecast_vals, 1):
    print(f"  Year +{i}: {val:.2f}%")

# 6. Get detailed forecast with uncertainty intervals
forecast_details = prophet_forecaster.get_forecast_components(steps=future_steps, freq='YS')
print(f"\nDetailed forecast:")
print(forecast_details[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])