import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from AdaptiveAutoRegForecaster import AdaptiveAutoRegForecaster
from prophet_forecaster import AdaptiveProphetForecaster
from MoD44BLoader import MoD44BLoader

def forecast_auto_regressive(pixel_ts_data, n_years, dataloader=None):
    # Fit model
    forecaster = AdaptiveAutoRegForecaster()
    forecaster.fit(pixel_ts_data)

    # Forecast
    pred_diff = forecaster.forecast(steps=n_years)
    
    # Invert differencing
    predicted = forecaster.invert_difference(
        pred_diff, 
        last_observed_value=pixel_ts_data[-1]
    )

    return predicted

def forecast_prophet(pixel_ts_data, n_years, dataloader):
    dates = pd.DatetimeIndex(dataloader.get_time_datetimes())
    prophet_forecaster = AdaptiveProphetForecaster(
    seasonality_mode='additive',
    yearly_seasonality=False,  # Disable for annual data
        changepoint_prior_scale=0.05
    )
    prophet_forecaster.fit(pixel_ts_data, dates=dates)
    forecast_vals = prophet_forecaster.forecast(steps=n_years, freq='YS')
    return forecast_vals

def main():
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(
        description="Run pixel-wise Forecasting on MoD44B data."
    )

    # Input/Output arguments
    parser.add_argument(
        "-i", "--input_dir", 
        type=str, 
        required=True, 
        help="Path to the directory containing HDF files (e.g., ../project/data/h17v07/)"
    )
    parser.add_argument(
        "-o", "--output_path", 
        type=str, 
        required=True, 
        help="Path where the resulting .npy file will be saved (e.g., ./results/forecast.npy)"
    )

    # Configuration arguments
    parser.add_argument(
        "--sds_name", 
        type=str, 
        default="Percent_Tree_Cover", 
        help="The SDS name to extract from HDF files."
    )
    parser.add_argument(
        "--resolution", 
        type=int, 
        default=100, 
        help="The target resolution (height/width) for the grid."
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=3, 
        help="Number of time steps to forecast."
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default='Autoregressive', 
        help="Type of model used for forecasting. Can be one of 'Autoregressive' or 'Prophet'"
    )

    args = parser.parse_args()

    # Validate paths
    input_path = Path(args.input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    # Ensure output directory exists
    output_file = Path(args.output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Initialize Loader
    print(f"Initializing Loader for {args.input_dir}...")
    loader = MoD44BLoader(
        str(input_path), 
        args.sds_name, 
        resolution=args.resolution 
    )

    print(f"Starting forecast for resolution {args.resolution}x{args.resolution} over {args.steps} steps...")

    # Initialize Result Array
    # Shape: (steps, height, width)
    forecast_results = np.empty((args.steps, args.resolution, args.resolution), dtype=np.float32)

    # Forecasting Loop
    # We use a counter to give simple progress feedback
    total_pixels = args.resolution * args.resolution
    count = 0
    
    for i in range(args.resolution):
        for j in range(args.resolution):
            
            # Get pixel time series
            pixel_ts_data = loader.get_pixel_time_series(i, j)

            if np.all(pixel_ts_data == pixel_ts_data[0]):
                print(f"Skipping forecasting, constant data for pixel ({i},{j})")
                forecast_results[:, i, j] = np.full((args.steps,), pixel_ts_data[0], dtype=np.float32)
                continue

            # Get predictions
            if args.model=='Autoregressive':
                predicted = forecast_auto_regressive(pixel_ts_data, args.steps, loader)
            elif args.model == 'Prophet':
                predicted = forecast_prophet(pixel_ts_data, args.steps, loader)

            # Store results
            forecast_results[:, i, j] = predicted[:]
            
            # Simple progress logging every 10%
            count += 1
            if count % (total_pixels // 10) == 0:
                print(f"Progress: {count / total_pixels:.0%}")

    # 6. Save Results
    print(f"Forecasting complete. Final shape: {forecast_results.shape}")
    print(f"Saving results to {output_file}...")
    np.save(output_file, forecast_results)
    print("Done.")

if __name__ == "__main__":
    main()
    # Example: Run as python predictor.py --input_dir "../project/data/h17v07/" --output_path "../project/data/h17v07/3f.npy" --model "Autoregressive"