import os
import json
import pandas as pd
import numpy as np
from skimage.transform import resize

# For AR forecasting
from AdaptiveAutoRegForecaster import AdaptiveAutoRegForecaster

# -------------------------------
# CONFIG
# -------------------------------
input_folder = "data/tiles_csv"
output_json = "data/processed_tiles_with_forecast.json"
target_resolution = 100  # downsample to 100×100

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def downsample_cube(cube, res):
    """Downsample cube (time, h, w) to (time, res, res) while handling NaNs"""
    down = []
    for t in range(cube.shape[0]):
        img = np.nan_to_num(cube[t], nan=0.0)  # replace NaNs with 0
        img_resized = resize(img, (res, res), order=1, preserve_range=True, anti_aliasing=True)
        down.append(img_resized.tolist())
    return down

def convert(o):
    """Convert numpy types to JSON-serializable types"""
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o

# Ensure output folder exists
os.makedirs(os.path.dirname(output_json), exist_ok=True)

# -------------------------------
# PROCESS ALL TILES
# -------------------------------
tiles_json = {}

for file in sorted(os.listdir(input_folder)):
    if not file.endswith(".csv"):
        continue

    tile_name = os.path.splitext(file)[0]
    csv_path = os.path.join(input_folder, file)

    print(f"Processing tile {tile_name}...")

    # Load CSV
    df = pd.read_csv(csv_path)
    year_columns = [col for col in df.columns if col not in ["x", "y"]]
    num_years = len(year_columns)
    years = [int(y) for y in year_columns]

    # Reconstruct full raster for historical data
    max_x = int(df["x"].max()) + 1
    max_y = int(df["y"].max()) + 1
    historical = np.full((num_years, max_y, max_x), np.nan, dtype=float)
    for i, year in enumerate(year_columns):
        historical[i][df["y"], df["x"]] = df[year].values

    # -------------------------------
    # FORECAST
    # -------------------------------
    height, width = historical.shape[1], historical.shape[2]
    forecast_horizon = num_years  # same as number of historical years
    forecast = np.full((forecast_horizon, height, width), np.nan, dtype=float)

    for row in range(height):
        for col in range(width):
            ts = historical[:, row, col]
            if np.isnan(ts).all():
                continue
            try:
                f = AdaptiveAutoRegForecaster()
                f.fit(ts)
                pred_diff = f.forecast(steps=forecast_horizon)
                pred_orig = f.invert_difference(pred_diff, last_observed_value=ts[-1])
                forecast[:, row, col] = pred_orig
            except Exception:
                continue

    # -------------------------------
    # DOWNSAMPLE
    # -------------------------------
    hist_down = downsample_cube(historical, target_resolution)
    forecast_down = downsample_cube(forecast, target_resolution)

    # -------------------------------
    # ADD TO JSON
    # -------------------------------
    tiles_json[tile_name] = {
        "years": years,
        "shape": [target_resolution, target_resolution],
        "data": hist_down,
        "forecast_years": [years[-1] + i + 1 for i in range(forecast_horizon)],
        "forecast": forecast_down
    }

# -------------------------------
# SAVE JSON
# -------------------------------
with open(output_json, "w") as f:
    json.dump(tiles_json, f, default=convert)

print(f"✅ Saved JSON for {len(tiles_json)} tiles to {output_json}")
