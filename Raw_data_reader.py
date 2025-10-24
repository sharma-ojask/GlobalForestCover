import rasterio
import pandas as pd
import numpy as np
import os

year = 2000
output_path = "data_raw/h17/"
dfs = []

# Loop through all files in directory
for file in os.listdir(output_path):
    file_path = os.path.join(output_path, file)

    # Skip non-HDF files just in case
    if not file_path.endswith(".hdf"):
        continue

    # Open the HDF file and pick the first subdataset
    with rasterio.open(file_path) as src:
        sub_path = src.subdatasets[0]
        print(f"Year {year}, Using subdataset: {sub_path}")

    # Read data
    with rasterio.open(sub_path) as sub:
        data = sub.read(1)
        transform = sub.transform

    # Get valid pixel indices and coordinates
    rows, cols = np.where(~np.isnan(data))
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    values = data[rows, cols]

    # Create DataFrame for this year
    df_year = pd.DataFrame({
        "x": xs,
        "y": ys,
        str(year): values
    })

    dfs.append(df_year)
    year += 1

# Merge all years on x and y
from functools import reduce

df_merged = reduce(lambda left, right: pd.merge(left, right, on=["x", "y"], how="outer"), dfs)

# Save to CSV
df_merged.to_csv("data/percent_tree_cover_all_years.csv", index=False)
print(f"Saved percent_tree_cover_all_years.csv with {len(df_merged)} rows")