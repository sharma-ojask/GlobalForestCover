from pyhdf.SD import SD, SDC
import pandas as pd
import numpy as np
import os
import re

input_folder = "data"
output_folder = os.path.join(input_folder, "tiles_csv")
os.makedirs(output_folder, exist_ok=True)

start_year = 2000

# loop through subfolders
for folder in sorted(os.listdir(input_folder)):
    folder_path = os.path.join(input_folder, folder)
    if not os.path.isdir(folder_path):
        continue

    hdf_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".hdf")])
    if not hdf_files:
        continue

    tile_name = folder
    output_csv = os.path.join(output_folder, f"{tile_name}.csv")

    # initialize stacked time-series
    data_layers = []
    years = []

    for i, file in enumerate(hdf_files):
        year = start_year + i
        hdf = SD(os.path.join(folder_path, file), SDC.READ)
        arr = hdf.select("Percent_Tree_Cover")[:]
        hdf.end()

        arr = np.where((arr > 100) | (arr < 0), np.nan, arr)
        data_layers.append(arr)
        years.append(str(year))

    data_layers = np.array(data_layers)

    # convert to long format
    rows, cols = np.where(~np.isnan(data_layers[0]))
    xs, ys = cols, rows

    data_dict = {"x": xs, "y": ys}
    for i, year in enumerate(years):
        data_dict[year] = data_layers[i, ys, xs]

    df = pd.DataFrame(data_dict)
    df.to_csv(output_csv, index=False)

    print(f"Saved {output_csv}")
