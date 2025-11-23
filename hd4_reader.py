from pyhdf.SD import SD, SDC
import pandas as pd
import numpy as np
import os
import re

input_folder = "data"
output_folder = os.path.join(input_folder, "tiles_csv")
os.makedirs(output_folder, exist_ok=True)

start_year = 2000

for file in sorted(os.listdir(input_folder)):
    if not file.endswith(".hdf"):
        continue

    # Extract tile name (e.g., h13v03)
    tile_match = re.search(r'h\d{2}v\d{2}', file)
    tile_name = tile_match.group(0) if tile_match else os.path.splitext(file)[0]

    output_csv = os.path.join(output_folder, f"{tile_name}.csv")

    hdf = SD(os.path.join(input_folder, file), SDC.READ)
    data = hdf.select("Percent_Tree_Cover")[:]
    hdf.end()

    # Mask invalid values
    data = np.where((data > 100) | (data < 0), np.nan, data)

    # Convert to long format
    rows, cols = np.where(~np.isnan(data))
    xs, ys = cols, rows
    values = data[rows, cols]

    df = pd.DataFrame({
        "x": xs,
        "y": ys,
        str(start_year): values
    })

    df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")
