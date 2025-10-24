import pandas as pd
import numpy as np

df = pd.read_csv("percent_tree_cover_all_years.csv")
df = df[(df["2000"] != 200) & (df["2000"] != 253)]

bin_size = 10000
df["x_bin"] = (df["x"] // bin_size).astype(int)
df["y_bin"] = (df["y"] // bin_size).astype(int)

# This line averages each *year column* across all points in the same (x_bin, y_bin)
df_coarse = df.groupby(["x_bin", "y_bin"]).mean(numeric_only=True).reset_index()

print(df_coarse.head())

df_coarse.to_csv("course_data.csv")