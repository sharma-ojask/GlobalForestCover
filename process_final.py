import os
import re
import json
import numpy as np
from pyhdf.SD import SD, SDC
from AdaptiveAutoRegForecaster import AdaptiveAutoRegForecaster

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = "my_data"
OUTPUT_JSON = os.path.join(BASE_DIR, "processed_tiles_with_forecast.json")

TARGET_DATASET = "Percent_Tree_Cover"
TARGET_YEARS = list(range(2010, 2029))      # 2010–2028
RAW_SIZE = 4800
TARGET_RES = 100
BLOCK = RAW_SIZE // TARGET_RES              # 48

# -----------------------------
# HELPERS
# -----------------------------
def clean_nans_2d(arr):
    """Convert NaN → None so JSON is valid."""
    return [[None if np.isnan(v) else float(v) for v in row] for row in arr]

def block_reduce_valid(arr):
    """Downsample by averaging valid pixels."""
    out = np.full((TARGET_RES, TARGET_RES), np.nan)

    for i in range(TARGET_RES):
        for j in range(TARGET_RES):

            block = arr[i*BLOCK:(i+1)*BLOCK, j*BLOCK:(j+1)*BLOCK]
            valid = block[~np.isnan(block)]

            if valid.size > 0:
                out[i, j] = valid.mean()

    return out

# -----------------------------
# MAIN
# -----------------------------
tiles_json = {}

tile_folders = [
    d for d in os.listdir(BASE_DIR)
    if os.path.isdir(os.path.join(BASE_DIR, d)) and re.match(r"h\d{2}v\d{2}", d)
]

for tile in tile_folders:
    print(f"\n🔍 Processing tile {tile}")
    folder = os.path.join(BASE_DIR, tile)

    # Load all years in folder
    year_raw = {}
    for f in os.listdir(folder):
        if not f.endswith(".hdf"):
            continue

        match = re.search(r"A(\d{4})", f)
        if not match:
            continue
        year = int(match.group(1))

        path = os.path.join(folder, f)

        try:
            hdf = SD(path, SDC.READ)
            arr = hdf.select(TARGET_DATASET)[:].astype(float)
            hdf.end()
        except Exception as e:
            print(f"❌ Could not read {f}: {e}")
            continue

        # Mask invalid values
        arr[(arr == 200) | (arr == 253) | (arr == 254)] = np.nan
        arr[(arr < 0) | (arr > 100)] = np.nan

        year_raw[year] = arr

    if len(year_raw) == 0:
        print(f"⚠️ No usable data for {tile}, skipping.")
        continue

    # -----------------------------
    # Build historical (downsampled)
    # -----------------------------
    hist_down = []
    mask = []  # 1=hist, 2=interp, 3=forecast

    for yr in TARGET_YEARS:
        if yr in year_raw:
            ds = block_reduce_valid(year_raw[yr])
            hist_down.append(ds)
            mask.append(np.ones((TARGET_RES, TARGET_RES)))  # historical
        else:
            hist_down.append(np.full((TARGET_RES, TARGET_RES), np.nan))
            mask.append(2 * np.ones((TARGET_RES, TARGET_RES)))  # interpolated placeholder

    hist_down = np.array(hist_down)
    mask = np.array(mask)

    # -----------------------------
    # Interpolation + Forecasting
    # -----------------------------
    full_data = hist_down.copy()
    full_mask = mask.copy()

    for r in range(TARGET_RES):
        for c in range(TARGET_RES):

            ts = full_data[:, r, c]

            if np.isnan(ts).all():
                continue

            real_idx = np.where(~np.isnan(ts))[0]
            if len(real_idx) == 0:
                continue

            # --- Linear Interpolation ---
            for i in range(len(real_idx) - 1):
                a, b = real_idx[i], real_idx[i+1]
                if b > a + 1:
                    v0, v1 = ts[a], ts[b]
                    step = (v1 - v0) / (b - a)
                    for k in range(a+1, b):
                        ts[k] = v0 + step * (k - a)
                        full_mask[k, r, c] = 2

            # --- Forecast (after last real year) ---
            last_idx = real_idx[-1]
            missing = len(TARGET_YEARS) - 1 - last_idx
            if missing > 0:
                observed = ts[~np.isnan(ts)]
                try:
                    f = AdaptiveAutoRegForecaster()
                    f.fit(observed)
                    pred_diff = f.forecast(missing)
                    pred = f.invert_difference(pred_diff, observed[-1])

                    ts[last_idx+1:last_idx+1+missing] = pred
                    full_mask[last_idx+1:last_idx+1+missing] = 3

                except:
                    pass

            full_data[:, r, c] = ts

    # -----------------------------
    # SAVE JSON (correct!)
    # -----------------------------
    tiles_json[tile] = {
        "years": TARGET_YEARS,
        "shape": [TARGET_RES, TARGET_RES],
        "data": [clean_nans_2d(layer) for layer in full_data],   # ✔ REAL FOREST COVER
        "mask": full_mask.tolist()                                # ✔ mask
    }

with open(OUTPUT_JSON, "w") as f:
    json.dump(tiles_json, f)

print(f"\n✅ Saved → {OUTPUT_JSON}")
