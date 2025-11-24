import argparse
import numpy as np
from pathlib import Path
from MoD44BLoader import MoD44BLoader
from movingavg import MovingAverageForecaster

def main():
    parser = argparse.ArgumentParser(description="Moving Average Forecast on MoD44B data")

    parser.add_argument(
        "-i", "--input_dir",
        type=str,
        required=True,
        help="Folder containing HDF files"
    )

    parser.add_argument(
        "-o", "--output_path",
        type=str,
        required=True,
        help="Where to save .npy results"
    )

    parser.add_argument(
        "--sds_name",
        type=str,
        default="Percent_Tree_Cover",
        help="SDS name from HDF"
    )

    parser.add_argument(
        "--resolution",
        type=int,
        default=100,
        help="Grid resolution"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=3,
        help="Forecast horizon"
    )

    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="Moving average window"
    )

    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize loader
    loader = MoD44BLoader(str(input_path), args.sds_name, resolution=args.resolution)

    forecast_results = np.empty((args.steps, args.resolution, args.resolution), dtype=np.float32)

    total_pixels = args.resolution * args.resolution
    count = 0

    for i in range(args.resolution):
        for j in range(args.resolution):

            ts = loader.get_pixel_time_series(i, j)

            # Skip constant sequences
            if np.all(ts == ts[0]):
                forecast_results[:, i, j] = ts[0]
                continue

            forecaster = MovingAverageForecaster(window=args.window)
            forecaster.fit(ts)

            preds = forecaster.forecast(steps=args.steps)

            forecast_results[:, i, j] = preds

            count += 1
            if count % (total_pixels // 10) == 0:
                print(f"Progress: {count / total_pixels:.0%}")

    print(f"Saving results to {output_path}")
    np.save(output_path, forecast_results)
    print("Done.")

if __name__ == "__main__":
    main()
