import os
import re
import datetime as dt
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
from pyhdf.SD import SD, SDC


FILE_NAME_PATTERN = re.compile(
    r"^MOD44B\.A(?P<year>\d{4})(?P<doy>\d{3})\.h(?P<h>\d{2})v(?P<v>\d{2})\.(?P<collection>\d{3})\.(?P<proc>\d{13})\.hdf$"
)


class MoD44BLoader:
    """
    Loads a directory of MOD44B HDF files, parses metadata from filenames,
    and constructs a time series array for a given SDS (e.g., 'Percent_Tree_Cover').

    Usage:
        loader = MoD44BLoader(data_dir="/path/to/dir", sds_name="Percent_Tree_Cover")
        metadata = loader.get_metadata()
        series = loader.get_time_series()
    """

    def __init__(self, data_dir: str, sds_name: str, resolution: int = -1) -> None:
        self.data_dir: Path = Path(data_dir)
        self.sds_name: str = sds_name

        if not self.data_dir.exists() or not self.data_dir.is_dir():
            raise ValueError(f"Data directory not found or not a directory: {self.data_dir}")

        # Internal state
        self._metadata: List[Dict] = [] #Sorted by acquisition date
        self._file_paths: List[str] = [] #Sorted by acquisition date
        self._time_series: Optional[np.ndarray] = None
        self._dimensions: Optional[Tuple[int, int]] = None  # (height, width)
        self.resolution: int = resolution

        # Build state on construction
        self._scan_and_parse()
        self._build_time_series()

    # ----------------------------
    # Public getters
    # ----------------------------
    def get_metadata(self) -> List[Dict]:
        return self._metadata

    def get_pixel_time_series(self, i: int, j: int) -> np.ndarray:
        """
        Gets the 1-D time series for a specific pixel.
        It returns a copy of the data so any changes made to the returned data
        will not be reflected in the internal state of the class.
        
        Args:
            i: The row index (height).
            j: The column index (width).
            
        Returns:
            A 1-D numpy array of shape (time_steps,).
            
        Raises:
            RuntimeError: If the time series data has not been loaded.
            IndexError: If the (i, j) coordinates are out of bounds.
        """

        if self._time_series is None or self._dimensions is None:
            raise RuntimeError("Time series data has not been initialized.")
            
        height, width = self._dimensions
        
        # Validate bounds for i (row/height)
        if not (0 <= i < height):
            raise IndexError(
                f"Row index i ({i}) is out of bounds for height {height}"
            )
            
        # Validate bounds for j (column/width)
        if not (0 <= j < width):
            raise IndexError(
                f"Column index j ({j}) is out of bounds for width {width}"
            )
            
        # Select all time steps (:) for the specific pixel (i, j)
        return self._time_series[:, i, j].copy().astype(np.int16)

    def get_time_series(self) -> np.ndarray:
        """
        Gets a copy of the full (time, height, width) data cube.
        
        Returns a copy to prevent accidental modification of the 
        loader's internal state.
        """
        if self._time_series is None:
            raise RuntimeError("Time series has not been initialized.")
            
        return self._time_series.copy().astype(np.int16)

    def get_file_paths(self) -> List[str]:
        return self._file_paths

    def get_sds_name(self) -> str:
        return self.sds_name

    def get_dimensions(self) -> Tuple[int, int]:
        assert self._dimensions is not None, "Dimensions are not available yet"
        return self._dimensions

    def get_time_datetimes(self) -> List[dt.datetime]:
        """
        Return a list of acquisition datetimes (sorted) for each time slice.
        Useful as the x-axis for time series plots.
        """
        return [m["acquisition_dt"] for m in self._metadata]

    def get_time_years(self) -> List[int]:
        """
        Return a list of acquisition years (sorted, one per time slice).
        """
        return [m["acquisition_dt"].year for m in self._metadata]

    def get_num_time_steps(self) -> int:
        """Return the number of time steps in the loaded cube."""
        return len(self._file_paths)

    def get_frame(self, t: int) -> np.ndarray:
        """
        Return a copy of the 2D slice at time index t.
        """
        if self._time_series is None:
            raise RuntimeError("Time series has not been initialized.")
        if not (0 <= t < self._time_series.shape[0]):
            raise IndexError(f"Time index t ({t}) is out of bounds for {self._time_series.shape[0]} steps")
        return self._time_series[t].copy().astype(np.int16)

    def get_change_between(self, t_early: int, t_late: int) -> np.ndarray:
        """
        Compute a change map (late - early) between two time indices.
        """
        early = self.get_frame(t_early).astype(np.float32)
        late = self.get_frame(t_late).astype(np.float32)
        return (late - early).astype(np.float32)

    def get_frame_stats(self, t: int) -> Dict[str, float]:
        """
        Compute basic statistics for the frame at time t, ignoring NaNs.
        Returns keys: mean, std, min, max.
        """
        frame = self.get_frame(t).astype(np.float32)
        valid = frame[~np.isnan(frame)] if np.issubdtype(frame.dtype, np.floating) else frame
        # If integer typed without NaNs, treat all values as valid
        if isinstance(valid, np.ndarray):
            values = valid
        else:
            values = frame
        if values.size == 0:
            return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    # ----------------------------
    # Internal helpers
    # ----------------------------
    def _scan_and_parse(self) -> None:
        """
        Scan the directory for files matching the expected pattern,
        parse metadata, and sort by acquisition date.
        """
        candidates: List[Path] = [p for p in self.data_dir.iterdir() if p.is_file() and FILE_NAME_PATTERN.match(p.name)]

        if len(candidates) == 0:
            raise ValueError(f"No MOD44B files found in {self.data_dir}")

        parsed: List[Dict] = []
        for path in candidates:
            meta = self._parse_filename(path.name)
            meta["path"] = str(path)
            parsed.append(meta)

        # Sort by acquisition datetime
        parsed.sort(key=lambda m: m["acquisition_dt"]) 

        self._metadata = parsed
        self._file_paths = [m["path"] for m in parsed]

        first_h, first_v = parsed[0]["h"], parsed[0]["v"]
        for meta in parsed[1:]:
            if meta["h"] != first_h or meta["v"] != first_v:
                raise ValueError(
                    f"Directory contains mixed tiles. "
                    f"Found file for h={meta['h']}, v={meta['v']} "
                    f"but expected h={first_h}, v={first_v} (from {parsed[0]['filename']})."
                )

    def _parse_filename(self, fname: str) -> Dict:
        m = FILE_NAME_PATTERN.match(fname)
        if not m:
            raise RuntimeError(f"Filename {fname} should match the expected pattern")

        year = int(m.group("year"))
        doy = int(m.group("doy"))
        h = int(m.group("h"))
        v = int(m.group("v"))
        collection = int(m.group("collection"))
        proc = m.group("proc")

        # Acquisition datetime from AYYYYDDD
        base = dt.date(year, 1, 1)
        acquisition_dt = dt.datetime.combine(base, dt.time()) + dt.timedelta(days=doy - 1)

        # Processing datetime from 13 digits: YYYY DDD HH MM SS
        p_year = int(proc[0:4])
        p_doy = int(proc[4:7])
        p_h = int(proc[7:9])
        p_m = int(proc[9:11])
        p_s = int(proc[11:13])
        p_base = dt.date(p_year, 1, 1)
        processing_dt = dt.datetime.combine(p_base, dt.time()) + dt.timedelta(
            days=p_doy - 1, hours=p_h, minutes=p_m, seconds=p_s
        )

        return {
            "filename": fname,
            "h": h,
            "v": v,
            "collection": collection,
            "acquisition_dt": acquisition_dt,
            "processing_dt": processing_dt,
        }

    def _load_data(self, file_path: str) -> np.ndarray:
        hdf_file = SD(file_path, SDC.READ)
        try:
            datasets = hdf_file.datasets().keys()
            if self.sds_name not in datasets:
                raise KeyError(f"Dataset {self.sds_name} not found in {file_path}. Available: {list(datasets)}")
            sds_obj = hdf_file.select(self.sds_name)
            data = sds_obj.get()
        finally:
            hdf_file.end()
        
        # Apply reduction if flag is set
        if self.resolution != -1:
            data = self._reduce_frame(data, self.resolution)
        return data
    
    def _reduce_frame(self, data: np.ndarray, target_dim: int) -> np.ndarray:
        """
        Reduces an image (H, W) to (target_dim, target_dim) by averaging blocks.
        Assumes input dimensions are divisible by target_dim (e.g. 4800 / 100 = 48).
        """
        h, w = data.shape
        
        # Calculate block size (e.g., 48 if going from 4800 -> 100)
        if h % target_dim != 0 or w % target_dim != 0:
            raise ValueError(
                f"Cannot reduce resolution: Input shape ({h}, {w}) "
                f"is not divisible by target dimension ({target_dim})."
            )
            
        block_h = h // target_dim
        block_w = w // target_dim

        # 1. Reshape to (Target_H, Block_H, Target_W, Block_W)
        # 2. Take the mean over the block axes (1 and 3)
        # 3. Cast back to original dtype (optional: remove astype if you want floats)
        return (
            data.reshape(target_dim, block_h, target_dim, block_w)
            .mean(axis=(1, 3))
            .astype(data.dtype)
        )

    def _build_time_series(self) -> None:
        # Load first to determine dimensions
        first_data = self._load_data(self._file_paths[0])

        height, width = first_data.shape
        self._dimensions = (height, width)

        time_steps = len(self._file_paths)
        time_series = np.empty((time_steps, height, width), dtype=first_data.dtype)

        time_series[0] = first_data
        for idx, file_path in enumerate(self._file_paths[1:], start=1):
            data = self._load_data(file_path)
            if data.shape != (height, width):
                raise ValueError(
                    f"Shape mismatch in {file_path}: expected {(height, width)}, got {data.shape}"
                )
            time_series[idx] = data

        self._time_series = time_series


