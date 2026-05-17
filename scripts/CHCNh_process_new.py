"""
GHCNh hourly station preprocessing (ported from StationCast `dataset/CHCNh_process_new.py`).

Reads ISD-style `.psv` yearly files, optional disk cache, merges stations per 6-hourly
timestamp, QC against ERA5 single-level NPY, and writes NetCDF under ``<root>/processed/<year>/``.

**Paths:** override defaults with environment variables (see below) or edit this file.

- ``STATIONCAST_HOME`` — **required** unless ``STATIONCAST_ROOT`` is set; StationCast dataset root.
- ``STATIONCAST_ROOT`` — dataset root (default ``$STATIONCAST_HOME/dataset``).
- ``STATIONCAST_ISD_BY_YEAR`` — ISD ``by-year`` tree (default ``$STATIONCAST_ROOT/ISD_raw/by-year``).
- ``STATIONCAST_CACHE`` — joblib cache dir (default ``$STATIONCAST_HOME/cache_new``).
- ``ERA5_NPY_ROOT`` — ERA5 np.25 root for QC fields (default ``/ecmwf-era5-datasets/era5_np.25``).
"""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from joblib import dump, load
from tqdm import tqdm

# Path layout (override with env for your site; defaults match original StationCast layout).
_station_home = os.environ.get("STATIONCAST_HOME", "").strip()
root = os.environ.get(
    "STATIONCAST_ROOT",
    os.path.join(_station_home, "dataset") if _station_home else "",
).strip()
folder_path = os.environ.get(
    "STATIONCAST_ISD_BY_YEAR",
    os.path.join(root, "ISD_raw", "by-year"),
)
cache_dir = os.environ.get(
    "STATIONCAST_CACHE",
    os.path.join(_station_home, "cache_new"),
)
ERA5_NPY_ROOT = os.environ.get("ERA5_NPY_ROOT", "/ecmwf-era5-datasets/era5_np.25")

var_names = [
    "temperature",
    "dew_point_temperature",
    "station_level_pressure",
    "sea_level_pressure",
    "wind_direction",
    "wind_speed",
    "wind_gust",
    "relative_humidity",
    "precipitation_3_hour",
    "precipitation_6_hour",
    "precipitation_9_hour",
    "precipitation_12_hour",
    "precipitation_24_hour",
]

short_name = {
    "temperature": "t2m",
    "dew_point_temperature": "d2m",
    "station_level_pressure": "sp",
    "sea_level_pressure": "msl",
    "wind_direction": "u10",
    "wind_speed": "v10",
    "wind_gust": "v10",
    "relative_humidity": "t2m",
    "precipitation_3_hour": "tp3h",
    "precipitation_6_hour": "tp6h",
    "precipitation_9_hour": "tp6h",
    "precipitation_12_hour": "tp6h",
    "precipitation_24_hour": "tp6h",
}

qc_vars = [
    "temperature",
    "dew_point_temperature",
    "station_level_pressure",
    "sea_level_pressure",
    "wind_speed",
]


def clean_year_cache(cache_dir_local: str, year_to_delete: int) -> None:
    for file_name in os.listdir(cache_dir_local):
        if file_name.startswith(f"{year_to_delete}_"):
            file_path = os.path.join(cache_dir_local, file_name)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                    print(f"Deleted cache file: {file_path}")
                except OSError as e:
                    print(f"Error deleting cache file {file_path}: {e}")


def load_file_with_disk_cache(file_path: str, year: int) -> pd.DataFrame | None:
    global current_year

    if current_year is None:
        current_year = year
    elif current_year != year:
        print(
            f"Year changed from {current_year} to {year}. Clearing cache for year {current_year}."
        )
        clean_year_cache(cache_dir, current_year)
        current_year = year

    file_name = os.path.basename(file_path)
    cache_file = os.path.join(cache_dir, f"{year}_{file_name}.joblib")

    if os.path.exists(cache_file):
        try:
            return load(cache_file)
        except Exception as e:
            tqdm.write(f"Error loading cache file {cache_file}: {e}")
            os.remove(cache_file)

    try:
        df = pd.read_csv(
            file_path,
            sep="|",
            low_memory=False,
            usecols=["DATE", "STATION", "ELEVATION", "LONGITUDE", "LATITUDE"] + var_names,
        )
        dump(df, cache_file)
        return df
    except Exception as e:
        tqdm.write(f"Error reading file {file_path}: {e}")
        return None


def process_file(file_path: str, hourly: datetime, start_time: datetime, end_time: datetime, year: int):
    try:
        df = load_file_with_disk_cache(file_path, year)
        if df is None:
            return None
        columns = df.columns.tolist()
    except Exception as e:
        tqdm.write(f"Error reading file {file_path}: {e}")
        return None

    if "DATE" in columns:
        try:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        except Exception as e:
            tqdm.write(f"Error processing DATE column in {file_path}: {e}")
            return None

    df = df.dropna(subset=["DATE"])
    result = df[(df["DATE"] >= start_time) & (df["DATE"] <= end_time)]
    if result.empty:
        return None

    averaged_data = {var: pd.to_numeric(result[var], errors="coerce").mean() for var in var_names}

    try:
        station_id = result["STATION"].iloc[0]
        time_ = result["DATE"].iloc[0]
        altitude = pd.to_numeric(result["ELEVATION"].iloc[0], errors="coerce")
        longitude = pd.to_numeric(result["LONGITUDE"].iloc[0], errors="coerce")
        latitude = pd.to_numeric(result["LATITUDE"].iloc[0], errors="coerce")
    except Exception as e:
        tqdm.write(f"Error extracting metadata in {file_path}: {e}")
        return None

    station_ds = xr.Dataset(
        data_vars={var: (["station"], [averaged_data[var]]) for var in var_names},
        coords={
            "station": [station_id],
            "time": time_,
            "longitude": longitude,
            "latitude": latitude,
            "altitude": altitude,
        },
    )
    return station_ds


def process_hourly(hourly: datetime, max_workers: int) -> None:
    station_datasets = []
    year = hourly.year
    file_list = glob.glob(f"{folder_path}/{year}/*.psv")
    start_time = hourly - timedelta(minutes=15)
    end_time = hourly + timedelta(minutes=15)
    if max_workers == 1:
        for i in range(len(file_list)):
            station_ds = process_file(file_list[i], hourly, start_time, end_time, year)
            if station_ds is not None:
                station_datasets.append(station_ds)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_file, fp, hourly, start_time, end_time, year)
                for fp in file_list
            ]
            for future in tqdm(
                futures, desc=f"Processing stations for {hourly}", leave=False, unit="station"
            ):
                station_ds = future.result()
                if station_ds is not None:
                    station_datasets.append(station_ds)

    if station_datasets:
        combined_ds = xr.concat(station_datasets, dim="station")
        altitude_data = combined_ds["altitude"].data
        altitude_data = np.where(altitude_data == "******", np.nan, altitude_data)
        combined_ds["altitude"].data = altitude_data

        wind_direction = combined_ds["wind_direction"].data
        wind_direction = np.where(wind_direction == 999.0, np.nan, wind_direction)
        combined_ds["wind_direction"].data = wind_direction

        unified_time = hourly
        combined_ds = combined_ds.assign_coords({"unified_time": unified_time})

        os.makedirs(f"{root}/processed/{year}", exist_ok=True)
        output_file = f"{root}/processed/{year}/{hourly}.nc"

        date, time = str(hourly).split(" ")
        ds_ERA5 = xr.Dataset(
            coords={
                "latitude": lat,
                "longitude": lon,
            }
        )
        for var in qc_vars:
            short_abb = short_name[var]
            npy_path = os.path.join(
                ERA5_NPY_ROOT, "single", str(year), date, f"{time}-{short_abb}.npy"
            )
            dense_data = np.load(npy_path)
            if short_abb == "d2m" or short_abb == "t2m":
                dense_data = dense_data - 273.15
                units = "Celsius"
            elif short_abb == "msl" or short_abb == "sp":
                dense_data = dense_data / 100
                units = "hPa"
            elif short_abb == "u10" or short_abb == "v10":
                units = "m/s"
                u10_data = np.load(
                    os.path.join(
                        ERA5_NPY_ROOT, "single", str(year), date, f"{time}-u10.npy"
                    )
                )
                v10_data = np.load(
                    os.path.join(
                        ERA5_NPY_ROOT, "single", str(year), date, f"{time}-v10.npy"
                    )
                )
                dense_data = np.sqrt(u10_data**2 + v10_data**2)
            elif "precipitation" in var:
                units = "mm"
            else:
                units = "unknown"

            ds_ERA5[var] = (["latitude", "longitude"], dense_data)
            ds_ERA5[var].attrs["units"] = units

        stations_lat = combined_ds["latitude"]
        stations_lon = combined_ds["longitude"]
        stations_lon_adjusted = stations_lon % 360
        ds = ds_ERA5.interp(
            latitude=stations_lat,
            longitude=stations_lon_adjusted,
            method="linear",
        )

        for var_name, ratio, _real_ratio in zip(
            [
                "temperature",
                "dew_point_temperature",
                "station_level_pressure",
                "sea_level_pressure",
                "wind_speed",
            ],
            [6, 6, 9000, 9000, 7],
            [10, 10, 10000, 10000, 10],
        ):
            station_temperature = combined_ds[var_name].data
            era5_temperature = ds[var_name].data
            problematic_stations = (station_temperature / era5_temperature) > ratio
            station_temperature_corrected = np.where(
                ~problematic_stations, station_temperature, era5_temperature
            )
            combined_ds[var_name].data = station_temperature_corrected
            uncorrected = int(problematic_stations.sum())
            print(f"Number of uncorrected stations for {var_name}: {uncorrected}")

        combined_ds.to_netcdf(output_file)
        tqdm.write(f"Saved combined dataset for {hourly} to {output_file}")
    else:
        tqdm.write(f"No data found for {hourly} across all stations.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process GHCNh ISD hourly station data with configurable time range."
    )
    parser.add_argument(
        "--start_time",
        type=str,
        default="2020-01-01 00:00:00",
        help="Start time (YYYY-MM-DD HH:MM:SS).",
    )
    parser.add_argument(
        "--end_time",
        type=str,
        default="2024-12-31 23:00:00",
        help="End time (YYYY-MM-DD HH:MM:SS).",
    )
    return parser.parse_args()


os.makedirs(cache_dir, exist_ok=True)
current_year = None

lat = np.linspace(90, -90, 721)
lon = np.linspace(0, 360, 1440)

if __name__ == "__main__":
    args = parse_args()
    start_time = args.start_time
    end_time = args.end_time
    print(f"Processing data from {start_time} to {end_time}")
    hourly_indexs = pd.date_range(start=start_time, end=end_time, freq="6h")
    for hourly in tqdm(hourly_indexs, desc="Processing 6-hourly timestamps"):
        max_workers = 8
        process_hourly(hourly, max_workers)
