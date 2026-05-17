"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import logging
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, minimum_filter
import xarray as xr

from src.common.saver import select_pressure_hpa

__all__ = ["Tracker"]

logger = logging.getLogger(__file__)
DEBUG_LOG_PATH = Path(__file__).resolve().parent / ".cursor" / "debug-3a66ca.log"
DEBUG_SESSION_ID = "3a66ca"


def _agent_debug_log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": DEBUG_SESSION_ID,
        "runId": f"run_{int(time.time())}",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


class NoEyeException(Exception):
    """Raised when no eye can be found."""


def get_box(
    variable: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
):
    """Get a square box for a variable."""
    # Make latitude selection.

    lat_mask = (lat_min <= lats) & (lats <= lat_max)
    box = variable[lat_mask, :]
    lats = lats[lat_mask]

    # Make longitude selection. Be careful when wrapping around.
    lon_min = lon_min % 360
    lon_max = lon_max % 360
    if lon_min <= lon_max:
        lon_mask = (lon_min <= lons) & (lons <= lon_max)
        box = box[:, lon_mask]
        lons = lons[lon_mask]
    else:
        lon_mask1 = lon_min <= lons
        lon_mask2 = lons <= lon_max
        box = np.concatenate((box[:, lon_mask1], box[:, lon_mask2]), axis=-1)
        lons = np.concatenate((lons[lon_mask1], lons[lon_mask2]))
    # import pdb
    # pdb.set_trace()
    return lats, lons, box


def havdist(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance between two latitude-longitude coordinates."""
    lat1, lat2 = np.deg2rad(lat1), np.deg2rad(lat2)
    lon1, lon2 = np.deg2rad(lon1), np.deg2rad(lon2)
    rad_earth_km = 6371
    inner = 1 - np.cos(lat2 - lat1) + np.cos(lat1) * np.cos(lat2) * (1 - np.cos(lon2 - lon1))
    return 2 * rad_earth_km * np.arcsin(np.sqrt(0.5 * inner))


def get_closest_min(
    variable: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    lat: float,
    lon: float,
    delta_lat: float = 5,
    delta_lon: float = 5,
    minimum_cap_size: int = 50,
) -> tuple[float, float]:
    """Get the minimum in `variable` that is closest to `lat` and `lon`."""
    # Create a box centred around the current latitude and longitude.
    lats, lons, box = get_box(
        variable,
        lats,
        lons,
        lat - delta_lat,
        lat + delta_lat,
        lon - delta_lon,
        lon + delta_lon,
    )

    # Smooth to avoid local minima due to noise.
    box = gaussian_filter(box, sigma=1)

    # Find local minima.
    local_minima = minimum_filter(box, size=(minimum_cap_size, minimum_cap_size)) == box
    # Remove minima at the edges: these occur when the tracker fails.
    local_minima[0, :] = 0
    local_minima[-1, :] = 0
    local_minima[:, 0] = 0
    local_minima[:, -1] = 0

    # If no local minima are left, no eye can be found. Try the next one.
    if local_minima.sum() == 0:
        raise NoEyeException()

    # Return the latitude and longitude of the closest local minimum.
    lat_inds, lon_inds = zip(*np.argwhere(local_minima))
    dists = havdist(lats[list(lat_inds)], lons[list(lon_inds)], lat, lon)
    i = np.argmin(dists)

    return lats[lat_inds[i]], lons[lon_inds[i]], dists[i]


def extrapolate(lats: list[float], lons: list[float]) -> tuple[float, float]:
    """Guess an initial latitude and longitude by extrapolating `lats` and `lons`."""
    assert len(lats) == len(lons)
    if len(lats) == 0:
        raise ValueError("Cannot extrapolate from empty lists.")
    elif len(lats) == 1:
        return lats[0], lons[0]
    else:
        # Linearly extrapolate using the last eight points.
        lats = lats[-8:]
        lons = lons[-8:]
        n = len(lats)
        fit = np.polyfit(np.arange(n), np.stack((lats, lons), axis=-1), 1)
        return np.polyval(fit, n)


class Tracker:
    """Simple tropical cyclone tracker.

    This algorithm was originally designed and implemented by Anna Allen. This particular
    implementation is by Wessel Bruinsma and features various improvements over the original design.
    """

    def __init__(
        self,
        init_lat: float,
        init_lon: float,
        init_time: datetime,
        distance_threshold = 560,
        wind_threshold = 8,   #m/s
        resolution = 0.09,    # degree
    ) -> None:
        self.tracked_times: list[datetime] = [init_time]
        self.tracked_lats: list[float] =[init_lat]
        self.tracked_lons: list[float] =[init_lon]
        self.tracked_msls: list[float] = [np.nan]
        self.tracked_winds: list[float] = [np.nan] 
        self.distance_threshold = distance_threshold # km
        self.wind_threshold = wind_threshold #8 m/s
        self.fails: int = 0
        self.end_flag = False
        if resolution == 0.25:
            self.minimum_cap_size = 8
        elif   resolution == 0.09:
            self.minimum_cap_size = 45
    def results(self) -> pd.DataFrame:
        """Assemble the track into a convenient DataFrame."""
        return pd.DataFrame(
            {
                "time": self.tracked_times,
                "lat": self.tracked_lats,
                "lon": self.tracked_lons,
                "msl": self.tracked_msls,
                "wind": self.tracked_winds,
            }
        )

    def step(self, ds: xr.Dataset) -> None:
        """Track the next step.

        Args:
            ds (:class:`xr.Dataset`): Prediction.
        """
        # Extract the relevant variables from the prediction.

        z850 = select_pressure_hpa(ds["z"], 850.0).values.squeeze()
        # region agent log
        _agent_debug_log(
            "H12",
            "tracker.py:Tracker.step",
            "z_level_selection",
            {
                "z_level_hpa": 850.0,
                "z_dims": list(ds["z"].dims),
                "z_shape": list(ds["z"].shape),
            },
        )
        # endregion
        msl =  ds["msl"].values.squeeze()
        u10 = ds["u10"].values.squeeze()
        v10 = ds["v10"].values.squeeze()
        wind = np.sqrt(u10 * u10 + v10 * v10)
        # lsm = batch.static_vars["lsm"].numpy()
        lats = ds.latitude.values
        lons = ds.longitude.values
        time = ds.time.values
        # import pdb
        # pdb.set_trace()
        # Provide an initial guess by extrapolating.
        lat, lon = extrapolate(self.tracked_lats, self.tracked_lons)
        lat = max(min(lat, 90), -90)
        lon = lon % 360

        # def is_clear(lat: float, lon: float, delta: float) -> bool:
        #     """Is a box centred at `lat` and `lon` with "radius" `delta` clear of land?"""
        #     _, _, lsm_box = get_box(
        #         lsm,
        #         lats,
        #         lons,
        #         lat - delta,
        #         lat + delta,
        #         lon - delta,
        #         lon + delta,
        #     )
        #     return lsm_box.max() < 0.5

        # Did we "snap" from the guess to a real nearby minimum?
        snap = False

        # Try MSL with increasingly small boxes.
        for delta in [5, 4, 3, 2, 1.5]:
            try:
                # if is_clear(lat, lon, delta):
                lat, lon, distance = get_closest_min(
                        msl,
                        lats,
                        lons,
                        lat,
                        lon,
                        delta_lat=delta,
                        delta_lon=delta,
                        minimum_cap_size=self.minimum_cap_size 
                    )
                snap = True
                break
            except NoEyeException:
                pass

        if not snap:
            # MSL didn't work. Try Z850. If it works, try to refine with MSL.
            try:
                lat, lon, distance = get_closest_min(
                    z850,
                    lats,
                    lons,
                    lat,
                    lon,
                    delta_lat=5,
                    delta_lon=5,
                    minimum_cap_size=self.minimum_cap_size,
                )
                snap = True
                
                for delta in [5, 4, 3, 2, 1.5]:
                    try:
                        # if is_clear(lat, lon, delta):
                        lat, lon, distance = get_closest_min(
                            msl,
                            lats,
                            lons,
                            lat,
                            lon,
                            delta_lat=delta,
                            delta_lon=delta,
                            minimum_cap_size=self.minimum_cap_size
                        )
                        break
                    except NoEyeException:
                        pass
                
            except NoEyeException:
                pass

        if not snap:
            distance = 100
            self.fails += 1
            if len(self.tracked_lats) > 1:
                logger.info(f"Failed at time {time}. Extrapolating in a silly way.")
            else:
                raise NoEyeException("Completely failed at the first step.")



        # Extract minimum MSL and maximum wind speed from a crop around the TC.
        _, _, msl_crop = get_box(
            msl,
            lats,
            lons,
            lat - 1.5,
            lat + 1.5,
            lon - 1.5,
            lon + 1.5,
        )
        _, _, wind_crop = get_box(
            wind,
            lats,
            lons,
            lat - 1.5,
            lat + 1.5,
            lon - 1.5,
            lon + 1.5,
        )
        if wind_crop.max()>self.wind_threshold and distance<self.distance_threshold:
            self.tracked_times.append(time)
            self.tracked_lats.append(lat)
            self.tracked_lons.append(lon)
            self.tracked_msls.append(msl_crop.min())
            self.tracked_winds.append(wind_crop.max())
        else:
            self.end_flag = True
def plot_typhoon_tracks(trajectories, output_path=None, extent=None):
    """
    Plot TC tracks for multiple members plus the mean track.
    
    Args:
        trajectories (dict): member tracks as {name: [(lat, lon, time), ...]}。
        output_path (str): output image path; if None, show interactively。
        extent (list): map extent [leftlon, rightlon, lowerlat, upperlat]。
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Set map extent
    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Add map features
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    
    # Accumulator for mean track
    mean_trajectory = {}
    
    # Plot each member track
    for member_name, trajectory in trajectories.items():
        if len(trajectory) > 0:  # skip empty tracks
            lats, lons, times = zip(*trajectory)  # unpack track as lats, lons, times
            # if member_name=="M 0":
            ax.plot(lons, lats, marker='o', markersize=3, linestyle='--', label=member_name, alpha=0.6)  # connect track points

            # Add points to mean-track accumulator
            if member_name != 'best':
                for idx, (lat, lon, t) in enumerate(trajectory):
                    t = str(t)
                    if t not in mean_trajectory: # initialize the time 
                        mean_trajectory[t] = {"lat_sum": 0, "lon_sum": 0, "count": 0}
                    mean_trajectory[t]["lat_sum"] += lat  # sum latitude
                    mean_trajectory[t]["lon_sum"] += lon  # sum longitude
                    mean_trajectory[t]["count"] += 1  # member count for this time
    
    # Build mean track
    mean_lats, mean_lons, mean_times = [], [], []
    for t, values in sorted(mean_trajectory.items()):  # sorted by time
        # if values["count"]<10:
        #     break
        mean_lats.append(values["lat_sum"] / values["count"])  # mean latitude
        mean_lons.append(values["lon_sum"] / values["count"])  # mean longitude
        mean_times.append(t)  # time label
    
    # Plot mean track
    if mean_lats and mean_lons:
        ax.plot(mean_lons, mean_lats, color='black', linewidth=1,  marker='^', markersize=4, label='Mean Track')  # bold black mean track

    # Legend (multi-column)
    ax.legend(loc='upper left', fontsize=10, title='Members',framealpha=0.5, ncol=4)  # legend columns
    
    # Title
    ax.set_title("Typhoon Tracks Across Members", fontsize=16)

    # Save or show figure
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

       
if __name__ == '__main__':
    import os
    from datetime import datetime, timedelta
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import pandas as pd
    import xarray as xr
    try:
        from ghr.utils.s3_client import s3_client
    except:
        s3_client = None
        
    timestamp = '2024-11-04T00:00:00'  # start time
    use_s3 = True
    # User-specified initial TC center
    init_lat, init_lon = 12.8, 131.8  # initial lat/lon

    s3 = s3_client('s_ssd')

    # Forecast lead configuration
    lead_times = 10  # forecast length in days (10)
    interval_hours = 6  # 6-hour interval
    num_member = 10
    
    # Store tracks per member
    trajectories = {}
    init_time = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")
    all_forecast_times = [
        (init_time + timedelta(hours=i * interval_hours)).strftime("%Y-%m-%dT%H:%M:%S")
        for i in range(0, lead_times * 24 // interval_hours)
    ]

    # get the ensembel tc forecast
    for member_idx in range(0, 50):  # example: 20 members
        member_name = f"M {member_idx}"
        print(f"Tracking for member {member_name}")
        member_forecasts = [
                f"{timestamp[:4]}/{timestamp}/{forecast_time}_{member_idx}.nc"
                for forecast_time in all_forecast_times
                ]
        tracker = Tracker(init_lat=init_lat, init_lon=init_lon, init_time=init_time, resolution=0.09)
        
        for member_file in member_forecasts:
            if not use_s3: 
                ds = xr.open_dataset(member_file)
            else:
                ds = s3.read_nc_from_BytesIO(member_file,bucket='nwp_predictions/FengWu_GHR/')
            # print(ds)
            tracker.step(ds)
            if tracker.end_flag:
                break
            
        trajectories[member_name] = []
        print(tracker.results())
        
        for index, point in tracker.results().iterrows():
            trajectories[member_name].append((point["lat"], point["lon"], point["time"]))
        # print(tracker.results())
    output_path = "./demos/typhoon_tracks.png"       
    extent = [85, 140, 5, 30]

    
    # User-specified time window
    # ----------------------
    start_time = "2024-11-04 00:00:00"  # start time
    end_time = "2024-11-14 23:59:59"    # end time
    from load_IBTrACS import BestTracker
    from tc_paths import ibtracs_csv_path

    tracker = BestTracker(ibtracs_csv_path())
    best_tc = tracker.get_typhoon_path(
        name="YINXING",
        season=2024,
        start_time=start_time,
        end_time=end_time,
        time_step=6,
    )
    # print(best_tc)


    # Plot all member tracks
    trajectories['best'] = []
    for forecast_time in all_forecast_times:
        row = best_tc[best_tc["ISO_TIME"] == forecast_time]
        if not row.empty:
            trajectories['best'].append((row["LAT"].values, row["LON"].values, forecast_time))



    # output path
    output_path = "./tc_result/FengWu-GHR_typhoon_tracks.png"
    # map extent [leftlon, rightlon, lowerlat, upperlat]
    extent = [85, 140, 5, 30]
    plot_typhoon_tracks(trajectories, output_path=output_path, extent=extent)
