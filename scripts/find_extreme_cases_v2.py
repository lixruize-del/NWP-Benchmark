import xarray as xr
import numpy as np
import pandas as pd
from scipy.ndimage import label, sum_labels, find_objects
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from src.common.repo_paths import nwp_outputs_dir  # noqa: E402

_HW = nwp_outputs_dir() / "era5_monthly_202506_v2/metrics/heatwave_object_v2/ifs"
mask_file = _HW / "aifs/lead_day_1/step2/hot_masks_p90.nc"
gt_file = _HW / "_shared/gt_tmax_daily_2025_20250701_20251231.nc"

print("Loading datasets...")
ds_mask = xr.open_dataset(mask_file)
ds_gt = xr.open_dataset(gt_file)

# We want events that are:
# 1. Identified as a heatwave object (mask == 1)
# 2. Temperature > 35 Celsius
extreme_mask = (ds_mask['gt_mask_p90'].values == 1) & (ds_gt['tmax_gt_c'].values > 35.0)

lats = ds_gt['latitude'].values
lons = ds_gt['longitude'].values
times = ds_gt['time'].values

print("Performing 3D labeling on (Mask=1 AND Tmax > 35)...")
structure = np.ones((3, 3, 3), dtype=int)
labeled_mask, num_features = label(extreme_mask, structure=structure)

print(f"Found {num_features} extreme candidates. Calculating Severity...")

# Cosine weights for area
lat_weights = np.cos(np.deg2rad(lats))
weights_3d = np.broadcast_to(lat_weights[np.newaxis, :, np.newaxis], extreme_mask.shape)

# Severity = weighted sum of (mask * weights)
event_sizes = sum_labels(weights_3d, labeled_mask, index=np.arange(1, num_features + 1))

# Get top 20 events to filter manually for geographic diversity
top_n = 20
top_indices = np.argsort(event_sizes)[::-1][:top_n]
slices = find_objects(labeled_mask)

print("\n" + "="*85)
print(f"🏆 Top {top_n} Extreme Heatwave Events (Tmax > 35 & P90) in 2025 H2")
print("="*85)

for i, idx in enumerate(top_indices):
    real_label = idx + 1
    sl = slices[idx]
    time_slice, lat_slice, lon_slice = sl
    
    start_date = pd.to_datetime(times[time_slice.start]).strftime('%Y-%m-%d')
    end_date = pd.to_datetime(times[time_slice.stop - 1]).strftime('%Y-%m-%d')
    
    lat_vals_slice = lats[lat_slice]
    lon_vals_slice = lons[lon_slice]
    lat_range = (lat_vals_slice.min(), lat_vals_slice.max())
    lon_range = (lon_vals_slice.min(), lon_vals_slice.max())
    
    # Peak determination
    event_3d = (labeled_mask[sl] == real_label)
    daily_stats = []
    for t_idx in range(time_slice.stop - time_slice.start):
        day_mask = event_3d[t_idx]
        if day_mask.any():
            day_tmax = ds_gt['tmax_gt_c'].values[time_slice.start + t_idx, lat_slice, lon_slice]
            # Average Tmax in the extreme area for that day
            mean_peak_t = day_tmax[day_mask].mean()
            # Max Tmax in the area
            max_peak_t = day_tmax[day_mask].max()
            area_weight = (day_mask * lat_weights[lat_slice, np.newaxis]).sum()
            daily_stats.append((t_idx, area_weight, mean_peak_t, max_peak_t))
    
    # Sort by area * mean_t to find best peak
    daily_stats.sort(key=lambda x: x[1], reverse=True)
    best_rel_t_idx, best_area, best_mean_t, best_max_t = daily_stats[0]
    peak_date = pd.to_datetime(times[time_slice.start + best_rel_t_idx]).strftime('%Y-%m-%d')
    
    severity = event_sizes[idx]
    
    print(f"Rank {i+1:2d} | Sev: {severity:8.0f} | {start_date} to {end_date} | Peak: {peak_date}")
    print(f"        Box: Lat[{lat_range[0]:6.1f}, {lat_range[1]:6.1f}] Lon[{lon_range[0]:6.1f}, {lon_range[1]:6.1f}]")
    print(f"        Peak Stats: AreaWt={best_area:6.1f}, MeanT={best_mean_t:4.1f}C, MaxT={best_max_t:4.1f}C")
    print("-" * 85)

ds_mask.close()
ds_gt.close()
