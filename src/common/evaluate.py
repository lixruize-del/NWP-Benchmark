import os
import logging
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from typing import Dict, List, Optional, Union, Tuple

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("NWPBench.Evaluator")

class MetricCalculator:
    """
    Static utility class for calculating meteorological metrics.
    """

    @staticmethod
    def compute_latitude_weights(latitudes: xr.DataArray) -> xr.DataArray:
        """
        Compute cosine latitude weights for global grids.
        
        Args:
            latitudes (xr.DataArray): 1D array of latitude values (degrees).
            
        Returns:
            xr.DataArray: Weights proportional to cosine(lat).
        """
        weights = np.cos(np.deg2rad(latitudes))
        # Normalize weights
        weights = weights / weights.mean()
        return weights

    @staticmethod
    def wrmse(pred: xr.DataArray, gt: xr.DataArray, weights: xr.DataArray = None) -> float:
        """
        Compute Weighted Root Mean Square Error (WRMSE).
        
        Args:
            pred (xr.DataArray): Prediction data.
            gt (xr.DataArray): Ground Truth data.
            weights (xr.DataArray): Latitude weights. If None, computes unweighted RMSE.
            
        Returns:
            float: Scalar WRMSE value.
        """
        diff_sq = (pred - gt) ** 2
        
        if weights is not None:
            # Broadcast weights to match dimensions (e.g., [Lat] -> [Lat, Lon])
            weights, diff_sq = xr.broadcast(weights, diff_sq)
            # Weighted average
            mse = (diff_sq * weights).mean()
        else:
            mse = diff_sq.mean()
            
        return np.sqrt(mse).item()

class Evaluator:
    def __init__(self, output_dir: str = "./evaluation_results"):
        """
        Initialize the Evaluator.

        Args:
            output_dir (str): Directory to save plots and metrics logs.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_and_align(self, nc_path: str, is_gt: bool = False) -> xr.Dataset:
        """
        Load NetCDF file and standardize dimension names for alignment.
        
        Args:
            nc_path (str): Path to .nc file.
            is_gt (bool): Flag indicating if this is Ground Truth (might need renaming).
            
        Returns:
            xr.Dataset: Loaded dataset with standard coords (latitude, longitude, level).
        """
        if not os.path.exists(nc_path):
            raise FileNotFoundError(f"File not found: {nc_path}")
            
        ds = xr.open_dataset(nc_path)
        
        # Standardize dimension names (Handle different conventions)
        # Target: latitude, longitude, level
        rename_map = {}
        if 'lat' in ds.coords: rename_map['lat'] = 'latitude'
        if 'lon' in ds.coords: rename_map['lon'] = 'longitude'
        if 'isobaricInhPa' in ds.coords: rename_map['isobaricInhPa'] = 'level'
        if 'pressure_level' in ds.coords: rename_map['pressure_level'] = 'level'
        
        if rename_map:
            ds = ds.rename(rename_map)
            
        return ds

    def evaluate_single_sample(self, 
                               pred_path: str, 
                               gt_path: str, 
                               var_name: str, 
                               level: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate metrics for a single prediction file against GT.
        
        Args:
            pred_path: Path to model prediction (.nc).
            gt_path: Path to ERA5 Ground Truth (.nc).
            var_name: Variable name to evaluate (e.g., 't2m', 'z').
            level: Pressure level (int) if applicable.
            
        Returns:
            Dict: Dictionary containing computed metrics (e.g., {'WRMSE': 1.23}).
        """
        logger.info(f"Evaluating {var_name} (Level: {level})")
        
        ds_pred = self._load_and_align(pred_path)
        ds_gt = self._load_and_align(gt_path, is_gt=True)
        
        # Select Variable
        try:
            da_pred = ds_pred[var_name]
            # Handle ERA5 naming mapping (Standard -> ERA5)
            # You might need a more robust mapping here depending on your GT files
            gt_var_map = {'t2m': '2t', 'z': 'z', 'u': 'u', 'v': 'v', 'msl': 'msl'}
            gt_var_name = gt_var_map.get(var_name, var_name)
            
            # Check if variable exists in GT
            if gt_var_name not in ds_gt:
                # Try finding by standard_name or fallback
                logger.warning(f"GT variable {gt_var_name} not found. Available: {list(ds_gt.data_vars)}")
                return {}
                
            da_gt = ds_gt[gt_var_name]

            # Select Level if applicable
            if level is not None:
                if 'level' in da_pred.coords:
                    da_pred = da_pred.sel(level=level)
                if 'level' in da_gt.coords:
                    da_gt = da_gt.sel(level=level)

            # --- ALIGNMENT & REGRIDDING ---
            # Automatically interp Pred to GT grid if different (handles Stormer vs ERA5)
            if da_pred.shape != da_gt.shape:
                logger.info(f"Grid mismatch detected. Interpolating Pred {da_pred.shape} to GT {da_gt.shape}...")
                da_pred = da_pred.interp_like(da_gt, method='linear')

            # Select first time step if multiple exist
            if 'time' in da_pred.dims: da_pred = da_pred.isel(time=0)
            if 'time' in da_gt.dims: da_gt = da_gt.isel(time=0)

            # Compute Weights
            weights = MetricCalculator.compute_latitude_weights(da_gt.latitude)
            
            # Compute Metric
            wrmse_val = MetricCalculator.wrmse(da_pred, da_gt, weights)
            logger.info(f"-> WRMSE: {wrmse_val:.4f}")
            
            return {"WRMSE": wrmse_val}

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {}

    def compare_models_visual(self, 
                              model_paths: Dict[str, str], 
                              gt_path: str, 
                              var_name: str, 
                              level: Optional[int] = None,
                              save_name: str = "comparison.png"):
        """
        Visualize comparison of multiple models against GT for a specific variable.
        
        Args:
            model_paths: Dict mapping model names to their .nc file paths.
                         e.g., {'Pangu': 'path/to/pangu.nc', 'Aurora': 'path/to/aurora.nc'}
            gt_path: Path to Ground Truth .nc file.
            var_name: Variable to visualize (e.g., 't2m').
            level: Pressure level (optional).
            save_name: Filename for the output plot.
        """
        logger.info(f"Generating comparison plot for {var_name}...")
        
        # Load GT
        ds_gt = self._load_and_align(gt_path, is_gt=True)
        # Mapping logic (simplified)
        gt_var_map = {'t2m': '2t', 'z': 'z', 'u': 'u', 'v': 'v', 'msl': 'msl'}
        gt_var = gt_var_map.get(var_name, var_name)
        
        if level is not None:
            da_gt = ds_gt[gt_var].sel(level=level).isel(time=0)
        else:
            da_gt = ds_gt[gt_var].isel(time=0)
            
        # Prepare Plot
        num_models = len(model_paths)
        cols = num_models + 1 # 1 col for GT, rest for models
        fig, axes = plt.subplots(2, cols, figsize=(5 * cols, 8), 
                                 subplot_kw={'projection': ccrs.PlateCarree()})
        
        # --- Row 1: Fields ---
        # Plot GT
        ax_gt = axes[0, 0]
        ax_gt.set_title("ERA5 Ground Truth")
        im_gt = ax_gt.pcolormesh(da_gt.longitude, da_gt.latitude, da_gt, 
                                 transform=ccrs.PlateCarree(), cmap='coolwarm')
        plt.colorbar(im_gt, ax=ax_gt, orientation='horizontal', pad=0.05)
        ax_gt.coastlines()
        
        # Global Min/Max for unified colorbar
        vmin, vmax = da_gt.min(), da_gt.max()

        # Plot Models
        preds = {}
        for idx, (model_name, path) in enumerate(model_paths.items(), start=1):
            ds_m = self._load_and_align(path)
            if level is not None:
                da_m = ds_m[var_name].sel(level=level).isel(time=0)
            else:
                da_m = ds_m[var_name].isel(time=0)
            
            # Align grid
            da_m = da_m.interp_like(da_gt, method='linear')
            preds[model_name] = da_m
            
            # Plot
            ax = axes[0, idx]
            ax.set_title(f"{model_name}")
            im = ax.pcolormesh(da_m.longitude, da_m.latitude, da_m, 
                               transform=ccrs.PlateCarree(), cmap='coolwarm', vmin=vmin, vmax=vmax)
            plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
            ax.coastlines()

        # --- Row 2: Errors (Bias) ---
        # Blank for GT column (or put 0)
        axes[1, 0].axis('off')
        
        # Compute Errors
        max_err = 0
        diffs = {}
        for model_name, da_pred in preds.items():
            diff = da_pred - da_gt
            diffs[model_name] = diff
            current_max = abs(diff).max()
            if current_max > max_err: max_err = current_max
            
        # Plot Errors
        for idx, (model_name, diff) in enumerate(diffs.items(), start=1):
            ax = axes[1, idx]
            ax.set_title(f"Diff: {model_name} - GT")
            
            # Calculate metrics for title
            weights = MetricCalculator.compute_latitude_weights(da_gt.latitude)
            wrmse = MetricCalculator.wrmse(preds[model_name], da_gt, weights)
            
            ax.set_title(f"Diff: {model_name}\nWRMSE: {wrmse:.3f}")
            
            im = ax.pcolormesh(diff.longitude, diff.latitude, diff, 
                               transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-max_err, vmax=max_err)
            plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.05)
            ax.coastlines()

        full_save_path = os.path.join(self.output_dir, save_name)
        plt.tight_layout()
        plt.savefig(full_save_path, dpi=150)
        logger.info(f"Comparison plot saved to: {full_save_path}")
        plt.close()

if __name__ == "__main__":
    # --- Example Usage ---
    
    # 1. Define Paths
    # Assume we have results from Pangu and Aurora for the same Init Time
    # Note: These paths are hypothetical based on your folder structure
    init_time = "2025120412"
    lead_time = "06"
    
    models_to_compare = {
        "Pangu": f"outputs/pangu/{init_time}/2025-1204-{lead_time}.nc",
        "Aurora": f"outputs/aurora/{init_time}/2025-1204-{lead_time}.nc",
        "Stormer": f"outputs/stormer/{init_time}/2025-1204-{lead_time}.nc" 
    }
    
    # GT Path (T+6h -> 18:00)
    gt_sfc = "assets/data/era5_aurora/gt_surface_2025120418.nc"
    gt_upper = "assets/data/era5_aurora/gt_upper_2025120418.nc"

    # 2. Initialize Evaluator
    evaluator = Evaluator(output_dir=f"outputs/benchmark_results/{init_time}_{lead_time}h")

    # 3. Run Comparison for T2M
    # Verify files exist before running to avoid crash in example
    valid_models = {k: v for k, v in models_to_compare.items() if os.path.exists(v)}
    
    if valid_models and os.path.exists(gt_sfc):
        evaluator.compare_models_visual(
            model_paths=valid_models,
            gt_path=gt_sfc,
            var_name="t2m",
            level=None,
            save_name="compare_t2m.png"
        )
    else:
        logger.warning("Skipping T2M comparison: missing model output or GT files.")

    # 4. Run Comparison for Z500
    if valid_models and os.path.exists(gt_upper):
        evaluator.compare_models_visual(
            model_paths=valid_models,
            gt_path=gt_upper,
            var_name="z",
            level=500,
            save_name="compare_z500.png"
        )