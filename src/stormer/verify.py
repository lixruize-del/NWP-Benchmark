import os
import sys
import logging
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from dataclasses import dataclass
from typing import Tuple, Optional

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Stormer.Verify")

@dataclass
class VerifyConfig:
    """Configuration for verification paths and parameters."""
    # Paths
    pred_path: str = "outputs/stormer/pred_2025120412_06h.npy"
    gt_sfc_path: str = "assets/data/era5_aurora/gt_surface_2025120418.nc"
    gt_upper_path: str = "assets/data/era5_aurora/gt_upper_2025120418.nc"
    output_dir: str = "outputs/stormer/verification"
    
    # Stormer Grid Definitions (1.40625 deg resolution)
    lat_res: int = 128
    lon_res: int = 256
    
    # Variable Indices in Stormer Output (based on VARIABLES list)
    idx_t2m: int = 0
    idx_z500: int = 11

class StormerVerifier:
    def __init__(self, config: VerifyConfig):
        """
        Initialize the verifier with configuration.
        
        Args:
            config (VerifyConfig): Configuration object containing paths and grid settings.
        """
        self.cfg = config
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        
        # Define target grid (Stormer resolution)
        self.target_lats = np.linspace(90, -90, self.cfg.lat_res)
        self.target_lons = np.linspace(0, 360, self.cfg.lon_res, endpoint=False)

    def load_prediction(self) -> np.ndarray:
        """
        Load the Stormer prediction .npy file.
        
        Returns:
            np.ndarray: Squeezed prediction tensor [Channels, Lat, Lon].
        """
        if not os.path.exists(self.cfg.pred_path):
            logger.error(f"Prediction file not found: {self.cfg.pred_path}")
            sys.exit(1)
            
        try:
            # Load and squeeze (B, C, H, W) -> (C, H, W)
            data = np.load(self.cfg.pred_path).squeeze()
            logger.info(f"Loaded prediction data. Shape: {data.shape}")
            return data
        except Exception as e:
            logger.error(f"Failed to load prediction: {e}")
            sys.exit(1)

    def load_and_regrid_gt(self, file_path: str, var_name: str, level: Optional[int] = None) -> np.ndarray:
        """
        Load ERA5 Ground Truth and regrid it to Stormer resolution.
        
        Args:
            file_path (str): Path to the ERA5 NetCDF file.
            var_name (str): Variable name in the NetCDF file (e.g., 't2m', 'z').
            level (int, optional): Pressure level to select (for upper air variables).

        Returns:
            np.ndarray: Regridded Ground Truth data [Lat, Lon].
        """
        if not os.path.exists(file_path):
            logger.error(f"Ground Truth file not found: {file_path}")
            sys.exit(1)

        try:
            ds = xr.open_dataset(file_path)
            
            # Handle variable name aliases (e.g., '2t' vs 't2m')
            if var_name not in ds:
                # Common ERA5 mappings
                aliases = {'t2m': '2t', 'u10': '10u', 'v10': '10v', 'z': 'geopotential'}
                var_name = aliases.get(var_name, var_name)
            
            da = ds[var_name]

            # Select pressure level if specified
            if level is not None:
                da = da.sel(level=level)

            # Perform linear interpolation to target grid
            logger.info(f"Regridding {var_name} from {da.shape} to ({self.cfg.lat_res}, {self.cfg.lon_res})...")
            da_interp = da.interp(
                latitude=self.target_lats,
                longitude=self.target_lons,
                method='linear'
            )
            
            return da_interp.values.squeeze()
            
        except Exception as e:
            logger.error(f"Failed to process Ground Truth {file_path}: {e}")
            sys.exit(1)

    def compute_rmse(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """Compute Root Mean Square Error."""
        return np.sqrt(np.mean((pred - gt) ** 2))

    def plot_comparison(self, gt: np.ndarray, pred: np.ndarray, 
                       title: str, filename: str, unit: str):
        """
        Generate and save a comparison plot (GT, Prediction, Difference).
        
        Args:
            gt (np.ndarray): Ground Truth data.
            pred (np.ndarray): Prediction data.
            title (str): Title for the plot.
            filename (str): Output filename.
            unit (str): Unit string for the colorbar.
        """
        fig, axes = plt.subplots(1, 3, figsize=(20, 5), subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Determine color scale limits based on data range
        vmin = min(np.min(gt), np.min(pred))
        vmax = max(np.max(gt), np.max(pred))
        
        # Plot 1: Ground Truth
        ax = axes[0]
        ax.set_title(f"ERA5 Ground Truth (Regridded)\n{title}")
        im0 = ax.pcolormesh(self.target_lons, self.target_lats, gt, 
                            transform=ccrs.PlateCarree(), cmap='jet', vmin=vmin, vmax=vmax)
        plt.colorbar(im0, ax=ax, orientation='horizontal', label=unit)
        ax.coastlines()
        
        # Plot 2: Prediction
        ax = axes[1]
        ax.set_title(f"Stormer Prediction\n{title}")
        im1 = ax.pcolormesh(self.target_lons, self.target_lats, pred, 
                            transform=ccrs.PlateCarree(), cmap='jet', vmin=vmin, vmax=vmax)
        plt.colorbar(im1, ax=ax, orientation='horizontal', label=unit)
        ax.coastlines()
        
        # Plot 3: Difference
        ax = axes[2]
        diff = pred - gt
        rmse = np.sqrt(np.mean(diff**2))
        limit = max(abs(np.min(diff)), abs(np.max(diff)))
        
        ax.set_title(f"Difference (Pred - GT)\nRMSE: {rmse:.4f} {unit}")
        im2 = ax.pcolormesh(self.target_lons, self.target_lats, diff, 
                            transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=-limit, vmax=limit)
        plt.colorbar(im2, ax=ax, orientation='horizontal', label=f"Diff ({unit})")
        ax.coastlines()
        
        save_full_path = os.path.join(self.cfg.output_dir, filename)
        plt.savefig(save_full_path, bbox_inches='tight', dpi=150)
        logger.info(f"Saved plot to: {save_full_path}")
        plt.close()

    def run(self):
        """Execute the full verification workflow."""
        logger.info("Starting Stormer verification workflow...")
        
        # 1. Load Prediction
        pred_all = self.load_prediction()
        pred_t2m = pred_all[self.cfg.idx_t2m]
        pred_z500 = pred_all[self.cfg.idx_z500]
        
        # 2. Load and Regrid Ground Truth
        # Note: ERA5 variable names might be 't2m' or '2t', handled inside the method
        gt_t2m = self.load_and_regrid_gt(self.cfg.gt_sfc_path, 't2m')
        gt_z500 = self.load_and_regrid_gt(self.cfg.gt_upper_path, 'z', level=500)
        
        # 3. Validation & Metrics
        logger.info("-" * 40)
        logger.info("METRICS REPORT")
        logger.info("-" * 40)
        
        # -- T2M Verification --
        rmse_t2m = self.compute_rmse(pred_t2m, gt_t2m)
        logger.info(f"T2M (2m Temperature):")
        logger.info(f"  Pred Mean: {np.mean(pred_t2m):.2f} K")
        logger.info(f"  GT Mean:   {np.mean(gt_t2m):.2f} K")
        logger.info(f"  RMSE:      {rmse_t2m:.4f} K")
        
        if rmse_t2m > 5.0:
            logger.warning("HIGH RMSE for T2M! Check normalization or channel indexing.")

        # -- Z500 Verification --
        rmse_z500 = self.compute_rmse(pred_z500, gt_z500)
        logger.info(f"Z500 (Geopotential):")
        logger.info(f"  Pred Mean: {np.mean(pred_z500):.2f}")
        logger.info(f"  GT Mean:   {np.mean(gt_z500):.2f}")
        logger.info(f"  RMSE:      {rmse_z500:.2f} m^2/s^2")
        
        # Unit sanity check for Geopotential
        if abs(np.mean(pred_z500) - 5000) < 1000:
             logger.warning("Predicted Z500 seems to be Geopotential Height (m), but ERA5 is Geopotential (m^2/s^2).")

        # 4. Visualization
        self.plot_comparison(gt_t2m, pred_t2m, "2m Temperature", "verify_t2m.png", "K")
        self.plot_comparison(gt_z500, pred_z500, "500hPa Geopotential", "verify_z500.png", "m^2/s^2")

if __name__ == "__main__":
    # Initialize configuration and run verifier
    config = VerifyConfig()
    verifier = StormerVerifier(config)
    verifier.run()