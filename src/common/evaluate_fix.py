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
        
        # Add renaming for the time dimension
        if 'valid_time' in ds.dims:
            # We need to rename both the dimension and the coordinate variable
            rename_map['valid_time'] = 'time'
            
        if rename_map:
            ds = ds.rename(rename_map)
            
        if 'longitude' in ds.coords and ds.longitude.values.min() < 0:
            logger.info(f"在 {os.path.basename(nc_path)} 中检测到经度范围 [{ds.longitude.values.min()}, {ds.longitude.values.max()}]。正在转换为 [0, 360]。")
            # 使用 assign_coords 来计算新的坐标值，此时数据本身尚未修改
            ds = ds.assign_coords(longitude=(((ds.longitude + 360) % 360)))
            # 使用 sortby 根据新的经度坐标对数据进行重新排序
            ds = ds.sortby('longitude')
            
        return ds

    def evaluate_single_sample(self, 
                               pred_path: str, 
                               gt_path: str, 
                               var_name: str, 
                               level: Optional[int] = None) -> Dict[str, float]:
        logger.info(f"Evaluating {var_name} (Level: {level})")
        
        # 1. 加载
        ds_pred = self._load_and_align(pred_path)
        ds_gt = self._load_and_align(gt_path, is_gt=True)
        
        # 2. 变量选择
        gt_var_map = {'t2m': '2t', 'z': 'z', 'u': 'u', 'v': 'v', 'msl': 'msl'}
        gt_var_name = gt_var_map.get(var_name, var_name)
        
        if var_name not in ds_pred: return {}
        if gt_var_name not in ds_gt: return {}
            
        da_pred = ds_pred[var_name]
        da_gt = ds_gt[gt_var_name]

        # 3. 层级选择
        if level is not None:
            if 'level' in da_pred.coords: da_pred = da_pred.sel(level=level)
            if 'level' in da_gt.coords: da_gt = da_gt.sel(level=level)

        # ======================================================================
        # 🕵️‍♂️ 审计环节：在时间切片之前打印元数据
        # ======================================================================
        logger.info(f"--- [AUDIT] Pre-Slicing Metadata ---")
        logger.info(f"Pred Time Coords: {da_pred.coords.get('time', 'No Time').values}")
        logger.info(f"GT   Time Coords: {da_gt.coords.get('time', 'No Time').values}")
        
        # 4. 时间切片 (原逻辑)
        if 'time' in da_pred.dims: 
            da_pred = da_pred.isel(time=0)
            logger.info("Pred: Selected index 0")
            
        if 'time' in da_gt.dims: 
            # ！！！重点怀疑对象！！！
            # 我们打印出这里到底选了哪个时间
            selected_gt = da_gt.isel(time=0)
            logger.info(f"GT  : Selected index 0 -> Time Value: {selected_gt.time.values}")
            da_gt = selected_gt

        # ======================================================================
        # 🕵️‍♂️ 审计环节：在对齐和计算之前
        # ======================================================================
        logger.info(f"--- [AUDIT] Alignment Check ---")
        
        # A. 检查时间是否匹配
        try:
            t_p = pd.to_datetime(da_pred.time.values)
            t_g = pd.to_datetime(da_gt.time.values)
            if t_p != t_g:
                logger.error(f"🚨 TIME MISMATCH DETECTED! Pred({t_p}) vs GT({t_g})")
                logger.error(f"   You are comparing two different times! This explains the high RMSE.")
            else:
                logger.info(f"✅ Time aligned: {t_p}")
        except Exception:
            logger.warning("Could not verify time alignment (missing coords?)")

        # B. 检查坐标范围
        logger.info(f"Pred Lat: {da_pred.latitude.values[0]:.4f} ... {da_pred.latitude.values[-1]:.4f}")
        logger.info(f"GT   Lat: {da_gt.latitude.values[0]:.4f} ... {da_gt.latitude.values[-1]:.4f}")
        
        # C. 检查对齐操作
        if da_pred.shape != da_gt.shape:
            logger.info(f"Grid mismatch detected ({da_pred.shape} vs {da_gt.shape}). Interpolating...")
            da_pred = da_pred.interp_like(da_gt, method='linear')
        
        # D. 检查数值统计 (防止填0或单位错误)
        p_mean = float(da_pred.mean())
        g_mean = float(da_gt.mean())
        logger.info(f"Pred Mean Value: {p_mean:.2f}")
        logger.info(f"GT   Mean Value: {g_mean:.2f}")
        if abs(p_mean - g_mean) > 1000:
             logger.error(f"🚨 HUGE VALUE DIFF! Check units (e.g. m vs m^2/s^2).")

        # ======================================================================

        # 5. 计算
        weights = MetricCalculator.compute_latitude_weights(da_gt.latitude)
        wrmse_val = MetricCalculator.wrmse(da_pred, da_gt, weights)
        
        logger.info(f"-> WRMSE: {wrmse_val:.4f}")
        return {"WRMSE": wrmse_val}

    def compare_models_visual(self, model_paths: Dict[str, str], gt_path: str, var_name: str, level: Optional[int] = None, save_name: str = "comparison.png"):
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
            da_gt = ds_gt[gt_var].sel(level=level).isel(time=-1)
        else:
            da_gt = ds_gt[gt_var].isel(time=-1)
            
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
"""
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
"""

if __name__ == "__main__":
    # --- 检验 NeuralGCM 输出 ---
    
    # 1. 定义时间和路径
    init_time = "2020082212"
    lead_time_hours = 6
    valid_time_str = "2020082218" # 预报生效时间 (12 + 6 = 18)

    # 模型输出文件路径
    neuralgcm_output_path = f"outputs/neuralgcm/{init_time}/2020-0822-06.nc"
    
    # 将 NeuralGCM 的输出放入一个字典中，以便 compare_models_visual 函数使用
    models_to_compare = {
        "NeuralGCM": neuralgcm_output_path
    }
    
    # 真实数据 (Ground Truth) 路径
    gt_sfc_path = f"assets/data/era5_neuralgcm/surface_{valid_time_str}.nc"
    gt_upper_path = f"assets/data/era5_neuralgcm/upper_{valid_time_str}.nc"

    # 2. 初始化评估器
    # 将结果保存在一个专门的文件夹中
    evaluator = Evaluator(output_dir=f"outputs/evaluation_results/neuralgcm_{init_time}_{lead_time_hours}h")

    # --- 3. 检验关键变量 ---

    # 首先，检查所有需要的文件是否存在
    required_files_exist = all([
        os.path.exists(neuralgcm_output_path),
        os.path.exists(gt_sfc_path),
        os.path.exists(gt_upper_path)
    ])

    if not required_files_exist:
        logger.error("一个或多个必需的输入文件（模型输出或GT）不存在。请检查路径。")
        # 打印出具体哪个文件缺失，方便调试
        print(f"NeuralGCM Output Exists: {os.path.exists(neuralgcm_output_path)} -> {neuralgcm_output_path}")
        print(f"GT Surface Exists: {os.path.exists(gt_sfc_path)} -> {gt_sfc_path}")
        print(f"GT Upper Exists: {os.path.exists(gt_upper_path)} -> {gt_upper_path}")

    else:
        logger.info("所有文件均已找到，开始进行评估...")
        
        # 检验 t850 (850hPa 温度) 来代替 t2m
        logger.info("--- 评估 850hPa 温度 (t850) ---")
        evaluator.compare_models_visual(
            model_paths=models_to_compare,
            gt_path=gt_upper_path,  # 温度在高空文件中
            var_name="t",          # 模型输出的变量名叫 't'
            level=850,             # 指定高度层
            save_name="comparison_t850.png"
        )
        
        # 检验 z500 (500hPa 位势高度)
        logger.info("--- 评估 500hPa 位势高度 (z500) ---")
        evaluator.compare_models_visual(
            model_paths=models_to_compare,
            gt_path=gt_upper_path,
            var_name="z",
            level=500,
            save_name="comparison_z500.png"
        )

        # 检验 u850 (850hPa U风)
        logger.info("--- 评估 850hPa U风 (u850) ---")
        evaluator.compare_models_visual(
            model_paths=models_to_compare,
            gt_path=gt_upper_path,
            var_name="u",
            level=850,
            save_name="comparison_u850.png"
        )