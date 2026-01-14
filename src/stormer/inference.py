import os
import sys
import logging
import argparse
import traceback
import numpy as np
import torch
from pathlib import Path
import types

# --- 路径设置 ---
CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent

# 将项目根目录加入 path，以便导入 src.common
sys.path.append(str(BASE_DIR))

# --- Import Saver ---
from src.common.saver import Saver

# Paths
PROCESSED_DIR = BASE_DIR / "assets" / "data" / "processed_stormer"
WEIGHTS_FILE = BASE_DIR / "assets" / "weights" / "stormer" / "stormer_1.40625_patch_size_2.ckpt"
OUTPUT_DIR = BASE_DIR / "outputs" / "stormer"
LOG_DIR = BASE_DIR / "logs" / "stormer"
NORM_DIR = CURRENT_DIR / "normalization_constants"
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
sys.path.append(str(CURRENT_DIR))

# Logging setup
log_file = LOG_DIR / "inference.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Stormer.Inference")

# ==============================================================================
# Monkey Patching System (Dependency Fix)
# ==============================================================================
class UniversalInstance:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return self
    def __getattr__(self, key): return self
    def __getitem__(self, key): return self
    def __len__(self): return 0
    def __repr__(self): return "UniversalInstance"

class MockModule(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__file__ = f"/tmp/{name}.py"
        self.__path__ = [] 
    def __getattr__(self, key):
        return type(key, (UniversalInstance,), {"__module__": self.__name__})

def register_mock_module(name):
    if name not in sys.modules:
        sys.modules[name] = MockModule(name)
    return sys.modules[name]

def apply_patches():
    logger.info("Applying environment patches for checkpoint loading...")
    try:
        import stormer.utils.lr_scheduler as local_scheduler
        import stormer.utils.metrics as local_metrics
        import stormer.utils.data_utils as local_data_utils
        
        # 1. Inject missing metric classes into local modules
        class DummyMetric(UniversalInstance): pass
        for missing in ["LatWeightedMSE", "LatWeightedACC", "LatWeightedRMSE", "MetricsMetaInfo"]:
            if not hasattr(local_metrics, missing):
                setattr(local_metrics, missing, DummyMetric)
        if not hasattr(local_data_utils, "MetricsMetaInfo"):
            local_data_utils.MetricsMetaInfo = DummyMetric

        # 2. Mock 'climate_learn' structure in sys.modules
        register_mock_module("climate_learn")
        
        # Map models.lr_scheduler to local implementation
        models = register_mock_module("climate_learn.models")
        models.lr_scheduler = local_scheduler
        sys.modules["climate_learn.models.lr_scheduler"] = local_scheduler
        
        # Map metrics to local implementation
        metrics_pkg = register_mock_module("climate_learn.metrics")
        metrics_pkg.metrics = local_metrics
        metrics_pkg.utils = local_metrics 
        sys.modules["climate_learn.metrics.metrics"] = local_metrics
        sys.modules["climate_learn.metrics.utils"] = local_metrics
        
        # Mock transforms (handling denormalize)
        register_mock_module("climate_learn.transforms")
        denorm_mod = register_mock_module("climate_learn.transforms.denormalize")
        class Denormalize(UniversalInstance): pass
        denorm_mod.Denormalize = Denormalize
        
        # Mock data module
        register_mock_module("climate_learn.data")
        
        logger.info("Patches applied successfully.")
    except Exception as e:
        logger.error(f"Patching failed: {e}")
        sys.exit(1)

apply_patches()

try:
    from stormer.models.hub.stormer import Stormer
    from stormer.models.iterative_module import GlobalForecastIterativeModule
except ImportError as e:
    logger.error(f"Failed to import Stormer model: {e}")
    sys.exit(1)


# ==============================================================================
# Variable Definitions & Adapter Logic
# ==============================================================================
VARIABLES = [
    "2m_temperature", "10m_u_component_of_wind", "10m_v_component_of_wind", "mean_sea_level_pressure",
    "geopotential_50", "geopotential_100", "geopotential_150", "geopotential_200", "geopotential_250", 
    "geopotential_300", "geopotential_400", "geopotential_500", "geopotential_600", "geopotential_700", 
    "geopotential_850", "geopotential_925", "geopotential_1000",
    "u_component_of_wind_50", "u_component_of_wind_100", "u_component_of_wind_150", "u_component_of_wind_200", 
    "u_component_of_wind_250", "u_component_of_wind_300", "u_component_of_wind_400", "u_component_of_wind_500", 
    "u_component_of_wind_600", "u_component_of_wind_700", "u_component_of_wind_850", "u_component_of_wind_925", 
    "u_component_of_wind_1000",
    "v_component_of_wind_50", "v_component_of_wind_100", "v_component_of_wind_150", "v_component_of_wind_200", 
    "v_component_of_wind_250", "v_component_of_wind_300", "v_component_of_wind_400", "v_component_of_wind_500", 
    "v_component_of_wind_600", "v_component_of_wind_700", "v_component_of_wind_850", "v_component_of_wind_925", 
    "v_component_of_wind_1000",
    "temperature_50", "temperature_100", "temperature_150", "temperature_200", "temperature_250", 
    "temperature_300", "temperature_400", "temperature_500", "temperature_600", "temperature_700", 
    "temperature_850", "temperature_925", "temperature_1000",
    "specific_humidity_50", "specific_humidity_100", "specific_humidity_150", "specific_humidity_200", 
    "specific_humidity_250", "specific_humidity_300", "specific_humidity_400", "specific_humidity_500", 
    "specific_humidity_600", "specific_humidity_700", "specific_humidity_850", "specific_humidity_925", 
    "specific_humidity_1000",
]

def get_standard_var_names(stormer_vars):
    """
    Adapter: Maps Stormer's long variable names to NWPBench standard short names.
    e.g., 'geopotential_500' -> 'z_500'
    """
    mapping = {
        "2m_temperature": "t2m",
        "10m_u_component_of_wind": "u10",
        "10m_v_component_of_wind": "v10",
        "mean_sea_level_pressure": "msl"
    }
    
    std_vars = []
    for v in stormer_vars:
        if v in mapping:
            std_vars.append(mapping[v])
        elif v.startswith("geopotential_"):
            std_vars.append(v.replace("geopotential_", "z_"))
        elif v.startswith("temperature_"):
            std_vars.append(v.replace("temperature_", "t_"))
        elif v.startswith("u_component_of_wind_"):
            std_vars.append(v.replace("u_component_of_wind_", "u_"))
        elif v.startswith("v_component_of_wind_"):
            std_vars.append(v.replace("v_component_of_wind_", "v_"))
        elif v.startswith("specific_humidity_"):
            std_vars.append(v.replace("specific_humidity_", "q_"))
        else:
            logger.warning(f"Unknown variable format: {v}, keeping as is.")
            std_vars.append(v)
    return std_vars

def get_target_date():
    if DATE_FILE.exists():
        with open(DATE_FILE) as f:
            return f.read().strip()
    return "2025120412"

def load_normalization_stats(device, dtype):
    mean_file = NORM_DIR / "normalize_mean.npz"
    std_file = NORM_DIR / "normalize_std.npz"
    
    if not mean_file.exists():
        mean_file = CURRENT_DIR / "normalization_constants" / "normalize_mean.npz"
        std_file = CURRENT_DIR / "normalization_constants" / "normalize_std.npz"

    logger.info(f"Loading normalization stats from: {mean_file.parent}")
    mean_dict = dict(np.load(mean_file))
    std_dict = dict(np.load(std_file))
    
    mean_list = [mean_dict[var] for var in VARIABLES]
    std_list = [std_dict[var] for var in VARIABLES]
    
    mean_arr = np.concatenate(mean_list, axis=0).astype(np.float32)
    std_arr = np.concatenate(std_list, axis=0).astype(np.float32)
    
    mean_gpu = torch.from_numpy(mean_arr).to(device, dtype=dtype).view(1, -1, 1, 1)
    std_gpu = torch.from_numpy(std_arr).to(device, dtype=dtype).view(1, -1, 1, 1)
    
    return mean_gpu, std_gpu

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None)
    args = parser.parse_args()
    
    target_date = args.date if args.date else get_target_date()
    
    # Device Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    logger.info(f"Target Date: {target_date}")
    logger.info(f"Device: {device} | Precision: {dtype}")

    # 1. Load Processed Data
    input_file = PROCESSED_DIR / f"input_{target_date}.npy"
    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        sys.exit(1)

    logger.info("Loading processed input tensor...")
    inp_np = np.load(input_file)
    inp_tensor = torch.from_numpy(inp_np).unsqueeze(0).to(device, dtype=dtype) # [1, C, H, W]

    # 2. Load Stats
    mean_gpu, std_gpu = load_normalization_stats(device, dtype)

    # 3. Load Model
    logger.info(f"Loading model from {WEIGHTS_FILE.name}...")
    if not WEIGHTS_FILE.exists():
        logger.error("Weights missing.")
        sys.exit(1)

    net = Stormer(in_img_size=[128, 256], variables=VARIABLES, patch_size=2, hidden_size=1024, depth=24, num_heads=16, mlp_ratio=4)
    
    try:
        checkpoint = torch.load(WEIGHTS_FILE, map_location='cpu')
        new_sd = {k.replace("net.", ""): v for k, v in checkpoint['state_dict'].items()}
        net.load_state_dict(new_sd, strict=False)
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        sys.exit(1)

    model = GlobalForecastIterativeModule(net).to(device, dtype=dtype).eval()

    # 4. Inference
    logger.info("Running inference (T+6h)...")
    try:
        # Normalize
        inp_norm = (inp_tensor - mean_gpu) / std_gpu
        
        # Prepare Interval Tensor (6 hours)
        interval_val = 6
        interval_tensor = torch.tensor([interval_val]).to(device, dtype=dtype) / 10.0
        interval_tensor = interval_tensor.repeat(inp_tensor.shape[0])
        
        with torch.no_grad():
            pred_norm = model.net(inp_norm, VARIABLES, interval_tensor)
        
        # Denormalize
        pred = pred_norm * std_gpu + mean_gpu
        
        # --- NEW: Save using Standard Saver ---
        # 4.1 Convert to numpy and standard format
        pred_np = pred.float().cpu().numpy() # [1, C, H, W]
        std_vars = get_standard_var_names(VARIABLES)
        
        # 4.2 Initialize Saver
        saver = Saver(save_root=str(OUTPUT_DIR))
        
        # 4.3 Generate Grid for Stormer (1.40625 deg resolution)
        # 128 Lat, 256 Lon
        lat_res, lon_res = 128, 256
        lats = np.linspace(90, -90, lat_res)
        lons = np.linspace(0, 360, lon_res, endpoint=False)
        
        # 4.4 Save
        saver.save(
            data=pred_np,
            channel_mapping=std_vars,
            init_time_str=target_date,
            lead_time_hours=interval_val,
            lat_values=lats,
            lon_values=lons
        )
        
        logger.info(f"Success! Prediction saved via Saver to {OUTPUT_DIR}")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()