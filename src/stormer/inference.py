import sys
import types
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
import torch
from torchvision.transforms import transforms

CURRENT_DIR = Path(__file__).resolve().parent
BASE_DIR = CURRENT_DIR.parent.parent
sys.path.append(str(BASE_DIR))

PROCESSED_DIR = BASE_DIR / "assets" / "data" / "processed_stormer"
WEIGHTS_FILE = BASE_DIR / "assets" / "weights" / "stormer" / "stormer_1.40625_patch_size_2.ckpt"
OUTPUT_DIR = BASE_DIR / "outputs" / "stormer"
NORM_DIR = CURRENT_DIR / "normalization_constants"
DATE_FILE = BASE_DIR / "assets" / "target_date.txt"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("Stormer.InferenceTry")


class UniversalInstance:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, key):
        return self

    def __getitem__(self, key):
        return self

    def __len__(self):
        return 0


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

    import src.stormer.utils.lr_scheduler as local_scheduler
    import src.stormer.utils.metrics as local_metrics
    import src.stormer.utils.data_utils as local_data_utils

    class DummyMetric(UniversalInstance):
        pass

    for missing in ["LatWeightedMSE", "LatWeightedACC", "LatWeightedRMSE", "MetricsMetaInfo"]:
        if not hasattr(local_metrics, missing):
            setattr(local_metrics, missing, DummyMetric)
    if not hasattr(local_data_utils, "MetricsMetaInfo"):
        local_data_utils.MetricsMetaInfo = DummyMetric

    register_mock_module("climate_learn")
    models = register_mock_module("climate_learn.models")
    models.lr_scheduler = local_scheduler
    sys.modules["climate_learn.models.lr_scheduler"] = local_scheduler

    metrics_pkg = register_mock_module("climate_learn.metrics")
    metrics_pkg.metrics = local_metrics
    metrics_pkg.utils = local_metrics
    sys.modules["climate_learn.metrics.metrics"] = local_metrics
    sys.modules["climate_learn.metrics.utils"] = local_metrics

    register_mock_module("climate_learn.transforms")
    denorm_mod = register_mock_module("climate_learn.transforms.denormalize")

    class Denormalize(UniversalInstance):
        pass

    denorm_mod.Denormalize = Denormalize
    register_mock_module("climate_learn.data")

    class StormerModuleFinder:
        def find_module(self, fullname, path=None):
            if fullname.startswith("stormer.") and fullname not in sys.modules:
                return self
            return None

        def load_module(self, fullname):
            new_name = "src." + fullname
            if new_name in sys.modules:
                module = sys.modules[new_name]
            else:
                import importlib
                module = importlib.import_module(new_name)
            sys.modules[fullname] = module
            return module

    sys.meta_path.insert(0, StormerModuleFinder())
    logger.info("Patches applied successfully.")


apply_patches()

from src.common.saver import Saver
from src.stormer.data.iterative_dataset import ERA5MultiLeadtimeDataset
from src.stormer.models.iterative_module import GlobalForecastIterativeModule
from src.stormer.models.hub.stormer import Stormer

PRESSURE_LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
variables = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    *[f"geopotential_{l}" for l in PRESSURE_LEVELS],
    *[f"u_component_of_wind_{l}" for l in PRESSURE_LEVELS],
    *[f"v_component_of_wind_{l}" for l in PRESSURE_LEVELS],
    *[f"temperature_{l}" for l in PRESSURE_LEVELS],
    *[f"specific_humidity_{l}" for l in PRESSURE_LEVELS],
]

SFC_TO_SHORT = {
    "2m_temperature": "t2m",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "mean_sea_level_pressure": "msl",
}
PL_BASE_TO_SHORT = {
    "geopotential": "z",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity": "q",
}


def get_target_date() -> str:
    return DATE_FILE.read_text().strip() if DATE_FILE.exists() else "2023010112"


def date_to_file_idx(date_str: str) -> Tuple[int, int]:
    dt = pd.to_datetime(date_str, format="%Y%m%d%H")
    year = dt.year
    start_of_year = pd.Timestamp(f"{year}-01-01 00:00:00")
    hours_since_start = (dt - start_of_year).total_seconds() / 3600
    file_idx = int(hours_since_start // 6)
    return year, file_idx


def build_channel_mapping(stormer_variables: List[str]) -> List[str]:
    mapping = []
    for v in stormer_variables:
        if v in SFC_TO_SHORT:
            mapping.append(SFC_TO_SHORT[v])
            continue
        base, level_str = v.rsplit("_", 1)
        if level_str.isdigit() and base in PL_BASE_TO_SHORT:
            mapping.append(f"{PL_BASE_TO_SHORT[base]}_{int(level_str)}")
        else:
            mapping.append(v)
    return mapping


def read_lat_lon_from_any_h5(root: Path) -> Tuple[np.ndarray, np.ndarray]:
    h5_files = sorted(root.glob("*.h5"))
    if not h5_files:
        return None, None
    with h5py.File(h5_files[0], "r") as f:
        lat = f["input/lat"][:] if "input/lat" in f else None
        lon = f["input/lon"][:] if "input/lon" in f else None
    return lat, lon


def ensure_batch_bvhw(t: torch.Tensor) -> torch.Tensor:
    """ERA5MultiLeadtimeDataset returns (V,H,W); some pipelines may already use (1,V,H,W). Stormer needs (B,V,H,W)."""
    if t.dim() == 3:
        return t.unsqueeze(0)
    if t.dim() == 4 and t.shape[0] == 1:
        return t
    raise ValueError(f"Expected input (V,H,W) or (1,V,H,W), got shape {tuple(t.shape)}")


def load_model(device: torch.device) -> GlobalForecastIterativeModule:
    net = Stormer(
        in_img_size=[128, 256],
        variables=variables,
        patch_size=2,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
    )
    return GlobalForecastIterativeModule(net, pretrained_path=str(WEIGHTS_FILE)).to(device).eval()


def load_transforms_and_stats(device: torch.device):
    mean_npz = dict(np.load(NORM_DIR / "normalize_mean.npz"))
    std_npz = dict(np.load(NORM_DIR / "normalize_std.npz"))

    normalize_mean = np.concatenate([mean_npz[v] for v in variables], axis=0).astype(np.float32)
    normalize_std = np.concatenate([std_npz[v] for v in variables], axis=0).astype(np.float32)

    inp_transform = transforms.Normalize(normalize_mean, normalize_std)

    out_transforms = {}
    for interval in [6, 12, 24]:
        diff_std_npz = dict(np.load(NORM_DIR / f"normalize_diff_std_{interval}.npz"))
        diff_std = np.concatenate([diff_std_npz[v] for v in variables], axis=0)
        out_transforms[interval] = transforms.Normalize(np.zeros_like(diff_std), diff_std)

    mean_t = torch.from_numpy(normalize_mean).to(device=device, dtype=torch.float32).view(1, -1, 1, 1)
    std_t = torch.from_numpy(normalize_std).to(device=device, dtype=torch.float32).view(1, -1, 1, 1)

    return inp_transform, out_transforms, mean_t, std_t


def run_inference_and_save(date_str: str, lead_times: List[int], list_intervals: List[int] = [6, 12, 24]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    inp_transform, out_transforms, mean_t, std_t = load_transforms_and_stats(device)
    model.set_transforms(inp_transform, out_transforms)

    dataset = ERA5MultiLeadtimeDataset(
        root_dir=str(PROCESSED_DIR),
        variables=variables,
        transform=inp_transform,
        list_lead_times=lead_times,
        data_freq=6,
    )
    if len(dataset) == 0:
        logger.error("Dataset is empty! Check if enough files exist for the lead times.")
        return

    year, file_idx = date_to_file_idx(date_str)
    target_filename = f"{year}_{file_idx:04d}.h5"
    found_idx = None
    for idx, path in enumerate(dataset.inp_file_paths):
        if target_filename in str(path):
            found_idx = idx
            break

    if found_idx is None:
        logger.error("Could not find %s in dataset input files", target_filename)
        return

    inp_data, out_data_dict, _ = dataset[found_idx]
    inp_data = ensure_batch_bvhw(inp_data).to(device, dtype=torch.float32)

    prediction_dict: Dict[int, torch.Tensor] = {}
    for lead_time in sorted(out_data_dict.keys()):
        all_preds = []
        for interval in list_intervals:
            if lead_time % interval == 0:
                steps = lead_time // interval
                with torch.no_grad():
                    pred = model.forward_validation(inp_data, variables, interval, steps)
                all_preds.append(pred)

        if not all_preds:
            logger.warning("Skip lead=%sh: no available interval", lead_time)
            continue

        mean_pred = torch.stack(all_preds, dim=0).mean(0)
        # explicit denormalization safeguard
        denorm_pred = mean_pred * std_t + mean_t
        prediction_dict[lead_time] = denorm_pred

    saver = Saver(save_root=str(OUTPUT_DIR))
    channel_mapping = build_channel_mapping(variables)
    lat_values, lon_values = read_lat_lon_from_any_h5(PROCESSED_DIR)

    for lead_time, pred in prediction_dict.items():
        saver.save(
            data=pred.squeeze(0).detach().cpu().numpy(),
            channel_mapping=channel_mapping,
            init_time_str=date_str,
            lead_time_hours=int(lead_time),
            lat_values=lat_values,
            lon_values=lon_values,
        )
        logger.info("Saved Stormer forecast with denormalization: init=%s lead=%sh", date_str, lead_time)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", type=str, default=None)
    p.add_argument("--lead_times", type=int, nargs="+", default=[6])
    args = p.parse_args()
    run_inference_and_save(args.date or get_target_date(), args.lead_times)


if __name__ == "__main__":
    main()
