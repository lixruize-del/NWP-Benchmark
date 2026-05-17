import os
import logging
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from typing import List, Optional, Union, Dict, Tuple

# Conditionally import torch to handle both JAX and PyTorch environments
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Configure logging for the module
logger = logging.getLogger("Saver")

# Configuration dictionary defining variable metadata
# This acts as the single source of truth for variable names, units, and identifiers
def pressure_level_dim_name(short_name: str) -> str:
    """Dimension name for one GRIB short (z/t/u/v) so merged Datasets do not align-join levels."""
    return f"plev_{short_name}"


def select_pressure_hpa(da: xr.DataArray, hpa: float) -> xr.DataArray:
    """
    Select one pressure level from a field saved by :class:`Saver`.

    Supports legacy files using ``isobaricInhPa`` and new files using ``plev_<short>``.
    """
    target = float(hpa)
    for d in da.dims:
        if d == "isobaricInhPa" or d.startswith("plev_"):
            vals = np.asarray(da.coords[d].values, dtype=np.float64)
            if vals.size == 0:
                continue
            idxs = np.flatnonzero(np.isclose(vals, target))
            if idxs.size:
                return da.isel({d: int(idxs[0])})
    raise KeyError(f"No pressure level coordinate for {hpa} hPa in {da.dims}")


def has_pressure_level(da: xr.DataArray, hpa: float) -> bool:
    target = float(hpa)
    for d in da.dims:
        if d == "isobaricInhPa" or d.startswith("plev_"):
            vals = np.asarray(da.coords[d].values, dtype=np.float64)
            if vals.size and np.any(np.isclose(vals, target)):
                return True
    return False


grib_para = {
    "z":    {"Name": "Geopotential",                  "ShortName": "z",    "Unit": "m^2 s^-2", "ParaID": 129},
    "t":    {"Name": "Temperature",                   "ShortName": "t",    "Unit": "K",        "ParaID": 130},
    "u":    {"Name": "U component of wind",           "ShortName": "u",    "Unit": "m s^-1",   "ParaID": 131},
    "v":    {"Name": "V component of wind",           "ShortName": "v",    "Unit": "m s^-1",   "ParaID": 132},
    "q":    {"Name": "Specific humidity",             "ShortName": "q",    "Unit": "kg kg^-1", "ParaID": 133},
    "r":    {"Name": "Relative humidity",             "ShortName": "r",    "Unit": "%",        "ParaID": 157},

    # GRIB2 parameter IDs for these are 244 (ciwc) and 245 (clwc) in neuralgcm
    "ciwc": {"Name": "Specific cloud ice water content",  "ShortName": "ciwc", "Unit": "kg kg^-1", "ParaID": 244},
    "clwc": {"Name": "Specific cloud liquid water content", "ShortName": "clwc", "Unit": "kg kg^-1", "ParaID": 245},
    
    "w":    {"Name": "Vertical velocity",             "ShortName": "w",    "Unit": "Pa s^-1",  "ParaID": 135},
    "u10":  {"Name": "10m U component of wind",       "ShortName": "u10",  "Unit": "m s^-1",   "ParaID": 165},
    "v10":  {"Name": "10m V component of wind",       "ShortName": "v10",  "Unit": "m s^-1",   "ParaID": 166},
    "u100": {"Name": "100m U component of wind",      "ShortName": "u100", "Unit": "m s^-1",   "ParaID": 246},
    "v100": {"Name": "100m V component of wind",      "ShortName": "v100", "Unit": "m s^-1",   "ParaID": 247},
    "tp6h": {"Name": "Total 6H Precipitation",        "ShortName": "tp6h", "Unit": "mm",       "ParaID": 228},
    "tp":   {"Name": "Total 1H Precipitation",        "ShortName": "tp",   "Unit": "mm",       "ParaID": 228},
    "t2m":  {"Name": "2 meter temperature",           "ShortName": "t2m",  "Unit": "K",        "ParaID": 167},
    "msl":  {"Name": "mean sea level pressure",       "ShortName": "msl",  "Unit": "Pa",       "ParaID": 151},
    "tcc":  {"Name": "Total cloud cover",             "ShortName": "tcc",  "Unit": "0-1",      "ParaID": 164},
    "sp":   {"Name": "Surface pressure",              "ShortName": "sp",   "Unit": "Pa",       "ParaID": 134},
    "ssr":  {"Name": "Surface net short-wave radiation", "ShortName": "ssr", "Unit": "J m-2",  "ParaID": 176},
    "ssr6h":{"Name": "Surface net short-wave radiation (6h)", "ShortName": "ssr6h", "Unit": "J m-2", "ParaID": 176},
}


class Saver:
    def __init__(self, save_root: str):
        """
        Initialize the Saver utility.

        Args:
            save_root (str): The root directory where output files will be stored.
                             Subdirectories based on initialization time will be created here.
        """
        self.save_root = save_root

    def _parse_variable_context(self, raw_name: str) -> Tuple[str, Optional[int], bool]:
        """
        Analyze the channel name to determine if it represents a surface variable or a pressure-level variable.

        Logic:
        - Exact match in grib_para implies a Surface variable (e.g., 'tcc', 'msl').
        - 'name_level' format where 'name' exists in grib_para implies a Pressure variable (e.g., 'z_500').

        Returns:
            Tuple containing:
            - short_name (str): The canonical short name of the variable.
            - level (int or None): Pressure level in hPa if applicable.
            - is_pressure (bool): Flag indicating if the variable belongs to a pressure level.
        """
        # Check for exact match in metadata dictionary (Surface variables)
        if raw_name in grib_para:
            return raw_name, None, False

        # Check for pressure level pattern (e.g., z_500, t_850)
        if '_' in raw_name:
            parts = raw_name.split('_')
            base_name = parts[0]
            level_str = parts[-1]

            if base_name in grib_para and level_str.isdigit():
                return base_name, int(level_str), True

        # Fallback for unrecognized variables, treating them as surface variables
        logger.warning(f"Variable '{raw_name}' not recognized in metadata configuration. Treating as surface variable.")
        return raw_name, None, False

    def save(self, 
             data: Union[np.ndarray, object], 
             channel_mapping: List[str], 
             init_time_str: str, 
             lead_time_hours: int,
             lat_values: Optional[np.ndarray] = None,
             lon_values: Optional[np.ndarray] = None,
             member: Optional[int] = None) -> None:
        """
        Orchestrates the conversion of tensor data to a NetCDF file with proper metadata.

        Args:
            data: Input tensor with shape [C, H, W] or [B, C, H, W]. Batch dimension must be 1.
            channel_mapping: List of strings describing each channel in the C dimension.
            init_time_str: Initialization time string in 'YYYYMMDDHH' format.
            lead_time_hours: Forecast lead time in hours.
            lat_values: Array of latitude coordinates. Defaults to global 90 to -90.
            lon_values: Array of longitude coordinates. Defaults to global 0 to 360.
            member: Optional ensemble member index for filename differentiation.
        """
        
        # Ensure input data is in numpy format, handling torch.Tensor if available
        if HAS_TORCH and isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        if data.ndim == 4:
            data = data.squeeze(0)
            
        C, H, W = data.shape
        if len(channel_mapping) != C:
            raise ValueError(f"Channel mapping length ({len(channel_mapping)}) does not match data channels ({C}).")

        # Generate default global coordinates if none are provided
        if lat_values is None:
            lat_values = np.linspace(90.0, -90.0, H)
        if lon_values is None:
            lon_values = np.linspace(0.0, 360.0, W, endpoint=False)

        # Parse time information
        try:
            init_time = datetime.strptime(init_time_str, "%Y%m%d%H")
        except ValueError:
            init_time = datetime.fromisoformat(init_time_str)
            
        valid_time = init_time + timedelta(hours=lead_time_hours)

        # Container dictionaries for segregating variables
        surface_dict = {}
        pressure_dict = {} 

        # Iterate through channels and classify variables
        for i, raw_name in enumerate(channel_mapping):
            short_name, level, is_pressure = self._parse_variable_context(raw_name)
            slice_data = data[i]

            if is_pressure:
                # Organize pressure data: {variable: {levels: [], data: []}}
                if short_name not in pressure_dict:
                    pressure_dict[short_name] = {'levels': [], 'data': []}
                pressure_dict[short_name]['levels'].append(level)
                pressure_dict[short_name]['data'].append(slice_data)
            else:
                # Create DataArray for surface variable immediately
                meta = grib_para.get(short_name, {"Name": raw_name, "Unit": "unknown"})
                
                da = xr.DataArray(
                    data=slice_data[None, :, :],
                    dims=["time", "latitude", "longitude"],
                    coords={
                        "time": [valid_time],
                        "latitude": lat_values,
                        "longitude": lon_values
                    },
                    attrs={
                        "units": meta["Unit"],
                        "long_name": meta["Name"],
                        "GRIB_shortName": short_name
                    }
                )
                surface_dict[short_name] = da

        # Construct the final list of Datasets
        ds_list = []
        
        # Append surface dataset if variables exist
        if surface_dict:
            ds_list.append(xr.Dataset(surface_dict))

        # Process and append pressure datasets
        for name, content in pressure_dict.items():
            meta = grib_para.get(name, {"Name": name, "Unit": "unknown"})
            
            # Sort levels numerically (ascending) to ensure monotonic coordinates
            sorted_pairs = sorted(zip(content['levels'], content['data']), key=lambda x: x[0])
            sorted_levels = [p[0] for p in sorted_pairs]
            sorted_data_list = [p[1] for p in sorted_pairs]
            
            # Stack data along the pressure level dimension
            stack_data = np.stack(sorted_data_list, axis=0)[None, ...]

            plev_dim = pressure_level_dim_name(name)
            da = xr.DataArray(
                data=stack_data,
                dims=["time", plev_dim, "latitude", "longitude"],
                coords={
                    "time": [valid_time],
                    plev_dim: xr.DataArray(
                        sorted_levels,
                        dims=[plev_dim],
                        attrs={"units": "hPa", "long_name": "isobaric level"},
                    ),
                    "latitude": lat_values,
                    "longitude": lon_values
                },
                attrs={
                    "units": meta["Unit"],
                    "long_name": meta["Name"],
                    "GRIB_shortName": name,
                    "positive": "down"
                }
            )
            ds_list.append(da.to_dataset(name=name))

        if not ds_list:
            logger.warning("No valid variables found. Aborting save operation.")
            return

        # Merge all datasets into a single entity
        final_ds = xr.merge(ds_list)
        
        # Assign global attributes complying with conventions
        final_ds.attrs["initial_time"] = init_time.isoformat()
        final_ds.attrs["forecast_lead_time"] = f"{lead_time_hours} hours"
        final_ds.attrs["creation_date"] = datetime.now().isoformat()
        final_ds.attrs["generator"] = "NWPBench Saver"
        
        """
        # Construct file path: root/InitTime/YYYY-MMDD-Lead.nc
        folder_name = init_time_str
        date_part = init_time.strftime("%Y-%m%d")
        
        if member is not None:
            file_name = f"{date_part}-{lead_time_hours:02d}_{member}.nc"
        else:
            file_name = f"{date_part}-{lead_time_hours:02d}.nc"
            
        save_dir = os.path.join(self.save_root, folder_name)
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, file_name)
        
        # Persist data to disk
        logger.info(f"Saving NetCDF file to: {save_path}")
        final_ds.to_netcdf(save_path)
        """

        # For temporary debugging, save to a fixed path
        import shutil
        import tempfile
        
        folder_name = init_time_str
        date_part = init_time.strftime("%Y-%m%d")
        if member is not None:
            file_name = f"{date_part}-{lead_time_hours:02d}_{member}.nc"
        else:
            file_name = f"{date_part}-{lead_time_hours:02d}.nc"
        save_dir = os.path.join(self.save_root, folder_name)
        os.makedirs(save_dir, exist_ok=True)
        final_path = os.path.join(save_dir, file_name)

        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        try:
            logger.info(f"Writing temporary NetCDF to: {tmp_path}")
            final_ds.to_netcdf(tmp_path)
            logger.info(f"Moving to final destination: {final_path}")
            shutil.move(tmp_path, final_path)
        except Exception as e:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            logger.error(f"Failed to save NetCDF: {e}")
            raise
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        logger.info(f"✅ Successfully saved NetCDF file to: {final_path}")