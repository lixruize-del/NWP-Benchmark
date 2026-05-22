# NWP-Benchmark

Large-scale AI weather model inference and evaluation on ERA5 and IFS layouts.

> Draft reorganized README (see `README.md` for the current repo default until this replaces it).

---

## 1. Environment

Create and use the conda environment `nwp_unified`:

```bash
conda create -n nwp_unified python=3.10 -y
conda activate nwp_unified
cd /path/to/NWP-Benchmark
pip install -r requirements-large-scale.txt
```

Install **PyTorch** and **ONNX Runtime** builds that match your GPU/CUDA stack (not pinned in `requirements-large-scale.txt`). Model weights are loaded via `NWP_WEIGHTS_ROOT` at run time.

---

## 2. Inference data

**Dataset paths for this release are not bundled in the repo; they will be published separately soon.** After download, point `--era5_root` / station / TC paths in the commands below to your local layout.

| Purpose | What you need |
|---------|----------------|
| ERA5 inference & grid metrics | ERA5 **np.25** NPY tree (pressure + `single/`) |
| IFS inference | IFS-era5 **analysis_np.25** NPY tree (or equivalent layout) |
| Model weights | ONNX / checkpoints under `NWP_WEIGHTS_ROOT` |
| Typhoon evaluation | IBTrACS CSV (default in repo: `data/tc/ibtracs.last3years.list.v04r01.csv`) |
| Station evaluation | Processed station obs + station lat/lon JSON |
| Heatwave / coldwave | ERA5 2 m temperature climate files for baselines; pre-built p90/p10 NetCDF optional |

---

## 3. Inference (examples)

Entrypoints:

- **ERA5:** `run_large_scale_v2.py`
- **IFS:** `run_large_scale_v2_ifs.py`

Both accept the same style of arguments as `run_large_scale.py` / `run_large_scale_ifs.py` (`--help` for full list). Use `--mode both` to save forecast NetCDF and write metrics in one pass.

### ERA5 (example)

```bash
conda activate nwp_unified
cd /path/to/NWP-Benchmark

python -u run_large_scale_v2.py \
  --model pangu \
  --init_time 2025060100 \
  --lead_times 6 12 18 24 \
  --era5_root /path/to/era5_np.25 \
  --mode both \
  --output_csv /path/to/nwp_outputs/era5_monthly_202506_v2/metrics/pangu_metrics.csv \
  --nc_dir /path/to/nwp_outputs/era5_monthly_202506_v2/forecasts/pangu \
  --save_lead_range 6 240 \
  --save_vars u10 v10 msl t2m u_850 v_850 u_500 v_500 z_500 \
  --eval_vars u10 v10 msl t2m u_850 v_850 u_500 v_500 z_500
```

For many inits, use `--start`, `--end`, and `--init_hours` instead of `--init_time`.

### IFS (example)

```bash
python -u run_large_scale_v2_ifs.py \
  --model aifs \
  --init_time 2025060100 \
  --lead_times 6 12 18 24 \
  --era5_root /path/to/analysis_np.25 \
  --mode both \
  --output_csv /path/to/nwp_outputs/ifs_monthly_202506_v2/metrics/aifs_metrics.csv \
  --nc_dir /path/to/nwp_outputs/ifs_monthly_202506_v2/forecasts/aifs \
  --save_lead_range 6 240 \
  --save_vars u10 v10 msl t2m u_850 v_850 u_500 v_500 z_500 \
  --eval_vars u10 v10 msl t2m u_850 v_850 u_500 v_500 z_500
```

Run **one model per GPU/process** for long windows (`CUDA_VISIBLE_DEVICES`, tmux, or your scheduler).

---

## 4. Evaluation

All evaluation assumes **saved forecasts** under  
`forecasts/<model>/<YYYYMMDDHH>/*.nc`  
(from step 3 or your own export).

| Topic | Script |
|-------|--------|
| **Station**  | `scripts/eval_station_metrics_from_saved_ifs_nc.py` |
| **Typhoon / TC**  | `scripts/evaluate_tc_by_storm.py` |
| **Heatwave**  | `scripts/run_heatwave_object_eval_batch_v2.py` (after p90 baseline) |
| **Cold surge**  | Same batch driver with p10 baseline and `--event-type coldwave` |

### Station

```bash
python -u scripts/eval_station_metrics_from_saved_ifs_nc.py \
  --forecasts-root /path/to/forecasts \
  --models pangu \
  --start 2025-06-01 --end 2025-06-30 \
  --station-root /path/to/station/processed/2025 \
  --station-latlon-json /path/to/station_latlon_2025.json \
  --out-root /path/to/metrics_station
```

### Typhoon / TC

```bash
python -u scripts/evaluate_tc_by_storm.py \
  --forecast-root /path/to/forecasts \
  --models pangu \
  --season 2025 \
  --start-date 2025-06-01 --end-date 2025-12-31 \
  --out-dir tc_eval_results/storm_centric
```

### Heatwave

1. Build baseline (once): `scripts/build_heatwave_baseline_percentile.py` → p90 NetCDF.  
2. Run object verification:

```bash
python scripts/run_heatwave_object_eval_batch_v2.py \
  --forecast-root /path/to/forecasts \
  --baseline-file /path/to/heatwave_baseline_p90_doy_001_366.nc \
  --models pangu \
  --event-type heatwave \
  --year 2025 \
  --start-date 2025-06-01 --end-date 2025-10-30 \
  --out-root /path/to/heatwave_object_v2_metrics
```

### Cold surge 

Use a **p10** baseline from the same baseline script (`--percentiles 10`), then:

```bash
python scripts/run_heatwave_object_eval_batch_v2.py \
  --forecast-root /path/to/forecasts \
  --baseline-file /path/to/coldwave_baseline_p10_doy_001_366.nc \
  --models pangu \
  --event-type coldwave \
  --year 2025 \
  --start-date 2025-06-01 --end-date 2025-10-30 \
  --out-root /path/to/coldwave_object_v2_metrics
```
