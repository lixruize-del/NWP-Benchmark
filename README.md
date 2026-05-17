# NWP-Benchmark (Large-Scale Inference Workspace)

This workspace hosts the large-scale inference and evaluation pipeline for AI weather models.

## Quick Start

- Environment: `conda activate nwp_unified`
- **ERA5 np.25 (class-based v2 runners):** `python -u run_large_scale_v2.py` — same CLI as `run_large_scale.py`, plus optional `--min_init_time YYYYMMDDHH` when iterating `--start` / `--end`.
- **IFS-era5 hybrid:** `python -u run_large_scale_v2_ifs.py` — same CLI as `run_large_scale_ifs.py` (v2 adapters for supported models).
- Legacy single-file entry: `python run_large_scale.py` / `python run_large_scale_ifs.py` (non-v2 `build_adapter` paths).
- Metrics: latitude-weighted `WRMSE/BIAS/MAE`, `ACC` (from daily climatology when available), `Activity` placeholder (`NaN` by project decision).

## Large-scale batch runs (v2 entrypoints)

Use `run_large_scale_v2.py` or `run_large_scale_v2_ifs.py` directly. Pass either `--init_time YYYYMMDDHH` for a single analysis time or `--start` / `--end` / `--init_hours` to loop over many inits (see `run_large_scale.py --help`). Choose `--mode both` (or offline / metrics-only) and set `--output_csv`, `--nc_dir`, `--save_vars`, `--eval_vars`, `--save_lead_range`, etc., to match your directory layout (for example under `nwp_outputs/era5_monthly_202506_v2/` or `nwp_outputs/ifs_monthly_202506_v2/`).

The old `scripts/run_era5_monthly_forecast_and_metrics*.py` orchestrators and `launch_all_models_parallel*.sh` / `v2_tmux_supervisor.sh` have been removed from this repo; run **one process per model (or per GPU)** with `CUDA_VISIBLE_DEVICES` / tmux / a scheduler, mirroring whatever shard of `--init_hours` and date range you need.

```bash
conda activate nwp_unified
cd /path/to/NWP-Benchmark

# Single init, short leads (ERA5 v2)
python -u run_large_scale_v2.py \
  --model pangu \
  --init_time 2025060100 \
  --lead_times 6 12 18 \
  --era5_root /ecmwf-era5-datasets/era5_np.25 \
  --mode both

# Date span (ERA5 v2) — same flags as run_large_scale.py
python -u run_large_scale_v2.py \
  --model stormer \
  --start 2025-06-01 --end 2025-06-30 \
  --init_hours 0 12 \
  --lead_times 6 12 18 24 \
  --era5_root /ecmwf-era5-datasets/era5_np.25 \
  --mode both
```

### Why GPU may look idle

`nvidia-smi` utilization can briefly show `0%` even when workers are active.
Check both process and GPU state together:

```bash
ps -eo pid,stat,etime,cmd | grep -E 'run_large_scale(_v2)?\\.py'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
```

- `Sl/Rl`: usually active compute stage.
- `D`: usually storage/I/O wait; progress may continue after wait clears.
- Track progress via the paths you pass to `--output_csv` / `--nc_dir`, plus `ps` / `nvidia-smi`. Older shared-root monthly runs also used `summary.csv` and `metrics/per_init/` under one `--out-root`.

## Metrics backfill from saved NetCDF

Use this when forecasts already exist under `<forecasts_root>/<model>/<YYYYMMDDHH>/*.nc` but you need to **append or rebuild** rows in each model’s `metrics/<model>_metrics.csv` without re-running inference.

`scripts/backfill_missing_metrics_from_nc_parallel.py` calls the same weighted-metric path as `run_large_scale.py` (including `use_eval_wrmse` parity via `src.common.era5_eval_regrid`).

- **Stormer** outputs on native `(128, 256)` are regridded to ERA5 `721×1440` with `interpolate_stormer_to_721` using lat/lon read from the NetCDF.
- **NeuralGCM** outputs that are not already `721×1440` are passed through `stack_native_to_era5_025` before GT comparison (same helper as the monthly pipeline).

Optional: build a CSV of missing inits with `scripts/list_missing_by_init_csv.py`, then pass it into the parallel backfill driver. See `--help` on both scripts for `--forecasts-root`, `--models`, `--by-init-dir`, `--workers`, and related flags.

## Typhoon / tropical cyclone track evaluation (storm-centric)

**Recommended entrypoint:** `scripts/evaluate_tc_by_storm.py` — loops **storm → init → lead**, reads IBTrACS directly (default CSV: `data/tc/ibtracs.last3years.list.v04r01.csv`, override with `--ibtracs-csv`), and resolves forecasts under `--forecast-root` / optional `--fallback-forecast-roots` / `--era5-fallback-root`. NetCDF names follow the same middle token as monthly saves: `YYYY-mdd-<lead>h.nc` under `forecasts/<model>/<YYYYMMDDHH>/`.

```bash
conda activate nwp_unified
cd /path/to/NWP-Benchmark
python -u scripts/evaluate_tc_by_storm.py \
  --forecast-root /path/to/nwp_outputs/era5_monthly_202506_v2/forecasts \
  --models pangu \
  --source ERA5 \
  --season 2025 \
  --start-date 2025-06-01 \
  --end-date 2025-12-31 \
  --lead-step 6 \
  --max-lead 240 \
  --out-dir tc_eval_results/storm_centric
```

Use `--resume` for long runs (skips finished inits; see `--resume-skip-reasons`, `--no-checkpoint-every-init`). For a single init and optional trajectory plot: `--single-init-time 2025082012` and `--plot-trajectory`. Tracker tuning: `--distance-threshold`, `--wind-threshold`, `--resolution`.

**Parallel by model:** `python scripts/run_tc_eval_parallel_by_model.py` (one subprocess per model; same CLI knobs as above via wrapper defaults).

**IFS layout on a cluster:** `bash scripts/start_tc_eval_ifs_tmux_parallel_models.sh` (tmux per model, primary/fallback IFS + ERA5 roots via env vars in the script header).

**Slow NFS / GraphCast:** `bash scripts/run_tc_eval_graphcast_ifs_cached.sh` (wraps the same eval with a larger `--forecast-nc-cache-size`).

**Tables and paper-style figures:** `scripts/build_tc_tables_and_figures.py`, `scripts/plot_tc_case_figure9_style.py` (point them at your `tc_eval_results/...` CSV tree).

## IFS layout: station metrics from saved forecasts

After `run_large_scale_v2_ifs.py` (or equivalent) has written forecast NetCDF under `forecasts/<model>/<YYYYMMDDHH>/`, evaluate **point station** scores with **`scripts/eval_station_metrics_from_saved_ifs_nc.py`**. It bilinearly samples the grid at each station location, compares to station observations under `--station-root`, and optionally uses `--era5-root` climatology for `activity` / `acc`. Outputs are CSV rows with `init_time`, `valid_time`, `lead_hours`, `variable`, `wrmse`, `bias`, `mae`, `activity`, `acc`.

**Paths you will almost always override:** `--forecasts-root` (or priority list `--forecasts-roots ...`), `--station-root`, `--station-latlon-json`, `--out-root`, and optionally `--by-init-dir` + `--write-by-init` for per-init shards. Defaults inside the script point at this workspace’s historical layout; treat them as examples only.

**Parallelism:** the script supports `--max-workers` (per-model sample workers) and `--model-parallel-jobs` (how many models run concurrently); cap CPUs with `--cpu-cap` or env `STATION_EVAL_CPU_CAP`. For large windows, a practical pattern is **one shell process per model** (e.g. separate tmux panes) each calling the script with `--models <one_model>` and the same `--out-root` / `--by-init-dir` so merged files accumulate consistently. Set `OMP_NUM_THREADS=1` when using many workers to avoid oversubscription.

```bash
conda activate nwp_unified
cd /path/to/NWP-Benchmark
export OMP_NUM_THREADS=1

python -u scripts/eval_station_metrics_from_saved_ifs_nc.py \
  --forecasts-root /path/to/nwp_outputs/ifs_monthly_202506_v2/forecasts \
  --models pangu \
  --start 2025-06-01 --end 2025-06-30 \
  --init-hours 0 6 12 18 \
  --max-lead 240 \
  --max-workers 8 \
  --station-root /path/to/StationCast/dataset/processed/2025 \
  --station-latlon-json /path/to/station_latlon_2025.json \
  --era5-root /ecmwf-era5-datasets/era5_np.25 \
  --out-root /path/to/nwp_outputs/ifs_monthly_202506_v2/metrics_station \
  --by-init-dir /path/to/nwp_outputs/ifs_monthly_202506_v2/metrics_station/by_init \
  --write-by-init \
  --resume \
  --flush-every-samples 500
```

Use `--forecasts-roots root1 root2 ...` (first wins per file) when forecasts are split across VEPFS and NAS copies. See `python scripts/eval_station_metrics_from_saved_ifs_nc.py --help` for `--replace-vars`, `--eval-vars`, and other tuning.

## GHCNh station preprocessing (ISD hourly → NetCDF)

These scripts were copied from `StationCast/dataset` for a **self-contained station pipeline** alongside `eval_station_metrics_from_saved_ifs_nc.py`: first fetch GHCNh hourly PSVs, then build hourly station cubes with ERA5-based QC.

| Script | Role |
|--------|------|
| `scripts/downloader.py` | Download GHCNh hourly files from NOAA NCEI (`by-station` or `by-year`). Requires `ghcnh-station-list.csv` in the **current working directory** (first column = station IDs). Writes under `./ISD_raw/by-station/` or `./ISD_raw/by-year/<year>/`. |
| `scripts/CHCNh_process_new.py` | Read yearly `.psv` from `STATIONCAST_ISD_BY_YEAR` (default: `$STATIONCAST_ROOT/ISD_raw/by-year`), merge stations per 6-hourly UTC time, optionally QC against ERA5 single-level NPY, and write `NetCDF` under `$STATIONCAST_ROOT/processed/<year>/<timestamp>.nc`. |

**Environment variables (preprocess):**

- `STATIONCAST_HOME` — default `/root/NWP/StationCast`; used with `STATIONCAST_ROOT` / `STATIONCAST_CACHE` if unset.
- `STATIONCAST_ROOT` — dataset root containing `ISD_raw/` and `processed/` (default `$STATIONCAST_HOME/dataset`).
- `STATIONCAST_ISD_BY_YEAR` — ISD tree (default `$STATIONCAST_ROOT/ISD_raw/by-year`).
- `STATIONCAST_CACHE` — joblib cache (default `$STATIONCAST_HOME/cache_new`).
- `ERA5_NPY_ROOT` — ERA5 np.25 root for QC `.npy` reads (default `/ecmwf-era5-datasets/era5_np.25`).

**Downloader CLI:**

```bash
cd /path/to/your_station_workdir   # must contain ghcnh-station-list.csv
# Optional: export NWP_HTTP_PROXY='http://user:pass@host:port/' if using --proxy special
python scripts/downloader.py --mode by_year --start_year 2020 --end_year 2020 --worker_num 8 --proxy direct
python scripts/downloader.py --mode by_station --worker_num 4 --proxy direct
```

**Preprocess CLI** (run from repo root or any `PYTHONPATH`; paths resolve via env above):

```bash
export STATIONCAST_HOME=/path/to/StationCast   # or set STATIONCAST_ROOT / STATIONCAST_ISD_BY_YEAR explicitly
export ERA5_NPY_ROOT=/ecmwf-era5-datasets/era5_np.25
python scripts/CHCNh_process_new.py --start_time "2020-06-01 00:00:00" --end_time "2020-06-01 18:00:00"
```

**Dependencies:** `pandas`, `xarray`, `numpy`, `tqdm`, `joblib` (preprocess); `requests`, `beautifulsoup4`, `urllib3` (downloader). Install via your `nwp_unified` env or `requirements-large-scale.txt` as appropriate.

## Heatwave / coldwave: file map and pipelines

**Workflow:** build ERA5 percentile baselines, then run **object verification v2** (daily Tmax/Tmin series + temporal IoU), usually via `run_heatwave_object_eval_batch_v2.py`. An older CSV-based init-level event eval was removed from this repo; recover from version control if you need it.

Shared libraries: `scripts/heatwave_common.py` (forecast filename pattern, `to_celsius`, ERA5 helpers—used by **baseline build** and **object v2 Step 1**); `scripts/heatwave_object_common_v2.py` (temporal IoU, greedy matching, mask/event helpers for Steps 2–3). Run CLIs as `python scripts/<script>.py` from the repo root (Python puts `scripts/` on `sys.path` for that invocation), or `cd scripts` then `python <script>.py`.

### File map

| Script | Role |
|--------|------|
| `scripts/heatwave_common.py` | Library for baseline + object v2 Step 1 (not the object-matching algorithm core). |
| `scripts/build_heatwave_baseline_percentile.py` | Build per-grid DOY **p90** (from historical **daily max** °C) and **p10** (from **daily min** °C) baselines from ERA5 climate NetCDFs. **Prerequisite** for object v2 exceedance masks. |
| `scripts/heatwave_object_common_v2.py` | **Library only** (no CLI): 1D span extraction, temporal IoU, greedy matching, metrics edge cases; used by object v2 steps. Optional Numba acceleration. |
| `scripts/build_heatwave_lead_timeseries_v2.py` | **Object v2 Step 1:** one pass over forecasts per model → per–lead-day dirs with `pred_tmax_daily.nc` / `pred_tmin_daily.nc` and shared yearly GT cache (`--gt-cache-file`). |
| `scripts/extract_heatwave_events_v2.py` | **Object v2 Step 2:** exceedance masks vs baseline (`heatwave`: daily Tmax above p90; `coldwave`: daily Tmin below p10) → e.g. `hot_masks_p90.nc` or `cold_masks_p10.nc`. |
| `scripts/eval_heatwave_object_metrics_v2.py` | **Object v2 Step 3:** reads masks NetCDF, spatial/temporal object stats → `counts_tp_fp_fn.nc`, `metrics_grid.nc`, `metrics_global.json`, `metrics_latband.csv`. |
| `scripts/run_heatwave_object_eval_batch_v2.py` | **Batch driver:** builds shared GT cache if missing; **Phase A** runs Step 1 once per model (thread pool); **Phase B** runs Step 2+3 per `(model, lead_day)`. Layout: `<out-root>/<init-source>/<model>/lead_day_<N>/step{1,2,3}/`. Flags: `--skip-phase-a`, `--gt-cache-dir`, `--phase-a-workers`, `--phase-b-workers`, `--dry-run`. |

### 1) Build DOY percentile baselines (`build_heatwave_baseline_percentile.py`)

Reads ERA5 **2 m temperature** climate NetCDFs under `--climate-dir` (default: `/ecmwf-era5-datasets/climate/2_metre_temperature`), converts to **°C**, then:

- Builds **true daily max** and **true daily min** by reducing any sub-daily dimensions and `resample(time="1D").max` / `.min`.
- For each day-of-year (DOY), pools all calendar days in a sliding window of `--window-days` (default 15, centered) and takes the requested empirical quantile(s).
- **Rule:** percentiles **≥ 50** use the **daily max** pool (e.g. **p90** for heatwave-like thresholds); percentiles **< 50** use the **daily min** pool (e.g. **p10** for cold-surge-like lows).

Output variables are named `t2m_p{int(percentile)}_c` (temperature in °C). Defaults if you omit percentile flags: `--percentiles 90 10`.

**Performance / robustness:** materializes daily max/min once (`load()`), then the DOY loop runs in memory. Use `--resume-existing` and per-run `--checkpoint-file`; `--log-file`, `--progress-every-doy`, `--output-dtype float32`, `--open-engine {h5netcdf,netcdf4}`.

Example: **1979–1990**, p90 + p10 in **two files**:

```bash
conda activate nwp_unified
cd /path/to/NWP-Benchmark

python scripts/build_heatwave_baseline_percentile.py \
  --climate-dir /ecmwf-era5-datasets/climate/2_metre_temperature \
  --start-date 1979-01-01 --end-date 1990-12-31 \
  --window-days 15 \
  --percentiles 90 10 \
  --doy-start 1 --doy-end 366 \
  --split-output \
  --heatwave-out-name heatwave_baseline_p90_doy_001_366.nc \
  --coldwave-out-name coldwave_baseline_p10_doy_001_366.nc \
  --out-dir /path/to/your_baseline_dir \
  --resume-existing \
  --checkpoint-file /path/to/your_baseline_dir/baseline_build_checkpoint_MYRUN.json \
  --log-file /path/to/your_baseline_dir/baseline_build_MYRUN.log \
  --progress-every-doy 5
```

Single NetCDF with both variables: omit `--split-output` and set `--out-name` (see `--help`).

### 2) Object-based pipeline v2

Run pieces manually with each script’s `--help`, or drive everything from:

```bash
python scripts/run_heatwave_object_eval_batch_v2.py \
  --forecast-root /path/to/nwp_outputs/.../forecasts \
  --baseline-file /path/to/heatwave_baseline_p90_doy_001_366.nc \
  --models pangu \
  --lead-days 1 3 7 10 \
  --year 2025 \
  --start-date 2025-06-01 \
  --end-date 2025-10-30 \
  --event-type heatwave \
  --out-root /path/to/heatwave_object_v2_metrics
```

For **coldwave**, use a **p10** baseline file and `--event-type coldwave` (batch runner wires `pred_tmin_daily.nc` / `gt_tmin` and `cold_masks_p10.nc`). Tuning: `--iou-threshold`, `--min-duration-days`, `--skip-missing`, `--skip-phase-a`, worker counts.

## Notes

- **`run_large_scale.py`** on ERA5 **`np.25`**: `pangu`, `stormer`, `graphcast`, `fengwu`, `fuxi`, `aurora`, `neuralgcm` (NeuralGCM needs full channel set, e.g. cloud water paths—see runner). IFS-HRES models (`aifs`, `graphcast_operational`, `fengwu_v2`) use separate roots and layouts documented per runner. `run_large_scale.py --mode both` supports offline-save + online-metrics in one inference pass.
- **Climatology / ACC:** daily climatology is read from `ERA5_ROOT/climate_mean_day/{1993-2016,single/1993-2016}/MM-DD/` and used for ACC in `run_large_scale.py` (no extra CLI flag). For native non-721 grids without direct climatology transform (currently NeuralGCM), ACC remains `NaN`.
- **Weights:** override with `NWP_WEIGHTS_ROOT` (FuXi expects `…/fuxi/{short,medium,long}.onnx`).
- **FuXi long rollouts:** optional `FUXI_STAGE_STEPS=a,b,c` for `[short,medium,long]` step counts (each step 6h); must sum to `max(lead_times)//6`.

### ERA5 `np.25` on-disk layout

Paths are defined by `Era5NpyLayout`:

- Pressure: `{root}/{year}/{YYYY-MM-DD}/{HH:MM:SS}-{var}-{level}.npy`
- Surface: `{root}/single/{year}/{YYYY-MM-DD}/{HH:MM:SS}-{var}.npy`

### Precipitation (`tp` vs `tp6h`)

| Consumer | Variable | File / convention |
|----------|-----------|-------------------|
| **GraphCast** | `total_precipitation_6hr` | `tp6h` \*.npy, depth in **metres** (`load_era5_tp6h_depth_m`) |
| **FuXi** | `TP` (last channel) | `tp6h` then **×1000** → **mm**, `clip(0,1000)` per [tpys/FuXi](https://github.com/tpys/FuXi) |
| **Pangu / FengWu / Aurora** (this repo) | — | No precip in current ONNX/Aurora stacks |

`load_snapshot_by_channel_names(..., "tp6h")` loads the `tp6h` file (not hourly `tp`).

## FuXi quick run

```bash
conda activate nwp_unified
cd /path/to/NWP-Benchmark
python -u run_large_scale.py \
  --model fuxi \
  --init_time 2025051212 \
  --lead_times 6 \
  --era5_root /ecmwf-era5-datasets/era5_np.25
```

Metrics append to `outputs/fuxi/metrics.csv`. Channel names match FuXi order (e.g. `Z500` is **geopotential** at 500 hPa, m²/s²).

## GraphCast Status (Strict Replica Mode)

- `src/models/graphcast_runner.py` is rewritten to follow `/home/NWP-Benchmark/src/graphcast/inference.py` and `prepare.py` structure.
- Critical static fields now prioritize `static.nc` (`z` and `lsm`) instead of reconstructed proxies.
- Default static path in this workspace: `static.nc` at repo root (override with `NWP_GRAPHCAST_STATIC_NC`).
- 6h check (`init=2025051212`) now returns `z500 WRMSE = 15.662233819601873` (target range met).

## Aurora metrics grid

- Aurora outputs `(720,1440)` after internal crop from `(721,1440)`. For GT regridding and latitude weights, `run_large_scale.py` uses `np.linspace(90,-90,721)[:-1]`, not cell-centered `89.875…-89.875`. See [Form of a Batch](https://microsoft.github.io/aurora/batch.html).

## Historical Conversation Context

- This workspace follows a strict policy from user dialogue: no speculative experiments when baseline code exists; first read `/home/NWP-Benchmark/src/<model>` fully, then replicate.
- Previous deviation (approximated GraphCast static geopotential) was superseded by direct `static.nc` usage to align with original script semantics.
- Transcript reference for this decision chain: [GraphCast strict alignment](68eb6c8a-0f0b-4645-b7d6-73663638c295).
