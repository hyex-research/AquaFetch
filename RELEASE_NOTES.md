## AquaFetch 1.2.1


-  LamaHIce now uses 1.5 instead of 2024 
- For CAMELS_NZ, the catchment boundaries are returned in wgs84
- For CAMELS_DE (daily), LamaHIce and CAMELS_DK, the catchment boundaries are returned in wgs84; CAMELS_DK no longer fails for its 29 catchments with holes


## AquaFetch 1.2.0

Everything below is in the `rr` submodule. The table is the complete per-dataset
list; the sections after it cover only what is common to all of them. Blank cell
= unchanged since 1.1.0.

| Dataset | Release | Stations | Dyn. feats | Static feats | Dynamic length | Other changes |
|---|---|---|---|---|---|---|
| **CAMELS_GB** | v2 default, `version=1` = old | | hourly: 7 (new timestep) | 145 → 219 (v1: 145) | 16436 → 18993 daily steps; +280512 hourly | Boundaries British National Grid→WGS84; v1 `rh_%`→`spechum_gkg`, `slope_`→`slope_fdc`; per-version folders/caches |
| **CAMELS_BR** | 1.2 default, `version='1.1'` = old | | 11 → 26 | 67 → 66 | 14245 → 16437 daily steps | 34 % of 1980-2018 streamflow revised; new `update_streamflow()`, `ana_streamflow()`; `folders`→`group_folders`; `all_stations(feature)`→`all_stations(group)`; `fetch_simulated_streamflow()` now 1.1 only |
| **CAMELS_CL** | 2022 default, `version=2018` = old | | 12 → 10 | 104 → 110 | → 44368 daily steps (1900-01-01→2021-06-22) | 2018: min air temp now to 2016-12-31 (was cut 2010-03-09), float32, numeric statics, timestamp `start`/`end` |
| **CAMELS_CH** | 0.6 → 0.9 | | hourly: 9 → 1 | | hourly: 1923-02-15→2021-02-08 | `timestep='H'` used to serve the daily data; `-9999`→NaN; `to_netcdf`/`float_precision`/`overwrite` were ignored; Caravan extension no longer downloaded |
| **CAMELS_COL** | Feb 2026; no `version` (old record restricted) | 347 → 346 | 6 → 5 | 255 → 79 | | 162 boundaries revised, areas ±17 %; EPSG:3395→WGS84 ellipsoidal (spherical one was 8.2 km off, `get_boundary()` returned metres); float32; soil water share → `water_bodies_soil_perc` |
| **CAMELS_FR** | 2.1 → 3.2 | | | | | `hym_q_questionable`, `hym_q_unqualified`, `hym_q_anomaly_inrae` recomputed at source; radiation J cm-2 day-1→W m-2; only the attributes archive re-downloaded |
| **CAMELS_FI** | 1.0.1 → 1.2.0, 1.0.1 withdrawn | | `pe_era5_land`→`pet_singer` | 106 → 112 | | 1.0.1 coords ~23 km off at all 320 gauges; ERA5-Land PET replaced (aridity 0.86→0.67); `snowdepth_m` was cm; `get_boundary()` returned EPSG:3067 metres; `baseflow_index_ladson`→`baseflow_index`; `temperature_mean`→`temperature_mean_annual`; radiation kJ m-2 day-1→W m-2 |
| **CAMELS_IND** | 2 → 2.2, `version='2'` from disk only | | | | | Release 2 gave 55 gauges of basins 12 & 15 the previous gauge's name/coords/area/elevation; streamflow revised at 181 gauges; `dspbar`→`dpsbar`; `slope_degrees`→`slope_%`; `evap_*` named per release's unit; streamflow read once (1.6 s vs 4.5 s); `to_netcdf`/`overwrite` were ignored; LSTM output not extracted |
| **CAMELS_NZ** | 2 → 5, release 2 no longer read | | | 40 → 37 | | Daily `pet_mm` was 24× too small; hourly timestamps were NZ local time with DST; `stn_coords()` raised for every gauge; `overwrite=True` skipped the download; only the requested timestep downloaded (296 MB vs 5.2 GB) |
| **CAMELS_LUX** | 1.1 → 2.1, 1.1 withdrawn | | 25 → 26 | names/values changed | 2004-11-01→2021-10-31, read from files (was hardcoded 2004-01-01→2021-12-31) | `q_cms_obs` differs on 11 % of days, ERA5 recomputed; `airtemp_C_mean`→`airtemp_C_mean_era5`; `AI_Oudin`/`AI_PM`→`P_PET_Oudin`/`P_PET_PM` (reciprocal); `spechum_gkg` was kg/kg; boundary gauge ids were invented; `cin` NaN where none (was 0); duplicate timestamps now dropped at all timesteps |
| **CAMELS_US** | | | 8 → 9 | | | Daymet `srad` is a daylight mean → `swdownrad_wm2_daylight`; 24-h mean `swdownrad_wm2` derived |
| **CAMELS_AUS** | | | | | | Solar radiation MJ m-2 day-1→W m-2 |
| **CABra** | | | | | | `srad_*` MJ m-2 day-1→W m-2 |
| **HYSETS** | | | | | | Radiation J m-2 day-1→W m-2; warns that the ERA5 `surface_net_thermal_radiation` is defective (92.9 % zeros) — use ERA5Land |
| **LamaHCE** | | | | | | Refactored: cached tables returned as copies, instance not pickled to pool workers, ~8× faster index; `-999`→NaN and duplicate-gauge warnings; net shortwave was under the downward name |
| **LamaHIce** | | | | | | Positive-upward net thermal kept under its source name, with a warning |
| **Bull** | | | | | | Six radiation columns are net → `swnetrad_wm2*`, `lwnetrad_wm2*` |
| **GRDCCaravan** | | | | | | Same net-radiation renaming as BULL |
| **Caravan_DK** | | | | | | Radiation newly mapped (net, W m-2) |
| **GSHA** | | | | | | `SHORTRAD_*`/`LONGRAD_*` relabelled net; netCDF encoding bug and unguarded print fixed |
| **CAMELS_PE** | | | | | | `srad` newly mapped, MJ m-2 day-1→W m-2 |
| **CAMELSH** | | | | | | `SWdown`/`LWdown` newly mapped |
| **CAMELS_DE** | | | | | | Hourly boundaries EPSG:3035→WGS84 now ellipsoidal; spherical-approximation warning removed; spatial min/max radiation names kept distinct from BULL's temporal ones |
| **CAMELS_SK** | | | | | | Radiation left unmapped — ERA5-Land running accumulation on hourly rows |
| **CAMELS_PL** | | | | | | `radiation_global_mean`→`swdownrad_wm2` |
| **EStreams** (+ Finland, Italy, Poland, Portugal, Slovenia) | | | | | | `swr_mean`→`swdownrad_wm2`; Portugal downloader uses shared `BROWSER_HEADERS` |
| **Ireland** | | | | | | Streamflow sorted by time before caching |
| **UKFlow15** | | | | | | Docs: 664 gauges overlap CAMELS_GB v2, whose hourly flow is this record hourly-averaged |

### New dataset
- **CAMELS_KR** — 282 South Korean catchments, 14 dynamic features (1981-01-01→2025-12-31,
  16436 daily steps) and 75 static. Simulated streamflow and HBV parameters are model
  output and are not extracted.


### Breaking: radiation feature names and units

One convention, documented in `aqua_fetch/rr/_map.py`:

    <band><direction>rad_wm2 [_<stat>] [_<source>]

`band` is `sw`/`lw` (shortwave = solar, longwave = thermal), `direction` is
`down`/`up`/`net`. `solrad_wm2*` → `swdownrad_wm2*`, `solradnet_wm2*` →
`swnetrad_wm2*`, `thermrad_wm2*` → `lwnetrad_wm2*` (merged; the two named one
quantity); `lwdownrad_wm2*` unchanged.

`wm2` was a label, not a promise: six datasets shipped J cm-2 day-1,
MJ m-2 day-1, J m-2 day-1, kJ m-2 day-1 or a daylight-period mean under it, and
five published *net* shortwave under the downward name. Both are fixed per the
table. Sign conventions are reported, not normalised: LamaH-CE and LamaH-Ice
publish net thermal positive-upward, so those series keep their source names.

### Caches

Cache files now carry a version, e.g. `camels_gb_D_v2.nc` instead of
`camels_gb_D.nc`, plus the release and timestep where a dataset has more than
one (`camels_br_D_1.2_v2.nc`). Old caches are ignored and rebuilt once from the
source files, not deleted — `.nc` files without `_v2` can be removed by hand.

### Downloading and extraction

For every dataset whose release changed, and for CAMELS_COL:

- whether to download is decided by the presence of the **extracted** data, not
  the archive, so `remove_zip=True` (the default of `RainfallRunoff`) no longer
  forces a re-download on the next initialization;
- extraction goes through a temporary folder, so an interrupted one is redone
  instead of being taken for complete;
- `overwrite=True` deletes the archives, extracted folders and caches first;
- files missing from a release are reported against the release's own file list
  instead of silently making the dataset smaller.

### Upgrading an existing data folder

Each release lives in its own sub-folder, so an old copy is never served as the
new one. On the first initialization CAMELS_FI and CAMELS_LUX delete it with its
caches; CAMELS_COL, CAMELS_IND and CAMELS_NZ warn and keep it (delete it
yourself to free the space); a CAMELS_BR 1.1 extracted directly into
`CAMELS_BR/` is still read from there. CAMELS_IND deliberately keeps the
archives of superseded releases, because Zenodo has restricted them and they
cannot be fetched again.

### Also

`aqua_fetch._geom_utils` now reprojects on the ellipsoid instead of a sphere,
accepts arrays, and gained `world_mercator_to_wgs84`; all inverses are checked
against pyproj, which remains not a dependency. `rr/utils.py` gained the shared
`cache_name()`, `ymd_index()` (raises on an impossible date instead of rolling it
into the next month) and `apply_dyn_factors()`.

## AquaFetch 1.1.0

### New datasets
- CAMELS_DE — Germany
- CAMELS_PL — Poland
- CAMELS_PE — Peru (136 catchments)
- UKFlow15 — UK 15-minute streamflow (1,369 gauges)
- CaravanQual — water-quality extension for Caravan
- GlobalRiverNutrients — total N / total P / streamflow (observational)
- NamalValleyPakistan

### Enhancements
- Faster / refactored HYSETS, GSHA, LamaH-Ice, EStreams, CAMELS-H, CAMELS-DE
- Efficient time-weighted resampler; time-weighted averaging for Ireland (OPW)
- New `free_disk_space` method (rr) and two new HYSETS methods
- Optional `seed` for reproducible station sampling in `fetch()`

### Fixes
- Restored core dependencies (pandas, requests) in `install_requires`
- Fixed CAMELS-CH coordinate bug and static-feature renaming bug
- Ireland OPW stations now assert timezone info
- Removed unguarded print statements
