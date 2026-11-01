"""
Tests for the CAMELS_CZ (Czechia, 249 catchments) dataset at daily and hourly
timestep.

They check that the class

    * downloads and extracts the archives of each timestep once, skips the
      per-gauge netCDF copies, and never re-downloads or re-extracts data that
      is already on disk,
    * returns every dynamic and static value unchanged (compared with an
      independent read of the raw csv files) apart from the documented kPa ->
      hPa conversion of the mean surface pressure and the degrees x 10 ->
      degrees conversion of the slope, and derives ``q_cms_obs`` with the
      areas the authors used,
    * reports feature lists, stations and the temporal extent of the data,
    * reads efficiently (process pool only for large reads, never with
      ``processes=1``).

Most tests that fetch through the public API read the netCDF cache; the csv
reader is exercised by the all-stations, float64, reader-spec and mini-archive
tests.

The first run downloads 1.6 GB (daily) and 2.3 GB (hourly) into CAMELS_CZ_PATH.
"""

import os
import sys
import time
import atexit
import shutil
import random
import zipfile
import logging
import tempfile
import warnings
import functools
import contextlib
import concurrent.futures as cf
from unittest import mock

import numpy as np
import pandas as pd
import pytest

# repository root and this folder (for the shared ``utils`` helpers)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    logging.basicConfig(filename='test_camels_cz.log', filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

import aqua_fetch
import aqua_fetch.download_zenodo as download_zenodo
from aqua_fetch import CAMELS_CZ
from aqua_fetch.rr._camels import _cz
from aqua_fetch._backend import xarray as xr, fiona

import utils as rr_utils

# The class appends ``CAMELS_CZ`` to this path. Replace it with the location on
# your machine.
CAMELS_CZ_PATH = '/mnt/hyexnas01/abbaa0a/data/gscad_database/raw/CAMELS'

NUM_STATIONS = 249       # gauges with time series (attributes_caravan_camelscz.csv)
UNSERVED = ['P4153300']  # has attributes but no time series
NUM_STATIC = 215
DAILY_LEN = 27603        # 1950-01-02 .. 2025-07-29
HOURLY_LEN = 79401       # 2016-06-16 22:00 .. 2025-07-08 06:00

CARAVAN, CAMELS, HOURLY = 'Caravan-Extension-CZ', 'CAMELS-CZ', 'CAMELS-CZ-Hourly'

# raw column -> name served by the class. Written out here (not imported) so
# that a wrong mapping in the class is caught.
DAILY_RAW_TO_STD = {
    'snow_depth_water_equivalent_mean': 'swe_mm',
    'surface_net_solar_radiation_mean': 'swnetrad_wm2',
    'surface_net_thermal_radiation_mean': 'lwnetrad_wm2',
    'surface_pressure_mean': 'airpres_hpa',
    'temperature_2m_mean': 'airtemp_C_mean_era5land',
    'dewpoint_temperature_2m_mean': 'dptemp_C_mean',
    'u_component_of_wind_10m_mean': 'windspeedu_mps',
    'v_component_of_wind_10m_mean': 'windspeedv_mps',
    'volumetric_soil_water_layer_1_mean': 'sml1',
    'volumetric_soil_water_layer_2_mean': 'sml2',
    'volumetric_soil_water_layer_3_mean': 'sml3',
    'volumetric_soil_water_layer_4_mean': 'sml4',
    'snow_depth_water_equivalent_min': 'swe_mm_min',
    'surface_net_solar_radiation_min': 'swnetrad_wm2_min',
    'surface_net_thermal_radiation_min': 'lwnetrad_wm2_min',
    'surface_pressure_min': 'surface_pressure_min',
    'temperature_2m_min': 'airtemp_C_era5land_min',
    'dewpoint_temperature_2m_min': 'dptemp_C_min',
    'u_component_of_wind_10m_min': 'windspeedu_mps_min',
    'v_component_of_wind_10m_min': 'windspeedv_mps_min',
    'volumetric_soil_water_layer_1_min': 'sml1_min',
    'volumetric_soil_water_layer_2_min': 'sml2_min',
    'volumetric_soil_water_layer_3_min': 'sml3_min',
    'volumetric_soil_water_layer_4_min': 'sml4_min',
    'snow_depth_water_equivalent_max': 'swe_mm_max',
    'surface_net_solar_radiation_max': 'swnetrad_wm2_max',
    'surface_net_thermal_radiation_max': 'lwnetrad_wm2_max',
    'surface_pressure_max': 'surface_pressure_max',
    'temperature_2m_max': 'airtemp_C_era5land_max',
    'dewpoint_temperature_2m_max': 'dptemp_C_max',
    'u_component_of_wind_10m_max': 'windspeedu_mps_max',
    'v_component_of_wind_10m_max': 'windspeedv_mps_max',
    'volumetric_soil_water_layer_1_max': 'sml1_max',
    'volumetric_soil_water_layer_2_max': 'sml2_max',
    'volumetric_soil_water_layer_3_max': 'sml3_max',
    'volumetric_soil_water_layer_4_max': 'sml4_max',
    'total_precipitation_sum': 'pcp_mm_era5land',
    'potential_evaporation_sum_ERA5_LAND': 'pevap_mm_era5land',
    'potential_evaporation_sum_FAO_PENMAN_MONTEITH': 'pet_mm_era5land',
    'streamflow': 'q_mm_obs',
    'temperature_2m_max_CHMI': 'airtemp_C_chmi_max',
    'temperature_2m_min_CHMI': 'airtemp_C_chmi_min',
    'temperature_2m_mean_CHMI': 'airtemp_C_mean_chmi',
    'total_precipitation_sum_CHMI': 'pcp_mm_chmi',
    'potential_evaporation_sum_FAO_PENMAN_MONTEITH_CHMI': 'pet_mm_chmi',
}
CHMI_COLS = [c for c in DAILY_RAW_TO_STD if c.endswith('_CHMI')]

HOURLY_RAW_TO_STD = {
    'streamflow': 'q_mm_obs',
    'temperature_2m_mean_CHMI': 'airtemp_C_mean_chmi',
    'total_precipitation_sum_CHMI': 'pcp_mm_chmi',
    'dewpoint_temperature_2m_ERA5_LAND': 'dptemp_C_mean',
    'potential_evaporation_ERA5_LAND': 'pevap_mm_era5land',
    'snow_depth_water_equivalent_ERA5_LAND': 'swe_mm',
    'surface_net_solar_radiation_ERA5_LAND': 'swnetrad_wm2',
    'surface_net_thermal_radiation_ERA5_LAND': 'lwnetrad_wm2',
    'surface_pressure_ERA5_LAND': 'airpres_hpa',
    'temperature_2m_ERA5_LAND': 'airtemp_C_mean_era5land',
    'total_precipitation_ERA5_LAND': 'pcp_mm_era5land',
    'u_component_of_wind_10m_ERA5_LAND': 'windspeedu_mps',
    'v_component_of_wind_10m_ERA5_LAND': 'windspeedv_mps',
    'volumetric_soil_water_layer_1_ERA5_LAND': 'sml1',
    'volumetric_soil_water_layer_2_ERA5_LAND': 'sml2',
    'volumetric_soil_water_layer_3_ERA5_LAND': 'sml3',
    'volumetric_soil_water_layer_4_ERA5_LAND': 'sml4',
}

# the only unit conversion of a dynamic feature: kPa -> hPa
RAW_FACTORS = {'surface_pressure_mean': 10.0, 'surface_pressure_ERA5_LAND': 10.0}

STATIC_RENAMED = {'area': 'area_km2', 'gauge_lat': 'lat', 'gauge_lon': 'long',
                  'ele_mt_sav': 'elev_catch_m', 'ele_mt_smn': 'elev_catch_min_m',
                  'ele_mt_smx': 'elev_catch_max_m', 'slp_dg_sav': 'slope_degrees'}
# HydroATLAS stores the slope in degrees x 10
STATIC_FACTORS = {'slope_degrees': 0.1}

t0 = time.time()
daily = CAMELS_CZ(path=CAMELS_CZ_PATH, verbosity=0)
print(f"CAMELS_CZ daily initialization took {time.time() - t0:.1f} s")
t0 = time.time()
hourly = CAMELS_CZ(path=CAMELS_CZ_PATH, timestep='H', verbosity=0)
print(f"CAMELS_CZ hourly initialization took {time.time() - t0:.1f} s")

DS_DIR = os.path.join(CAMELS_CZ_PATH, 'CAMELS_CZ')


def _rng() -> random.Random:
    """the same random choices in every run, whichever tests ran before"""
    return random.Random(313)


def _root(product: str, ds_dir: str = DS_DIR) -> str:
    return os.path.join(ds_dir, product, product)


def _ts_file(product: str, stn: str) -> str:
    return os.path.join(_root(product), 'timeseries', 'csv', 'camelscz', f'camelscz_{stn}.csv')


def _attr_file(product: str, kind: str, ds_dir: str = DS_DIR) -> str:
    return os.path.join(_root(product, ds_dir), 'attributes', 'camelscz', f'attributes_{kind}_camelscz.csv')


# ---------------------------------------------------------------------------
# independent raw readers (exact float parser, float64)
# ---------------------------------------------------------------------------

def _read_raw(product: str, stn: str) -> pd.DataFrame:
    tcol = 'time' if product == HOURLY else 'date'
    return pd.read_csv(_ts_file(product, stn), index_col=tcol, parse_dates=True,
                       float_precision='round_trip')


def raw_daily(stn: str) -> pd.DataFrame:
    """the Caravan extension with the CHMI columns of CAMELS-CZ.zip, raw names"""
    car = _read_raw(CARAVAN, stn)
    chmi = _read_raw(CAMELS, stn)[CHMI_COLS]
    assert chmi.index.isin(car.index).all(), f"{stn}: CAMELS-CZ.zip has days the Caravan extension lacks"
    return car.join(chmi)


def raw_hourly(stn: str) -> pd.DataFrame:
    return _read_raw(HOURLY, stn)


def _float_columns(product: str, kind: str):
    """numeric columns of an attribute file, parsed below with Python's exact
    float() instead of the parser the class uses"""
    head = pd.read_csv(_attr_file(product, kind), nrows=5)
    return [c for c in head.columns if head[c].dtype.kind == 'f']


def raw_static(product: str) -> pd.DataFrame:
    dfs = []
    for kind in ('other', 'caravan', 'hydroatlas'):
        df = pd.read_csv(_attr_file(product, kind), dtype={'gauge_id': str, 'gauge_name': str},
                         converters={c: float for c in _float_columns(product, kind)})
        df.index = pd.Index(df.pop('gauge_id').str.replace('camelscz_', '', regex=False), name='gauge_id')
        dfs.append(df)
    return pd.concat(dfs, axis=1)


def expected(raw: pd.DataFrame, raw_col: str, fp) -> np.ndarray:
    """raw values after the documented conversion, cast to the class precision"""
    return (raw[raw_col] * RAW_FACTORS.get(raw_col, 1.0)).to_numpy(dtype=fp)


def _basin_areas():
    """BasinArea (m2) of the boundary shapefile, read with fiona"""
    shp = os.path.join(_root(HOURLY), 'shapefiles', 'camelscz', 'camelscz_basin_shapes')
    if fiona is None:
        # not independent of the class, but still checks the conversion
        dbf = _cz._read_dbf(f"{shp}.dbf", ('gauge_id', 'BasinArea'))
        pairs = zip(dbf['gauge_id'], dbf['BasinArea'])
    else:
        with fiona.open(f"{shp}.shp") as src:
            pairs = [(f['properties']['gauge_id'], f['properties']['BasinArea']) for f in src]
    return {gid[9:]: int(area) / 1e6 for gid, area in pairs}


# the area (km2) the authors divided the CHMI discharge by: ``area`` of the
# attributes for daily data, ``BasinArea`` of the boundary shapefile for hourly
# data (both verified against CHMI's open data)
Q_AREA = {'D': raw_static(CAMELS)['area'].to_dict(), 'H': _basin_areas()}


def assert_frame_matches_raw(ds, stn: str, df: pd.DataFrame, raw: pd.DataFrame):
    """every served column equals its raw column (cast to the class precision),
    q_cms_obs equals q_mm_obs * area / seconds, and the rows after the end of
    the file (if any) are NaN padding at the dataset's time step"""
    mapping = HOURLY_RAW_TO_STD if ds.timestep == 'H' else DAILY_RAW_TO_STD
    freq = pd.Timedelta('1h') if ds.timestep == 'H' else pd.Timedelta('1D')
    assert df.index[:len(raw)].equals(raw.index), f"{stn}: time index changed"
    assert df.iloc[len(raw):].isna().all().all(), f"{stn}: padding is not NaN"
    assert (np.diff(df.index[len(raw) - 1:].values) == freq).all(), f"{stn}: padding is irregular"
    df = df.iloc[:len(raw)]
    for raw_c, std_c in mapping.items():
        assert np.array_equal(df[std_c].to_numpy(), expected(raw, raw_c, ds.fp), equal_nan=True), \
            f"{stn}:{std_c} values changed"
    want = (raw['streamflow'] * Q_AREA[ds.timestep][stn] * 1e3 / freq.total_seconds()).to_numpy(dtype=ds.fp)
    assert np.array_equal(df['q_cms_obs'].to_numpy(), want, equal_nan=True), f"{stn}: q_cms_obs"


# ---------------------------------------------------------------------------
# stubs and small copies of the archives
# ---------------------------------------------------------------------------

def _raise(*args, **kwargs):
    raise AssertionError("must not be called")


@contextlib.contextmanager
def _patched(download=_raise, extractall=None, to_netcdf=None):
    """replaces the download (wherever the class may look it up), and
    optionally the extraction and the netCDF writer"""
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(download_zenodo, 'download_from_zenodo', download))
        stack.enter_context(mock.patch.object(_cz, 'download_from_zenodo', download, create=True))
        if extractall is not None:
            stack.enter_context(mock.patch.object(zipfile.ZipFile, 'extractall', extractall))
        if to_netcdf is not None:
            stack.enter_context(mock.patch.object(xr.Dataset, 'to_netcdf', to_netcdf))
        yield


MINI_STATIONS = ['B4403000', '24042409']


@functools.lru_cache(maxsize=None)
def _mini_archives() -> dict:
    """product -> path of a small copy of ``<product>.zip``: the files of
    MINI_STATIONS (and the unserved gauge in the attributes), the real .dbf
    (it holds BasinArea), empty other boundary files and a netCDF copy that
    must not be extracted. Built once per session."""
    outdir = tempfile.mkdtemp()
    atexit.register(shutil.rmtree, outdir, True)
    paths = {}
    for product in (CARAVAN, CAMELS, HOURLY):
        attr_src = CAMELS if product == CARAVAN else product
        paths[product] = os.path.join(outdir, f"{product}.zip")
        with zipfile.ZipFile(paths[product], 'w') as zf:
            for kind in ('other', 'caravan', 'hydroatlas'):
                df = pd.read_csv(_attr_file(attr_src, kind), dtype=str)
                keep = df['gauge_id'].str[9:].isin(MINI_STATIONS + UNSERVED)
                zf.writestr(f"{product}/attributes/camelscz/attributes_{kind}_camelscz.csv",
                            df[keep].to_csv(index=False))
            for stn in MINI_STATIONS:
                zf.write(_ts_file(product, stn), f"{product}/timeseries/csv/camelscz/camelscz_{stn}.csv")
                zf.writestr(f"{product}/timeseries/netcdf/camelscz/camelscz_{stn}.nc", b"not extracted")
            shp = os.path.join(_root(attr_src), 'shapefiles', 'camelscz', 'camelscz_basin_shapes')
            zf.write(f"{shp}.dbf", f"{product}/shapefiles/camelscz/camelscz_basin_shapes.dbf")
            for ext in ('.shp', '.shx', '.prj'):
                zf.writestr(f"{product}/shapefiles/camelscz/camelscz_basin_shapes{ext}", b"")
    return paths


@contextlib.contextmanager
def _mini(products=()):
    """a temporary ``path`` whose CAMELS_CZ folder holds the mini archives of
    ``products``. A download is answered with the mini archives and recorded
    in ``calls``; it fails if the archive it would write is already there,
    because ``download()`` would then write ``<name>.zip1``."""
    tmp = tempfile.mkdtemp()
    ds_dir = os.path.join(tmp, 'CAMELS_CZ')
    os.makedirs(ds_dir)
    minis = _mini_archives()
    for product in products:
        shutil.copy(minis[product], ds_dir)
    calls = []

    def fake_download(outdir, doi, include=None, **kwargs):
        calls.append((doi, sorted(include)))
        for fname in include:
            target = os.path.join(outdir, fname)
            assert not os.path.exists(target), f"stale {fname} not removed before download"
            shutil.copy(minis[fname[:-4]], target)

    try:
        with _patched(download=fake_download):
            yield tmp, ds_dir, calls
    finally:
        shutil.rmtree(tmp)


@contextlib.contextmanager
def _linked(products):
    """a temporary ``path`` whose CAMELS_CZ folder links to the extracted
    folders of ``products`` only, without any archive"""
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, 'CAMELS_CZ'))
    for product in products:
        os.symlink(os.path.join(DS_DIR, product), os.path.join(tmp, 'CAMELS_CZ', product))
    try:
        yield tmp
    finally:
        shutil.rmtree(tmp)  # removes the symlinks, not their targets


def _messages(w, word):
    return [str(x.message) for x in w if word in str(x.message)]


# ---------------------------------------------------------------------------
# registration, features, stations
# ---------------------------------------------------------------------------

def test_registration():
    logger.info("test_registration")
    assert 'CAMELS_CZ' in aqua_fetch.ALL_DATASETS
    from aqua_fetch import RainfallRunoff
    # RainfallRunoff defaults to remove_zip=True, which deletes the archives;
    # keep them so that the later tests can use them
    archives = [os.path.join(DS_DIR, f"{p}.zip") for p in (CARAVAN, CAMELS, HOURLY)]
    present = [a for a in archives if os.path.exists(a)]
    for timestep in ('D', 'H'):
        rr = RainfallRunoff('CAMELS_CZ', path=CAMELS_CZ_PATH, timestep=timestep,
                            verbosity=0, remove_zip=False)
        assert len(rr.stations()) == NUM_STATIONS
    assert all(os.path.exists(a) for a in present), "remove_zip=False deleted an archive"


def test_invalid_timestep():
    logger.info("test_invalid_timestep")
    for bad in ('d', 'h', '15min', 'M'):
        with pytest.raises(ValueError):
            CAMELS_CZ(path=CAMELS_CZ_PATH, timestep=bad, verbosity=0)


def test_feature_names():
    logger.info("test_feature_names")
    stn = daily.stations()[0]
    car = pd.read_csv(_ts_file(CARAVAN, stn), nrows=0).columns[1:].tolist()
    assert daily.dynamic_features == [DAILY_RAW_TO_STD[c] for c in car + CHMI_COLS] + ['q_cms_obs']
    hrs = pd.read_csv(_ts_file(HOURLY, stn), nrows=0).columns[1:].tolist()
    assert hourly.dynamic_features == [HOURLY_RAW_TO_STD[c] for c in hrs] + ['q_cms_obs']
    for ds, product in ((daily, CAMELS), (hourly, HOURLY)):
        assert len(ds.dynamic_features) == len(set(ds.dynamic_features))
        raw = raw_static(product)
        assert ds.static_features == [STATIC_RENAMED.get(c, c) for c in raw.columns]
        assert len(ds.static_features) == len(set(ds.static_features)) == NUM_STATIC


def test_cache_name():
    """the default float32 cache has no precision in its name, as in the other
    classes; another precision gets its own cache"""
    logger.info("test_cache_name")
    assert daily.dyn_fname == 'camels_cz_D_v2.nc' and hourly.dyn_fname == 'camels_cz_H_v2.nc'
    ds = CAMELS_CZ(path=CAMELS_CZ_PATH, verbosity=0, to_netcdf=False, float_precision=np.float64)
    assert ds.dyn_fname == 'camels_cz_D_float64_v2.nc'


def test_returns_copies():
    logger.info("test_returns_copies")
    for ds in (daily, hourly):
        for getter in (ds.stations, lambda: ds.dynamic_features, lambda: ds.static_features):
            items = getter()
            items.append('XXX')
            assert 'XXX' not in getter()
        df = ds._static_data()
        df.iloc[0, 0] = -1
        df.drop(columns=df.columns[1], inplace=True)
        assert ds._static_data().shape == (NUM_STATIONS, NUM_STATIC)
        assert ds._static_data().iloc[0, 0] != -1
        stns = ds.stations()
        stns.sort(reverse=True)
        assert ds.stations() != stns
        ds.dyn_map['streamflow'] = 'XXX'
        assert ds.dyn_map['streamflow'] == 'q_mm_obs'


def test_unserved_gauge_warns():
    logger.info("test_unserved_gauge_warns")
    other = pd.read_csv(_attr_file(CAMELS, 'other'))['gauge_id'].str[9:].tolist()
    assert sorted(set(other) - set(daily.stations())) == UNSERVED
    assert not os.path.exists(_ts_file(CAMELS, UNSERVED[0]))
    assert not os.path.exists(_ts_file(CARAVAN, UNSERVED[0]))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        CAMELS_CZ(path=CAMELS_CZ_PATH, verbosity=0)
    msgs = _messages(w, 'not served')
    assert len(msgs) == 1 and UNSERVED[0] in msgs[0], msgs
    # the hourly archive does not list it at all
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        CAMELS_CZ(path=CAMELS_CZ_PATH, timestep='H', verbosity=0)
    assert not _messages(w, 'not served')


def test_missing_file_warns():
    logger.info("test_missing_file_warns")
    stn = daily.stations()[5]
    orig = _cz._cz_fname
    with mock.patch.object(_cz, '_cz_fname', lambda s: orig(s) + ("x" if s == stn else "")):
        for timestep, n in (('D', 2), ('H', 1)):
            # at initialization and when called directly
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ds = CAMELS_CZ(path=CAMELS_CZ_PATH, timestep=timestep, verbosity=0)
                ds._check_manifest()
            msgs = _messages(w, 'missing')
            assert len(msgs) == 2 and all(f'{n} files are missing' in m for m in msgs), (timestep, msgs)

    # a missing boundary shapefile (and its .dbf/.shx/.prj siblings) warns; a
    # missing station list raises
    with mock.patch.object(CAMELS_CZ, 'boundary_file',
                           property(lambda self: os.path.join(DS_DIR, 'nowhere', 'x.shp'))):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            daily._check_manifest()
    msgs = _messages(w, 'missing')
    assert len(msgs) == 1 and '4 files are missing' in msgs[0], msgs

    orig_attr = CAMELS_CZ._attr_path
    with mock.patch.object(CAMELS_CZ, '_attr_path',
                           lambda self, kind: orig_attr(self, kind) + ('x' if kind == 'caravan' else '')):
        with pytest.raises(FileNotFoundError):
            daily._check_manifest()


def test_each_timestep_needs_only_its_own_archives():
    """daily reads only the two daily archives and hourly only the hourly one,
    including the attributes and boundaries"""
    logger.info("test_each_timestep_needs_only_its_own_archives")
    for timestep, products in (('D', (CARAVAN, CAMELS)), ('H', (HOURLY,))):
        with _linked(products) as tmp, _patched(extractall=_raise):
            ds = CAMELS_CZ(path=tmp, timestep=timestep, verbosity=0, to_netcdf=False)
            assert len(ds.stations()) == NUM_STATIONS
            assert ds.fetch_static_features('all', 'all').shape == (NUM_STATIONS, NUM_STATIC)
            stn = ds.stations()[0]
            assert len(ds._read_stn_dyn(stn)) == (HOURLY_LEN if timestep == 'H' else DAILY_LEN)
            if fiona is not None:
                assert ds.get_boundary(stn)['type'] in ('Polygon', 'MultiPolygon')


# ---------------------------------------------------------------------------
# dynamic data
# ---------------------------------------------------------------------------

def _raw_in_worker(args):
    timestep, stn = args
    raw = raw_hourly(stn) if timestep == 'H' else raw_daily(stn)
    vals = np.abs(raw.to_numpy('float64'))
    vals = vals[np.isfinite(vals) & (vals != 0)]
    # every value is well inside the float32 range: the cast only rounds
    in_range = bool((vals < np.finfo(np.float32).max).all() and (vals > np.finfo(np.float32).tiny).all())
    return stn, raw, in_range


def _check_all_stations(ds, length):
    """every station, from the csv files and (if built) from the netCDF cache"""
    with cf.ProcessPoolExecutor(16) as ex:
        res = {stn: (raw, ok) for stn, raw, ok in ex.map(_raw_in_worker, [(ds.timestep, s) for s in ds.stations()])}

    t = time.time()
    from_csv = ds._read_dynamic(ds.stations(), 'all')
    t_csv = time.time() - t
    sources = {'csv': from_csv}
    if ds.dyn_fpath_exists and xr is not None:
        t = time.time()
        _, sources['netCDF'] = ds.fetch(as_dataframe=True)
        print(f"{ds.timestep}: all stations in {time.time() - t:.2f} s from netCDF, {t_csv:.2f} s from csv")
    else:
        print(f"{ds.timestep}: all stations in {t_csv:.2f} s from csv")

    mapping = HOURLY_RAW_TO_STD if ds.timestep == 'H' else DAILY_RAW_TO_STD
    freq = pd.Timedelta('1h') if ds.timestep == 'H' else pd.Timedelta('1D')
    starts, ends = set(), set()
    for stn, (raw, in_range) in res.items():
        starts.add(raw.index.min())
        ends.add(raw.index.max())
        assert in_range, stn
        # every gauge has exactly the expected columns
        assert set(raw.columns) == set(mapping), (stn, set(raw.columns) ^ set(mapping))
        # neither missing nor duplicated time steps in the source
        assert raw.index.is_unique and (np.diff(raw.index.values) == freq).all(), stn
        for name, dyn in sources.items():
            assert len(dyn[stn]) == length, (name, stn)
            assert dyn[stn].dtypes.eq(ds.fp).all(), (name, stn)
            assert_frame_matches_raw(ds, stn, dyn[stn], raw)
    assert min(starts) == ds.start and max(ends) == ds.end, (starts, ends)
    if ds.timestep == 'D':
        assert len(starts) == len(ends) == 1, (starts, ends)


def test_all_stations_fidelity_daily():
    logger.info("test_all_stations_fidelity_daily")
    _check_all_stations(daily, DAILY_LEN)


def test_all_stations_fidelity_hourly():
    logger.info("test_all_stations_fidelity_hourly")
    _check_all_stations(hourly, HOURLY_LEN)


def test_float64_values_are_exact():
    """with float_precision=np.float64 the csv values are parsed exactly (the
    default parser is off by one ulp in many of them)"""
    logger.info("test_float64_values_are_exact")
    for timestep, reader in (('D', raw_daily), ('H', raw_hourly)):
        ds = CAMELS_CZ(path=CAMELS_CZ_PATH, timestep=timestep, verbosity=0, to_netcdf=False,
                       float_precision=np.float64)
        stn = 'B4403000'
        df = ds._read_dynamic([stn], 'all')[stn]
        assert df.dtypes.eq(np.float64).all()
        assert_frame_matches_raw(ds, stn, df, reader(stn))


def test_units():
    """the one converted unit: the raw surface pressure is in kPa, so
    airpres_hpa is in hPa, at every gauge"""
    logger.info("test_units")
    _, dyn = daily.fetch(dynamic_features=['airpres_hpa', 'surface_pressure_min'], as_dataframe=True)
    for stn, df in dyn.items():
        df = df.astype('float64')
        # the catchments lie at ~250-1100 m on average
        assert 880 < df['airpres_hpa'].mean() < 1000, stn
        assert 88 < df['surface_pressure_min'].mean() < 100, stn  # kPa


def test_mm_to_cms_matches_served():
    """the public conversion helpers use the same areas as q_cms_obs"""
    logger.info("test_mm_to_cms_matches_served")
    for ds in (daily, hourly):
        stn = _rng().choice(ds.stations())
        df = ds._read_dynamic([stn], ['q_mm_obs', 'q_cms_obs'])[stn].astype('float64')
        q_mm = df['q_mm_obs'].rename(stn)
        assert np.allclose(ds.mm_to_cms(q_mm), df['q_cms_obs'], rtol=1e-6, equal_nan=True)
        assert np.allclose(ds.cms_to_mm(ds.mm_to_cms(q_mm)), q_mm, rtol=1e-12, equal_nan=True)


def test_reader_spec_reads_only_needed_columns():
    logger.info("test_reader_spec_reads_only_needed_columns")
    car_dir, cz_dir = (os.path.dirname(_ts_file(p, 'x')) for p in (CARAVAN, CAMELS))
    spec = daily._reader_spec(['pcp_mm_chmi'])
    # the Caravan extension is always read, at least its dates
    assert spec['files'] == [(car_dir, ['date']), (cz_dir, ['date', 'total_precipitation_sum_CHMI'])]
    spec = daily._reader_spec(['q_cms_obs'])
    assert spec['files'] == [(car_dir, ['date', 'streamflow'])]

    stn = _rng().choice(daily.stations())
    raw = raw_daily(stn)
    # a CHMI-only request has the same time index as a full one
    df = daily._read_dynamic([stn], ['pcp_mm_chmi'])[stn]
    assert df.index.equals(raw.index)
    assert np.array_equal(df['pcp_mm_chmi'].to_numpy(), expected(raw, 'total_precipitation_sum_CHMI', daily.fp),
                          equal_nan=True)


class _RecordingPool(cf.ProcessPoolExecutor):
    used = 0

    def __init__(self, *args, **kwargs):
        _RecordingPool.used += 1
        super().__init__(*args, **kwargs)


def test_process_pool_use():
    """one station is read without a pool, and processes=1 never uses one,
    even for a read that otherwise does"""
    logger.info("test_process_pool_use")
    serial = CAMELS_CZ(path=CAMELS_CZ_PATH, verbosity=0, processes=1)
    _RecordingPool.used = 0
    with mock.patch.object(_cz.cf, 'ProcessPoolExecutor', _RecordingPool):
        daily._read_dynamic(daily.stations()[:1], 'all')
        assert _RecordingPool.used == 0, "one station needs no pool"
        # 15 stations are ~170 MB of csv, enough for a pool with any start method
        assert len(serial._read_dynamic(serial.stations()[:15], 'all')) == 15
        assert _RecordingPool.used == 0, "processes=1 must not use a pool"
        assert len(daily._read_dynamic(daily.stations()[:15], 'all')) == 15
        assert _RecordingPool.used == 1, "the same read without processes=1 uses a pool"


def test_empty_requests():
    logger.info("test_empty_requests")
    with pytest.raises(ValueError):
        daily._read_dynamic(daily.stations()[:1], [])
    assert daily._read_dynamic([], 'all') == {}


def test_fetch_outside_data_range():
    logger.info("test_fetch_outside_data_range")
    stn = daily.stations()[0]
    _, before = daily.fetch(stn, st='1900-01-01', en='1940-12-31', as_dataframe=True)
    assert len(before[stn]) == 0
    _, inside = daily.fetch(stn, st='2010-01-01', en='2010-12-31', as_dataframe=True)
    assert len(inside[stn]) == 365
    _, day = hourly.fetch(stn, st='2020-01-01', en='2020-01-01 23:00', as_dataframe=True)
    assert len(day[stn]) == 24


def test_invalid_station():
    logger.info("test_invalid_station")
    for ds in (daily, hourly):
        for bad in ('X0000000', [ds.stations()[0], 'X0000000'], UNSERVED[0]):
            with pytest.raises((ValueError, AssertionError, KeyError)):
                ds.fetch(bad, as_dataframe=True)


# ---------------------------------------------------------------------------
# static data, boundaries, overlaps
# ---------------------------------------------------------------------------

def test_dbf_reader_matches_fiona():
    logger.info("test_dbf_reader_matches_fiona")
    if fiona is None:
        pytest.skip("fiona is not installed")
    shp = hourly.boundary_file
    with fiona.open(shp) as src:
        props = [dict(f['properties']) for f in src]
    cols = tuple(props[0])
    dbf = _cz._read_dbf(f"{os.path.splitext(shp)[0]}.dbf", cols)
    assert len(props) == len(dbf['gauge_id']) == NUM_STATIONS
    for col in cols:
        assert dbf[col] == [str(p[col]) for p in props], col


def test_static_fidelity():
    logger.info("test_static_fidelity")
    for ds, product in ((daily, CAMELS), (hourly, HOURLY)):
        raw = raw_static(product).loc[ds.stations()].rename(columns=STATIC_RENAMED)
        for col, factor in STATIC_FACTORS.items():
            raw[col] = raw[col] * factor
        got = ds.fetch_static_features('all', 'all')
        pd.testing.assert_frame_equal(got, raw, check_exact=True)
    # the hourly archive's attributes are those of the daily one (up to ~6e-14)
    pd.testing.assert_frame_equal(hourly._static_data(), daily._static_data().loc[hourly.stations()])


def test_coords_area_boundary():
    logger.info("test_coords_area_boundary")
    other = raw_static(CAMELS).loc[daily.stations()]
    coords = daily.stn_coords()
    assert np.array_equal(coords['lat'].to_numpy(), other['gauge_lat'].to_numpy(dtype=daily.fp))
    assert np.array_equal(coords['long'].to_numpy(), other['gauge_lon'].to_numpy(dtype=daily.fp))
    assert np.array_equal(daily.area().to_numpy(), other['area'].to_numpy(dtype=daily.fp))
    if fiona is None:
        pytest.skip("fiona is not installed")
    for ds in (daily, hourly):
        ids = set(ds._create_boundary_id_map().keys())
        assert set(ds.stations()) <= ids, set(ds.stations()) - ids
        # each boundary belongs to its own gauge
        rr_utils.test_boundaries_around_gauges(ds)


def _edit_mini_attrs(ds_dir, product, kind, edit):
    fpath = _attr_file(product, kind, ds_dir)
    edit(pd.read_csv(fpath, dtype=str)).to_csv(fpath, index=False)


def test_duplicate_gauges_are_reported_at_init():
    """the source has no gauge names (gauge_name is the id), so a gauge with
    another id at the same place must still be reported, and served"""
    logger.info("test_duplicate_gauges_are_reported_at_init")
    src, dup = 'B4403000', 'X0000000'
    with _mini((CARAVAN, CAMELS)) as (tmp, ds_dir, _):
        CAMELS_CZ(path=tmp, verbosity=0, to_netcdf=False)       # extracts
        for product in (CARAVAN, CAMELS):
            for kind in ('other', 'caravan', 'hydroatlas'):
                _edit_mini_attrs(ds_dir, product, kind, lambda df: pd.concat(
                    [df, df[df['gauge_id'] == f'camelscz_{src}'].assign(gauge_id=f'camelscz_{dup}')]))
            folder = os.path.join(_root(product, ds_dir), 'timeseries', 'csv', 'camelscz')
            shutil.copy(os.path.join(folder, f'camelscz_{src}.csv'), os.path.join(folder, f'camelscz_{dup}.csv'))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ds = CAMELS_CZ(path=tmp, verbosity=0, to_netcdf=False)
        msgs = _messages(w, 'duplicate')
        assert len(msgs) == 1 and src in msgs[0] and dup in msgs[0], msgs
        assert dup in ds.stations()


# ---------------------------------------------------------------------------
# download, extraction, overwrite, cache
# ---------------------------------------------------------------------------

def test_no_redownload_or_reextract():
    logger.info("test_no_redownload_or_reextract")
    with _patched(extractall=_raise, to_netcdf=_raise):
        for timestep in ('D', 'H'):
            t = time.time()
            ds = CAMELS_CZ(path=CAMELS_CZ_PATH, timestep=timestep, verbosity=0)
            print(f"{timestep}: initialization with the data on disk took {time.time() - t:.2f} s")
            assert len(ds.stations()) == NUM_STATIONS


def test_interrupted_extraction_is_redone():
    """the first daily init downloads both archives in one call; an extraction
    interrupted there is not taken as complete, and the next init extracts
    (without downloading), skips the netCDF copies and serves the raw values"""
    logger.info("test_interrupted_extraction_is_redone")
    orig_ext = zipfile.ZipFile.extractall

    def extract_then_fail(zf, path=None, members=None, pwd=None):
        for m in members[:3]:
            zf.extract(m, path)
        raise KeyboardInterrupt

    with _mini() as (tmp, ds_dir, calls):
        with mock.patch.object(zipfile.ZipFile, 'extractall', extract_then_fail):
            with pytest.raises(KeyboardInterrupt):
                CAMELS_CZ(path=tmp, verbosity=0, to_netcdf=False)
        assert calls == [(daily.url, [f"{CAMELS}.zip", f"{CARAVAN}.zip"])]
        assert not os.path.exists(os.path.join(ds_dir, CARAVAN)), "a partial extraction looks complete"

        assert zipfile.ZipFile.extractall is orig_ext
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ds = CAMELS_CZ(path=tmp, verbosity=0, to_netcdf=False)
        assert len(calls) == 1
        assert ds.stations() == [s for s in daily.stations() if s in MINI_STATIONS]
        assert any(UNSERVED[0] in m for m in _messages(w, 'not served'))
        for product in (CARAVAN, CAMELS):
            assert os.path.isdir(os.path.join(_root(product, ds_dir), 'timeseries', 'csv'))
            assert not os.path.exists(os.path.join(_root(product, ds_dir), 'timeseries', 'netcdf'))
            assert not os.path.exists(os.path.join(ds_dir, f"{product}_extracting"))
        for stn in MINI_STATIONS:
            assert_frame_matches_raw(daily, stn, ds._read_stn_dyn(stn), raw_daily(stn))


def test_remove_zip_deletes_only_this_timesteps_archives():
    logger.info("test_remove_zip_deletes_only_this_timesteps_archives")
    with _mini((CARAVAN, CAMELS, HOURLY)) as (tmp, ds_dir, calls):
        CAMELS_CZ(path=tmp, verbosity=0, to_netcdf=False, remove_zip=True)
        assert sorted(f for f in os.listdir(ds_dir) if f.endswith('.zip')) == [f"{HOURLY}.zip"]
        # without its archives, the next daily init neither downloads nor extracts
        with _patched(extractall=_raise):
            CAMELS_CZ(path=tmp, verbosity=0, to_netcdf=False)
        assert calls == []


def test_overwrite_replaces_stale_data_of_its_timestep_only():
    """overwrite=True removes the archives, extracted folders and cache of its
    timestep before downloading, and leaves those of the other timestep"""
    logger.info("test_overwrite_replaces_stale_data_of_its_timestep_only")
    with _mini((CARAVAN, CAMELS, HOURLY)) as (tmp, ds_dir, calls):
        ds = CAMELS_CZ(path=tmp, verbosity=0, to_netcdf=False)
        CAMELS_CZ(path=tmp, timestep='H', verbosity=0, to_netcdf=False)
        for product in (CARAVAN, CAMELS):
            with open(os.path.join(ds_dir, f"{product}.zip"), 'w') as fp:
                fp.write('stale')
            with open(os.path.join(_root(product, ds_dir), 'stale.txt'), 'w') as fp:
                fp.write('stale')
        h_cache = os.path.join(ds_dir, hourly.dyn_fname)
        for cache in (ds.dyn_fpath, h_cache):
            with open(cache, 'w') as fp:
                fp.write('stale')

        ds = CAMELS_CZ(path=tmp, verbosity=0, to_netcdf=False, overwrite=True)
        assert calls == [(daily.url, [f"{CAMELS}.zip", f"{CARAVAN}.zip"])]
        assert not os.path.exists(ds.dyn_fpath)
        for product in (CARAVAN, CAMELS):
            assert not os.path.exists(os.path.join(_root(product, ds_dir), 'stale.txt'))
        stn = MINI_STATIONS[1]
        assert_frame_matches_raw(daily, stn, ds._read_stn_dyn(stn), raw_daily(stn))
        # the hourly data is left alone
        assert os.path.exists(h_cache), "overwrite of D deleted the H cache"
        assert os.path.exists(os.path.join(ds_dir, f"{HOURLY}.zip"))
        assert os.path.isdir(os.path.join(_root(HOURLY, ds_dir), 'timeseries'))


# ---------------------------------------------------------------------------
# the suite shared by all rainfall-runoff datasets
# ---------------------------------------------------------------------------

def test_standard_suite_daily():
    logger.info("test_standard_suite_daily")
    random.seed(313)  # the shared suite draws from the global random state
    rr_utils.test_dataset(daily,
                          num_stations=NUM_STATIONS,
                          dyn_data_len=DAILY_LEN,
                          num_static_attrs=NUM_STATIC,
                          num_dyn_attrs=len(DAILY_RAW_TO_STD) + 1,
                          yearly_steps=366,
                          st="20040101", en="20041231")


def test_standard_suite_hourly():
    logger.info("test_standard_suite_hourly")
    random.seed(313)
    rr_utils.test_dataset(hourly,
                          num_stations=NUM_STATIONS,
                          dyn_data_len=HOURLY_LEN,
                          num_static_attrs=NUM_STATIC,
                          num_dyn_attrs=len(HOURLY_RAW_TO_STD) + 1,
                          yearly_steps=366 * 24,
                          st="20200101", en="20201231 23:00")


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith('test_') and callable(func):
            t = time.time()
            try:
                func()
            except pytest.skip.Exception as e:
                print(f"{name} skipped: {e}")
                continue
            print(f"{name} passed in {time.time() - t:.1f} s")
    print("*** All CAMELS_CZ tests passed ***")
