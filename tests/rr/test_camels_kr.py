"""
Tests for the CAMELS_KR (South Korea, 282 catchments) dataset.

They check that the class

    * downloads and extracts the data once, skips the model output, and never
      re-downloads or re-extracts data that is already on disk,
    * returns every dynamic and static value unchanged (compared with an
      independent read of the raw csv files) apart from the documented
      MJ m-2 day-1 -> W m-2 conversion of solar radiation,
    * reports feature lists, stations and the temporal extent of the data,
    * reads efficiently (process pool only for large reads, never with
      ``processes=1``).

The first run downloads ~360 MB into ``CAMELS_KR_PATH``.
"""

import os
import sys
import time
import shutil
import random
import logging
import tempfile
import warnings
import concurrent.futures as cf

import numpy as np
import pandas as pd
import pytest

# repository root and this folder (for the shared ``utils`` helpers)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    logging.basicConfig(filename='test_camels_kr.log', filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

import aqua_fetch
import aqua_fetch.download_zenodo as download_zenodo
from aqua_fetch import CAMELS_KR
from aqua_fetch.rr import _camels
from aqua_fetch.rr import utils as base_utils
from aqua_fetch._backend import xarray as xr, fiona

import utils as rr_utils

# the same random stations in every run, with pytest or as a script
random.seed(313)

# The class appends ``CAMELS_KR`` to this path. Replace it with the location on
# your machine.
CAMELS_KR_PATH = '/mnt/hyexnas01/abbaa0a/data/gscad_database/raw/CAMELS/CAMELS'

NUM_STATIONS = 282   # Lee et al., 2026
DYN_LEN = 16436      # daily steps 1981-01-01 .. 2025-12-31

HYDRO = 'Hydrological'
METEO = 'Meteorological'

# raw column -> name served by the class. Written out here (not imported) so
# that a wrong mapping in the class is caught.
RAW_TO_STD = {
    'discharge_vol': 'q_cms_obs', 'discharge_spec': 'q_mm_obs', 'water_level': 'wl_m_obs',
    'prec': 'pcp_mm', 'temp_min': 'airtemp_C_min', 'temp_max': 'airtemp_C_max',
    'temp_avg': 'airtemp_C_mean', 'rel_hum': 'rh_%', 'wind_speed': 'windspeed_mps',
    'wind_speed_max': 'windgust_mps_max', 'solar_radiation': 'swdownrad_wm2',
    'pet': 'pet_mm', 'pet_gleam': 'pet_mm_gleam', 'aet_gleam': 'aet_mm_gleam',
}
# the only unit conversion: MJ m-2 day-1 -> W m-2
RAW_FACTORS = {'solar_radiation': 1e6 / 86400.0}

STATIC_RENAMED = {
    'basin_area': 'area_km2', 'gauge_lat': 'lat', 'gauge_lon': 'long',
    'gauge_elev': 'elev_gauge_m', 'dens_2000': 'pop_density_2000_km2',
    'dens_2010': 'pop_density_2010_km2', 'dens_2020': 'pop_density_2020_km2',
    'dens_2024': 'pop_density_2024_km2',
}

ATTR_FILES = ["CAMELS_KR_location_attributes.csv", "CAMELS_KR_topography_attributes.csv",
              "CAMELS_KR_climate_attributes.csv", "CAMELS_KR_hydrology_attributes.csv",
              "CAMELS_KR_land cover_attributes.csv", "CAMELS_KR_soil_attributes.csv",
              "CAMELS_KR_human influence_attributes.csv"]

MODEL_OUTPUT = ("Simulated hydrological time series", "HBV_model_parameters")

t0 = time.time()
dataset = CAMELS_KR(path=CAMELS_KR_PATH, verbosity=0)
print(f"CAMELS_KR initialization took {time.time() - t0:.1f} s")

ROOT = os.path.join(CAMELS_KR_PATH, 'CAMELS_KR', 'CAMELS-KR')
ARCHIVE = os.path.join(CAMELS_KR_PATH, 'CAMELS_KR', 'CAMELS-KR.zip')


# ---------------------------------------------------------------------------
# independent raw readers
# ---------------------------------------------------------------------------

def _raw_file(kind: str, stn: str) -> str:
    return os.path.join(ROOT, f"{kind} time series", f"CAMELS_KR_{kind}_timeseries_{stn}.csv")


def raw_ts(stn: str) -> pd.DataFrame:
    """both raw time-series files of a station, float64, raw column names"""
    h = pd.read_csv(_raw_file(HYDRO, stn), index_col='date', parse_dates=True)
    m = pd.read_csv(_raw_file(METEO, stn), index_col='date', parse_dates=True)
    assert h.index.equals(m.index), f"{stn}: hydrological and meteorological dates differ"
    return pd.concat([h, m], axis=1)


def expected(raw: pd.DataFrame, raw_col: str) -> np.ndarray:
    """raw values after the documented conversion, cast to the class precision"""
    return (raw[raw_col] * RAW_FACTORS.get(raw_col, 1.0)).to_numpy(dtype=dataset.fp)


def raw_static() -> pd.DataFrame:
    dfs = [pd.read_csv(os.path.join(ROOT, f), dtype={'gauge_id': str}, encoding='utf-8-sig'
                       ).set_index('gauge_id') for f in ATTR_FILES]
    return pd.concat(dfs, axis=1)


def assert_frame_matches_raw(stn: str, df: pd.DataFrame, raw: pd.DataFrame):
    assert df.index.equals(raw.index), f"{stn}: time index changed"
    for raw_c, std_c in RAW_TO_STD.items():
        got = df[std_c].to_numpy()
        assert np.array_equal(np.isnan(got), raw[raw_c].isna().to_numpy()), \
            f"{stn}:{std_c} NaN pattern changed"
        assert np.array_equal(got, expected(raw, raw_c), equal_nan=True), \
            f"{stn}:{std_c} values changed"


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_registration():
    logger.info("test_registration")
    assert 'CAMELS_KR' in aqua_fetch.ALL_DATASETS
    from aqua_fetch import RainfallRunoff
    # RainfallRunoff defaults to remove_zip=True, which deletes the archive
    # (honoured on every init); keep it so the later tests can use it
    rr = RainfallRunoff('CAMELS_KR', path=CAMELS_KR_PATH, verbosity=0, remove_zip=False)
    assert len(rr.stations()) == NUM_STATIONS
    assert os.path.exists(ARCHIVE), "the archive must survive remove_zip=False"


def test_feature_names():
    logger.info("test_feature_names")
    raw = raw_ts(dataset.stations()[0])
    assert dataset.dynamic_features == [RAW_TO_STD[c] for c in raw.columns]

    raw_st = raw_static()
    assert dataset.static_features == [STATIC_RENAMED.get(c, c) for c in raw_st.columns]
    assert len(dataset.static_features) == len(set(dataset.static_features)) == 75
    # inconsistent elevation statistics keep their raw names
    for c in ('elev_mean', 'elev_min', 'elev_5', 'elev_95', 'elev_max'):
        assert c in dataset.static_features
    for f in ('elev_catch_m', 'elev_catch_min_m', 'elev_catch_max_m'):
        assert f not in dataset.static_features
    # no simulated streamflow
    assert not any('sim' in f for f in dataset.dynamic_features)


def test_elevation_columns_are_inconsistent():
    """guards the reason for not renaming the elev_* columns: if a new release
    fixes them, this fails and the renaming should be reconsidered."""
    logger.info("test_elevation_columns_are_inconsistent")
    topo = pd.read_csv(os.path.join(ROOT, "CAMELS_KR_topography_attributes.csv"), encoding='utf-8-sig')
    assert (topo['elev_5'] > topo['elev_max']).all()


def test_model_output_not_extracted():
    logger.info("test_model_output_not_extracted")
    for folder in MODEL_OUTPUT:
        assert not os.path.exists(os.path.join(ROOT, folder)), folder
    if os.path.exists(ARCHIVE):
        import zipfile
        with zipfile.ZipFile(ARCHIVE) as zf:
            names = zf.namelist()
        for folder in MODEL_OUTPUT:
            assert any(n.startswith(f"CAMELS-KR/{folder}/") for n in names), \
                f"{folder} not in the archive, the exclusion is a no-op"


def test_manifest_completeness():
    logger.info("test_manifest_completeness")
    loc = pd.read_csv(os.path.join(ROOT, ATTR_FILES[0]), dtype={'gauge_id': str}, encoding='utf-8-sig')
    manifest = set(loc['gauge_id'])
    assert len(manifest) == NUM_STATIONS == len(loc)
    assert dataset.stations() == loc['gauge_id'].tolist()
    for f in ATTR_FILES:
        df = pd.read_csv(os.path.join(ROOT, f), dtype={'gauge_id': str}, encoding='utf-8-sig')
        assert set(df['gauge_id']) == manifest and df['gauge_id'].is_unique, f
    for kind in (HYDRO, METEO):
        folder = os.path.join(ROOT, f"{kind} time series")
        stns = {f.split('_')[-1][:-4] for f in os.listdir(folder)}
        assert stns == manifest, kind


def test_missing_file_warns():
    logger.info("test_missing_file_warns")
    orig = _camels._kr_ts_fname
    _camels._kr_ts_fname = lambda kind, stn: orig(kind, stn) + ("x" if stn == '1001620' else "")
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dataset._check_manifest()
    finally:
        _camels._kr_ts_fname = orig
    msgs = [str(x.message) for x in w if 'missing' in str(x.message)]
    assert len(msgs) == 1 and '2 files are missing' in msgs[0], msgs

    # a missing boundary shapefile (and its .dbf/.shx/.prj siblings, which fiona
    # needs) warns; a missing location file (the manifest) raises
    orig_bnd, orig_attr = CAMELS_KR.boundary_file, CAMELS_KR._attr_path
    CAMELS_KR.boundary_file = property(
        lambda self: os.path.join(ROOT, 'nowhere', 'CAMELS_KR_catchments.shp'))
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dataset._check_manifest()
    finally:
        CAMELS_KR.boundary_file = orig_bnd
    msgs = [str(x.message) for x in w if 'missing' in str(x.message)]
    assert len(msgs) == 1 and '4 files are missing' in msgs[0], msgs

    CAMELS_KR._attr_path = lambda self, name: orig_attr(self, name) + ('x' if name == 'location' else '')
    try:
        with pytest.raises(FileNotFoundError):
            dataset._check_manifest()
    finally:
        CAMELS_KR._attr_path = orig_attr


def test_returns_copies():
    logger.info("test_returns_copies")
    for getter in (dataset.stations, lambda: dataset.dynamic_features,
                   lambda: dataset.static_features):
        items = getter()
        items.append('XXX')
        assert 'XXX' not in getter()
    df = dataset._static_data()
    df.iloc[0, 0] = -1
    df.drop(columns=df.columns[1], inplace=True)
    assert dataset._static_data().shape == (NUM_STATIONS, 75)
    assert dataset._static_data().iloc[0, 0] != -1


def test_read_stn_dyn_fidelity():
    """the csv read path, including the stations with implausible raw values
    which must be served unchanged"""
    logger.info("test_read_stn_dyn_fidelity")
    stns = random.sample(dataset.stations(), 3) + ['3014610', '2016650', '4009665']
    for stn in stns:
        raw = raw_ts(stn)
        df = dataset._read_stn_dyn(stn)
        assert df.shape == (DYN_LEN, 14)
        assert (df.dtypes == dataset.fp).all()
        assert_frame_matches_raw(stn, df, raw)
    raw = raw_ts('3014610')
    n = int((raw['discharge_vol'] == -999).sum())
    assert n > 0 and int((dataset._read_stn_dyn('3014610')['q_cms_obs'] == -999).sum()) == n


def _read_raw_in_worker(stn):
    return stn, raw_ts(stn)


def test_all_stations_fidelity_and_extent():
    """every station, from both the netCDF cache and the csv path"""
    logger.info("test_all_stations_fidelity_and_extent")
    with cf.ProcessPoolExecutor(16) as ex:
        raws = dict(ex.map(_read_raw_in_worker, dataset.stations()))

    t = time.time()
    _, from_nc = dataset.fetch(as_dataframe=True)
    t_nc = time.time() - t
    t = time.time()
    from_csv = dataset._read_dynamic(dataset.stations(), 'all')
    t_csv = time.time() - t
    print(f"fetching all {NUM_STATIONS} stations: {t_nc:.2f} s from netCDF, {t_csv:.2f} s from csv")

    assert len(from_nc) == len(from_csv) == NUM_STATIONS
    starts, ends = set(), set()
    for stn, raw in raws.items():
        starts.add(raw.index.min())
        ends.add(raw.index.max())
        # no missing or duplicated days in the source
        assert raw.index.equals(pd.date_range(raw.index.min(), raw.index.max(), freq='D')), stn
        assert_frame_matches_raw(stn, from_csv[stn], raw)
        assert_frame_matches_raw(stn, from_nc[stn], raw)
        # float32 rounding is negligible
        vals, ref = from_csv[stn][list(RAW_TO_STD.values())].to_numpy('float64'), \
            np.column_stack([raw[c] * RAW_FACTORS.get(c, 1.0) for c in RAW_TO_STD])
        ok = np.isfinite(ref) & (ref != 0)
        assert np.nanmax(np.abs(vals[ok] - ref[ok]) / np.abs(ref[ok])) < 1e-6, stn
    assert starts == {dataset.start} and ends == {dataset.end}, (starts, ends)


def test_units():
    logger.info("test_units")
    area = dataset.area()
    for stn in random.sample(dataset.stations(), 10):
        raw = raw_ts(stn)
        q = raw.dropna(subset=['discharge_vol'])
        q = q[q['discharge_vol'] > 1]
        # mm day-1 == m3 s-1 * 86400 s / (area km2 * 1e6 m2) * 1e3 mm m-1
        assert np.allclose(q['discharge_spec'], q['discharge_vol'] * 86.4 / area[stn], rtol=2e-3)
    # the clearness index (surface / top-of-atmosphere daily mean irradiance) is
    # physically 0.25-0.8 on average (0.38-0.73 here); an unconverted or doubly
    # converted MJ m-2 day-1 series lands outside by a factor of 11.6
    _, dyn = dataset.fetch(dynamic_features='swdownrad_wm2', as_dataframe=True)
    lats = dataset.stn_coords()['lat']
    for stn, df in dyn.items():
        phi = np.radians(float(lats[stn]))
        j = df.index.dayofyear.values
        dr = 1 + 0.033 * np.cos(2 * np.pi * j / 365.0)
        dec = 0.409 * np.sin(2 * np.pi * j / 365.0 - 1.39)
        ws = np.arccos(-np.tan(phi) * np.tan(dec))
        toa = (1367.0 / np.pi) * dr * (ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))
        kt = np.nanmean(df['swdownrad_wm2'].to_numpy('float64') / toa)
        assert 0.25 < kt < 0.8, f"{stn}: clearness index {kt}"


def test_static_fidelity():
    logger.info("test_static_fidelity")
    raw = raw_static().rename(columns=STATIC_RENAMED)
    got = dataset.fetch_static_features('all', 'all')
    pd.testing.assert_frame_equal(got, raw, check_names=False)


def test_coords_area_boundary():
    logger.info("test_coords_area_boundary")
    loc = pd.read_csv(os.path.join(ROOT, ATTR_FILES[0]), dtype={'gauge_id': str},
                      encoding='utf-8-sig').set_index('gauge_id')
    coords = dataset.stn_coords()
    assert np.array_equal(coords['lat'].to_numpy(), loc['gauge_lat'].to_numpy(dtype=dataset.fp))
    assert np.array_equal(coords['long'].to_numpy(), loc['gauge_lon'].to_numpy(dtype=dataset.fp))
    assert np.array_equal(dataset.area().to_numpy(), loc['basin_area'].to_numpy(dtype=dataset.fp))
    if fiona is None:
        pytest.skip("fiona is not installed")
    from aqua_fetch.rr.utils import _make_boundary_2d
    ids = dataset._create_boundary_id_map().keys()
    assert set(ids) == set(dataset.stations())
    for stn in random.sample(dataset.stations(), 20):
        for ring in _make_boundary_2d(dataset.get_boundary(stn)):
            assert (ring[:, 0] > 126).all() and (ring[:, 0] < 130).all()
            assert (ring[:, 1] > 34).all() and (ring[:, 1] < 39.5).all()


def test_q_mm_is_observed_mm():
    logger.info("test_q_mm_is_observed_mm")
    stn = random.choice(dataset.stations())
    q = dataset.q_mm(stn)
    assert np.array_equal(q[stn].to_numpy(), expected(raw_ts(stn), 'discharge_spec'), equal_nan=True)


def test_reader_spec_skips_unrequested_file():
    logger.info("test_reader_spec_skips_unrequested_file")
    spec = dataset._reader_spec(['pcp_mm', 'swdownrad_wm2'])
    assert list(spec['files']) == [METEO]
    assert spec['files'][METEO][1] == ['date', 'prec', 'solar_radiation']
    spec = dataset._reader_spec(['wl_m_obs'])
    assert list(spec['files']) == [HYDRO]

    stn = random.choice(dataset.stations())
    raw = raw_ts(stn)
    dyn = dataset._read_dynamic([stn], ['swdownrad_wm2', 'q_cms_obs'], st='2001-01-01', en='2001-12-31')
    df = dyn[stn]
    assert df.columns.tolist() == ['swdownrad_wm2', 'q_cms_obs']
    assert len(df) == 365
    sub = raw.loc['2001-01-01':'2001-12-31']
    assert np.array_equal(df['swdownrad_wm2'].to_numpy(), expected(sub, 'solar_radiation'))
    assert np.array_equal(df['q_cms_obs'].to_numpy(), expected(sub, 'discharge_vol'), equal_nan=True)


class _RecordingPool(cf.ProcessPoolExecutor):
    used = 0

    def __init__(self, *args, **kwargs):
        _RecordingPool.used += 1
        super().__init__(*args, **kwargs)


def test_n_workers():
    """thresholds measured on 48 cores: fork pays off from ~5 MB of csv,
    forkserver from ~60 MB and spawn from ~120 MB"""
    logger.info("test_n_workers")
    mp = base_utils.mp
    orig, orig_all = mp.get_start_method, mp.get_all_start_methods
    cpus = min(base_utils.get_cpus(), 32)
    try:
        for method, small, large in (('fork', 4e6, 6e6), ('forkserver', 50e6, 70e6),
                                     ('spawn', 110e6, 130e6)):
            # the start method set by the user, and the platform default when none is set
            for get, get_all in ((lambda allow_none=False, m=method: m, orig_all),
                                 (lambda allow_none=False: None, lambda m=method: [m, 'spawn'])):
                mp.get_start_method, mp.get_all_start_methods = get, get_all
                assert base_utils.n_workers(small, 50) == 1, method
                assert base_utils.n_workers(large, 50) == min(cpus, 50), method
                assert base_utils.n_workers(large, 3) == min(cpus, 3), method
                assert base_utils.n_workers(1e9, 50, processes=1) == 1
                assert base_utils.n_workers(1e9, 1) == 1
                assert base_utils.n_workers(1e9, 50, processes=4) == 4
    finally:
        mp.get_start_method, mp.get_all_start_methods = orig, orig_all


def test_process_pool_use():
    logger.info("test_process_pool_use")
    orig = _camels.cf.ProcessPoolExecutor
    _camels.cf.ProcessPoolExecutor = _RecordingPool
    _RecordingPool.used = 0
    try:
        # two stations' files are ~4.7 MB, below every threshold
        dataset._read_dynamic(dataset.stations()[:2], 'all')
        assert _RecordingPool.used == 0
        dataset._read_dynamic(dataset.stations(), 'all')
        assert _RecordingPool.used == 1

        # 60 stations (~140 MB) would use a pool with any start method
        serial = CAMELS_KR(path=CAMELS_KR_PATH, verbosity=0, processes=1)
        out = serial._read_dynamic(serial.stations()[:60], 'all')
        assert _RecordingPool.used == 1, "processes=1 must not use a pool"
        assert len(out) == 60
    finally:
        _camels.cf.ProcessPoolExecutor = orig


def test_fetch_outside_data_range():
    logger.info("test_fetch_outside_data_range")
    stn = dataset.stations()[0]
    _, before = dataset.fetch(stn, st='1950-01-01', en='1970-12-31', as_dataframe=True)
    assert len(before[stn]) == 0
    _, inside = dataset.fetch(stn, st='2010-01-01', en='2010-12-31', as_dataframe=True)
    assert len(inside[stn]) == 365


class _Other:
    """stands in for another dataset, without downloading one"""

    def __init__(self, ids=None, coords=None):
        self.ids, self.coords = ids, coords

    def stations(self):
        return list(self.ids)

    def stn_coords(self, stations='all'):
        return self.coords


def test_common_stations_by_id():
    """CAMELS_SK uses the same official Korean gauge codes, so the shared gauges
    are found by matching ids"""
    logger.info("test_common_stations_by_id")
    stns = dataset.stations()
    assert dataset.common_stations([stns[7], stns[0], 'not_an_id']) == [stns[0], stns[7]]
    assert dataset.common_stations(_Other(ids=[stns[7], stns[0]])) == [stns[0], stns[7]]
    assert dataset.common_stations([]) == []

    # the real CAMELS_SK ids, read from its files so that nothing is downloaded
    sk_dir = os.path.join(CAMELS_KR_PATH, 'CAMELS_SK', 'timeseries', 'timeseries')
    if not os.path.isdir(sk_dir):
        pytest.skip("CAMELS_SK data is not available")
    sk_ids = [f.split('.')[0] for f in os.listdir(sk_dir) if f.endswith('.csv')]
    assert len(sk_ids) == 178
    shared = dataset.common_stations(sk_ids)
    assert len(shared) == 115, len(shared)
    assert set(shared) == set(stns) & set(sk_ids)


def test_common_stations_by_distance():
    """a dataset with different ids (e.g. GSHA) is matched by distance. One
    degree of latitude is 111.19 km on the sphere used here (110.99 km on the
    WGS84 ellipsoid, i.e. 0.2% shorter)."""
    logger.info("test_common_stations_by_distance")
    coords = dataset.stn_coords()
    stn = dataset.stations()[8]
    one_degree_north = coords.loc[[stn]].copy()
    one_degree_north['lat'] = one_degree_north['lat'] + 1.0

    assert stn not in dataset.common_stations(_Other(coords=one_degree_north), max_dist_km=111.0)
    assert stn in dataset.common_stations(_Other(coords=one_degree_north), max_dist_km=111.3)

    assert dataset.common_stations(_Other(coords=coords.loc[[stn]]), max_dist_km=0.1) == [stn]
    far = coords.loc[[stn]].copy()
    far['lat'] = far['lat'] + 5.0
    assert dataset.common_stations(_Other(coords=far), max_dist_km=1.0) == []

    # a list of ids carries no coordinates
    with pytest.raises(TypeError):
        dataset.common_stations([stn], max_dist_km=1.0)


def test_no_dynamic_feature_requested():
    logger.info("test_no_dynamic_feature_requested")
    with pytest.raises(ValueError):
        dataset._read_dynamic(dataset.stations()[:1], [])


def test_invalid_station():
    logger.info("test_invalid_station")
    for bad in ('0000000', ['1001620', '0000000']):
        try:
            dataset.fetch(bad, as_dataframe=True)
        except (ValueError, AssertionError):
            pass
        else:
            raise AssertionError(f"invalid station {bad} accepted")


def test_duplicate_check():
    logger.info("test_duplicate_check")
    calls = []
    orig = _camels._warn_duplicate_gauges
    _camels._warn_duplicate_gauges = lambda name, meta: calls.append(meta) or orig(name, meta)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CAMELS_KR(path=CAMELS_KR_PATH, verbosity=0)
    finally:
        _camels._warn_duplicate_gauges = orig
    # the check runs at init on all gauges and, for this data, finds nothing
    assert len(calls) == 1 and calls[0]['gauge_id'].tolist() == dataset.stations()
    assert not [x for x in w if 'duplicate' in str(x.message).lower()]

    loc = pd.read_csv(os.path.join(ROOT, ATTR_FILES[0]), dtype={'gauge_id': str}, encoding='utf-8-sig')
    # same name, coordinates differing below the rounding -> a duplicate
    dup = loc.iloc[[0]].assign(gauge_id='X', gauge_lat=loc['gauge_lat'].iloc[0] + 1e-6)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _camels._warn_duplicate_gauges('CAMELS_KR', pd.concat([loc, dup]))
    msgs = [str(x.message) for x in w if 'duplicate' in str(x.message).lower()]
    assert len(msgs) == 1 and "'X'" in msgs[0], msgs


def _raise(*args, **kwargs):
    raise AssertionError("must not be called")


def test_no_redownload_or_reextract():
    logger.info("test_no_redownload_or_reextract")
    orig_dl, orig_ext = download_zenodo.download_from_zenodo, _camels.zipfile.ZipFile.extractall
    orig_nc = xr.Dataset.to_netcdf if xr is not None else None
    download_zenodo.download_from_zenodo = _raise
    _camels.zipfile.ZipFile.extractall = _raise
    if xr is not None:
        xr.Dataset.to_netcdf = _raise
    try:
        ds = CAMELS_KR(path=CAMELS_KR_PATH, verbosity=0)
        assert len(ds.stations()) == NUM_STATIONS
    finally:
        download_zenodo.download_from_zenodo = orig_dl
        _camels.zipfile.ZipFile.extractall = orig_ext
        if xr is not None:
            xr.Dataset.to_netcdf = orig_nc


def test_no_redownload_without_archive():
    """after remove_zip=True only the extracted folder is left"""
    logger.info("test_no_redownload_without_archive")
    tmp = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp, 'CAMELS_KR'))
    os.symlink(ROOT, os.path.join(tmp, 'CAMELS_KR', 'CAMELS-KR'))
    orig_dl, orig_ext = download_zenodo.download_from_zenodo, _camels.zipfile.ZipFile.extractall
    download_zenodo.download_from_zenodo = _raise
    _camels.zipfile.ZipFile.extractall = _raise
    try:
        ds = CAMELS_KR(path=tmp, verbosity=0, to_netcdf=False)
        assert len(ds.stations()) == NUM_STATIONS
        assert not os.path.exists(os.path.join(tmp, 'CAMELS_KR', 'CAMELS-KR.zip'))
    finally:
        download_zenodo.download_from_zenodo = orig_dl
        _camels.zipfile.ZipFile.extractall = orig_ext
        shutil.rmtree(tmp)  # removes the symlink, not its target


def test_interrupted_extraction_and_overwrite():
    """an interrupted extraction is not taken as complete and is redone, and
    overwrite=True replaces a stale archive, extracted folder and netCDF cache
    before downloading"""
    logger.info("test_interrupted_extraction_and_overwrite")
    if not os.path.exists(ARCHIVE):
        pytest.skip(f"{ARCHIVE} was removed")
    tmp = tempfile.mkdtemp()
    ds_dir = os.path.join(tmp, 'CAMELS_KR')
    root = os.path.join(ds_dir, 'CAMELS-KR')
    os.makedirs(ds_dir)
    os.symlink(ARCHIVE, os.path.join(ds_dir, 'CAMELS-KR.zip'))
    orig_dl, orig_ext = download_zenodo.download_from_zenodo, _camels.zipfile.ZipFile.extractall

    def extract_then_fail(zf, path=None, members=None, pwd=None):
        for m in members[:100]:
            zf.extract(m, path)
        raise KeyboardInterrupt

    try:
        download_zenodo.download_from_zenodo = _raise

        # 1) extraction interrupted after 100 files
        _camels.zipfile.ZipFile.extractall = extract_then_fail
        try:
            CAMELS_KR(path=tmp, verbosity=0, to_netcdf=False)
        except KeyboardInterrupt:
            pass
        _camels.zipfile.ZipFile.extractall = orig_ext
        assert not os.path.exists(root), "a partial extraction looks complete"
        assert os.listdir(os.path.join(ds_dir, 'CAMELS-KR_extracting'))

        # 2) the next init extracts again; remove_zip=True then deletes the
        # archive (here a symlink, not the real archive)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ds = CAMELS_KR(path=tmp, verbosity=0, to_netcdf=False, remove_zip=True)
        assert not os.path.exists(os.path.join(ds_dir, 'CAMELS-KR_extracting'))
        assert len(os.listdir(os.path.join(root, 'Meteorological time series'))) == NUM_STATIONS
        for folder in MODEL_OUTPUT:
            assert not os.path.exists(os.path.join(root, folder)), f"{folder} was extracted"
        assert any('model output' in str(x.message) for x in w)
        assert_frame_matches_raw('1001620', ds._read_stn_dyn('1001620'), raw_ts('1001620'))
        assert not os.path.lexists(os.path.join(ds_dir, 'CAMELS-KR.zip')), "remove_zip ignored"
        assert os.path.exists(ARCHIVE)

        # no archive left: the next init must neither download nor extract
        _camels.zipfile.ZipFile.extractall = _raise
        CAMELS_KR(path=tmp, verbosity=0, to_netcdf=False)
        _camels.zipfile.ZipFile.extractall = orig_ext

        # 3) overwrite: stale archive, extracted folder and cache
        stale_archive = os.path.join(ds_dir, 'CAMELS-KR.zip')
        if os.path.lexists(stale_archive):
            # never write through a symlink to the real archive
            raise RuntimeError(f"{stale_archive} still exists")
        with open(stale_archive, 'w') as fp:
            fp.write('stale')
        with open(os.path.join(ds_dir, 'CAMELS-KR', 'stale.txt'), 'w') as fp:
            fp.write('stale')
        with open(ds.dyn_fpath, 'w') as fp:
            fp.write('stale')
        calls = []

        def fake_download(outdir, doi, include=None, **kwargs):
            calls.append(doi)
            target = os.path.join(outdir, include[0])
            # download() would write <name>.zip1 if the stale archive were still there
            assert not os.path.exists(target), "stale archive not removed before download"
            assert not os.path.exists(os.path.join(ds_dir, 'CAMELS-KR')), "stale data not removed"
            assert not os.path.exists(ds.dyn_fpath), "stale netCDF cache not removed"
            os.symlink(ARCHIVE, target)

        download_zenodo.download_from_zenodo = fake_download
        CAMELS_KR(path=tmp, verbosity=0, to_netcdf=False, overwrite=True)
        assert calls == [dataset.url]
        assert not os.path.exists(os.path.join(ds_dir, 'CAMELS-KR', 'stale.txt'))
        assert os.path.isdir(os.path.join(ds_dir, 'CAMELS-KR', 'Hydrological time series'))
    finally:
        download_zenodo.download_from_zenodo = orig_dl
        _camels.zipfile.ZipFile.extractall = orig_ext
        shutil.rmtree(tmp)


def test_interrupted_cache_build_is_not_reused():
    """an interrupted netCDF cache build must leave nothing behind. A partial
    cache used to be taken as complete and then served missing or empty
    stations without any warning."""
    logger.info("test_interrupted_cache_build_is_not_reused")
    if xr is None:
        pytest.skip("xarray is not installed")
    tmp = tempfile.mkdtemp()
    ds_dir = os.path.join(tmp, 'CAMELS_KR')
    os.makedirs(ds_dir)
    os.symlink(ROOT, os.path.join(ds_dir, 'CAMELS-KR'))
    orig_dl, orig_nc = download_zenodo.download_from_zenodo, xr.Dataset.to_netcdf
    download_zenodo.download_from_zenodo = _raise

    cache = os.path.join(ds_dir, CAMELS_KR(path=tmp, verbosity=0, to_netcdf=False).dyn_fname)
    written_to = []

    def write_5_stations_then_die(self, path=None, *args, **kwargs):
        # the cache must never be written under its final name: a hard kill
        # (SIGKILL, power loss) skips every cleanup handler, so only writing
        # somewhere else and renaming keeps a partial file from being served
        written_to.append(str(path))
        assert str(path) != cache, "the cache is written under its final name"
        orig_nc(self[list(self.data_vars)[:5]], path, *args, **kwargs)
        raise KeyboardInterrupt

    xr.Dataset.to_netcdf = write_5_stations_then_die
    try:
        try:
            CAMELS_KR(path=tmp, verbosity=0)
        except KeyboardInterrupt:
            pass
        assert written_to and all(p.endswith('.part') for p in written_to), written_to
        left = [f for f in os.listdir(ds_dir) if f.endswith('.nc') or f.endswith('.part')]
        assert left == [], f"an interrupted cache build left {left}"

        # the next initialization builds a complete cache and leaves no .part
        xr.Dataset.to_netcdf = orig_nc
        ds = CAMELS_KR(path=tmp, verbosity=0)
        assert ds.dyn_fpath_exists
        assert not os.path.exists(f"{ds.dyn_fpath}.part"), "temporary cache file left behind"
        _, dyn = ds.fetch(as_dataframe=True)
        assert len(dyn) == NUM_STATIONS
        stn = ds.stations()[-1]
        assert_frame_matches_raw(stn, dyn[stn], raw_ts(stn))
    finally:
        download_zenodo.download_from_zenodo = orig_dl
        xr.Dataset.to_netcdf = orig_nc
        shutil.rmtree(tmp)


def test_remove_stale_unlinks_symlinks():
    """overwrite=True on data kept elsewhere through a symlink removes the link,
    not the data it points to"""
    logger.info("test_remove_stale_unlinks_symlinks")
    tmp = tempfile.mkdtemp()
    try:
        target = os.path.join(tmp, 'target')
        os.makedirs(target)
        with open(os.path.join(target, 'data.csv'), 'w') as fp:
            fp.write('data')
        for name, dst in (('dir_link', target), ('file_link', os.path.join(target, 'data.csv')),
                          ('broken_link', os.path.join(tmp, 'missing'))):
            os.symlink(dst, os.path.join(tmp, name))
        stale_dir = os.path.join(tmp, 'stale_dir')
        os.makedirs(os.path.join(stale_dir, 'sub'))
        _camels._remove_stale([os.path.join(tmp, n) for n in
                               ('dir_link', 'file_link', 'broken_link', 'stale_dir', 'absent')], 0)
        assert sorted(os.listdir(tmp)) == ['target']
        assert os.listdir(target) == ['data.csv']
    finally:
        shutil.rmtree(tmp)


def test_standard_suite():
    """the comprehensive suite shared by all rainfall-runoff datasets"""
    logger.info("test_standard_suite")
    rr_utils.test_dataset(dataset,
                          num_stations=NUM_STATIONS,
                          dyn_data_len=DYN_LEN,
                          num_static_attrs=75,
                          num_dyn_attrs=14,
                          yearly_steps=366,
                          st="20040101", en="20041231")


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
    print("*** All CAMELS_KR tests passed ***")
