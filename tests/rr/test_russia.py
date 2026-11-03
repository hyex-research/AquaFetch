"""
Tests for the Russia (1,886 Roshydromet gauges) dataset.

They check that the class

    * returns every served value unchanged and in the source units (compared
      with an independent read of the raw files),
    * leaves out exactly what it documents (GLEAM, the precipitation and the
      GRDC catchments by default) and says so,
    * downloads and extracts the data only once,
    * reads many stations in parallel, but never with ``processes=1``.

The data must already be in ``RUSSIA_PATH``; the first run there extracts it.
No test downloads the data or writes into ``RUSSIA_PATH``.
"""

import os
import inspect
import sys
import time
import types
import shutil
import random
import logging
import zipfile
import tempfile
import warnings
import functools
import contextlib
import multiprocessing as mp
import concurrent.futures as cf

import numpy as np
import pandas as pd
import pytest

# repository root and this folder (for the shared ``utils`` helpers)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    logging.basicConfig(filename='test_russia.log', filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

import aqua_fetch
import aqua_fetch.download_zenodo as download_zenodo
from aqua_fetch import Russia
from aqua_fetch.rr import _russia
from aqua_fetch._backend import netCDF4, fiona

import utils as rr_utils

# the same random stations in every run, with pytest or as a script
random.seed(313)

# The class appends ``Russia`` to this path. Replace it with the location on
# your machine.
RUSSIA_PATH = '/mnt/hyexnas01/abbaa0a/data/gscad_database/raw'

NUM_STATIONS = 1886   # Roshydromet gauges
NUM_GRDC = 278        # GRDC gauges, served with grdc_stns=True except the 6 below
# GRDC gauges that are Roshydromet gauges of the dataset under another id
GRDC_DUPLICATES = {'2311208': '11411', '2910562': '10150', '2911306': '12010',
                   '6984500': '84108', '6984700': '84454', '6984800': '84346'}
NUM_FULL = NUM_STATIONS + NUM_GRDC - len(GRDC_DUPLICATES)
DYN_LEN = 4749        # daily steps 2008-01-01 .. 2020-12-31
NUM_STATIC = 25

TOP = 'Russia_HydroMeteo_Database_v04'
ROOT = os.path.join(RUSSIA_PATH, 'Russia', TOP)

# raw variable -> name served by the class. Written out here (not imported)
# so that a wrong mapping in the class is caught.
OBS = {'q_cms_s': 'q_cms_obs', 'q_mm_day': 'q_mm_obs', 'lvl_sm': 'wl_cm_obs', 'lvl_mbs': 'lvl_mbs'}
TEMP = {'t_max_e5': 'airtemp_C_era5_max', 't_min_e5': 'airtemp_C_era5_min',
        't_max_e5l': 'airtemp_C_era5land_max', 't_min_e5l': 'airtemp_C_era5land_min'}
# served only with suspect_pcp=True
PCP = {'prcp_e5l': 'pcp_mm_era5land', 'prcp_mswep': 'pcp_mm_mswep', 'prcp_imerg': 'pcp_mm_imerg',
       'prcp_e5': 'pcp_mm_era5', 'prcp_gpcp': 'pcp_mm_gpcp'}
RAW_TO_STD = {**OBS, **TEMP}               # served by default
ALL_SERVED = {**OBS, **PCP, **TEMP}        # served with suspect_pcp=True
NUM_DYN = len(RAW_TO_STD)

STATIC_RENAMED = {'ws_area': 'area_km2', 'height_bs': 'elev_gauge_m',
                  'ele_mt_sav': 'elev_catch_m', 'slp_dg_sav': 'slope_degrees'}

# pool workers started with spawn or forkserver (Windows, macOS, Linux from
# Python 3.14) import this module again; only the main process builds datasets
if mp.parent_process() is None:
    t0 = time.time()
    dataset = Russia(path=RUSSIA_PATH, verbosity=0)
    print(f"Russia initialization took {time.time() - t0:.2f} s")
    # every gauge and every precipitation product
    full = Russia(path=RUSSIA_PATH, verbosity=0, grdc_stns=True, suspect_pcp=True)


# ---------------------------------------------------------------------------
# independent raw readers
# ---------------------------------------------------------------------------

def raw_file(stn: str) -> str:
    return os.path.join(ROOT, 'nc_concat', f'{stn}.nc')


def raw_stn(stn: str, columns=None) -> pd.DataFrame:
    """the variables ``columns`` (default: all) of a station file, float64,
    raw names, without netCDF4's masking; the time axis decoded by
    netCDF4.num2date"""
    with netCDF4.Dataset(raw_file(stn)) as nc:
        nc.set_auto_mask(False)
        name = 'date' if 'date' in nc.variables else 'index'
        t = nc.variables[name]
        index = pd.DatetimeIndex(netCDF4.num2date(t[:], t.units, t.calendar,
                                                  only_use_cftime_datetimes=False,
                                                  only_use_python_datetimes=True))
        return pd.DataFrame({v: np.asarray(nc.variables[v][:], dtype='float64').ravel()
                             for v in nc.variables
                             if v not in (name, 'gauge_id') and (columns is None or v in columns)},
                            index=index)


def _raw_in_worker(stn, columns=None):
    return stn, raw_stn(stn, columns)


def read_raw(stations, columns=None) -> dict:
    with cf.ProcessPoolExecutor(16) as ex:
        return dict(ex.map(functools.partial(_raw_in_worker, columns=columns), stations,
                           chunksize=16))


def raw_static() -> pd.DataFrame:
    return pd.read_csv(os.path.join(ROOT, 'attributes', 'russia_cis_static.csv'),
                       dtype={'gauge_id': str}).set_index('gauge_id')


def assert_frame_matches_raw(stn: str, df: pd.DataFrame, raw: pd.DataFrame, mapping=None):
    mapping = mapping or RAW_TO_STD
    assert df.index.equals(raw.index), f"{stn}: time index changed"
    assert (df.dtypes == dataset.fp).all(), f"{stn}: {df.dtypes.unique()}"
    for raw_c, std_c in mapping.items():
        got = df[std_c].to_numpy()
        want = raw[raw_c].to_numpy(dtype=dataset.fp) if raw_c in raw else np.full(len(raw), np.nan, dataset.fp)
        assert np.array_equal(got, want, equal_nan=True), f"{stn}:{std_c} values changed"


def _raise(*args, **kwargs):
    raise AssertionError("must not be called")


@contextlib.contextmanager
def patched(*replacements):
    """sets each ``(owner, attribute, value)`` for the duration of the block"""
    originals = [(owner, name, getattr(owner, name)) for owner, name, _ in replacements]
    for owner, name, value in replacements:
        setattr(owner, name, value)
    try:
        yield
    finally:
        for owner, name, value in reversed(originals):
            setattr(owner, name, value)


NO_DOWNLOAD = (download_zenodo, 'download_from_zenodo', _raise)
NO_EXTRACT = (zipfile.ZipFile, 'extractall', _raise)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_registration():
    logger.info("test_registration")
    assert 'Russia' in aqua_fetch.ALL_DATASETS
    from aqua_fetch import RainfallRunoff
    # remove_zip=False: the default True would delete the archive
    rr = RainfallRunoff('Russia', path=RUSSIA_PATH, verbosity=0, remove_zip=False)
    assert len(rr.stations()) == NUM_STATIONS


def test_options_and_warnings():
    """what is left out is reported regardless of verbosity, and each option
    brings back exactly what it documents"""
    logger.info("test_options_and_warnings")
    for grdc in (False, True):
        for pcp in (False, True):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                ds = Russia(path=RUSSIA_PATH, verbosity=0, grdc_stns=grdc, suspect_pcp=pcp)
            msgs = [str(x.message) for x in w if 'not served' in str(x.message)]
            assert len(msgs) == 1, msgs
            assert 'GLEAM' in msgs[0]
            assert ('GRDC catchments' in msgs[0]) != grdc
            assert ('suspect_pcp=True includes it' in msgs[0]) != pcp
            assert ('All precipitation is scaled down' in msgs[0]) == pcp
            dups = [str(x.message) for x in w if 'are left out because' in str(x.message)]
            assert len(dups) == grdc, dups
            if grdc:
                assert all(f"{g}={r}" in dups[0] for g, r in GRDC_DUPLICATES.items()), dups
            assert len(ds.stations()) == (NUM_FULL if grdc else NUM_STATIONS)
            assert len(ds.dynamic_features) == NUM_DYN + (len(PCP) if pcp else 0)
            assert ds.fetch_static_features().shape == (len(ds.stations()), NUM_STATIC)
    # the GRDC duplicates are left out, while their Roshydromet ids are served
    assert not set(GRDC_DUPLICATES) & set(full.stations())
    assert set(GRDC_DUPLICATES.values()) <= set(full.stations())


def test_all_stations_fidelity_and_extent():
    """every value of every gauge, as DataFrames and as xarray Dataset, and
    every file shares one complete daily time axis"""
    logger.info("test_all_stations_fidelity_and_extent")
    raws = read_raw(full.stations(), list(ALL_SERVED))

    t = time.time()
    _, served = full.fetch(as_dataframe=True)
    print(f"fetching all {len(served)} stations and {len(full.dynamic_features)} "
          f"features took {time.time() - t:.2f} s")

    axis = pd.date_range('2008-01-01', '2020-12-31', freq='D')
    for stn, raw in raws.items():
        assert raw.index.equals(axis), stn
        assert_frame_matches_raw(stn, served[stn], raw, ALL_SERVED)
    assert dataset.start == axis[0] and dataset.end == axis[-1]

    stns = random.sample(dataset.stations(), 5)
    _, xds = dataset.fetch(stns)
    for stn in stns:
        df = xds[stn].to_pandas()
        assert df.columns.tolist() == dataset.dynamic_features
        assert_frame_matches_raw(stn, df, raws[stn])

    # float32 keeps every published digit of the observations: discharge and
    # level in cm are read back as the same numbers, level in m (computed from
    # cm by the authors, tenths of cm at some gauges) to the mm
    for raw_c in ('q_cms_s', 'lvl_sm', 'lvl_mbs'):
        values = np.unique(np.concatenate([raw[raw_c].dropna().to_numpy() for raw in raws.values()]))
        f32 = values.astype(np.float32)
        if raw_c == 'lvl_mbs':
            assert np.array_equal(np.rint(f32.astype('float64') * 1000), np.rint(values * 1000))
        else:
            assert all(float(str(v)) == w for v, w in zip(f32, values)), raw_c


def test_units():
    logger.info("test_units")
    area = full.area()
    raws = read_raw(random.sample(dataset.stations(), 100),
                    ['q_cms_s', 'q_mm_day', 'lvl_sm', 'lvl_mbs', 't_max_e5', 't_min_e5',
                     't_max_e5l', 't_min_e5l', 'prcp_e5l', 'prcp_mswep'])
    for stn, raw in raws.items():
        q = raw.dropna(subset=['q_cms_s'])
        q = q[q['q_cms_s'] > 0]
        # mm day-1 == m3 s-1 * 86400 s / (area km2 * 1e6 m2) * 1e3 mm m-1
        assert np.allclose(q['q_mm_day'], q['q_cms_s'] * 86.4 / area[stn], rtol=1e-6), stn
        # lvl_mbs (m) is lvl_sm (cm) above one datum per gauge
        offset = (raw['lvl_mbs'] - raw['lvl_sm'] / 100).dropna().round(6).unique()
        assert len(offset) <= 1, (stn, offset[:3])
        # whole cm, at 10 gauges tenths of cm
        lvl = raw['lvl_sm'].dropna().to_numpy()
        assert np.allclose(lvl * 10, np.rint(lvl * 10), rtol=0, atol=1e-6), f"{stn}: lvl_sm not in cm"
        for t in ('t_max_e5', 't_min_e5', 't_max_e5l', 't_min_e5l'):
            assert raw[t].dropna().between(-65, 45).all(), (stn, t)   # deg C
        for p in ('prcp_e5l', 'prcp_mswep'):
            assert raw[p].dropna().between(-1e-3, 250).all(), (stn, p)   # mm day-1
    # HydroATLAS stores slope in degrees x 10 (Caravan: up to ~340); here it
    # is in degrees
    slope = dataset.fetch_static_features(static_features='slope_degrees')['slope_degrees']
    assert 0 < slope.min() and slope.max() < 35


def test_static_fidelity():
    """the static table equals the csv file and the two GeoPackages, read
    with fiona instead of the class's sqlite3 reader, value for value and
    with the same dtypes"""
    logger.info("test_static_fidelity")
    if fiona is None:
        pytest.skip("fiona is not installed")
    with fiona.open(os.path.join(ROOT, 'geometry', 'russia_cis_gauges.gpkg')) as src:
        assert src.crs.to_epsg() == 4326
        gauges = pd.DataFrame([dict(gauge_id=f['properties']['gauge_id'],
                                    name_ru=f['properties']['name_ru'],
                                    name_en=f['properties']['name_en'],
                                    lat=f['geometry']['coordinates'][1],
                                    long=f['geometry']['coordinates'][0]) for f in src]).set_index('gauge_id')
    with fiona.open(os.path.join(ROOT, 'geometry', 'russia_cis_ws.gpkg')) as src:
        assert src.crs.to_epsg() == 4326
        ais = pd.Series({f['properties']['gauge_id']: f['properties']['ais_dif'] for f in src},
                        name='ais_dif')
    raw = pd.concat([gauges, raw_static(), ais], axis=1).rename(columns=STATIC_RENAMED)
    for ds in (dataset, full):
        got = ds.fetch_static_features('all', 'all')
        pd.testing.assert_frame_equal(got, raw.loc[ds.stations()], check_names=False,
                                      check_exact=True)
        assert (got.dtypes.drop(['name_ru', 'name_en']) == 'float64').all()


def test_coords_area_boundary():
    """coordinates and areas equal the source, for all stations and for a few
    in any order, and every catchment boundary lies around its own gauge"""
    logger.info("test_coords_area_boundary")
    static = dataset.fetch_static_features(static_features=['lat', 'long', 'area_km2'])
    for stns in ('all', ['72818', '75299']):
        ids = dataset.stations() if stns == 'all' else stns
        coords = dataset.stn_coords(stns).loc[ids, ['lat', 'long']]
        assert np.array_equal(coords.to_numpy(), static.loc[ids, ['lat', 'long']].to_numpy(dtype=dataset.fp))
        area = dataset.area(stns).loc[ids]
        assert np.array_equal(area.to_numpy(), static.loc[ids, 'area_km2'].to_numpy(dtype=dataset.fp))
    if fiona is None:
        pytest.skip("fiona is not installed")
    rr_utils.test_boundaries_around_gauges(dataset)


def test_q_mm_is_observed_mm():
    logger.info("test_q_mm_is_observed_mm")
    stns = ['72818', '75299']
    q = dataset.q_mm(stns)
    for stn in stns:
        assert np.array_equal(q[stn].to_numpy(), raw_stn(stn)['q_mm_day'].to_numpy(dtype=dataset.fp),
                              equal_nan=True), stn


class _RecordingPool(cf.ProcessPoolExecutor):
    used = 0

    def __init__(self, *args, **kwargs):
        _RecordingPool.used += 1
        super().__init__(*args, **kwargs)


def test_process_pool_use():
    logger.info("test_process_pool_use")
    _RecordingPool.used = 0
    # only the class sees the recording pool, not concurrent.futures itself
    with patched((_russia, 'cf', types.SimpleNamespace(ProcessPoolExecutor=_RecordingPool))):
        # two station files are ~1.9 MB, below every threshold
        dataset._read_dynamic(dataset.stations()[:2], 'all')
        assert _RecordingPool.used == 0
        t = time.time()
        out = dataset._read_dynamic(dataset.stations(), 'all')
        print(f"all stations from the station files with a pool: {time.time() - t:.2f} s")
        assert _RecordingPool.used == 1 and len(out) == NUM_STATIONS

        serial = Russia(path=RUSSIA_PATH, verbosity=0, processes=1)
        t = time.time()
        out = serial._read_dynamic(serial.stations()[:200], 'all')
        print(f"200 stations with processes=1: {time.time() - t:.2f} s")
        assert _RecordingPool.used == 1, "processes=1 must not use a pool"
        assert len(out) == 200


def test_period_with_process_pool():
    """the period (and the feature selection) is applied when many stations
    are read in parallel, not only when they are read one by one"""
    logger.info("test_period_with_process_pool")
    stns = dataset.stations()[:200]   # ~190 MB of files: a pool with any start method
    _RecordingPool.used = 0
    with patched((_russia, 'cf', types.SimpleNamespace(ProcessPoolExecutor=_RecordingPool))):
        _, served = dataset.fetch(stns, dynamic_features=['wl_cm_obs', 'q_cms_obs'],
                                  st='2012-01-01', en='2012-12-31', as_dataframe=True)
    assert _RecordingPool.used == 1, "the stations were not read in parallel"
    raws = read_raw(stns, ['lvl_sm', 'q_cms_s'])
    for stn in stns:
        df = served[stn]
        assert df.columns.tolist() == ['wl_cm_obs', 'q_cms_obs'], stn
        assert df.index.equals(pd.date_range('2012-01-01', '2012-12-31', freq='D')), stn
        assert_frame_matches_raw(stn, df, raws[stn].loc['2012-01-01':'2012-12-31'],
                                 {'lvl_sm': 'wl_cm_obs', 'q_cms_s': 'q_cms_obs'})


def test_duplicate_check():
    """the check runs at init on the served gauges, with their names and
    coordinates, and for this data finds nothing"""
    logger.info("test_duplicate_check")
    calls = []
    orig = _russia._warn_duplicate_gauges
    with patched((_russia, '_warn_duplicate_gauges',
                  lambda name, meta: calls.append(meta) or orig(name, meta))):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            Russia(path=RUSSIA_PATH, verbosity=0)
            Russia(path=RUSSIA_PATH, verbosity=0, grdc_stns=True)
    assert [len(c) for c in calls] == [NUM_STATIONS, NUM_FULL]
    static = full.fetch_static_features(static_features=['name_en', 'lat', 'long'])
    for meta, ds in zip(calls, (dataset, full)):
        assert meta['gauge_id'].tolist() == ds.stations()
        want = static.loc[ds.stations()]
        assert meta['gauge_name'].tolist() == want['name_en'].tolist()
        assert np.array_equal(meta['gauge_lat'].to_numpy(), want['lat'].to_numpy())
        assert np.array_equal(meta['gauge_lon'].to_numpy(), want['long'].to_numpy())
    assert not [x for x in w if 'duplicate' in str(x.message).lower()]


def _mini_archive(fpath: str):
    """a small archive with the layout of the real one, built from the
    extracted files: 5 station files, the static table, the gauge
    GeoPackage, a file of a left-out folder and macOS metadata"""
    with zipfile.ZipFile(fpath, 'w') as dst:
        for stn in ('75299', '1001', '72818', '49001', '6458555'):
            dst.write(raw_file(stn), f"{TOP}/nc_concat/{stn}.nc")
        dst.write(os.path.join(ROOT, 'attributes', 'russia_cis_static.csv'),
                  f"{TOP}/attributes/russia_cis_static.csv")
        dst.write(os.path.join(ROOT, 'geometry', 'russia_cis_gauges.gpkg'),
                  f"{TOP}/geometry/russia_cis_gauges.gpkg")
        dst.write(raw_file('75299'), f"{TOP}/nc_all_q/75299.nc")
        for name in (f"{TOP}/.DS_Store", f"{TOP}/geometry/.DS_Store",
                     f"__MACOSX/{TOP}/nc_concat/._75299.nc"):
            dst.writestr(name, b'macOS metadata')
    return


def test_download_extraction_and_overwrite():
    """the first initialization downloads and extracts; an interrupted
    extraction is not taken as complete and is redone; missing files are
    reported; remove_zip=True deletes the archive; overwrite=True replaces a
    stale archive and extracted folder before downloading"""
    logger.info("test_download_extraction_and_overwrite")
    tmp = tempfile.mkdtemp()
    ds_dir = os.path.join(tmp, 'Russia')
    root = os.path.join(ds_dir, TOP)
    archive = os.path.join(ds_dir, f'{TOP}.zip')
    mini = os.path.join(tmp, 'mini.zip')
    _mini_archive(mini)
    calls = []

    def fake_download(outdir, doi, include=None, **kwargs):
        calls.append((doi, include))
        target = os.path.join(outdir, include[0])
        # download() would write <name>.zip1 if a stale archive were still there
        assert not os.path.exists(target), "stale archive not removed before download"
        assert not os.path.exists(root), "stale data not removed"
        shutil.copy(mini, target)

    def extract_then_fail(zf, path=None, members=None, pwd=None):
        for m in members[:2]:
            zf.extract(m, path)
        raise KeyboardInterrupt

    try:
        # 1) the first initialization downloads; the extraction is interrupted
        with patched((download_zenodo, 'download_from_zenodo', fake_download),
                     (zipfile.ZipFile, 'extractall', extract_then_fail)):
            with pytest.raises(KeyboardInterrupt):
                Russia(path=tmp, verbosity=0)
        assert calls == [(dataset.url, [f'{TOP}.zip'])]
        assert os.path.exists(archive)
        assert not os.path.exists(root), "a partial extraction looks complete"
        assert os.listdir(os.path.join(ds_dir, f'{TOP}_extracting'))

        # 2) the next init extracts again, only the needed folders and without
        # .DS_Store; the files missing from the mini archive (the boundary
        # file and 2159 station files) are reported
        with patched(NO_DOWNLOAD), warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ds = Russia(path=tmp, verbosity=0, remove_zip=True)
        assert not os.path.exists(os.path.join(ds_dir, f'{TOP}_extracting'))
        assert sorted(os.listdir(root)) == ['attributes', 'geometry', 'nc_concat']
        assert sorted(os.listdir(os.path.join(root, 'geometry'))) == ['russia_cis_gauges.gpkg']
        assert len(os.listdir(os.path.join(root, 'nc_concat'))) == 5
        assert any('2160 files are missing' in str(x.message) for x in w), [str(x.message) for x in w]
        assert_frame_matches_raw('75299', ds._read_stn_dyn('75299'), raw_stn('75299'))
        assert not os.path.exists(archive), "remove_zip ignored"

        # no archive left: the next init must neither download nor extract
        with patched(NO_DOWNLOAD, NO_EXTRACT):
            Russia(path=tmp, verbosity=0)

        # 3) overwrite: stale archive and extracted folder
        with open(archive, 'w') as fp:
            fp.write('stale')
        with open(os.path.join(root, 'stale.txt'), 'w') as fp:
            fp.write('stale')
        calls.clear()
        with patched((download_zenodo, 'download_from_zenodo', fake_download)):
            Russia(path=tmp, verbosity=0, overwrite=True)
        assert calls == [(dataset.url, [f'{TOP}.zip'])]
        assert not os.path.exists(os.path.join(root, 'stale.txt'))
        assert len(os.listdir(os.path.join(root, 'nc_concat'))) == 5
    finally:
        shutil.rmtree(tmp)


def test_url_is_latest_release():
    """the class downloads the latest version of the concept record 7789303.
    Needs only the records' metadata, not the data."""
    logger.info("test_url_is_latest_release")
    import requests
    record = dataset.url.rstrip('/').split('/')[-1]
    try:
        # answered with a redirect to the latest version
        r = requests.get("https://zenodo.org/api/records/7789303/versions/latest",
                         timeout=30, allow_redirects=False)
        latest = r.headers['Location'] if r.is_redirect else r.json()['links']['self']
        files = requests.get(f"https://zenodo.org/api/records/{record}", timeout=30).json()['files']
    except Exception as e:
        pytest.skip(f"zenodo metadata is not available: {e!r}")
    assert latest.rstrip('/').split('/')[-1] == record
    assert [f['key'] for f in files] == [f'{TOP}.zip']


def test_docstring_examples():
    """runs the examples of the class docstring, on RUSSIA_PATH, and compares
    the printed outputs"""
    logger.info("test_docstring_examples")
    # Python 3.13 removes the indentation of docstrings, older versions keep it
    lines = inspect.cleandoc(Russia.__doc__).split('Examples\n--------\n')[1].splitlines()
    statements, i = [], 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line.startswith('>>>'):
            continue
        code = line[4:]
        while i < len(lines) and lines[i].strip().startswith('...') \
                and not lines[i].strip().startswith('... #'):
            code += '\n' + lines[i].strip()[4:]
            i += 1
        expected = []
        while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(('>>>', '...')):
            expected.append(lines[i].strip())
            i += 1
        statements.append((code, expected))

    namespace = {}
    for code, expected in statements:
        code = code.replace("Russia()", f"Russia(path={RUSSIA_PATH!r}, verbosity=0)")
        code = code.replace("Russia(grdc_stns", f"Russia(path={RUSSIA_PATH!r}, verbosity=0, grdc_stns")
        try:
            value = eval(code, namespace)
        except SyntaxError:
            exec(code, namespace)
            value = None
        if expected:
            got = [' '.join(line.split()) for line in repr(value).splitlines()]
            want = [' '.join(line.split()) for line in expected]
            assert got == want, (code, got, want)
    assert len(statements) > 15


def test_standard_suite():
    """the comprehensive suite shared by all rainfall-runoff datasets"""
    logger.info("test_standard_suite")
    rr_utils.test_dataset(dataset,
                          num_stations=NUM_STATIONS,
                          dyn_data_len=DYN_LEN,
                          num_static_attrs=NUM_STATIC,
                          num_dyn_attrs=NUM_DYN,
                          yearly_steps=366,
                          st="20120101", en="20121231")


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
    print("*** All Russia tests passed ***")
