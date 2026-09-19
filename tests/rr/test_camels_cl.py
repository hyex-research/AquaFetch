"""
Tests for CAMELS_CL (Chile, 516 catchments), mainly for the 2022 release.

They check that the class

    * serves the raw values and units unchanged (compared with an independent
      read of the raw files), on the union of the files' dates,
    * reads the static attributes and boundaries faithfully,
    * downloads once, does not download or extract again, and on
      ``overwrite=True`` deletes only the stale files of that release before
      downloading, and warns about missing files,
    * still reads the 2018 release, without truncating its minimum temperature,
    * fetches all stations quickly.

The first run downloads the 2022 archive (288 MB) into ``CAMELS_CL_PATH``.
Run as a script or with pytest.
"""

import os
import csv
import copy
import site
import time
import random
import shutil
import logging
import zipfile
import tempfile
import warnings
import functools

import numpy as np
import pandas as pd

# add the repository root to the path so that ``aqua_fetch`` and the shared
# ``utils`` test helpers can be imported
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

if __name__ == "__main__":
    logging.basicConfig(filename='test_camels_cl.log', filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

import aqua_fetch
from aqua_fetch import CAMELS_CL
from aqua_fetch.rr import _camels
from aqua_fetch.utils import BROWSER_HEADERS
from aqua_fetch._backend import xarray as xr, fiona

from utils import test_dataset as run_shared_tests

# The class appends ``CAMELS_CL`` to this path; both releases can live there.
# Replace with the location on your machine.
CAMELS_CL_PATH = '/mnt/hyexnas01/abbaa0a/data/gscad_database/raw/CAMELS/CAMELS'

NUM_STATIONS = 516
NUM_STATIC = 110
NUM_DYNAMIC = 10
DYN_LEN = 44368  # daily steps 1900-01-01 .. 2021-06-22

# standardized feature -> raw 2022 file. Written out here rather than taken
# from the class, so that a wrong mapping in the class is caught.
RAW_FILES = {
    'q_cms_obs': 'q_m3s_day.csv',
    'q_mm_obs': 'q_mm_day.csv',
    'pcp_mm_cr2met': 'precip_cr2met_mm_day.csv',
    'pcp_mm_chirps': 'precip_chirps_mm_day.csv',
    'pcp_mm_mswep': 'precip_mswep_mm_day.csv',
    'pcp_mm_tmpa': 'precip_tmpa_mm_day.csv',
    'airtemp_C_min': 'tmin_cr2met_C_day.csv',
    'airtemp_C_max': 'tmax_cr2met_C_day.csv',
    'airtemp_C_mean': 'tmean_cr2met_C_day.csv',
    'pet_mm_hargreaves': 'pet_hargreaves_mm_day.csv',
}

# the same for the 2018 release: standardized feature -> folder and file name
RAW_FILES_2018 = {
    'q_cms_obs': '2_CAMELScl_streamflow_m3s',
    'q_mm_obs': '3_CAMELScl_streamflow_mm',
    'pcp_mm_cr2met': '4_CAMELScl_precip_cr2met',
    'pcp_mm_chirps': '5_CAMELScl_precip_chirps',
    'pcp_mm_mswep': '6_CAMELScl_precip_mswep',
    'pcp_mm_tmpa': '7_CAMELScl_precip_tmpa',
    'airtemp_C_min': '8_CAMELScl_tmin_cr2met',
    'airtemp_C_max': '9_CAMELScl_tmax_cr2met',
    'airtemp_C_mean': '10_CAMELScl_tmean_cr2met',
    'pet_mm_modis': '11_CAMELScl_pet_8d_modis',
    'pet_mm_hargreaves': '12_CAMELScl_pet_hargreaves',
    'swe': '13_CAMELScl_swe',
}

dataset = CAMELS_CL(path=CAMELS_CL_PATH, verbosity=0)

RAW_DIR = os.path.join(dataset.path, 'CAMELS_CL_v202201', 'CAMELS_CL_v202201')


@functools.lru_cache(maxsize=None)
def _raw(feature: str, stations: tuple) -> pd.DataFrame:
    """independent read of the raw 2022 file of ``feature``, at full precision"""
    df = pd.read_csv(os.path.join(RAW_DIR, RAW_FILES[feature]), float_precision='round_trip',
                     usecols=['date', *stations], index_col='date', parse_dates=True)
    return df[list(stations)]


def _assert_faithful(dyn: dict, stations: tuple):
    """the served values equal the raw values cast to ``dataset.fp``, on the raw dates"""
    for feature in RAW_FILES:
        raw = _raw(feature, stations)
        for stn in stations:
            served = dyn[stn][feature].reindex(raw.index)
            # no raw date is dropped and no value is changed, NaNs included
            assert np.array_equal(served.to_numpy(), raw[stn].to_numpy(dataset.fp),
                                  equal_nan=True), f"{stn}:{feature} differs from the raw file"
            # the float32 cast is harmless: relative error below 1e-7, tiny values included
            ok = (raw[stn] != 0) & raw[stn].notna()
            err = np.abs(served[ok].to_numpy(np.float64) / raw[stn][ok] - 1)
            assert (err < 1e-7).all(), f"{stn}:{feature} precision loss {err.max()}"
            # dates outside this file's record are NaN
            outside = ~dyn[stn].index.isin(raw.index)
            assert dyn[stn].loc[outside, feature].isna().all(), f"{stn}:{feature} filled outside its file"
    return


def _number(text: str):
    """``text`` as a number, NaN if it is a missing value, None if it is not a number"""
    if text in ('NA', ''):
        return float('nan')
    try:
        return float(text)
    except ValueError:
        return None


def _assert_same_attributes(static: pd.DataFrame, raw: dict, renamed: dict):
    """``static`` equals ``raw`` (attribute -> its values as text, in the order of
    ``static.index``): numbers exactly, whole numbers as integers, text unchanged"""
    assert list(static.columns) == [renamed.get(name, name) for name in raw]
    for name, values in raw.items():
        served = static[renamed.get(name, name)]
        values = [value.strip() for value in values]
        numbers = [_number(value) for value in values]
        if None in numbers:  # a text attribute
            assert served.dtype == object, name
            assert [v.strip() if isinstance(v, str) else 'NA' for v in served] == \
                   [value or 'NA' for value in values], name
        else:
            assert np.array_equal(served.to_numpy(float), numbers, equal_nan=True), name
            if all(value.lstrip('-').isdigit() for value in values):
                assert served.dtype.kind == 'i', name
    return


def test_registration():
    """the class is registered and can be built by the RainfallRunoff factory"""
    logger.info("test_registration")
    assert 'CAMELS_CL' in aqua_fetch.ALL_DATASETS
    from aqua_fetch import RainfallRunoff
    # remove_zip defaults to True in RainfallRunoff and would delete the downloaded archive
    rr = RainfallRunoff('CAMELS_CL', path=CAMELS_CL_PATH, remove_zip=False, verbosity=0)
    assert len(rr.stations()) == NUM_STATIONS
    return


def test_version_argument():
    """2022 is the default, each release has its own cache and an unknown
    version is refused before anything is created on disk"""
    logger.info("test_version_argument")
    assert dataset.version == 2022
    assert dataset.dyn_fname == 'camels_cl_D_2022_v2.nc', dataset.dyn_fname
    # a numpy integer is a valid version and names the same cache
    assert CAMELS_CL(path=CAMELS_CL_PATH, version=np.int64(2022), verbosity=0).dyn_fpath == dataset.dyn_fpath

    tmp = tempfile.mkdtemp()
    try:
        # 2022.0 == 2022, but would name a second cache camels_cl_D_2022.0_v2.nc
        for bad in (2020, '2022', None, 2022.0, True):
            try:
                CAMELS_CL(path=tmp, version=bad, verbosity=0)
            except ValueError:
                pass
            else:
                raise AssertionError(f"version={bad!r} was accepted")
        assert os.listdir(tmp) == [], os.listdir(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_feature_names():
    """standardized names, the raw names they replace are gone"""
    logger.info("test_feature_names")
    assert dataset.dynamic_features == list(RAW_FILES), dataset.dynamic_features

    static = dataset.static_features
    assert len(static) == len(set(static)) == NUM_STATIC, len(static)
    for name in ('area_km2', 'lat', 'long', 'slope_%', 'elev_catch_m', 'elev_catch_med_m',
                 'elev_catch_min_m', 'elev_catch_max_m', 'gauge_name'):
        assert name in static, name
    for raw in ('gauge_lat', 'gauge_lon', 'mean_slope_perc', 'mean_elev', 'med_elev', 'min_elev', 'max_elev'):
        assert raw not in static, raw
    return


def test_read_dynamic_fidelity():
    """the csv reader serves the raw values, with threads for a few stations,
    processes for many stations or float64, and neither when processes=1"""
    logger.info("test_read_dynamic_fidelity")
    stations = tuple(random.sample(dataset.stations(), 5))

    real_pools = {name: getattr(_camels.cf, name) for name in ('ThreadPoolExecutor', 'ProcessPoolExecutor')}
    used = []

    def recording(name):
        def pool(*args, **kwargs):
            used.append(name)
            return real_pools[name](*args, **kwargs)
        return pool

    try:
        for name in real_pools:
            setattr(_camels.cf, name, recording(name))
        dyn = dataset._read_dynamic(list(stations), 'all')
        assert used == ['ThreadPoolExecutor'], f"a few stations should be read with threads: {used}"

        many = dataset._read_dynamic(dataset.stations()[:100], ['q_mm_obs', 'airtemp_C_max'])
        assert used[1:] == ['ProcessPoolExecutor'], f"many stations should be read with processes: {used}"

        # float64 is parsed exactly, with processes because that parser holds the GIL
        exact = copy.copy(dataset)
        exact.fp = np.float64
        dyn64 = exact._read_dynamic(list(stations[:2]), 'all')
        assert used[2:] == ['ProcessPoolExecutor'], f"float64 should be read with processes: {used}"

        serial = copy.copy(dataset)
        serial.processes = 1
        dyn_serial = serial._read_dynamic(list(stations), 'all')
        many_serial = serial._read_dynamic(dataset.stations()[:100], ['q_mm_obs', 'airtemp_C_max'])
        assert len(used) == 3, f"processes=1 must not start threads or processes: {used}"
    finally:
        for name, pool in real_pools.items():
            setattr(_camels.cf, name, pool)

    for stn in many:
        assert many[stn].equals(many_serial[stn]), stn

    for feature in RAW_FILES:
        raw = _raw(feature, stations)
        for stn in stations[:2]:
            assert np.array_equal(dyn64[stn][feature].reindex(raw.index).to_numpy(), raw[stn].to_numpy(),
                                  equal_nan=True), f"float64 {stn}:{feature} differs from the raw file"

    _assert_faithful(dyn, stations)
    for stn in stations:
        assert dyn[stn].shape == (DYN_LEN, NUM_DYNAMIC), dyn[stn].shape
        assert (dyn[stn].dtypes == dataset.fp).all()
        assert dyn[stn].equals(dyn_serial[stn]), stn

    # _read_stn_dyn is the same reader for one station
    assert dataset._read_stn_dyn(stations[0]).equals(dyn[stations[0]])
    return


def test_fetch_fidelity():
    """fetch (from the netCDF cache) serves the raw values"""
    logger.info("test_fetch_fidelity")
    assert dataset.dyn_fpath_exists or xr is None
    stations = tuple(random.sample(dataset.stations(), 5))
    _, dyn = dataset.fetch(list(stations), as_dataframe=True)
    _assert_faithful(dyn, stations)

    # a flood of 4371 m3/s, which the 2018 release stores as 4.0
    _, q = dataset.fetch('7383001', dynamic_features='q_cms_obs', st='2017-06-26',
                         en='2017-06-26', as_dataframe=True)
    assert q['7383001']['q_cms_obs'].tolist() == [4371.0], q['7383001']

    # a tiny value that pandas' default csv parser reads as 0
    raw = pd.read_csv(os.path.join(RAW_DIR, RAW_FILES['pcp_mm_cr2met']), usecols=['date', '9437002'],
                      dtype=str, index_col='date')
    assert raw.loc['2001-10-27', '9437002'] == '0.0000000000000000015595638'
    _, p = dataset.fetch('9437002', dynamic_features='pcp_mm_cr2met', st='2001-10-27',
                         en='2001-10-27', as_dataframe=True)
    assert p['9437002']['pcp_mm_cr2met'].tolist() == [np.float32(1.5595638e-18)], p['9437002']
    return


def test_time_index():
    """the time index is the union of the dates in the raw files (no date is
    invented or dropped) and start/end are its first and last dates"""
    logger.info("test_time_index")
    dates = pd.DatetimeIndex([])
    for fname in RAW_FILES.values():
        raw = pd.read_csv(os.path.join(RAW_DIR, fname), usecols=['date'], parse_dates=['date'])
        dates = dates.union(pd.DatetimeIndex(raw['date']))
    assert len(dates) == DYN_LEN, len(dates)

    stn = random.choice(dataset.stations())
    assert dataset._read_stn_dyn(stn).index.equals(dates.rename('time'))
    _, dyn = dataset.fetch(stn, as_dataframe=True)
    assert dyn[stn].index.equals(dates)
    # also when the first features read do not cover all dates, as in the netCDF cache
    for features in (['pcp_mm_tmpa'], ['airtemp_C_min', 'pcp_mm_chirps']):
        fresh = copy.copy(dataset)
        fresh._all_dates = None  # as in a new instance
        assert fresh._read_dynamic([stn], features)[stn].index.equals(dates), features
        _, dyn = dataset.fetch(stn, dynamic_features=features, as_dataframe=True)
        assert dyn[stn].index.equals(dates), features
    assert dataset.start == dates[0] == pd.Timestamp('1900-01-01'), dataset.start
    assert dataset.end == dates[-1] == pd.Timestamp('2021-06-22'), dataset.end
    return


def test_units():
    """values are consistent with the documented units"""
    logger.info("test_units")
    _, dyn = dataset.fetch(list(random.sample(dataset.stations(), 50)), as_dataframe=True)
    df = pd.concat(dyn.values())
    area = dataset.area(list(dyn))

    # q_mm_obs (mm/day) == q_cms_obs (m3/s) * 86400 s / (area km2 * 1e6 m2) * 1000 mm/m
    for stn, stn_df in dyn.items():
        both = stn_df[['q_cms_obs', 'q_mm_obs']].dropna()
        both = both[both['q_cms_obs'] > 0]
        expected = both['q_cms_obs'].astype(float) * 86.4 / float(area[stn])
        assert np.allclose(both['q_mm_obs'], expected, rtol=1e-5), stn

    temps = df[['airtemp_C_min', 'airtemp_C_mean', 'airtemp_C_max']].dropna()
    assert temps.min().min() > -40 and temps.max().max() < 50, temps.describe()
    assert (temps['airtemp_C_min'] <= temps['airtemp_C_max']).all()
    for name in ('pcp_mm_cr2met', 'pcp_mm_chirps', 'pcp_mm_mswep', 'pcp_mm_tmpa',
                 'pet_mm_hargreaves', 'q_cms_obs', 'q_mm_obs'):
        assert df[name].min() >= 0, name
    assert df['pet_mm_hargreaves'].max() < 20, "Hargreaves PET is not mm/day"
    return


def test_q_mm():
    """q_mm returns the raw q_mm_day values, not a conversion from m3/s"""
    logger.info("test_q_mm")
    stations = tuple(random.sample(dataset.stations(), 2))
    q = dataset.q_mm(list(stations))
    raw = _raw('q_mm_obs', stations)
    for stn in stations:
        assert np.array_equal(q[stn].reindex(raw.index).to_numpy(), raw[stn].to_numpy(dataset.fp),
                              equal_nan=True), stn
    return


def test_static_fidelity():
    """static attributes equal catchment_attributes.csv, read independently by the csv module"""
    logger.info("test_static_fidelity")
    with open(os.path.join(RAW_DIR, 'catchment_attributes.csv'), newline='') as f:
        rows = list(csv.reader(f))
    ids = [row[0] for row in rows[1:]]
    assert dataset.stations() == ids
    static = dataset.fetch_static_features('all', 'all')
    assert static.shape == (NUM_STATIONS, NUM_STATIC)

    raw = {name: [row[i] for row in rows[1:]] for i, name in enumerate(rows[0]) if i > 0}
    renamed = {'gauge_lat': 'lat', 'gauge_lon': 'long', 'mean_slope_perc': 'slope_%',
               'mean_elev': 'elev_catch_m', 'med_elev': 'elev_catch_med_m',
               'min_elev': 'elev_catch_min_m', 'max_elev': 'elev_catch_max_m'}
    _assert_same_attributes(static.loc[ids], raw, renamed)

    coords = dataset.stn_coords()
    assert np.allclose(coords['lat'], [float(v) for v in raw['gauge_lat']], atol=1e-4)
    assert np.allclose(coords['long'], [float(v) for v in raw['gauge_lon']], atol=1e-4)
    assert np.allclose(dataset.area(), [float(v) for v in raw['area_km2']], rtol=1e-6)
    return


def test_returns_copies():
    """editing returned lists and frames does not change the dataset"""
    logger.info("test_returns_copies")
    stns = dataset.stations()
    stns.append('XXX')
    assert 'XXX' not in dataset.stations()

    feats = dataset.static_features
    feats.append('YYY')
    assert 'YYY' not in dataset.static_features

    static = dataset._static_data()
    static.loc[:, 'area_km2'] = -1
    assert (dataset._static_data()['area_km2'] > 0).all()
    return


def test_fetch_period():
    """st/en select a period; a period outside the data is empty"""
    logger.info("test_fetch_period")
    stn = dataset.stations()[0]
    _, inside = dataset.fetch(stn, st='2000-01-01', en='2000-12-31', as_dataframe=True)
    assert len(inside[stn]) == 366, inside[stn].shape
    csv = dataset._read_dynamic([stn], 'all', st='2000-01-01', en='2000-12-31')[stn]
    assert csv.index.equals(inside[stn].index.rename('time'))
    assert np.array_equal(csv.to_numpy(), inside[stn].to_numpy(), equal_nan=True)

    before = dataset._read_dynamic([stn], 'all', st='1800-01-01', en='1850-12-31')[stn]
    assert before.shape == (0, NUM_DYNAMIC), before.shape
    return


def test_boundaries():
    """every gauge has a WGS84 boundary inside Chile, the one of the 2022 shapefile"""
    logger.info("test_boundaries")
    if fiona is None:
        return
    assert sorted(dataset._create_boundary_id_map()) == sorted(dataset.stations())
    with fiona.open(os.path.join(RAW_DIR, 'camels_cl_boundaries', 'camels_cl_boundaries.shp')) as src:
        raw = {str(int(feature['properties']['gauge_id'])): feature['geometry'] for feature in src}
    from aqua_fetch.rr.utils import _make_boundary_2d
    for stn in random.sample(dataset.stations(), 5) + ['4714001']:  # 4714001 was redrawn in 2021
        geometry = dataset.get_boundary(stn)
        assert geometry.coordinates == raw[stn].coordinates, stn
        for ring in _make_boundary_2d(geometry):
            assert ((ring[:, 0] > -76) & (ring[:, 0] < -66)).all(), stn
            assert ((ring[:, 1] > -56) & (ring[:, 1] < -17)).all(), stn
    return


def test_init_warnings():
    """initialization warns loudly about missing files and duplicate gauges, and
    stays quiet for the complete data"""
    logger.info("test_init_warnings")

    def boom(*args, **kwargs):
        raise AssertionError("nothing must be downloaded")

    original = _camels.download
    _camels.download = boom
    tmp = tempfile.mkdtemp()
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CAMELS_CL(path=CAMELS_CL_PATH, verbosity=0)
        msgs = [str(x.message) for x in w]
        assert not [m for m in msgs if 'missing' in m or 'duplicate' in m], msgs

        # an extracted 2022 folder holding only the attributes, with the first gauge twice
        root = os.path.join(tmp, 'CAMELS_CL', 'CAMELS_CL_v202201', 'CAMELS_CL_v202201')
        os.makedirs(root)
        with open(os.path.join(RAW_DIR, 'catchment_attributes.csv'), 'rb') as f:
            header, first = f.readline(), f.readline()
        with open(os.path.join(root, 'catchment_attributes.csv'), 'wb') as f:
            f.writelines([header, first, first])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CAMELS_CL(path=tmp, to_netcdf=False, verbosity=0)
        msgs = [str(x.message) for x in w]
        assert any('14 of 15 files are missing' in m for m in msgs), msgs
        assert any('duplicate' in m for m in msgs), msgs
    finally:
        _camels.download = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_no_redownload_or_reextract():
    """initializing again neither downloads, extracts nor rebuilds the cache"""
    logger.info("test_no_redownload_or_reextract")

    def boom(*args, **kwargs):
        raise AssertionError("must not be called when the data exists")

    originals = _camels.download, zipfile.ZipFile, (xr.Dataset.to_netcdf if xr is not None else None)
    _camels.download = zipfile.ZipFile = boom
    if xr is not None:
        xr.Dataset.to_netcdf = boom
    try:
        again = CAMELS_CL(path=CAMELS_CL_PATH, verbosity=0)
        assert len(again.stations()) == NUM_STATIONS
    finally:
        _camels.download, zipfile.ZipFile = originals[:2]
        if xr is not None:
            xr.Dataset.to_netcdf = originals[2]
    return


def _patched_download(tmp, version, on_download):
    """a copy of the dataset for ``version`` in ``tmp``, and a stand-in for
    ``download`` that records its calls and writes a small zip archive"""
    calls = []

    def download(url, outdir, fname, verbosity, headers):
        calls.append((url, outdir, fname, headers))
        on_download()
        with zipfile.ZipFile(os.path.join(outdir, fname), 'w') as zf:
            zf.writestr('data.csv', 'date,1001001\n2000-01-01,1.5\n')

    ds = copy.copy(dataset)
    ds._path, ds.version = tmp, version
    return ds, calls, download


def test_download():
    """a missing release is downloaded into a new folder (2022 from cr2.cl with a
    browser user agent, 2018 as 15 archives from PANGAEA) and each archive is
    extracted into its own folder; once extracted nothing is downloaded again,
    and remove_zip then deletes the archives"""
    logger.info("test_download")
    for version, n_archives, host, headers in ((2022, 1, 'www.cr2.cl', BROWSER_HEADERS),
                                               (2018, 15, 'store.pangaea.de', None)):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, 'CAMELS_CL')  # does not exist yet
        original = _camels.download
        try:
            ds, calls, _camels.download = _patched_download(path, version, lambda: None)
            ds._download_camels_cl()

            assert len(calls) == n_archives, calls
            for url, outdir, fname, sent in calls:
                assert url.startswith(f'https://{host}/') and outdir == path and fname.endswith('.zip'), calls
                assert sent == headers, (version, sent)
            folders = [fname[:-len('.zip')] for _, _, fname, _ in calls]
            assert sorted(os.listdir(path)) == sorted(folders + [fname for _, _, fname, _ in calls])
            for folder in folders:
                assert os.listdir(os.path.join(path, folder)) == ['data.csv'], folder

            calls.clear()
            ds.remove_zip = True
            ds._download_camels_cl()
            assert calls == [], f"downloaded again although the data is extracted: {calls}"
            assert sorted(os.listdir(path)) == sorted(folders), "remove_zip left archives"
        finally:
            _camels.download = original
            shutil.rmtree(tmp, ignore_errors=True)
    return


def test_interrupted_extraction():
    """an extraction that stops half way leaves no folder that looks complete,
    so the next initialization extracts the archive again"""
    logger.info("test_interrupted_extraction")
    tmp = tempfile.mkdtemp()
    original_download, original_extractall = _camels.download, zipfile.ZipFile.extractall

    def interrupted(self, path, *args, **kwargs):
        os.makedirs(path, exist_ok=True)
        open(os.path.join(path, 'half_written.csv'), 'wb').close()
        raise KeyboardInterrupt

    try:
        ds, calls, _camels.download = _patched_download(tmp, 2022, lambda: None)
        zipfile.ZipFile.extractall = interrupted
        try:
            ds._download_camels_cl()
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("the extraction was not interrupted")
        finally:
            zipfile.ZipFile.extractall = original_extractall
        assert not os.path.exists(os.path.join(tmp, 'CAMELS_CL_v202201'))

        ds._download_camels_cl()
        assert len(calls) == 1, f"the archive was downloaded again: {calls}"
        assert sorted(os.listdir(tmp)) == ['CAMELS_CL_v202201', 'CAMELS_CL_v202201.zip'], os.listdir(tmp)
        assert os.listdir(os.path.join(tmp, 'CAMELS_CL_v202201')) == ['data.csv']
    finally:
        _camels.download = original_download
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_partial_and_corrupt_downloads():
    """of a partly downloaded 2018 release only the missing parts are downloaded
    or extracted; an archive that is not a zip file is deleted with a clear error
    and downloaded again on the next initialization"""
    logger.info("test_partial_and_corrupt_downloads")
    tmp = tempfile.mkdtemp()
    original = _camels.download
    try:
        ds, calls, _camels.download = _patched_download(tmp, 2018, lambda: None)
        stems = [fname[:-len('.zip')] for fname in CAMELS_CL.urls[2018]]
        for stem in stems[:13]:  # extracted, and one of them keeps its archive
            os.makedirs(os.path.join(tmp, stem))
            open(os.path.join(tmp, stem, 'original.csv'), 'wb').close()
        open(os.path.join(tmp, f'{stems[0]}.zip'), 'wb').close()
        with zipfile.ZipFile(os.path.join(tmp, f'{stems[13]}.zip'), 'w') as zf:  # downloaded only
            zf.writestr('data.csv', 'x')
        os.makedirs(os.path.join(tmp, f'{stems[13]}_extracting'))  # an interrupted extraction
        # stems[14] is missing altogether

        ds._download_camels_cl()
        assert [c[2] for c in calls] == [f'{stems[14]}.zip'], calls
        for stem in stems[:13]:
            assert os.listdir(os.path.join(tmp, stem)) == ['original.csv'], stem
        for stem in stems[13:]:
            assert os.listdir(os.path.join(tmp, stem)) == ['data.csv'], stem
        assert not [f for f in os.listdir(tmp) if f.endswith('_extracting')]

        # an error page saved as the 2022 archive, and a zip file whose compressed data is damaged
        ds.version = 2022
        archive = os.path.join(tmp, 'CAMELS_CL_v202201.zip')
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('data.csv', ''.join(f'2000-01-01,{i * 0.37}\n' for i in range(20000)))
        with open(archive, 'r+b') as f:
            f.seek(200)
            f.write(b'\x00' * 50)
        damaged = open(archive, 'rb').read()
        for content in (b'<html>Service unavailable</html>', damaged):
            with open(archive, 'wb') as f:
                f.write(content)
            try:
                ds._download_camels_cl()
            except ValueError as e:
                assert 'is corrupt' in str(e), e
            else:
                raise AssertionError("a corrupt archive was accepted")
            assert not os.path.exists(archive), "the corrupt archive was kept"
            assert not os.path.exists(f"{archive[:-len('.zip')]}_extracting"), "the partial extraction was kept"
        ds._download_camels_cl()
        assert calls[-1][2] == 'CAMELS_CL_v202201.zip', calls
        assert os.listdir(os.path.join(tmp, 'CAMELS_CL_v202201')) == ['data.csv']
    finally:
        _camels.download = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_date_checks():
    """start/end are the first and last dates over all files, a missing file
    raises instead of narrowing them, and a file whose dates are not sorted or
    repeat a date is refused instead of being misread"""
    logger.info("test_date_checks")
    tmp = tempfile.mkdtemp()

    def write(fpath, dates):
        with open(fpath, 'w') as f:
            f.write('date,year,month,day,1001001\n' + ''.join(f'{d},0,0,0,1.0\n' for d in dates))

    try:
        ds = copy.copy(dataset)
        ds._path = tmp
        root = os.path.join(tmp, 'CAMELS_CL_v202201', 'CAMELS_CL_v202201')
        os.makedirs(root)
        for fname in RAW_FILES.values():
            write(os.path.join(root, fname), ['2000-01-02', '2000-01-03'])
        write(os.path.join(root, 'precip_tmpa_mm_day.csv'), ['2000-01-01', '2000-01-02', '2000-01-05'])
        ds.__dict__.pop('_time_extent', None)
        assert (ds.start, ds.end) == (pd.Timestamp('2000-01-01'), pd.Timestamp('2000-01-05'))

        os.remove(os.path.join(root, 'precip_tmpa_mm_day.csv'))
        ds.__dict__.pop('_time_extent', None)
        try:
            _ = ds.start
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("a missing file silently narrowed start/end")

        fpath = os.path.join(root, 'q_m3s_day.csv')
        for dates in (['2000-01-03', '2000-01-02'], ['2000-01-02', '2000-01-02']):
            write(fpath, dates)
            for read in (lambda: _camels._read_camels_cl_dates(fpath, b','),
                         lambda: _camels._read_camels_cl_ts(fpath, ['1001001'], np.float32, 'legacy',
                                                            ',', 'date', None)):
                try:
                    read()
                except ValueError as e:
                    assert 'not sorted or not unique' in str(e), e
                else:
                    raise AssertionError(f"dates {dates} were accepted")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_float64():
    """with float_precision=np.float64 fetch serves the exact values, also when
    the float32 cache exists, because each precision has its own cache"""
    logger.info("test_float64")
    ds64 = CAMELS_CL(path=CAMELS_CL_PATH, float_precision=np.float64, to_netcdf=False, verbosity=0)
    assert ds64.dyn_fname == 'camels_cl_D_2022_float64_v2.nc', ds64.dyn_fname
    assert dataset.dyn_fname == 'camels_cl_D_2022_v2.nc', dataset.dyn_fname

    _, p = ds64.fetch('9437002', dynamic_features='pcp_mm_cr2met', st='2001-10-27',
                      en='2001-10-27', as_dataframe=True)
    assert p['9437002']['pcp_mm_cr2met'].dtype == np.float64
    assert p['9437002']['pcp_mm_cr2met'].tolist() == [float('0.0000000000000000015595638')], p['9437002']
    return


def test_download_helper_headers():
    """``utils.download`` sends ``headers`` when given, and is unchanged without
    them. A local server stands in for cr2.cl, which refuses urllib's user agent."""
    logger.info("test_download_helper_headers")
    import threading
    import urllib.error
    from http.server import HTTPServer, BaseHTTPRequestHandler
    from aqua_fetch.utils import download

    import http.client
    payload = b'PK\x03\x04 not really a zip'

    class Handler(BaseHTTPRequestHandler):
        """/cr2/ refuses urllib's user agent, /open/ does not; /short/ announces
        more bytes than it sends and /chunked/ stops within a chunk, like dropped
        connections (cr2.cl sends the archive in chunks)"""
        def do_GET(self):
            if not self.path.startswith('/open/') and 'Mozilla' not in self.headers.get('User-Agent', ''):
                self.send_error(403)
                return
            self.send_response(200)
            if self.path.startswith('/chunked/'):
                self.send_header('Transfer-Encoding', 'chunked')
                self.end_headers()
                self.wfile.write(b'%x\r\n' % (len(payload) * 50) + payload)
                return
            self.send_header('Content-Length', str(len(payload) * (50 if self.path.startswith('/short/') else 1)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def url(route):
        return f"http://127.0.0.1:{server.server_address[1]}/{route}/?wpdmdl=1"

    tmp = tempfile.mkdtemp()  # download() writes its temporary file here too
    try:
        for route, headers, error in (('cr2', None, urllib.error.HTTPError),
                                      ('short', BROWSER_HEADERS, urllib.error.ContentTooShortError),
                                      ('chunked', BROWSER_HEADERS, http.client.IncompleteRead)):
            try:
                download(url(route), outdir=tmp, fname='a.zip', verbosity=0, headers=headers)
            except error:
                pass
            else:
                raise AssertionError(f"{route}: the failed download was accepted")
            # no partial temporary file or archive is left
            assert os.listdir(tmp) == [], (route, os.listdir(tmp))

        fpath = download(url('cr2'), outdir=tmp, fname='a.zip', verbosity=0, headers=BROWSER_HEADERS)
        assert fpath == os.path.join(tmp, 'a.zip')
        with open(fpath, 'rb') as f:
            assert f.read() == payload

        # without headers, as before: the file is saved, and a second copy gets a '1' suffix
        for suffix in ('', '1'):
            fpath = download(url('open'), outdir=tmp, fname='b.zip', verbosity=0)
            assert fpath == os.path.join(tmp, f'b.zip{suffix}'), fpath
            with open(fpath, 'rb') as f:
                assert f.read() == payload
        assert sorted(os.listdir(tmp)) == ['a.zip', 'b.zip', 'b.zip1'], os.listdir(tmp)
    finally:
        server.shutdown()
        server.server_close()
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_overwrite_removes_stale_before_download():
    """overwrite=True deletes the archives, extracted folders and cache of one
    release before downloading it, and leaves the other release alone"""
    logger.info("test_overwrite_removes_stale_before_download")
    for version, other in ((2022, 2018), (2018, 2022)):
        tmp = tempfile.mkdtemp(prefix='cl[1]')  # '[' must not break finding the caches

        def fake_release(v):
            """cache, archives and extracted folders of release ``v`` in ``tmp``"""
            archives = [os.path.join(tmp, fname) for fname in CAMELS_CL.urls[v]]
            folders = [archive[:-len('.zip')] for archive in archives]
            files = [os.path.join(tmp, f'camels_cl_D_{v}_v2.nc'),
                     os.path.join(tmp, f'camels_cl_D_{v}_float64_v2.nc'), *archives,
                     *[os.path.join(folder, 'data.csv') for folder in folders]]
            for folder in folders:
                os.makedirs(folder)
            for fpath in files:
                open(fpath, 'wb').close()
            return files + folders

        stale, kept = fake_release(version), fake_release(other)
        checked = []

        def stale_gone():
            if not checked:  # the first download
                checked.append(True)
                assert not any(map(os.path.exists, stale)), "stale files still exist when downloading"

        original = _camels.download
        try:
            ds, calls, _camels.download = _patched_download(tmp, version, stale_gone)
            ds._download_camels_cl(overwrite=True)
            assert checked and len(calls) == len(CAMELS_CL.urls[version]), calls
            assert all(map(os.path.exists, kept)), f"overwriting {version} removed {other} files"
        finally:
            _camels.download = original
            shutil.rmtree(tmp, ignore_errors=True)
    return


def check_tmin_2018(ds):
    """minimum temperature of the 2018 release is served until 2016-12-31, as in
    its raw file. The class used to cut it at 2010-03-09."""
    stations = ['8350001', '1001001', '12876004']
    raw = pd.read_csv(os.path.join(ds.path, '8_CAMELScl_tmin_cr2met', '8_CAMELScl_tmin_cr2met.txt'),
                      sep='\t', index_col='gauge_id', usecols=['gauge_id', *stations],
                      na_values=' ', parse_dates=True)
    _, dyn = ds.fetch(stations, dynamic_features='airtemp_C_min', as_dataframe=True)
    for stn in stations:
        served = dyn[stn]['airtemp_C_min'].dropna()
        assert served.index[-1] == pd.Timestamp('2016-12-31'), f"{stn} tmin ends {served.index[-1]}"
        assert np.array_equal(dyn[stn]['airtemp_C_min'].reindex(raw.index).to_numpy(np.float32),
                              raw[stn].to_numpy(np.float32), equal_nan=True), stn
    return


def test_version_2018():
    """the 2018 release is read by changing only ``version`` and its minimum
    temperature is not truncated"""
    logger.info("test_version_2018")

    def boom(*args, **kwargs):
        raise AssertionError("the 2018 data exists, nothing must be downloaded")

    # the user's 2018 archives lie next to their folders; they must not be extracted again
    originals = _camels.download, zipfile.ZipFile
    _camels.download = zipfile.ZipFile = boom
    try:
        ds = CAMELS_CL(path=CAMELS_CL_PATH, version=2018, to_netcdf=False, verbosity=0)
    finally:
        _camels.download, zipfile.ZipFile = originals

    assert ds.path == dataset.path and ds.dyn_fpath != dataset.dyn_fpath
    assert len(ds.stations()) == NUM_STATIONS
    assert len(ds.static_features) == 104
    assert ds.dynamic_features == list(RAW_FILES_2018), ds.dynamic_features
    assert (ds.start, ds.end) == (pd.Timestamp('1913-02-15'), pd.Timestamp('2018-03-09'))
    check_tmin_2018(ds)

    # all features equal the raw files, on the union of their dates
    stations = list(dict.fromkeys(['8350001', '1001001', *random.sample(ds.stations(), 2)]))
    dyn = ds._read_dynamic(stations, 'all')
    for feature, stem in RAW_FILES_2018.items():
        raw = pd.read_csv(os.path.join(ds.path, stem, f'{stem}.txt'), sep='\t', index_col='gauge_id',
                          usecols=['gauge_id', *stations], na_values=' ', parse_dates=True,
                          float_precision='round_trip')
        for stn in stations:
            served = dyn[stn][feature]
            assert len(served) == 38374, (stn, feature, len(served))
            assert np.array_equal(served.reindex(raw.index).to_numpy(), raw[stn].to_numpy(ds.fp),
                                  equal_nan=True), (stn, feature)
            assert served[~served.index.isin(raw.index)].isna().all(), (stn, feature)

    # the static attributes, compared with an independent read by the csv module.
    # In the file, attributes are rows and gauges are columns.
    with open(os.path.join(ds.path, '1_CAMELScl_attributes', '1_CAMELScl_attributes.txt'), newline='') as f:
        rows = list(csv.reader(f, delimiter='\t'))
    ids = [gauge.strip() for gauge in rows[0][1:]]
    renamed = {'area': 'area_km2', 'slope_mean': 'slope_mkm-1', 'gauge_lat': 'lat', 'gauge_lon': 'long'}
    _assert_same_attributes(ds._static_data().loc[ids], {row[0]: row[1:] for row in rows[1:]}, renamed)
    return


def test_efficiency():
    """all stations and features are fetched in seconds, from the cache and from
    the csv files, and so is a tenth of the stations from the csv files"""
    logger.info("test_efficiency")
    reads = (  # name, number of stations, limit in seconds, read
        ('all stations from the cache', NUM_STATIONS, 10, lambda: dataset.fetch('all', as_dataframe=True)[1]),
        ('all stations from csv', NUM_STATIONS, 30, lambda: dataset._read_dynamic(dataset.stations(), 'all')),
        ('51 stations from csv', 51, 10, lambda: dataset._read_dynamic(random.sample(dataset.stations(), 51), 'all')),
    )
    timings = {}
    for name, n_stations, limit, read in reads:
        start = time.time()
        dyn = read()
        timings[name] = round(time.time() - start, 2)
        assert len(dyn) == n_stations, name
        assert timings[name] < limit, timings
    logger.info(f"seconds to read all features: {timings}")
    return


def test_shared_suite():
    """the checks shared by all rainfall-runoff datasets"""
    logger.info("test_shared_suite")
    run_shared_tests(dataset,
                     num_stations=NUM_STATIONS,
                     dyn_data_len=DYN_LEN,
                     num_static_attrs=NUM_STATIC,
                     num_dyn_attrs=NUM_DYNAMIC,
                     yearly_steps=366,
                     st="20040101", en="20041231")
    return


def test_shared_suite_2018():
    """the same checks for the 2018 release (moved here from test_camels.py)"""
    logger.info("test_shared_suite_2018")
    run_shared_tests(CAMELS_CL(path=CAMELS_CL_PATH, version=2018, verbosity=0),
                     num_stations=NUM_STATIONS,
                     dyn_data_len=38374,  # daily steps 1913-02-15 .. 2018-03-09
                     num_static_attrs=104,
                     num_dyn_attrs=len(RAW_FILES_2018),
                     yearly_steps=366,
                     st="20040101", en="20041231")
    return


if __name__ == "__main__":
    random.seed(313)

    test_registration()
    test_version_argument()
    test_feature_names()
    test_read_dynamic_fidelity()
    test_fetch_fidelity()
    test_time_index()
    test_units()
    test_q_mm()
    test_static_fidelity()
    test_returns_copies()
    test_fetch_period()
    test_boundaries()
    test_init_warnings()
    test_no_redownload_or_reextract()
    test_download()
    test_interrupted_extraction()
    test_partial_and_corrupt_downloads()
    test_date_checks()
    test_float64()
    test_download_helper_headers()
    test_overwrite_removes_stale_before_download()
    test_version_2018()
    test_efficiency()
    test_shared_suite()
    test_shared_suite_2018()

    print("*** All CAMELS_CL tests passed ***")
