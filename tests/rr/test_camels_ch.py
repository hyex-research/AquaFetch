"""
Tests for CAMELS_CH (hydrologic Switzerland, 331 catchments), version 0.9.

They check that the class

    * serves the raw values and units of the source files unchanged, at both
      timesteps, compared with an independent read of those files,
    * reads version 0.9 (comma separated, ``timeseries`` folder) and keeps an
      older release on disk untouched,
    * downloads only the files it reads, downloads and extracts once, and on
      ``overwrite=True`` deletes the stale files first,
    * serves hourly discharge for ``timestep='H'``, from the file the inventory
      names, with the ``-9999`` no-data marker as NaN,
    * honours ``to_netcdf``, ``overwrite`` and ``float_precision``,
    * fetches all stations quickly.

The first run downloads 247 MB (and 417 MB more for the hourly tests) into
``CAMELS_CH_PATH``. Run as a script or with pytest.
"""

import os
import copy
import site
import time
import random
import shutil
import logging
import zipfile
import tempfile
import warnings

import numpy as np
import pandas as pd

# add the repository root to the path so that ``aqua_fetch`` and the shared
# ``utils`` test helpers can be imported
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

if __name__ == "__main__":
    logging.basicConfig(filename='test_camels_ch.log', filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

import aqua_fetch
from aqua_fetch import CAMELS_CH
from aqua_fetch.rr import _camels
from aqua_fetch._backend import xarray as xr, netCDF4, fiona

from utils import test_dataset as run_shared_tests

# The class appends ``CAMELS_CH`` to this path.
# Replace with the location on your machine.
CAMELS_CH_PATH = '/mnt/hyexnas01/abbaa0a/data/gscad_database/raw/CAMELS/CAMELS'

NUM_STATIONS = 331
NUM_STATIC = 209
NUM_DYNAMIC = 9
DYN_LEN = 14610          # daily steps 1981-01-01 .. 2020-12-31
NUM_HOURLY_STATIONS = 170
HOURLY_LEN_2009 = 401017  # hourly steps of gauge 2009, 1974-01-01 .. 2019-10-01

# standardized dynamic feature -> its column in the raw csv files, in the order
# of those files. Written out here rather than taken from the class, so that a
# wrong mapping in the class is caught.
RAW_DYN = {
    'q_cms_obs': 'discharge_vol(m3/s)',
    'q_mm_obs': 'discharge_spec(mm/d)',
    'waterlevel(m)': 'waterlevel(m)',
    'pcp_mm': 'precipitation(mm/d)',
    'airtemp_C_min': 'temperature_min(degC)',
    'airtemp_C_mean': 'temperature_mean(degC)',
    'airtemp_C_max': 'temperature_max(degC)',
    'rel_sun_dur(%)': 'rel_sun_dur(%)',
    'swe_mm': 'swe(mm)',
}

# the ten static attribute files, as (file name, sub folder), and how many
# columns each of them contributes
RAW_STATIC = {
    'CAMELS_CH_climate_attributes_obs.csv': ('', 14),
    'CAMELS_CH_geology_attributes.csv': ('', 15),
    'CAMELS_CH_geology_attributes_supplement.csv': ('supplements', 21),
    'CAMELS_CH_glacier_attributes.csv': ('', 4),
    'CAMELS_CH_humaninfluence_attributes.csv': ('', 14),
    'CAMELS_CH_hydrogeology_attributes.csv': ('', 10),
    'CAMELS_CH_hydrology_attributes_obs.csv': ('', 16),
    'CAMELS_CH_landcover_attributes.csv': ('', 13),
    'CAMELS_CH_soil_attributes.csv': ('', 80),
    'CAMELS_CH_topographic_attributes.csv': ('', 22),
}

dataset = CAMELS_CH(path=CAMELS_CH_PATH, verbosity=0)

RAW_DIR = os.path.join(dataset.path, 'camels_ch_v0.9', 'camels_ch')


def _raw_daily(station: str) -> pd.DataFrame:
    """independent read of the raw daily file of one catchment, at full precision"""
    return pd.read_csv(
        os.path.join(RAW_DIR, 'timeseries', 'observation_based',
                     f'CAMELS_CH_obs_based_{station}.csv'),
        sep=',', index_col='date', parse_dates=True, float_precision='round_trip')


def _raw_hourly(fname: str) -> pd.Series:
    """independent read of one raw hourly discharge file, markers included"""
    df = pd.read_csv(os.path.join(dataset.path, 'DischargeDBHydroCH', 'DischargeDBHydroCH',
                                  'CH', 'FOEN', fname), sep='\t', float_precision='round_trip')
    index = pd.to_datetime(dict(year=df['YYYY'], month=df['MM'], day=df['DD'], hour=df['HH']))
    return pd.Series(df.iloc[:, 4].to_numpy(), index=index)


def _hourly_dataset() -> CAMELS_CH:
    """the hourly dataset, built once and reused by the hourly tests"""
    global _HOURLY
    try:
        return _HOURLY
    except NameError:
        _HOURLY = CAMELS_CH(path=CAMELS_CH_PATH, timestep='H', verbosity=0)
    return _HOURLY


def test_registration():
    """the class is registered and can be built by the RainfallRunoff factory"""
    logger.info("test_registration")
    assert 'CAMELS_CH' in aqua_fetch.ALL_DATASETS
    from aqua_fetch import RainfallRunoff
    # remove_zip defaults to True in RainfallRunoff and would delete the archive
    rr = RainfallRunoff('CAMELS_CH', path=CAMELS_CH_PATH, remove_zip=False, verbosity=0)
    assert len(rr.stations()) == NUM_STATIONS
    return


def test_version():
    """version 0.9 is read, from its own folder and into its own cache, and the
    pre-0.7 layout is not used"""
    logger.info("test_version")
    assert _camels._CH_VERSION == '0.9'
    assert dataset.url['camels_ch.zip'] == 'https://zenodo.org/records/15025258'
    assert dataset.camels_path == os.path.join(dataset.path, 'camels_ch_v0.9', 'camels_ch')
    assert dataset.dyn_fname == 'camels_ch_D_0.9_v2.nc', dataset.dyn_fname
    # version 0.7 renamed this folder and moved this file
    assert os.path.isdir(os.path.join(dataset.camels_path, 'timeseries'))
    assert not os.path.exists(os.path.join(dataset.camels_path, 'time_series'))
    assert os.path.exists(os.path.join(dataset.static_path, 'supplements',
                                       'CAMELS_CH_geology_attributes_supplement.csv'))
    return


def test_url_is_not_mutated():
    """building a dataset leaves the class level ``url`` alone, so that a later
    hourly dataset still knows where to download the hourly archive from"""
    logger.info("test_url_is_not_mutated")
    before = dict(CAMELS_CH.url)
    CAMELS_CH(path=CAMELS_CH_PATH, verbosity=0)
    assert CAMELS_CH.url == before, CAMELS_CH.url
    assert 'DischargeDBHydroCH.zip' in CAMELS_CH.url
    return


def test_timestep_is_validated():
    """an unknown timestep is refused before anything is created on disk"""
    logger.info("test_timestep_is_validated")
    tmp = tempfile.mkdtemp()
    try:
        for bad in ('h', 'd', 'hourly', None, 1):
            try:
                CAMELS_CH(path=tmp, timestep=bad, verbosity=0)
            except ValueError:
                pass
            else:
                raise AssertionError(f"timestep={bad!r} was accepted")
        assert os.listdir(tmp) == [], os.listdir(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_feature_names():
    """standardized names, in the order of the columns of the raw files"""
    logger.info("test_feature_names")
    assert dataset.dynamic_features == list(RAW_DYN), dataset.dynamic_features

    static = dataset.static_features
    assert len(static) == len(set(static)) == NUM_STATIC, len(static)
    assert sum(n for _, n in RAW_STATIC.values()) == NUM_STATIC
    for name in ('area_km2', 'lat', 'long', 'slope_degrees', 'gauge_name'):
        assert name in static, name
    for raw in ('area', 'gauge_lat', 'gauge_lon', 'slope_mean'):
        assert raw not in static, raw
    return


def test_stations():
    """all 331 catchments, as strings, in the order of the attribute files"""
    logger.info("test_stations")
    stations = dataset.stations()
    assert len(stations) == len(set(stations)) == NUM_STATIONS
    assert all(isinstance(stn, str) for stn in stations)
    assert stations[:3] == ['2004', '2007', '2009'], stations[:3]
    raw = pd.read_csv(os.path.join(dataset.static_path, 'CAMELS_CH_glacier_attributes.csv'),
                      sep=',', skiprows=1)['gauge_id']
    assert stations == [str(stn) for stn in raw]
    return


def test_time_index():
    """start and end are read from the files, not hardcoded, and every station
    has the same gapless daily index"""
    logger.info("test_time_index")
    assert dataset.start == pd.Timestamp('1981-01-01'), dataset.start
    assert dataset.end == pd.Timestamp('2020-12-31'), dataset.end

    expected = pd.date_range(dataset.start, dataset.end, freq='D')
    assert len(expected) == DYN_LEN
    for stn in random.sample(dataset.stations(), 5):
        assert dataset._read_stn_dyn(stn).index.equals(expected), stn

    # a station whose file reaches further must move the extent, which a
    # hardcoded start/end would not do
    tmp = tempfile.mkdtemp()
    try:
        ds = copy.copy(dataset)
        ds._path = tmp
        # copy.copy carries the caches of the original over
        for cached in ('_extent', '_all_stns', '_static', '_dyn_feats'):
            ds.__dict__.pop(cached, None)
        root = os.path.join(tmp, 'camels_ch_v0.9', 'camels_ch')
        os.makedirs(root)
        os.symlink(dataset.static_path, os.path.join(root, 'static_attributes'))
        ts_dir = os.path.join(root, 'timeseries', 'observation_based')
        os.makedirs(ts_dir)
        for fname in os.listdir(dataset.dynamic_path):
            os.symlink(os.path.join(dataset.dynamic_path, fname), os.path.join(ts_dir, fname))

        longer = os.path.join(ts_dir, 'CAMELS_CH_obs_based_2004.csv')
        rows = open(os.path.join(dataset.dynamic_path, 'CAMELS_CH_obs_based_2004.csv')).read()
        os.unlink(longer)
        with open(longer, 'w') as f:
            f.write(rows + '2021-01-01' + ',NaN' * NUM_DYNAMIC + '\n')

        assert ds.start == pd.Timestamp('1981-01-01'), ds.start
        assert ds.end == pd.Timestamp('2021-01-01'), ds.end
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_read_dynamic_fidelity():
    """the served daily values are the raw values cast to ``dataset.fp``, with
    no row, column or NaN changed"""
    logger.info("test_read_dynamic_fidelity")
    stations = random.sample(dataset.stations(), 5)
    dyn = dataset._read_dynamic(stations, 'all')

    for stn in stations:
        raw = _raw_daily(stn)
        served = dyn[stn]
        assert served.shape == (DYN_LEN, NUM_DYNAMIC), served.shape
        assert (served.dtypes == dataset.fp).all()
        assert served.index.equals(raw.index), stn
        for feature, column in RAW_DYN.items():
            assert np.array_equal(served[feature].to_numpy(),
                                  raw[column].to_numpy(dataset.fp),
                                  equal_nan=True), f"{stn}:{feature} differs from the raw file"
            # the float32 cast is harmless: relative error below 1e-7
            ok = (raw[column] != 0) & raw[column].notna()
            err = np.abs(served[feature][ok].to_numpy(np.float64) / raw[column][ok] - 1)
            assert (err < 1e-7).all(), f"{stn}:{feature} precision loss {err.max()}"

    # processes=1 must not start a pool and must give the same values
    serial = copy.copy(dataset)
    serial.processes = 1
    real_pool = _camels.cf.ProcessPoolExecutor
    _camels.cf.ProcessPoolExecutor = lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("processes=1 must not start a process pool"))
    try:
        dyn_serial = serial._read_dynamic(stations, 'all')
    finally:
        _camels.cf.ProcessPoolExecutor = real_pool
    for stn in stations:
        assert dyn[stn].equals(dyn_serial[stn]), stn
    return


def test_units():
    """the values are in the units the standardized names promise"""
    logger.info("test_units")
    stations = random.sample(dataset.stations(), 20)
    dyn = dataset._read_dynamic(stations, 'all')
    served = pd.concat(dyn.values())
    area = dataset.area(stations)

    # the two discharges are the same flow in m3/s and in mm/day over the
    # catchment area, which fixes both units. A wrong unit would be off by a
    # factor of 86400 or 1000; the 10 % tolerance is for the published area,
    # which at gauges 2282 and 2309 disagrees with the ratio by 6 to 7 %
    errors = {}
    for stn in stations:
        both = dyn[stn][['q_cms_obs', 'q_mm_obs']].dropna()
        both = both[both['q_mm_obs'].abs() > 0.01]
        if both.empty:
            continue
        mm_per_day = both['q_cms_obs'] * 86400.0 / (float(area[stn]) * 1e6) * 1000.0
        errors[stn] = np.abs(mm_per_day / both['q_mm_obs'] - 1).median()
    assert errors, "no station had both discharges"
    worst = max(errors, key=errors.get)
    assert errors[worst] < 0.1, f"{worst}: q_mm_obs is not q_cms_obs over the area"
    assert np.median(list(errors.values())) < 1e-2, errors
    assert served['q_cms_obs'].max() < 6000, served['q_cms_obs'].max()

    # water level is metres above sea level, not a stage of a few metres
    medians = pd.Series({stn: dyn[stn]['waterlevel(m)'].median() for stn in stations
                         if dyn[stn]['waterlevel(m)'].notna().any()})
    if not medians.empty:
        assert (medians > 150).all(), medians.min()
    # precipitation is a daily total in mm, never negative
    assert served['pcp_mm'].min() >= 0 and served['pcp_mm'].max() < 500
    # degree Celsius, not Kelvin, and ordered min <= mean <= max
    assert -50 < served['airtemp_C_min'].min() and served['airtemp_C_max'].max() < 50
    ok = served[['airtemp_C_min', 'airtemp_C_mean', 'airtemp_C_max']].notna().all(axis=1)
    ordered = served[ok]
    assert (ordered['airtemp_C_min'] <= ordered['airtemp_C_mean']).all()
    assert (ordered['airtemp_C_mean'] <= ordered['airtemp_C_max']).all()
    # relative sunshine duration and snow water equivalent
    assert served['rel_sun_dur(%)'].min() >= 0 and served['rel_sun_dur(%)'].max() <= 100
    assert served['swe_mm'].min() >= 0
    return


def test_static_fidelity():
    """every static file contributes its columns unchanged, only the four
    columns of ``static_map`` are renamed"""
    logger.info("test_static_fidelity")
    static = dataset.fetch_static_features()
    assert static.shape == (NUM_STATIONS, NUM_STATIC), static.shape
    assert list(static.index) == dataset.stations()

    renamed = {'area': 'area_km2', 'slope_mean': 'slope_degrees',
               'gauge_lat': 'lat', 'gauge_lon': 'long'}
    served = 0
    for fname, (subdir, n_columns) in RAW_STATIC.items():
        raw = pd.read_csv(os.path.join(dataset.static_path, subdir, fname), sep=',',
                          skiprows=1, index_col='gauge_id', encoding='latin-1',
                          float_precision='round_trip')
        assert raw.shape == (NUM_STATIONS, n_columns), (fname, raw.shape)
        raw.index = raw.index.astype(str)
        served += n_columns
        for column in raw.columns:
            values = static[renamed.get(column, column)]
            if raw[column].dtype == object:
                assert values.astype(str).tolist() == raw[column].astype(str).tolist(), (fname, column)
            else:
                assert np.allclose(values.to_numpy(float), raw[column].to_numpy(float),
                                   rtol=1e-6, equal_nan=True), (fname, column)
    assert served == NUM_STATIC

    # the typo which version 0.7 corrected
    assert static.loc['2247', 'water_body_name'] == 'Doubs'
    return


def test_returns_copies():
    """the caller cannot corrupt the cached state of the instance"""
    logger.info("test_returns_copies")
    stations = dataset.stations()
    stations.append('not a station')
    assert len(dataset.stations()) == NUM_STATIONS

    features = dataset.dynamic_features
    features.clear()
    assert len(dataset.dynamic_features) == NUM_DYNAMIC

    static = dataset.fetch_static_features('2004')
    static.iloc[0, 0] = -999.0
    assert dataset.fetch_static_features('2004').iloc[0, 0] != -999.0
    return


def test_coords_and_area():
    """coordinates are WGS84 and derived from the precise easting/northing, and
    areas are the km2 of the attribute file"""
    logger.info("test_coords_and_area")
    coords = dataset.stn_coords()
    assert coords.shape == (NUM_STATIONS, 2)
    assert list(coords.columns) == ['lat', 'long']
    assert coords['lat'].between(45.5, 48.2).all()
    assert coords['long'].between(5.5, 11.0).all()
    assert np.allclose(coords.loc['2004'].to_numpy(), [46.930752, 7.116924], atol=1e-5)

    # they are finer than the dataset's own two-decimal gauge_lat/gauge_lon
    published = dataset.fetch_static_features(static_features=['lat', 'long']).astype(float)
    assert (published.round(2) == published).all().all(), "gauge_lat/lon are no longer rounded"
    assert not np.allclose(coords['lat'].astype(float), published['lat'])

    # the conversion is right and not an arbitrary shift: it agrees with the
    # published gauge_lat/gauge_lon up to their 0.005 degree rounding
    from aqua_fetch._geom_utils import epsg2056_point_to_wgs84
    topo = dataset.topo_attrs()
    lat, long = epsg2056_point_to_wgs84(topo['gauge_easting'].to_numpy(float),
                                        topo['gauge_northing'].to_numpy(float))
    assert np.abs(lat - topo['gauge_lat'].to_numpy(float)).max() < 6e-3
    assert np.abs(long - topo['gauge_lon'].to_numpy(float)).max() < 6e-3

    raw = pd.read_csv(os.path.join(dataset.static_path, 'CAMELS_CH_topographic_attributes.csv'),
                      sep=',', skiprows=1, index_col='gauge_id', encoding='latin-1')
    area = dataset.area()
    assert np.allclose(area.to_numpy(float), raw['area'].to_numpy(float))
    return


def test_boundaries():
    """one boundary per catchment, converted from EPSG:2056 to WGS84"""
    logger.info("test_boundaries")
    if fiona is None:
        return
    assert os.path.exists(dataset.boundary_file)
    # 2004 is a simple Polygon, 2022 a Polygon with an interior ring (a hole)
    # and 3028 a MultiPolygon: the geometry type and rings must be preserved
    for stn, gtype in (('2004', 'Polygon'), ('2022', 'Polygon'), ('3028', 'MultiPolygon')):
        assert dataset.get_boundary(stn).type == gtype, stn

    for stn in random.sample(dataset.stations(), 5) + ['2004', '2022', '3028']:
        boundary = dataset.get_boundary(stn)
        rings = ([ring for polygon in boundary.coordinates for ring in polygon]
                 if boundary.type == 'MultiPolygon' else list(boundary.coordinates))
        xy = np.concatenate([np.asarray(ring, dtype=float)[:, :2] for ring in rings])
        assert xy[:, 0].min() > 5.5 and xy[:, 0].max() < 11.0, stn   # longitude
        assert xy[:, 1].min() > 45.5 and xy[:, 1].max() < 48.2, stn  # latitude
    return


def test_float64():
    """float_precision is honoured and gets its own cache name"""
    logger.info("test_float64")
    exact = CAMELS_CH(path=CAMELS_CH_PATH, verbosity=0, to_netcdf=False,
                      float_precision=np.float64)
    assert exact.dyn_fname == 'camels_ch_D_0.9_float64_v2.nc', exact.dyn_fname
    served = exact._read_stn_dyn('2004')
    assert (served.dtypes == np.float64).all()
    raw = _raw_daily('2004')
    for feature, column in RAW_DYN.items():
        assert np.array_equal(served[feature].to_numpy(), raw[column].to_numpy(float),
                              equal_nan=True), feature
    return


def test_to_netcdf_is_honoured():
    """``to_netcdf=False`` does not write a cache, ``to_netcdf=True`` does, and
    the cache serves exactly what the csv files do"""
    logger.info("test_to_netcdf_is_honoured")
    no_cache = CAMELS_CH(path=CAMELS_CH_PATH, verbosity=0, to_netcdf=False,
                         float_precision=np.float64)
    assert no_cache.to_netcdf is False
    assert not os.path.exists(no_cache.dyn_fpath), "a cache was written although to_netcdf=False"

    if netCDF4 is None:
        return
    assert dataset.to_netcdf is True
    assert os.path.exists(dataset.dyn_fpath)
    from_cache = dataset.fetch(stations=['2004', '2009'], as_dataframe=True)[1]
    for stn in ('2004', '2009'):
        assert from_cache[stn].equals(dataset._read_stn_dyn(stn)[dataset.dynamic_features]), stn
    return


def test_overwrite_is_forwarded():
    """``overwrite`` reaches the parent class, so that a stale netCDF cache is
    rebuilt instead of being served"""
    logger.info("test_overwrite_is_forwarded")
    stubs = ('_download_camels_ch', '_check_manifest', '_check_duplicates', '_maybe_to_netcdf')
    originals = {name: getattr(CAMELS_CH, name) for name in stubs}
    tmp = tempfile.mkdtemp()
    try:
        for name in stubs:
            setattr(CAMELS_CH, name, lambda self, *args, **kwargs: None)
        assert CAMELS_CH(path=tmp, verbosity=0, overwrite=True).overwrite is True
        assert CAMELS_CH(path=tmp, verbosity=0).overwrite is False
    finally:
        for name, original in originals.items():
            setattr(CAMELS_CH, name, original)
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_no_redownload_or_reextract():
    """initializing again neither downloads, extracts nor rebuilds the cache"""
    logger.info("test_no_redownload_or_reextract")

    def boom(*args, **kwargs):
        raise AssertionError("must not be called when the data exists")

    # ZipFile itself is left alone: pandas uses it to read the .xlsx inventory
    originals = (_camels.download_from_zenodo, zipfile.ZipFile.extractall,
                 xr.Dataset.to_netcdf if xr is not None else None)
    _camels.download_from_zenodo = zipfile.ZipFile.extractall = boom
    if xr is not None:
        xr.Dataset.to_netcdf = boom
    try:
        again = CAMELS_CH(path=CAMELS_CH_PATH, verbosity=0)
        assert len(again.stations()) == NUM_STATIONS
        hourly = CAMELS_CH(path=CAMELS_CH_PATH, timestep='H', verbosity=0)
        assert len(hourly.stations()) == NUM_HOURLY_STATIONS
    finally:
        _camels.download_from_zenodo, zipfile.ZipFile.extractall = originals[:2]
        if xr is not None:
            xr.Dataset.to_netcdf = originals[2]
    return


def _fake_record(tmp: str):
    """a stand-in for ``download_from_zenodo`` which records its calls and writes
    the requested files of a small CAMELS-CH like record"""
    calls = []

    def download_from_zenodo(outdir, doi, include, verbosity=1, **kwargs):
        calls.append((doi, tuple(include), outdir))
        for fname in include:
            fpath = os.path.join(outdir, fname)
            if fname.endswith('.zip'):
                with zipfile.ZipFile(fpath, 'w') as zf:
                    zf.writestr(f"{fname[:-len('.zip')]}/data.csv", 'gauge_id,x\n2004,1\n')
            else:
                open(fpath, 'wb').close()
    return calls, download_from_zenodo


def test_download():
    """only the files the class reads are downloaded, into a version named
    folder, and the archive is extracted once; afterwards nothing is downloaded
    or extracted again, and ``remove_zip`` deletes the archives"""
    logger.info("test_download")
    tmp = tempfile.mkdtemp()
    original = _camels.download_from_zenodo
    try:
        for timestep, n_calls in (('D', 1), ('H', 2)):
            path = os.path.join(tmp, timestep)
            ds = copy.copy(dataset)
            ds._path, ds.timestep, ds.remove_zip = path, timestep, False
            calls, _camels.download_from_zenodo = _fake_record(path)

            ds._download_camels_ch()
            assert len(calls) == n_calls, calls
            # the 546 MB Caravan extension and the 0.6 MB word document are never asked for
            assert calls[0] == ('https://zenodo.org/records/15025258', ('camels_ch.zip',),
                                os.path.join(path, 'camels_ch_v0.9')), calls[0]
            if timestep == 'H':
                assert calls[1] == ('https://zenodo.org/records/7691294',
                                    ('DischargeDBHydroCH.zip', 'Inventory_discharge_hydroCH.xlsx'),
                                    path), calls[1]
            assert os.listdir(os.path.join(path, 'camels_ch_v0.9', 'camels_ch')) == ['data.csv']
            assert not [f for f in os.listdir(os.path.join(path, 'camels_ch_v0.9'))
                        if f.endswith('_extracting')]

            calls.clear()
            ds.remove_zip = True
            ds._download_camels_ch()
            assert calls == [], f"downloaded again although the data is extracted: {calls}"
            assert not os.path.exists(os.path.join(path, 'camels_ch_v0.9', 'camels_ch.zip')), \
                "remove_zip left the archive"

            # and a third time, with the archive gone: still nothing to do
            ds._download_camels_ch()
            assert calls == [], f"remove_zip forced a re-download: {calls}"
    finally:
        _camels.download_from_zenodo = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_older_release_is_left_alone():
    """a version 0.6 folder next to the 0.9 one is neither read nor deleted"""
    logger.info("test_older_release_is_left_alone")
    tmp = tempfile.mkdtemp()
    original = _camels.download_from_zenodo
    try:
        old = os.path.join(tmp, 'camels_ch', 'camels_ch', 'time_series')
        os.makedirs(old)
        open(os.path.join(old, 'CAMELS_CH_obs_based_2004.csv'), 'wb').close()

        ds = copy.copy(dataset)
        ds._path, ds.remove_zip = tmp, False
        calls, _camels.download_from_zenodo = _fake_record(tmp)
        ds._download_camels_ch()

        assert os.path.exists(os.path.join(old, 'CAMELS_CH_obs_based_2004.csv')), \
            "the older release was deleted"
        assert [c[1] for c in calls] == [('camels_ch.zip',)], calls
        assert ds.camels_path == os.path.join(tmp, 'camels_ch_v0.9', 'camels_ch')
    finally:
        _camels.download_from_zenodo = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_interrupted_extraction():
    """an extraction that stops half way leaves no folder that looks complete,
    so the next initialization extracts the archive again without downloading it"""
    logger.info("test_interrupted_extraction")
    tmp = tempfile.mkdtemp()
    original_download, original_extractall = _camels.download_from_zenodo, zipfile.ZipFile.extractall

    def interrupted(self, path, *args, **kwargs):
        os.makedirs(path, exist_ok=True)
        open(os.path.join(path, 'half_written.csv'), 'wb').close()
        raise KeyboardInterrupt

    try:
        ds = copy.copy(dataset)
        ds._path, ds.remove_zip = tmp, False
        calls, _camels.download_from_zenodo = _fake_record(tmp)

        zipfile.ZipFile.extractall = interrupted
        try:
            ds._download_camels_ch()
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("the extraction was not interrupted")
        finally:
            zipfile.ZipFile.extractall = original_extractall

        assert not os.path.exists(ds.camels_path)
        ds._download_camels_ch()
        assert len(calls) == 1, f"the archive was downloaded again: {calls}"
        assert os.listdir(ds.camels_path) == ['data.csv']
    finally:
        _camels.download_from_zenodo = original_download
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_corrupt_archive():
    """an archive which is not a zip file is deleted with a clear error and
    downloaded again on the next initialization"""
    logger.info("test_corrupt_archive")
    tmp = tempfile.mkdtemp()
    original = _camels.download_from_zenodo
    try:
        ds = copy.copy(dataset)
        ds._path, ds.remove_zip = tmp, False
        calls, _camels.download_from_zenodo = _fake_record(tmp)

        archive = os.path.join(tmp, 'camels_ch_v0.9', 'camels_ch.zip')
        os.makedirs(os.path.dirname(archive))
        with open(archive, 'wb') as f:
            f.write(b'<html>Service unavailable</html>')
        try:
            ds._download_camels_ch()
        except ValueError as e:
            assert 'is corrupt' in str(e), e
        else:
            raise AssertionError("a corrupt archive was accepted")
        assert not os.path.exists(archive), "the corrupt archive was kept"
        assert not os.path.exists(f"{ds.camels_path}_extracting"), "the partial extraction was kept"

        ds._download_camels_ch()
        assert [c[1] for c in calls] == [('camels_ch.zip',)], calls
        assert os.listdir(ds.camels_path) == ['data.csv']
    finally:
        _camels.download_from_zenodo = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_overwrite_removes_stale_before_download():
    """``overwrite=True`` deletes the extracted folder, the archive and the
    netCDF caches of this timestep before downloading, and leaves the files of
    the other timestep alone"""
    logger.info("test_overwrite_removes_stale_before_download")
    tmp = tempfile.mkdtemp()
    original = _camels.download_from_zenodo
    try:
        ds = copy.copy(dataset)
        ds._path, ds.remove_zip = tmp, False
        calls, _camels.download_from_zenodo = _fake_record(tmp)

        os.makedirs(os.path.join(tmp, 'camels_ch_v0.9', 'camels_ch'))
        stale_csv = os.path.join(tmp, 'camels_ch_v0.9', 'camels_ch', 'stale.csv')
        open(stale_csv, 'wb').close()
        for fname in ('camels_ch_D.nc', 'camels_ch_D_0.9_v2.nc', 'camels_ch_H_0.9_v2.nc'):
            open(os.path.join(tmp, fname), 'wb').close()

        ds._download_camels_ch(overwrite=True)

        assert not os.path.exists(stale_csv), "the stale extracted data was kept"
        assert os.listdir(ds.camels_path) == ['data.csv']
        assert [c[1] for c in calls] == [('camels_ch.zip',)], calls
        assert not os.path.exists(os.path.join(tmp, 'camels_ch_D.nc'))
        assert not os.path.exists(os.path.join(tmp, 'camels_ch_D_0.9_v2.nc'))
        assert os.path.exists(os.path.join(tmp, 'camels_ch_H_0.9_v2.nc')), \
            "the cache of the other timestep was deleted"
    finally:
        _camels.download_from_zenodo = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_missing_files_warn():
    """a file this class reads which is missing is reported, loudly, instead of
    being silently left out of the dataset"""
    logger.info("test_missing_files_warn")
    tmp = tempfile.mkdtemp()
    try:
        ds = copy.copy(dataset)
        ds._path = tmp
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds._check_manifest()
        assert len(caught) == 1 and 'missing' in str(caught[0].message), caught
        assert '14 files are missing' in str(caught[0].message), caught[0].message

        # everything but one time series file
        version_dir = os.path.join(tmp, 'camels_ch_v0.9', 'camels_ch')
        os.makedirs(version_dir)
        for name in ('static_attributes', 'catchment_delineations'):
            os.symlink(os.path.join(dataset.camels_path, name), os.path.join(version_dir, name))
        ts_dir = os.path.join(version_dir, 'timeseries', 'observation_based')
        os.makedirs(ts_dir)
        real_ts = os.path.join(dataset.dynamic_path)
        for fname in os.listdir(real_ts):
            if fname != 'CAMELS_CH_obs_based_2004.csv':
                os.symlink(os.path.join(real_ts, fname), os.path.join(ts_dir, fname))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds._check_manifest()
        assert len(caught) == 1 and '1 of 345 files are missing' in str(caught[0].message), caught[0].message

        # and none when everything is there
        os.symlink(os.path.join(real_ts, 'CAMELS_CH_obs_based_2004.csv'),
                   os.path.join(ts_dir, 'CAMELS_CH_obs_based_2004.csv'))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds._check_manifest()
        assert caught == [], caught
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


# ---------------------------------------------------------------------------
# hourly
# ---------------------------------------------------------------------------

def test_hourly_is_hourly():
    """``timestep='H'`` serves the hourly discharge of the 170 catchments which
    have it, not the daily data"""
    logger.info("test_hourly_is_hourly")
    hourly = _hourly_dataset()
    assert hourly.timestep == 'H'
    assert hourly.dynamic_features == ['q_cms_obs'], hourly.dynamic_features
    assert len(hourly.stations()) == NUM_HOURLY_STATIONS
    assert set(hourly.stations()) <= set(dataset.stations())
    assert hourly.dyn_fname == 'camels_ch_H_0.9_v2.nc', hourly.dyn_fname
    # a cache of 170 ragged records over 98 years would be 584 MB of mostly NaN
    assert hourly.to_netcdf is False

    served = hourly._read_stn_dyn('2009')
    assert served.shape == (HOURLY_LEN_2009, 1), served.shape
    assert pd.infer_freq(served.index) in ('H', 'h')
    assert served.index[0] == pd.Timestamp('1974-01-01') and \
           served.index[-1] == pd.Timestamp('2019-10-01')
    assert (served.dtypes == hourly.fp).all()

    # the extent spans every station's record
    assert hourly.start == pd.Timestamp('1923-02-15'), hourly.start
    assert hourly.end == pd.Timestamp('2021-02-08 23:00'), hourly.end
    for stn in random.sample(hourly.stations(), 5):
        index = hourly._read_stn_dyn(stn).index
        assert hourly.start <= index[0] and index[-1] <= hourly.end, stn
        assert index.equals(pd.date_range(index[0], index[-1], freq='h')), f"{stn} has gaps"
    return


def test_hourly_fidelity():
    """the served hourly values are the values of the file the inventory names,
    with only the -9999 no-data marker turned into NaN"""
    logger.info("test_hourly_fidelity")
    hourly = _hourly_dataset()
    stations = random.sample(hourly.stations(), 5) + ['2199', '2446']
    inventory = hourly._inventory

    for stn in stations:
        raw = _raw_hourly(inventory.loc[stn, 'Filename'])
        served = hourly._read_stn_dyn(stn)['q_cms_obs']
        assert served.index.equals(pd.DatetimeIndex(raw.index)), stn

        expected = raw.to_numpy(float)
        marker = expected == -9999.0
        expected[marker] = np.nan
        assert np.array_equal(served.to_numpy(), expected.astype(hourly.fp),
                              equal_nan=True), f"{stn} differs from the raw file"
        assert served.isna().sum() == marker.sum(), stn
        assert not (served == -9999.0).any(), f"{stn} serves the no-data marker as a discharge"

    # gauges 2446 and 2447 are channels with bidirectional flow: their negative
    # values are real and must survive
    negative = hourly._read_stn_dyn('2446')['q_cms_obs']
    assert (negative < 0).sum() == 41173, (negative < 0).sum()
    assert negative.min() > -500
    return


def test_hourly_station_2160():
    """gauge 2160 is the Sarine at Broc: two files in the FOEN folder carry that
    id and the inventory says which one belongs to the gauge"""
    logger.info("test_hourly_station_2160")
    hourly = _hourly_dataset()
    foen = [f for f in hourly.foen_stations() if '_Q_2160_hourly.asc' in f]
    assert sorted(foen) == ['SarBrf_Q_2160_hourly.asc', 'SarBro_Q_2160_hourly.asc'], foen

    assert os.path.basename(hourly._hourly_paths['2160']) == 'SarBro_Q_2160_hourly.asc'
    served = hourly.read_hourly_q_ch('2160')['q_cms_obs']
    right, wrong = (_raw_hourly(f'SarBr{suffix}_Q_2160_hourly.asc') for suffix in ('o', 'f'))
    assert np.array_equal(served.to_numpy(), right.to_numpy(hourly.fp), equal_nan=True)
    assert not np.array_equal(served.to_numpy(), wrong.to_numpy(hourly.fp), equal_nan=True)
    # and the gauge is the one CAMELS-CH describes
    assert dataset.fetch_static_features('2160', 'id6').iloc[0, 0] == 'SarBro'
    return


def test_hourly_stations_lists():
    """the inventory holds more gauges than CAMELS-CH does"""
    logger.info("test_hourly_stations_lists")
    hourly = _hourly_dataset()
    everything = hourly.all_hourly_stations()
    assert len(everything) == 291
    assert len(hourly.hourly_stations()) == NUM_HOURLY_STATIONS
    assert set(hourly.hourly_stations()) == set(everything) & set(dataset.stations())
    everything.clear()
    assert len(hourly.all_hourly_stations()) == 291
    return


def test_hourly_fetch():
    """fetching returns one frame per station, each on its own record"""
    logger.info("test_hourly_fetch")
    hourly = _hourly_dataset()
    stations = random.sample(hourly.stations(), 3)
    _, dyn = hourly.fetch(stations=stations, as_dataframe=True)
    assert sorted(dyn) == sorted(stations)
    for stn in stations:
        assert list(dyn[stn].columns) == ['q_cms_obs']
        assert dyn[stn].index.name == 'time'
        assert dyn[stn].equals(hourly._read_stn_dyn(stn))

    static, _ = hourly.fetch(stations=stations, dynamic_features=None, static_features='all')
    assert static.shape == (3, NUM_STATIC)
    return


# ---------------------------------------------------------------------------
# efficiency and the shared suite
# ---------------------------------------------------------------------------

def test_efficiency():
    """a documented one liner must not take minutes"""
    logger.info("test_efficiency")
    start = time.time()
    CAMELS_CH(path=CAMELS_CH_PATH, verbosity=0)
    init = time.time() - start
    assert init < 10, f"initialization took {init:.1f} s"

    start = time.time()
    _, dyn = dataset.fetch(stations='all', as_dataframe=True)
    fetch = time.time() - start
    assert len(dyn) == NUM_STATIONS
    assert fetch < 60, f"fetching all stations took {fetch:.1f} s"

    start = time.time()
    coords = dataset.stn_coords()
    assert len(coords) == NUM_STATIONS
    assert time.time() - start < 5

    logger.info(f"init {init:.2f} s, fetch all {fetch:.2f} s")
    return


def test_shared_suite():
    """the checks every rainfall-runoff dataset has to pass"""
    logger.info("test_shared_suite")
    run_shared_tests(dataset, NUM_STATIONS, DYN_LEN, NUM_STATIC, NUM_DYNAMIC)
    return


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith('test_') and callable(test):
            print(f"running {name}")
            test()
    print("all tests passed")
