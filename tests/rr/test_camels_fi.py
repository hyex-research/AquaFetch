"""
Tests for CAMELS_FI (Finland, 320 catchments), release 1.2.0.

They check that the class

    * serves the raw values, dates and units unchanged, with the two documented
      conversions (snow depth cm -> m, global radiation kJ m-2 day-1 -> W m-2),
    * reads the static attributes, coordinates, areas and boundaries faithfully
      and serves release 1.2.0, not the withdrawn release 1.0.1,
    * downloads once, neither downloads nor extracts again, survives an
      interrupted extraction and, on ``overwrite=True``, deletes the stale files
      first,
    * replaces a release 1.0.1 left on disk by an earlier version of this class,
      including the netCDF caches built from it,
    * warns instead of silently serving a truncated dataset,
    * fetches all stations quickly.

The first run downloads release 1.2.0 (382 MB) into ``CAMELS_FI_PATH``.
Run as a script or with pytest.
"""

import os
import copy
import site
import time
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
    logging.basicConfig(filename='test_camels_fi.log', filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

import aqua_fetch
from aqua_fetch import download_zenodo   # so that aqua_fetch.download_zenodo can be patched
from aqua_fetch import CAMELS_FI
from aqua_fetch._backend import xarray as xr, fiona
from aqua_fetch._geom_utils import tmerc_to_wgs84, _make_boundary_2d
from aqua_fetch.rr._map import KJ_M2_DAY_TO_WM2

from utils import test_dataset as run_shared_tests

# The class appends ``CAMELS_FI`` to this path. Replace with the location on
# your machine, or set the CAMELS_FI_PATH environment variable.
CAMELS_FI_PATH = os.environ.get(
    'CAMELS_FI_PATH',
    '/mnt/hyexnas01/abbaa0a/data/gscad_database/raw/CAMELS/CAMELS')

NUM_STATIONS = 320
NUM_STATIC = 112
NUM_DYNAMIC = 16
DYN_LEN = 23010          # daily steps 1961-01-01 .. 2023-12-31
START, END = '1961-01-01', '2023-12-31'

# EPSG:3067 (ETRS89 / TM35FIN), the projection of the boundaries and of the
# gauge eastings/northings
TM35FIN = dict(lon_0=27.0, k0=0.9996, false_easting=500000.0, false_northing=0.0)

# standardized name -> (raw column, factor). Written out here rather than taken
# from the class, so that a wrong mapping or a forgotten factor is caught.
RAW_COLUMNS = {
    'q_cms_obs': ('discharge_vol', 1.0),
    'q_mm_obs': ('discharge_spec', 1.0),
    'pcp_mm': ('precipitation', 1.0),
    'pet_mm': ('pet', 1.0),
    'pet_fmi': ('pet_fmi', 1.0),
    'pet_singer': ('pet_singer', 1.0),
    'snow_evaporation': ('snow_evaporation', 1.0),
    'swe_mm_era5': ('swe', 1.0),
    'swe_mm_cci3-1': ('swe_cci3-1', 1.0),
    'snowdepth_m': ('snow_depth', 0.01),          # published in cm
    'temperature_gmin': ('temperature_gmin', 1.0),
    'airtemp_C_min': ('temperature_min', 1.0),
    'airtemp_C_mean': ('temperature_mean', 1.0),
    'airtemp_C_max': ('temperature_max', 1.0),
    'rh_%': ('humidity_rel', 1.0),
    'swdownrad_wm2': ('radiation_global', KJ_M2_DAY_TO_WM2),  # published in kJ m-2 day-1
}

ATTRIBUTE_FILES = ('CAMELS_FI_meta_attributes.csv',
                   'CAMELS_FI_topographic_attributes.csv',
                   'CAMELS_FI_climatic_attributes.csv',
                   'CAMELS_FI_hydrologic_attributes.csv',
                   'CAMELS_FI_landcover_attributes.csv',
                   'CAMELS_FI_soil_attributes.csv',
                   'CAMELS_FI_geology_attributes.csv',
                   'CAMELS_FI_humaninfluence_attributes.csv')

# the download of release 1.2.0 happens here, once
dataset = CAMELS_FI(path=CAMELS_FI_PATH, verbosity=0)


def raw_timeseries(stn: str) -> pd.DataFrame:
    """the station's file read independently of the class"""
    fpath = os.path.join(dataset.ts_path,
                         f"CAMELS_FI_hydromet_timeseries_{stn}_19610101-20231231.csv")
    return pd.read_csv(fpath, index_col=0, parse_dates=True)


def raw_attributes(fname: str) -> pd.DataFrame:
    """one attribute file read independently of the class"""
    return pd.read_csv(os.path.join(dataset.data_path, fname),
                       index_col=0, dtype={0: str})


def _copy_for(path: str) -> CAMELS_FI:
    """a copy of the dataset which points at ``path`` instead of the downloaded
    data, so that the download code can be exercised without a network"""
    ds = copy.copy(dataset)
    ds._path = path
    return ds


def _fake_download(calls, marker: bool = True):
    """a stand-in for download_from_zenodo which records its calls and writes a
    small CAMELS-FI.zip holding the folder layout of the real archive. With
    ``marker=False`` it writes the layout of release 1.0.1 instead, which has no
    geology attributes."""
    def download_from_zenodo(outdir, doi, include=None, verbosity=1, **kwargs):
        calls.append((outdir, doi, tuple(include or ())))
        files = {'CAMELS-FI/data/CAMELS_FI_meta_attributes.csv': 'gauge_id,gauge_lat\n1156,62.4\n'}
        if marker:
            files['CAMELS-FI/data/CAMELS_FI_geology_attributes.csv'] = 'gauge_id,volcanic_perc\n1156,0.0\n'
        with zipfile.ZipFile(os.path.join(outdir, 'CAMELS-FI.zip'), 'w') as zf:
            for name, content in files.items():
                zf.writestr(name, content)
    return download_from_zenodo


def _old_release(path: str) -> dict:
    """lays down a release 1.0.1 as an earlier version of this class left it:
    the archive extracted into ``CAMELS-FI/CAMELS-FI/`` and the caches built
    from it. Returns the paths it created."""
    data = os.path.join(path, 'CAMELS-FI', 'CAMELS-FI', 'data')
    os.makedirs(os.path.join(data, 'timeseries'))
    created = {'data': data,
               'root': os.path.join(path, 'CAMELS-FI'),
               'archive': os.path.join(path, 'CAMELS-FI.zip'),
               'cache_v1': os.path.join(path, 'camels_fi_D.nc'),
               'cache_v2': os.path.join(path, 'camels_fi_D_v2.nc')}
    open(os.path.join(data, 'CAMELS_FI_meta_attributes.csv'), 'w').close()
    for key in ('archive', 'cache_v1', 'cache_v2'):
        open(created[key], 'w').close()
    return created


# ------------------------------------------------------------------ contents

def test_stations_and_features():
    """the release's own 320 gauges, 16 dynamic and 112 static features"""
    logger.info("test_stations_and_features")
    stations = dataset.stations()
    assert len(stations) == NUM_STATIONS, len(stations)
    assert len(set(stations)) == NUM_STATIONS, "a station is listed twice"

    # the gauges of the metadata file, not whatever csv files are on disk
    expected = set(raw_attributes('CAMELS_FI_meta_attributes.csv').index)
    assert set(stations) == expected, expected.symmetric_difference(stations)

    assert len(dataset.dynamic_features) == NUM_DYNAMIC, dataset.dynamic_features
    assert set(dataset.dynamic_features) == set(RAW_COLUMNS), \
        set(dataset.dynamic_features).symmetric_difference(RAW_COLUMNS)
    assert len(dataset.static_features) == NUM_STATIC, len(dataset.static_features)

    # the features that release 1.0.1 has and 1.2.0 does not, and the other way
    assert 'pe_era5_land' not in dataset.dynamic_features
    assert 'pet_singer' in dataset.dynamic_features
    assert 'baseflow_index_lfstat' not in dataset.static_features
    assert 'baseflow_index' in dataset.static_features
    assert 'temperature_mean_annual' in dataset.static_features
    assert 'volcanic_perc' in dataset.static_features, "no geology attributes"
    return


def test_dynamic_values_and_units():
    """every served value is the published one, times the documented factor, on
    the published date"""
    logger.info("test_dynamic_values_and_units")
    stations = sorted(dataset.stations())
    for stn in (stations[0], stations[len(stations) // 2], stations[-1], '1156'):
        raw = raw_timeseries(stn)
        _, dyn = dataset.fetch(stations=stn, as_dataframe=True)
        served = dyn[stn]

        assert served.shape == (DYN_LEN, NUM_DYNAMIC), served.shape
        # the full daily index, not one built from the first and last dates
        assert served.index.equals(pd.date_range(START, END, freq='D')), stn
        assert served.index.equals(raw.index), stn

        for name, (column, factor) in RAW_COLUMNS.items():
            expected = raw[column].values * factor
            np.testing.assert_allclose(
                served[name].values.astype('float64'), expected,
                rtol=1e-6, atol=0, equal_nan=True,
                err_msg=f"{stn}: {name} is not {column} x {factor}")
    return


def test_snow_depth_is_meters():
    """snow depth is published in cm and the canonical name promises m, so the
    values must be a hundredth of the published ones"""
    logger.info("test_snow_depth_is_meters")
    raw = raw_timeseries('1156')
    _, dyn = dataset.fetch(stations='1156', as_dataframe=True)
    served = dyn['1156']['snowdepth_m']

    np.testing.assert_allclose(served.values.astype('float64'),
                               raw['snow_depth'].values / 100.0,
                               rtol=1e-6, equal_nan=True)
    # 1.84 m is the deepest snow in the whole release; cm values would be ~184
    assert 0.0 <= served.max() <= 3.0, served.max()
    return


def test_static_values():
    """static attributes are the published ones, with the land-cover fractions
    of the static_map converted from percent and the rest left as published"""
    logger.info("test_static_values")
    static = dataset.fetch_static_features('all', static_features='all')
    assert static.shape == (NUM_STATIONS, NUM_STATIC), static.shape

    meta = raw_attributes('CAMELS_FI_meta_attributes.csv')
    landcover = raw_attributes('CAMELS_FI_landcover_attributes.csv')
    geology = raw_attributes('CAMELS_FI_geology_attributes.csv')
    static = static.loc[meta.index]

    np.testing.assert_allclose(static['area_km2'].astype('float64'), meta['area'])
    np.testing.assert_allclose(static['lat'].astype('float64'), meta['gauge_lat'])
    np.testing.assert_allclose(static['long'].astype('float64'), meta['gauge_lon'])
    assert list(static['gauge_name']) == list(meta['gauge_name'])

    # converted from percent, and their sister attributes are not
    np.testing.assert_allclose(static['crop_frac_2018'].astype('float64'),
                               landcover['crop_perc_2018'] / 100.0)
    np.testing.assert_allclose(static['wetland_perc_2018'].astype('float64'),
                               landcover['wetland_perc_2018'])
    np.testing.assert_allclose(static['volcanic_perc'].astype('float64'),
                               geology['volcanic_perc'])
    return


def test_coordinates_are_release_1_2_0():
    """the served coordinates agree with the eastings and northings the release
    publishes for the same gauges. Release 1.0.1 disagrees with its own
    projection by up to 0.4 degrees, so this fails if it is served."""
    logger.info("test_coordinates_are_release_1_2_0")
    meta = raw_attributes('CAMELS_FI_meta_attributes.csv')
    coords = dataset.stn_coords().loc[meta.index]

    reprojected = [tmerc_to_wgs84(easting, northing, **TM35FIN)
                   for easting, northing in zip(meta['gauge_easting'], meta['gauge_northing'])]
    lat = np.array([point[0] for point in reprojected])
    lon = np.array([point[1] for point in reprojected])

    # tmerc_to_wgs84 itself is good to ~5 cm, so 1e-4 degrees (~10 m) is loose
    np.testing.assert_allclose(coords['lat'].astype('float64'), lat, atol=1e-4)
    np.testing.assert_allclose(coords['long'].astype('float64'), lon, atol=1e-4)
    return


def test_area_and_boundary():
    """areas come from the attributes and the boundaries are readable"""
    logger.info("test_area_and_boundary")
    meta = raw_attributes('CAMELS_FI_meta_attributes.csv')
    area = dataset.area('all')
    np.testing.assert_allclose(area.loc[meta.index].astype('float64'), meta['area'], rtol=1e-6)

    if fiona is not None:
        geometry = dataset.get_boundary('1156')
        assert geometry.type in ('Polygon', 'MultiPolygon'), geometry.type
    return


def test_boundary_is_wgs84():
    """get_boundary reprojects the EPSG:3067 shapefile to WGS84 without losing
    or moving a vertex, and to_wgs84=False still returns the published meters"""
    logger.info("test_boundary_is_wgs84")
    if fiona is None:
        logger.info("fiona is not installed, skipped")
        return

    coords = dataset.stn_coords()

    # a small, a medium, the largest (1388, 50672 km2) and the smallest
    # (2643, 6.9 km2) catchment
    for stn in ['1156', '896', '1388', '2643']:
        degrees = np.concatenate(_make_boundary_2d(dataset.get_boundary(stn)))
        meters = np.concatenate(_make_boundary_2d(dataset.get_boundary(stn, to_wgs84=False)))

        # to_wgs84=False leaves the published eastings/northings untouched
        assert meters[:, 0].min() > 1e4 and meters[:, 1].min() > 1e6, stn

        # no vertex is dropped or duplicated by the transformation
        assert degrees.shape == meters.shape, (stn, degrees.shape, meters.shape)

        # Finland, and nothing but Finland
        assert 19.0 < degrees[:, 0].min() and degrees[:, 0].max() < 32.0, stn
        assert 59.5 < degrees[:, 1].min() and degrees[:, 1].max() < 70.5, stn

        # each vertex is where an independent reprojection of the published
        # meters puts it
        lat, long_ = tmerc_to_wgs84(meters[:, 0], meters[:, 1], **TM35FIN)
        np.testing.assert_allclose(degrees[:, 0], long_, atol=1e-9, err_msg=stn)
        np.testing.assert_allclose(degrees[:, 1], lat, atol=1e-9, err_msg=stn)

        # cross-check against the other source of coordinates in the release:
        # the gauge from the metadata csv lies inside the shapefile's outline
        long_g, lat_g = coords.loc[stn, 'long'], coords.loc[stn, 'lat']
        assert degrees[:, 0].min() <= long_g <= degrees[:, 0].max(), stn
        assert degrees[:, 1].min() <= lat_g <= degrees[:, 1].max(), stn
    return


def test_cache_matches_csv():
    """the netCDF cache serves exactly what the csv files hold"""
    logger.info("test_cache_matches_csv")
    if xr is None or not dataset.dyn_fpath_exists:
        logger.info("no netCDF cache, skipped")
        return

    assert dataset.release in os.path.basename(dataset.dyn_fpath), dataset.dyn_fpath

    stations = sorted(dataset.stations())[:5]
    _, from_cache = dataset.fetch(stations=stations, as_dataframe=True)
    for stn in stations:
        raw = raw_timeseries(stn)
        for name, (column, factor) in RAW_COLUMNS.items():
            np.testing.assert_allclose(
                from_cache[stn][name].values.astype('float64'),
                raw[column].values * factor,
                rtol=1e-6, equal_nan=True, err_msg=f"{stn}: {name} in the cache")
    return


def test_fetch_all_stations():
    """all 320 stations are fetched in a few seconds"""
    logger.info("test_fetch_all_stations")
    start = time.time()
    static, dyn = dataset.fetch(stations='all', static_features='all', as_dataframe=True)
    taken = time.time() - start

    assert static.shape == (NUM_STATIONS, NUM_STATIC), static.shape
    assert len(dyn) == NUM_STATIONS, len(dyn)
    assert {df.shape for df in dyn.values()} == {(DYN_LEN, NUM_DYNAMIC)}
    logger.info(f"fetched all {NUM_STATIONS} stations in {taken:.1f} s")
    assert taken < 300, f"fetching all stations took {taken:.0f} s"
    return


def test_shared():
    """the checks every rainfall-runoff dataset has to pass"""
    logger.info("test_shared")
    # test_latlong_ranges=True checks that get_boundary reprojects the
    # EPSG:3067 boundaries to WGS84
    run_shared_tests(dataset, NUM_STATIONS, DYN_LEN, NUM_STATIC, NUM_DYNAMIC,
                     test_latlong_ranges=True)
    return


# ------------------------------------------------------------------ download

def test_no_redownload_or_reextract():
    """initializing again neither downloads, extracts nor rebuilds the cache"""
    logger.info("test_no_redownload_or_reextract")

    def boom(*args, **kwargs):
        raise AssertionError("must not be called when the data exists")

    originals = (aqua_fetch.download_zenodo.download_from_zenodo,
                 zipfile.ZipFile.extractall,
                 xr.Dataset.to_netcdf if xr is not None else None)
    aqua_fetch.download_zenodo.download_from_zenodo = boom
    zipfile.ZipFile.extractall = boom
    if xr is not None:
        xr.Dataset.to_netcdf = boom
    try:
        again = CAMELS_FI(path=CAMELS_FI_PATH, verbosity=0)
        assert len(again.stations()) == NUM_STATIONS
    finally:
        aqua_fetch.download_zenodo.download_from_zenodo = originals[0]
        zipfile.ZipFile.extractall = originals[1]
        if xr is not None:
            xr.Dataset.to_netcdf = originals[2]
    return


def test_download_and_extract():
    """the archive is downloaded once and extracted into ``CAMELS-FI``; once
    extracted nothing is downloaded again and remove_zip deletes the archive"""
    logger.info("test_download_and_extract")
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'CAMELS_FI')       # does not exist yet
    calls = []
    original = aqua_fetch.download_zenodo.download_from_zenodo
    try:
        aqua_fetch.download_zenodo.download_from_zenodo = _fake_download(calls)
        ds = _copy_for(path)
        ds.remove_zip = False
        ds._download_camels_fi()

        assert len(calls) == 1, calls
        outdir, doi, include = calls[0]
        assert outdir == path and doi == CAMELS_FI.url
        assert include == ('CAMELS-FI.zip',), "the support document was downloaded too"
        # the nesting of the archive is not repeated: CAMELS-FI/data, not
        # CAMELS-FI/CAMELS-FI/data
        assert os.path.isdir(ds.data_path), os.listdir(path)
        assert sorted(os.listdir(path)) == ['CAMELS-FI', 'CAMELS-FI.zip'], os.listdir(path)

        calls.clear()
        ds.remove_zip = True
        ds._download_camels_fi()
        assert calls == [], f"downloaded again although the data is extracted: {calls}"
        assert os.listdir(path) == ['CAMELS-FI'], "remove_zip left the archive"

        # and with the archive gone it is still not downloaded again
        ds._download_camels_fi()
        assert calls == [], f"downloaded again after remove_zip: {calls}"
    finally:
        aqua_fetch.download_zenodo.download_from_zenodo = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_interrupted_extraction():
    """an extraction that stops half way leaves no folder which looks complete,
    so the next initialization extracts the archive again without downloading"""
    logger.info("test_interrupted_extraction")
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'CAMELS_FI')
    calls = []
    original_download = aqua_fetch.download_zenodo.download_from_zenodo
    original_extractall = zipfile.ZipFile.extractall
    try:
        aqua_fetch.download_zenodo.download_from_zenodo = _fake_download(calls)
        ds = _copy_for(path)
        ds.remove_zip = False

        def die(self, *args, **kwargs):
            original_extractall(self, *args, **kwargs)
            raise KeyboardInterrupt("interrupted")

        zipfile.ZipFile.extractall = die
        try:
            ds._download_camels_fi()
        except KeyboardInterrupt:
            pass
        zipfile.ZipFile.extractall = original_extractall

        assert not os.path.exists(ds._root), "a half extracted folder was left behind"
        assert os.path.exists(os.path.join(path, 'CAMELS-FI.zip'))

        # the archive is on disk, so it is extracted rather than downloaded again
        ds._download_camels_fi()
        assert len(calls) == 1, f"the archive was downloaded again: {calls}"
        assert os.path.isdir(ds.data_path)
    finally:
        aqua_fetch.download_zenodo.download_from_zenodo = original_download
        zipfile.ZipFile.extractall = original_extractall
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_old_release_is_replaced():
    """a release 1.0.1 left by an earlier version of this class, and the caches
    built from it, are removed and release 1.2.0 downloaded in their place"""
    logger.info("test_old_release_is_replaced")
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'CAMELS_FI')
    calls = []
    original = aqua_fetch.download_zenodo.download_from_zenodo
    try:
        aqua_fetch.download_zenodo.download_from_zenodo = _fake_download(calls)
        old = _old_release(path)
        ds = _copy_for(path)
        ds.remove_zip = False

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds._download_camels_fi()
        messages = [str(w.message) for w in caught]
        assert any('1.0.1' in m and CAMELS_FI.release in m for m in messages), messages

        assert len(calls) == 1, f"release 1.2.0 was not downloaded: {calls}"
        assert not os.path.exists(old['data']), "the release 1.0.1 files survived"
        assert not os.path.exists(old['cache_v1']), "camels_fi_D.nc survived"
        assert not os.path.exists(old['cache_v2']), "camels_fi_D_v2.nc survived"
        # nothing was written as CAMELS-FI.zip1 next to the old archive
        assert not [f for f in os.listdir(path) if f.endswith('.zip1')], os.listdir(path)
        assert os.path.isdir(ds.data_path)
    finally:
        aqua_fetch.download_zenodo.download_from_zenodo = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_old_release_cache_is_not_served():
    """the cache of this release carries the release in its name, so a cache
    built from release 1.0.1 can never be read as this release"""
    logger.info("test_old_release_cache_is_not_served")
    assert os.path.basename(dataset.dyn_fpath) != 'camels_fi_D_v2.nc'
    assert os.path.basename(dataset.dyn_fpath).startswith(f'camels_fi_D_{CAMELS_FI.release}')
    return


def test_overwrite_removes_stale_before_download():
    """overwrite=True deletes the archive, the extracted files and the netCDF
    caches before downloading again"""
    logger.info("test_overwrite_removes_stale_before_download")
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'CAMELS_FI')
    calls = []
    original = aqua_fetch.download_zenodo.download_from_zenodo
    try:
        aqua_fetch.download_zenodo.download_from_zenodo = _fake_download(calls)
        ds = _copy_for(path)
        ds.remove_zip = False
        ds._download_camels_fi()

        stale = os.path.join(ds.data_path, 'stale.csv')
        open(stale, 'w').close()
        open(ds.dyn_fpath, 'w').close()
        open(os.path.join(path, 'camels_fi_D_v2.nc'), 'w').close()

        calls.clear()
        ds._download_camels_fi(overwrite=True)

        assert len(calls) == 1, calls
        assert not os.path.exists(stale), "the stale extracted file survived"
        assert not os.path.exists(os.path.join(path, 'camels_fi_D_v2.nc')), \
            "the release 1.0.1 cache survived"
        assert not os.path.exists(ds.dyn_fpath), "the stale cache survived"
        assert not [f for f in os.listdir(path) if f.endswith('.zip1')], os.listdir(path)
    finally:
        aqua_fetch.download_zenodo.download_from_zenodo = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_missing_files_warn():
    """a file deleted by hand is reported instead of being served as a smaller
    dataset"""
    logger.info("test_missing_files_warn")
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, 'CAMELS_FI')
        data = os.path.join(path, 'CAMELS-FI', 'data')
        os.makedirs(os.path.join(data, 'timeseries'))
        shutil.copy(os.path.join(dataset.data_path, 'CAMELS_FI_meta_attributes.csv'), data)
        ds = _copy_for(path)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds._check_manifest()
        messages = [str(w.message) for w in caught]
        # the 7 other attribute files, 4 shapefile files and 320 time series
        expected = f'{7 + 4 + NUM_STATIONS} of {8 + 4 + NUM_STATIONS} files are missing'
        assert any(expected in m for m in messages), messages

        # and a complete release does not warn
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dataset._check_manifest()
        assert [str(w.message) for w in caught] == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


if __name__ == "__main__":
    for name, test in sorted(list(globals().items())):
        if name.startswith('test_') and callable(test):
            print(f"running {name}")
            start = time.time()
            test()
            print(f"   ok in {time.time() - start:.1f} s")
