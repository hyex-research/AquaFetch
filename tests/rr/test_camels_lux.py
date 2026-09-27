"""
Tests for CAMELS_LUX (Luxembourg, 56 partly nested gauges), release 2.1.

They check that the class

    * serves the raw values, dates and units unchanged, with the one documented
      conversion (specific humidity kg/kg -> g/kg),
    * reads the static attributes, coordinates, areas and boundaries faithfully
      and serves release 2.1, not the superseded release 1.1,
    * maps every catchment boundary to its own gauge, although release 2.1
      reordered the attributes of the shapefile,
    * derives its temporal extent from the data instead of a hardcoded one,
    * downloads once, neither downloads nor extracts again, survives an
      interrupted extraction and, on ``overwrite=True``, deletes the stale
      files first,
    * replaces a release 1.1 left on disk by an earlier version of this class,
      including the netCDF caches built from it,
    * warns instead of silently serving a truncated dataset,
    * still reads after free_disk_space("redundant") has removed the csv files,
    * fetches all stations quickly.

The first run downloads release 2.1 (881 MB) into ``CAMELS_LUX_PATH``.
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
    logging.basicConfig(filename='test_camels_lux.log', filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

import aqua_fetch
from aqua_fetch import download_zenodo   # so that aqua_fetch.download_zenodo can be patched
from aqua_fetch import CAMELS_LUX
from aqua_fetch._backend import xarray as xr, fiona

from utils import test_dataset as run_shared_tests

# The class appends ``CAMELS_LUX`` to this path. Replace with the location on
# your machine, or set the CAMELS_LUX_PATH environment variable. Point it at a
# folder that does not already hold release 1.1: initializing the class there
# deletes that release and its netCDF caches (see test_old_release_is_replaced)
# and downloads release 2.1 in their place.
CAMELS_LUX_PATH = os.environ.get(
    'CAMELS_LUX_PATH',
    '/mnt/storage1/atr/data/gscad_database/raw/CAMELS')

NUM_STATIONS = 56
NUM_STATIC = 61
NUM_DYNAMIC = 26

# steps and bounds of each timestep, derived from the release itself
DYN_LEN = {'D': 6209, 'H': 149016, '15Min': 596061}
EXTENT = {'D': ('2004-11-01', '2021-10-31'),
          'H': ('2004-11-01 01:00:00', '2021-11-01 00:00:00'),
          '15Min': ('2004-11-01 01:00:00', '2021-11-01 00:00:00')}

# standardized name -> (raw column, factor). Written out here rather than taken
# from the class, so that a wrong mapping or a forgotten factor is caught.
RAW_COLUMNS = {
    'q_cms_obs': ('Q', 1.0),
    'q_mm_obs': ('Qspec', 1.0),
    'Qflag': ('Qflag', 1.0),
    'pcp_mm_radar': ('RR_rad', 1.0),
    'RR_min_rad': ('RR_min_rad', 1.0),
    'RR_max_rad': ('RR_max_rad', 1.0),
    'RR_flag_rad': ('RR_flag_rad', 1.0),
    'pcp_mm_station': ('RR_stn', 1.0),
    'pcp_mm_era5': ('tp', 1.0),
    'airtemp_C_mean_era5': ('t2m', 1.0),
    'airtemp_C_mean_station': ('T_stn', 1.0),   # added by release 2.1
    'pet_mm_oudin': ('PET_Oudin', 1.0),
    'pet_mm_pm': ('PET_PM', 1.0),
    'cape': ('cape', 1.0),
    'cin': ('cin', 1.0),
    'kx': ('kx', 1.0),
    'spechum_gkg': ('q', 1000.0),               # published in kg/kg
    'rh_%': ('rh', 1.0),
    'tcwv': ('tcwv', 1.0),
    'windspeed_mps': ('ws10500', 1.0),
    'lls': ('lls', 1.0),
    'dls': ('dls', 1.0),
    'sml1': ('swvl1', 1.0),
    'sml2': ('swvl2', 1.0),
    'sml3': ('swvl3', 1.0),
    'sml4': ('swvl4', 1.0),
}

ATTRIBUTE_FILES = ('CAMELS_LUX_meta_attributes.csv',
                   'CAMELS_LUX_climatic_attributes.csv',
                   'CAMELS_LUX_geologic_attributes.csv',
                   'CAMELS_LUX_landuse_attributes.csv',
                   'CAMELS_LUX_topographic_attributes.csv')

# the download of release 2.1 happens here, once
dataset = CAMELS_LUX(path=CAMELS_LUX_PATH, verbosity=0)


def raw_timeseries(stn: str, timestep: str = 'D') -> pd.DataFrame:
    """the gauge's file read independently of the class"""
    token = {'D': '_daily', 'H': 'hourly', '15Min': '15min'}[timestep]
    folder = {'D': 'daily', 'H': 'hourly', '15Min': '15min'}[timestep]
    fpath = os.path.join(dataset.ts_path, folder,
                         f"CAMELS_LUX_hydromet_timeseries_{token}_{stn}.csv")
    return pd.read_csv(fpath, index_col=0, parse_dates=True)


def raw_attributes(fname: str) -> pd.DataFrame:
    """one attribute file read independently of the class"""
    return pd.read_csv(os.path.join(dataset._root, fname), index_col=0, dtype={0: str})


def _copy_for(path: str) -> CAMELS_LUX:
    """a copy of the dataset which points at ``path`` instead of the downloaded
    data, so that the download code can be exercised without a network"""
    ds = copy.copy(dataset)
    ds._path = path
    return ds


def _fake_download(calls, release: str = '2.1'):
    """a stand-in for download_from_zenodo which records its calls and writes
    small zips holding the folder layout of the real archives. With
    ``release='1.1'`` it writes the layout of the superseded release instead,
    which has basin_id.csv and differently named time series files."""
    def download_from_zenodo(outdir, doi, include=None, verbosity=1, **kwargs):
        calls.append((outdir, doi, tuple(include or ())))
        if release == '2.1':
            files = {'CAMELS_LUX_meta_attributes.csv': 'gauge_id,Lat,Lon\nID_01,49.5,6.1\n',
                     'timeseries/daily/CAMELS_LUX_hydromet_timeseries__daily_ID_01.csv':
                         'Date,Q\n2004-11-01,0.883\n'}
        else:
            files = {'CAMELS_LUX_meta_attributes.csv': 'gauge_id,Lat,Lon\nID_01,49.5,6.1\n',
                     'basin_id.csv': 'ID_01\n',
                     'timeseries/daily/CAMELS_LUX_hydromet_timeseries_ID_01.csv':
                         'Date,Q\n2004-11-01,0.883\n'}
        if 'CAMELS-LUX.zip' in (include or ()):
            with zipfile.ZipFile(os.path.join(outdir, 'CAMELS-LUX.zip'), 'w') as zf:
                for name, content in files.items():
                    zf.writestr(name, content)
        if 'CAMELS-LUX_shapefiles.zip' in (include or ()):
            with zipfile.ZipFile(os.path.join(outdir, 'CAMELS-LUX_shapefiles.zip'), 'w') as zf:
                zf.writestr('catchments_CAMELS-LUX.shp', 'x')
    return download_from_zenodo


def _old_release(path: str) -> dict:
    """lays down a release 1.1 as an earlier version of this class left it: the
    archives extracted into CAMELS-LUX/ and CAMELS-LUX_shapefiles/, and the
    caches built from it. Returns the paths it created."""
    root = os.path.join(path, 'CAMELS-LUX')
    os.makedirs(os.path.join(root, 'timeseries', 'daily'))
    os.makedirs(os.path.join(path, 'CAMELS-LUX_shapefiles'))
    created = {'root': root,
               'basin_id': os.path.join(root, 'basin_id.csv'),
               'ts': os.path.join(root, 'timeseries', 'daily',
                                  'CAMELS_LUX_hydromet_timeseries_ID_01.csv'),
               'boundaries': os.path.join(path, 'CAMELS-LUX_shapefiles'),
               'archive': os.path.join(path, 'CAMELS-LUX.zip'),
               'cache_v1': os.path.join(path, 'camels_lux_D.nc'),
               'cache_v2': os.path.join(path, 'camels_lux_D_v2.nc'),
               'cache_h': os.path.join(path, 'camels_lux_H_v2.nc')}
    for key in ('basin_id', 'ts', 'archive', 'cache_v1', 'cache_v2', 'cache_h'):
        open(created[key], 'w').close()
    return created


# ------------------------------------------------------------------ contents

def test_stations_and_features():
    """the release's own 56 gauges, 26 dynamic and 61 static features"""
    logger.info("test_stations_and_features")
    stations = dataset.stations()
    assert len(stations) == NUM_STATIONS, len(stations)
    assert len(set(stations)) == NUM_STATIONS, "a gauge is listed twice"
    assert stations[0] == 'ID_01' and stations[-1] == 'ID_56', stations[:3]

    # the gauges of the metadata file: release 2.1 dropped basin_id.csv, which
    # an earlier version of this class read
    expected = set(raw_attributes('CAMELS_LUX_meta_attributes.csv').index)
    assert set(stations) == expected, expected.symmetric_difference(stations)
    assert not os.path.exists(os.path.join(dataset._root, 'basin_id.csv')), \
        "basin_id.csv is not part of release 2.1"

    assert len(dataset.dynamic_features) == NUM_DYNAMIC, dataset.dynamic_features
    assert set(dataset.dynamic_features) == set(RAW_COLUMNS), \
        set(dataset.dynamic_features).symmetric_difference(RAW_COLUMNS)
    assert len(dataset.static_features) == NUM_STATIC, len(dataset.static_features)

    # the features release 2.1 has and release 1.1 did not, and the other way
    assert 'airtemp_C_mean_station' in dataset.dynamic_features
    assert 'P_PET_Oudin' in dataset.static_features
    assert 'AI_Oudin' not in dataset.static_features, "release 1.1 aridity index"
    assert 't2m_mean' not in dataset.static_features, "release 1.1 climatic temperature"
    assert 'T_stn_mean' in dataset.static_features
    assert 'limestone_dolomite' in dataset.static_features
    assert 'limestone_dolomites' not in dataset.static_features, "release 1.1 spelling"
    return


def test_dynamic_values_and_units():
    """every value read from a gauge's file is the published one, times the
    documented factor, on the published date"""
    logger.info("test_dynamic_values_and_units")
    stations = dataset.stations()
    for stn in (stations[0], stations[len(stations) // 2], stations[-1], 'ID_02'):
        raw = raw_timeseries(stn)
        served = dataset._read_stn_dyn(stn)

        assert served.shape == (len(raw), NUM_DYNAMIC), served.shape
        assert served.index.equals(raw.index), stn

        for name, (column, factor) in RAW_COLUMNS.items():
            expected = raw[column].values * factor
            np.testing.assert_allclose(
                served[name].values.astype('float64'), expected,
                rtol=1e-12, atol=0, equal_nan=True,
                err_msg=f"{stn}: {name} is not {column} x {factor}")
    return


def test_specific_humidity_is_g_per_kg():
    """specific humidity is published in kg/kg and spechum_gkg promises g/kg, so
    the served values must be a thousand times the published ones"""
    logger.info("test_specific_humidity_is_g_per_kg")
    raw = raw_timeseries('ID_02')
    _, dyn = dataset.fetch(stations='ID_02', as_dataframe=True)
    served = dyn['ID_02']['spechum_gkg']

    np.testing.assert_allclose(served.values.astype('float64'),
                               raw['q'].values * 1000.0, rtol=1e-6, equal_nan=True)
    # a few g/kg at 700 hPa; kg/kg values would be ~0.003
    assert 0.1 <= float(served.mean()) <= 20.0, served.mean()
    return


def test_short_gauges_are_padded_not_shifted():
    """a gauge whose file starts late keeps its own rows: read from the file it
    has only its own dates, and fetched through the netCDF cache it is padded
    with NaN before them instead of being moved to the start of the record"""
    logger.info("test_short_gauges_are_padded_not_shifted")
    raw = raw_timeseries('ID_55')
    assert raw.index[0] == pd.Timestamp('2015-01-01'), raw.index[0]

    from_file = dataset._read_stn_dyn('ID_55')
    assert from_file.index.equals(raw.index)
    assert len(from_file) < DYN_LEN['D'], len(from_file)

    _, dyn = dataset.fetch(stations='ID_55', as_dataframe=True)
    served = dyn['ID_55']
    assert len(served) == DYN_LEN['D'], len(served)
    assert served.index[0] == pd.Timestamp(EXTENT['D'][0]), served.index[0]
    # NaN before the gauge's first date, its own values from then on
    assert served.loc[:'2014-12-31', 'q_cms_obs'].isna().all()
    np.testing.assert_allclose(
        served.loc[raw.index, 'q_cms_obs'].values.astype('float64'),
        raw['Q'].values, rtol=1e-5, equal_nan=True)
    return


def test_static_values():
    """static attributes are the published ones, with the land use attributes of
    the static_map converted from percent and the rest left as published"""
    logger.info("test_static_values")
    static = dataset.fetch_static_features('all', static_features='all')
    assert static.shape == (NUM_STATIONS, NUM_STATIC), static.shape

    meta = raw_attributes('CAMELS_LUX_meta_attributes.csv')
    landuse = raw_attributes('CAMELS_LUX_landuse_attributes.csv')
    topo = raw_attributes('CAMELS_LUX_topographic_attributes.csv')
    climatic = raw_attributes('CAMELS_LUX_climatic_attributes.csv')
    static = static.loc[meta.index]

    np.testing.assert_allclose(static['area_km2'].astype('float64'), meta['area_km2'])
    np.testing.assert_allclose(static['lat'].astype('float64'), meta['Lat'])
    np.testing.assert_allclose(static['long'].astype('float64'), meta['Lon'])
    np.testing.assert_allclose(static['perimeter_km'].astype('float64'), meta['perimeter_km'])
    np.testing.assert_allclose(static['elev_catch_m'].astype('float64'), topo['Z_MEAN'])
    np.testing.assert_allclose(static['slope_degree'].astype('float64'), topo['SLOPE_MEAN'])

    # converted from percent, and their sister attributes are not
    for standardized, published in (('grass_frac', 'grassland'),
                                    ('crop_frac', 'agricultural_land'),
                                    ('urban_frac', 'urban')):
        np.testing.assert_allclose(static[standardized].astype('float64'),
                                   landuse[published] / 100.0, rtol=1e-6)
    np.testing.assert_allclose(static['forests_naturalareas'].astype('float64'),
                               landuse['forests_naturalareas'])
    np.testing.assert_allclose(static['runoffratio'].astype('float64'),
                               climatic['runoffratio'])
    return


def test_static_feature_order_is_deterministic():
    """the static features are read from a fixed list of files, so their order
    does not depend on the order the file system returns them in"""
    logger.info("test_static_feature_order_is_deterministic")
    again = CAMELS_LUX(path=CAMELS_LUX_PATH, verbosity=0)
    assert again.static_features == dataset.static_features

    # the blocks follow _attr_files: the metadata file first, the topographic
    # one last, each in the order the file has them
    expected = []
    for fname in ATTRIBUTE_FILES:
        columns = raw_attributes(fname).columns
        expected += [dataset.static_map.get(column, column) for column in columns]
    assert dataset.static_features == expected, \
        set(dataset.static_features).symmetric_difference(expected)
    return


def test_temporal_extent_is_derived():
    """start and end are the first and last timestamp of the data, not the
    hardcoded 2004-01-01 .. 2021-12-31 an earlier version of this class used"""
    logger.info("test_temporal_extent_is_derived")
    start, end = EXTENT['D']
    assert dataset.start == pd.Timestamp(start), dataset.start
    assert dataset.end == pd.Timestamp(end), dataset.end

    # and they really are the bounds over all gauges, ten of which start late
    firsts, lasts = [], []
    for stn in dataset.stations():
        index = raw_timeseries(stn).index
        firsts.append(index[0])
        lasts.append(index[-1])
    assert dataset.start == min(firsts), (dataset.start, min(firsts))
    assert dataset.end == max(lasts), (dataset.end, max(lasts))
    return


def test_boundary_maps_to_the_right_gauge():
    """release 2.1 put ``Area`` first in the catchments shapefile and added
    ``gauge_id``; without using the latter every boundary would be filed under
    an id like ID_258344800 and get_boundary would raise"""
    logger.info("test_boundary_maps_to_the_right_gauge")
    if fiona is None:
        logger.info("fiona is not installed, skipped")
        return

    bounds = dataset._create_boundary_id_map()
    assert set(bounds) == set(dataset.stations()), \
        set(bounds).symmetric_difference(dataset.stations())

    # the gauge is inside its own catchment's bounding box
    coords = dataset.stn_coords()
    for stn in ('ID_01', 'ID_02', 'ID_50', 'ID_56'):
        geometry = dataset.get_boundary(stn)
        assert geometry.type in ('Polygon', 'MultiPolygon'), geometry.type
        xy = np.array([point for ring in _rings(geometry['coordinates']) for point in ring])
        lat, lon = float(coords.loc[stn, 'lat']), float(coords.loc[stn, 'long'])
        assert xy[:, 0].min() - 0.01 <= lon <= xy[:, 0].max() + 0.01, stn
        assert xy[:, 1].min() - 0.01 <= lat <= xy[:, 1].max() + 0.01, stn
    return


def _rings(coords):
    """every ring of a Polygon or MultiPolygon as a list of (long, lat)"""
    if coords and isinstance(coords[0][0], (int, float)):
        return [coords]
    out = []
    for part in coords:
        out.extend(_rings(part))
    return out


def test_area_and_boundary():
    """areas come from the attributes and the boundaries are readable"""
    logger.info("test_area_and_boundary")
    meta = raw_attributes('CAMELS_LUX_meta_attributes.csv')
    area = dataset.area('all')
    np.testing.assert_allclose(area.loc[meta.index].astype('float64'),
                               meta['area_km2'], rtol=1e-6)
    return


def test_static_data_is_a_copy():
    """a caller editing the returned table does not corrupt the cached one"""
    logger.info("test_static_data_is_a_copy")
    first = dataset._static_data()
    first.iloc[0, 0] = 'corrupted'
    assert dataset._static_data().iloc[0, 0] != 'corrupted'

    stations = dataset.stations()
    stations.append('ID_99')
    assert len(dataset.stations()) == NUM_STATIONS

    features = dataset.dynamic_features
    features.append('nonsense')
    assert len(dataset.dynamic_features) == NUM_DYNAMIC
    return


def test_cache_matches_csv():
    """the netCDF cache serves exactly what the csv files hold"""
    logger.info("test_cache_matches_csv")
    if xr is None:
        logger.info("xarray is not installed, skipped")
        return
    assert dataset.dyn_fname == f'camels_lux_D_{CAMELS_LUX.release}_v2.nc', dataset.dyn_fname
    assert dataset.dyn_fpath_exists, "the cache was not built"

    _, from_cache = dataset.fetch(stations='ID_02', as_dataframe=True)
    served = from_cache['ID_02']
    from_file = dataset._read_stn_dyn('ID_02')

    # the full index of the release, not one built from a file's own two ends
    assert served.index.equals(pd.date_range(*EXTENT['D'], freq='D')), served.index
    assert served.index.equals(from_file.index)
    # the cache is written at the dataset's float precision, hence rtol=1e-5
    for name in RAW_COLUMNS:
        np.testing.assert_allclose(
            served[name].values.astype('float64'),
            from_file[name].values.astype('float64'), rtol=1e-5, equal_nan=True,
            err_msg=f"cache disagrees with the csv for {name}")
    return


def test_other_timesteps():
    """the hourly and 15 minute data have their own files, lengths and extent"""
    logger.info("test_other_timesteps")
    for timestep in ('H', '15Min'):
        # to_netcdf=False: the 15 minute cache alone is 6.7 GB
        ds = CAMELS_LUX(path=CAMELS_LUX_PATH, timestep=timestep,
                        to_netcdf=False, verbosity=0)
        assert len(ds.stations()) == NUM_STATIONS
        assert len(ds.dynamic_features) == NUM_DYNAMIC, ds.dynamic_features
        assert ds.dyn_fname == f'camels_lux_{timestep}_{CAMELS_LUX.release}_v2.nc'

        served = ds._read_stn_dyn('ID_02')
        assert len(served) == DYN_LEN[timestep], (timestep, len(served))
        assert not served.index.has_duplicates, timestep

        start, end = EXTENT[timestep]
        assert ds.start == pd.Timestamp(start), (timestep, ds.start)
        assert ds.end == pd.Timestamp(end), (timestep, ds.end)

        raw = raw_timeseries('ID_02', timestep)
        np.testing.assert_allclose(served['q_cms_obs'].values.astype('float64'),
                                   raw['Q'].values, rtol=1e-6, equal_nan=True)
    return


def test_timestep_is_validated():
    """a timestep the dataset does not have is refused with a clear error"""
    logger.info("test_timestep_is_validated")
    for bad in ('15min', 'M', 'daily'):
        try:
            CAMELS_LUX(path=CAMELS_LUX_PATH, timestep=bad, verbosity=0)
        except ValueError as e:
            assert 'timestep must be one of' in str(e), e
        else:
            raise AssertionError(f"timestep={bad!r} was accepted")
    return


def test_fetch_all_stations():
    """all 56 gauges with all features, from the cache and from the csv files"""
    logger.info("test_fetch_all_stations")
    start = time.time()
    _, dyn = dataset.fetch(stations='all', as_dataframe=True)
    took = time.time() - start
    assert len(dyn) == NUM_STATIONS, len(dyn)
    assert all(df.shape[1] == NUM_DYNAMIC for df in dyn.values())
    logger.info(f"fetched all {NUM_STATIONS} gauges in {took:.1f} s")
    assert took < 120, f"fetching all gauges took {took:.1f} s"
    return


def test_shared():
    """the checks every rainfall-runoff dataset has to pass"""
    logger.info("test_shared")
    run_shared_tests(dataset, NUM_STATIONS, DYN_LEN['D'], NUM_STATIC, NUM_DYNAMIC,
                     st="20120101", en="20121231", yearly_steps=366)
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
        again = CAMELS_LUX(path=CAMELS_LUX_PATH, verbosity=0)
        assert len(again.stations()) == NUM_STATIONS
    finally:
        aqua_fetch.download_zenodo.download_from_zenodo = originals[0]
        zipfile.ZipFile.extractall = originals[1]
        if xr is not None:
            xr.Dataset.to_netcdf = originals[2]
    return


def test_download_and_extract():
    """both archives are downloaded once and extracted into their own folder;
    once extracted nothing is downloaded again and remove_zip deletes them"""
    logger.info("test_download_and_extract")
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'CAMELS_LUX')       # does not exist yet
    calls = []
    original = aqua_fetch.download_zenodo.download_from_zenodo
    try:
        aqua_fetch.download_zenodo.download_from_zenodo = _fake_download(calls)
        ds = _copy_for(path)
        ds.remove_zip = False
        ds._download_camels_lux()

        assert len(calls) == 1, calls
        outdir, doi, include = calls[0]
        assert outdir == path and doi == CAMELS_LUX.url
        assert sorted(include) == ['CAMELS-LUX.zip', 'CAMELS-LUX_shapefiles.zip'], \
            "the dataset description was downloaded too"
        # the archives hold their files at the root, so the folders are named
        # after the archives and not nested twice
        assert os.path.isfile(os.path.join(ds._root, 'CAMELS_LUX_meta_attributes.csv')), \
            os.listdir(ds._root)
        assert os.path.isdir(ds._boundary_dir), os.listdir(path)

        calls.clear()
        ds.remove_zip = True
        ds._download_camels_lux()
        assert calls == [], f"downloaded again although the data is extracted: {calls}"
        assert sorted(os.listdir(path)) == ['CAMELS-LUX', 'CAMELS-LUX_shapefiles'], \
            "remove_zip left an archive"

        # and with the archives gone it is still not downloaded again
        ds._download_camels_lux()
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
    path = os.path.join(tmp, 'CAMELS_LUX')
    calls = []
    original_download = aqua_fetch.download_zenodo.download_from_zenodo
    original_extract = zipfile.ZipFile.extractall
    try:
        aqua_fetch.download_zenodo.download_from_zenodo = _fake_download(calls)
        ds = _copy_for(path)
        ds.remove_zip = False

        def die(self, *args, **kwargs):
            original_extract(self, *args, **kwargs)
            raise KeyboardInterrupt("interrupted half way")

        zipfile.ZipFile.extractall = die
        try:
            ds._download_camels_lux()
        except KeyboardInterrupt:
            pass
        zipfile.ZipFile.extractall = original_extract

        assert not os.path.isdir(ds._root), "a half extracted folder looks complete"

        calls.clear()
        ds._download_camels_lux()
        assert calls == [], f"downloaded again although the archive is there: {calls}"
        assert os.path.isfile(os.path.join(ds._root, 'CAMELS_LUX_meta_attributes.csv'))
    finally:
        aqua_fetch.download_zenodo.download_from_zenodo = original_download
        zipfile.ZipFile.extractall = original_extract
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_old_release_is_replaced():
    """a release 1.1 left by an earlier version of this class is deleted, warned
    about and replaced by release 2.1, caches included"""
    logger.info("test_old_release_is_replaced")
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'CAMELS_LUX')
    calls = []
    original = aqua_fetch.download_zenodo.download_from_zenodo
    try:
        aqua_fetch.download_zenodo.download_from_zenodo = _fake_download(calls)
        created = _old_release(path)
        ds = _copy_for(path)
        ds.remove_zip = False

        assert ds._holds_old_release(), "release 1.1 was not recognized"

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds._download_camels_lux()
        messages = [str(w.message) for w in caught]
        assert any('holds release 1.1' in m for m in messages), messages

        assert len(calls) == 1, calls
        # everything of release 1.1 is gone
        assert not os.path.exists(created['basin_id']), "basin_id.csv survived"
        assert not os.path.exists(created['ts']), "a release 1.1 time series survived"
        for key in ('cache_v1', 'cache_v2', 'cache_h'):
            assert not os.path.exists(created[key]), f"{key} survived"
        # and release 2.1 is in its place
        assert not ds._holds_old_release()
        assert os.path.isfile(os.path.join(
            ds._root, 'timeseries', 'daily',
            'CAMELS_LUX_hydromet_timeseries__daily_ID_01.csv'))
    finally:
        aqua_fetch.download_zenodo.download_from_zenodo = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_old_release_cache_is_not_served():
    """the cache name carries the release, so a cache built from release 1.1 is
    never read as release 2.1"""
    logger.info("test_old_release_cache_is_not_served")
    assert CAMELS_LUX.release == '2.1'
    assert dataset.release in dataset.dyn_fname, dataset.dyn_fname
    old_names = {os.path.basename(p) for p in dataset._old_release_caches}
    assert 'camels_lux_D.nc' in old_names and 'camels_lux_D_v2.nc' in old_names, old_names
    assert os.path.basename(dataset.dyn_fpath) not in old_names
    return


def test_overwrite_removes_stale_before_download():
    """overwrite=True deletes the archives, the extracted files and the netCDF
    caches before downloading again"""
    logger.info("test_overwrite_removes_stale_before_download")
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'CAMELS_LUX')
    calls = []
    original = aqua_fetch.download_zenodo.download_from_zenodo
    try:
        aqua_fetch.download_zenodo.download_from_zenodo = _fake_download(calls)
        ds = _copy_for(path)
        ds.remove_zip = False
        ds._download_camels_lux()

        stale = os.path.join(ds._root, 'stale.csv')
        open(stale, 'w').close()
        open(ds.dyn_fpath, 'w').close()
        open(os.path.join(path, 'camels_lux_D_v2.nc'), 'w').close()

        calls.clear()
        ds._download_camels_lux(overwrite=True)

        assert len(calls) == 1, calls
        assert not os.path.exists(stale), "the stale extracted file survived"
        assert not os.path.exists(os.path.join(path, 'camels_lux_D_v2.nc')), \
            "the release 1.1 cache survived"
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
        path = os.path.join(tmp, 'CAMELS_LUX')
        root = os.path.join(path, 'CAMELS-LUX')
        os.makedirs(os.path.join(root, 'timeseries', 'daily'))
        shutil.copy(os.path.join(dataset._root, 'CAMELS_LUX_meta_attributes.csv'), root)
        ds = _copy_for(path)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds._check_manifest()
        messages = [str(w.message) for w in caught]
        # the 4 other attribute files, 4 shapefile files and 56 time series
        expected = f'{4 + 4 + NUM_STATIONS} of {5 + 4 + NUM_STATIONS} files are missing'
        assert any(expected in m for m in messages), messages

        # and a complete release does not warn
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dataset._check_manifest()
        assert [str(w.message) for w in caught] == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_reads_after_free_disk_space():
    """once the csv files are removed the class reads the feature names and the
    temporal extent from the netCDF cache instead of re-downloading"""
    logger.info("test_reads_after_free_disk_space")
    if xr is None:
        logger.info("xarray is not installed, skipped")
        return
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, 'CAMELS_LUX')
        root = os.path.join(path, 'CAMELS-LUX')
        os.makedirs(root)
        for fname in ATTRIBUTE_FILES:
            shutil.copy(os.path.join(dataset._root, fname), root)
        shutil.copytree(dataset._boundary_dir, os.path.join(path, 'CAMELS-LUX_shapefiles'))
        shutil.copy(dataset.dyn_fpath, os.path.join(path, os.path.basename(dataset.dyn_fpath)))

        ds = _copy_for(path)
        ds.__dict__.pop('_dyn_features', None)
        ds.__dict__.pop('_time_extent', None)
        ds.__dict__.pop('_static_df', None)

        # the timeseries folder is gone, as free_disk_space("redundant") leaves it
        assert not os.path.isdir(ds._ts_dir('D'))
        assert not ds._holds_old_release(), "an absent timeseries folder is not release 1.1"

        assert len(ds.dynamic_features) == NUM_DYNAMIC, ds.dynamic_features
        assert set(ds.dynamic_features) == set(RAW_COLUMNS)
        assert ds.start == pd.Timestamp(EXTENT['D'][0]), ds.start
        assert ds.end == pd.Timestamp(EXTENT['D'][1]), ds.end

        # the pairs free_disk_space would remove point at this release's caches
        for folder, cache in ds._redundant_after_consolidation():
            assert CAMELS_LUX.release in os.path.basename(cache), cache
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
