"""
Tests for CAMELS_IND (Peninsular India, 472 catchments), mainly for release 2.2.

They check that the class

    * serves the raw values, units and dates unchanged (compared with an
      independent read of the raw files),
    * reads the static attributes, coordinates, areas and boundaries faithfully,
      and that the gauge metadata of release 2.2 describes the catchment whose
      boundary and time series are served under the same id,
    * downloads once, neither downloads nor extracts again, and on
      ``overwrite=True`` deletes the stale files of that release only,
    * does not extract the LSTM simulated streamflow (model output),
    * keeps release 2 readable through ``version='2'`` without mixing the two,
    * fetches all stations quickly.

The first run downloads release 2.2 (350 MB, 881 MB extracted) into
``CAMELS_IND_PATH``. The few tests which need release 2 download another 360 MB.
Run as a script or with pytest.
"""

import os
import sys
import copy
import time
import inspect
import contextlib
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
sys.path.insert(0, wd_dir)

if __name__ == "__main__":
    logging.basicConfig(filename='test_camels_ind.log', filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

import aqua_fetch
from aqua_fetch import CAMELS_IND
from aqua_fetch import download_zenodo
from aqua_fetch._backend import xarray as xr, fiona

from utils import test_dataset as run_shared_tests

# The class appends ``CAMELS_IND`` to this path; both releases can live there.
# Replace with the location on your machine.
CAMELS_IND_PATH = '/mnt/hyexnas01/abbaa0a/data/gscad_database/raw/CAMELS/CAMELS'

NUM_STATIONS = 472
NUM_STATIC = 210
NUM_DYNAMIC = 20
DYN_LEN = 14976          # daily steps 1980-01-01 .. 2020-12-31

# standardized name -> column of the forcing file. Written out here rather than
# taken from the class, so that a wrong mapping is caught.
RAW_COLUMNS = {
    'pcp_mm': 'prcp(mm/day)',
    'airtemp_C_max': 'tmax(C)',
    'airtemp_C_min': 'tmin(C)',
    'airtemp_C_mean': 'tavg(C)',
    'lwdownrad_wm2': 'srad_lw(w/m2)',
    'swdownrad_wm2': 'srad_sw(w/m2)',
    'windspeedu_mps': 'wind_u(m/s)',
    'windspeedv_mps': 'wind_v(m/s)',
    'windspeed_mps': 'wind(m/s)',
    'rh_%': 'rel_hum(%)',
    'pet_mm': 'pet(mm/day)',
    'pet_mm_gleam': 'pet_gleam(mm/day)',
    'aet_mm_gleam': 'aet_gleam(mm/day)',
}

# the forcing columns which keep the name the files give them
UNMAPPED_COLUMNS = {
    '2': ('evap_canopy(kg/m2/s)', 'evap_surface(kg/m2/s)'),
    '2.2': ('evap_canopy(mm/day)', 'evap_surface(mm/day)'),
}

# the 8 attribute files of a release, without the release's prefix
ATTRIBUTE_FILES = ('name', 'topo', 'clim', 'hydro', 'land', 'soil', 'geol', 'anth')

# gauges whose metadata release 2 attaches to the next catchment
REKEYED_GAUGES = ('12001', '12042', '15001')

# what the download stub writes when it has no archive to copy
NOT_A_ZIP = b'<html>404 not found</html>'

# the download of release 2.2 happens here, once
dataset = CAMELS_IND(path=CAMELS_IND_PATH, verbosity=0)

_datasets = {'2.2': dataset}


def dataset_of(version: str) -> CAMELS_IND:
    """the dataset of ``version``, built once"""
    if version not in _datasets:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _datasets[version] = CAMELS_IND(path=CAMELS_IND_PATH, version=version,
                                            verbosity=0)
    return _datasets[version]


def version_dir(ds: CAMELS_IND) -> str:
    """folder which holds the files of ``ds``, worked out from the layout of the
    release rather than taken from the class: release 2 is extracted directly
    into ``path``, release 2.2 into a folder named after its archive"""
    if ds.version == '2':
        return ds.path
    return os.path.join(ds.path, 'CAMELS_IND_All_Catchments')


def attr_prefix(version: str) -> str:
    return 'camels_India_' if version == '2' else 'camels_ind_'


def _raw_attributes(ds: CAMELS_IND, name: str) -> pd.DataFrame:
    """one attribute file, read independently of the class"""
    fpath = os.path.join(version_dir(ds), 'attributes_txt',
                         f"{attr_prefix(ds.version)}{name}.txt")
    return pd.read_csv(fpath, sep=';', index_col=0, dtype={0: str})


def _raw_forcings(ds: CAMELS_IND, station: str) -> pd.DataFrame:
    """one forcing file, read independently of the class"""
    folder = os.path.join(version_dir(ds), 'catchment_mean_forcings')
    gauge_id = station.zfill(5)
    if ds.version == '2':
        folder = os.path.join(folder, gauge_id[:2])
    df = pd.read_csv(os.path.join(folder, f"{gauge_id}.csv"))
    df.index = pd.to_datetime(df[['year', 'month', 'day']])
    return df.drop(columns=['year', 'month', 'day'])


def _raw_streamflow(ds: CAMELS_IND, station: str) -> pd.Series:
    """the streamflow column of one gauge, read independently of the class"""
    fpath = os.path.join(version_dir(ds), 'streamflow_timeseries',
                         'streamflow_observed.csv')
    df = pd.read_csv(fpath, usecols=['year', 'month', 'day', station])
    df.index = pd.to_datetime(df[['year', 'month', 'day']])
    return df[station]


def _polygon_area_km2(geometry) -> float:
    """
    Area of a fiona (Multi)Polygon in WGS84, by the shoelace formula on an
    equirectangular projection around the polygon's own latitude. Good to a few
    percent, which is all the alignment checks need.
    """
    polygons = (geometry['coordinates'] if geometry['type'] == 'MultiPolygon'
                else [geometry['coordinates']])
    lat0 = np.radians(polygons[0][0][0][1])
    radius = 6371.0088

    area = 0.0
    for polygon in polygons:
        for n, ring in enumerate(polygon):
            xy = np.asarray(ring, dtype=float)
            x = np.radians(xy[:, 0]) * np.cos(lat0) * radius
            y = np.radians(xy[:, 1]) * radius
            shoelace = abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])) / 2
            # the first ring is the polygon, the others are holes
            area += shoelace if n == 0 else -shoelace
    return area


def _mini_copy(version: str = '2.2', stations=('3001', '12001'), into=None) -> str:
    """
    A temporary copy of ``version`` which holds only ``stations``: the attribute
    files (with the name file trimmed to those gauges), their forcing files, the
    streamflow of those gauges and the boundary files. The class is fully usable
    on it, so the tests which write into the dataset folder do not touch the
    real copy and stay fast.
    """
    ds = dataset_of(version)
    src, tmp = version_dir(ds), into or tempfile.mkdtemp()
    dst = os.path.join(tmp, 'CAMELS_IND')
    if version != '2':
        dst = os.path.join(dst, 'CAMELS_IND_All_Catchments')

    static = os.path.join(dst, 'attributes_txt')
    os.makedirs(static, exist_ok=True)
    for name in ATTRIBUTE_FILES:
        fname = f"{attr_prefix(version)}{name}.txt"
        shutil.copy(os.path.join(src, 'attributes_txt', fname),
                    os.path.join(static, fname))

    # the name file decides which gauges the class expects to find
    name_file = os.path.join(static, f"{attr_prefix(version)}name.txt")
    names = pd.read_csv(name_file, sep=';', dtype={0: str})
    keep = [stn.zfill(5) for stn in stations]
    names[names['gauge_id'].isin(keep)].to_csv(name_file, sep=';', index=False)

    forcings = os.path.join(dst, 'catchment_mean_forcings')
    for stn in keep:
        folder = os.path.join(forcings, stn[:2]) if version == '2' else forcings
        os.makedirs(folder, exist_ok=True)
        src_folder = (os.path.join(src, 'catchment_mean_forcings', stn[:2])
                      if version == '2' else os.path.join(src, 'catchment_mean_forcings'))
        shutil.copy(os.path.join(src_folder, f"{stn}.csv"),
                    os.path.join(folder, f"{stn}.csv"))

    if version == '2':
        # the class checks that every archive of release 2 is extracted; its
        # attributes_csv duplicates attributes_txt and is never read
        os.makedirs(os.path.join(dst, 'attributes_csv'), exist_ok=True)

    q_dir = os.path.join(dst, 'streamflow_timeseries')
    os.makedirs(q_dir, exist_ok=True)
    q = pd.read_csv(os.path.join(src, 'streamflow_timeseries', 'streamflow_observed.csv'),
                    usecols=['year', 'month', 'day'] + list(stations))
    q.to_csv(os.path.join(q_dir, 'streamflow_observed.csv'), index=False)

    boundary = os.path.dirname(ds.boundary_file)
    target = os.path.join(dst, os.path.relpath(boundary, src))
    os.makedirs(target, exist_ok=True)
    for fname in os.listdir(boundary):
        if fname.startswith('all_catchments'):
            shutil.copy(os.path.join(boundary, fname), os.path.join(target, fname))
    return tmp


def _archive_of(mini: str, version: str = '2.2') -> str:
    """
    Zips the mini copy the way the Zenodo record ships it, with a simulated
    streamflow file added, and returns the folder holding the archive(s).
    """
    root = os.path.join(mini, 'CAMELS_IND')
    source = root if version == '2' else os.path.join(root, 'CAMELS_IND_All_Catchments')
    out = tempfile.mkdtemp()

    for stem in CAMELS_IND._ARCHIVES[version]:
        folder = os.path.join(source, stem) if version == '2' else source
        if not os.path.isdir(folder):   # e.g. attributes_csv, which is not copied
            os.makedirs(folder)
        with zipfile.ZipFile(os.path.join(out, f"{stem}.zip"), 'w') as zf:
            for dirpath, _, fnames in os.walk(folder):
                for fname in fnames:
                    fpath = os.path.join(dirpath, fname)
                    zf.write(fpath, os.path.relpath(fpath, folder))
            lstm = ('streamflow_timeseries/lstm_pred_streamflow.csv' if version != '2'
                    else 'LSTM_pred_streamflow.csv')
            if stem in ('CAMELS_IND_All_Catchments', 'streamflow_timeseries'):
                zf.writestr(lstm, 'year,month,day,3001\n1980,1,1,1.5\n')
    return out


def _fake_zenodo(calls, archives: str):
    """a stand-in for download_from_zenodo which records its calls and copies
    the prepared archives instead of downloading them"""
    def download_from_zenodo(outdir, doi, include=None, verbosity=1, **kwargs):
        calls.append((doi, outdir, tuple(include)))
        os.makedirs(outdir, exist_ok=True)
        for fname in include:
            src = os.path.join(archives, fname)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(outdir, fname))
            else:               # the data description, or a damaged download
                with open(os.path.join(outdir, fname), 'wb') as f:
                    f.write(NOT_A_ZIP)
    return download_from_zenodo


@contextlib.contextmanager
def _stubbed_zenodo(archives: str, calls: list, record=None):
    """
    Replaces the Zenodo download and the record listing for the duration of the
    block, so that the download tests neither reach the network nor depend on
    what Zenodo offers today. ``record`` says which files the record is to hold
    and defaults to the prepared archives plus the data description of each
    release.
    """
    if record is None:
        record = os.listdir(archives) + list(CAMELS_IND._DOC_FILE.values())
    original_download = download_zenodo.download_from_zenodo
    original_record = CAMELS_IND.__dict__['_record_files']
    try:
        download_zenodo.download_from_zenodo = _fake_zenodo(calls, archives)
        CAMELS_IND._record_files = property(lambda self: list(record))
        yield
    finally:
        download_zenodo.download_from_zenodo = original_download
        CAMELS_IND._record_files = original_record


def _fresh(path: str, version: str = '2.2', **kwargs) -> CAMELS_IND:
    """a CAMELS_IND of ``version`` under ``path``, without the netCDF cache"""
    return CAMELS_IND(path=path, version=version, to_netcdf=False,
                      verbosity=0, **kwargs)


def test_registration():
    """the class is registered under its own name"""
    logger.info("test_registration")
    assert 'CAMELS_IND' in aqua_fetch.ALL_DATASETS
    assert aqua_fetch.rr.DATASETS['CAMELS_IND'] is CAMELS_IND
    return


def test_version_argument():
    """the default is release 2.2 and every release has its own Zenodo record"""
    logger.info("test_version_argument")
    assert dataset.version == '2.2'
    assert sorted(CAMELS_IND.urls) == ['2', '2.2']
    assert dataset._latest_version == '2.2' and dataset._is_latest
    assert not dataset_of('2')._is_latest
    assert dataset.url == "https://zenodo.org/records/14999580"
    assert CAMELS_IND.urls['2'] == "https://zenodo.org/records/13221214"

    # numbers and strings are both accepted, and both reach the right record
    for value, record in ((2, '13221214'), ('2', '13221214'),
                          (2.2, '14999580'), ('2.2', '14999580')):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds = CAMELS_IND(path=CAMELS_IND_PATH, version=value, verbosity=0)
        assert ds.version == str(value)
        assert ds.url.endswith(record), (value, ds.url)
        assert os.path.isdir(ds.static_path), ds.static_path

    for wrong in (1, '1', 2.1, '2.20', None, 'latest'):
        try:
            CAMELS_IND(path=CAMELS_IND_PATH, version=wrong, verbosity=0)
        except ValueError:
            continue
        raise AssertionError(f"version={wrong!r} was accepted")
    return


def test_layout():
    """each release keeps its files, and its cache, where the release puts them"""
    logger.info("test_layout")
    assert dataset._version_dir == os.path.join(dataset.path, 'CAMELS_IND_All_Catchments')
    assert dataset.static_path == os.path.join(dataset._version_dir, 'attributes_txt')
    assert dataset.q_path == os.path.join(dataset._version_dir, 'streamflow_timeseries')
    assert dataset.forcings_path == os.path.join(dataset._version_dir,
                                                 'catchment_mean_forcings')
    assert dataset.boundary_file.endswith(os.path.join('shapefiles_catchment',
                                                       'merged', 'all_catchments.shp'))
    assert os.path.dirname(dataset.dyn_fpath) == dataset._version_dir

    for fpath in (dataset.static_path, dataset.q_path, dataset.forcings_path,
                  dataset.boundary_file, dataset._q_file):
        assert os.path.exists(fpath), fpath

    # the forcing files of release 2.2 are in one folder, not one per basin
    assert dataset.stn_forcing_path('12001') == os.path.join(
        dataset.forcings_path, '12001.csv')

    old = dataset_of('2')
    assert old._version_dir == old.path
    assert old.boundary_file.endswith(os.path.join('Merged', 'all_catchments.shp'))
    assert old.stn_forcing_path('12001') == os.path.join(
        old.forcings_path, '12', '12001.csv')
    assert old.dyn_fpath != dataset.dyn_fpath, "the releases share a netCDF cache"
    return


def test_feature_names():
    """the features are named after the files of the release they come from"""
    logger.info("test_feature_names")
    assert len(dataset.stations()) == NUM_STATIONS
    assert len(dataset.static_features) == NUM_STATIC
    assert len(dataset.dynamic_features) == NUM_DYNAMIC

    for name in list(RAW_COLUMNS) + list(UNMAPPED_COLUMNS['2.2']) + ['q_cms_obs']:
        assert name in dataset.dynamic_features, name
    for name in UNMAPPED_COLUMNS['2']:
        assert name not in dataset.dynamic_features, name

    # release 2.2 corrects the unit label of the two evaporation forcings and
    # the spelling of the mean drainage path slope
    assert 'dpsbar' in dataset.static_features
    assert 'dspbar' not in dataset.static_features

    old = dataset_of('2')
    assert len(old.static_features) == NUM_STATIC
    assert len(old.dynamic_features) == NUM_DYNAMIC
    assert 'dspbar' in old.static_features and 'dpsbar' not in old.static_features
    for name in UNMAPPED_COLUMNS['2']:
        assert name in old.dynamic_features, name
    return


def test_read_dynamic_fidelity():
    """the served values are the values of the files, in their own units"""
    logger.info("test_read_dynamic_fidelity")
    for version in ('2.2', '2'):
        ds = dataset_of(version)
        for stn in ('3001', '12001', '17025'):
            served = ds._read_stn_dyn(stn)
            forcings = _raw_forcings(ds, stn)
            q = _raw_streamflow(ds, stn)

            assert served.index.equals(forcings.index)
            for name, raw in RAW_COLUMNS.items():
                assert np.allclose(served[name].values, forcings[raw].values,
                                   equal_nan=True), (version, stn, name)
            for name in UNMAPPED_COLUMNS[version]:
                assert np.allclose(served[name].values, forcings[name].values,
                                   equal_nan=True), (version, stn, name)
            assert np.allclose(served['q_cms_obs'].values, q.values.astype(np.float32),
                               equal_nan=True), (version, stn)
            assert (served.dtypes == np.float32).all(), served.dtypes.unique()
            # soil moisture is served under the raw name as well
            assert np.allclose(served['sm_lvl1(kg/m2)'].values,
                               forcings['sm_lvl1(kg/m2)'].values, equal_nan=True)
    return


def test_read_dynamic_matches_per_station():
    """the bulk reader, which reads the streamflow file once for all gauges,
    gives what reading one gauge at a time gives"""
    logger.info("test_read_dynamic_matches_per_station")
    stations = ['3001', '12001', '15013', '17025']
    bulk = dataset._read_dynamic(stations, 'all')
    assert list(bulk) == stations
    for stn in stations:
        one = dataset._read_stn_dyn(stn)
        assert list(bulk[stn].columns) == list(one.columns), stn
        assert bulk[stn].index.equals(one.index), stn
        assert np.array_equal(bulk[stn].values, one.values, equal_nan=True), stn
        assert bulk[stn].index.name == 'time'
        assert bulk[stn].columns.name == 'dynamic_features'

    # a subset of the features and a period, as fetch() asks for them
    part = dataset._read_dynamic(['3001'], ['pcp_mm', 'q_cms_obs'],
                                 st='2000-01-01', en='2000-12-31')
    assert part['3001'].shape == (366, 2)
    assert list(part['3001'].columns) == ['pcp_mm', 'q_cms_obs']

    # without a process pool the answer is the same
    serial = CAMELS_IND(path=CAMELS_IND_PATH, processes=1, verbosity=0)
    one_by_one = serial._read_dynamic(stations, 'all')
    for stn in stations:
        assert np.array_equal(one_by_one[stn].values, bulk[stn].values,
                              equal_nan=True), stn
    return


def test_start_end_follow_the_files():
    """the period is the one the files hold, for both releases"""
    logger.info("test_start_end_follow_the_files")
    for version in ('2.2', '2'):
        ds = dataset_of(version)
        q = _raw_streamflow(ds, '3001')
        assert ds.start == q.index.min(), version
        assert ds.end == q.index.max(), version
        # the forcing files share that time axis
        for stn in ('3001', '12001', '17025'):
            forcings = _raw_forcings(ds, stn).index
            assert forcings.min() == ds.start and forcings.max() == ds.end, (version, stn)
            assert len(forcings) == DYN_LEN
    return


def test_fetch_matches_the_files():
    """fetch(), which reads the netCDF cache, serves the values of the files"""
    logger.info("test_fetch_matches_the_files")
    stations = ['3001', '12001', '15001']
    _, dyn = dataset.fetch(stations=stations, as_dataframe=True)
    for stn in stations:
        served = dyn[stn]
        assert served.shape == (DYN_LEN, NUM_DYNAMIC)
        assert (served.dtypes == np.float32).all(), served.dtypes.unique()
        forcings = _raw_forcings(dataset, stn)
        assert served.index.equals(forcings.index)
        assert np.allclose(served['pcp_mm'].values, forcings['prcp(mm/day)'].values,
                           equal_nan=True), stn
        assert np.allclose(served['q_cms_obs'].values,
                           _raw_streamflow(dataset, stn).values.astype(np.float32),
                           equal_nan=True), stn
    return


def test_time_index():
    """every day between start and end is in the index, none of them twice"""
    logger.info("test_time_index")
    _, dyn = dataset.fetch(stations='3001', as_dataframe=True)
    index = dyn['3001'].index
    expected = pd.date_range(dataset.start, dataset.end, freq='D')
    assert index.equals(expected), (index[:3], expected[:3])
    assert len(index) == DYN_LEN
    # start and end are what the files hold, not a guess
    raw = _raw_forcings(dataset, '3001').index
    assert raw.min() == dataset.start and raw.max() == dataset.end
    return


def test_units():
    """the values stay in the units of the source files"""
    logger.info("test_units")
    _, dyn = dataset.fetch(stations=['3001', '12001'], as_dataframe=True)
    for stn, df in dyn.items():
        assert 0 <= df['pcp_mm'].max() < 1500, df['pcp_mm'].max()      # mm/day
        assert -10 < df['airtemp_C_min'].min() < 40                    # deg C
        assert df['airtemp_C_max'].mean() > df['airtemp_C_min'].mean()
        assert 0 < df['rh_%'].max() <= 100                             # %
        assert 0 < df['swdownrad_wm2'].mean() < 500                    # W m-2
        assert 0 < df['pet_mm'].max() < 50                             # mm/day
        assert 0 <= df['windspeed_mps'].max() < 30                     # m s-1
        # the two evaporation forcings are mm/day, not kg m-2 s-1
        assert 0 <= df['evap_canopy(mm/day)'].max() < 20, stn
    return


def test_documented_coverage():
    """the streamflow coverage the docstring quotes is what the files hold"""
    logger.info("test_documented_coverage")
    q = dataset._read_q()
    assert q.shape == (DYN_LEN, NUM_STATIONS)
    covered = q.notna().mean()
    assert int((covered > 0).sum()) == 313, int((covered > 0).sum())
    assert int((covered > 0.3).sum()) == 242, int((covered > 0.3).sum())

    # release 2 is the one with fewer well covered gauges
    old_q = dataset_of('2')._read_q()
    assert int((old_q.notna().mean() > 0.3).sum()) == 228
    return


def test_static_fidelity():
    """the static features are the values of the attribute files"""
    logger.info("test_static_fidelity")
    static = dataset.fetch_static_features(stations='all')
    assert static.shape == (NUM_STATIONS, NUM_STATIC)

    topo = _raw_attributes(dataset, 'topo')
    clim = _raw_attributes(dataset, 'clim')
    for stn in ('3001', '12001', '15013'):
        gauge_id = stn.zfill(5)
        assert static.loc[stn, 'area_km2'] == topo.loc[gauge_id, 'cwc_area']
        assert static.loc[stn, 'lat'] == topo.loc[gauge_id, 'cwc_lat']
        assert static.loc[stn, 'long'] == topo.loc[gauge_id, 'cwc_lon']
        assert static.loc[stn, 'slope_%'] == topo.loc[gauge_id, 'slope_mean']
        assert static.loc[stn, 'dpsbar'] == topo.loc[gauge_id, 'dpsbar']
        assert static.loc[stn, 'p_mean'] == clim.loc[gauge_id, 'p_mean']

    # every attribute file contributes its columns, none of them twice
    columns = []
    for name in ATTRIBUTE_FILES:
        columns += _raw_attributes(dataset, name).columns.tolist()
    assert len(columns) == NUM_STATIC and len(set(columns)) == NUM_STATIC
    return


def test_coords_and_area():
    """the coordinates and areas are the published ones and lie within India"""
    logger.info("test_coords_and_area")
    coords = dataset.stn_coords()
    assert coords.shape == (NUM_STATIONS, 2)
    assert coords['lat'].between(8, 26).all(), coords['lat'].describe()
    assert coords['long'].between(68, 88).all(), coords['long'].describe()

    topo = _raw_attributes(dataset, 'topo')
    areas = dataset.area()
    assert len(areas) == NUM_STATIONS
    assert areas.min() > 0
    for stn in ('3001', '12001', '17021'):
        assert areas[stn] == topo.loc[stn.zfill(5), 'cwc_area']
        assert np.isclose(dataset.stn_coords(stn)['lat'].iloc[0],
                          topo.loc[stn.zfill(5), 'cwc_lat'])
    return


def _area_mismatches(ds: CAMELS_IND, tol: float = 0.05):
    """gauges whose boundary polygon is more than ``tol`` away from the area
    the release publishes for them"""
    topo = _raw_attributes(ds, 'topo')
    off = []
    for stn in ds.stations():
        published = topo.loc[stn.zfill(5), 'ghi_area']
        polygon = _polygon_area_km2(ds.get_boundary(stn))
        if abs(polygon - published) / published > tol:
            off.append(stn)
    return off


def test_gauge_metadata_alignment():
    """
    The metadata of every gauge of release 2.2 describes the catchment whose
    boundary is served under the same id, which release 2 does not for the
    gauges of basins 12 and 15 whose metadata it attached to the next gauge.
    """
    logger.info("test_gauge_metadata_alignment")
    if fiona is None:
        logger.info("fiona is not installed, skipping")
        return

    assert _area_mismatches(dataset) == [], \
        "the published area does not describe the boundary served for these gauges"

    old = dataset_of('2')
    off = _area_mismatches(old)
    assert len(off) > 40, \
        f"release 2 agrees with its own boundaries for all but {len(off)} gauges, " \
        f"check the test"
    assert all(stn[:2] in ('12', '15') for stn in off), off

    # the coordinates moved with the areas
    for stn in REKEYED_GAUGES:
        assert stn in off, stn
        assert not np.isclose(dataset.stn_coords(stn)['lat'].iloc[0],
                              old.stn_coords(stn)['lat'].iloc[0]), stn
    return


def test_releases_do_not_mix():
    """the two releases are read from their own files, not from each other's"""
    logger.info("test_releases_do_not_mix")
    old = dataset_of('2')
    assert old.area('12001').iloc[0] != dataset.area('12001').iloc[0]
    assert old._version_dir != dataset._version_dir
    assert os.path.exists(old.dyn_fpath) and os.path.exists(dataset.dyn_fpath)
    # both caches are read, not one of them twice
    _, new_dyn = dataset.fetch(stations='12001', as_dataframe=True)
    _, old_dyn = old.fetch(stations='12001', as_dataframe=True)
    assert not np.array_equal(new_dyn['12001']['q_cms_obs'].values,
                              old_dyn['12001']['q_cms_obs'].values,
                              equal_nan=True)
    return


def test_q_mm():
    """streamflow in mm/day follows from the published area"""
    logger.info("test_q_mm")
    q_mm = dataset.q_mm(['3001'])
    _, dyn = dataset.fetch(stations='3001', as_dataframe=True)
    q_cms = dyn['3001']['q_cms_obs']
    area_m2 = dataset.area('3001').iloc[0] * 1e6
    expected = q_cms * 86400 / area_m2 * 1e3
    assert np.allclose(q_mm['3001'].values, expected.values, equal_nan=True)
    return


def test_boundaries():
    """the boundary of every gauge is found by its id"""
    logger.info("test_boundaries")
    if fiona is None:
        logger.info("fiona is not installed, skipping")
        return
    for version in ('2.2', '2'):
        ds = dataset_of(version)
        # the id is mapped by the class itself, not by a name check in the base
        assert ds.boundary_id_map == 'gauge_id'
        assert ds._boundary_catch_id('03001') == '3001'
        boundaries = ds._create_boundary_id_map()
        assert len(boundaries) == NUM_STATIONS, (version, len(boundaries))
        assert set(boundaries) == set(ds.stations()), version
        geometry = ds.get_boundary('3001')
        assert geometry['type'] in ('Polygon', 'MultiPolygon')

    from aqua_fetch.rr import utils as rr_utils
    source = inspect.getsource(rr_utils._RainfallRunoff._create_boundary_id_map)
    assert 'CAMELS_IND' not in source, \
        "the base class still singles CAMELS_IND out"
    return


def test_returns_copies():
    """what the class returns can be modified without corrupting it"""
    logger.info("test_returns_copies")
    stations = dataset.stations()
    stations.append('junk')
    assert len(dataset.stations()) == NUM_STATIONS

    dynamic = dataset.dynamic_features
    dynamic.clear()
    assert len(dataset.dynamic_features) == NUM_DYNAMIC

    static = dataset.static_features
    static.clear()
    assert len(dataset.static_features) == NUM_STATIC

    table = dataset._static_data()
    table['junk'] = 1
    assert 'junk' not in dataset._static_data().columns
    return


def test_getstate_leaves_the_heavy_tables_behind():
    """the big cached tables must not travel with a pickled dataset, for the
    paths which still pickle one (this class hands its pool a module level
    function instead)"""
    logger.info("test_getstate_leaves_the_heavy_tables_behind")
    import pickle
    if fiona is None:
        logger.info("fiona is not installed, skipping")
        return
    ds = CAMELS_IND(path=CAMELS_IND_PATH, verbosity=0)
    ds._static_data()
    ds.get_boundary('3001')
    for cached in CAMELS_IND._NOT_PICKLED:
        assert cached in ds.__dict__ and len(ds.__dict__[cached]) > 0, cached

    small = len(pickle.dumps(ds))
    original = CAMELS_IND.__dict__['__getstate__']
    try:
        del CAMELS_IND.__getstate__                     # pickle everything
        big = len(pickle.dumps(ds))
    finally:
        CAMELS_IND.__getstate__ = original
    assert CAMELS_IND.__dict__['__getstate__'] is original, "the class was left without it"
    assert small < big / 5, (small, big)
    assert small < 60_000, small

    restored = pickle.loads(pickle.dumps(ds))
    assert restored.stations() == ds.stations()
    assert restored._read_stn_dyn('3001').shape == (DYN_LEN, NUM_DYNAMIC)
    assert len(restored._static_data()) == NUM_STATIONS   # rebuilt in the worker
    return


def test_to_netcdf_and_overwrite_are_honoured():
    """both arguments reach the parent class instead of being swallowed"""
    logger.info("test_to_netcdf_and_overwrite_are_honoured")
    original_download = CAMELS_IND._download_camels_ind
    CAMELS_IND._download_camels_ind = lambda self, overwrite=False: None
    # _maybe_to_netcdf is inherited, so it is removed again rather than restored:
    # assigning it back would put a copy on the subclass and cut the class off
    # from a later change to _RainfallRunoff._maybe_to_netcdf
    assert '_maybe_to_netcdf' not in CAMELS_IND.__dict__
    CAMELS_IND._maybe_to_netcdf = lambda self: None
    try:
        ds = CAMELS_IND(path=CAMELS_IND_PATH, to_netcdf=False, verbosity=0)
        assert ds.to_netcdf is False, "to_netcdf=False was ignored"
        assert ds.overwrite is False
        ds = CAMELS_IND(path=CAMELS_IND_PATH, overwrite=True, verbosity=0)
        assert ds.overwrite is True, "overwrite=True did not reach the parent"
        assert ds.to_netcdf is True
    finally:
        CAMELS_IND._download_camels_ind = original_download
        del CAMELS_IND._maybe_to_netcdf
    return


def test_no_redownload_or_reextract():
    """initializing again neither downloads, extracts nor rebuilds the cache"""
    logger.info("test_no_redownload_or_reextract")

    def boom(*args, **kwargs):
        raise AssertionError("must not be called when the data exists")

    originals = (download_zenodo.download_from_zenodo, zipfile.ZipFile,
                 xr.Dataset.to_netcdf if xr is not None else None)
    download_zenodo.download_from_zenodo = zipfile.ZipFile = boom
    if xr is not None:
        xr.Dataset.to_netcdf = boom
    try:
        again = CAMELS_IND(path=CAMELS_IND_PATH, verbosity=0)
        assert len(again.stations()) == NUM_STATIONS
    finally:
        download_zenodo.download_from_zenodo, zipfile.ZipFile = originals[:2]
        if xr is not None:
            xr.Dataset.to_netcdf = originals[2]
    return


def test_download():
    """
    Only the archive of the release is downloaded, into a folder of its own
    name, and once extracted nothing is downloaded or extracted again.
    """
    logger.info("test_download")
    mini = _mini_copy()
    archives = _archive_of(mini)
    tmp = tempfile.mkdtemp()
    calls = []
    try:
        with _stubbed_zenodo(archives, calls):
            with warnings.catch_warnings(record=True) as raised:
                warnings.simplefilter("always")
                ds = _fresh(tmp)

            assert len(calls) == 1, calls
            doi, outdir, include = calls[0]
            assert doi == CAMELS_IND.urls['2.2']
            assert outdir == ds.path
            # the data description of release 2.2 is inside its archive, and the
            # 242 gauge subset archive is a copy of what the archive already holds
            assert include == ('CAMELS_IND_All_Catchments.zip',), include

            assert os.path.isdir(ds._version_dir)
            assert not [f for f in os.listdir(ds.path) if f.endswith('_extracting')]
            assert sorted(ds.stations()) == ['12001', '3001']
            assert len(ds.fetch(stations='3001', as_dataframe=True)[1]['3001']) == DYN_LEN

            # the simulated streamflow is model output and is not extracted
            assert os.listdir(ds.q_path) == ['streamflow_observed.csv']
            assert any('model output' in str(w.message) for w in raised), \
                [str(w.message) for w in raised]

            calls.clear()
            original_zipfile = zipfile.ZipFile
            zipfile.ZipFile = lambda *a, **k: (_ for _ in ()).throw(
                AssertionError("extracted again"))
            try:
                again = _fresh(tmp)
                assert calls == [], f"downloaded again although extracted: {calls}"
                assert len(again.stations()) == 2
            finally:
                zipfile.ZipFile = original_zipfile

            # remove_zip deletes the archive of this release, which does not bring
            # the download back
            again = _fresh(tmp, remove_zip=True)
            assert not [f for f in os.listdir(again.path) if f.endswith('.zip')]
            calls.clear()
            _fresh(tmp)
            assert calls == [], "the archive was downloaded again after remove_zip"
    finally:
        for folder in (mini, archives, tmp):
            shutil.rmtree(folder, ignore_errors=True)
    return


def test_interrupted_extraction():
    """an extraction that stops half way leaves no folder which looks complete,
    so the next initialization extracts the archive again without downloading"""
    logger.info("test_interrupted_extraction")
    mini = _mini_copy()
    archives = _archive_of(mini)
    tmp = tempfile.mkdtemp()
    calls = []
    original_extractall = zipfile.ZipFile.extractall
    try:
        with _stubbed_zenodo(archives, calls):

            def die(self, *args, **kwargs):
                original_extractall(self, *args, **kwargs)
                raise KeyboardInterrupt("interrupted")

            zipfile.ZipFile.extractall = die
            try:
                _fresh(tmp)
            except KeyboardInterrupt:
                pass
            finally:
                zipfile.ZipFile.extractall = original_extractall

            path = os.path.join(tmp, 'CAMELS_IND')
            assert not os.path.exists(os.path.join(path, 'CAMELS_IND_All_Catchments')), \
                "the half extracted folder looks complete"
            assert os.path.exists(os.path.join(path, 'CAMELS_IND_All_Catchments.zip'))

            ds = _fresh(tmp)
            assert len(calls) == 1, "the archive on disk was downloaded again"
            assert sorted(ds.stations()) == ['12001', '3001']
    finally:
        zipfile.ZipFile.extractall = original_extractall
        for folder in (mini, archives, tmp):
            shutil.rmtree(folder, ignore_errors=True)
    return


def test_corrupt_archive_is_set_aside():
    """an archive which is not a zip file is moved aside, not deleted: the
    bytes of a release Zenodo no longer serves are all the user has"""
    logger.info("test_corrupt_archive_is_set_aside")
    tmp = tempfile.mkdtemp()
    calls = []
    try:
        # the stub writes an empty file for anything it does not have
        with _stubbed_zenodo(tmp, calls, record=['CAMELS_IND_All_Catchments.zip']):
            try:
                _fresh(tmp)
            except ValueError as e:
                assert 'not a readable zip file' in str(e), e
            else:
                raise AssertionError("a corrupt archive was accepted")
        archive = os.path.join(tmp, 'CAMELS_IND', 'CAMELS_IND_All_Catchments.zip')
        assert not os.path.exists(archive), "the unreadable archive was left in place"
        assert open(f"{archive}.corrupt", 'rb').read() == NOT_A_ZIP, \
            "the bytes of the unreadable archive were not kept"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_overwrite_removes_stale_before_download():
    """overwrite=True deletes the archive, the extracted folder and the cache of
    this release before downloading it again, and leaves the other release alone"""
    logger.info("test_overwrite_removes_stale_before_download")
    mini = _mini_copy()
    archives = _archive_of(mini)
    tmp = tempfile.mkdtemp()
    calls = []
    try:
        with _stubbed_zenodo(archives, calls):
            ds = _fresh(tmp)

            stale = os.path.join(ds._version_dir, 'stale.txt')
            open(stale, 'w').close()
            cache = os.path.join(ds._version_dir, ds.dyn_fname)
            open(cache, 'w').close()
            other = os.path.join(ds.path, ds.dyn_fname)    # the cache of release 2
            open(other, 'w').close()
            os.makedirs(os.path.join(ds.path, 'attributes_txt'), exist_ok=True)

            calls.clear()
            _fresh(tmp, overwrite=True)

            assert len(calls) == 1, calls
            assert not os.path.exists(stale), "the stale extracted folder survived"
            assert not os.path.exists(cache), "the stale cache survived"
            assert os.path.exists(other), "the cache of release 2 was deleted"
            assert os.path.isdir(os.path.join(ds.path, 'attributes_txt')), \
                "the files of release 2 were deleted"
            assert not [f for f in os.listdir(ds.path) if f.endswith('.zip1')], \
                os.listdir(ds.path)
    finally:
        for folder in (mini, archives, tmp):
            shutil.rmtree(folder, ignore_errors=True)
    return


def test_remove_zip_files_keeps_the_other_release():
    """remove_zip deletes the archives of this release, not the other's"""
    logger.info("test_remove_zip_files_keeps_the_other_release")
    mini = _mini_copy()
    archives = _archive_of(mini)
    tmp = tempfile.mkdtemp()
    try:
        with _stubbed_zenodo(archives, []):
            ds = _fresh(tmp)
            other = os.path.join(ds.path, 'attributes_txt.zip')    # of release 2
            open(other, 'w').close()

            ds.remove_zip_files()
            assert not os.path.exists(os.path.join(ds.path,
                                                   'CAMELS_IND_All_Catchments.zip'))
            assert os.path.exists(other), "the archive of release 2 was deleted"
    finally:
        for folder in (mini, archives, tmp):
            shutil.rmtree(folder, ignore_errors=True)
    return


def test_a_restricted_release_cannot_be_downloaded():
    """
    Zenodo lists no file for the releases superseded by 2.2, so asking for one
    without its files on disk says so instead of failing inside the downloader.
    """
    logger.info("test_a_restricted_release_cannot_be_downloaded")
    tmp = tempfile.mkdtemp()
    calls = []
    try:
        with _stubbed_zenodo(tmp, calls, record=[]):      # a restricted record
            try:
                _fresh(tmp, version='2')
            except FileNotFoundError as e:
                assert 'cannot be downloaded' in str(e), e
                assert 'attributes_txt.zip' in str(e), e
            else:
                raise AssertionError("a restricted release was downloaded")
        assert calls == [], "the downloader was called for a restricted record"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_overwrite_keeps_what_it_cannot_download_again():
    """
    overwrite=True asks Zenodo before deleting: with neither a record that
    serves the files nor an archive on disk, the extracted copy is the only one
    left and must survive.
    """
    logger.info("test_overwrite_keeps_what_it_cannot_download_again")
    mini = _mini_copy()
    archives = _archive_of(mini)
    tmp = tempfile.mkdtemp()
    calls = []
    try:
        with _stubbed_zenodo(archives, calls):
            ds = _fresh(tmp, remove_zip=True)     # nothing to re-extract from
        assert not [f for f in os.listdir(ds.path) if f.endswith('.zip')]
        cache = os.path.join(ds._version_dir, ds.dyn_fname)
        open(cache, 'w').close()

        calls.clear()
        with _stubbed_zenodo(archives, calls, record=[]):   # restricted now
            try:
                _fresh(tmp, overwrite=True)
            except FileNotFoundError as e:
                assert 'Nothing was deleted' in str(e), e
            else:
                raise AssertionError("overwrite went ahead without a source")

        assert calls == [], calls
        assert os.path.isdir(ds._version_dir), "the only copy was deleted"
        assert os.path.exists(cache), "the cache was deleted"
        assert sorted(_fresh(tmp).stations()) == ['12001', '3001'], \
            "the release is no longer readable"
    finally:
        for folder in (mini, archives, tmp):
            shutil.rmtree(folder, ignore_errors=True)
    return


def test_remove_zip_keeps_a_superseded_release():
    """
    remove_zip=True must not delete the archives of a release which is not the
    newest: unlike release 2.2's, they cannot be fetched again.
    """
    logger.info("test_remove_zip_keeps_a_superseded_release")
    mini = _mini_copy('2')
    try:
        archive = os.path.join(mini, 'CAMELS_IND', 'attributes_txt.zip')
        open(archive, 'w').close()

        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter("always")
            ds = _fresh(mini, version='2', remove_zip=True)

        assert ds.remove_zip is True
        assert ds._is_latest is False and dataset._is_latest is True
        assert os.path.exists(archive), "an archive Zenodo cannot serve again was deleted"
        assert any('are kept although remove_zip is True' in str(w.message)
                   for w in raised), [str(w.message) for w in raised]
    finally:
        shutil.rmtree(mini, ignore_errors=True)
    return


def test_a_missing_description_does_not_block_extraction():
    """
    The data description is not read by the class, so its absence must not stop
    an extraction that needs no download - which is what it did for release 2,
    whose record offers nothing at all.
    """
    logger.info("test_a_missing_description_does_not_block_extraction")
    mini = _mini_copy('2')
    archives = _archive_of(mini, '2')
    calls = []
    try:
        root = os.path.join(mini, 'CAMELS_IND')
        # one folder gone, its archive lying next to the data, no description
        shutil.rmtree(os.path.join(root, 'attributes_txt'))
        shutil.copy(os.path.join(archives, 'attributes_txt.zip'), root)
        assert not os.path.exists(os.path.join(root, CAMELS_IND._DOC_FILE['2']))

        with _stubbed_zenodo(archives, calls, record=[]):   # a restricted record
            ds = _fresh(mini, version='2')

        assert calls == [], "the description was fetched although nothing else was"
        assert sorted(os.listdir(ds.static_path)) == \
            [f"camels_India_{name}.txt" for name in sorted(ATTRIBUTE_FILES)], \
            "the archive on disk was not extracted"
        assert sorted(ds.stations()) == ['12001', '3001']
    finally:
        for folder in (mini, archives):
            shutil.rmtree(folder, ignore_errors=True)
    return


def test_overwrite_re_extracts_what_it_cannot_download():
    """
    overwrite=True on a release Zenodo no longer serves must repair it from the
    archives on disk instead of refusing: they are the only copy there is, so
    they are kept and their folders extracted from them again.
    """
    logger.info("test_overwrite_re_extracts_what_it_cannot_download")
    mini = _mini_copy('2')
    archives = _archive_of(mini, '2')
    calls = []
    try:
        root = os.path.join(mini, 'CAMELS_IND')
        for fname in os.listdir(archives):
            shutil.copy(os.path.join(archives, fname), root)

        # an extraction that stopped half way leaves a folder which looks complete
        os.remove(os.path.join(root, 'attributes_txt', 'camels_India_topo.txt'))

        with _stubbed_zenodo(archives, calls, record=[]):    # a restricted record
            with warnings.catch_warnings(record=True) as raised:
                warnings.simplefilter("always")
                ds = _fresh(mini, version='2', overwrite=True)

        assert calls == [], "a restricted record was asked for a download"
        assert any('are kept although overwrite is True' in str(w.message)
                   for w in raised), [str(w.message) for w in raised]
        assert os.path.exists(os.path.join(root, 'attributes_txt.zip')), \
            "the only copy of the archive was deleted"
        assert os.path.exists(ds._attr_file('topo')), "the folder was not repaired"
        assert sorted(ds.stations()) == ['12001', '3001']
    finally:
        for folder in (mini, archives):
            shutil.rmtree(folder, ignore_errors=True)
    return


def test_the_newest_release_does_not_depend_on_how_urls_is_written():
    """
    _is_latest decides whether an archive may be deleted, so it must follow the
    release numbers and not the order the ``urls`` dict happens to be written in.
    """
    logger.info("test_the_newest_release_does_not_depend_on_how_urls_is_written")
    original = CAMELS_IND.urls
    try:
        for order in (['2', '2.2'], ['2.2', '2'], ['2', '2.2', '2.1']):
            CAMELS_IND.urls = {v: original.get(v, 'https://zenodo.org/records/0')
                               for v in order}
            ds = copy.copy(dataset)
            assert ds._latest_version == '2.2', (order, ds._latest_version)
            assert ds._is_latest is True, order
            ds.version = '2'
            assert ds._is_latest is False, order
    finally:
        CAMELS_IND.urls = original
    return


def test_a_bad_archive_does_not_take_the_release_with_it():
    """
    overwrite=True on a release Zenodo cannot serve replaces each folder only
    once its own archive has been read, so one unreadable archive leaves the
    rest of the release where it was instead of deleting it all.
    """
    logger.info("test_a_bad_archive_does_not_take_the_release_with_it")
    mini = _mini_copy('2')
    archives = _archive_of(mini, '2')
    calls = []
    try:
        root = os.path.join(mini, 'CAMELS_IND')
        for fname in os.listdir(archives):
            shutil.copy(os.path.join(archives, fname), root)
        # one of the archives is a truncated download
        damaged = os.path.join(root, 'catchment_mean_forcings.zip')
        with open(damaged, 'wb') as f:
            f.write(NOT_A_ZIP)

        before = sorted(os.listdir(root))
        with _stubbed_zenodo(archives, calls, record=[]):
            try:
                _fresh(mini, version='2', overwrite=True)
            except ValueError as e:
                assert 'not a readable zip file' in str(e), e
            else:
                raise AssertionError("an unreadable archive was accepted")

        assert calls == []
        assert sorted(os.listdir(root)) == before, \
            "overwrite deleted files although it could not replace them"
        assert os.path.exists(damaged), "the archive the user has to replace was moved"
        # and the release is still readable, with its data intact
        ds = _fresh(mini, version='2')
        assert sorted(ds.stations()) == ['12001', '3001']
        assert ds._read_stn_dyn('3001').shape == (DYN_LEN, NUM_DYNAMIC)
    finally:
        for folder in (mini, archives):
            shutil.rmtree(folder, ignore_errors=True)
    return


def test_a_folder_that_cannot_be_deleted_is_not_half_deleted():
    """
    The folder being replaced is swapped out by rename, not deleted in place,
    so a filesystem which refuses a delete cannot leave the release without its
    data while the finished replacement waits unused beside it.
    """
    logger.info("test_a_folder_that_cannot_be_deleted_is_not_half_deleted")
    mini = _mini_copy('2')
    archives = _archive_of(mini, '2')
    calls = []
    root = os.path.join(mini, 'CAMELS_IND')
    try:
        for fname in os.listdir(archives):
            shutil.copy(os.path.join(archives, fname), root)

        # one entry of the folder cannot be unlinked, so a delete in place
        # empties the folder around it and then cannot be renamed over
        locked = os.path.join(root, 'attributes_txt', 'locked')
        os.makedirs(locked)
        open(os.path.join(locked, 'held'), 'w').close()
        os.chmod(locked, 0o500)

        with _stubbed_zenodo(archives, calls, record=[]):
            ds = _fresh(mini, version='2', overwrite=True)

        for name in ATTRIBUTE_FILES:
            assert os.path.exists(ds._attr_file(name)), name
        assert sorted(ds.stations()) == ['12001', '3001']
        assert ds._read_stn_dyn('3001').shape == (DYN_LEN, NUM_DYNAMIC)
    finally:
        for dirpath, dirnames, _ in os.walk(mini):
            for name in dirnames:
                try:
                    os.chmod(os.path.join(dirpath, name), 0o700)
                except OSError:
                    pass
        for folder in (mini, archives):
            shutil.rmtree(folder, ignore_errors=True)
    return


def test_missing_files_warn():
    """a file deleted by hand is reported instead of being served as a smaller
    dataset"""
    logger.info("test_missing_files_warn")
    mini = _mini_copy()
    try:
        ds = _fresh(mini)
        os.remove(ds.stn_forcing_path('12001'))
        os.remove(os.path.join(ds.static_path, 'camels_ind_clim.txt'))

        with warnings.catch_warnings(record=True) as raised:
            warnings.simplefilter("always")
            _fresh(mini)
        messages = [str(w.message) for w in raised]
        assert any('2 files are missing' in m for m in messages), messages

        # the name and topography files are what the class needs to know its
        # gauges at all, so their absence stops it instead of warning
        for fname in ('camels_ind_topo.txt', 'camels_ind_name.txt'):
            os.remove(os.path.join(ds.static_path, fname))
            try:
                _fresh(mini)
            except FileNotFoundError:
                continue
            raise AssertionError(f"a missing {fname} went unnoticed")
    finally:
        shutil.rmtree(mini, ignore_errors=True)
    return


def test_station_argument_types():
    """the documented ways of naming stations work"""
    logger.info("test_station_argument_types")
    _, one = dataset.fetch(stations='3001', as_dataframe=True)
    assert list(one) == ['3001']
    _, two = dataset.fetch(stations=['3001', '12001'], as_dataframe=True)
    assert sorted(two) == ['12001', '3001']
    _, some = dataset.fetch(stations=3, as_dataframe=True, seed=13)
    assert len(some) == 3
    _, fraction = dataset.fetch(0.01, as_dataframe=True, seed=13)
    assert len(fraction) == int(NUM_STATIONS * 0.01)
    _, again = dataset.fetch(stations=3, as_dataframe=True, seed=13)
    assert list(again) == list(some), "the seed does not fix the sample"
    return


def test_efficiency():
    """fetching every station stays quick"""
    logger.info("test_efficiency")
    start = time.time()
    _, dyn = dataset.fetch(as_dataframe=True)
    took = time.time() - start
    assert len(dyn) == NUM_STATIONS
    logger.info(f"fetched {NUM_STATIONS} stations in {took:.1f} s")
    assert took < 5, f"fetching all stations took {took:.1f} s"

    start = time.time()
    dataset.stn_coords()
    dataset.area()
    dataset.stations()
    assert time.time() - start < 1, "the documented one liners are slow"

    start = time.time()
    dyn = dataset._read_dynamic(dataset.stations(), 'all')
    assert len(dyn) == NUM_STATIONS
    logger.info(f"read {NUM_STATIONS} stations from the source files in "
                f"{time.time() - start:.1f} s")
    return


def test_the_streamflow_file_is_read_once_per_call():
    """
    Every gauge's streamflow is in one 19 MB file. Reading it once per station
    instead of once per call cost 4.4 s for all 472 gauges here and 53 s with
    processes=1 - but the healthy time on a 2 cpu runner is 4 s too, so this
    counts the reads instead of timing them.
    """
    logger.info("test_the_streamflow_file_is_read_once_per_call")
    # through a process pool the reads happen in the workers, where this counter
    # cannot see them, so the check has to run without one
    serial = CAMELS_IND(path=CAMELS_IND_PATH, processes=1, verbosity=0)
    serial.dynamic_features, serial.start       # not what is being counted

    calls, original = [], CAMELS_IND._read_q

    def counted(self, stations=None):
        calls.append(stations)
        return original(self, stations)

    CAMELS_IND._read_q = counted
    try:
        dyn = serial._read_dynamic(serial.stations()[:20], 'all')
    finally:
        CAMELS_IND._read_q = original

    assert len(dyn) == 20
    assert len(calls) == 1, \
        f"the streamflow file was read {len(calls)} times for 20 stations, not once"
    return


def test_shared_suite():
    """the checks every rainfall-runoff dataset has to pass"""
    logger.info("test_shared_suite")
    run_shared_tests(dataset, NUM_STATIONS, DYN_LEN, NUM_STATIC, NUM_DYNAMIC)
    return


def test_shared_suite_2():
    """the same checks for release 2"""
    logger.info("test_shared_suite_2")
    run_shared_tests(dataset_of('2'), NUM_STATIONS, DYN_LEN, NUM_STATIC, NUM_DYNAMIC)
    return


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith('test_') and callable(test):
            print(f"{name} ...", flush=True)
            start = time.time()
            test()
            print(f"{name} passed in {time.time() - start:.1f} s", flush=True)
    print("all tests passed")
