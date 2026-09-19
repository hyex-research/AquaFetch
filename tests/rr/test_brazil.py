"""
Tests for the two Brazilian datasets of ``aqua_fetch/rr/_brazil.py``: CAMELS_BR
(897 catchments, mainly release 1.2) and CABra (735 catchments).

They check that the classes

    * serve the raw values, units and dates unchanged (compared with an
      independent read of the raw files),
    * read the static attributes, coordinates, areas and boundaries faithfully,
    * download once, neither download nor extract again, and on
      ``overwrite=True`` delete the stale files of that release first,
    * keep CAMELS_BR release 1.1 readable through ``version='1.1'``,
    * extend the released streamflow with the ANA download without ever
      replacing a released value,
    * fetch all stations quickly.

Everything up to the CABra section is about CAMELS_BR; ``test_cabra`` runs the
shared suite on CABra once per meteorological source.

The first run downloads CAMELS_BR release 1.2 (about 1.0 GB) into
``CAMELS_BR_PATH``. Run as a script or with pytest.
"""

import os
import copy
import json
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
    logging.basicConfig(filename='test_brazil.log', filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

import aqua_fetch
from aqua_fetch import CAMELS_BR, CABra
from aqua_fetch.rr import _brazil
from aqua_fetch._backend import xarray as xr, fiona

from utils import test_dataset as run_shared_tests

# The classes append ``CAMELS_BR`` (both releases can live there) and ``CABra``
# to these paths. Replace with the locations on your machine.
CAMELS_BR_PATH = '/mnt/hyexnas01/abbaa0a/data/gscad_database/raw/CAMELS/CAMELS'
CABRA_PATH = '/mnt/hyexnas01/abbaa0a/data/gscad_database/raw'

NUM_STATIONS = 897
NUM_STATIC = 66
NUM_DYNAMIC = 26
DYN_LEN = 16437          # daily steps 1980-01-01 .. 2024-12-31
NUM_BRDWGD = 864         # catchments with BR-DWGD products, the rest are NaN
NUM_GAUGES_OUTSIDE_BOUNDARY = 22   # gauges the release places outside their own
                                   # catchment, the farthest by 0.089 deg (~10 km)

# release 1.1, for the tests which check that it is still readable
NUM_STATIC_11 = 67
NUM_DYNAMIC_11 = 11
DYN_LEN_11 = 14245       # daily steps 1980-01-01 .. 2018-12-31

# standardized feature -> (folder, file name suffix, raw column). Written out
# here rather than taken from the class, so that a wrong mapping is caught.
RAW_COLUMNS = {
    'q_cms_obs': ('03_CAMELS_BR_streamflow_selected_catchments', '_streamflow.txt', 'streamflow_m3s'),
    'q_mm_obs': ('03_CAMELS_BR_streamflow_selected_catchments', '_streamflow.txt', 'streamflow_mm'),
    'pcp_mm_cpc': ('05_CAMELS_BR_precipitation', '_precipitation.txt', 'p_cpc'),
    'pcp_mm_mswep': ('05_CAMELS_BR_precipitation', '_precipitation.txt', 'p_mswep'),
    'pcp_mm_chirps': ('05_CAMELS_BR_precipitation', '_precipitation.txt', 'p_chirps'),
    'pcp_mm_era5land': ('05_CAMELS_BR_precipitation', '_precipitation.txt', 'p_era5land'),
    'pcp_mm_brdwgd': ('05_CAMELS_BR_precipitation', '_precipitation.txt', 'p_brdwgd'),
    'aet_mm_gleam': ('06_CAMELS_BR_actual_evapotransp', '_actual_evapotransp.txt', 'aet_gleam'),
    'aet_mm_mgb': ('06_CAMELS_BR_actual_evapotransp', '_actual_evapotransp.txt', 'aet_mgb'),
    'aet_mm_era5land': ('06_CAMELS_BR_actual_evapotransp', '_actual_evapotransp.txt', 'aet_era5land'),
    'pet_mm_gleam': ('07_CAMELS_BR_potential_evapotransp', '_potential_evapotransp.txt', 'pet_gleam'),
    'pet_mm_era5land': ('07_CAMELS_BR_potential_evapotransp', '_potential_evapotransp.txt', 'pet_era5land'),
    'eto_mm_brdwgd': ('08_CAMELS_BR_reference_evapotransp', '_reference_evapotransp.txt', 'eto_brdwgd'),
    'airtemp_C_cpc_min': ('09_CAMELS_BR_temperature', '_temperature.txt', 'tmin_cpc'),
    'airtemp_C_cpc_max': ('09_CAMELS_BR_temperature', '_temperature.txt', 'tmax_cpc'),
    'airtemp_C_era5land_min': ('09_CAMELS_BR_temperature', '_temperature.txt', 'tmin_era5land'),
    'airtemp_C_mean_era5land': ('09_CAMELS_BR_temperature', '_temperature.txt', 'tmean_era5land'),
    'airtemp_C_era5land_max': ('09_CAMELS_BR_temperature', '_temperature.txt', 'tmax_era5land'),
    'airtemp_C_brdwgd_min': ('09_CAMELS_BR_temperature', '_temperature.txt', 'tmin_brdwgd'),
    'airtemp_C_brdwgd_max': ('09_CAMELS_BR_temperature', '_temperature.txt', 'tmax_brdwgd'),
    'sm_surface_gleam': ('10_CAMELS_BR_soil_moisture', '_soil_moisture.txt', 'sm_surface_gleam'),
    'sm_rootzone_gleam': ('10_CAMELS_BR_soil_moisture', '_soil_moisture.txt', 'sm_rootzone_gleam'),
    'sml1_era5land': ('10_CAMELS_BR_soil_moisture', '_soil_moisture.txt', 'sm_layer1_era5land'),
    'sml2_era5land': ('10_CAMELS_BR_soil_moisture', '_soil_moisture.txt', 'sm_layer2_era5land'),
    'sml3_era5land': ('10_CAMELS_BR_soil_moisture', '_soil_moisture.txt', 'sm_layer3_era5land'),
    'sml4_era5land': ('10_CAMELS_BR_soil_moisture', '_soil_moisture.txt', 'sm_layer4_era5land'),
}

ATTRIBUTE_FILES = ('camels_br_climate.txt', 'camels_br_geology.txt',
                   'camels_br_human_intervention.txt', 'camels_br_hydrology.txt',
                   'camels_br_land_cover.txt', 'camels_br_location.txt',
                   'camels_br_quality_check.txt', 'camels_br_soil.txt',
                   'camels_br_topography.txt')

# the download of release 1.2 happens here, once
dataset = CAMELS_BR(path=CAMELS_BR_PATH, use_ana_update=False, verbosity=0)


def release_dir(ds: CAMELS_BR) -> str:
    """folder which holds the files of ``ds``, worked out from the layout on
    disk rather than taken from the class: each release lives in a folder of its
    own name, except a release 1.1 which an earlier version of the class had
    extracted directly into ``path``"""
    legacy = os.path.join(ds.path, '03_CAMELS_BR_streamflow_mm_selected_catchments')
    if ds.version == '1.1' and os.path.isdir(legacy):
        return ds.path
    return os.path.join(ds.path, ds.version)


RELEASE_DIR = release_dir(dataset)


def _raw_frame(path: str, station: str, folder: str, suffix: str) -> pd.DataFrame:
    """one raw time series file, read independently of the class"""
    fpath = os.path.join(path, folder, folder, f"{station}{suffix}")
    df = pd.read_csv(fpath, sep=' ')
    df.index = pd.to_datetime(df[['year', 'month', 'day']])
    return df.drop(columns=['year', 'month', 'day'])


def _mini_copy(stations=('10500000', '11400000'), version='1.2', into=None) -> str:
    """
    A temporary copy of ``version`` which holds only ``stations``: the attribute
    and boundary files plus the time series files of those gauges. The class is
    fully usable on it, so the tests which write into the dataset folder do not
    touch the real copy and stay fast.
    """
    tmp = into or tempfile.mkdtemp()
    src = release_dir(dataset_of(version))
    dst = os.path.join(tmp, 'CAMELS_BR', version)

    # every archive of the release must be there, or __init__ downloads it
    for fname in CAMELS_BR.urls[version]:
        stem = fname[:-len('.zip')]
        os.makedirs(os.path.join(dst, stem, stem), exist_ok=True)

    groups = dict(CAMELS_BR._DYN_GROUPS[version], **CAMELS_BR._EXTRA_GROUPS[version])
    for folder, suffix, _ in groups.values():
        for stn in stations:
            fpath = os.path.join(src, folder, folder, f"{stn}{suffix}")
            if os.path.exists(fpath):
                shutil.copy(fpath, os.path.join(dst, folder, folder, f"{stn}{suffix}"))

    attributes = '01_CAMELS_BR_attributes'
    for fname in ATTRIBUTE_FILES:
        shutil.copy(os.path.join(src, attributes, attributes, fname),
                    os.path.join(dst, attributes, attributes, fname))

    boundary = os.path.dirname(dataset_of(version).boundary_file)
    target = os.path.join(dst, os.path.relpath(boundary, src))
    os.makedirs(target, exist_ok=True)
    for fname in os.listdir(boundary):
        shutil.copy(os.path.join(boundary, fname), os.path.join(target, fname))
    return tmp


_datasets = {'1.2': dataset}


def dataset_of(version: str) -> CAMELS_BR:
    """the dataset of ``version``, built once"""
    if version not in _datasets:
        _datasets[version] = CAMELS_BR(path=CAMELS_BR_PATH, version=version,
                                       use_ana_update=False, verbosity=0)
    return _datasets[version]


def test_registration():
    """the class is registered and can be built by the RainfallRunoff factory"""
    logger.info("test_registration")
    assert 'CAMELS_BR' in aqua_fetch.ALL_DATASETS
    from aqua_fetch import RainfallRunoff
    # remove_zip defaults to True in RainfallRunoff and would delete the archives
    rr = RainfallRunoff('CAMELS_BR', path=CAMELS_BR_PATH, remove_zip=False,
                        use_ana_update=False, verbosity=0)
    assert len(rr.stations()) == NUM_STATIONS
    return


def test_version_argument():
    """1.2 is the default, each release has its own caches and an unknown
    version is refused before anything is created on disk"""
    logger.info("test_version_argument")
    assert dataset.version == '1.2'
    assert dataset.dyn_fname == 'camels_br_D_1.2_v2.nc', dataset.dyn_fname
    assert dataset.url.endswith('15025488'), dataset.url

    # a float names the same release, and the caches of the two releases differ
    as_float = CAMELS_BR(path=CAMELS_BR_PATH, version=1.2, use_ana_update=False, verbosity=0)
    assert as_float.dyn_fpath == dataset.dyn_fpath
    assert as_float._static_fpath == dataset._static_fpath
    assert dataset_of('1.1').dyn_fname == 'camels_br_D_1.1_v2.nc', dataset_of('1.1').dyn_fname
    assert dataset_of('1.1')._static_fpath != dataset._static_fpath

    tmp = tempfile.mkdtemp()
    try:
        for bad in ('1.0', 1.0, None, 'latest', 12):
            try:
                CAMELS_BR(path=tmp, version=bad, verbosity=0)
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
    assert dataset.dynamic_features == list(RAW_COLUMNS), dataset.dynamic_features
    assert len(dataset.dynamic_features) == NUM_DYNAMIC

    static = dataset.static_features
    assert len(static) == len(set(static)) == NUM_STATIC, len(static)
    for name in ('area_km2', 'lat', 'long', 'slope_degrees'):
        assert name in static, name
    for raw in ('area', 'gauge_lat', 'gauge_lon', 'slope_mean'):
        assert raw not in static, raw
    # the quality columns are not dynamic features
    for flag in ('qual_flag', 'qual_control_by_ana'):
        assert flag not in dataset.dynamic_features, flag

    assert dataset_of('1.1').dynamic_features[-1] == 'q_cms_obs'
    assert len(dataset_of('1.1').dynamic_features) == NUM_DYNAMIC_11
    assert len(dataset_of('1.1').static_features) == NUM_STATIC_11
    return


def test_read_dynamic_fidelity():
    """every served value is the raw value of its file, on the union of the
    dates of the files of that gauge"""
    logger.info("test_read_dynamic_fidelity")
    stations = random.sample(dataset.stations(), 3)
    _, dyn = dataset.fetch(stations=stations, as_dataframe=True)

    for stn in stations:
        served = dyn[stn]
        assert list(served.columns) == list(RAW_COLUMNS), stn
        assert served.index.name == 'time' and served.columns.name == 'dynamic_features'

        for feature, (folder, suffix, column) in RAW_COLUMNS.items():
            fpath = os.path.join(RELEASE_DIR, folder, folder, f"{stn}{suffix}")
            if not os.path.exists(fpath):   # BR-DWGD reference ET, 864 of 897
                assert served[feature].isna().all(), f"{stn}:{feature} is not empty"
                continue
            raw = _raw_frame(RELEASE_DIR, stn, folder, suffix)[column]
            common = served.index.intersection(raw.index)
            assert len(common) == len(raw), f"{stn}:{feature} lost dates"
            both = raw.notna() & served.loc[common, feature].notna()
            err = np.abs(served.loc[common, feature][both] - raw[both]) / np.abs(raw[both]).clip(lower=1e-9)
            assert (err < 1e-6).all(), f"{stn}:{feature} changed, max {err.max()}"
            assert served.loc[common, feature].isna().equals(raw.isna()), \
                f"{stn}:{feature} gained or lost missing values"
            # dates outside this file's record are NaN
            outside = ~served.index.isin(raw.index)
            assert served.loc[outside, feature].isna().all(), f"{stn}:{feature} filled outside its file"
    return


def test_time_index():
    """the index is the union of the dates of the files of the gauge, daily and
    without holes, and is not built from the first and the last row"""
    logger.info("test_time_index")
    stn = random.choice(dataset.stations())
    _, dyn = dataset.fetch(stations=stn, as_dataframe=True)
    index = dyn[stn].index

    expected = None
    for folder, suffix, _ in CAMELS_BR._DYN_GROUPS['1.2'].values():
        fpath = os.path.join(RELEASE_DIR, folder, folder, f"{stn}{suffix}")
        if not os.path.exists(fpath):
            continue
        dates = _raw_frame(RELEASE_DIR, stn, folder, suffix).index
        expected = dates if expected is None else expected.union(dates)

    assert index.equals(expected), (index[[0, -1]], expected[[0, -1]])
    assert index.is_monotonic_increasing and not index.has_duplicates
    assert (index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()
    assert len(index) == DYN_LEN, len(index)

    assert dataset.start == pd.Timestamp('1980-01-01'), dataset.start
    assert dataset.end == pd.Timestamp('2024-12-31'), dataset.end
    return


def test_units():
    """the served values are in the units of the source"""
    logger.info("test_units")
    stations = random.sample(dataset.stations(), 5)
    _, dyn = dataset.fetch(stations=stations, as_dataframe=True)
    areas = dataset.area(stations)

    # wide enough for the artifacts which the sources themselves contain and
    # which are served unchanged (single days of -42 C in the CPC minimum
    # temperature of both releases, 179 slightly negative GLEAM and 33 negative
    # ERA5-Land potential evapotranspiration values), narrow enough to catch a
    # wrong unit
    # measured over all 897 gauges: pcp_cpc 0..395, pcp_era5land 0..278,
    # aet_gleam 0..15, pet_gleam -0.1..12, pet_era5land -7.4..73.5,
    # eto_brdwgd 0.5..9, tmax_cpc -42..53, tmin_cpc -42..44,
    # tmean_era5land -2..35, soil moisture 0.06..0.55
    ranges = {'pcp_mm_cpc': (0, 450), 'pcp_mm_era5land': (0, 350),
              'aet_mm_gleam': (0, 20), 'pet_mm_gleam': (-1, 20),
              'pet_mm_era5land': (-10, 90),
              'eto_mm_brdwgd': (0, 15), 'airtemp_C_cpc_max': (-45, 60),
              'airtemp_C_cpc_min': (-45, 50), 'airtemp_C_mean_era5land': (-10, 40),
              'sm_surface_gleam': (0, 1), 'sml4_era5land': (0, 1)}

    for stn in stations:
        df = dyn[stn]
        for feature, (low, high) in ranges.items():
            values = df[feature].dropna()
            if values.empty:   # a catchment without BR-DWGD
                continue
            assert values.min() >= low and values.max() <= high, (stn, feature, values.min(), values.max())

        # mm/day is the m3/s value over the GSIM area, as the release computed it
        both = df[['q_cms_obs', 'q_mm_obs']].dropna()
        if both.empty:
            continue
        mm = both['q_cms_obs'] * 86400.0 / (float(areas[stn]) * 1e6) * 1e3
        # the published mm column is rounded to six decimals, which dominates the
        # difference for the small values
        tolerance = 1e-6 + 1e-4 * both['q_mm_obs'].abs()
        diff = np.abs(mm - both['q_mm_obs'])
        assert (diff <= tolerance).all(), (stn, diff.max(), both['q_mm_obs'][diff.idxmax()])
    return


def test_partial_coverage():
    """BR-DWGD covers 864 of the 897 catchments; the other 33 get the columns as
    NaN instead of losing them, and this is not reported as a missing download"""
    logger.info("test_partial_coverage")
    folder, suffix, _ = CAMELS_BR._DYN_GROUPS['1.2']['reference_evapotransp']
    have = {fname[:-len(suffix)] for fname
            in os.listdir(os.path.join(RELEASE_DIR, folder, folder))
            if fname.endswith(suffix)}
    assert len(have) == NUM_BRDWGD, len(have)

    without = sorted(set(dataset.stations()) - have)
    assert len(without) == NUM_STATIONS - NUM_BRDWGD, len(without)

    served = dataset._read_stn_dyn(without[0])
    assert list(served.columns) == list(RAW_COLUMNS)
    for feature in ('eto_mm_brdwgd', 'pcp_mm_brdwgd', 'airtemp_C_brdwgd_max'):
        assert served[feature].isna().all(), feature
    assert served['pcp_mm_cpc'].notna().any()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CAMELS_BR(path=CAMELS_BR_PATH, use_ana_update=False, verbosity=0)
    assert not [w for w in caught if 'incomplete' in str(w.message)], [str(w.message) for w in caught]
    return


def test_static_fidelity():
    """the static table is the attribute files of the release, unchanged and
    restricted to the 897 catchments"""
    logger.info("test_static_fidelity")
    attributes = '01_CAMELS_BR_attributes'
    raw = pd.concat(
        [pd.read_csv(os.path.join(RELEASE_DIR, attributes, attributes, fname),
                     sep=' ', index_col='gauge_id', dtype={'gauge_id': str})
         for fname in ATTRIBUTE_FILES], axis=1)

    static = dataset.fetch_static_features()
    assert static.shape == (NUM_STATIONS, NUM_STATIC), static.shape
    assert sorted(static.index) == sorted(dataset.stations())

    renamed = dataset.static_map
    for column in raw.columns:
        served = static[renamed.get(column, column)]
        values = raw.loc[static.index, column]
        if pd.api.types.is_numeric_dtype(values):
            assert np.allclose(served.to_numpy(float), values.to_numpy(float),
                               equal_nan=True), column
        else:
            assert list(served) == list(values), column
    assert 'gauge_region' not in static.columns   # dropped in release 1.2
    return


def test_coords_and_area():
    """coordinates and areas are those of the location file"""
    logger.info("test_coords_and_area")
    attributes = '01_CAMELS_BR_attributes'
    raw = pd.read_csv(os.path.join(RELEASE_DIR, attributes, attributes,
                                   'camels_br_location.txt'),
                      sep=' ', index_col='gauge_id', dtype={'gauge_id': str})

    coords = dataset.stn_coords()
    assert coords.shape == (NUM_STATIONS, 2), coords.shape
    assert list(coords.columns) == ['lat', 'long']
    assert np.allclose(coords['lat'].to_numpy(float), raw.loc[coords.index, 'gauge_lat'].to_numpy(float))
    assert np.allclose(coords['long'].to_numpy(float), raw.loc[coords.index, 'gauge_lon'].to_numpy(float))

    area = dataset.area()
    assert len(area) == NUM_STATIONS and area.name == 'area_km2'
    assert np.allclose(area.to_numpy(float), raw.loc[area.index, 'area_gsim'].to_numpy(float), rtol=1e-6)
    ana = dataset.area(source='ana')
    assert np.allclose(ana.to_numpy(float), raw.loc[ana.index, 'area_ana'].to_numpy(float),
                       rtol=1e-6, equal_nan=True)
    assert not np.allclose(area.to_numpy(float), ana.to_numpy(float))

    try:
        dataset.area(source='whatever')
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown area source was accepted")
    return


def test_q_mm():
    """q_mm serves the published mm/day column"""
    logger.info("test_q_mm")
    stations = random.sample(dataset.stations(), 3)
    q = dataset.q_mm(stations)
    assert list(q.columns) == stations and len(q) == DYN_LEN
    for stn in stations:
        raw = _raw_frame(RELEASE_DIR, stn, *RAW_COLUMNS['q_mm_obs'][:2])['streamflow_mm']
        served = q[stn].reindex(raw.index)
        both = raw.notna()
        err = np.abs(served[both] - raw[both]) / raw[both].abs().clip(lower=1e-9)
        assert err.max() < 1e-6, (stn, err.max())
    return


def test_boundaries():
    """the boundary GeoPackage is read and its float gauge ids are matched"""
    logger.info("test_boundaries")
    if fiona is None:
        logger.info("fiona is not installed, skipping")
        return
    assert dataset.boundary_file.endswith('camels_br_catchments.gpkg')
    ids = dataset._create_boundary_id_map()
    assert len(ids) == NUM_STATIONS, len(ids)
    assert set(ids) == set(dataset.stations()), sorted(set(ids) ^ set(dataset.stations()))[:5]

    def points(coordinates):
        """every point of a (Multi)Polygon, whatever its nesting"""
        if isinstance(coordinates[0], (int, float)):
            yield coordinates
        else:
            for part in coordinates:
                yield from points(part)

    stn = random.choice(dataset.stations())
    geometry = dataset.get_boundary(stn)
    assert geometry['type'] in ('Polygon', 'MultiPolygon'), geometry['type']
    assert len(np.array(list(points(geometry['coordinates'])))) > 2, stn

    # Every gauge lies within the bounding box of its catchment, except the
    # gauges the release itself places outside it. Counting them is what makes
    # this catch a broken projection or a wrong id map: both move the count at
    # once, while a tolerance alone would hide them.
    coords = dataset.stn_coords()
    outside = {}
    for gauge, geometry in dataset._create_boundary_id_map().items():
        xy = np.array(list(points(geometry['coordinates'])))
        lat, lon = coords.loc[gauge, 'lat'], coords.loc[gauge, 'long']
        offset = max(xy[:, 0].min() - lon, lon - xy[:, 0].max(),
                     xy[:, 1].min() - lat, lat - xy[:, 1].max(), 0.0)
        if offset:
            outside[gauge] = offset

    assert len(outside) == NUM_GAUGES_OUTSIDE_BOUNDARY, sorted(outside.items())[:5]
    assert max(outside.values()) < 0.1, sorted(outside.items(), key=lambda kv: -kv[1])[:3]
    return


def test_returns_copies():
    """a caller cannot corrupt the cached state of the dataset"""
    logger.info("test_returns_copies")
    stations = dataset.stations()
    stations.append('not a gauge')
    assert 'not a gauge' not in dataset.stations()

    static = dataset._static_data()
    column = static.columns[0]
    static[column] = -999
    assert not (dataset._static_data()[column] == -999).all()
    return


def test_fetch_period():
    """st and en slice the record"""
    logger.info("test_fetch_period")
    stn = random.choice(dataset.stations())
    _, dyn = dataset.fetch(stations=stn, st='2000-01-01', en='2000-12-31', as_dataframe=True)
    df = dyn[stn]
    assert len(df) == 366, len(df)
    assert df.index[0] == pd.Timestamp('2000-01-01') and df.index[-1] == pd.Timestamp('2000-12-31')
    return


def test_raw_and_simulated_streamflow():
    """the streamflow of every gauge of the release is served with its quality
    columns, and 'all' is refused because the folder does not fit in memory"""
    logger.info("test_raw_and_simulated_streamflow")
    gauges = dataset.all_stations('streamflow_m3s_raw')
    assert len(gauges) == 4025, len(gauges)
    assert set(dataset.stations()).issubset(gauges)

    stns = random.sample(gauges, 2)
    served = dataset.fetch_raw_streamflow(stns)
    assert list(served.columns) == ['q_cms_obs', 'qual_control_by_ana', 'qual_flag']
    assert served.index.names == ['gauge_id', 'time']
    for stn in stns:
        raw = _raw_frame(RELEASE_DIR, stn, '02_CAMELS_BR_streamflow_all_catchments', '_streamflow.txt')
        assert np.allclose(served.loc[stn, 'q_cms_obs'].to_numpy(float),
                           raw['streamflow_m3s'].to_numpy(float), equal_nan=True), stn
        assert np.allclose(served.loc[stn, 'qual_flag'].to_numpy(float),
                           raw['qual_flag'].to_numpy(float), equal_nan=True), stn

    for bad in (None, 'all'):
        try:
            dataset.fetch_raw_streamflow(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"stations={bad!r} was accepted")

    # the MGB-SA simulation is model output and is only shipped with release 1.1
    try:
        dataset.fetch_simulated_streamflow('10500000')
    except ValueError as e:
        assert '1.1' in str(e), e
    else:
        raise AssertionError("release 1.2 served simulated streamflow")

    old = dataset_of('1.1')
    simulated = old.fetch_simulated_streamflow('10500000')
    assert list(simulated.columns) == ['q_cms_sim'], simulated.columns
    assert len(old.all_stations('simulated_streamflow_m3s')) == 593
    return


def test_all_stations():
    """all_stations knows the groups of this release and rejects others"""
    logger.info("test_all_stations")
    assert len(dataset.all_stations('streamflow')) == NUM_STATIONS
    assert len(dataset.all_stations('soil_moisture')) == NUM_STATIONS
    assert len(dataset.all_stations('reference_evapotransp')) == NUM_BRDWGD
    try:
        dataset.all_stations('simulated_streamflow_m3s')   # release 1.1 only
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown group was accepted")
    return


def test_no_redownload_or_reextract():
    """initializing again neither downloads, extracts nor rebuilds the cache"""
    logger.info("test_no_redownload_or_reextract")

    def boom(*args, **kwargs):
        raise AssertionError("must not be called when the data exists")

    originals = _brazil.download, zipfile.ZipFile, (xr.Dataset.to_netcdf if xr is not None else None)
    _brazil.download = zipfile.ZipFile = boom
    if xr is not None:
        xr.Dataset.to_netcdf = boom
    try:
        again = CAMELS_BR(path=CAMELS_BR_PATH, use_ana_update=False, verbosity=0)
        assert len(again.stations()) == NUM_STATIONS
    finally:
        _brazil.download, zipfile.ZipFile = originals[:2]
        if xr is not None:
            xr.Dataset.to_netcdf = originals[2]
    return


def _copy_for(path: str, version: str) -> CAMELS_BR:
    """a copy of the dataset which points at ``path`` and ``version``, with the
    cached properties of the original dropped so that none of them describes the
    release it no longer is"""
    ds = copy.copy(dataset)
    ds._path, ds.version = path, version
    # every cached property, so that a new one cannot be forgotten here
    ds.__dict__ = {name: value for name, value in ds.__dict__.items()
                   if not isinstance(getattr(type(ds), name, None), functools.cached_property)}
    return ds


def _fake_download(calls):
    """a stand-in for download which records its calls and writes a small zip
    archive holding a folder of the archive's own name, as the real ones do"""
    def download(url, outdir, fname, verbosity=1):
        calls.append((url, outdir, fname))
        stem = fname[:-len('.zip')]
        with zipfile.ZipFile(os.path.join(outdir, fname), 'w') as zf:
            zf.writestr(f'{stem}/data.txt', 'year month day value\n1980 1 1 1.5\n')
    return download


def test_download():
    """every archive of the release is downloaded into a new folder and
    extracted into a folder of its own name; once extracted nothing is
    downloaded again and remove_zip then deletes the archives"""
    logger.info("test_download")
    for version, n_archives in (('1.2', 11), ('1.1', 15)):
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, 'CAMELS_BR')   # does not exist yet
        calls = []
        original = _brazil.download
        try:
            _brazil.download = _fake_download(calls)
            ds = _copy_for(path, version)
            ds._download_camels_br()

            assert len(calls) == n_archives, calls
            release = os.path.join(path, version)
            for url, outdir, fname in calls:
                assert url.startswith('https://zenodo.org/records/') and url.endswith(fname)
                assert outdir == release and fname.endswith('.zip')
            folders = [fname[:-len('.zip')] for _, _, fname in calls]
            assert sorted(os.listdir(release)) == sorted(folders + [f for _, _, f in calls])
            for folder in folders:
                assert os.listdir(os.path.join(release, folder)) == [folder], folder

            calls.clear()
            ds.remove_zip = True
            ds._download_camels_br()
            assert calls == [], f"downloaded again although the data is extracted: {calls}"
            assert sorted(os.listdir(release)) == sorted(folders), "remove_zip left archives"
        finally:
            _brazil.download = original
            shutil.rmtree(tmp, ignore_errors=True)
    return


def test_interrupted_extraction():
    """an extraction that stops half way leaves no folder which looks complete,
    so the next initialization extracts the archive again without downloading"""
    logger.info("test_interrupted_extraction")
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'CAMELS_BR')
    calls = []
    original_download, original_extractall = _brazil.download, zipfile.ZipFile.extractall
    try:
        _brazil.download = _fake_download(calls)
        ds = _copy_for(path, '1.2')

        def die(self, *args, **kwargs):
            original_extractall(self, *args, **kwargs)
            raise KeyboardInterrupt("interrupted")

        zipfile.ZipFile.extractall = die
        try:
            ds._download_camels_br()
        except KeyboardInterrupt:
            pass
        zipfile.ZipFile.extractall = original_extractall

        release = os.path.join(path, '1.2')
        first = sorted(os.listdir(release))
        assert not [name for name in first if name.endswith('_extracting')], first
        extracted = [name for name in first if not name.endswith('.zip')]
        assert len(extracted) < 11, extracted   # not all archives were extracted
        interrupted = calls[-1][2]              # its archive is on disk, unextracted

        # the archive of the interrupted extraction is not downloaded again, it
        # is only extracted
        ds._download_camels_br()
        assert [fname for _, _, fname in calls].count(interrupted) == 1, \
            f"{interrupted} was downloaded again although it is on disk"
        assert len([n for n in os.listdir(release) if not n.endswith('.zip')]) == 11
    finally:
        _brazil.download, zipfile.ZipFile.extractall = original_download, original_extractall
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_overwrite_removes_stale_before_download():
    """overwrite=True deletes the archives, the extracted folders and the files
    derived from them of this release before downloading it again"""
    logger.info("test_overwrite_removes_stale_before_download")
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'CAMELS_BR')
    calls = []
    original = _brazil.download
    try:
        _brazil.download = _fake_download(calls)
        ds = _copy_for(path, '1.2')
        ds._download_camels_br()

        release = os.path.join(path, '1.2')
        stale = os.path.join(release, '12_CAMELS_BR_catchment_boundaries', 'stale.txt')
        open(stale, 'w').close()
        open(ds._static_fpath, 'w').close()
        cache = os.path.join(path, 'camels_br_D_1.2_v2.nc')
        open(cache, 'w').close()
        other_release = os.path.join(path, 'camels_br_D_1.1_v2.nc')
        open(other_release, 'w').close()

        calls.clear()
        ds._download_camels_br(overwrite=True)

        assert len(calls) == 11, calls
        assert not os.path.exists(stale), "the stale extracted folder survived"
        assert not os.path.exists(ds._static_fpath), "the stale static table survived"
        assert not os.path.exists(cache), "the stale cache survived"
        assert os.path.exists(other_release), "the cache of the other release was deleted"
        assert not [f for f in os.listdir(release) if f.endswith('.zip1')], os.listdir(release)
    finally:
        _brazil.download = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_missing_files_warn():
    """a file deleted by hand is reported instead of being served as a smaller
    dataset"""
    logger.info("test_missing_files_warn")
    tmp = _mini_copy()
    try:
        path = os.path.join(tmp, 'CAMELS_BR', '1.2')
        folder, suffix, _ = CAMELS_BR._DYN_GROUPS['1.2']['precipitation']
        os.remove(os.path.join(path, folder, folder, f"10500000{suffix}"))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            CAMELS_BR(path=tmp, to_netcdf=False, verbosity=0)
        messages = [str(w.message) for w in caught]
        assert any('incomplete' in m and folder in m for m in messages), messages

        attributes = os.path.join(path, '01_CAMELS_BR_attributes', '01_CAMELS_BR_attributes')
        os.remove(os.path.join(attributes, 'camels_br_soil.txt'))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            CAMELS_BR(path=tmp, to_netcdf=False, verbosity=0)
        assert any('camels_br_soil.txt' in str(w.message) for w in caught), \
            [str(w.message) for w in caught]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


# ---------------------------------------------------------------- ANA update

# one gauge and two months as the ANA service serves them: January 2025 twice,
# raw and reviewed, and March 2025 with a blank day; February is not served
ANA_XML = """<?xml version="1.0" encoding="utf-8"?>
<DataTable xmlns="http://MRCS/">
  <SerieHistorica>
    <EstacaoCodigo>10500000</EstacaoCodigo><NivelConsistencia>1</NivelConsistencia>
    <DataHora>2025-01-01 00:00:00</DataHora>
    <Vazao01>10.0</Vazao01><Vazao01Status>1</Vazao01Status>
    <Vazao02>20.0</Vazao02><Vazao02Status>1</Vazao02Status>
  </SerieHistorica>
  <SerieHistorica>
    <EstacaoCodigo>10500000</EstacaoCodigo><NivelConsistencia>2</NivelConsistencia>
    <DataHora>2025-01-01 00:00:00</DataHora>
    <Vazao01>11.5</Vazao01><Vazao01Status>1</Vazao01Status>
    <Vazao02>21.5</Vazao02><Vazao02Status>2</Vazao02Status>
  </SerieHistorica>
  <SerieHistorica>
    <EstacaoCodigo>10500000</EstacaoCodigo><NivelConsistencia>1</NivelConsistencia>
    <DataHora>2025-03-01 00:00:00</DataHora>
    <Vazao01>30.0</Vazao01><Vazao01Status>1</Vazao01Status>
    <Vazao02></Vazao02><Vazao02Status></Vazao02Status>
    <Vazao03>0.0</Vazao03><Vazao03Status>6</Vazao03Status>
  </SerieHistorica>
</DataTable>"""


def test_ana_parse():
    """the xml of the service becomes one row per day: the reviewed version of a
    month wins, blanks are missing values and unserved months are absent"""
    logger.info("test_ana_parse")
    df = _brazil._ana_parse(ANA_XML, pd.Timestamp('2025-01-01'), pd.Timestamp('2025-12-31'))

    assert list(df.columns) == ['streamflow_m3s', 'status', 'nivel_consistencia']
    assert df.index.name == 'date' and df.index.is_monotonic_increasing
    # January is served at both levels, the reviewed values are kept
    assert df.loc['2025-01-01', 'streamflow_m3s'] == 11.5
    assert df.loc['2025-01-02', 'streamflow_m3s'] == 21.5
    assert (df.loc['2025-01-01':'2025-01-31', 'nivel_consistencia'] == 2).all()
    assert df.loc['2025-01-02', 'status'] == '2'
    # the month has 31 days although only two carry a value
    assert len(df.loc['2025-01-01':'2025-01-31']) == 31
    assert df.loc['2025-01-03':'2025-01-31', 'streamflow_m3s'].isna().all()
    # February is not served at all
    assert len(df.loc['2025-02-01':'2025-02-28']) == 0
    # March: a blank day is a missing value, a dry river bed is a zero
    assert df.loc['2025-03-01', 'streamflow_m3s'] == 30.0
    assert np.isnan(df.loc['2025-03-02', 'streamflow_m3s'])
    assert df.loc['2025-03-03', 'streamflow_m3s'] == 0.0
    assert df.loc['2025-03-03', 'status'] == '6'
    assert (df.loc['2025-03-01':'2025-03-31', 'nivel_consistencia'] == 1).all()

    # days outside the requested period are dropped
    short = _brazil._ana_parse(ANA_XML, pd.Timestamp('2025-03-01'), pd.Timestamp('2025-03-31'))
    assert short.index.min() == pd.Timestamp('2025-03-01') and len(short) == 31
    return


def test_ana_failure_is_not_no_data():
    """a request which fails leaves no file, so the gauge is asked again instead
    of being taken for a gauge without data"""
    logger.info("test_ana_failure_is_not_no_data")
    tmp = tempfile.mkdtemp()
    original = _brazil._ana_request
    try:
        _brazil._ana_request = lambda *a, **k: None
        out = _brazil._ana_fetch_station('10500000', pd.Timestamp('2025-01-01'),
                                         pd.Timestamp('2025-01-31'), tmp)
        assert out['status'] == 'failed' and out['days'] == 0, out
        assert os.listdir(tmp) == [], os.listdir(tmp)

        _brazil._ana_request = lambda *a, **k: "<DataTable></DataTable>"
        out = _brazil._ana_fetch_station('10500000', pd.Timestamp('2025-01-01'),
                                         pd.Timestamp('2025-01-31'), tmp)
        assert out['status'] == 'no data' and out['days'] == 0, out
        assert os.listdir(tmp) == ['10500000.csv'], os.listdir(tmp)

        _brazil._ana_request = lambda *a, **k: ANA_XML
        out = _brazil._ana_fetch_station(
            '10500000', pd.Timestamp('2025-01-01'), pd.Timestamp('2025-12-31'), tmp)
        assert (out['status'], out['days'], out['first'], out['last']) == \
            ('ok', 4, '2025-01-01', '2025-03-03'), out

        # a second, narrower request keeps the days the file already holds and
        # reports the values ANA has changed since
        _brazil._ana_request = lambda *a, **k: _ana_xml('10500000', pd.Timestamp('2025-03-01'), 99.0)
        out = _brazil._ana_fetch_station(
            '10500000', pd.Timestamp('2025-03-01'), pd.Timestamp('2025-03-31'), tmp)
        # January is untouched, 2025-03-01 changed from 30.0 to 99.0 and
        # 2025-03-03 is dropped: ANA served March again without its 0.0, which
        # means the value has been withdrawn
        assert out['days'] == 3 and out['first'] == '2025-01-01', out
        assert out['changed'] == 1 and out['withdrawn'] == 1 and out['dropped'] == 0, out
    finally:
        _brazil._ana_request = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def _stub_ana(monkey_values):
    """makes _ana_request serve ``monkey_values[station]``"""
    def request(station, start, end, retries=5, timeout=180.0):
        return monkey_values.get(str(station))
    return request


def test_update_streamflow():
    """the download extends the released record without replacing any released
    value, is not repeated on the next call and is served afterwards"""
    logger.info("test_update_streamflow")
    tmp = _mini_copy(stations=('10500000',))
    original = _brazil._ana_request
    try:
        ds = CAMELS_BR(path=tmp, to_netcdf=False, verbosity=0)
        released = ds._read_stn_dyn('10500000')
        last_released = ds._released_q_end('10500000')
        assert last_released is not None

        # ANA serves the month after the released record, plus a value on a day
        # the release already has (which must not be used)
        served_month = (last_released + pd.Timedelta(days=1)).replace(day=1)
        xml = (f'<DataTable><SerieHistorica><EstacaoCodigo>10500000</EstacaoCodigo>'
               f'<NivelConsistencia>1</NivelConsistencia>'
               f'<DataHora>{served_month.strftime("%Y-%m-%d")} 00:00:00</DataHora>'
               + ''.join(f'<Vazao{day:02d}>{day}.0</Vazao{day:02d}>'
                         f'<Vazao{day:02d}Status>1</Vazao{day:02d}Status>'
                         for day in range(1, 29))
               + '</SerieHistorica><SerieHistorica><EstacaoCodigo>10500000</EstacaoCodigo>'
               '<NivelConsistencia>2</NivelConsistencia>'
               '<DataHora>2000-01-01 00:00:00</DataHora>'
               '<Vazao01>-5.0</Vazao01><Vazao01Status>1</Vazao01Status>'
               '</SerieHistorica></DataTable>')
        _brazil._ana_request = _stub_ana({'10500000': xml})

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            q = ds.update_streamflow(en=served_month + pd.offsets.MonthEnd(0))
        assert any('ANA served data for 1' in str(w.message) for w in caught), \
            [str(w.message) for w in caught]

        # a released value is never replaced
        assert q.loc['2000-01-01', '10500000'] != -5.0
        released_days = released['q_cms_obs'].dropna().index
        assert np.allclose(q.loc[released_days, '10500000'].to_numpy(float),
                           released['q_cms_obs'].dropna().to_numpy(float), rtol=1e-6)
        # and the days it did not have are filled
        assert q.loc[served_month + pd.Timedelta(days=4), '10500000'] == 5.0
        assert ds.end >= served_month + pd.Timedelta(days=27)

        manifest = json.load(open(os.path.join(ds._ana_dir, 'manifest.json')))
        assert manifest['stations']['10500000']['status'] == 'ok'
        assert manifest['stations']['10500000']['days'] == 28

        raw = ds.ana_streamflow('10500000')
        assert list(raw.columns) == ['streamflow_m3s', 'status', 'nivel_consistencia']
        assert raw.index.names == ['gauge_id', 'time']

        # a second call does not download again
        def boom(*args, **kwargs):
            raise AssertionError("downloaded again although the data is on disk")
        _brazil._ana_request = boom
        ds.update_streamflow(en=served_month + pd.offsets.MonthEnd(0))

        # a new dataset serves the extension, one built with use_ana_update=False
        # serves the released record
        extended = CAMELS_BR(path=tmp, to_netcdf=False, verbosity=0)
        assert extended._serves_ana
        assert 'ana' in extended.dyn_fname
        assert extended.end > ds._released_q_end('10500000')

        plain = CAMELS_BR(path=tmp, to_netcdf=False, use_ana_update=False, verbosity=0)
        assert not plain._serves_ana
        assert plain.dyn_fname == 'camels_br_D_1.2_v2.nc'
        assert plain.end == pd.Timestamp('2024-12-31')
        assert plain._read_stn_dyn('10500000')['q_cms_obs'].last_valid_index() == last_released
    finally:
        _brazil._ana_request = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_ana_mm_conversion():
    """the streamflow added from ANA is converted to mm/day with the same area
    the release used"""
    logger.info("test_ana_mm_conversion")
    tmp = _mini_copy(stations=('10500000',))
    original = _brazil._ana_request
    try:
        ds = CAMELS_BR(path=tmp, to_netcdf=False, verbosity=0)
        last = ds._released_q_end('10500000')
        month = (last + pd.Timedelta(days=1)).replace(day=1)
        xml = (f'<DataTable><SerieHistorica><EstacaoCodigo>10500000</EstacaoCodigo>'
               f'<NivelConsistencia>2</NivelConsistencia>'
               f'<DataHora>{month.strftime("%Y-%m-%d")} 00:00:00</DataHora>'
               f'<Vazao01>100.0</Vazao01><Vazao01Status>1</Vazao01Status>'
               f'</SerieHistorica></DataTable>')
        _brazil._ana_request = _stub_ana({'10500000': xml})
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ds.update_streamflow(en=month + pd.offsets.MonthEnd(0))

        df = CAMELS_BR(path=tmp, to_netcdf=False, verbosity=0)._read_stn_dyn('10500000')
        area = float(pd.read_csv(os.path.join(
            tmp, 'CAMELS_BR', '1.2', '01_CAMELS_BR_attributes', '01_CAMELS_BR_attributes',
            'camels_br_location.txt'), sep=' ', index_col='gauge_id',
            dtype={'gauge_id': str}).loc['10500000', 'area_gsim'])

        assert df.loc[month, 'q_cms_obs'] == 100.0
        expected = 100.0 * 86400.0 / (area * 1e6) * 1e3
        assert abs(df.loc[month, 'q_mm_obs'] - expected) / expected < 1e-5, \
            (df.loc[month, 'q_mm_obs'], expected)
    finally:
        _brazil._ana_request = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_update_streamflow_live():
    """one gauge is really downloaded from ANA. The only test which touches the
    web service, so that the download path is exercised once; set
    AQUA_FETCH_NO_NETWORK to skip it."""
    logger.info("test_update_streamflow_live")
    if os.environ.get('AQUA_FETCH_NO_NETWORK'):
        logger.info("AQUA_FETCH_NO_NETWORK is set, skipping")
        return
    tmp = _mini_copy(stations=('10500000',))
    try:
        ds = CAMELS_BR(path=tmp, to_netcdf=False, verbosity=0)
        last = ds._released_q_end('10500000')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # a dead service must not hold the suite for 5 x 180 s
            q = ds.update_streamflow(st=last + pd.Timedelta(days=1),
                                     en=last + pd.Timedelta(days=90),
                                     retries=2, timeout=30)
        if any('did not answer for 1' in str(w.message) for w in caught):
            logger.info("the ANA service did not answer, skipping")
            return
        added = q['10500000'].loc[last + pd.Timedelta(days=1):]
        assert added.notna().any(), "ANA served nothing for a gauge it has data for"
        assert added.dropna().min() >= 0, added.dropna().min()
        raw = ds.ana_streamflow('10500000')
        assert raw['nivel_consistencia'].isin([1, 2]).all()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_version_11():
    """release 1.1 is still readable and unchanged"""
    logger.info("test_version_11")
    old = dataset_of('1.1')
    assert len(old.stations()) == NUM_STATIONS
    assert old.start == pd.Timestamp('1980-01-01') and old.end == pd.Timestamp('2018-12-31')

    stn = random.choice(old.stations())
    df = old._read_stn_dyn(stn)
    assert df.shape == (DYN_LEN_11, NUM_DYNAMIC_11), df.shape

    raw = _raw_frame(release_dir(old), stn, '03_CAMELS_BR_streamflow_mm_selected_catchments',
                     '_streamflow_mm.txt')['streamflow_mm']
    both = raw.notna()
    err = np.abs(df.loc[raw.index, 'q_mm_obs'][both] - raw[both]) / raw[both].abs().clip(lower=1e-9)
    assert err.max() < 1e-6, err.max()

    # m3/s is computed from mm/day with the GSIM area
    area = float(old.area(stn).iloc[0])
    expected = df['q_mm_obs'] * 1e-3 * area * 1e6 / 86400
    diff = np.abs(df['q_cms_obs'] - expected) / expected.abs().clip(lower=1e-9)
    assert np.nanmax(diff) < 1e-4, np.nanmax(diff)

    # the boundary file of 1.1 is the shapefile and its ids are matched as well
    if fiona is not None:
        assert old.boundary_file.endswith('.shp')
        assert len(old._create_boundary_id_map()) == NUM_STATIONS
    return


def test_releases_do_not_mix():
    """both releases ship a folder named 01_CAMELS_BR_attributes with different
    content, so each release must read its own"""
    logger.info("test_releases_do_not_mix")
    new, old = dataset, dataset_of('1.1')
    assert release_dir(new) != release_dir(old), release_dir(new)

    location = os.path.join('01_CAMELS_BR_attributes', '01_CAMELS_BR_attributes',
                            'camels_br_location.txt')
    rows = {version: len(pd.read_csv(os.path.join(release_dir(ds), location), sep=' '))
            for version, ds in (('1.2', new), ('1.1', old))}
    assert rows == {'1.2': 4025, '1.1': 3679}, rows

    assert 'gauge_region' in old.static_features and 'gauge_region' not in new.static_features
    assert len(old.static_features) == NUM_STATIC_11 and len(new.static_features) == NUM_STATIC
    assert old._static_fpath != new._static_fpath
    # the unversioned table the previous implementation wrote is not read
    assert os.path.basename(old._static_fpath) == 'static_features_1.1.csv'
    assert os.path.basename(new._static_fpath) == 'static_features_1.2.csv'
    # and the two releases keep their downloads from ANA apart
    assert release_dir(old) in old._ana_dir and release_dir(new) in new._ana_dir
    assert old._ana_dir != new._ana_dir
    return


def test_overwrite_on_a_legacy_11_layout():
    """a release 1.1 which lies directly in ``path`` is refreshed in place:
    overwrite must not delete the folder which tells the layout apart and then
    download into a folder nothing reads"""
    logger.info("test_overwrite_on_a_legacy_11_layout")
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, 'CAMELS_BR')
    calls = []
    original = _brazil.download
    try:
        # the layout the previous implementation left behind
        for fname in CAMELS_BR.urls['1.1']:
            stem = fname[:-len('.zip')]
            os.makedirs(os.path.join(path, stem, stem))
            with open(os.path.join(path, stem, stem, 'data.txt'), 'w') as f:
                f.write('old\n')

        _brazil.download = _fake_download(calls)
        ds = _copy_for(path, '1.1')
        assert ds._release_dir == path, ds._release_dir
        ds._download_camels_br(overwrite=True)

        assert len(calls) == 15, calls
        assert {outdir for _, outdir, _ in calls} == {path}, calls
        sample = os.path.join(path, '03_CAMELS_BR_streamflow_mm_selected_catchments',
                              '03_CAMELS_BR_streamflow_mm_selected_catchments')
        assert os.path.isdir(sample), "the data was deleted and not replaced"
        assert not os.path.exists(os.path.join(path, '1.1')), \
            "downloaded into a folder the reader does not look at"
    finally:
        _brazil.download = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def _ana_xml(station: str, month: pd.Timestamp, value: float, days: int = 1,
             level: int = 1) -> str:
    """the answer of the service for one gauge and one month"""
    return ('<DataTable><SerieHistorica>'
            f'<EstacaoCodigo>{station}</EstacaoCodigo>'
            f'<NivelConsistencia>{level}</NivelConsistencia>'
            f'<DataHora>{month.strftime("%Y-%m-%d")} 00:00:00</DataHora>'
            + ''.join(f'<Vazao{day:02d}>{value}</Vazao{day:02d}>'
                      f'<Vazao{day:02d}Status>1</Vazao{day:02d}Status>'
                      for day in range(1, days + 1))
            + '</SerieHistorica></DataTable>')


def test_two_updates_on_one_day():
    """a second update on the same day is served, not answered from the cache of
    the first one"""
    logger.info("test_two_updates_on_one_day")
    tmp = _mini_copy(stations=('10500000', '11400000'))
    original = _brazil._ana_request
    try:
        ds = CAMELS_BR(path=tmp, verbosity=0)          # with the netCDF cache
        first_month = (ds._released_q_end('10500000') + pd.Timedelta(days=1)).replace(day=1)
        en = first_month + pd.offsets.MonthEnd(0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _brazil._ana_request = _stub_ana({'10500000': _ana_xml('10500000', first_month, 111.0)})
            ds.update_streamflow(stations=['10500000'], en=en)
            cache_after_first = ds.dyn_fname

            _brazil._ana_request = _stub_ana({'11400000': _ana_xml('11400000', first_month, 222.0)})
            q = ds.update_streamflow(stations=['11400000'], en=en)

        assert ds.dyn_fname != cache_after_first, \
            "the cache name did not change although the served record did"
        assert q.loc[first_month, '11400000'] == 222.0, q.loc[first_month].to_dict()

        served = CAMELS_BR(path=tmp, verbosity=0)
        _, dyn = served.fetch(stations=['10500000', '11400000'], as_dataframe=True)
        assert dyn['10500000'].loc[first_month, 'q_cms_obs'] == 111.0
        assert dyn['11400000'].loc[first_month, 'q_cms_obs'] == 222.0

        # the cache of the superseded download is gone, not left behind
        caches = [f for f in os.listdir(served.path) if f.endswith('.nc')]
        assert sorted(caches) == sorted({served.dyn_fname, 'camels_br_D_1.2_v2.nc'}), caches
    finally:
        _brazil._ana_request = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_update_keeps_an_earlier_wider_download():
    """a routine update after one with an earlier ``st`` must not throw away the
    days that earlier download brought"""
    logger.info("test_update_keeps_an_earlier_wider_download")
    tmp = _mini_copy(stations=('10500000',))
    original = _brazil._ana_request
    try:
        ds = CAMELS_BR(path=tmp, to_netcdf=False, verbosity=0)
        last = ds._released_q_end('10500000')
        old_month = pd.Timestamp('2000-06-01')          # a gap inside the record
        new_month = (last + pd.Timedelta(days=1)).replace(day=1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _brazil._ana_request = _stub_ana(
                {'10500000': _ana_xml('10500000', old_month, 55.0, days=3)})
            ds.update_streamflow(st='2000-01-01', en='2000-12-31')
            wide = ds.ana_streamflow('10500000')
            assert len(wide) > 0

            _brazil._ana_request = _stub_ana(
                {'10500000': _ana_xml('10500000', new_month, 77.0, days=3)})
            ds.update_streamflow(en=new_month + pd.offsets.MonthEnd(0))

        after = ds.ana_streamflow('10500000')
        assert old_month in after.index.get_level_values('time'), \
            "the days of the earlier download were lost"
        assert after.loc[('10500000', old_month), 'streamflow_m3s'] == 55.0
        assert after.loc[('10500000', new_month), 'streamflow_m3s'] == 77.0

        info = json.load(open(ds._ana_manifest_fpath))['stations']['10500000']
        assert pd.Timestamp(info['start']) <= old_month, info
        assert pd.Timestamp(info['end']) >= new_month, info

        # overwrite=True replaces the file, which is reported and recorded
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _brazil._ana_request = _stub_ana(
                {'10500000': _ana_xml('10500000', new_month, 88.0, days=3)})
            ds.update_streamflow(en=new_month + pd.offsets.MonthEnd(0), overwrite=True)
        assert any('dropped 3 days' in str(w.message) for w in caught), \
            [str(w.message) for w in caught]

        replaced = ds.ana_streamflow('10500000')
        assert old_month not in replaced.index.get_level_values('time')
        info = json.load(open(ds._ana_manifest_fpath))['stations']['10500000']
        assert pd.Timestamp(info['start']) == new_month, info
    finally:
        _brazil._ana_request = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_ana_days_before_the_release_start():
    """days which ANA serves before the release begins are not clipped away"""
    logger.info("test_ana_days_before_the_release_start")
    tmp = _mini_copy(stations=('10500000',))
    original = _brazil._ana_request
    try:
        ds = CAMELS_BR(path=tmp, to_netcdf=False, verbosity=0)
        assert ds.start == pd.Timestamp('1980-01-01')
        early = pd.Timestamp('1975-06-01')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _brazil._ana_request = _stub_ana(
                {'10500000': _ana_xml('10500000', early, 42.0)})
            q = ds.update_streamflow(st='1975-01-01', en='1975-12-31')

        assert q.index[0] <= early, q.index[0]
        assert q.loc[early, '10500000'] == 42.0
        assert ds.start == early, ds.start
    finally:
        _brazil._ana_request = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_ana_download_belongs_to_its_release():
    """the days to download follow the record of the release, so a download made
    for one release is not served by the other"""
    logger.info("test_ana_download_belongs_to_its_release")
    tmp = _mini_copy(stations=('10500000',))
    original = _brazil._ana_request
    try:
        ds = CAMELS_BR(path=tmp, to_netcdf=False, verbosity=0)
        month = (ds._released_q_end('10500000') + pd.Timedelta(days=1)).replace(day=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _brazil._ana_request = _stub_ana({'10500000': _ana_xml('10500000', month, 9.0)})
            ds.update_streamflow(en=month + pd.offsets.MonthEnd(0))

        assert os.path.isdir(ds._ana_dir)
        assert os.path.relpath(ds._ana_dir, os.path.join(tmp, 'CAMELS_BR')) == \
            os.path.join('1.2', 'ana_streamflow')

        _mini_copy(stations=('10500000',), version='1.1', into=tmp)
        old = CAMELS_BR(path=tmp, version='1.1', to_netcdf=False, verbosity=0)
        assert not old._serves_ana, "the download of release 1.2 is served by 1.1"
    finally:
        _brazil._ana_request = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_incomplete_release_is_not_cached():
    """a release with missing files is reported and its truncated record is not
    written to the cache that a complete one would use"""
    logger.info("test_incomplete_release_is_not_cached")
    tmp = _mini_copy(stations=('10500000',))
    try:
        path = os.path.join(tmp, 'CAMELS_BR', '1.2')
        attributes = os.path.join(path, '01_CAMELS_BR_attributes', '01_CAMELS_BR_attributes')
        os.remove(os.path.join(attributes, 'camels_br_location.txt'))

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds = CAMELS_BR(path=tmp, verbosity=0)      # to_netcdf is on by default
        messages = [str(w.message) for w in caught]
        assert any('incomplete' in m for m in messages), messages
        assert any('cache is not built' in m for m in messages), messages
        assert not os.path.exists(ds.dyn_fpath), "an incomplete release was cached"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_getstate_leaves_the_heavy_tables_behind():
    """the dataset is pickled once per station by the process pool, so the big
    cached tables must not travel with it"""
    logger.info("test_getstate_leaves_the_heavy_tables_behind")
    import pickle
    ds = CAMELS_BR(path=CAMELS_BR_PATH, use_ana_update=False, verbosity=0)
    ds._static_data(), ds.stn_coords(), ds.area()      # fill the caches
    for cached in ('_static_table', '_location_table'):
        assert cached in ds.__dict__, cached

    small = len(pickle.dumps(ds))
    original = CAMELS_BR.__dict__['__getstate__']
    try:
        del CAMELS_BR.__getstate__                      # pickle everything
        big = len(pickle.dumps(ds))
    finally:
        CAMELS_BR.__getstate__ = original
    assert CAMELS_BR.__dict__['__getstate__'] is original, "the class was left without it"
    assert small < big / 5, (small, big)
    assert small < 60_000, small
    restored = pickle.loads(pickle.dumps(ds))
    assert restored.stations() == ds.stations()
    assert len(restored._static_data()) == NUM_STATIONS   # rebuilt in the worker
    return


def test_station_argument_types():
    """a wrong type is refused with a clear error instead of failing deep in a read"""
    logger.info("test_station_argument_types")
    for accessor in (dataset.fetch_raw_streamflow, dataset.update_streamflow):
        for bad in (5, 1.5, {'10500000': 1}):
            try:
                accessor(bad)
            except TypeError:
                pass
            else:
                raise AssertionError(f"{accessor.__name__}(stations={bad!r}) was accepted")
    # an iterable of ids is materialized, a generator is not emptied by a first pass
    assert _brazil._as_station_list(pd.Series(['1', '2'])) == ['1', '2']
    assert _brazil._as_station_list(iter(['1', '2'])) == ['1', '2']
    assert _brazil._as_station_list({'1': 0}.keys()) == ['1']
    # a tuple and a set are iterables of ids and are accepted
    served = dataset.fetch_raw_streamflow(('10500000',))
    assert served.index.get_level_values('gauge_id').unique().tolist() == ['10500000']
    return


def test_revised_values_are_served():
    """when ANA revises the values of days it already served, the refreshed
    record is what fetch() returns, not the cache built before"""
    logger.info("test_revised_values_are_served")
    tmp = _mini_copy(stations=('10500000',))
    original = _brazil._ana_request
    try:
        ds = CAMELS_BR(path=tmp, verbosity=0)          # with the netCDF cache
        month = (ds._released_q_end('10500000') + pd.Timedelta(days=1)).replace(day=1)
        en = month + pd.offsets.MonthEnd(0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _brazil._ana_request = _stub_ana({'10500000': _ana_xml('10500000', month, 5.0, days=3)})
            ds.update_streamflow(en=en)
            first_cache = ds.dyn_fname
            assert ds.fetch(stations='10500000', as_dataframe=True)[1]['10500000'].loc[month, 'q_cms_obs'] == 5.0

            # the same days, other values, which is what ANA's review does
            _brazil._ana_request = _stub_ana({'10500000': _ana_xml('10500000', month, 6.0, days=3)})
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                q = ds.update_streamflow(en=en, overwrite=True)

        assert ds.dyn_fname != first_cache, "the cache name ignored the revision"
        assert q.loc[month, '10500000'] == 6.0, q.loc[month].to_dict()

        served = CAMELS_BR(path=tmp, verbosity=0)
        _, dyn = served.fetch(stations='10500000', as_dataframe=True)
        assert dyn['10500000'].loc[month, 'q_cms_obs'] == 6.0, "the stale cache was served"
    finally:
        _brazil._ana_request = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_failed_request_keeps_what_was_downloaded():
    """a gauge the service does not answer for keeps the data an earlier call
    downloaded instead of disappearing from the record"""
    logger.info("test_failed_request_keeps_what_was_downloaded")
    tmp = _mini_copy(stations=('10500000',))
    original = _brazil._ana_request
    try:
        ds = CAMELS_BR(path=tmp, to_netcdf=False, verbosity=0)
        month = (ds._released_q_end('10500000') + pd.Timedelta(days=1)).replace(day=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _brazil._ana_request = _stub_ana({'10500000': _ana_xml('10500000', month, 7.0, days=3)})
            ds.update_streamflow(en=month + pd.offsets.MonthEnd(0))

            _brazil._ana_request = lambda *a, **k: None      # the service is down
            ds.update_streamflow(en=month + pd.Timedelta(days=90))

        info = json.load(open(ds._ana_manifest_fpath))['stations']['10500000']
        assert info['status'] == 'ok' and info['days'] == 3, info
        assert info['last_attempt'] == 'failed', info

        after = CAMELS_BR(path=tmp, to_netcdf=False, verbosity=0)
        assert after._serves_ana and '10500000' in after._ana_stations
        assert after._read_stn_dyn('10500000').loc[month, 'q_cms_obs'] == 7.0
    finally:
        _brazil._ana_request = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_withdrawn_value_is_not_resurrected():
    """a day ANA serves without a value has been withdrawn: the merge must not
    keep the old value and pair it with the new consistency level"""
    logger.info("test_withdrawn_value_is_not_resurrected")
    tmp = tempfile.mkdtemp()
    original = _brazil._ana_request
    try:
        month = pd.Timestamp('2025-01-01')
        _brazil._ana_request = lambda *a, **k: _ana_xml('10500000', month, 10.0, days=2, level=1)
        first = _brazil._ana_fetch_station('10500000', month, month + pd.offsets.MonthEnd(0), tmp)
        assert first['days'] == 2, first

        # ANA serves the month again, reviewed, with the second day withdrawn
        _brazil._ana_request = lambda *a, **k: _ana_xml('10500000', month, 11.0, days=1, level=2)
        second = _brazil._ana_fetch_station('10500000', month, month + pd.offsets.MonthEnd(0), tmp)
        assert second['days'] == 1, second
        assert second['changed'] == 1 and second['withdrawn'] == 1, second
        assert second['dropped'] == 0, "a merge does not drop days, it only follows ANA"

        held = _brazil._read_ana_csv(os.path.join(tmp, '10500000.csv'))
        assert held.loc[month, 'streamflow_m3s'] == 11.0
        assert np.isnan(held.loc[month + pd.Timedelta(days=1), 'streamflow_m3s']), \
            "a value ANA has withdrawn is still served"
        assert (held.loc[month:month + pd.offsets.MonthEnd(0), 'nivel_consistencia'] == 2).all()
    finally:
        _brazil._ana_request = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_incomplete_release_11_does_not_crash():
    """an incomplete release 1.1 is reported, like 1.2, instead of raising while
    telling the user to use overwrite=True"""
    logger.info("test_incomplete_release_11_does_not_crash")
    tmp = _mini_copy(stations=('10500000',), version='1.1')
    try:
        attributes = os.path.join(tmp, 'CAMELS_BR', '1.1', '01_CAMELS_BR_attributes',
                                  '01_CAMELS_BR_attributes')
        os.remove(os.path.join(attributes, 'camels_br_location.txt'))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds = CAMELS_BR(path=tmp, version='1.1', verbosity=0)
        messages = [str(w.message) for w in caught]
        assert any('incomplete' in m for m in messages), messages
        assert not os.path.exists(ds.dyn_fpath)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_overwrite_when_ana_has_nothing_left():
    """overwrite=True empties the file when ANA no longer serves the period, so
    the record must stop being served instead of coming from the old cache"""
    logger.info("test_overwrite_when_ana_has_nothing_left")
    tmp = _mini_copy(stations=('10500000',))
    original = _brazil._ana_request
    try:
        ds = CAMELS_BR(path=tmp, verbosity=0)           # with the netCDF cache
        month = (ds._released_q_end('10500000') + pd.Timedelta(days=1)).replace(day=1)
        en = month + pd.offsets.MonthEnd(0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _brazil._ana_request = _stub_ana({'10500000': _ana_xml('10500000', month, 1000.0, days=3)})
            ds.update_streamflow(en=en)
            first_cache = ds.dyn_fname
            assert ds.fetch(stations='10500000', as_dataframe=True)[1]['10500000'].loc[month, 'q_cms_obs'] == 1000.0

            # ANA now answers that it has nothing for this gauge and period
            _brazil._ana_request = lambda *a, **k: "<DataTable></DataTable>"
            ds.update_streamflow(en=en, overwrite=True)

        assert _brazil._read_ana_csv(ds._ana_file('10500000')).empty
        info = json.load(open(ds._ana_manifest_fpath))['stations']['10500000']
        assert info['status'] == 'no data' and info['days'] == 0, info

        fresh = CAMELS_BR(path=tmp, verbosity=0)
        assert not fresh._serves_ana, "an emptied download is still served"
        assert fresh.dyn_fname != first_cache
        _, dyn = fresh.fetch(stations='10500000', as_dataframe=True)
        assert np.isnan(dyn['10500000'].loc[month, 'q_cms_obs']), "the stale cache was served"
    finally:
        _brazil._ana_request = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_withdrawn_and_dropped_are_reported_apart():
    """a value ANA has withdrawn and a day lost to overwrite=True are different
    events and are reported as such"""
    logger.info("test_withdrawn_and_dropped_are_reported_apart")
    tmp = _mini_copy(stations=('10500000',))
    original = _brazil._ana_request
    try:
        ds = CAMELS_BR(path=tmp, to_netcdf=False, verbosity=0)
        month = (ds._released_q_end('10500000') + pd.Timedelta(days=1)).replace(day=1)
        en = month + pd.offsets.MonthEnd(0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _brazil._ana_request = _stub_ana({'10500000': _ana_xml('10500000', month, 3.0, days=3)})
            ds.update_streamflow(en=en)

        # the same month, reviewed, with only the first day left. st reaches
        # before what is on disk, so the month is requested again
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _brazil._ana_request = _stub_ana(
                {'10500000': _ana_xml('10500000', month, 4.0, days=1, level=2)})
            ds.update_streamflow(st=month - pd.Timedelta(days=5), en=en)
        messages = [str(w.message) for w in caught]
        assert any('no longer served by ANA' in m and '2 days' in m for m in messages), messages
        assert not any('overwrite=True' in m for m in messages), messages
    finally:
        _brazil._ana_request = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_a_gauge_named_twice_is_downloaded_once():
    """a repeated gauge must not be requested twice, which would have two
    threads writing its file at the same time"""
    logger.info("test_a_gauge_named_twice_is_downloaded_once")
    tmp = _mini_copy(stations=('10500000',))
    original = _brazil._ana_request
    requests = []
    try:
        ds = CAMELS_BR(path=tmp, to_netcdf=False, verbosity=0)
        month = (ds._released_q_end('10500000') + pd.Timedelta(days=1)).replace(day=1)

        def request(station, start, end, retries=5, timeout=180.0):
            requests.append(str(station))
            return _ana_xml('10500000', month, 2.0, days=3)

        _brazil._ana_request = request
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            q = ds.update_streamflow(stations=['10500000', '10500000'],
                                     en=month + pd.offsets.MonthEnd(0))
        assert requests == ['10500000'], requests
        assert list(q.columns) == ['10500000'], q.columns
        assert q.loc[month, '10500000'] == 2.0
    finally:
        _brazil._ana_request = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_efficiency():
    """fetching every station stays quick"""
    logger.info("test_efficiency")
    start = time.time()
    _, dyn = dataset.fetch(as_dataframe=True)
    took = time.time() - start
    assert len(dyn) == NUM_STATIONS
    logger.info(f"fetched {NUM_STATIONS} stations in {took:.1f} s")
    assert took < 15, f"fetching all stations took {took:.1f} s"

    start = time.time()
    dataset.stn_coords()
    dataset.area()
    dataset.stations()
    assert time.time() - start < 1, "the documented one liners are slow"
    return


def test_shared_suite():
    """the checks every rainfall-runoff dataset has to pass"""
    logger.info("test_shared_suite")
    run_shared_tests(dataset, NUM_STATIONS, DYN_LEN, NUM_STATIC, NUM_DYNAMIC)
    return


def test_shared_suite_11():
    """the same checks for release 1.1"""
    logger.info("test_shared_suite_11")
    run_shared_tests(dataset_of('1.1'), NUM_STATIONS, DYN_LEN_11,
                     NUM_STATIC_11, NUM_DYNAMIC_11)
    return


# --------------------------------------------------------------------- CABra


def test_cabra():
    """the shared suite on CABra, once per meteorological source"""
    logger.info("test_cabra")
    cabra_dir = os.path.join(CABRA_PATH, 'CABra')
    if not os.path.exists(cabra_dir):
        # CABra is a download of its own, do not start it by accident
        print(f"skipping CABra, {cabra_dir} does not exist")
        return

    for source in ['era5', 'ref', 'ens']:
        ds = CABra(path=CABRA_PATH, met_src=source, verbosity=4)
        run_shared_tests(ds, 735, 10957, 87, 13)
    return


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith('test_') and callable(test):
            print(f"{name} ...", flush=True)
            start = time.time()
            test()
            print(f"{name} passed in {time.time() - start:.1f} s", flush=True)
    print("all tests passed")
