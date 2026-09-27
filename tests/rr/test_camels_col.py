"""
Tests for the CAMELS_COL (Colombia, 346 catchments) dataset.

The class serves the February 2026 release of CAMELS-COL
(`zenodo 18794895 <https://zenodo.org/records/18794895>`_). The tests verify
that it

    * exposes the correct number of stations (346), dynamic (5) and static (79)
      features, and a temporal extent derived from the data rather than from a
      hardcoded literal,
    * fetches the time series and the attributes **without changing the values
      or their units**, compared against a fresh, independent read of the raw
      files,
    * reads the dd/mm/yyyy dates day first (a naive read shifts 37 of the 346
      gauges by months),
    * does not re-download or re-extract when the release is already on disk,
    * does **not** mistake files of the superseded 2025 release, which lie
      directly in ``path``, for a completed download of this release,
    * converts the gauge coordinates and the catchment boundaries from
      EPSG:3395 (World Mercator, meters) to WGS84, matching ``pyproj``.

The projection tests need neither ``pyproj`` nor the dataset: the expected
values are hardcoded from a ``pyproj`` run. The tests that need the data are
skipped when it is not on disk.
"""

import os
import site
import shutil
import zipfile
import logging
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd

# add the repository root to the path so that ``aqua_fetch`` can be imported
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

if __name__ == "__main__":
    logging.basicConfig(filename='test_camels_col.log', filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

import aqua_fetch.download_zenodo
import aqua_fetch.rr.utils
from aqua_fetch import CAMELS_COL
from aqua_fetch._backend import fiona
from aqua_fetch._geom_utils import world_mercator_to_wgs84
from aqua_fetch.rr._camels import _extract_zip, _read_col_csv


raw_data_path = '/path/to/raw/data'   # replace with the real raw data root

NUM_STATIONS = 346
NUM_DYNAMIC = 5
NUM_STATIC = 79
NUM_STEPS = 15340          # daily steps from 1981-01-01 to 2022-12-31

# Three real CAMELS-COL gauges, chosen as the southern-most, a middle and the
# northern-most one, so the worst-case latitudes are covered:
#   gauge id: ((easting, northing) as given in 02_CAMELS_COL_Catchment_information,
#              (lat, lon) from pyproj Transformer.from_crs("EPSG:3395", "EPSG:4326"))
# Note the record names the northing column ``gauge_lat`` and the easting
# column ``gauge_lon``.
REF_COORDS = {
    '47077010': ((-8102899.35, -159753.98), (-1.4446111016660301, -72.78958331787135)),
    '21257110': ((-8338377.18, 567537.59), (5.125694471257405, -74.90491665547434)),
    '15017020': ((-8231136.31, 1255256.04), (11.277638919565327, -73.94155552944159)),
}

# the first, middle and last vertex of the outer ring of catchment 35067040:
# (index, (easting, northing) in the shapefile, (lon, lat) from pyproj)
BOUNDARY_REF_STN = '35067040'
BOUNDARY_REF = [
    (0, (-8195346.143816721, 549390.4838304777), (-73.62004699640542, 4.962221635259413)),
    (1598, (-8209877.608888865, 517986.7109880647), (-73.75058536815499, 4.679233839965909)),
    (3196, (-8195346.143816721, 549390.4838304777), (-73.62004699640542, 4.962221635259413)),
]

# a gauge whose record starts on the 1st of a month, which a month-first read
# of 01/03/1981 would turn into the 3rd of January
DAY_FIRST_STN = '12017020'
DAY_FIRST_START = pd.Timestamp('1981-03-01')

# the only two gauges of the release which share a position
DUPLICATE_GAUGES = ('21247040', '21247050')

# the only gauge of the release whose values do not all survive the float32
# downcast at the 3 decimals the record writes: 6 of its discharges, all above
# 16000 m3/s, move by 0.001 m3/s
WORST_DOWNCAST_STN = '31097010'

_CACHED = {}


def _camels_col_path() -> str:
    return os.path.join(raw_data_path, 'CAMELS')


def _data_available() -> bool:
    return os.path.exists(os.path.join(
        _camels_col_path(), 'CAMELS_COL', 'camels_col',
        '02_CAMELS_COL_Catchment_information.csv'))


def _dataset() -> CAMELS_COL:
    """one instance shared by the tests, so the release is read once"""
    if 'ds' not in _CACHED:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _CACHED['ds'] = CAMELS_COL(path=_camels_col_path(), verbosity=0,
                                       remove_zip=False)
    return _CACHED['ds']


def _skip_without_data():
    """
    Skips a test that needs the release. The skip is loud when ``raw_data_path``
    is still the checked in placeholder, because a silent skip makes the suite
    look green while it tests almost nothing.
    """
    if raw_data_path == '/path/to/raw/data':
        warnings.warn(
            "tests/rr/test_camels_col.py: raw_data_path is still the placeholder, "
            "so every test that needs the CAMELS_COL release is skipped. Set it to "
            "the raw data root to run them.", UserWarning)
    if not _data_available():
        raise unittest.SkipTest("CAMELS_COL data is not available")


def _cache_name(fp=np.float32) -> str:
    """
    The name of the netCDF cache, derived from the class rather than written
    out, so that a fixture which plants a stale cache cannot go on planting a
    name the class stopped using.
    """
    stub = CAMELS_COL.__new__(CAMELS_COL)
    stub.name = CAMELS_COL.__name__
    stub.timestep = 'D'
    stub.fp = fp
    return stub.dyn_fname


def _raw_ts(dataset: CAMELS_COL, stn: str) -> pd.DataFrame:
    """the raw time series file of one gauge, read independently of the class"""
    df = pd.read_csv(dataset._ts_file(stn), sep='\t')
    df.index = pd.to_datetime(df.pop('Fecha'), format='%d/%m/%Y')
    return df


# --------------------------------------------------------------------------
# coordinate conversion
# --------------------------------------------------------------------------

def _check_world_mercator_matches_pyproj():
    """the helper reproduces pyproj to better than 1e-9 degrees (~0.1 mm)."""
    logger.info("test_world_mercator_matches_pyproj")
    easting = np.array([xy[0] for xy, _ in REF_COORDS.values()])
    northing = np.array([xy[1] for xy, _ in REF_COORDS.values()])
    exp_lat = np.array([ll[0] for _, ll in REF_COORDS.values()])
    exp_lon = np.array([ll[1] for _, ll in REF_COORDS.values()])

    lat, lon = world_mercator_to_wgs84(easting, northing)

    np.testing.assert_allclose(lat, exp_lat, atol=1e-9)
    np.testing.assert_allclose(lon, exp_lon, atol=1e-9)
    return


def _check_spherical_inverse_is_not_accurate_enough():
    """the *spherical* inverse the class used before is off by 1.1 km at the
    southern-most gauge and 8.2 km at the northern-most one, so this asserts
    the helper is genuinely the ellipsoidal one and would fail if it were
    replaced by ``2*atan(exp(y/a)) - pi/2`` again."""
    logger.info("test_spherical_inverse_is_not_accurate_enough")
    a = 6378137.0
    for (easting, northing), (exp_lat, _) in REF_COORDS.values():
        spherical = np.degrees(2 * np.arctan(np.exp(northing / a)) - np.pi / 2)
        ellipsoidal, _ = world_mercator_to_wgs84(easting, northing)

        # the helper is right to ~0.01 mm ...
        assert abs(float(ellipsoidal) - exp_lat) < 1e-9, (ellipsoidal, exp_lat)
        # ... and it is not the spherical formula: the two differ by more than a
        # kilometre everywhere, and by more the further the gauge is from the
        # equator (1.1 km at -1.44 deg, 8.2 km at 11.28 deg)
        apart_m = abs(float(ellipsoidal) - spherical) * 111319.0
        assert apart_m > 1000.0, f"the helper is only {apart_m:.0f} m from the spherical inverse"
    return


def _check_stn_coords_are_degrees():
    """all gauge coordinates are degrees inside the Colombian bounding box."""
    logger.info("test_stn_coords_are_degrees")
    _skip_without_data()
    coords = _dataset().stn_coords('all')
    assert coords.shape == (NUM_STATIONS, 2), coords.shape
    assert coords['lat'].between(-4.5, 13.5).all(), coords['lat'].describe()
    assert coords['long'].between(-79.5, -66.5).all(), coords['long'].describe()

    # and they agree with the pyproj reference. ``stn_coords`` serves float32
    # (``self.fp``), so the tolerance is 1e-5 deg (~1 m); the conversion itself
    # is exact to ~0.01 mm, which test_world_mercator_matches_pyproj asserts.
    for sid, (_, (exp_lat, exp_lon)) in REF_COORDS.items():
        assert abs(float(coords.loc[sid, 'lat']) - exp_lat) < 1e-5, sid
        assert abs(float(coords.loc[sid, 'long']) - exp_lon) < 1e-5, sid
    return


def _check_boundary_is_transformed_to_wgs84():
    """``get_boundary`` converts EPSG:3395 meters to degrees while keeping the
    geometry type and the ring structure, for plain polygons, polygons with
    holes and MultiPolygons alike."""
    logger.info("test_boundary_is_transformed_to_wgs84")
    _skip_without_data()
    if fiona is None:
        raise unittest.SkipTest("fiona is not installed")

    def rings(geometry):
        if geometry.type == 'MultiPolygon':
            return [ring for polygon in geometry.coordinates for ring in polygon]
        return list(geometry.coordinates)

    dataset = _dataset()

    n_multi = n_holed = 0
    for stn in dataset.stations():
        projected = dataset.get_boundary(stn, to_wgs84=False)
        geographic = dataset.get_boundary(stn, to_wgs84=True)

        assert geographic.type == projected.type, (stn, projected.type)
        src, dst = rings(projected), rings(geographic)
        assert len(src) == len(dst), stn

        n_multi += projected.type == 'MultiPolygon'
        n_holed += projected.type == 'Polygon' and len(src) > 1

        for src_ring, dst_ring in zip(src, dst):
            assert len(src_ring) == len(dst_ring), stn
            xy = np.asarray(dst_ring, dtype=float)
            assert (np.abs(xy[:, 0]) <= 180).all(), stn   # lon
            assert (np.abs(xy[:, 1]) <= 90).all(), stn    # lat
            # inside Colombia
            assert xy[:, 0].min() > -84.0 and xy[:, 0].max() < -60.0, stn
            assert xy[:, 1].min() > -6.0 and xy[:, 1].max() < 16.0, stn

    # the structures a single-polygon shortcut would have missed
    assert n_multi > 0, "expected MultiPolygon boundaries"
    assert n_holed > 0, "expected polygons with interior rings"

    # and the converted vertices are the pyproj values, not merely in range
    ring = dataset.get_boundary(BOUNDARY_REF_STN).coordinates[0]
    projected = dataset.get_boundary(BOUNDARY_REF_STN, to_wgs84=False).coordinates[0]
    for i, (easting, northing), (exp_lon, exp_lat) in BOUNDARY_REF:
        np.testing.assert_allclose(projected[i], (easting, northing), rtol=0, atol=1e-6)
        np.testing.assert_allclose(ring[i], (exp_lon, exp_lat), rtol=0, atol=1e-9)
    return


# --------------------------------------------------------------------------
# what the release holds
# --------------------------------------------------------------------------

def _check_counts():
    """the release has 346 gauges, 5 dynamic and 79 static features."""
    logger.info("test_counts")
    _skip_without_data()
    dataset = _dataset()

    assert len(dataset.stations()) == NUM_STATIONS, len(dataset.stations())
    assert len(set(dataset.stations())) == NUM_STATIONS, "station ids are not unique"
    assert dataset.stations() == sorted(dataset.stations()), "stations are not sorted"

    assert dataset.dynamic_features == [
        'pcp_mm', 'pet_mm', 'airtemp_C_min', 'airtemp_C_max', 'q_cms_obs'
    ], dataset.dynamic_features

    assert len(dataset.static_features) == NUM_STATIC, len(dataset.static_features)
    # 13077030 was dropped by this release
    assert '13077030' not in dataset.stations()
    return


def _check_temporal_extent_is_derived():
    """``start``/``end`` come from the files, not from a hardcoded literal."""
    logger.info("test_temporal_extent_is_derived")
    _skip_without_data()
    dataset = _dataset()

    # scanned independently over every file
    first, last = [], []
    for stn in dataset.stations():
        dates = pd.read_csv(dataset._ts_file(stn), sep='\t', usecols=['Fecha'])['Fecha']
        dates = pd.to_datetime(dates, format='%d/%m/%Y')
        first.append(dates.min())
        last.append(dates.max())

    assert dataset.start == min(first), (dataset.start, min(first))
    assert dataset.end == max(last), (dataset.end, max(last))
    assert len(pd.date_range(dataset.start, dataset.end, freq='D')) == NUM_STEPS
    return


def _check_stations_and_attributes_agree():
    """every gauge with a time series file also has attributes, and no gauge of
    the attribute tables is without a time series file."""
    logger.info("test_stations_and_attributes_agree")
    _skip_without_data()
    dataset = _dataset()
    static = dataset._static_data()

    assert static.shape == (NUM_STATIONS, NUM_STATIC), static.shape
    assert static.index.tolist() == dataset.stations()
    assert not static.index.has_duplicates
    # no gauge is entirely without attributes
    assert not static.isna().all(axis=1).any()
    return


def _check_attribute_names_are_unique():
    """the water body share of the land cover and of the soil table are kept
    apart, so no attribute name is unreachable."""
    logger.info("test_attribute_names_are_unique")
    _skip_without_data()
    features = _dataset().static_features

    assert len(features) == len(set(features)), \
        [f for f in features if features.count(f) > 1]
    assert 'water_bodies_perc' in features        # from the land cover table
    assert 'water_bodies_soil_perc' in features   # from the soil table
    # the trailing spaces of the physiographic table are stripped
    assert not [f for f in features if f != f.strip()], features
    return


# --------------------------------------------------------------------------
# fidelity: the served values are the raw values
# --------------------------------------------------------------------------

def _check_dynamic_values_and_units_unchanged():
    """the served time series are the raw values, in the raw units."""
    logger.info("test_dynamic_values_and_units_unchanged")
    _skip_without_data()
    dataset = _dataset()

    raw_to_served = {
        'Precipitacion': 'pcp_mm',               # mm/day, CHIRPS v2.0
        'ETP_': 'pet_mm',                        # mm/day, MSWX
        'Temperatura_minima': 'airtemp_C_min',   # degC, MSWX
        'Temperatura_maxima': 'airtemp_C_max',   # degC, MSWX
        'Caudal': 'q_cms_obs',                   # m3/s, IDEAM
    }

    stations = dataset.stations()[::37]   # 10 gauges spread over the list
    _, dynamic = dataset.fetch(stations, as_dataframe=True)

    for stn in stations:
        raw = _raw_ts(dataset, stn)
        served = dynamic[stn]

        assert served.shape[1] == NUM_DYNAMIC, served.shape
        # the raw dates are all served, and their values are unchanged
        assert raw.index.isin(served.index).all(), stn
        # exactly: the documented float32 downcast is the only thing that may
        # happen to a value, so the served numbers are compared bit for bit
        # with the float32 of the source, not with a loose rtol
        assert (served.dtypes == np.float32).all(), served.dtypes.to_dict()
        on_raw = served.loc[raw.index]
        for raw_col, served_col in raw_to_served.items():
            np.testing.assert_array_equal(
                on_raw[served_col].values,
                raw[raw_col].values.astype(np.float32),
                err_msg=f"{stn}: {raw_col} was changed")

        # the days the file does not hold are NaN, never zero or interpolated
        padded = served.index.difference(raw.index)
        if len(padded):
            assert served.loc[padded].isna().all().all(), stn
    return


def _check_read_stn_dyn_returns_the_raw_values():
    """
    The reader itself, not only the netCDF cache, serves the raw values. The
    other fidelity test goes through ``fetch``, which reads the cache whenever
    it exists, so a defect in ``_read_stn_dyn`` would hide behind it.
    """
    logger.info("test_read_stn_dyn_returns_the_raw_values")
    _skip_without_data()
    dataset = _dataset()

    raw_to_served = {
        'Precipitacion': 'pcp_mm', 'ETP_': 'pet_mm',
        'Temperatura_minima': 'airtemp_C_min', 'Temperatura_maxima': 'airtemp_C_max',
        'Caudal': 'q_cms_obs',
    }

    for stn in dataset.stations()[::37]:
        raw = _raw_ts(dataset, stn)
        served = dataset._read_stn_dyn(stn)

        assert (served.dtypes == np.float32).all(), served.dtypes.to_dict()
        for raw_col, served_col in raw_to_served.items():
            np.testing.assert_array_equal(
                served.loc[raw.index, served_col].values,
                raw[raw_col].values.astype(np.float32),
                err_msg=f"{stn}: _read_stn_dyn changed {raw_col}")
    return


def _check_the_float32_downcast_keeps_the_source_precision():
    """
    The values are served as float32, a downcast of the float64 the source
    parses to, so the loss it may cause is bounded here. Rounded back to the 3
    decimals the record writes, every value of the sampled gauges is unchanged.
    The one exception in the whole record is gauge 31097010, whose 6 discharges
    above 16000 m3/s move by 0.001 m3/s, their 8th significant digit; it is
    checked by name so that a larger loss cannot hide behind the sample.
    """
    logger.info("test_the_float32_downcast_keeps_the_source_precision")
    _skip_without_data()
    dataset = _dataset()

    worst = 0.0
    for stn in list(dataset.stations()[::37]) + [WORST_DOWNCAST_STN]:
        raw = _raw_ts(dataset, stn)
        for col in raw.columns:
            f64 = raw[col].to_numpy(dtype=np.float64)
            f64 = f64[np.isfinite(f64)]
            moved = np.abs(f64.astype(np.float32).astype(np.float64) - f64)
            # never more than 1e-7 of the value itself
            np.testing.assert_array_less(
                moved, np.abs(f64) * 1e-7 + 1e-12,
                err_msg=f"{stn}: float32 moved {col} by more than 1e-7 relative")
            if stn != WORST_DOWNCAST_STN:
                # and, at the precision the record writes, not at all
                np.testing.assert_array_equal(
                    np.round(f64.astype(np.float32).astype(np.float64), 3),
                    np.round(f64, 3),
                    err_msg=f"{stn}: float32 changed {col} at 3 decimals")
            else:
                worst = max(worst, moved.max())

    # the documented worst case of the whole record
    assert worst <= 0.001, worst

    # and the class itself serves that gauge as the float32 of its file: the
    # sample above steps over 31097010, so without this the worst gauge of the
    # record would never go through the reader
    raw = _raw_ts(dataset, WORST_DOWNCAST_STN)
    served = dataset._read_stn_dyn(WORST_DOWNCAST_STN)
    assert (served.dtypes == np.float32).all(), served.dtypes.to_dict()
    np.testing.assert_array_equal(
        served.loc[raw.index, 'q_cms_obs'].values,
        raw['Caudal'].values.astype(np.float32),
        err_msg=f"{WORST_DOWNCAST_STN}: the served discharge is not the float32 "
                f"of the file")
    return


def _check_the_cache_name_carries_the_release_and_the_precision():
    """
    The netCDF cache is named for the release and for the precision it holds.
    Both matter on disk: the superseded 2025 release wrote a camels_col_D_v2.nc
    of its own, and so did this class while it still served float64, and
    neither may be read as this cache.
    """
    logger.info("test_the_cache_name_carries_the_release_and_the_precision")
    _skip_without_data()
    dataset = _dataset()

    assert dataset.dyn_fname == 'camels_col_D_2026_v2.nc', dataset.dyn_fname

    fp = dataset.fp
    try:
        dataset.fp = np.float64
        assert dataset.dyn_fname == 'camels_col_D_2026_float64_v2.nc', dataset.dyn_fname
    finally:
        dataset.fp = fp
    return


def _check_every_gauge_is_padded_to_the_common_index():
    """
    ``_read_stn_dyn`` pads to the full daily index itself, so the time axis does
    not depend on the netCDF cache or on which gauges are asked for. Without
    this, a fetch of two short gauges returned their own shorter index.
    """
    logger.info("test_every_gauge_is_padded_to_the_common_index")
    _skip_without_data()
    dataset = _dataset()

    for stn in ('11017010', '11027010', dataset.stations()[-1]):
        raw = _raw_ts(dataset, stn)
        served = dataset._read_stn_dyn(stn)

        assert len(served) == NUM_STEPS, (stn, len(served))
        assert served.index[0] == dataset.start and served.index[-1] == dataset.end
        # the padded days are NaN, never zero or interpolated
        padded = served.index.difference(raw.index)
        assert len(padded) == NUM_STEPS - len(raw), stn
        assert served.loc[padded].isna().all().all(), stn
        # and the observed days are untouched
        assert served.loc[raw.index].notna().all().all(), stn
    return


def _check_dates_are_read_day_first():
    """the dd/mm/yyyy dates are read day first. A month-first read turns the
    01/03/1981 of gauge 12017020 into 1981-01-03 and shifts its whole series."""
    logger.info("test_dates_are_read_day_first")
    _skip_without_data()
    dataset = _dataset()

    # the series is padded to the common index, so the first *observed* day is
    # what the date format decides
    series = dataset._read_stn_dyn(DAY_FIRST_STN)
    observed = series.dropna(how='all')
    assert observed.index[0] == DAY_FIRST_START, observed.index[0]

    # and the raw file really is the ambiguous case this guards against
    with open(dataset._ts_file(DAY_FIRST_STN)) as f:
        f.readline()
        assert f.readline().split('\t')[0] == '01/03/1981'

    # the dates are strictly increasing, which a mixed-up parse would break
    assert observed.index.is_monotonic_increasing
    return


def _check_a_truncated_release_is_reported():
    """
    An interrupted extraction leaves a time series folder with fewer than the
    346 files. ``ts_path`` still exists, so the count is checked against the
    record's own list of gauges instead of against what happens to be on disk.
    """
    logger.info("test_a_truncated_release_is_reported")
    _skip_without_data()

    dataset = CAMELS_COL.__new__(CAMELS_COL)      # no download, no __init__
    dataset.verbosity = 0
    dataset.path = _camels_col_path()             # the setter appends the class name
    # six gauges never extracted
    full = _dataset().stations()
    dataset.__dict__['_stn_ids'] = full[:-6]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dataset._check_manifest()
    messages = " ".join(str(w.message) for w in caught)

    assert 'time series files' in messages, messages
    for stn in full[-6:]:
        assert stn in messages, f"{stn} not reported as missing: {messages}"

    # and with every file there, it says nothing
    dataset.__dict__['_stn_ids'] = full
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dataset._check_manifest()
    assert not [w for w in caught if 'time series files' in str(w.message)], \
        [str(w.message) for w in caught]
    return


def _check_static_values_and_units_unchanged():
    """the served attributes are the raw values, except the gauge position,
    which is converted from EPSG:3395 meters to WGS84 degrees."""
    logger.info("test_static_values_and_units_unchanged")
    _skip_without_data()
    dataset = _dataset()
    static = dataset._static_data()
    root = dataset._release_dir

    # area (km2), perimeter (km) and mean elevation (m) of the physiographic table
    physio = pd.read_csv(os.path.join(root, '10_CAMELS_COL_Physiograpic_characteristics.csv'),
                         sep=';', index_col=0, dtype={0: str})
    physio.index = physio.index.astype(str)
    for raw_col, served_col in [('area', 'area_km2'), ('perimeter', 'perimeter_km'),
                                ('mean_ele', 'elev_catch_m'),
                                ('maximum_ele', 'elev_catch_max_m'),
                                ('minimum_ele', 'elev_catch_min_m')]:
        np.testing.assert_allclose(
            static.loc[physio.index, served_col].values.astype(float),
            physio[raw_col].values.astype(float), rtol=1e-9,
            err_msg=f"{raw_col} was changed")

    # the hydrological signatures, the only comma separated table
    signatures = pd.read_csv(os.path.join(root, '09_CAMELS_COL_Hydrological_signatures.csv'),
                             sep=',', index_col=0, dtype={0: str})
    signatures.index = signatures.index.astype(str)
    np.testing.assert_allclose(static.loc[signatures.index, 'runoff_ratio'].values.astype(float),
                               signatures['runoff_ratio'].values.astype(float), rtol=1e-9)

    # the catchment information, the only latin-1 table: its text survives
    info = pd.read_csv(os.path.join(root, '02_CAMELS_COL_Catchment_information.csv'),
                       sep=';', index_col=0, dtype={0: str}, encoding='latin-1')
    info.index = info.index.astype(str)
    assert (static.loc[info.index, 'gauge_department'] == info['gauge_department']).all()
    # the record's own start of each record is kept as its dd/mm/yyyy text
    assert static.loc['11017010', 'gauge_star'] == '19/05/1981'
    np.testing.assert_allclose(static.loc[info.index, 'elev_gauge_m'].values.astype(float),
                               info['gauge_elev'].values.astype(float), rtol=1e-9)
    return


def _check_land_use_capability_table():
    """the 16037 empty rows of the land use capability table are dropped and
    its ``11017010.00`` ids are normalized."""
    logger.info("test_land_use_capability_table")
    _skip_without_data()
    dataset = _dataset()
    static = dataset._static_data()

    classes = [f'class_{i}' for i in range(1, 9)] + ['Urban_area', 'No_classified']
    for col in classes:
        assert col in static.columns, col
    # every gauge has a share, and the shares are percentages
    assert static[classes].notna().all().all()
    assert static[classes].to_numpy(dtype=float).max() <= 100.0
    # and they are shares of the whole catchment, so they add up
    np.testing.assert_allclose(static[classes].to_numpy(dtype=float).sum(axis=1),
                               100.0, atol=0.05)

    raw = pd.read_csv(os.path.join(dataset._release_dir, '11_CAMELS_COL_Land_use_capability.csv'),
                      sep=';', dtype={0: str})
    assert len(raw) > NUM_STATIONS, "expected the padded raw table"
    assert raw['gauge_id'].dropna().iloc[0].endswith('.00'), "expected float-like raw ids"
    assert len(raw[raw['gauge_id'].notna()]) == NUM_STATIONS, "only blank rows may be dropped"

    # every table keeps all its gauges: a row with one blank cell must survive
    for name in CAMELS_COL._TABLES:
        table = _read_col_csv(dataset._release_file(name))
        assert len(table) == NUM_STATIONS, (name, table.shape)
    return


def _check_static_data_and_stations_return_copies():
    """a caller cannot corrupt the cached attribute table or station list."""
    logger.info("test_static_data_and_stations_return_copies")
    _skip_without_data()
    dataset = _dataset()

    first = dataset._static_data()
    first.iloc[0, 0] = 'corrupted'
    first.drop(columns=first.columns[1], inplace=True)

    second = dataset._static_data()
    assert second.iloc[0, 0] != 'corrupted'
    assert second.shape == (NUM_STATIONS, NUM_STATIC), second.shape

    stns = dataset.stations()
    stns.append('not a gauge')
    assert len(dataset.stations()) == NUM_STATIONS
    return


def _check_duplicate_gauges_are_warned_about():
    """the two gauges which share a position are reported, not excluded."""
    logger.info("test_duplicate_gauges_are_warned_about")
    _skip_without_data()
    dataset = _dataset()

    # a fresh instance, so that the warning of the cached table is raised again
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fresh = CAMELS_COL(path=_camels_col_path(), verbosity=0, remove_zip=False,
                           to_netcdf=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fresh._static_data()
    messages = " ".join(str(w.message) for w in caught)

    assert 'only the coordinates are compared' in messages, messages
    for stn in DUPLICATE_GAUGES:
        assert stn in messages, f"{stn} not reported as a duplicate: {messages}"
        assert stn in dataset.stations(), f"{stn} was excluded"
    return


# --------------------------------------------------------------------------
# download behaviour
# --------------------------------------------------------------------------

class _Downloaded(Exception):
    """raised by the stubs to show that a download or extraction was started"""


def _check_near_duplicate_positions_are_warned_about():
    """
    The duplicate check rounds the coordinates before comparing. The only real
    duplicate pair of this release has bit-identical coordinates, so exact
    matching happens to find it too and nothing would notice if the rounding
    were dropped -- yet CLAUDE.md asks for it, because the same gauge in two
    source collections disagrees in the last decimals. Two gauges moved 10 m
    apart, which no exact comparison can see, must still be reported.
    """
    logger.info("test_near_duplicate_positions_are_warned_about")
    _skip_without_data()

    tmp = tempfile.mkdtemp()
    try:
        dataset = _dataset()
        name = '02_CAMELS_COL_Catchment_information.csv'
        # this one table of the record is latin-1, the others are utf-8
        table = pd.read_csv(dataset._release_file(name), sep=';', dtype={0: str},
                            encoding='latin-1')
        ids = table.columns[0]
        first, second = table.index[0], table.index[1]
        # the second gauge is put 10 m east and 10 m north of the first, which
        # is ~1e-4 degrees: inside the 3 decimals, outside exact equality
        for col, shift in (('gauge_lat', 10.0), ('gauge_lon', 10.0)):
            table.loc[second, col] = float(table.loc[first, col]) + shift
        moved = (str(table.loc[first, ids]), str(table.loc[second, ids]))
        assert table.loc[first, 'gauge_lat'] != table.loc[second, 'gauge_lat']

        mirror = _mirror(dataset, tmp,
                         replace={name: table.to_csv(sep=';', index=False)})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            static = mirror._static_table
        messages = " ".join(str(w.message) for w in caught)

        assert 'share a position' in messages, messages
        for stn in moved:
            assert stn in messages, (stn, messages)
        # and both are kept, never dropped
        assert len(static) == NUM_STATIONS, static.shape
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def _check_no_redownload_or_reextraction():
    """initializing again, with the release on disk, neither downloads nor
    extracts nor rebuilds the netCDF cache."""
    logger.info("test_no_redownload_or_reextraction")
    _skip_without_data()

    def refuse(*args, **kwargs):
        raise _Downloaded("the class downloaded or extracted again")

    zenodo = aqua_fetch.download_zenodo.download_from_zenodo
    extractall = zipfile.ZipFile.extractall
    to_netcdf = aqua_fetch.rr.utils.atomic_to_netcdf
    aqua_fetch.download_zenodo.download_from_zenodo = refuse
    zipfile.ZipFile.extractall = refuse
    aqua_fetch.rr.utils.atomic_to_netcdf = refuse
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            again = CAMELS_COL(path=_camels_col_path(), verbosity=0, remove_zip=False)
        assert len(again.stations()) == NUM_STATIONS
        # and remove_zip=False really kept them: the gate, not only the choice
        # of which files remove_zip_files() picks, has to work
        for name in CAMELS_COL._ARCHIVES:
            assert os.path.exists(again._release_file(name)), \
                f"{name} was deleted although remove_zip=False"
    finally:
        aqua_fetch.download_zenodo.download_from_zenodo = zenodo
        zipfile.ZipFile.extractall = extractall
        aqua_fetch.rr.utils.atomic_to_netcdf = to_netcdf
    return


def _check_an_extracted_release_without_its_archives_is_not_downloaded_again():
    """
    After ``remove_zip=True`` the archives are gone but the extracted folders
    are there, and that must be enough: the "already downloaded?" decision is
    made on the folders, never on the archives. Gating it on the archives would
    re-download 199 MB on every initialization, which is the first thing
    CLAUDE.md warns about and which no other test can see, because they all run
    with the archives in place.
    """
    logger.info("test_an_extracted_release_without_its_archives_is_not_downloaded_again")
    _skip_without_data()

    tmp = tempfile.mkdtemp()
    try:
        # the release as remove_zip=True leaves it: everything but the 3 zips
        dataset = _dataset()
        mirror = _mirror(dataset, tmp, drop=CAMELS_COL._ARCHIVES)
        for name in CAMELS_COL._ARCHIVES:
            assert not os.path.exists(mirror._release_file(name)), name
            assert os.path.isdir(mirror._folder_path(name)), name

        def refuse(*args, **kwargs):
            raise _Downloaded("the class downloaded or extracted again")

        zenodo = aqua_fetch.download_zenodo.download_from_zenodo
        extractall = zipfile.ZipFile.extractall
        aqua_fetch.download_zenodo.download_from_zenodo = refuse
        zipfile.ZipFile.extractall = refuse
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                again = CAMELS_COL(path=tmp, verbosity=0, to_netcdf=False,
                                   remove_zip=False)
        finally:
            aqua_fetch.download_zenodo.download_from_zenodo = zenodo
            zipfile.ZipFile.extractall = extractall

        assert len(again.stations()) == NUM_STATIONS
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def _check_files_of_the_superseded_release_do_not_block_the_download():
    """
    The 2025 release wrote its files directly into ``path``, some under names
    this release also uses. A ``path`` which holds them must still be filled
    with this release: the check is made against *this* release's files, which
    live in their own sub-folder.
    """
    logger.info("test_files_of_the_superseded_release_do_not_block_the_download")

    tmp = tempfile.mkdtemp()
    try:
        old = os.path.join(tmp, 'CAMELS_COL')
        # the 2025 release as it lies on disk: attribute workbooks, the folders
        # its archives were extracted into, and its netCDF cache
        os.makedirs(os.path.join(old, '03_CAMELS_COL_Basin_boundary',
                                 '03_CAMELS_COL_Basin_boundary'))
        os.makedirs(os.path.join(old, '04_CAMELS_COL_Hydrometeorological_data',
                                 '04_CAMELS_COL_Hydrometeorological_data'))
        for name in ('02_CAMELS_COL_Catchment_information.xlsx',
                     '07_CAMELS_COL_Soil_characteristics.xlsx',
                     '04_CAMELS_COL_Hydrometeorological_data.zip',
                     '00_CAMELS-COL  Description.docx'):
            open(os.path.join(old, name), 'w').close()
        open(os.path.join(old, 'camels_col_D_v2.nc'), 'w').close()

        asked = {}

        def stub(outdir, doi=None, include=None, **kwargs):
            asked['outdir'] = outdir
            asked['include'] = list(include or [])
            raise _Downloaded

        zenodo = aqua_fetch.download_zenodo.download_from_zenodo
        aqua_fetch.download_zenodo.download_from_zenodo = stub
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    CAMELS_COL(path=tmp, verbosity=0, to_netcdf=False)
                except _Downloaded:
                    pass
                else:
                    raise AssertionError(
                        "the old release on disk was taken for a complete download")
            messages = " ".join(str(w.message) for w in caught)
        finally:
            aqua_fetch.download_zenodo.download_from_zenodo = zenodo

        # every file of this release was asked for, into its own sub-folder
        assert asked['outdir'] == os.path.join(old, 'camels_col'), asked['outdir']
        for name in CAMELS_COL._TABLES + CAMELS_COL._ARCHIVES:
            assert name in asked['include'], name

        # and the user is told that the old files are there and are ignored
        assert 'superseded' in messages, messages
        assert '15554735' in messages, messages
        # every pattern of _SUPERSEDED is covered by the fixture: 2 xlsx, a zip,
        # a docx and the old netCDF cache
        assert 'holds 7 files and folders' in messages, messages

        # nothing of the old release was deleted
        assert os.path.exists(os.path.join(old, '02_CAMELS_COL_Catchment_information.xlsx'))
        assert os.path.exists(os.path.join(old, 'camels_col_D_v2.nc'))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def _check_overwrite_downloads_a_genuinely_fresh_copy():
    """
    ``overwrite=True`` removes this release's archives, extracted folders,
    tables and netCDF cache and asks for them again, and touches nothing of the
    superseded release.
    """
    logger.info("test_overwrite_downloads_a_genuinely_fresh_copy")

    tmp = tempfile.mkdtemp()
    try:
        root = os.path.join(tmp, 'CAMELS_COL')
        release = os.path.join(root, 'camels_col')
        os.makedirs(release)
        # a complete looking release
        for name in CAMELS_COL._ARCHIVES:
            open(os.path.join(release, name), 'w').close()
            os.makedirs(os.path.join(release, name[:-len('.zip')]))
        for name in CAMELS_COL._TABLES:
            open(os.path.join(release, name), 'w').close()
        # the cache of this release, and the one this class wrote before it
        # served float32: overwrite=True must remove both
        for cache in (_cache_name(), 'camels_col_D_v2.nc'):
            open(os.path.join(release, cache), 'w').close()
        # and the superseded release next to it
        old = [os.path.join(root, '02_CAMELS_COL_Catchment_information.xlsx'),
               os.path.join(root, '04_CAMELS_COL_Hydrometeorological_data.zip'),
               os.path.join(root, 'camels_col_D.nc')]
        for fpath in old:
            open(fpath, 'w').close()

        asked = {}

        def stub(outdir, doi=None, include=None, **kwargs):
            asked['include'] = list(include or [])
            raise _Downloaded

        zenodo = aqua_fetch.download_zenodo.download_from_zenodo
        aqua_fetch.download_zenodo.download_from_zenodo = stub
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    CAMELS_COL(path=tmp, verbosity=0, to_netcdf=False, overwrite=True)
                except _Downloaded:
                    pass
                else:
                    raise AssertionError("overwrite=True did not download again")
        finally:
            aqua_fetch.download_zenodo.download_from_zenodo = zenodo

        # everything of this release is gone and asked for again
        for name in CAMELS_COL._ARCHIVES:
            assert not os.path.exists(os.path.join(release, name)), name
            assert not os.path.exists(os.path.join(release, name[:-len('.zip')])), name
        for name in CAMELS_COL._TABLES:
            assert not os.path.exists(os.path.join(release, name)), name
        for cache in (_cache_name(), 'camels_col_D_v2.nc'):
            assert not os.path.exists(os.path.join(release, cache)), \
                f"{cache} survived overwrite=True"
        for name in CAMELS_COL._TABLES + CAMELS_COL._ARCHIVES:
            assert name in asked['include'], name

        # and nothing of the superseded release was touched
        for fpath in old:
            assert os.path.exists(fpath), fpath
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def _reader_on(tmp: str, rows: str) -> CAMELS_COL:
    """a CAMELS_COL which reads one hand written time series file"""
    dataset = CAMELS_COL.__new__(CAMELS_COL)      # no download, no __init__
    dataset.verbosity = 0
    dataset.path = tmp
    os.makedirs(dataset.ts_path)
    with open(dataset._ts_file('99999999'), 'w') as f:
        f.write("Fecha\tPrecipitacion\tETP_\tTemperatura_minima\t"
                "Temperatura_maxima\tCaudal\n")
        f.write(rows)
    dataset.__dict__['_time_extent'] = (pd.Timestamp('2000-01-01'),
                                        pd.Timestamp('2000-01-04'))
    return dataset


def _check_duplicated_dates_are_rejected():
    """
    A repeated date is raised on, not warned about: ``_read_stn_dyn`` runs in a
    process pool worker whose warnings never reach the caller, and serving one
    of the two rows would be a silent guess. The message names the file and the
    date.
    """
    logger.info("test_duplicated_dates_are_rejected")

    tmp = tempfile.mkdtemp()
    try:
        dataset = _reader_on(tmp,
                             "01/01/2000\t1\t1\t1\t1\t1\n"
                             "02/01/2000\t2\t2\t2\t2\t2\n"
                             "02/01/2000\t9\t9\t9\t9\t9\n"
                             "03/01/2000\t3\t3\t3\t3\t3\n")
        try:
            dataset._read_stn_dyn('99999999')
        except ValueError as err:
            assert 'duplicated dates' in str(err), err
            assert '2000-01-02' in str(err), err
            assert '99999999' in str(err), err
        else:
            raise AssertionError("a duplicated date was served")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def _check_days_outside_the_extent_are_rejected():
    """
    Padding to the common index would drop any day outside it, which only an
    unsorted file can produce, so it is raised on instead.
    """
    logger.info("test_days_outside_the_extent_are_rejected")

    tmp = tempfile.mkdtemp()
    try:
        dataset = _reader_on(tmp,
                             "01/01/2000\t1\t1\t1\t1\t1\n"
                             "01/01/1999\t7\t7\t7\t7\t7\n")
        try:
            dataset._read_stn_dyn('99999999')
        except ValueError as err:
            assert '1999-01-01' in str(err), err
            assert 'not sorted' in str(err), err
        else:
            raise AssertionError("a day outside the extent was dropped silently")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def _check_free_disk_space_spares_the_superseded_release():
    """
    ``free_disk_space`` walks the whole of ``path`` in the base class, which
    would delete the archives of the superseded release. Only this release's
    archives may be listed.
    """
    logger.info("test_free_disk_space_spares_the_superseded_release")

    tmp = tempfile.mkdtemp()
    try:
        root = os.path.join(tmp, 'CAMELS_COL')
        release = os.path.join(root, 'camels_col')
        os.makedirs(release)
        for name in CAMELS_COL._ARCHIVES:
            open(os.path.join(release, name), 'w').close()
            os.makedirs(os.path.join(release, name[:-len('.zip')]))
        # the superseded release: an archive next to its extracted folder, which
        # is exactly what the base class looks for
        old_archive = os.path.join(root, '04_CAMELS_COL_Hydrometeorological_data.zip')
        open(old_archive, 'w').close()
        os.makedirs(os.path.join(root, '04_CAMELS_COL_Hydrometeorological_data'))

        dataset = CAMELS_COL.__new__(CAMELS_COL)
        dataset.verbosity = 0
        dataset.path = tmp

        listed = dataset._archive_files()
        assert sorted(os.path.basename(f) for f in listed) == sorted(CAMELS_COL._ARCHIVES)
        for fpath in listed:
            assert os.path.commonpath([release, fpath]) == release, fpath
        assert old_archive not in listed, "the superseded release would be deleted"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def _check_pickling_drops_the_heavy_caches_but_stays_correct():
    """
    ``_read_dynamic`` hands this object to a process pool, so the big cached
    tables must not travel with it, and a worker must rebuild what it needs and
    read exactly what the serial path reads.
    """
    logger.info("test_pickling_drops_the_heavy_caches_but_stays_correct")
    _skip_without_data()
    import pickle

    dataset = _dataset()
    stn = dataset.stations()[0]
    serial = dataset._read_stn_dyn(stn)     # fills _daily_index and _time_extent
    dataset.static_features                 # fills _static_table

    state = dataset.__getstate__()
    for dropped in CAMELS_COL._NOT_PICKLED:
        assert dropped not in state, f"{dropped} is shipped to every worker"
    # but the cheap extent is kept, so a worker does not rescan the 346 files
    assert '_time_extent' in state, "the worker would rescan every file"

    revived = pickle.loads(pickle.dumps(dataset))
    assert len(pickle.dumps(dataset)) < 50_000, len(pickle.dumps(dataset))
    pd.testing.assert_frame_equal(revived._read_stn_dyn(stn), serial)
    return


def _check_a_truncated_table_is_reported_not_raised_on():
    """
    A release copied only in part (an interrupted rsync) must warn, not raise
    an opaque pandas error out of ``__init__``.
    """
    logger.info("test_a_truncated_table_is_reported_not_raised_on")
    _skip_without_data()

    tmp = tempfile.mkdtemp()
    try:
        release = os.path.join(tmp, 'CAMELS_COL', 'camels_col')
        os.makedirs(release)
        real = _dataset()._release_dir
        # link everything, then replace the first table with a zero byte file
        for name in os.listdir(real):
            os.symlink(os.path.join(real, name), os.path.join(release, name))
        os.unlink(os.path.join(release, CAMELS_COL._TABLES[0]))
        open(os.path.join(release, CAMELS_COL._TABLES[0]), 'w').close()

        dataset = CAMELS_COL.__new__(CAMELS_COL)
        dataset.verbosity = 0
        dataset.path = tmp
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dataset._check_manifest()       # must not raise
        messages = " ".join(str(w.message) for w in caught)
        assert 'cannot be read' in messages, messages
        assert 'overwrite=True' in messages, messages
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def _check_an_empty_time_series_folder_raises_a_clear_error():
    """
    With no time series file at all the period cannot be read; the message must
    say which folder is empty instead of ``min() arg is an empty sequence``.
    """
    logger.info("test_an_empty_time_series_folder_raises_a_clear_error")

    tmp = tempfile.mkdtemp()
    try:
        dataset = CAMELS_COL.__new__(CAMELS_COL)
        dataset.verbosity = 0
        dataset.path = tmp
        os.makedirs(dataset.ts_path)
        try:
            dataset.start
        except FileNotFoundError as err:
            assert 'Hydromet_data' in str(err), err
            assert 'overwrite=True' in str(err), err
        else:
            raise AssertionError("an empty time series folder was accepted")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def _check_an_empty_time_series_folder_is_reported_not_crashed_on():
    """an extraction that produced no file at all warns instead of raising."""
    logger.info("test_an_empty_time_series_folder_is_reported_not_crashed_on")
    _skip_without_data()

    dataset = CAMELS_COL.__new__(CAMELS_COL)
    dataset.verbosity = 0
    dataset.path = _camels_col_path()
    dataset.__dict__['_stn_ids'] = []          # nothing was extracted

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dataset._check_manifest()              # must not raise IndexError
    messages = " ".join(str(w.message) for w in caught)
    assert 'time series files' in messages, messages
    return


def _mirror(dataset: CAMELS_COL, tmp: str, drop=(), replace=None) -> CAMELS_COL:
    """
    A CAMELS_COL whose release folder links to the real files, so that a single
    file can be damaged without touching the original. ``drop`` names files to
    leave out, ``replace`` maps a name to the text to write in its place.
    """
    release = os.path.join(tmp, 'CAMELS_COL', 'camels_col')
    os.makedirs(release)
    replace = replace or {}
    for name in os.listdir(dataset._release_dir):
        if name in drop:
            continue
        os.symlink(os.path.join(dataset._release_dir, name), os.path.join(release, name))
    for name, text in replace.items():
        target = os.path.join(release, name)
        if os.path.lexists(target):
            os.unlink(target)
        with open(target, 'w') as f:
            f.write(text)

    mirror = CAMELS_COL.__new__(CAMELS_COL)      # no download, no __init__
    mirror.verbosity = 0
    mirror.name = CAMELS_COL.__name__            # normally set by the base __init__
    mirror.path = tmp
    return mirror


def _ts_mirror(dataset: CAMELS_COL, tmp: str, stn: str, text: str) -> CAMELS_COL:
    """a mirror in which the time series file of ``stn`` is hand written"""
    mirror = _mirror(dataset, tmp, drop=('04_CAMELS_COL_Hydrometeorological_data',))
    inner = os.path.join(mirror._release_dir, '04_CAMELS_COL_Hydrometeorological_data',
                         '3_Hydrometeorological_data')
    os.makedirs(inner)
    for name in os.listdir(dataset.ts_path):
        os.symlink(os.path.join(dataset.ts_path, name), os.path.join(inner, name))
    os.unlink(mirror._ts_file(stn))
    with open(mirror._ts_file(stn), 'w') as f:
        f.write("Fecha\tPrecipitacion\tETP_\tTemperatura_minima\t"
                "Temperatura_maxima\tCaudal\n")
        f.write(text)
    return mirror


def _check_time_extent_is_read_day_first():
    """
    ``_time_extent`` reads only the first and last row of each file, so it needs
    the date format just as much as ``_read_stn_dyn``: 37 of the 346 files begin
    on an ambiguous date.
    """
    logger.info("_check_time_extent_is_read_day_first")

    tmp = tempfile.mkdtemp()
    try:
        dataset = CAMELS_COL.__new__(CAMELS_COL)
        dataset.verbosity = 0
        dataset.path = tmp
        os.makedirs(dataset.ts_path)
        with open(dataset._ts_file('99999999'), 'w') as f:
            f.write("Fecha\tPrecipitacion\tETP_\tTemperatura_minima\t"
                    "Temperatura_maxima\tCaudal\n")
            f.write("01/03/1981\t1\t1\t1\t1\t1\n")   # 1 March, not 3 January
            f.write("12/11/1981\t2\t2\t2\t2\t2\n")   # 12 November, not 11 December
        assert dataset.start == pd.Timestamp('1981-03-01'), dataset.start
        assert dataset.end == pd.Timestamp('1981-11-12'), dataset.end
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def _check_an_unsorted_file_is_rejected():
    """the extent is read from two rows, so a file whose last date precedes its
    first would give a wrong extent and is rejected."""
    logger.info("_check_an_unsorted_file_is_rejected")

    tmp = tempfile.mkdtemp()
    try:
        dataset = CAMELS_COL.__new__(CAMELS_COL)
        dataset.verbosity = 0
        dataset.path = tmp
        os.makedirs(dataset.ts_path)
        with open(dataset._ts_file('99999999'), 'w') as f:
            f.write("Fecha\tPrecipitacion\tETP_\tTemperatura_minima\t"
                    "Temperatura_maxima\tCaudal\n")
            f.write("31/12/2000\t1\t1\t1\t1\t1\n")
            f.write("01/01/2000\t2\t2\t2\t2\t2\n")
        try:
            dataset.start
        except ValueError as err:
            assert 'not sorted' in str(err), err
        else:
            raise AssertionError("an unsorted file was accepted")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def _check_a_file_without_data_rows_is_rejected():
    """a header with no data row must say so, not fail on parsing 'Fecha'."""
    logger.info("_check_a_file_without_data_rows_is_rejected")

    tmp = tempfile.mkdtemp()
    try:
        dataset = CAMELS_COL.__new__(CAMELS_COL)
        dataset.verbosity = 0
        dataset.path = tmp
        os.makedirs(dataset.ts_path)
        with open(dataset._ts_file('99999999'), 'w') as f:
            f.write("Fecha\tPrecipitacion\tETP_\tTemperatura_minima\t"
                    "Temperatura_maxima\tCaudal\n")
        try:
            dataset.start
        except ValueError as err:
            assert 'no data row' in str(err), err
        else:
            raise AssertionError("a file without data rows was accepted")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def _check_stations_are_sorted_whatever_the_file_system_says():
    """
    ``stations()`` must not depend on the order the file system lists the
    files in. The real data sits on an NFS mount that happens to return sorted
    entries, so this is checked on files created out of order in a temp dir.
    """
    logger.info("_check_stations_are_sorted_whatever_the_file_system_says")

    tmp = tempfile.mkdtemp()
    try:
        dataset = CAMELS_COL.__new__(CAMELS_COL)
        dataset.verbosity = 0
        dataset.path = tmp
        os.makedirs(dataset.ts_path)
        scrambled = ['35067040', '11017010', '54077030', '21247040', '12017020']
        for stn in scrambled:
            open(dataset._ts_file(stn), 'w').close()
        assert dataset.stations() == sorted(scrambled), dataset.stations()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def _check_a_gauge_missing_from_any_table_is_reported():
    """
    A gauge absent from one attribute table keeps all its other attributes, so
    nothing else notices. It must still be reported rather than served with a
    block of NaN.
    """
    logger.info("_check_a_gauge_missing_from_any_table_is_reported")
    _skip_without_data()

    tmp = tempfile.mkdtemp()
    try:
        dataset = _dataset()
        table = '06_CAMELS_COL_Land_cover_characteristics.csv'
        lines = open(os.path.join(dataset._release_dir, table)).read().splitlines(True)
        dropped = lines[-1].split(';')[0]
        mirror = _mirror(dataset, tmp, replace={table: "".join(lines[:-1])})

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            static = mirror._static_table
        messages = " ".join(str(w.message) for w in caught)

        assert dropped in messages and table in messages, messages
        # the gauge is still served, with the missing table's columns as NaN
        assert dropped in static.index
        assert static.loc[dropped, 'forest_perc'] != static.loc[dropped, 'forest_perc']
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def _check_a_missing_release_file_is_reported():
    """one absent shapefile sidecar is named, and does not hide the gauge count
    check that follows it."""
    logger.info("_check_a_missing_release_file_is_reported")
    _skip_without_data()

    tmp = tempfile.mkdtemp()
    try:
        dataset = _dataset()
        mirror = _mirror(dataset, tmp, drop=('03_CAMELS_COL_Basin_boundary',))
        os.makedirs(os.path.join(mirror._release_dir, '03_CAMELS_COL_Basin_boundary',
                                 '03_CAMELS_COL_Basin_boundary'))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mirror._check_manifest()
        messages = " ".join(str(w.message) for w in caught)

        assert 'of the release are missing' in messages, messages
        assert '.prj' in messages and '.dbf' in messages, messages
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def _check_an_unknown_time_series_column_is_reported():
    """a column the class does not serve must be named, not dropped quietly."""
    logger.info("_check_an_unknown_time_series_column_is_reported")
    _skip_without_data()

    tmp = tempfile.mkdtemp()
    try:
        dataset = _dataset()
        first = dataset.stations()[0]
        mirror = _ts_mirror(dataset, tmp, first,
                            "01/01/2000\t1\t1\t1\t1\t1\t5\n")
        # add the extra column to the header the mirror wrote
        text = open(mirror._ts_file(first)).read().replace(
            "Caudal\n", "Caudal\tNuevo\n", 1)
        with open(mirror._ts_file(first), 'w') as f:
            f.write(text)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            mirror._check_manifest()
        messages = " ".join(str(w.message) for w in caught)
        assert 'Nuevo' in messages and 'does not serve' in messages, messages
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def _check_a_blank_cell_does_not_drop_a_gauge():
    """
    A gauge whose row has one blank cell keeps its other attributes. Dropping
    the whole row (``dropna(how='any')``) would lose a gauge without a word.
    """
    logger.info("_check_a_blank_cell_does_not_drop_a_gauge")
    _skip_without_data()

    tmp = tempfile.mkdtemp()
    try:
        dataset = _dataset()
        table = '08_CAMELS_COL_Climatic_indices.csv'
        lines = open(os.path.join(dataset._release_dir, table)).read().splitlines(True)
        fields = lines[1].rstrip('\n').split(';')
        blanked = fields[0]
        fields[1] = ''                                    # aridity is now blank
        lines[1] = ';'.join(fields) + '\n'
        mirror = _mirror(dataset, tmp, replace={table: "".join(lines)})

        table_df = _read_col_csv(mirror._release_file(table))
        assert len(table_df) == NUM_STATIONS, table_df.shape
        assert blanked in table_df.index, "a gauge was dropped for one blank cell"
        assert pd.isna(table_df.loc[blanked, 'aridity'])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def _check_a_gauge_without_a_time_series_file_is_dropped():
    """
    A gauge listed in the attribute tables but with no time series file cannot
    be served, so it is reported and left out rather than returned with an
    all-NaN record.
    """
    logger.info("_check_a_gauge_without_a_time_series_file_is_dropped")
    _skip_without_data()

    tmp = tempfile.mkdtemp()
    try:
        dataset = _dataset()
        table = CAMELS_COL._TABLES[0]
        text = open(os.path.join(dataset._release_dir, table), encoding='latin-1').read()
        ghost = '99999999'
        extra_row = ghost + ';Tolima;519409.28;-8344301.85;522;1/01/1988;28/02/2003;5119;7.566\n'
        mirror = _mirror(dataset, tmp, replace={table: text.rstrip('\n') + '\n' + extra_row})

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            static = mirror._static_table
        messages = " ".join(str(w.message) for w in caught)

        assert ghost in messages, messages
        assert static.index.tolist() == dataset.stations(), \
            "a gauge with no time series file was served"
        assert len(static) == NUM_STATIONS, static.shape
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def _check_an_empty_time_series_file_is_rejected():
    """
    A zero byte file made the shared row reader raise a bare ``IndexError``,
    and a blank line after the header made the guard claim the file had no data
    when it had. Both must give the same clear message, for any gauge, not only
    the first.
    """
    logger.info("_check_an_empty_time_series_file_is_rejected")

    header = ("Fecha\tPrecipitacion\tETP_\tTemperatura_minima\t"
              "Temperatura_maxima\tCaudal\n")
    rows = "01/01/2000\t1\t1\t1\t1\t1\n31/12/2000\t2\t2\t2\t2\t2\n"

    for label, text, should_raise in [
        ('zero byte', "", True),
        ('header only', header, True),
        ('blank line after the header', header + "\n" + rows, False),
    ]:
        tmp = tempfile.mkdtemp()
        try:
            dataset = CAMELS_COL.__new__(CAMELS_COL)
            dataset.verbosity = 0
            dataset.path = tmp
            os.makedirs(dataset.ts_path)
            # two gauges, so the damaged one is not stations()[0] either
            with open(dataset._ts_file('11111111'), 'w') as f:
                f.write(header + rows)
            with open(dataset._ts_file('99999999'), 'w') as f:
                f.write(text)

            if should_raise:
                try:
                    dataset.start
                except ValueError as err:
                    assert 'no data row' in str(err), (label, err)
                    assert '99999999' in str(err), (label, err)
                else:
                    raise AssertionError(f"{label}: an empty file was accepted")
            else:
                # the file does hold data, so it must not be rejected
                assert dataset.start == pd.Timestamp('2000-01-01'), (label, dataset.start)
                assert dataset.end == pd.Timestamp('2000-12-31'), (label, dataset.end)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return


def _check_a_gauge_listed_twice_is_rejected():
    """
    A gauge with two rows in one table would make ``pd.concat`` raise
    ``InvalidIndexError``, which names neither the table nor the gauge.
    """
    logger.info("_check_a_gauge_listed_twice_is_rejected")
    _skip_without_data()

    tmp = tempfile.mkdtemp()
    try:
        dataset = _dataset()
        table = '08_CAMELS_COL_Climatic_indices.csv'
        lines = open(os.path.join(dataset._release_dir, table)).read().splitlines(True)
        repeated = lines[1].split(';')[0]
        mirror = _mirror(dataset, tmp, replace={table: "".join(lines) + lines[1]})
        try:
            mirror._static_table
        except ValueError as err:
            assert repeated in str(err), err
            assert table in str(err), err
            assert 'more than once' in str(err), err
        else:
            raise AssertionError("a gauge listed twice was accepted")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def _check_the_superseded_warning_counts_the_extracted_folders():
    """
    The 2025 release left extracted folders next to its archives. They are the
    bulk of the space, so the warning must count and name them, and it must say
    that the deletion cannot be undone.
    """
    logger.info("_check_the_superseded_warning_counts_the_extracted_folders")

    tmp = tempfile.mkdtemp()
    try:
        root = os.path.join(tmp, 'CAMELS_COL')
        os.makedirs(root)
        open(os.path.join(root, '02_CAMELS_COL_Catchment_information.xlsx'), 'w').close()
        # an extracted folder of the old release, far bigger than the loose files
        folder = os.path.join(root, '04_CAMELS_COL_Hydrometeorological_data')
        os.makedirs(folder)
        with open(os.path.join(folder, 'Hydromet_data_11017010.txt.txt'), 'w') as f:
            f.write('x' * 200_000)

        dataset = CAMELS_COL.__new__(CAMELS_COL)
        dataset.verbosity = 0
        dataset.path = tmp
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dataset._warn_superseded_files()
        messages = " ".join(str(w.message) for w in caught)

        assert '2 files and folders' in messages, messages
        assert '04_CAMELS_COL_Hydrometeorological_data' in messages, messages
        assert 'cannot be undone' in messages, messages
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def _check_cache_and_paths_live_in_the_release_folder():
    """the netCDF cache and every file the class reads are inside the release
    sub-folder, so a cache of the superseded release in ``path`` is never read
    as this one."""
    logger.info("test_cache_and_paths_live_in_the_release_folder")
    _skip_without_data()
    dataset = _dataset()

    release = dataset._release_dir
    assert os.path.basename(release) == 'camels_col'
    for path in (dataset.dyn_fpath, dataset.ts_path, dataset.boundary_file,
                 dataset._ts_file(dataset.stations()[0])):
        assert os.path.commonpath([release, path]) == release, path
    return


def _check_an_interrupted_extraction_is_redone():
    """a half extracted folder is never taken for a complete one: the archive
    is unpacked into a temporary folder which is renamed only once done."""
    logger.info("test_an_interrupted_extraction_is_redone")

    tmp = tempfile.mkdtemp()
    try:
        release = os.path.join(tmp, 'CAMELS_COL', 'camels_col')
        os.makedirs(release)
        archive = os.path.join(release, '01_CAMELS_COL_Attributes.zip')
        with zipfile.ZipFile(archive, 'w') as zf:
            zf.writestr('01_CAMELS_COL_Attributes/readme.txt', 'hello')

        folder = archive[:-len('.zip')]
        # what an extraction interrupted half way leaves behind
        os.makedirs(f"{folder}_extracting")
        open(os.path.join(f"{folder}_extracting", 'partial.txt'), 'w').close()

        _extract_zip(archive, folder, verbosity=0)

        assert os.path.exists(os.path.join(folder, '01_CAMELS_COL_Attributes', 'readme.txt'))
        assert not os.path.exists(f"{folder}_extracting"), "the temporary folder was left"
        assert not os.path.exists(os.path.join(folder, 'partial.txt')), \
            "the half extracted folder was kept"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def _check_a_corrupt_archive_is_reported_and_deleted():
    """an error page saved as an archive is deleted with a clear message
    instead of being extracted again on every initialization."""
    logger.info("test_a_corrupt_archive_is_reported_and_deleted")

    tmp = tempfile.mkdtemp()
    try:
        archive = os.path.join(tmp, '01_CAMELS_COL_Attributes.zip')
        with open(archive, 'w') as f:
            f.write("<html>404</html>")

        try:
            _extract_zip(archive, archive[:-len('.zip')], verbosity=0)
        except ValueError as err:
            assert 'corrupt' in str(err), err
        else:
            raise AssertionError("a corrupt archive was accepted")
        assert not os.path.exists(archive), "the corrupt archive was kept"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def _check_remove_zip_keeps_the_old_release():
    """
    ``remove_zip`` deletes an archive of this release once its folder is
    extracted. It never touches the archives of the superseded release next to
    them, and never an archive whose folder is not there -- that one would
    have to be downloaded again.
    """
    logger.info("test_remove_zip_keeps_the_old_release")

    tmp = tempfile.mkdtemp()
    try:
        root = os.path.join(tmp, 'CAMELS_COL')
        release = os.path.join(root, 'camels_col')
        os.makedirs(release)
        *extracted, not_extracted = CAMELS_COL._ARCHIVES
        for name in CAMELS_COL._ARCHIVES:
            open(os.path.join(release, name), 'w').close()
        for name in extracted:
            os.makedirs(os.path.join(release, name[:-len('.zip')]))
        old_archive = os.path.join(root, '04_CAMELS_COL_Hydrometeorological_data.zip')
        open(old_archive, 'w').close()

        dataset = CAMELS_COL.__new__(CAMELS_COL)
        dataset.verbosity = 0
        dataset.path = tmp          # the setter appends the class name
        assert dataset.path == root, dataset.path
        dataset.remove_zip_files()

        for name in extracted:
            assert not os.path.exists(os.path.join(release, name)), name
        assert os.path.exists(os.path.join(release, not_extracted)), \
            "an archive which is not extracted yet was deleted"
        assert os.path.exists(old_archive), "the archive of the old release was deleted"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


class TestCamelsCol(unittest.TestCase):

    def test_world_mercator_matches_pyproj(self):
        _check_world_mercator_matches_pyproj()

    def test_spherical_inverse_is_not_accurate_enough(self):
        _check_spherical_inverse_is_not_accurate_enough()

    def test_stn_coords_are_degrees(self):
        _check_stn_coords_are_degrees()

    def test_boundary_is_transformed_to_wgs84(self):
        _check_boundary_is_transformed_to_wgs84()

    def test_counts(self):
        _check_counts()

    def test_temporal_extent_is_derived(self):
        _check_temporal_extent_is_derived()

    def test_stations_and_attributes_agree(self):
        _check_stations_and_attributes_agree()

    def test_attribute_names_are_unique(self):
        _check_attribute_names_are_unique()

    def test_dynamic_values_and_units_unchanged(self):
        _check_dynamic_values_and_units_unchanged()

    def test_read_stn_dyn_returns_the_raw_values(self):
        _check_read_stn_dyn_returns_the_raw_values()

    def test_the_float32_downcast_keeps_the_source_precision(self):
        _check_the_float32_downcast_keeps_the_source_precision()

    def test_the_cache_name_carries_the_release_and_the_precision(self):
        _check_the_cache_name_carries_the_release_and_the_precision()

    def test_every_gauge_is_padded_to_the_common_index(self):
        _check_every_gauge_is_padded_to_the_common_index()

    def test_dates_are_read_day_first(self):
        _check_dates_are_read_day_first()

    def test_a_truncated_release_is_reported(self):
        _check_a_truncated_release_is_reported()

    def test_static_values_and_units_unchanged(self):
        _check_static_values_and_units_unchanged()

    def test_land_use_capability_table(self):
        _check_land_use_capability_table()

    def test_static_data_and_stations_return_copies(self):
        _check_static_data_and_stations_return_copies()

    def test_duplicate_gauges_are_warned_about(self):
        _check_duplicate_gauges_are_warned_about()

    def test_no_redownload_or_reextraction(self):
        _check_no_redownload_or_reextraction()

    def test_an_extracted_release_without_its_archives_is_not_downloaded_again(self):
        _check_an_extracted_release_without_its_archives_is_not_downloaded_again()

    def test_near_duplicate_positions_are_warned_about(self):
        _check_near_duplicate_positions_are_warned_about()

    def test_files_of_the_superseded_release_do_not_block_the_download(self):
        _check_files_of_the_superseded_release_do_not_block_the_download()

    def test_overwrite_downloads_a_genuinely_fresh_copy(self):
        _check_overwrite_downloads_a_genuinely_fresh_copy()

    def test_duplicated_dates_are_rejected(self):
        _check_duplicated_dates_are_rejected()

    def test_days_outside_the_extent_are_rejected(self):
        _check_days_outside_the_extent_are_rejected()

    def test_free_disk_space_spares_the_superseded_release(self):
        _check_free_disk_space_spares_the_superseded_release()

    def test_pickling_drops_the_heavy_caches_but_stays_correct(self):
        _check_pickling_drops_the_heavy_caches_but_stays_correct()

    def test_a_truncated_table_is_reported_not_raised_on(self):
        _check_a_truncated_table_is_reported_not_raised_on()

    def test_an_empty_time_series_folder_raises_a_clear_error(self):
        _check_an_empty_time_series_folder_raises_a_clear_error()

    def test_an_empty_time_series_folder_is_reported_not_crashed_on(self):
        _check_an_empty_time_series_folder_is_reported_not_crashed_on()

    def test_time_extent_is_read_day_first(self):
        _check_time_extent_is_read_day_first()

    def test_an_unsorted_file_is_rejected(self):
        _check_an_unsorted_file_is_rejected()

    def test_a_file_without_data_rows_is_rejected(self):
        _check_a_file_without_data_rows_is_rejected()

    def test_stations_are_sorted_whatever_the_file_system_says(self):
        _check_stations_are_sorted_whatever_the_file_system_says()

    def test_a_gauge_missing_from_any_table_is_reported(self):
        _check_a_gauge_missing_from_any_table_is_reported()

    def test_a_missing_release_file_is_reported(self):
        _check_a_missing_release_file_is_reported()

    def test_an_unknown_time_series_column_is_reported(self):
        _check_an_unknown_time_series_column_is_reported()

    def test_a_blank_cell_does_not_drop_a_gauge(self):
        _check_a_blank_cell_does_not_drop_a_gauge()

    def test_a_gauge_without_a_time_series_file_is_dropped(self):
        _check_a_gauge_without_a_time_series_file_is_dropped()

    def test_an_empty_time_series_file_is_rejected(self):
        _check_an_empty_time_series_file_is_rejected()

    def test_a_gauge_listed_twice_is_rejected(self):
        _check_a_gauge_listed_twice_is_rejected()

    def test_the_superseded_warning_counts_the_extracted_folders(self):
        _check_the_superseded_warning_counts_the_extracted_folders()

    def test_cache_and_paths_live_in_the_release_folder(self):
        _check_cache_and_paths_live_in_the_release_folder()

    def test_an_interrupted_extraction_is_redone(self):
        _check_an_interrupted_extraction_is_redone()

    def test_a_corrupt_archive_is_reported_and_deleted(self):
        _check_a_corrupt_archive_is_reported_and_deleted()

    def test_remove_zip_keeps_the_old_release(self):
        _check_remove_zip_keeps_the_old_release()


if __name__ == "__main__":
    unittest.main()
