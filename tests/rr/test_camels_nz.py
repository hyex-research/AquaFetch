"""
Tests for CAMELS_NZ (New Zealand, 369 catchments), release 5, at the daily and
the hourly timestep. They check that the class

    * serves the values and the dates of the published files unchanged, with the
      one documented conversion (air temperature Kelvin -> degree Celsius),
    * serves release 5, i.e. daily potential evapotranspiration as a daily total
      and an hourly index without the gaps and repeats that the local-time
      stamps of release 2 had,
    * reads the stations, the features and the time period from the files
      instead of from hardcoded literals,
    * reads the attributes, coordinates, areas and boundaries faithfully,
    * downloads only the archives of its timestep, once, and neither downloads
      nor extracts again, survives an interrupted extraction, and deletes the
      stale files first on ``overwrite=True``,
    * reports, and does not delete, a release 2 left in ``path``,
    * warns instead of silently serving a truncated dataset,
    * and passes the generic rainfall-runoff suite.

Set ``raw_data_path`` and run with pytest, or as a script. The first run
downloads release 5: 315 MB for the daily and 4.9 GB for the hourly data.
"""

import os
import copy
import site
import time
import random
import shutil
import zipfile
import logging
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

# add the repository root to the path so that ``aqua_fetch`` and the shared
# ``utils`` test helpers can be imported
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

from aqua_fetch import CAMELS_NZ
from aqua_fetch.rr import _camels
from aqua_fetch.rr.utils import cache_name
from aqua_fetch._backend import xarray as xr, fiona

# the generic rainfall-runoff suite; imported under another name so that pytest
# does not collect it as a test of this module
from utils import test_dataset as run_generic_suite

if __name__ == "__main__":
    logging.basicConfig(filename='test_camels_nz.log', filemode='w', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

raw_data_path = '/path/to/raw/data'  # replace with actual path
NZ_PATH = os.path.join(raw_data_path, 'CAMELS')

NUM_STATIONS = 369
NUM_STATIC = 37
NUM_DYNAMIC = 5
DAILY_STEPS = 19208       # 1972-01-01 .. 2024-08-02
HOURLY_STEPS = 460978     # 1972-01-01 00:00 .. 2024-08-02 09:00

# raw column of the published files -> served name and the factor between them.
# Written out here rather than taken from the class, so that a wrong mapping or
# a forgotten conversion is caught.
RAW_TO_STD = {
    'PET': 'pet_mm',
    'precipitation': 'pcp_mm',
    'Relative_humidity': 'rh_%',
    'temperature': 'airtemp_C_mean',
    'flow': 'q_cms_obs',
}

ATTRIBUTE_FILES = ("1.CAMELS_NZ_Catchment_information.csv",
                   "2.CAMELS_NZ_Climatic_attribute.csv",
                   "3.CAMELS_NZ_Landcover_attribute.csv",
                   "4.CAMELS_NZ_Geology.csv",
                   "5.CAMELS_NZ_Anthropogenic_attribute.csv")

# the gauges whose streamflow needs the owner's permission, from 1.Readme.txt of
# the streamflow archive of the release
RESTRICTED = ('75253', '75261', '75265', '75276', '75294', '15408', '15410',
              '15453', '33356', '52916', '74318', '74321', '74368', '1114629')

# release 5 downloads here, once per timestep
dataset = CAMELS_NZ(path=NZ_PATH, verbosity=0)
dataset_h = CAMELS_NZ(path=NZ_PATH, timestep='H', verbosity=0)

TIMESTEPS = pytest.mark.parametrize(
    "ds, n_steps", [(dataset, DAILY_STEPS), (dataset_h, HOURLY_STEPS)], ids=['D', 'H'])


def _sample(seq, k: int) -> list:
    """the same random stations whichever tests run before"""
    return random.Random(313).sample(sorted(seq), k)


# ---------------------------------------------------------------------------
# independent readers of the published files
# ---------------------------------------------------------------------------

def raw_series(ds: CAMELS_NZ, stn: str, variable: str) -> pd.Series:
    """
    One published file, read without the class: the dates are parsed by pandas
    (day first where the file uses slashes, which the test verifies separately)
    and the values are kept at full precision.
    """
    fpath = ds._stn_file(stn, variable)
    df = pd.read_csv(fpath, index_col=0, na_values=['NA  '])
    dates = df.index.astype(str)
    # pandas decides, not the class: day first where the file uses slashes
    df.index = (pd.to_datetime(dates, dayfirst=True) if '/' in dates[0]
                else pd.to_datetime(dates, format='ISO8601'))
    return df[variable]


def raw_attributes(ds: CAMELS_NZ, fname: str) -> pd.DataFrame:
    """one attribute file read independently of the class"""
    return pd.read_csv(os.path.join(ds.static_path, fname), index_col=0, dtype={0: str})


def _copy_for(path: str, ds: CAMELS_NZ = None) -> CAMELS_NZ:
    """
    A copy of the dataset which points at ``path`` instead of the downloaded
    data, so that the download code can be exercised without the network and
    without touching the real data.
    """
    copied = copy.copy(ds or dataset)
    copied._path = path
    copied._stns = None
    copied._static_feats = None
    copied._var_dirs_ = None
    copied._period_ = None
    return copied


def _fake_download(calls):
    """
    A stand-in for :func:`aqua_fetch.rr._camels.download` which records its
    calls and writes an archive holding one csv, so that extraction has
    something to do.
    """
    def download(url, outdir=None, fname=None, verbosity=1, **kwargs):
        calls.append(fname)
        with zipfile.ZipFile(os.path.join(outdir, fname), 'w') as zf:
            zf.writestr('a.csv', '"time","flow"\n1972-01-01,1.0\n')
    return download


def _release_2(path: str) -> dict:
    """
    Lays down a release 2 the way an earlier version of this class left it: the
    archive, the folder it was extracted into and the netCDF caches built from
    it. Returns the paths it created.
    """
    created = {'folder': os.path.join(path, 'camels_nz'),
               'archive': os.path.join(path, 'camels_nz.zip'),
               'cache_d': os.path.join(path, 'camels_nz_D.nc'),
               'cache_h': os.path.join(path, 'camels_nz_H.nc'),
               'cache_d_v2': os.path.join(path, cache_name('camels_nz_D.nc'))}
    os.makedirs(created['folder'])
    for key in ('archive', 'cache_d', 'cache_h', 'cache_d_v2'):
        open(created[key], 'w').close()
    return created


# ---------------------------------------------------------------------------
# contents
# ---------------------------------------------------------------------------

def test_stations_and_features():
    """the release's own 369 gauges, 5 dynamic and 37 static features"""
    logger.info("test_stations_and_features")
    stations = dataset.stations()
    assert len(stations) == NUM_STATIONS, len(stations)
    assert len(set(stations)) == NUM_STATIONS, "a station is listed twice"

    # the gauges of the catchment information file, not whatever csv files are
    # on disk
    expected = set(raw_attributes(dataset, ATTRIBUTE_FILES[0]).index)
    assert set(stations) == expected, expected.symmetric_difference(stations)

    assert dataset.dynamic_features == list(RAW_TO_STD.values()), dataset.dynamic_features
    assert dataset_h.dynamic_features == list(RAW_TO_STD.values())

    static = dataset.static_features
    assert len(static) == NUM_STATIC, static
    # the coordinates are served under their canonical names; before release 5
    # the mapping used the spelling of the other four attribute files and
    # stn_coords() raised
    assert 'lat' in static and 'long' in static, static
    assert 'area_km2' in static and 'elev_gauge_m' in static and 'slope_degrees' in static
    # a copy, so that a caller cannot corrupt the cached lists
    static.append('nonsense')
    dataset.stations().append('nonsense')
    assert len(dataset.static_features) == NUM_STATIC
    assert len(dataset.stations()) == NUM_STATIONS
    return


@TIMESTEPS
def test_dynamic_values_and_units(ds, n_steps):
    """every served value is the published one, on the published timestamp"""
    logger.info(f"test_dynamic_values_and_units {ds.timestep}")
    stations = _sample(ds.stations(), 3) + ['74321', '57521']

    for stn in stations:
        _, dyn = ds.fetch(stations=stn, as_dataframe=True)
        served = dyn[stn]

        assert served.shape == (n_steps, NUM_DYNAMIC), (stn, served.shape)
        # the union of the records, not a range built from one file
        assert served.index.equals(
            pd.date_range(ds.start, ds.end, freq='D' if ds.timestep == 'D' else 'h')), stn

        for variable, name in RAW_TO_STD.items():
            if variable == 'flow' and stn in RESTRICTED:
                assert served[name].isna().all(), f"{stn} needs permission"
                continue
            raw = raw_series(ds, stn, variable)
            expected = raw.reindex(served.index).astype('float64')
            if variable == 'temperature':
                expected -= 273.15       # published in Kelvin
            np.testing.assert_allclose(
                served[name].values.astype('float64'), expected.values,
                rtol=1e-6, atol=0, equal_nan=True,
                err_msg=f"{stn}: {name} is not the published {variable}")
    return


def test_daily_pet_is_a_daily_total():
    """
    Release 2 published the daily potential evapotranspiration as a mean hourly
    rate, 24 times too small (~36 instead of ~870 mm a year). Release 5 gives a
    daily total, which must be what is served.
    """
    logger.info("test_daily_pet_is_a_daily_total")
    for stn in _sample(dataset.stations(), 5):
        _, dyn = dataset.fetch(stations=stn, as_dataframe=True)
        pet = dyn[stn]['pet_mm'].dropna()
        yearly = pet.resample('YE').sum(min_count=365).dropna()
        assert 300 < yearly.mean() < 2000, \
            f"{stn}: {yearly.mean():.0f} mm/year is not a daily total"

        # and it is exactly the published value, not a rescaling by this class
        np.testing.assert_allclose(
            pet.values.astype('float64'),
            raw_series(dataset, stn, 'PET').reindex(pet.index).values,
            rtol=1e-6, equal_nan=True)
    return


def test_daily_pet_is_the_sum_of_the_hourly_one():
    """the daily file of release 5 is the sum of the 24 hourly values"""
    logger.info("test_daily_pet_is_the_sum_of_the_hourly_one")
    stn = _sample(dataset.stations(), 1)[0]
    hourly = raw_series(dataset_h, stn, 'PET')
    daily = raw_series(dataset, stn, 'PET')

    summed = hourly.resample('D').sum(min_count=24).dropna()
    common = daily.index.intersection(summed.index)
    assert len(common) > 19000, len(common)
    np.testing.assert_allclose(daily.loc[common].values, summed.loc[common].values,
                               rtol=1e-9, atol=1e-9)
    return


@TIMESTEPS
def test_time_axes_are_uniform(ds, n_steps):
    """
    Every file of a variable has the same first and last timestamp, which is
    what lets :attr:`_period` read one file per variable instead of all 1845 of
    them. Reading the first and the last row of every file is cheap.
    """
    logger.info(f"test_time_axes_are_uniform {ds.timestep}")
    bounds = {}
    for variable in ds._variables:
        seen = set()
        for stn in ds.stations():
            if variable == 'flow' and stn in RESTRICTED:
                continue
            first, last = _camels._first_and_last_row(ds._stn_file(stn, variable))
            seen.add((ds._timestamp(first), ds._timestamp(last)))
        assert len(seen) == 1, f"{variable}: files with different time axes: {sorted(seen)}"
        bounds[variable] = seen.pop()

    # the streamflow and the meteorology cover different periods
    assert bounds['flow'] != bounds['PET'], bounds
    # and start/end is their union, read from the files
    assert ds.start == min(b[0] for b in bounds.values()), (ds.start, bounds)
    assert ds.end == max(b[1] for b in bounds.values()), (ds.end, bounds)
    assert len(pd.date_range(ds.start, ds.end,
                             freq='D' if ds.timestep == 'D' else 'h')) == n_steps
    return


def test_hourly_index_has_no_gaps_or_repeats():
    """
    The hourly timestamps of release 2 were New Zealand local time, which left
    50 hours missing and 50 repeated per station at the daylight saving
    switches. Release 5 is a regular hourly index.
    """
    logger.info("test_hourly_index_has_no_gaps_or_repeats")
    for stn in _sample(dataset_h.stations(), 2):
        for variable in dataset_h._variables:
            if variable == 'flow' and stn in RESTRICTED:
                continue
            index = raw_series(dataset_h, stn, variable).index
            assert not index.has_duplicates, f"{stn} {variable}: repeated timestamps"
            assert index.equals(pd.date_range(index[0], index[-1], freq='h')), \
                f"{stn} {variable}: missing hours"
    return


@TIMESTEPS
def test_restricted_gauges_are_nan(ds, n_steps):
    """
    The streamflow of the 14 gauges which need the owner's permission is not in
    the release; their files hold blank rows and the class serves NaN, without
    dropping the gauge or its meteorology.
    """
    logger.info(f"test_restricted_gauges_are_nan {ds.timestep}")
    assert set(ds._nodata_stns) == set(RESTRICTED), set(ds._nodata_stns) ^ set(RESTRICTED)
    assert set(RESTRICTED).issubset(ds.stations())

    for stn in (RESTRICTED[0], RESTRICTED[-1]):
        _, dyn = ds.fetch(stations=stn, as_dataframe=True)
        served = dyn[stn]
        assert served.shape == (n_steps, NUM_DYNAMIC)
        assert served['q_cms_obs'].isna().all()
        assert served['pcp_mm'].notna().any(), f"{stn} lost its meteorology"

        # the published file indeed holds no values
        with open(ds._stn_file(stn, 'flow')) as f:
            assert not [row for row in f.read().splitlines() if row.strip(', ')]

    # and a gauge which is not restricted does have streamflow
    other = [stn for stn in ds.stations() if stn not in RESTRICTED][0]
    _, dyn = ds.fetch(stations=other, as_dataframe=True)
    assert dyn[other]['q_cms_obs'].notna().any(), other
    return


@TIMESTEPS
def test_slash_dates_are_day_first(ds, n_steps):
    """
    A few streamflow files were exported with day-first slashes
    (``13/01/1972``) instead of ISO dates. They must be read day first, which a
    day above 12 proves, and land on the same axis as the other gauges.
    """
    logger.info(f"test_slash_dates_are_day_first {ds.timestep}")
    slashed = []
    for stn in ds.stations():
        if stn in RESTRICTED:
            continue
        fpath = ds._stn_file(stn, 'flow')
        first, _ = _camels._first_and_last_row(fpath)
        if b'/' in first:
            slashed.append(stn)

    assert slashed, "no file with slash dates; the special case is untested"
    logger.info(f"slash dates in {slashed}")

    reference = None
    for stn in ds.stations():
        if stn not in RESTRICTED and stn not in slashed:
            reference = raw_series(ds, stn, 'flow').index
            break

    for stn in slashed:
        with open(ds._stn_file(stn, 'flow')) as f:
            days = {row.split(',')[0].strip('"').split('/')[0] for row in f.read().splitlines()[1:]}
        assert any(day.isdigit() and int(day) > 12 for day in days), \
            f"{stn}: no day above 12, the format is ambiguous"

        _, dyn = ds.fetch(stations=stn, as_dataframe=True)
        served = dyn[stn]['q_cms_obs'].dropna()
        assert served.index.is_monotonic_increasing
        assert served.index.isin(reference).all(), f"{stn} is not on the common axis"
        np.testing.assert_allclose(
            served.values.astype('float64'),
            raw_series(ds, stn, 'flow').reindex(served.index).values, rtol=1e-6)
    return


def test_static_values():
    """the attributes are the published ones, joined without losing a column"""
    logger.info("test_static_values")
    static = dataset.fetch_static_features(stations='all', static_features='all')
    assert static.shape == (NUM_STATIONS, NUM_STATIC), static.shape

    info = raw_attributes(dataset, ATTRIBUTE_FILES[0])
    renamed = {'Latitude (WGS 84)': 'lat', 'Longitude(WGS 84)': 'long', 'uparea': 'area_km2',
               'elevation': 'elev_gauge_m', 'usAveSlope': 'slope_degrees'}
    for column in info.columns:
        served = static[renamed.get(column, column)]
        expected = info[column]
        if pd.api.types.is_numeric_dtype(expected):
            np.testing.assert_allclose(served.astype('float64').values,
                                       expected.astype('float64').values,
                                       rtol=1e-12, equal_nan=True, err_msg=column)
        else:
            assert served.tolist() == expected.tolist(), column

    # the identity columns which the other four files repeat are dropped: the
    # name and the coordinates come from the catchment information file, and
    # RID appears once instead of five times
    assert not static.columns.duplicated().any(), \
        static.columns[static.columns.duplicated()].tolist()
    assert list(static.columns).count('RID') == 1
    for column in ('StationName', 'latitude', 'longitude'):
        assert column not in static.columns, f"{column} is repeated"

    # and every attribute of those files is kept
    for fname in ATTRIBUTE_FILES[1:]:
        for column in raw_attributes(dataset, fname).columns:
            if column not in ('RID', 'StationName', 'latitude', 'longitude'):
                assert column in static.columns, f"{fname}: {column} is missing"
    return


def test_coordinates_and_area():
    """coordinates in New Zealand and areas in km2, from the published file"""
    logger.info("test_coordinates_and_area")
    coords = dataset.stn_coords()
    assert coords.shape == (NUM_STATIONS, 2), coords.shape
    assert coords.columns.tolist() == ['lat', 'long'], coords.columns.tolist()
    assert coords['lat'].between(-47.5, -34).all(), coords['lat'].describe()
    assert coords['long'].between(166, 179).all(), coords['long'].describe()

    info = raw_attributes(dataset, ATTRIBUTE_FILES[0])
    np.testing.assert_allclose(coords['lat'].astype('float64').values,
                               info['Latitude (WGS 84)'].values, rtol=1e-6)
    np.testing.assert_allclose(coords['long'].astype('float64').values,
                               info['Longitude(WGS 84)'].values, rtol=1e-6)

    area = dataset.area()
    np.testing.assert_allclose(area.astype('float64').values, info['uparea'].values, rtol=1e-6)
    assert 0.3 < area.min() and area.max() < 10000, (area.min(), area.max())
    return


def test_boundary():
    """
    The boundaries are New Zealand Map Grid (EPSG:27200) metres and are served
    as published; this class does not reproject them.
    """
    logger.info("test_boundary")
    if fiona is None:
        logger.info("fiona is not installed, skipped")
        return
    boundary = dataset.get_boundary('74321')
    assert boundary.type in ('Polygon', 'MultiPolygon'), boundary.type
    xy = np.asarray(boundary.coordinates[0][0], dtype='float64')
    assert xy.min() > 1000, "the boundaries look like degrees, not NZMG metres"
    return


@TIMESTEPS
def test_cache_matches_csv(ds, n_steps):
    """the netCDF cache carries the release in its name and serves the csv values"""
    logger.info(f"test_cache_matches_csv {ds.timestep}")
    assert f"_v{ds.release}_" in os.path.basename(ds.dyn_fpath), ds.dyn_fpath
    if xr is None or not ds.dyn_fpath_exists:
        logger.info("no netCDF cache, skipped")
        return

    for stn in _sample(ds.stations(), 2):
        _, from_cache = ds.fetch(stations=stn, as_dataframe=True)
        for variable, name in RAW_TO_STD.items():
            if variable == 'flow' and stn in RESTRICTED:
                continue
            expected = raw_series(ds, stn, variable).reindex(from_cache[stn].index).astype('float64')
            if variable == 'temperature':
                expected -= 273.15
            np.testing.assert_allclose(
                from_cache[stn][name].values.astype('float64'), expected.values,
                rtol=1e-6, equal_nan=True, err_msg=f"{stn}: {name} in the cache")
    return


def test_fetch_all_stations():
    """all 369 stations of the daily data are fetched in a few seconds"""
    logger.info("test_fetch_all_stations")
    start = time.time()
    static, dyn = dataset.fetch(stations='all', static_features='all', as_dataframe=True)
    taken = time.time() - start

    assert static.shape == (NUM_STATIONS, NUM_STATIC), static.shape
    assert len(dyn) == NUM_STATIONS, len(dyn)
    assert {df.shape for df in dyn.values()} == {(DAILY_STEPS, NUM_DYNAMIC)}
    logger.info(f"fetched all {NUM_STATIONS} stations in {taken:.1f} s")
    assert taken < 300, f"fetching all stations took {taken:.0f} s"
    return


@TIMESTEPS
def test_shared(ds, n_steps):
    """the checks every rainfall-runoff dataset has to pass"""
    logger.info(f"test_shared {ds.timestep}")
    # test_latlong_ranges=False: the boundaries are NZMG metres, not WGS84
    run_generic_suite(ds, NUM_STATIONS, n_steps, NUM_STATIC, NUM_DYNAMIC,
                      yearly_steps=366 if ds.timestep == 'D' else 8760,
                      test_latlong_ranges=False,
                      dyn_fraction=0.1 if ds.timestep == 'D' else 0.02)
    return


# ---------------------------------------------------------------------------
# download, extraction and the layout on disk
# ---------------------------------------------------------------------------

def test_timestep_is_validated():
    """an unknown timestep is refused instead of quietly giving the daily data"""
    logger.info("test_timestep_is_validated")
    with pytest.raises(ValueError, match='timestep'):
        CAMELS_NZ(path=NZ_PATH, timestep='hourly', verbosity=0)
    return


@TIMESTEPS
def test_no_redownload_or_reextract(ds, n_steps):
    """initializing again neither downloads, extracts nor rebuilds the cache"""
    logger.info(f"test_no_redownload_or_reextract {ds.timestep}")

    def boom(*args, **kwargs):
        raise AssertionError("must not be called when the data exists")

    originals = (_camels.download, zipfile.ZipFile.extractall,
                 xr.Dataset.to_netcdf if xr is not None else None)
    _camels.download = boom
    zipfile.ZipFile.extractall = boom
    if xr is not None:
        xr.Dataset.to_netcdf = boom
    try:
        again = CAMELS_NZ(path=NZ_PATH, timestep=ds.timestep, verbosity=0)
        assert len(again.stations()) == NUM_STATIONS
        assert again.start == ds.start and again.end == ds.end
    finally:
        _camels.download = originals[0]
        zipfile.ZipFile.extractall = originals[1]
        if xr is not None:
            xr.Dataset.to_netcdf = originals[2]
    return


def test_only_the_archives_of_the_timestep_are_downloaded():
    """
    The seven archives of a timestep are downloaded and extracted into the
    folder of the release; the five of the other timestep are left alone. Once
    extracted nothing is downloaded again, also after ``remove_zip`` deleted the
    archives.
    """
    logger.info("test_only_the_archives_of_the_timestep_are_downloaded")
    tmp = tempfile.mkdtemp()
    calls = []
    original = _camels.download
    try:
        _camels.download = _fake_download(calls)
        ds = _copy_for(os.path.join(tmp, 'CAMELS_NZ'))
        ds.remove_zip = False
        ds._download_camels_nz()

        assert sorted(calls) == sorted(f"{name}.zip" for name in ds._archives), calls
        assert len(calls) == 7, calls
        assert not any('hourly' in name for name in calls), calls
        assert sorted(os.listdir(ds._root)) == sorted(
            [name for name in ds._archives] + [f"{name}.zip" for name in ds._archives])

        calls.clear()
        ds._download_camels_nz()
        assert calls == [], f"downloaded again although the data is extracted: {calls}"

        # with the folder gone but the archive still there it is extracted again
        # without downloading, and remove_zip then deletes the archive
        first = ds._archives[0]
        ds.remove_zip = True
        shutil.rmtree(os.path.join(ds._root, first))
        ds._download_camels_nz()
        assert calls == [], f"downloaded although the archive is there: {calls}"
        assert os.path.isdir(os.path.join(ds._root, first))
        assert not os.path.exists(os.path.join(ds._root, f"{first}.zip")), \
            "remove_zip left the archive"

        # neither the archive nor the folder: downloaded again, once
        shutil.rmtree(os.path.join(ds._root, first))
        ds._download_camels_nz()
        assert calls == [f"{first}.zip"], calls

        # and the data of the other six archives is still not downloaded again
        calls.clear()
        ds._download_camels_nz()
        assert calls == [], f"downloaded again after remove_zip: {calls}"

        # the hourly instance asks for the hourly archives of the same folder
        hourly = _copy_for(os.path.join(tmp, 'CAMELS_NZ'), dataset_h)
        assert sum('hourly' in name for name in hourly._archives) == 5, hourly._archives
    finally:
        _camels.download = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_interrupted_extraction():
    """
    An extraction which stops half way leaves no folder that looks complete, so
    the next initialization extracts it again without downloading.
    """
    logger.info("test_interrupted_extraction")
    tmp = tempfile.mkdtemp()
    calls = []
    original = _camels.download
    try:
        _camels.download = _fake_download(calls)
        ds = _copy_for(os.path.join(tmp, 'CAMELS_NZ'))
        ds.remove_zip = False
        ds._download_camels_nz()
        calls.clear()

        name = ds._archives[0]
        folder = os.path.join(ds._root, name)
        # as an interrupted extraction leaves it: a *_extracting folder and no
        # folder under the final name
        shutil.move(folder, f"{folder}_extracting")

        ds._download_camels_nz()
        assert calls == [], f"downloaded again although the archive is there: {calls}"
        assert os.path.isdir(folder), "the archive was not extracted again"
        assert not os.path.exists(f"{folder}_extracting"), "the partial folder survived"
    finally:
        _camels.download = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_corrupt_archive_is_deleted():
    """an error page saved under the archive's name is deleted, not extracted"""
    logger.info("test_corrupt_archive_is_deleted")
    tmp = tempfile.mkdtemp()
    try:
        archive = os.path.join(tmp, 'CAMELS_NZ_daily_PET.zip')
        with open(archive, 'w') as f:
            f.write('<html>404</html>')
        with pytest.raises(ValueError, match='corrupt'):
            _camels._extract_zip(archive, os.path.join(tmp, 'CAMELS_NZ_daily_PET'), verbosity=0)
        assert not os.path.exists(archive)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_overwrite_removes_stale_before_download():
    """
    ``overwrite=True`` deletes this timestep's archives, folders and cache first,
    so that the download helper cannot write ``name.zip1`` next to a file which
    is never read, and downloads them again.
    """
    logger.info("test_overwrite_removes_stale_before_download")
    tmp = tempfile.mkdtemp()
    calls = []
    original = _camels.download
    try:
        _camels.download = _fake_download(calls)
        ds = _copy_for(os.path.join(tmp, 'CAMELS_NZ'))
        ds.remove_zip = False
        ds._download_camels_nz()
        assert len(calls) == 7, calls

        # a cache and a file which must not survive the overwrite
        os.makedirs(os.path.dirname(ds.dyn_fpath), exist_ok=True)
        open(ds.dyn_fpath, 'w').close()
        stale = os.path.join(ds._root, ds._archives[0], 'stale.csv')
        open(stale, 'w').close()

        calls.clear()
        ds._download_camels_nz(overwrite=True)

        assert len(calls) == 7, f"overwrite did not download again: {calls}"
        assert not os.path.exists(stale), "the old extracted folder survived"
        assert not os.path.exists(ds.dyn_fpath), "the old netCDF cache survived"
        assert not any(name.endswith('.zip1') for name in os.listdir(ds._root)), \
            os.listdir(ds._root)
    finally:
        _camels.download = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_release_2_is_reported_and_kept():
    """
    A release 2 left in ``path`` is not read any more. It is reported, whatever
    the verbosity, and it is not deleted: it may be the only copy the user has.
    """
    logger.info("test_release_2_is_reported_and_kept")
    tmp = tempfile.mkdtemp()
    calls = []
    original = _camels.download
    try:
        _camels.download = _fake_download(calls)
        path = os.path.join(tmp, 'CAMELS_NZ')
        os.makedirs(path)
        old = _release_2(path)

        ds = _copy_for(path)
        ds.verbosity = 0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds._warn_old_release()
        messages = [str(w.message) for w in caught]
        assert any('release 2' in m for m in messages), messages
        assert all(os.path.exists(p) for p in old.values()), "release 2 was deleted"

        # the folder and the cache of release 5 are elsewhere
        ds._download_camels_nz()
        assert os.path.basename(ds._root) == 'camels_nz_v5', ds._root
        assert os.path.basename(ds.dyn_fpath) == cache_name('camels_nz_v5_D.nc')
        assert ds.dyn_fpath != old['cache_d'] and ds.dyn_fpath != old['cache_d_v2']
        assert all(os.path.exists(p) for p in old.values()), "release 2 was deleted"

        # and a folder with only release 5 says nothing
        ds2 = _copy_for(os.path.join(tmp, 'other'))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds2._warn_old_release()
        assert [str(w.message) for w in caught] == []
    finally:
        _camels.download = original
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_nested_pet_folder_is_resolved():
    """
    The daily PET archive of release 5 holds one more folder of the same name,
    so its files are one level deeper than in the other archives. Both layouts
    must be found.
    """
    logger.info("test_nested_pet_folder_is_resolved")
    assert os.path.basename(os.path.dirname(dataset.pet_path)) == \
        os.path.basename(dataset.pet_path), \
        f"the daily PET folder of release 5 is nested, got {dataset.pet_path}"
    assert os.path.isfile(dataset._stn_file(dataset.stations()[0], 'PET'))
    # the other archives are flat
    assert os.path.dirname(dataset.precip_path) == dataset._root, dataset.precip_path
    assert os.path.dirname(dataset_h.pet_path) == dataset_h._root, dataset_h.pet_path

    # a flat layout, as the other archives and older releases have it
    tmp = tempfile.mkdtemp()
    try:
        ds = _copy_for(os.path.join(tmp, 'CAMELS_NZ'))
        flat = os.path.join(ds._root, 'CAMELS_NZ_daily_PET')
        os.makedirs(flat)
        assert ds.pet_path == flat, ds.pet_path
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_missing_files_warn():
    """a half extracted release warns instead of being served as complete"""
    logger.info("test_missing_files_warn")
    tmp = tempfile.mkdtemp()
    try:
        ds = _copy_for(os.path.join(tmp, 'CAMELS_NZ'))
        os.makedirs(ds.static_path)
        shutil.copy(os.path.join(dataset.static_path, ATTRIBUTE_FILES[0]), ds.static_path)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds._check_manifest()
        messages = [str(w.message) for w in caught]
        # the 4 other attribute files, the 4 shapefile files and 5 x 369 series
        expected = f"{4 + 4 + 5 * NUM_STATIONS} of {5 + 4 + 5 * NUM_STATIONS} files"
        assert any(expected in m for m in messages), messages

        # and a complete release does not warn
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            dataset._check_manifest()
        assert [str(w.message) for w in caught] == []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_free_disk_space_knows_this_release():
    """
    The variable folders of a timestep are only redundant once that timestep's
    cache of *this* release exists.
    """
    logger.info("test_free_disk_space_knows_this_release")
    pairs = dict(dataset._redundant_after_consolidation())
    assert len(pairs) == 10, pairs          # five variables of two timesteps
    for folder, cache in pairs.items():
        assert folder.startswith(dataset._root), folder
        assert f"_v{dataset.release}_" in os.path.basename(cache), cache
    assert dataset.pet_path.startswith(
        os.path.join(dataset._root, 'CAMELS_NZ_daily_PET'))
    assert (os.path.join(dataset._root, 'CAMELS_NZ_hourly_PET'),
            os.path.join(dataset.path, cache_name('camels_nz_v5_H.nc'))) in pairs.items()
    return


if __name__ == "__main__":
    for name, test in sorted(list(globals().items())):
        if name.startswith('test_') and callable(test):
            params = [(dataset, DAILY_STEPS), (dataset_h, HOURLY_STEPS)] \
                if hasattr(test, 'pytestmark') else [()]
            for args in params:
                print(f"running {name} {getattr(args[0], 'timestep', '') if args else ''}")
                start = time.time()
                test(*args)
                print(f"   ok in {time.time() - start:.1f} s")
