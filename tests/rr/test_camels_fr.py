"""
Tests for CAMELS_FR (France, 654 catchments), release 3.2.

They check that the class

    * serves the raw values, units and dates of the daily time series unchanged
      (compared with an independent read of the raw files),
    * downloads release 3.2 and, on a copy left by an earlier version of
      aqua_fetch, replaces only the superseded release 2.1 attributes,
    * neither downloads nor extracts again once the data is extracted, even when
      the archives were deleted by ``remove_zip``,
    * reports missing files instead of serving a truncated dataset,
    * fetches all stations quickly.

The first run downloads release 3.2 (about 372 MB) into ``CAMELS_FR_PATH``.
Run as a script or with pytest.
"""

import os
import io
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
    logging.basicConfig(filename='test_camels_fr.log', filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

import aqua_fetch
from aqua_fetch import CAMELS_FR
from aqua_fetch.rr import _camels
from aqua_fetch.rr._map import (observed_streamflow_cms, observed_streamflow_mm,
                                total_precipitation, mean_air_temp,
                                solar_radiation, downward_longwave_radiation)

from utils import test_dataset as run_shared_tests

# The class appends ``CAMELS_FR`` to this path. Replace with the location on
# your machine.
CAMELS_FR_PATH = '/mnt/hyexnas01/abbaa0a/data/gscad_database/raw/CAMELS/CAMELS'

NUM_STATIONS = 654
NUM_STATIC = 344
NUM_DYNAMIC = 22
DYN_LEN = 18993          # daily steps 1970-01-01 .. 2021-12-31

STATION = 'J421191001'

# the columns that release 3.0 recomputed, and the file they live in
CORRECTED = ['hym_q_questionable', 'hym_q_unqualified', 'hym_q_anomaly_inrae']
HYDROMETRY_FILE = 'CAMELS_FR_hydrometry_statistics.csv'

# J cm-2 day-1 -> W m-2, the factor the class declares for both radiation series
J_CM2_DAY_TO_WM2 = 1e4 / 86400.0

# the download happens here, once
dataset = CAMELS_FR(path=CAMELS_FR_PATH, verbosity=0, remove_zip=False)


def _bare_instance(path: str, remove_zip: bool = False) -> CAMELS_FR:
    """
    A CAMELS_FR whose ``__init__`` has not run, pointing at ``path``. Used by the
    download tests so that they exercise ``_download_camels_fr`` on a small
    temporary folder without reading the real dataset.
    """
    ds = CAMELS_FR.__new__(CAMELS_FR)
    ds._path = path
    ds.name = 'CAMELS_FR'
    ds.timestep = 'D'
    ds.verbosity = 0
    ds.remove_zip = remove_zip
    return ds


def _fake_install(root: str, hydrometry_bytes: int, with_archives: bool = False,
                  n_daily: int = 3) -> str:
    """
    A folder with the layout of an extracted CAMELS-FR whose files are stubs.
    ``hydrometry_bytes`` sets the size of CAMELS_FR_hydrometry_statistics.csv,
    which is what tells release 2.1 (38976) from release 3.2 (39467) apart.
    """
    path = os.path.join(root, 'CAMELS_FR')
    attrs = os.path.join(path, 'CAMELS_FR_attributes', 'CAMELS_FR_attributes')
    static = os.path.join(attrs, 'static_attributes')
    stats = os.path.join(attrs, 'time_series_statistics')
    geog = os.path.join(path, 'CAMELS_FR_geography', 'CAMELS_FR_geography')
    daily = os.path.join(path, 'CAMELS_FR_time_series', 'CAMELS_FR_time_series', 'daily')
    licenses = os.path.join(path, 'ADDITIONAL_LICENSES', 'ADDITIONAL_LICENSES')
    for folder in (static, stats, geog, daily, licenses):
        os.makedirs(folder, exist_ok=True)

    for fname in CAMELS_FR._STATIC_ATTR_FILES:
        open(os.path.join(static, fname), 'w').close()
    for fname in CAMELS_FR._TS_STAT_FILES:
        open(os.path.join(stats, fname), 'w').close()
    for fname in CAMELS_FR._GEOG_FILES:
        open(os.path.join(geog, fname), 'w').close()
    for i in range(n_daily):
        open(os.path.join(daily, f"CAMELS_FR_tsd_A10500300{i}.csv"), 'w').close()

    with open(os.path.join(stats, HYDROMETRY_FILE), 'wb') as fp:
        fp.write(b'x' * hydrometry_bytes)

    for fname in ('README.md', 'NEWS.md', 'CAMELS-FR_description.ods'):
        open(os.path.join(path, fname), 'w').close()
    if with_archives:
        for fname in CAMELS_FR.url:
            if fname.endswith('.zip'):
                open(os.path.join(path, fname), 'w').close()
    return path


class _Recorder:
    """stands in for ``aqua_fetch.rr._camels.download`` and writes a zip whose
    single entry is the folder the real archive holds"""

    def __init__(self):
        self.fetched = []

    def __call__(self, url, outdir=None, fname=None, verbosity=0, **kwargs):
        self.fetched.append(fname)
        fpath = os.path.join(outdir, fname)
        if fname.endswith('.zip'):
            stem = fname[:-len('.zip')]
            with zipfile.ZipFile(fpath, 'w') as zf:
                zf.writestr(f"{stem}/placeholder.txt", "downloaded")
        else:
            open(fpath, 'w').close()
        return fpath


def _no_download(*args, **kwargs):
    raise AssertionError(f"download was called: {args} {kwargs}")


def _patched_download(monkeypatched):
    """context-manager-free swap of the module level ``download``"""
    previous = _camels.download
    _camels.download = monkeypatched
    return previous


def test_registration():
    """the class is registered and can be built by the RainfallRunoff factory"""
    logger.info("test_registration")
    assert 'CAMELS_FR' in aqua_fetch.ALL_DATASETS
    from aqua_fetch import RainfallRunoff
    # remove_zip defaults to True in RainfallRunoff and would delete the archives
    rr = RainfallRunoff('CAMELS_FR', path=CAMELS_FR_PATH, remove_zip=False,
                        verbosity=0)
    assert len(rr.stations()) == NUM_STATIONS
    return


def test_release_32_is_served():
    """
    The attributes on disk are release 3.2, not the release 2.1 that aqua_fetch
    downloaded until now. Release 3.0 recomputed three columns, so their values
    say which release the class is serving. The expected numbers come from the
    release 3.2 archive, not from the class.
    """
    logger.info("test_release_32_is_served")
    assert os.path.getsize(dataset._hydrometry_file) == 39467
    assert not dataset._stale_attributes()

    static, _ = dataset.fetch(stations='all', static_features='all',
                              dynamic_features=None)
    # release 2.1 gives 0.196 / 0.016 / 3.603 and maxima 2.1 / 4.3 / 40.8
    expected_mean = {'hym_q_questionable': 3.8096, 'hym_q_unqualified': 0.3407,
                     'hym_q_anomaly_inrae': 3.3385}
    expected_max = {'hym_q_questionable': 35.9, 'hym_q_unqualified': 73.8,
                    'hym_q_anomaly_inrae': 29.0}
    for col in CORRECTED:
        assert np.isclose(static[col].mean(), expected_mean[col], atol=1e-3), \
            f"{col}: {static[col].mean()}"
        assert np.isclose(static[col].max(), expected_max[col]), \
            f"{col}: {static[col].max()}"
    return


def test_url_is_release_32():
    """the record's own file ids of the files that release 3.2 changed"""
    logger.info("test_url_is_release_32")
    assert dataset.url['CAMELS_FR_attributes.zip'].endswith('621683')
    assert dataset.url['README.md'].endswith('621685')
    # NEWS.md, the changelog, exists only from release 3.0 on
    assert dataset.url['NEWS.md'].endswith('621689')
    assert os.path.exists(os.path.join(dataset.path, 'NEWS.md'))
    # unchanged since release 1.0 and not re-downloaded by the upgrade
    assert dataset.url['CAMELS_FR_time_series.zip'].endswith('343470')
    assert dataset.url['CAMELS_FR_geography.zip'].endswith('343465')
    return


def test_counts():
    """the shape of the dataset, which release 3.2 did not change"""
    logger.info("test_counts")
    assert len(dataset.stations()) == NUM_STATIONS
    assert len(dataset.static_features) == NUM_STATIC
    assert len(dataset.dynamic_features) == NUM_DYNAMIC
    assert dataset.start == pd.Timestamp('1970-01-01')
    assert dataset.end == pd.Timestamp('2021-12-31')

    _, dyn = dataset.fetch(stations=STATION, as_dataframe=True)
    assert dyn[STATION].shape == (DYN_LEN, NUM_DYNAMIC), dyn[STATION].shape
    return


def test_values_and_units_unchanged():
    """
    The served values are those of the raw csv, with only the two radiation
    series scaled from J cm-2 day-1 to W m-2. Read independently of the class.
    """
    logger.info("test_values_and_units_unchanged")
    fpath = os.path.join(dataset.daily_ts_path, f"CAMELS_FR_tsd_{STATION}.csv")
    raw = pd.read_csv(fpath, sep=';', index_col=0, parse_dates=True, comment='#')

    _, dyn = dataset.fetch(stations=STATION, as_dataframe=True)
    served = dyn[STATION]

    assert served.index.equals(raw.index)
    assert len(raw) == DYN_LEN

    unscaled = {'tsd_q_l': observed_streamflow_cms(),
                'tsd_q_mm': observed_streamflow_mm(),
                'tsd_prec': total_precipitation(),
                'tsd_temp': mean_air_temp()}
    for column, feature in unscaled.items():
        pd.testing.assert_series_equal(
            served[feature].astype('float64'), raw[column].astype('float64'),
            check_names=False, rtol=1e-6)

    for column, feature in (('tsd_rad_ssi', solar_radiation()),
                            ('tsd_rad_dli', downward_longwave_radiation())):
        pd.testing.assert_series_equal(
            served[feature].astype('float64'),
            (raw[column] * J_CM2_DAY_TO_WM2).astype('float64'),
            check_names=False, rtol=1e-6)
    return


def test_no_download_when_extracted():
    """
    Data that is extracted is never downloaded or extracted again, even when the
    archives are gone because ``remove_zip`` deleted them. The decision must be
    made on the extracted folders; keying it on the archives forces a 372 MB
    download on every initialization.
    """
    logger.info("test_no_download_when_extracted")
    tmp = tempfile.mkdtemp()
    try:
        path = _fake_install(tmp, hydrometry_bytes=39467, with_archives=False)
        ds = _bare_instance(path)

        previous_download = _patched_download(_no_download)
        previous_extract = zipfile.ZipFile.extractall
        zipfile.ZipFile.extractall = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("extractall was called"))
        try:
            ds._download_camels_fr()
        finally:
            _camels.download = previous_download
            zipfile.ZipFile.extractall = previous_extract
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_release_21_is_replaced():
    """
    A copy left by an earlier version of aqua_fetch is release 2.1. It must be
    detected, warned about unconditionally, and its attributes archive - and
    nothing else - downloaded again.
    """
    logger.info("test_release_21_is_replaced")
    tmp = tempfile.mkdtemp()
    try:
        path = _fake_install(tmp, hydrometry_bytes=38976, with_archives=False)
        ds = _bare_instance(path)
        assert ds._stale_attributes()

        recorder = _Recorder()
        previous = _patched_download(recorder)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ds._download_camels_fr()
        finally:
            _camels.download = previous

        messages = [str(w.message) for w in caught]
        assert any('2.1' in m for m in messages), messages

        # only the attributes archive and the two documentation files
        assert 'CAMELS_FR_attributes.zip' in recorder.fetched, recorder.fetched
        assert 'CAMELS_FR_time_series.zip' not in recorder.fetched, recorder.fetched
        assert 'CAMELS_FR_geography.zip' not in recorder.fetched, recorder.fetched
        assert 'ADDITIONAL_LICENSES.zip' not in recorder.fetched, recorder.fetched
        assert 'README.md' in recorder.fetched, recorder.fetched

        # the stale attributes are gone, the replacement is extracted
        assert os.path.exists(os.path.join(path, 'CAMELS_FR_attributes',
                                           'CAMELS_FR_attributes', 'placeholder.txt'))
        assert not os.path.exists(os.path.join(
            path, 'CAMELS_FR_attributes', 'CAMELS_FR_attributes',
            'time_series_statistics', HYDROMETRY_FILE))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_release_32_is_left_alone():
    """a copy that is already release 3.2 is not downloaded again"""
    logger.info("test_release_32_is_left_alone")
    tmp = tempfile.mkdtemp()
    try:
        path = _fake_install(tmp, hydrometry_bytes=39467, with_archives=False)
        ds = _bare_instance(path)
        assert not ds._stale_attributes()

        previous = _patched_download(_no_download)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                ds._download_camels_fr()
        finally:
            _camels.download = previous
        assert not [w for w in caught if '2.1' in str(w.message)]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_remove_zip_is_honoured():
    """
    ``remove_zip=True`` deletes the archives once they are extracted, and the
    next initialization still does not download anything.
    """
    logger.info("test_remove_zip_is_honoured")
    tmp = tempfile.mkdtemp()
    try:
        path = _fake_install(tmp, hydrometry_bytes=39467, with_archives=True)
        ds = _bare_instance(path, remove_zip=True)

        previous = _patched_download(_no_download)
        try:
            ds._download_camels_fr()
            archives = [f for f in os.listdir(path) if f.endswith('.zip')]
            assert archives == [], archives
            # the extracted folders survive, so nothing is downloaded again
            ds._download_camels_fr()
        finally:
            _camels.download = previous
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_overwrite_removes_stale_files():
    """
    ``overwrite=True`` removes the archives, the extracted folders and the
    netCDF caches, so that the new download cannot land next to stale files.
    """
    logger.info("test_overwrite_removes_stale_files")
    tmp = tempfile.mkdtemp()
    try:
        path = _fake_install(tmp, hydrometry_bytes=39467, with_archives=True)
        cache = os.path.join(path, 'camels_fr_D_v2.nc')
        open(cache, 'w').close()
        ds = _bare_instance(path)

        recorder = _Recorder()
        previous = _patched_download(recorder)
        try:
            ds._download_camels_fr(overwrite=True)
        finally:
            _camels.download = previous

        assert not os.path.exists(cache), "the stale netCDF cache survived"
        assert sorted(recorder.fetched) == sorted(CAMELS_FR.url), recorder.fetched
        # a re-download that lands on an existing file is saved as `name.zip1`
        assert not [f for f in os.listdir(path) if f.endswith('1')]
        # the old attribute files are gone, only the fresh extraction is there
        assert not os.path.exists(os.path.join(
            path, 'CAMELS_FR_attributes', 'CAMELS_FR_attributes',
            'static_attributes'))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_interrupted_extraction_is_redone():
    """
    An extraction that dies half way must not leave a folder that the next
    initialization takes for complete.
    """
    logger.info("test_interrupted_extraction_is_redone")
    tmp = tempfile.mkdtemp()
    try:
        path = os.path.join(tmp, 'CAMELS_FR')
        os.makedirs(path)
        ds = _bare_instance(path)

        def dying_extractall(self, target):
            os.makedirs(target, exist_ok=True)
            raise EOFError("interrupted")

        recorder = _Recorder()
        previous = _patched_download(recorder)
        previous_extract = zipfile.ZipFile.extractall
        zipfile.ZipFile.extractall = dying_extractall
        try:
            try:
                ds._download_camels_fr()
            except Exception:
                pass  # the archive is reported as corrupt and deleted
        finally:
            _camels.download = previous
            zipfile.ZipFile.extractall = previous_extract

        folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
        assert folders == [], f"an incomplete extraction was kept: {folders}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_manifest_reports_missing_files():
    """
    A file deleted by hand, or left out by an interrupted extraction, is
    reported instead of silently making the dataset smaller.
    """
    logger.info("test_manifest_reports_missing_files")
    tmp = tempfile.mkdtemp()
    try:
        path = _fake_install(tmp, hydrometry_bytes=39467, n_daily=3)
        ds = _bare_instance(path)
        ds._stations = ['a', 'b', 'c']

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds._check_manifest()
        assert not caught, [str(w.message) for w in caught]

        os.remove(os.path.join(ds.static_attr_path, 'CAMELS_FR_geology_attributes.csv'))
        os.remove(os.path.join(ds.geog_path, 'CAMELS_FR_catchment_boundaries.gpkg'))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds._check_manifest()
        assert len(caught) == 1, [str(w.message) for w in caught]
        message = str(caught[0].message)
        assert 'CAMELS_FR_geology_attributes.csv' in message, message
        assert 'CAMELS_FR_catchment_boundaries.gpkg' in message, message

        # a time series file short of the number of stations is reported too
        ds._stations = ['a', 'b', 'c', 'd']
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ds._check_manifest()
        assert 'daily time series files' in str(caught[0].message)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_coords_area_and_boundary():
    """the ancillary accessors work and stay quick"""
    logger.info("test_coords_area_and_boundary")
    start = time.time()
    coords = dataset.stn_coords()
    areas = dataset.area()
    assert len(dataset.stations()) == NUM_STATIONS
    assert time.time() - start < 5, "the documented one liners are slow"

    assert coords.shape == (NUM_STATIONS, 2), coords.shape
    assert len(areas) == NUM_STATIONS
    # metropolitan France and Corsica
    assert coords['lat'].between(41, 52).all()
    assert coords['long'].between(-6, 10).all()
    assert areas.min() > 0
    return


def test_efficiency():
    """fetching every station stays quick"""
    logger.info("test_efficiency")
    start = time.time()
    _, dyn = dataset.fetch(as_dataframe=True)
    took = time.time() - start
    assert len(dyn) == NUM_STATIONS
    logger.info(f"fetched {NUM_STATIONS} stations in {took:.1f} s")
    assert took < 60, f"fetching all stations took {took:.1f} s"
    return


def test_shared_suite():
    """the checks every rainfall-runoff dataset has to pass"""
    logger.info("test_shared_suite")
    run_shared_tests(dataset, NUM_STATIONS, DYN_LEN, NUM_STATIC, NUM_DYNAMIC,
                     test_latlong_ranges=False)
    return


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith('test_') and callable(test):
            print(f"{name} ...", flush=True)
            start = time.time()
            test()
            print(f"{name} passed in {time.time() - start:.1f} s", flush=True)
    print("all tests passed")
