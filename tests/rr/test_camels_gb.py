"""
Tests for CAMELS_GB: version 2 daily (the default) and hourly, plus checks that
version 1 keeps working. They check that the class

    * downloads version 2 once, completes an interrupted download or extraction
      and deletes stale files before an ``overwrite``,
    * serves the time series and attributes without changing values or units,
    * reads stations, features and the time period from the files,
    * keeps the files and netCDF cache of each version and timestep apart,
    * and passes the generic rainfall-runoff suite (``utils.test_dataset``).

Set ``raw_data_path`` and run with pytest. The first run downloads version 2:
~0.8 GB for the daily and ~10.6 GB for the hourly data.
"""
import io
import os
import csv
import copy
import glob
import json
import math
import site
import time
import random
import shutil
import zipfile
import warnings
import concurrent.futures as cf

import numpy as np
import pandas as pd
import pytest

# add the repository root to the path so that ``aqua_fetch`` can be imported
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

import aqua_fetch
from aqua_fetch import CAMELS_GB, RainfallRunoff
from aqua_fetch.rr import _camels
from aqua_fetch._backend import xarray as xr, fiona, netCDF4

# the generic rainfall-runoff suite; imported under another name so that pytest
# does not collect it as a test of this module
from utils import test_dataset as run_generic_suite

raw_data_path = '/path/to/raw/data'  # replace with actual path
GB_PATH = os.path.join(raw_data_path, 'CAMELS')

NUM_STATIONS = 671
V1_ID = "8344e4f3-d2ea-44f5-8afa-86d2987543a9"
V2_ID = "9a46d428-958f-4ac1-86eb-94eee70c0955"

# raw column -> served name, in the column order of the files. Written out here
# rather than imported, so that a wrong mapping in the library is caught.
V2_RAW_TO_STD = {
    'precipitation_cehgear': 'pcp_mm_cehgear',
    'precipitation_haduk': 'pcp_mm_haduk',
    'pet_chess': 'pet_mm_chess',
    'peti_chess': 'pet_mm_intercep_chess',
    'pet_hydrope': 'pet_mm_hydrope',
    'peti_hydrope': 'pet_mm_intercep_hydrope',
    'temperature_chess': 'airtemp_C_mean_chess',
    'temperature_haduk': 'airtemp_C_mean_haduk',
    'discharge_spec': 'q_mm_obs',
    'discharge_vol': 'q_cms_obs',
}
V2_HOURLY_RAW_TO_STD = {
    'precipitation_cehgear': 'pcp_mm_cehgear',
    'precipitation_gradgb': 'pcp_mm_gradgb',
    'discharge_spec': 'q_mm_obs',
    'discharge_vol': 'q_cms_obs',
    'discharge_flag': 'discharge_flag',  # quality codes keep their names
    'level': 'wl_m_obs',
    'level_flag': 'level_flag',
}
V1_RAW_TO_STD = {
    'precipitation': 'pcp_mm',
    'pet': 'pet_mm',
    'temperature': 'airtemp_C_mean',
    'discharge_spec': 'q_mm_obs',
    'discharge_vol': 'q_cms_obs',
    'peti': 'pet_mm_intercep',
    'humidity': 'spechum_gkg',  # specific humidity, g kg-1
    'shortwave_rad': 'swdownrad_wm2',
    'longwave_rad': 'lwdownrad_wm2',
    'windspeed': 'windspeed_mps',
}
# slope_fdc is a streamflow signature, not a terrain slope, and is served
# under its published name
STATIC_RENAMED = {'area': 'area_km2', 'gauge_lat': 'lat', 'gauge_lon': 'long'}
CATEGORIES = ['climatic', 'humaninfluence', 'hydrogeology', 'hydrologic',
              'hydrometry', 'landcover', 'soil', 'topographic']

HOURLY_STEPS = 280512  # 1990-10-01 09:00 to 2022-10-01 08:00 UTC

dataset = CAMELS_GB(path=GB_PATH, verbosity=0)
dataset_v1 = CAMELS_GB(path=GB_PATH, version=1, verbosity=0)
dataset_h = CAMELS_GB(path=GB_PATH, timestep='H', verbosity=0)

# the daily datasets, which share nearly all of their behaviour
VERSIONS = pytest.mark.parametrize(
    "ds, version, raw_to_std, n_steps, n_static",
    [(dataset, 2, V2_RAW_TO_STD, 18993, 219), (dataset_v1, 1, V1_RAW_TO_STD, 16436, 145)],
    ids=['v2', 'v1'])
# every dataset, including the hourly one
ALL_TIMESTEPS = pytest.mark.parametrize(
    "ds, version", [(dataset, 2), (dataset_v1, 1), (dataset_h, 2)],
    ids=['v2', 'v1', 'v2 hourly'])


def _sample(seq, k: int) -> list:
    """the same random stations whichever tests run before"""
    return random.Random(313).sample(list(seq), k)


# ---------------------------------------------------------------------------
# independent readers of the published files
# ---------------------------------------------------------------------------

def _version_dir(version: int) -> str:
    return os.path.join(GB_PATH, 'CAMELS_GB', 'camels_gb_v2' if version == 2 else 'camels_gb')


def _raw_ts_files(version: int, timestep: str = 'D') -> list:
    if version == 2:
        folder = 'daily' if timestep == 'D' else 'hourly'
        pattern = os.path.join(_version_dir(2), 'Catchment_Timeseries', 'hydro-meteorological',
                               folder, f'camels_gb_v2_hydromet_{folder}_timeseries_*.csv')
    else:
        pattern = os.path.join(_version_dir(1), 'camels_gb', 'data', 'timeseries',
                               'CAMELS_GB_hydromet_timeseries_*.csv')
    return glob.glob(pattern)


def _raw_ts(version: int, stn: str, timestep: str = 'D') -> pd.DataFrame:
    files = [f for f in _raw_ts_files(version, timestep)
             if os.path.basename(f).split('_')[-2] == stn]
    assert len(files) == 1, files
    return pd.read_csv(files[0], index_col='date', parse_dates=True)


def _raw_attr_file(version: int, category: str) -> str:
    if version == 2:
        return os.path.join(_version_dir(2), 'Catchment_Attributes',
                            f'camels_gb_v2_{category}_attributes.csv')
    return os.path.join(_version_dir(1), 'camels_gb', 'data',
                        f'CAMELS_GB_{category}_attributes.csv')


def _raw_attr_rows(fpath: str):
    """header and ``{gauge_id: fields}`` read with the csv module. Fields beyond
    the header's length are part of the (free-text) last column."""
    with open(fpath, newline='', encoding='utf-8') as fp:
        rows = list(csv.reader(fp))
    n = len(rows[0])
    return rows[0], {r[0]: r[:n - 1] + [','.join(r[n - 1:])] for r in rows[1:]}


def _same(value, text: str) -> bool:
    """whether a served attribute value equals the published text"""
    if text in ('', 'NaN', 'NA'):
        return pd.isna(value)
    try:
        return math.isclose(float(value), float(text), rel_tol=1e-12)
    except (TypeError, ValueError):
        return str(value).lower() == text.lower()


# ---------------------------------------------------------------------------
# small fake records in tmp_path, for the download tests
# ---------------------------------------------------------------------------

def _tmp_dataset(ds, base) -> CAMELS_GB:
    """a copy of ``ds`` whose files are under ``base``; the caches of the real
    data are dropped, otherwise the copy would answer from them"""
    tmp = copy.copy(ds)
    tmp._path = str(base)
    tmp._static_df = tmp._ts_fnames = tmp._dyn_feats = tmp._period_ = None
    return tmp


def _manifest_json(files: dict) -> str:
    """content of a ro-crate-metadata.json listing ``{@id: bytes}``"""
    graph = [{'@id': rel, '@type': 'File', 'bytes': nbytes} for rel, nbytes in files.items()]
    graph.append({'@id': './', '@type': 'Dataset'})
    return json.dumps({'@graph': graph})


def _manifest_zip(files: dict) -> bytes:
    """a supporting-documents zip with the manifest of ``files``"""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w') as zf:
        zf.writestr('ro-crate-metadata.json', _manifest_json(files))
    return buffer.getvalue()


def _fake_v2_download(files: dict, calls: list):
    """a stand-in for ``download`` which writes a file of the manifest size"""
    sizes = {os.path.basename(rel): nbytes for rel, nbytes in files.items()}

    def fake_download(url, outdir, fname, verbosity):
        assert not os.path.exists(os.path.join(outdir, fname)), "must be deleted first"
        calls.append((url, outdir, fname))
        content = _manifest_zip(files) if fname == 'supporting_documents.zip' else b'x' * sizes[fname]
        with open(os.path.join(outdir, fname), 'wb') as fp:
            fp.write(content)

    return fake_download


def _fake_v1_zip(path):
    """a zip laid out like the version 1 record, with the boundary zip inside"""
    boundaries = io.BytesIO()
    with zipfile.ZipFile(boundaries, 'w') as zf:
        zf.writestr('CAMELS_GB_catchment_boundaries.shp', 'shp')
    with zipfile.ZipFile(path, 'w') as zf:
        zf.writestr(f'{V1_ID}/data/timeseries/CAMELS_GB_hydromet_timeseries_1001_19701001-20150930.csv', 'date')
        zf.writestr(f'{V1_ID}/data/CAMELS_GB_catchment_boundaries.zip', boundaries.getvalue())


DAILY = 'Catchment_Timeseries/hydro-meteorological/daily'
HOURLY = 'Catchment_Timeseries/hydro-meteorological/hourly'
# files of a fake version 2 record which the class uses ...
USED_V2 = {
    'data/Catchment_Attributes/camels_gb_v2_soil_attributes.csv': 10,
    f'data/{DAILY}/camels_gb_v2_hydromet_daily_timeseries_1001_19701001-20220930.csv': 20,
    'data/Catchment_Boundaries/camels_gb_v2_catchment_boundaries.shp': 30,
}
# ... and all its files
FAKE_V2 = {
    **USED_V2,
    'data/Catchment_Attributes/camels_gb_v2_groundwaterwell_attributes.csv': 5,
    'data/Catchment_Timeseries/hydro-meteorological/hourly/x.csv': 5,
    'data/Catchment_Timeseries/groundwater/daily/x.csv': 5,
    'supporting-documents/camels_gb_v2_eidc_supp_info.docx': 5,
}


# ---------------------------------------------------------------------------
# construction, versions and download
# ---------------------------------------------------------------------------

def test_registration():
    assert 'CAMELS_GB' in aqua_fetch.ALL_DATASETS
    assert dataset.version == 2
    rr = RainfallRunoff('CAMELS_GB', path=GB_PATH, version=1, verbosity=0)
    assert rr.dataset.version == 1
    assert len(rr.stations()) == NUM_STATIONS


@pytest.mark.parametrize('version', [0, 3, '2', None, True])
def test_invalid_version(tmp_path, version):
    with pytest.raises(ValueError):
        CAMELS_GB(path=str(tmp_path), version=version, verbosity=0)
    assert os.listdir(tmp_path) == [], "nothing may be created for an invalid version"


def test_versions_are_kept_apart():
    """each version has its own folder and netCDF cache, and each timestep its
    own cache name. The version 1 cache built by older aqua_fetch (directly in
    CAMELS_GB/) named humidity ``rh_%`` and must not be read."""
    assert dataset._version_dir == _version_dir(2)
    assert dataset_v1._version_dir == _version_dir(1)
    for ds in (dataset, dataset_v1):
        assert os.path.dirname(ds.dyn_fpath) == ds._version_dir
        assert ds.dyn_fpath_exists or netCDF4 is None
    assert dataset.data_path != dataset_v1.data_path

    # the hourly data shares the folder of version 2 but not the time series
    # folder, and has no cache of its own by default
    assert dataset_h._version_dir == dataset._version_dir
    assert dataset_h.ts_dir != dataset.ts_dir
    assert dataset_h.dyn_fpath != dataset.dyn_fpath
    assert dataset_h.to_netcdf is False and not dataset_h.dyn_fpath_exists


def test_hourly_only_in_version_2(tmp_path):
    with pytest.raises(ValueError, match='hourly'):
        CAMELS_GB(path=str(tmp_path), version=1, timestep='H', verbosity=0)
    with pytest.raises(ValueError, match='timestep'):
        CAMELS_GB(path=str(tmp_path), timestep='hourly', verbosity=0)
    assert os.listdir(tmp_path) == []


def test_no_redownload_or_rebuild(monkeypatch):
    """a second initialization must neither download, extract, delete nor
    rebuild the cache"""
    def boom(*args, **kwargs):
        raise AssertionError("must not be called when the data is on disk")

    for module, name in ((_camels, 'download'), (zipfile, 'ZipFile'), (cf, 'ThreadPoolExecutor'),
                         (shutil, 'rmtree'), (os, 'remove'), (os, 'rename')):
        monkeypatch.setattr(module, name, boom)
    if xr is not None:
        monkeypatch.setattr(xr.Dataset, 'to_netcdf', boom)

    for version, timestep in ((1, 'D'), (2, 'D'), (2, 'H')):
        ds = CAMELS_GB(path=GB_PATH, version=version, timestep=timestep, verbosity=0)
        assert len(ds.stations()) == NUM_STATIONS


@pytest.mark.parametrize('ds, folder', [(dataset, DAILY), (dataset_h, HOURLY)], ids=['D', 'H'])
def test_v2_manifest_matches_disk(ds, folder):
    """the files used are those listed by the record for this timestep, and all
    of them are complete"""
    manifest = ds._v2_manifest()
    folders = pd.Series([os.path.dirname(rel) for rel in manifest]).value_counts().to_dict()
    assert folders == {folder: NUM_STATIONS, 'Catchment_Attributes': 8,
                       'Catchment_Boundaries': 7}, folders
    assert not any('groundwater' in rel for rel in manifest)
    for rel, nbytes in manifest.items():
        assert os.path.getsize(os.path.join(_version_dir(2), rel)) == nbytes, rel


def test_v2_interrupted_download_is_completed(tmp_path, monkeypatch):
    """only missing files and files whose size differs from the manifest are
    downloaded; files of other folders are not"""
    root = tmp_path / 'CAMELS_GB' / 'camels_gb_v2'
    (root / 'supporting_documents').mkdir(parents=True)
    (root / 'supporting_documents' / 'ro-crate-metadata.json').write_text(_manifest_json(FAKE_V2))
    complete = root / 'Catchment_Attributes' / 'camels_gb_v2_soil_attributes.csv'
    cut = root / DAILY / 'camels_gb_v2_hydromet_daily_timeseries_1001_19701001-20220930.csv'
    for fpath, nbytes in ((complete, 10), (cut, 12)):
        fpath.parent.mkdir(parents=True, exist_ok=True)
        fpath.write_bytes(b'o' * nbytes)

    calls, workers = [], []
    monkeypatch.setattr(_camels, 'download', _fake_v2_download(FAKE_V2, calls))
    real_pool = cf.ThreadPoolExecutor

    class SpyPool(real_pool):
        def __init__(self, max_workers=None, *args, **kwargs):
            workers.append(max_workers)
            super().__init__(max_workers, *args, **kwargs)

    monkeypatch.setattr(cf, 'ThreadPoolExecutor', SpyPool)
    ds = _tmp_dataset(dataset, tmp_path / 'CAMELS_GB')
    ds.processes = 1
    with pytest.warns(UserWarning, match='1 files .* incomplete'):
        ds._download()
    assert workers == [1], "processes=1 downloads one file at a time"

    base = f"https://catalogue.ceh.ac.uk/datastore/eidchub/{V2_ID}"
    assert sorted(calls) == sorted([
        (f"{base}/{DAILY}/{cut.name}", str(root / DAILY), cut.name),
        (f"{base}/Catchment_Boundaries/camels_gb_v2_catchment_boundaries.shp",
         str(root / 'Catchment_Boundaries'), 'camels_gb_v2_catchment_boundaries.shp'),
    ])
    assert complete.read_bytes() == b'o' * 10
    assert cut.stat().st_size == 20


def test_v2_wrong_size_after_download_raises(tmp_path, monkeypatch):
    root = tmp_path / 'CAMELS_GB' / 'camels_gb_v2'
    (root / 'supporting_documents').mkdir(parents=True)
    (root / 'supporting_documents' / 'ro-crate-metadata.json').write_text(_manifest_json(FAKE_V2))
    wrong_sizes = {rel: nbytes + 1 for rel, nbytes in FAKE_V2.items()}
    monkeypatch.setattr(_camels, 'download', _fake_v2_download(wrong_sizes, []))

    with pytest.raises(RuntimeError, match='size'):
        _tmp_dataset(dataset, tmp_path / 'CAMELS_GB')._download()


@pytest.mark.parametrize('state', ['missing', 'cut short', 'corrupt zip'])
def test_v2_manifest_recovery(tmp_path, monkeypatch, state):
    """a missing or cut-short manifest is extracted again from the zip on disk,
    which is then deleted if ``remove_zip`` is set; a corrupt zip is downloaded
    again"""
    root = tmp_path / 'CAMELS_GB' / 'camels_gb_v2'
    root.mkdir(parents=True)
    zip_path = root / 'supporting_documents.zip'
    zip_path.write_bytes(b'not a zip' if state == 'corrupt zip' else _manifest_zip(FAKE_V2))
    if state == 'cut short':
        (root / 'supporting_documents').mkdir()
        (root / 'supporting_documents' / 'ro-crate-metadata.json').write_text(_manifest_json(FAKE_V2)[:50])

    calls = []
    monkeypatch.setattr(_camels, 'download', _fake_v2_download(FAKE_V2, calls))
    ds = _tmp_dataset(dataset, tmp_path / 'CAMELS_GB')
    ds.remove_zip = True

    assert ds._v2_manifest() == {rel[len('data/'):]: n for rel, n in USED_V2.items()}
    assert len(calls) == (1 if state == 'corrupt zip' else 0)
    assert not zip_path.exists()


@pytest.mark.parametrize('ds, other_folder', [(dataset, HOURLY), (dataset_h, DAILY)],
                         ids=['D', 'H'])
def test_v2_overwrite_deletes_before_download(tmp_path, monkeypatch, ds, other_folder):
    """the files of this timestep are deleted and downloaded again, those of the
    other timestep are left alone"""
    base = tmp_path / 'CAMELS_GB'
    root = base / 'camels_gb_v2'
    (root / 'supporting_documents').mkdir(parents=True)
    (root / 'supporting_documents' / 'ro-crate-metadata.json').write_text(_manifest_json(FAKE_V2))
    stale = root / 'Catchment_Attributes' / 'camels_gb_v2_soil_attributes.csv'
    stale.parent.mkdir(parents=True)
    stale.write_text('old')
    cache = root / ds.dyn_fname
    cache.write_text('old')

    other = root / other_folder / 'other_timestep.csv'
    other.parent.mkdir(parents=True)
    other.write_text('other')
    other_cache = root / (dataset_h if ds is dataset else dataset).dyn_fname
    other_cache.write_text('other cache')

    calls = []
    fake_download = _fake_v2_download(FAKE_V2, calls)

    def checked_download(url, outdir, fname, verbosity):
        assert not cache.exists(), "the cache must be deleted before downloading"
        fake_download(url, outdir, fname, verbosity)

    monkeypatch.setattr(_camels, 'download', checked_download)
    _tmp_dataset(ds, base)._download(overwrite=True)

    # attributes and boundaries, plus the time series of this timestep only
    folder = DAILY if ds is dataset else HOURLY
    expected = [rel for rel in FAKE_V2
                if (rel.startswith(('data/Catchment_Attributes/', 'data/Catchment_Boundaries/'))
                    and 'groundwaterwell' not in rel) or rel.startswith(f'data/{folder}/')]
    fnames = [fname for _, _, fname in calls]
    assert fnames[0] == 'supporting_documents.zip', "the manifest must be downloaded afresh"
    assert sorted(fnames[1:]) == sorted(os.path.basename(rel) for rel in expected)
    assert stale.read_bytes() == b'x' * 10
    assert other.read_text() == 'other', "the data of the other timestep was deleted"
    assert other_cache.read_text() == 'other cache'


def test_v1_overwrite_deletes_before_download(tmp_path, monkeypatch):
    base = tmp_path / 'CAMELS_GB'
    old_file = base / 'camels_gb' / 'camels_gb' / 'data' / 'old.csv'
    old_file.parent.mkdir(parents=True)
    old_file.write_text('old')
    zip_path = base / 'camels_gb.zip'
    _fake_v1_zip(zip_path)

    calls = []

    def fake_download(url, outdir, fname, verbosity):
        assert not zip_path.exists() and not old_file.exists(), "stale files must be deleted first"
        assert url == f"https://data-package.ceh.ac.uk/data/{V1_ID}.zip"
        calls.append(fname)
        _fake_v1_zip(os.path.join(outdir, fname))

    monkeypatch.setattr(_camels, 'download', fake_download)
    ds = _tmp_dataset(dataset_v1, base)
    ds.remove_zip = True
    ds._download(overwrite=True)

    assert calls == ['camels_gb.zip']
    assert os.path.exists(ds.boundary_file) and os.path.isdir(ds.ts_dir)
    assert not zip_path.exists(), "remove_zip deletes the zip after extraction"


@pytest.mark.parametrize('zip_state', ['valid', 'corrupt'])
def test_v1_interrupted_extraction(tmp_path, monkeypatch, zip_state):
    """a partly extracted folder is deleted and the zip on disk is extracted
    again; only a corrupt zip is downloaded again"""
    base = tmp_path / 'CAMELS_GB'
    # a leftover which would stop renaming the extracted folder
    partial = base / 'camels_gb' / 'camels_gb' / 'readme.html'
    partial.parent.mkdir(parents=True)
    partial.write_text('x')
    zip_path = base / 'camels_gb.zip'
    if zip_state == 'valid':
        _fake_v1_zip(zip_path)
    else:
        zip_path.write_bytes(b'not a zip')

    calls = []

    def fake_download(url, outdir, fname, verbosity):
        calls.append(fname)
        _fake_v1_zip(os.path.join(outdir, fname))

    monkeypatch.setattr(_camels, 'download', fake_download)
    ds = _tmp_dataset(dataset_v1, base)
    ds._download()

    assert calls == ([] if zip_state == 'valid' else ['camels_gb.zip'])
    assert not partial.exists()
    assert os.path.exists(ds.boundary_file) and os.path.isdir(ds.ts_dir)


def test_v1_failed_extraction_is_repeated(tmp_path, monkeypatch):
    """an extraction that fails part way leaves no folder that looks complete,
    so it is repeated at the next initialization"""
    base = tmp_path / 'CAMELS_GB'
    base.mkdir()
    _fake_v1_zip(base / 'camels_gb.zip')
    monkeypatch.setattr(_camels, 'download', None)
    ds = _tmp_dataset(dataset_v1, base)
    real_extractall = zipfile.ZipFile.extractall

    def failing_for(zip_name):
        def extractall(self, path=None, *args, **kwargs):
            if os.path.basename(self.filename) == zip_name:
                self.extract(self.namelist()[0], path)
                raise OSError(28, 'No space left on device')
            return real_extractall(self, path, *args, **kwargs)
        return extractall

    monkeypatch.setattr(zipfile.ZipFile, 'extractall', failing_for('camels_gb.zip'))
    with pytest.raises(OSError):
        ds._download()
    assert not os.path.exists(ds.data_path)

    # the zip of the boundaries, inside the data folder
    monkeypatch.setattr(zipfile.ZipFile, 'extractall', failing_for('CAMELS_GB_catchment_boundaries.zip'))
    with pytest.raises(OSError):
        ds._download()
    assert os.path.exists(ds.data_path) and not os.path.exists(os.path.dirname(ds.boundary_file))

    monkeypatch.setattr(zipfile.ZipFile, 'extractall', real_extractall)
    ds._download()
    assert os.path.exists(ds.boundary_file)


# ---------------------------------------------------------------------------
# features, stations and period
# ---------------------------------------------------------------------------

@VERSIONS
def test_feature_names(ds, version, raw_to_std, n_steps, n_static):
    assert ds.dynamic_features == list(raw_to_std.values())
    static = ds.static_features
    assert len(static) == len(set(static)) == n_static
    for name in STATIC_RENAMED.values():
        assert name in static
    # the slope of the flow duration curve is a streamflow signature; it must
    # not be served as the terrain slope
    assert 'slope_fdc' in static
    assert 'slope_' not in static

    # the categories are those of the published attribute files; version 1 also
    # has static_features.csv (written by older aqua_fetch) in the same folder
    attr_files = glob.glob(os.path.join(os.path.dirname(_raw_attr_file(version, 'soil')), '*_attributes.csv'))
    on_disk = {os.path.basename(f).split('_')[-2] for f in attr_files} - {'groundwaterwell'}
    assert sorted(ds.static_attribute_categories) == sorted(on_disk)


@VERSIONS
def test_stations_match_files(ds, version, raw_to_std, n_steps, n_static):
    ids = sorted(os.path.basename(f).split('_')[-2] for f in _raw_ts_files(version))
    assert ds.stations() == ids
    assert len(ids) == NUM_STATIONS
    assert sorted(ds.fetch_static_features().index) == ids


@VERSIONS
def test_period_matches_file_contents(ds, version, raw_to_std, n_steps, n_static):
    """start/end are read from the files; they must equal the first and last
    date in every file, and the file names must agree with them"""
    firsts, lasts = [], []
    for fpath in _raw_ts_files(version):
        with open(fpath, 'rb') as fp:
            lines = fp.read(200).splitlines()
            first = lines[1].split(b',')[0].decode()
            fp.seek(-200, os.SEEK_END)
            last = fp.read().splitlines()[-1].split(b',')[0].decode()
        # the file name carries the same period, as YYYYMMDD-YYYYMMDD
        named = os.path.basename(fpath)[:-len('.csv')].split('_')[-1].split('-')
        assert [first.replace('-', ''), last.replace('-', '')] == named, fpath
        firsts.append(first)
        lasts.append(last)
    assert ds.start == pd.Timestamp(min(firsts)) == pd.Timestamp('1970-10-01')
    assert ds.end == pd.Timestamp(max(lasts))
    assert ds.end == pd.Timestamp('2022-09-30' if version == 2 else '2015-09-30')
    assert len(pd.date_range(ds.start, ds.end)) == n_steps


# ---------------------------------------------------------------------------
# faithful values and units
# ---------------------------------------------------------------------------

def test_period_of_unusual_files(tmp_path):
    """the first and last timestamp are found even with a trailing blank line or
    a row longer than the tail that is read, and a file without rows is named"""
    ds = _tmp_dataset(dataset, tmp_path / 'CAMELS_GB')
    ts_dir = tmp_path / 'CAMELS_GB' / 'camels_gb_v2' / DAILY
    ts_dir.mkdir(parents=True)
    header = 'date,' + ','.join(V2_RAW_TO_STD) + '\n'
    rows = ['1970-10-01,' + ','.join('1' * 400 for _ in V2_RAW_TO_STD),  # a very long row
            '2022-09-30,' + ','.join('2' for _ in V2_RAW_TO_STD)]
    (ts_dir / 'camels_gb_v2_hydromet_daily_timeseries_1001_19701001-20220930.csv').write_text(
        header + '\n'.join(rows) + '\n\n')  # ends with a blank line
    assert ds._period() == (pd.Timestamp('1970-10-01'), pd.Timestamp('2022-09-30'))

    ds = _tmp_dataset(dataset, tmp_path / 'CAMELS_GB')
    (ts_dir / 'camels_gb_v2_hydromet_daily_timeseries_1002_19701001-20220930.csv').write_text(header)
    with pytest.raises(ValueError, match='1002'):
        ds._period()


@VERSIONS
def test_dynamic_values_unchanged(ds, version, raw_to_std, n_steps, n_static):
    """csv reader and netCDF cache both serve the published values unchanged,
    on the published dates"""
    for stn in _sample(ds.stations(), 5):
        raw = _raw_ts(version, stn)
        assert len(raw) == n_steps
        expected = raw.rename(columns=raw_to_std)

        csv_df = ds._read_stn_dyn(stn)
        _, cached = ds.fetch(stn, as_dataframe=True)
        for df in (csv_df, cached[stn]):
            assert df.index.equals(expected.index), stn
            assert list(df.columns) == list(expected.columns)
            np.testing.assert_array_equal(df.to_numpy(), expected.to_numpy(), err_msg=stn)


@VERSIONS
def test_static_values_unchanged(ds, version, raw_to_std, n_steps, n_static):
    """every attribute of every gauge equals the published text"""
    served = ds.fetch_static_features()
    names = []
    for category in CATEGORIES:
        header, rows = _raw_attr_rows(_raw_attr_file(version, category))
        assert len(rows) == NUM_STATIONS
        for col_idx, col in enumerate(header[1:], start=1):
            name = STATIC_RENAMED.get(col, col)
            names.append(name)
            values = served[name]
            bad = [gid for gid, fields in rows.items() if not _same(values[gid], fields[col_idx])]
            assert not bad, f"{category}:{col} differs for {bad[:5]}"
    assert sorted(names) == sorted(served.columns)


def test_v2_unquoted_commas_in_hydrometry_comments():
    """two published rows have an unquoted comma in the last, free-text column;
    the comment is restored and no other column is shifted"""
    fpath = _raw_attr_file(2, 'hydrometry')
    with pytest.raises(pd.errors.ParserError):
        pd.read_csv(fpath)

    cols = ['station_quality_qmed', 'station_quality_hourlyflow_issues',
            'station_quality_hourlyflow_comment']
    served = dataset.fetch_static_features(['27038', '42010'], cols)
    assert served.loc['27038', 'station_quality_hourlyflow_comment'] == (
        'Flows look unstable and have a clear divide - before 2000s with low mean,'
        ' standard deviation and maximum flows and after')
    assert served.loc['27038', 'station_quality_hourlyflow_issues'] == 'full_continuity_confirmed'
    assert served.loc['42010', 'station_quality_hourlyflow_comment'] == (
        'One of the spreadsheet had very divergent values, recorded every 15 days')
    assert served.loc['42010', 'station_quality_hourlyflow_issues'] == 'partial_other_issues_confirmed'


def test_other_malformed_attribute_rows_raise(tmp_path):
    """only the two known rows of the hydrometry file are repaired"""
    # a known gauge id, but not in the hydrometry file
    soil = tmp_path / 'camels_gb_v2_soil_attributes.csv'
    soil.write_text('gauge_id,sand_perc,clay_perc\n1001,20.5,30.1\n27038,21,5,30.2\n')
    with pytest.raises(ValueError):
        _camels._read_camels_gb_attributes(str(soil))

    with open(_raw_attr_file(2, 'hydrometry'), encoding='utf-8') as fp:
        lines = fp.read().splitlines()
    row_27038 = next(line for line in lines if line.startswith('27038,'))

    # the known row as first data row is repaired too
    first = tmp_path / 'first.csv'
    first.write_text('\n'.join([lines[0], row_27038] + [l for l in lines[1:] if l != row_27038]))
    served = _camels._read_camels_gb_attributes(str(first))
    assert served.loc['27038', 'station_quality_hourlyflow_comment'].endswith('flows and after')

    # the known row with a second extra comma
    two_commas = tmp_path / 'two_commas.csv'
    two_commas.write_text('\n'.join(line + ', again' if line == row_27038 else line for line in lines))
    with pytest.raises(ValueError):
        _camels._read_camels_gb_attributes(str(two_commas))

    # a comma in any other row of the hydrometry file
    other = tmp_path / 'other.csv'
    fields = lines[1].split(',')
    fields[lines[0].split(',').index('quncert_meta')] = 'Calculated, discharge uncertainties'
    other.write_text('\n'.join([lines[0], ','.join(fields)] + lines[2:]))
    with pytest.raises(ValueError):
        _camels._read_camels_gb_attributes(str(other))


def test_v1_humidity_is_specific_humidity():
    """version 1 publishes specific humidity in g kg-1; it used to be served as
    relative humidity ``rh_%``"""
    assert 'rh_%' not in dataset_v1.dynamic_features
    stns = _sample(dataset_v1.stations(), 20)
    _, dyn = dataset_v1.fetch(stns, dynamic_features='spechum_gkg')
    values = np.stack([dyn[stn].values for stn in stns])
    # UK specific humidity is ~2-15 g kg-1; relative humidity would be ~40-100 %
    assert 0 < np.nanmin(values) and np.nanmax(values) < 25


def test_v2_units():
    stns = _sample(dataset.stations(), 20)
    _, dyn = dataset.fetch(stns, as_dataframe=True)
    area = dataset.area(stns)
    for stn in stns:
        df = dyn[stn]
        for col in ('pcp_mm_cehgear', 'pcp_mm_haduk'):
            assert df[col].min() >= 0 and df[col].max() < 400, (stn, col)
        for col in ('pet_mm_chess', 'pet_mm_intercep_chess', 'pet_mm_hydrope', 'pet_mm_intercep_hydrope'):
            assert df[col].min() >= 0 and df[col].max() < 15, (stn, col)
        for col in ('airtemp_C_mean_chess', 'airtemp_C_mean_haduk'):
            assert -30 < df[col].min() and df[col].max() < 40, (stn, col)
        # q_mm_obs (mm day-1) = q_cms_obs (m3 s-1) * 86.4 / area (km2); the
        # published values are rounded to 2 decimals, so larger flows are used
        q = df[df['q_cms_obs'] > 1]
        if len(q) > 100:
            ratio = (q['q_cms_obs'] * 86.4 / area[stn] / q['q_mm_obs']).median()
            assert abs(ratio - 1) < 0.01, (stn, ratio)

    # CEH-GEAR and CHESS end on 2019-12-31
    df = dyn[stns[0]]
    assert df.loc['2020':, ['pcp_mm_cehgear', 'pet_mm_chess', 'airtemp_C_mean_chess']].isna().all().all()


def test_hourly_features_and_values():
    """the hourly data is served with its quality flags, unchanged"""
    assert dataset_h.dynamic_features == list(V2_HOURLY_RAW_TO_STD.values())
    assert dataset_h.stations() == dataset.stations()
    assert dataset_h.static_features == dataset.static_features  # the same attributes

    for stn in _sample(dataset_h.stations(), 3):
        raw = _raw_ts(2, stn, timestep='H')
        expected = raw.rename(columns=V2_HOURLY_RAW_TO_STD)
        df = dataset_h._read_stn_dyn(stn)
        assert df.shape == (HOURLY_STEPS, 7), (stn, df.shape)
        assert df.index.equals(expected.index)
        assert list(df.columns) == list(expected.columns)
        np.testing.assert_array_equal(df.to_numpy(), expected.to_numpy(), err_msg=stn)
        # the flags are whole-numbered quality codes (a station without flags
        # has them as NaN, which makes the column float)
        for col in ('discharge_flag', 'level_flag'):
            codes = df[col].dropna()
            assert (codes == codes.astype('int64')).all(), (stn, col)


def test_hourly_period_and_units():
    # the file names give only dates; the data starts at 09:00 and ends at 08:00,
    # so the period has to come from the files themselves
    assert dataset_h.start == pd.Timestamp('1990-10-01 09:00')
    assert dataset_h.end == pd.Timestamp('2022-10-01 08:00')
    assert len(pd.date_range(dataset_h.start, dataset_h.end, freq='h')) == HOURLY_STEPS
    assert {os.path.basename(f)[:-4].split('_')[-1] for f in _raw_ts_files(2, 'H')} == \
        {'19901001-20221001'}

    stns = _sample(dataset_h.stations(), 3)
    area = dataset_h.area(stns)
    for stn in stns:
        df = dataset_h._read_stn_dyn(stn)
        # every hourly step is there, none is repeated or missing
        assert df.index.to_series().diff().dropna().eq(pd.Timedelta('1h')).all(), stn
        # q_mm_obs (mm hour-1) = q_cms_obs (m3 s-1) * 3.6 / area (km2)
        q = df[df['q_cms_obs'] > 1]
        if len(q) > 100:
            ratio = (q['q_cms_obs'] * 3.6 / area[stn] / q['q_mm_obs']).median()
            assert abs(ratio - 1) < 0.01, (stn, ratio)
        # 101 catchments have no river level at all, so the column can be empty
        level = df['wl_m_obs'].dropna()
        assert level.empty or level.max() < 50, stn  # metres above the river bed
        assert df['pcp_mm_cehgear'].max() < 200, stn  # mm in one hour


def test_hourly_coverage():
    """what the hourly sources really cover, which the dataset's own table gets
    wrong for CEH-GEAR1hr (it says 2019 while every file ends in 2016)"""
    empty_level = 0
    for stn in _sample(dataset_h.stations(), 10) + ['18017']:  # 18017 has no flow
        df = dataset_h._read_stn_dyn(stn)
        assert df['pcp_mm_cehgear'].last_valid_index() == pd.Timestamp('2016-12-31 23:00'), stn
        assert df['pcp_mm_gradgb'].first_valid_index() == pd.Timestamp('2006-01-01 01:00'), stn
        empty_level += df['wl_m_obs'].notna().sum() == 0
    assert empty_level > 0, "expected at least one catchment without river level"
    # the 7 catchments without hourly flow are marked in the attributes
    no_flow = dataset_h.fetch_static_features(static_features=['hourly_flow_perc_complete'])
    assert (no_flow['hourly_flow_perc_complete'] == 0).sum() == 7
    assert dataset_h._read_stn_dyn('18017')['q_cms_obs'].notna().sum() == 0


def test_hourly_fetch_period():
    stn = _sample(dataset_h.stations(), 1)[0]
    _, dyn = dataset_h.fetch(stn, st='2020-01-01', en='2020-01-31 23:00', as_dataframe=True)
    assert dyn[stn].shape == (744, 7)  # 31 days of hourly steps
    assert dyn[stn].index[0] == pd.Timestamp('2020-01-01')


def test_q_mm_is_published_discharge_spec():
    stn = _sample(dataset.stations(), 1)[0]
    q = dataset.q_mm(stn)
    np.testing.assert_array_equal(q[stn].to_numpy(), _raw_ts(2, stn)['discharge_spec'].to_numpy())


def test_fetch_period():
    stn = _sample(dataset.stations(), 1)[0]
    _, dyn = dataset.fetch(stn, st='2020-01-01', en='2020-12-31', as_dataframe=True)
    assert dyn[stn].shape == (366, 10)
    assert dyn[stn].index[0] == pd.Timestamp('2020-01-01')


# ---------------------------------------------------------------------------
# coordinates, boundaries and duplicates
# ---------------------------------------------------------------------------

@VERSIONS
def test_coordinates_in_great_britain(ds, version, raw_to_std, n_steps, n_static):
    coords = ds.stn_coords()
    assert coords.shape == (NUM_STATIONS, 2)
    assert coords['lat'].between(49.8, 61).all() and coords['long'].between(-8.7, 2).all()


@pytest.mark.skipif(fiona is None, reason="fiona is not installed")
@ALL_TIMESTEPS
def test_boundaries(ds, version):
    """every gauge has a boundary. It is served in WGS84 lon/lat around the
    gauge's own coordinates (so the ids are not mixed up), while the published
    British National Grid metres are returned with ``to_wgs84=False``."""
    from aqua_fetch._geom_utils import _make_boundary_2d
    assert set(ds._create_boundary_id_map()) == set(ds.stations())

    coords = ds.stn_coords()
    en = ds.fetch_static_features(static_features=['gauge_easting', 'gauge_northing'])
    for stn in _sample(ds.stations(), 10):
        lon_lat = np.concatenate(_make_boundary_2d(ds.get_boundary(stn)))
        assert lon_lat[:, 0].min() > -8.7 and lon_lat[:, 0].max() < 2, stn
        assert lon_lat[:, 1].min() > 49.8 and lon_lat[:, 1].max() < 61, stn
        # the gauge lies inside the bounding box of its own catchment (a
        # boundary is at most ~1 deg wide, so a mix-up cannot pass this)
        lat, long = coords.loc[stn]
        assert lon_lat[:, 0].min() - 0.02 < long < lon_lat[:, 0].max() + 0.02, stn
        assert lon_lat[:, 1].min() - 0.02 < lat < lon_lat[:, 1].max() + 0.02, stn

        # unconverted, the same boundary is in metres around the gauge's
        # published easting and northing
        xy = np.concatenate(_make_boundary_2d(ds.get_boundary(stn, to_wgs84=False)))
        east, north = en.loc[stn]
        assert xy[:, 0].min() - 1000 < east < xy[:, 0].max() + 1000, stn
        assert xy[:, 1].min() - 1000 < north < xy[:, 1].max() + 1000, stn


def test_duplicate_gauges(tmp_path, monkeypatch):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dataset._warn_duplicate_gauges()
        dataset_v1._warn_duplicate_gauges()
    assert not [x for x in w if 'duplicate' in str(x.message)]

    topo = pd.read_csv(_raw_attr_file(2, 'topographic'))
    doctored = tmp_path / 'topographic.csv'
    pd.concat([topo, topo.iloc[[0]]]).to_csv(doctored, index=False)
    ds = copy.copy(dataset)
    monkeypatch.setattr(ds, '_attr_fpath', lambda category: str(doctored))
    with pytest.warns(UserWarning, match='duplicate'):
        ds._warn_duplicate_gauges()


# ---------------------------------------------------------------------------
# caching, copies and efficiency
# ---------------------------------------------------------------------------

def test_returns_copies():
    stns = dataset.stations()
    stns.append('x')
    assert 'x' not in dataset.stations()

    feats = dataset.dynamic_features
    feats.append('x')
    assert 'x' not in dataset.dynamic_features

    static = dataset._static_data()
    static.loc[:, 'area_km2'] = -1.0
    assert (dataset._static_data()['area_km2'] > 0).all()


def test_process_pool_only_when_it_pays_off(monkeypatch):
    # ~140 MB of csv files: enough for a pool with every start method
    stns = dataset.stations()[:120]
    serial_ds = copy.copy(dataset)
    serial_ds.processes = 1

    real_pool = cf.ProcessPoolExecutor
    used = []

    class SpyPool(real_pool):
        def __init__(self, *args, **kwargs):
            used.append(args)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(cf, 'ProcessPoolExecutor', SpyPool)

    serial = serial_ds._read_dynamic(stns, 'all')
    dataset._read_dynamic(stns[:3], 'all')
    assert used == [], "no pool for processes=1 or for a few stations"

    parallel_ds = copy.copy(dataset)
    parallel_ds.processes = 4
    parallel = parallel_ds._read_dynamic(stns, 'all', st='1980-01-01', en='1989-12-31')
    assert len(used) == 1
    for stn in stns:
        pd.testing.assert_frame_equal(parallel[stn], serial[stn].loc['1980-01-01':'1989-12-31'])


# ---------------------------------------------------------------------------
# the generic rainfall-runoff suite (moved here from test_camels.py)
# ---------------------------------------------------------------------------

def test_generic_suite_v2():
    run_generic_suite(dataset, NUM_STATIONS, 18993, 219, 10)


def test_generic_suite_v1():
    run_generic_suite(dataset_v1, NUM_STATIONS, 16436, 145, 10)


def test_generic_suite_hourly():
    # the suite fetches 2004-01-01 to 2004-12-31, which ends at midnight and so
    # holds 8761 hourly steps; a smaller fraction of the stations is used
    # because one hourly file is ~16 MB
    run_generic_suite(dataset_h, NUM_STATIONS, HOURLY_STEPS, 219, 7,
                      yearly_steps=8761, dyn_fraction=0.01)


@pytest.mark.skipif(xr is None, reason="xarray is not installed")
def test_fetch_all_stations_is_fast():
    start = time.time()
    _, dyn = dataset.fetch('all')
    dyn = dyn.load()  # the netCDF values are read lazily
    elapsed = time.time() - start
    assert len(dyn.data_vars) == NUM_STATIONS
    assert dict(dyn.sizes) == {'time': 18993, 'dynamic_features': 10}
    assert elapsed < 30, f"fetching all stations from the netCDF cache took {elapsed:.0f} s"
