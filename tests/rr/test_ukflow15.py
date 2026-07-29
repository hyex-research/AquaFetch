"""
Tests for the UKFlow15 (UK, 1369 gauges, 15-min river flow) dataset.

UK-Flow15 differs from the CAMELS-style datasets: it provides observed river
discharge only (no meteorological forcing, no catchment boundaries) at 15-min
resolution over 1948-2023, and the full archive is ~64 GB. The class therefore
downloads only the small (~0.4 MB) metadata at initialisation and fetches each
station's flow file lazily on first access.

The tests verify that the class

    * downloads the metadata during initialisation and does **not** re-download
      it (nor an already-cached station file) on a subsequent access,
    * exposes the correct number of stations (1369), dynamic (3) and static
      (11) features,
    * fetches the discharge time series **without changing the values or their
      units** (compared against a fresh, independent read of the raw csv),
    * preserves the zero-padded 3-digit QC ``flag`` (leading zeros are
      meaningful) and the missing (``NaN``) observations,
    * reads discharge at full float64 precision,
    * reports a temporal extent derived from the data (not a hardcoded literal),
    * converts British National Grid coordinates to WGS84 correctly, computes
      catchment area and q_mm, and exposes the QC metadata unchanged.

Most tests use only the metadata plus two of the smallest station files
(``024007`` ~12 KB and ``069011`` ~1.6 MB). One robustness test additionally
fetches a reproducible random sample of 100 shorter-record stations (~2.4 GB,
downloaded in parallel and cached), so the whole file runs in ~1.5 minutes on
the first run and in under a minute thereafter.
"""

import os
import site
import random
import shutil
import logging
import tempfile
import warnings

import numpy as np
import pandas as pd

# add the repository root to the path so that ``aqua_fetch`` can be imported
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

if __name__ == "__main__":
    logging.basicConfig(filename='test_ukflow15.log', filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

import aqua_fetch
from aqua_fetch import UKFlow15
from aqua_fetch.rr import _ukflow15
from aqua_fetch._geom_utils import osgb36_to_wgs84

# a stable location so the metadata is downloaded only once across test runs.
# The class appends ``UKFlow15`` to this path; we deliberately do NOT point at
# any real data directory (that would append the name again / trigger re-downloads).
UKFLOW15_PATH = os.path.join(tempfile.gettempdir(), 'aqua_fetch_ukflow15_test')

NUM_STATIONS = 1369
NUM_STATIC = 11
EXPECTED_DYN = ['q_cms_obs', 'resolution', 'flag']

# two of the smallest station files -> fast to download & parse
SMALL = '24007'    # 024007.csv, ~12 KB, 356 rows, flags 100/500
GAPS = '69011'     # 069011.csv, ~1.6 MB, 48860 rows, flags 000/002/NaN

# pyproj-independent reference coordinates (EPSG:27700 -> EPSG:4326), computed
# with pyproj offline; the pure-python helper matches these to ~4 mm.
REF_COORDS = {
    '1001':  (58.476196, -3.267060),
    '24007': (54.810458, -1.744724),
    '69011': (53.396728, -2.219374),
}

# download the metadata once (this is the only network I/O at init)
dataset = UKFlow15(path=UKFLOW15_PATH, verbosity=0)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _raw_ts(stn) -> pd.DataFrame:
    """independent read of one already-cached raw station csv (no transform)."""
    stem = str(stn).zfill(6)
    fpath = os.path.join(dataset.flow_dir, f"{stem}.csv")
    return pd.read_csv(fpath, dtype={'flag': str, 'datetime': str})


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_registration():
    """the class is importable, registered, and buildable via the factory."""
    logger.info("test_registration")
    assert 'UKFlow15' in aqua_fetch.ALL_DATASETS
    from aqua_fetch.rr import DATASETS
    assert DATASETS['UKFlow15'] is UKFlow15
    from aqua_fetch import RainfallRunoff
    rr = RainfallRunoff('UKFlow15', path=UKFLOW15_PATH, verbosity=0)
    assert len(rr.stations()) == NUM_STATIONS


def test_stations():
    """1369 unique, non zero-padded station ids that map 1:1 onto the flow files."""
    logger.info("test_stations")
    stns = dataset.stations()
    assert len(stns) == NUM_STATIONS
    assert len(set(stns)) == NUM_STATIONS, "duplicate station ids"
    assert all(isinstance(s, str) for s in stns)
    # non zero-padded (matching NRFA / CAMELS_GB convention)
    assert '1001' in stns and '001001' not in stns
    # returns a fresh list -> mutating it must not corrupt the cached state
    stns.append('xxx')
    assert len(dataset.stations()) == NUM_STATIONS


def test_feature_names():
    """dynamic features are exactly the 3 expected and static has area/coords."""
    logger.info("test_feature_names")
    assert dataset.dynamic_features == EXPECTED_DYN, dataset.dynamic_features
    sf = dataset.static_features
    assert len(sf) == NUM_STATIC, sf
    for f in ('area_km2', 'lat', 'long', 'Easting', 'Northing'):
        assert f in sf, f
    # reading the dynamic feature names must not download a station file
    unexpected = [f for f in os.listdir(dataset.flow_dir)]
    _ = dataset.dynamic_features
    assert os.listdir(dataset.flow_dir) == unexpected, "dynamic_features triggered a download"


def test_start_end_from_data():
    """the temporal extent is derived from the metadata (min Start_date / max
    End_date). The expected values are recomputed here from an independent read
    of the raw metadata, so a hardcoded-wrong extent would be caught."""
    logger.info("test_start_end_from_data")
    meta = pd.read_csv(os.path.join(dataset.meta_dir, '00_station_id_meta.csv'))
    exp_start = pd.to_datetime(meta['Start_date'], format='%d/%m/%Y %H:%M').min()
    exp_end = pd.to_datetime(meta['End_date'], format='%d/%m/%Y %H:%M').max()
    assert dataset.start == exp_start, (dataset.start, exp_start)
    assert dataset.end == exp_end, (dataset.end, exp_end)
    # and they match the dataset's documented 1948-2023 coverage
    assert exp_start == pd.Timestamp('1948-08-09 22:45:00')
    assert exp_end == pd.Timestamp('2023-12-31 23:45:00')


def test_fetch_values_and_units_unchanged():
    """fetched discharge equals the raw ``value`` column (units m3/s unchanged,
    no scaling), on a continuous 15-min index. Discharge is returned at the
    configured float precision (float32 by default) which losslessly preserves
    the source's 3-decimal values."""
    logger.info("test_fetch_values_and_units_unchanged")
    _, dyn = dataset.fetch(SMALL, as_dataframe=True)
    df = dyn[SMALL]
    assert list(df.columns) == EXPECTED_DYN
    assert df.shape == (356, 3), df.shape

    raw = _raw_ts(SMALL)
    # default float32 preserves the source's 3-decimal values exactly (no
    # scaling / unit change / meaningful precision loss)
    assert df['q_cms_obs'].dtype == dataset.fp
    np.testing.assert_array_equal(
        np.round(df['q_cms_obs'].values.astype(np.float64), 3),
        np.round(raw['value'].values.astype(np.float64), 3))
    # opting into float64 gives byte-exact values
    ds64 = UKFlow15(path=UKFLOW15_PATH, verbosity=0, float_precision=np.float64)
    d64 = ds64._read_stn_dyn(SMALL)
    assert d64['q_cms_obs'].dtype == np.float64
    np.testing.assert_array_equal(d64['q_cms_obs'].values, raw['value'].values)
    # index is a strictly-increasing 15-min DatetimeIndex
    assert isinstance(df.index, pd.DatetimeIndex)
    steps = df.index.to_series().diff().dropna().unique()
    assert len(steps) == 1 and steps[0] == pd.Timedelta('15min'), steps


def test_fetch_many_random_stations():
    """Robustness: fetch a reproducible random sample of 100 stations (not just
    the 2 used elsewhere) and verify the parse is faithful for every one.

    The sample is drawn from the 250 shortest-record stations only, to bound the
    download on this ~64 GB per-station-download dataset while still exercising
    100 distinct real files spanning a few hundred to several-hundred-thousand
    rows (including a NaN-area station). Fetching >1 station also drives the
    base class's multiprocessing path."""
    logger.info("test_fetch_many_random_stations")
    sd = dataset._static_data()
    est_rows = (
        (pd.to_datetime(sd['End_date'], format='%d/%m/%Y %H:%M')
         - pd.to_datetime(sd['Start_date'], format='%d/%m/%Y %H:%M')
         ).dt.total_seconds() / 900.0)
    pool = est_rows.sort_values(kind='stable').index[:250].tolist()
    stns = sorted(random.Random(15).sample(pool, 100))

    # a moderate worker count keeps concurrent load on the CEH datastore low
    # (the module-level downloader also retries transient HTTP errors)
    ds = UKFlow15(path=UKFLOW15_PATH, processes=6, verbosity=0)
    _, dyn = ds.fetch(stns, as_dataframe=True)
    assert len(dyn) == 100, len(dyn)

    for stn in stns:
        df = dyn[stn]
        assert list(df.columns) == EXPECTED_DYN, (stn, list(df.columns))
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.is_monotonic_increasing, stn
        assert df['q_cms_obs'].dtype == dataset.fp, stn
        # every non-null flag is a zero-padded 3-digit QC code string
        flags = df['flag'].dropna().unique()
        assert all(isinstance(f, str) and len(f) == 3 and f.isdigit()
                   for f in flags), (stn, flags[:5])
        # values + missing pattern faithful vs an independent raw read of the
        # class-downloaded file (float32 preserved to the source's 3 decimals)
        raw = _raw_ts(stn)
        assert len(df) == len(raw), (stn, len(df), len(raw))
        np.testing.assert_array_equal(
            np.round(df['q_cms_obs'].values.astype(np.float64), 3),
            np.round(raw['value'].values.astype(np.float64), 3), err_msg=stn)
        assert np.array_equal(df['q_cms_obs'].isna().values,
                              raw['value'].isna().values), stn


def test_flag_leading_zeros_and_missing():
    """the 3-digit QC flag keeps its leading zeros (read as string) and missing
    observations (value/flag) are preserved as NaN, matching the raw file."""
    logger.info("test_flag_leading_zeros_and_missing")
    df = dataset._read_stn_dyn(GAPS)
    raw = _raw_ts(GAPS)

    # flags are zero-padded strings, NOT integers 0/2
    present = set(df['flag'].dropna().unique())
    assert present == {'000', '002'}, present
    assert '0' not in present and '2' not in present
    assert df['flag'].dtype == object

    # exact match against the raw (string) flag column, NaNs included
    assert df['flag'].isna().sum() == raw['flag'].isna().sum()
    pd.testing.assert_series_equal(
        df['flag'].reset_index(drop=True),
        raw['flag'].reset_index(drop=True),
        check_names=False)

    # missing discharge preserved (not dropped, not filled)
    assert df['q_cms_obs'].isna().sum() == raw['value'].isna().sum() == 4
    # native-resolution provenance retained (30-min steps exist)
    assert 30.0 in set(df['resolution'].dropna().astype(float).unique())


def test_area():
    """catchment area (km2) matches the source; the 9 area-less stations
    surface as NaN rather than being silently dropped."""
    logger.info("test_area")
    a = dataset.area(SMALL)
    assert abs(float(a.iloc[0]) - 44.6) < 0.1, a
    all_area = dataset.area('all')
    assert len(all_area) == NUM_STATIONS
    assert int(all_area.isna().sum()) == 9


def test_osgb_to_wgs84_and_stn_coords():
    """British National Grid -> WGS84 conversion is accurate and all gauges lie
    within the UK bounding box."""
    logger.info("test_osgb_to_wgs84_and_stn_coords")
    for sid, (rlat, rlon) in REF_COORDS.items():
        c = dataset.stn_coords(sid)
        assert abs(float(c['lat'].iloc[0]) - rlat) < 1e-3, (sid, c)
        assert abs(float(c['long'].iloc[0]) - rlon) < 1e-3, (sid, c)
    coords = dataset.stn_coords('all')
    assert coords['lat'].between(49.5, 61.0).all()
    assert coords['long'].between(-8.5, 2.0).all()
    # direct helper check (scalar in, scalar-array out)
    lat, lon = osgb36_to_wgs84(np.array([326202.0]), np.array([954915.0]))
    assert abs(float(lat[0]) - 58.476196) < 1e-3
    assert abs(float(lon[0]) - (-3.267060)) < 1e-3


def test_q_mm():
    """q_mm is the correct cms->mm/15-min conversion using catchment area."""
    logger.info("test_q_mm")
    q = dataset.q_mm(SMALL)
    assert q.shape == (356, 1)
    _, dyn = dataset.fetch(SMALL, dynamic_features='q_cms_obs', as_dataframe=True)
    q_cms = dyn[SMALL]['q_cms_obs']
    area_m2 = float(dataset.area(SMALL).iloc[0]) * 1e6
    expected = (q_cms * 900.0 / area_m2) * 1e3  # cms -> mm per 15 min
    np.testing.assert_allclose(q[SMALL].values, expected.values, rtol=1e-5)


def test_qc_metadata():
    """the QC/traceability tables are returned unchanged, cached, and copied."""
    logger.info("test_qc_metadata")
    expected_shapes = {
        'common_sense_anomalies': (327, 4),
        'uk_products': (1369, 11),
        'traditional_qc': (1369, 9),
        'high_flows_qc': (1369, 12),
        'resolution': (1369, 8),
        'duplicates': (66, 3),
    }
    for name, shape in expected_shapes.items():
        assert dataset.qc_metadata(name).shape == shape, name
    # unknown name rejected
    try:
        dataset.qc_metadata('bogus')
        raise AssertionError("bogus name should raise ValueError")
    except ValueError:
        pass
    # returns a copy: mutating the result must not corrupt the cache
    t = dataset.qc_metadata('traditional_qc')
    t.iloc[0, 1] = -999
    assert dataset.qc_metadata('traditional_qc').iloc[0, 1] != -999


def test_no_redownload(monkeypatch):
    """re-initialising with the metadata already on disk, and reading an
    already-cached station file, must NOT hit the network."""
    logger.info("test_no_redownload")

    def _boom(*a, **k):
        raise AssertionError("download() called when data is already cached")

    # ensure the small station is cached first (may download once here)
    dataset.fetch(SMALL, as_dataframe=True)

    monkeypatch.setattr(_ukflow15, 'download', _boom)
    # re-init: metadata is present -> no download
    ds2 = UKFlow15(path=UKFLOW15_PATH, verbosity=0)
    assert len(ds2.stations()) == NUM_STATIONS
    # reading a cached station -> no download
    df = ds2._read_stn_dyn(SMALL)
    assert df.shape[0] == 356


def test_overwrite_redownloads_metadata(monkeypatch, tmp_path):
    """overwrite=True must remove and re-download every metadata file."""
    logger.info("test_overwrite_redownloads_metadata")
    calls = {'n': 0}

    def _fake_download(url=None, outdir=None, fname=None, verbosity=0):
        calls['n'] += 1
        os.makedirs(outdir, exist_ok=True)
        # write a minimal placeholder so the "exists" gate is satisfied
        with open(os.path.join(outdir, fname), 'w') as f:
            f.write("station_id\n1001\n")

    monkeypatch.setattr(_ukflow15, 'download', _fake_download)
    # first init downloads all 7 metadata files
    UKFlow15(path=str(tmp_path), verbosity=0)
    assert calls['n'] == 7, calls['n']
    # overwrite re-downloads all 7 again (old files removed first)
    UKFlow15(path=str(tmp_path), overwrite=True, verbosity=0)
    assert calls['n'] == 14, calls['n']


def test_boundary_not_implemented():
    """UK-Flow15 ships no catchment boundaries; the accessor fails clearly."""
    logger.info("test_boundary_not_implemented")
    try:
        dataset.get_boundary(SMALL)
        raise AssertionError("get_boundary should raise NotImplementedError")
    except NotImplementedError:
        pass


def test_to_netcdf_ignored():
    """to_netcdf=True is refused (with a warning) for this 64 GB 15-min dataset."""
    logger.info("test_to_netcdf_ignored")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds = UKFlow15(path=UKFLOW15_PATH, to_netcdf=True, verbosity=0)
        assert any("netCDF" in str(x.message) for x in w)
    assert ds.to_netcdf is False


def test_duplicate_warning():
    """potential duplicate stations are warned about, not excluded."""
    logger.info("test_duplicate_warning")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds = UKFlow15(path=UKFLOW15_PATH, verbosity=1)
        # force the (cached-once) duplicate check to run on this instance
        ds._static_data()
        assert any("duplicate" in str(x.message).lower() for x in w)
    # all stations retained despite duplicates
    assert len(ds.stations()) == NUM_STATIONS


def test_fetch_modes_and_slicing():
    """multi-station fetch, dynamic-feature subsetting, date slicing, the empty
    range, and the default xarray path (with the string ``flag`` column)."""
    logger.info("test_fetch_modes_and_slicing")
    # multi-station + dynamic subset
    _, dyn = dataset.fetch([SMALL, GAPS], dynamic_features=['q_cms_obs'],
                           as_dataframe=True)
    assert {k: v.shape for k, v in dyn.items()} == {SMALL: (356, 1), GAPS: (48860, 1)}

    # date slicing narrows the returned index
    _, dyn = dataset.fetch(SMALL, st='1971-08-14', en='1971-08-15',
                           as_dataframe=True)
    idx = dyn[SMALL].index
    assert idx.min() >= pd.Timestamp('1971-08-14')
    assert idx.max() <= pd.Timestamp('1971-08-15 23:59:59')
    assert len(idx) > 0

    # an out-of-range window yields an empty frame, not a crash
    _, dyn = dataset.fetch(SMALL, st='2000-01-01', en='2001-01-01',
                           as_dataframe=True)
    assert dyn[SMALL].shape == (0, 3)

    # default (xarray) path must handle the mixed float+string columns
    try:
        import xarray  # noqa: F401
    except Exception:
        return
    _, dsx = dataset.fetch(SMALL, as_dataframe=False)
    assert type(dsx).__name__ == 'Dataset'
    assert dict(dsx.sizes) == {'time': 356, 'dynamic_features': 3}


def test_q_mm_nan_area_warns(monkeypatch):
    """the 9 area-less stations are retained (not dropped); q_mm for them warns
    unconditionally and returns NaN rather than silently substituting."""
    logger.info("test_q_mm_nan_area_warns")
    area_all = dataset.area('all')
    nan_stns = area_all[area_all.isna()].index.tolist()
    assert len(nan_stns) == 9, nan_stns
    stn = nan_stns[0]

    # avoid downloading this station's (potentially large) real flow file
    idx = pd.date_range('2000-01-01', periods=4, freq='15min', name='time')
    dummy = pd.DataFrame({'q_cms_obs': [1.0, 2.0, 3.0, 4.0],
                          'resolution': [15.0] * 4, 'flag': ['000'] * 4}, index=idx)
    dummy.columns.name = 'dynamic_features'
    monkeypatch.setattr(dataset, '_read_stn_dyn', lambda s: dummy)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        q = dataset.q_mm(stn)
        assert any('area' in str(x.message).lower() for x in w), [str(x.message) for x in w]
    assert q[stn].isna().all()


def test_download_all_stations(monkeypatch):
    """download_all_stations early-returns for cached stations and calls the
    (mocked) downloader once per uncached station, with the ~64 GB warning."""
    logger.info("test_download_all_stations")
    # make sure the small station is cached (a real ~12 KB download at most)
    dataset.fetch(SMALL, as_dataframe=True)

    made = []

    def _fake_dl(stem, flow_dir, verbosity=0):
        made.append(stem)
        open(os.path.join(flow_dir, f"{stem}.csv"), 'a').close()

    monkeypatch.setattr(_ukflow15, '_download_station_file', _fake_dl)

    # already-cached -> early return, downloader not called
    dataset.download_all_stations(stations=[SMALL])
    assert made == []

    # a valid but uncached station -> downloaded once, with a size warning
    uncached_fp = os.path.join(dataset.flow_dir, '002001.csv')
    if os.path.exists(uncached_fp):
        os.remove(uncached_fp)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            dataset.download_all_stations(stations=['2001'])
            assert any('GB' in str(x.message) for x in w)
        assert made == ['002001']
    finally:
        # remove the empty placeholder so it can't corrupt other reads
        if os.path.exists(uncached_fp):
            os.remove(uncached_fp)


def test_download_all_stations_overwrite_parallel():
    """download_all_stations with overwrite=True and processes>1 removes the
    cached files and re-downloads them through a process pool. Uses only the
    two small stations, so it stays fast and leaves a valid cache."""
    logger.info("test_download_all_stations_overwrite_parallel")
    # ensure both small stations are present first
    dataset.download_all_stations(stations=[SMALL, GAPS])
    paths = {s: os.path.join(dataset.flow_dir, str(s).zfill(6) + '.csv')
             for s in (SMALL, GAPS)}
    assert all(os.path.exists(p) for p in paths.values())

    # overwrite=True + processes=2 -> the parallel re-download branch
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dataset.download_all_stations(stations=[SMALL, GAPS],
                                      processes=2, overwrite=True)
        assert any('GB' in str(x.message) for x in w), "expected the ~64 GB warning"

    # files still present and faithful after the parallel re-download
    assert all(os.path.exists(p) for p in paths.values())
    assert dataset._read_stn_dyn(SMALL).shape == (356, 3)
    assert dataset._read_stn_dyn(GAPS).shape == (48860, 3)


def test_overwrite_refreshes_station_file(monkeypatch, tmp_path):
    """overwrite=True refreshes a cached station file end-to-end (removes the
    stale copy and re-downloads it) exactly once per station per session."""
    logger.info("test_overwrite_refreshes_station_file")

    # a self-contained instance with its own metadata (mocked) and one cached
    # station file, so we don't touch the shared cache.
    def _fake_meta_download(url=None, outdir=None, fname=None, verbosity=0):
        os.makedirs(outdir, exist_ok=True)
        # a minimal but valid 00_station_id_meta.csv is enough for init
        with open(os.path.join(outdir, fname), 'w') as f:
            if fname == '00_station_id_meta.csv':
                f.write("station_id,River,Location,Start_date,End_date,"
                        "Missing_values_%,Catchment_Area,Easting,Northing,"
                        "nrfa_quality_status_v14\n"
                        "2001,R,L,21/06/1974 09:00,31/12/2023 23:45,0,551.4,"
                        "299839,918163,Pooling\n")
            else:
                f.write("station_id\n2001\n")

    monkeypatch.setattr(_ukflow15, 'download', _fake_meta_download)
    ds = UKFlow15(path=str(tmp_path), overwrite=True, verbosity=0)

    stem = '002001'
    fp = os.path.join(ds.flow_dir, f'{stem}.csv')
    # place a stale cached file
    with open(fp, 'w') as f:
        f.write("datetime,value,resolution,flag\n2000-01-01 00:00:00,9.9,15,000\n")

    dl_calls = []

    def _fake_station_dl(s, flow_dir, verbosity=0):
        dl_calls.append(s)
        with open(os.path.join(flow_dir, f"{s}.csv"), 'w') as f:
            f.write("datetime,value,resolution,flag\n2000-01-01 00:00:00,1.1,15,000\n")

    monkeypatch.setattr(_ukflow15, '_download_station_file', _fake_station_dl)

    # first read with overwrite=True: stale file removed and re-downloaded
    df1 = ds._read_stn_dyn('2001')
    assert dl_calls == [stem], dl_calls
    assert abs(float(df1['q_cms_obs'].iloc[0]) - 1.1) < 1e-4  # fresh copy, not the stale 9.9
    # second read: not refreshed again (once per session)
    ds._read_stn_dyn('2001')
    assert dl_calls == [stem], dl_calls


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, '-v', '-s'])
