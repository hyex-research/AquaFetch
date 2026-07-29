"""
Tests for the ``NamalValleyPakistan`` (Namal Valley, Pakistan) dataset.

This dataset is a point sensor network, not a set of gauged catchments. It
provides, at a 10-minute time step, precipitation (14 stations) and water level
(8 stations: 7 stream-stage gauges + the Namal Dam lake gauge). It has no
discharge, no catchment areas and no catchment boundaries.

The tests verify that the class

    * downloads the four source ``.xlsx`` files during initialisation, converts
      them to csv once, and does **not** re-download or re-convert on a
      subsequent init (asserted by monkeypatching the network / excel readers to
      raise),
    * exposes 14 stations and exactly two dynamic features
      (``pcp_mm`` incremental precipitation, ``wl_ft_obs`` water level in feet),
      i.e. only the observed 10-minute data and none of the derived
      cumulative / hourly / daily / monthly products,
    * fetches every value **without changing it or its units** (compared against
      a fresh, independent read of the raw excel sheets), including the exact
      missing (NaN) pattern -> no silent data loss,
    * keeps measurement values at full float64 precision,
    * keeps rainfall-only stations' water-level column as all-NaN (not dropped),
    * reports a temporal extent derived from the data (not a hardcoded literal),
    * reads station coordinates from the published inventory, warns (without
      excluding) about duplicate stations, ignores ``to_netcdf`` and raises a
      clear error when ``openpyxl`` is unavailable for the one-time conversion.

The source excel files (~22 MB) are downloaded once to a stable temp directory,
so the whole file runs in ~1 minute on the first run and in a few seconds
thereafter.
"""

import os
import sys
import site
import shutil
import logging
import tempfile
import warnings
import builtins

import numpy as np
import pandas as pd

# add the repository root to the path so that ``aqua_fetch`` can be imported
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

if __name__ == "__main__":
    logging.basicConfig(filename='test_namal.log', filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

import aqua_fetch
from aqua_fetch import NamalValleyPakistan
import aqua_fetch.utils as afu
from aqua_fetch.rr import _misc

# a stable location so the data is downloaded only once across test runs.
# The class appends ``NamalValleyPakistan`` to this path.
NAMAL_PATH = os.path.join(tempfile.gettempdir(), 'aqua_fetch_namal_test')

NUM_STATIONS = 14
EXPECTED_DYN = ['pcp_mm', 'wl_ft_obs']
EXPECTED_STATIC = ['station_name', 'station_type', 'lat', 'long',
                   'elev_gauge_m', 'deploy_date', 'status']
# the 8 water-level stations (7 stream gauges + the Namal Dam lake gauge ND)
WL_STATIONS = {'BW', 'DB', 'KB', 'LW', 'RK', 'GB', 'SF', 'ND'}
RAIN_ONLY = {'BS', 'DK', 'DP', 'KL', 'KH', 'NR'}

# download + convert once (the only network / excel I/O at module import)
dataset = NamalValleyPakistan(path=NAMAL_PATH, verbosity=0)
_PDIR = dataset.path


# ---------------------------------------------------------------------------
# helpers: independent (un-transformed) reads of the raw excel sheets
# ---------------------------------------------------------------------------
_RAW_CACHE = {}


def _raw_sheet(fname, sheet):
    key = (fname, sheet)
    if key not in _RAW_CACHE:
        df = pd.read_excel(os.path.join(_PDIR, fname), sheet_name=sheet, header=0)
        df = df.iloc[3:].copy()                       # drop lat/long/unit rows
        df = df.rename(columns={df.columns[0]: 't'})
        df['t'] = pd.to_datetime(df['t'])
        df = df.set_index('t')
        df = df[~df.index.duplicated(keep='first')].sort_index()
        _RAW_CACHE[key] = df.apply(pd.to_numeric, errors='coerce')
    return _RAW_CACHE[key]


def _raw_precip(stn):
    return _raw_sheet("Namal_Catchment_Precipitation_Data.xlsx",
                      "10- minute precipitation Rate")[stn]


def _raw_stream(stn):
    return _raw_sheet("Namal_Catchment_Stream_Levels_Data.xlsx",
                      "10-minutes Stream Level")[stn]


def _raw_lake():
    return _raw_sheet("Namal_Catchment_Lake_Level_Data.xlsx",
                      "10-minutes Lake Level")["ND"]


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
def test_registration():
    """importable, registered in ALL_DATASETS and the DATASETS factory, and
    actually buildable via ``RainfallRunoff(name)``."""
    logger.info("test_registration")
    assert 'NamalValleyPakistan' in aqua_fetch.ALL_DATASETS
    from aqua_fetch.rr import NamalValleyPakistan as N2, DATASETS
    assert N2 is NamalValleyPakistan
    assert DATASETS['NamalValleyPakistan'] is NamalValleyPakistan
    from aqua_fetch import RainfallRunoff
    rr = RainfallRunoff('NamalValleyPakistan', path=NAMAL_PATH, verbosity=0)
    assert len(rr.stations()) == NUM_STATIONS


def test_stations():
    """14 unique, string station ids; a returned list is a fresh copy."""
    logger.info("test_stations")
    stns = dataset.stations()
    assert len(stns) == NUM_STATIONS
    assert len(set(stns)) == NUM_STATIONS
    assert all(isinstance(s, str) for s in stns)
    assert set(stns) == RAIN_ONLY | WL_STATIONS
    # mutating the returned list must not corrupt the cached state
    stns.append('xxx')
    assert len(dataset.stations()) == NUM_STATIONS


def test_feature_names():
    """exactly two observed dynamic features, and the published static inventory.

    Only the 10-minute incremental precipitation and water level are exposed;
    the derived cumulative / hourly / daily / monthly products are not."""
    logger.info("test_feature_names")
    assert dataset.dynamic_features == EXPECTED_DYN, dataset.dynamic_features
    assert len(dataset.dynamic_features) == 2
    assert dataset.static_features == EXPECTED_STATIC, dataset.static_features
    for f in ('lat', 'long', 'elev_gauge_m'):
        assert f in dataset.static_features


def test_start_end_derived_from_data():
    """the temporal extent is the min/max timestamp across all three raw sheets,
    recomputed independently here so a hardcoded-wrong extent would be caught."""
    logger.info("test_start_end_derived_from_data")
    p = _raw_sheet("Namal_Catchment_Precipitation_Data.xlsx", "10- minute precipitation Rate")
    s = _raw_sheet("Namal_Catchment_Stream_Levels_Data.xlsx", "10-minutes Stream Level")
    lk = _raw_sheet("Namal_Catchment_Lake_Level_Data.xlsx", "10-minutes Lake Level")
    exp_start = min(p.index.min(), s.index.min(), lk.index.min())
    exp_end = max(p.index.max(), s.index.max(), lk.index.max())
    assert dataset.start == exp_start, (dataset.start, exp_start)
    assert dataset.end == exp_end, (dataset.end, exp_end)
    # sanity: 2020-11-25 .. 2024-09-30 (NOT the 2022 of the paper abstract)
    assert dataset.start == pd.Timestamp('2020-11-25 00:00:00')
    assert dataset.end == pd.Timestamp('2024-09-30 23:50:00')


def test_precipitation_values_and_units_unchanged():
    """fetched precipitation equals the raw ``Rate`` sheet (mm per 10-min,
    unchanged, full float64 precision) with the identical missing pattern."""
    logger.info("test_precipitation_values_and_units_unchanged")
    for stn in ('DK', 'LW', 'ND'):
        df = dataset._read_stn_dyn(stn)
        assert list(df.columns) == EXPECTED_DYN
        pcp = df['pcp_mm']
        raw = _raw_precip(stn)
        assert pcp.dtype == np.float64, pcp.dtype
        # identical missing pattern -> no silent drop / fill
        assert int(pcp.notna().sum()) == int(raw.notna().sum()), stn
        m = pd.DataFrame({'f': pcp, 'r': raw}).dropna()
        assert len(m) == int(raw.notna().sum())
        assert np.allclose(m['f'].values, m['r'].values, rtol=0, atol=1e-9), stn


def test_waterlevel_values_and_units_unchanged():
    """stream stage (7 gauges) and lake level (ND) are returned in feet,
    unchanged and NOT converted to cm, with the identical missing pattern."""
    logger.info("test_waterlevel_values_and_units_unchanged")
    # stream stage
    for stn in ('LW', 'BW', 'SF'):
        wl = dataset._read_stn_dyn(stn)['wl_ft_obs']
        raw = _raw_stream(stn)
        assert wl.dtype == np.float64
        assert int(wl.notna().sum()) == int(raw.notna().sum()), stn
        m = pd.DataFrame({'f': wl, 'r': raw}).dropna()
        assert np.array_equal(m['f'].values, m['r'].values), stn
    # lake level (absolute, ~1152-1164 ft) for ND
    wl = dataset._read_stn_dyn('ND')['wl_ft_obs']
    raw = _raw_lake()
    assert int(wl.notna().sum()) == int(raw.notna().sum())
    m = pd.DataFrame({'f': wl, 'r': raw}).dropna()
    assert np.array_equal(m['f'].values, m['r'].values)
    # the lake datum is genuinely different from the stream stage (~<10 ft)
    assert m['f'].min() > 1000.0


def test_rain_only_station_has_nan_waterlevel():
    """rainfall-only stations keep an all-NaN water-level column (not dropped),
    so every station shares the same two-column dynamic schema."""
    logger.info("test_rain_only_station_has_nan_waterlevel")
    for stn in sorted(RAIN_ONLY):
        df = dataset._read_stn_dyn(stn)
        assert list(df.columns) == EXPECTED_DYN
        assert df['wl_ft_obs'].isna().all(), stn
        assert df['pcp_mm'].notna().any(), stn


def test_offgrid_timestamp_preserved():
    """the single off-grid sensor timestamp (2024-03-31 23:49:59.995) is kept
    verbatim rather than being rounded to the 10-minute grid."""
    logger.info("test_offgrid_timestamp_preserved")
    idx = dataset._read_stn_dyn('ND').index
    assert isinstance(idx, pd.DatetimeIndex)
    assert idx.is_monotonic_increasing
    assert pd.Timestamp('2024-03-31 23:49:59.995') in idx


def test_stn_coords_from_inventory():
    """coordinates come from the station inventory and lie in valid ranges."""
    logger.info("test_stn_coords_from_inventory")
    coords = dataset.stn_coords()
    assert coords.shape == (NUM_STATIONS, 2)
    assert set(coords.columns) == {'lat', 'long'}
    assert coords['lat'].between(32.5, 32.8).all()
    assert coords['long'].between(71.7, 72.1).all()
    # a specific station matches the metadata (within float32 tolerance)
    nd = dataset.stn_coords('ND')
    assert abs(float(nd['lat'].iloc[0]) - 32.665219) < 1e-4
    assert abs(float(nd['long'].iloc[0]) - 71.802257) < 1e-4


def test_static_values_match_metadata():
    """static columns carry the published inventory verbatim."""
    logger.info("test_static_values_match_metadata")
    sd = dataset._static_data()
    assert sd.loc['ND', 'station_name'] == 'Namal Dam (ND)'
    assert sd.loc['ND', 'station_type'] == 'R, WL'
    assert sd.loc['BS', 'station_type'] == 'R'
    assert abs(float(sd.loc['DP', 'elev_gauge_m']) - 653.0) < 1e-6
    # every water-level station is flagged 'R, WL'; rain-only ones 'R'
    for stn in WL_STATIONS:
        assert 'WL' in sd.loc[stn, 'station_type'], stn
    for stn in RAIN_ONLY:
        assert sd.loc[stn, 'station_type'] == 'R', stn


def test_fetch_modes_and_slicing():
    """multi-station fetch, dynamic-feature subsetting, date slicing, the empty
    range and the default xarray path."""
    logger.info("test_fetch_modes_and_slicing")
    # single precipitation feature for two stations
    _, dyn = dataset.fetch(['BS', 'LW'], dynamic_features='pcp_mm', as_dataframe=True)
    assert isinstance(dyn, dict) and set(dyn) == {'BS', 'LW'}
    assert dyn['BS'].shape[1] == 1 and list(dyn['BS'].columns) == ['pcp_mm']

    # date slicing narrows the index
    _, dyn = dataset.fetch('ND', st='2023-01-01', en='2023-12-31', as_dataframe=True)
    idx = dyn['ND'].index
    assert idx.min() >= pd.Timestamp('2023-01-01')
    assert idx.max() <= pd.Timestamp('2023-12-31 23:59:59')
    assert len(idx) > 0

    # out-of-range window -> empty frame, not a crash
    _, dyn = dataset.fetch('ND', st='2015-01-01', en='2015-02-01', as_dataframe=True)
    assert dyn['ND'].shape == (0, 2)

    # default (xarray) path
    try:
        import xarray  # noqa: F401
    except Exception:
        return
    _, dsx = dataset.fetch(['ND', 'BS'], as_dataframe=False)
    assert type(dsx).__name__ == 'Dataset'
    assert len(dsx.data_vars) == 2
    assert dict(dsx.sizes)['dynamic_features'] == 2


def test_fetch_all_stations():
    """all 14 stations fetch to the same two-column schema."""
    logger.info("test_fetch_all_stations")
    _, dyn = dataset.fetch('all', as_dataframe=True)
    assert len(dyn) == NUM_STATIONS
    for stn, df in dyn.items():
        assert list(df.columns) == EXPECTED_DYN, stn
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.is_monotonic_increasing, stn


def test_no_redownload_no_reconvert(monkeypatch):
    """re-initialising with the data already on disk must NOT hit the network
    nor re-read the excel files."""
    logger.info("test_no_redownload_no_reconvert")

    def _boom_dl(*a, **k):
        raise AssertionError("download() called when data already present")

    def _boom_xls(self, *a, **k):
        raise AssertionError("_read_10min_sheet called when csv already present")

    monkeypatch.setattr(afu, 'download', _boom_dl)
    monkeypatch.setattr(NamalValleyPakistan, '_read_10min_sheet', _boom_xls)
    ds2 = NamalValleyPakistan(path=NAMAL_PATH, verbosity=0)
    assert len(ds2.stations()) == NUM_STATIONS
    assert ds2._read_stn_dyn('BS').shape[1] == 2


def test_overwrite_reprocesses(monkeypatch, tmp_path):
    """overwrite=True removes and re-fetches every source file and rebuilds the
    csv cache. The (mocked) downloader copies the already-cached xlsx so no
    network is used but the real conversion still runs."""
    logger.info("test_overwrite_reprocesses")
    calls = {'n': 0}

    def _fake_download(url, outdir=None, fname=None, verbosity=0):
        calls['n'] += 1
        os.makedirs(outdir, exist_ok=True)
        shutil.copy(os.path.join(_PDIR, fname), os.path.join(outdir, fname))

    monkeypatch.setattr(afu, 'download', _fake_download)
    # first init -> 4 source files fetched and converted
    ds = NamalValleyPakistan(path=str(tmp_path), verbosity=0)
    assert calls['n'] == 4, calls['n']
    assert len(ds.stations()) == NUM_STATIONS
    # a stale marker must be wiped by overwrite
    with open(os.path.join(ds.path, 'STALE.txt'), 'w') as f:
        f.write('x')
    ds2 = NamalValleyPakistan(path=str(tmp_path), overwrite=True, verbosity=0)
    assert calls['n'] == 8, calls['n']
    assert not os.path.exists(os.path.join(ds2.path, 'STALE.txt'))
    assert len(ds2.stations()) == NUM_STATIONS


def test_remove_zip_reclaims_xlsx_without_forcing_redownload(monkeypatch, tmp_path):
    """remove_zip=True deletes the source xlsx once the csv cache is built, and a
    later init (xlsx gone, csv present) must NOT re-download."""
    logger.info("test_remove_zip_reclaims_xlsx_without_forcing_redownload")

    def _fake_download(url, outdir=None, fname=None, verbosity=0):
        os.makedirs(outdir, exist_ok=True)
        shutil.copy(os.path.join(_PDIR, fname), os.path.join(outdir, fname))

    monkeypatch.setattr(afu, 'download', _fake_download)
    ds = NamalValleyPakistan(path=str(tmp_path), remove_zip=True, verbosity=0)
    files = os.listdir(ds.path)
    assert not any(f.endswith('.xlsx') for f in files), files      # xlsx reclaimed
    assert any(f.endswith('.csv') for f in files)                  # csv kept

    # xlsx gone, csv present -> re-init must not hit the (now-raising) downloader
    monkeypatch.setattr(afu, 'download',
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("re-download despite csv cache")))
    ds2 = NamalValleyPakistan(path=str(tmp_path), verbosity=0)
    assert len(ds2.stations()) == NUM_STATIONS
    assert ds2._read_stn_dyn('ND').shape[1] == 2


def test_flat_gauge_warns_on_first_precip_read():
    """station GB records precipitation that is entirely zero; the class warns
    (unconditionally) on first precipitation read — including on the common
    cached-read path — while keeping the values unchanged (not dropped)."""
    logger.info("test_flat_gauge_warns_on_first_precip_read")
    # a fresh instance -> its precipitation cache is empty, so first read warns
    ds = NamalValleyPakistan(path=NAMAL_PATH, verbosity=0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ds.fetch(['GB'], dynamic_features='pcp_mm', as_dataframe=True)
    assert any('GB' in str(x.message) and 'zero' in str(x.message) for x in w), \
        [str(x.message) for x in w]
    # values are kept unchanged (all zero, not dropped / not filled)
    gb = ds._read_stn_dyn('GB')['pcp_mm']
    assert gb.notna().sum() > 0 and (gb.dropna() == 0).all()


def test_duplicate_check_warns_but_keeps():
    """the duplicate check (name + rounded coords) warns without excluding.

    The real inventory has no duplicates, so it must NOT warn; a crafted
    duplicate must warn -> the check can actually fail."""
    logger.info("test_duplicate_check_warns_but_keeps")
    # real inventory: no duplicates -> no warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        NamalValleyPakistan._warn_duplicates(dataset._static_data())
        assert not any('duplicate' in str(x.message).lower() for x in w)
    # crafted duplicate -> warns
    dup = pd.DataFrame({
        'station_name': ['A (A)', 'A (A)'],
        'lat': [32.1, 32.1], 'long': [71.1, 71.1],
    }, index=['A', 'B'])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        NamalValleyPakistan._warn_duplicates(dup)
        assert any('duplicate' in str(x.message).lower() for x in w)


def test_unavailable_methods_raise_clearly():
    """this point-sensor network has no areas / discharge / boundaries; the
    corresponding accessors fail with a clear NotImplementedError, not a cryptic
    crash."""
    logger.info("test_unavailable_methods_raise_clearly")
    for call in (lambda: dataset.area('ND'),
                 lambda: dataset.q_mm('ND'),
                 lambda: dataset.get_boundary('ND')):
        try:
            call()
            raise AssertionError("expected NotImplementedError")
        except NotImplementedError:
            pass


def test_to_netcdf_does_not_write_nc():
    """the 10-minute data is served from csv; no .nc file is produced even when
    ``to_netcdf=True`` is requested and a fetch is performed."""
    logger.info("test_to_netcdf_does_not_write_nc")
    ds = NamalValleyPakistan(path=NAMAL_PATH, to_netcdf=True, verbosity=0)
    ds.fetch(['BS'], as_dataframe=True)
    nc = [f for f in os.listdir(_PDIR) if f.endswith('.nc')]
    assert nc == [], nc


def test_openpyxl_required(monkeypatch, tmp_path):
    """when the csv cache is absent and openpyxl cannot be imported, the
    one-time conversion raises a clear, actionable ImportError."""
    logger.info("test_openpyxl_required")
    # seed a fresh dir with only the source xlsx (no csv cache) so _process runs
    pdir = os.path.join(str(tmp_path), 'NamalValleyPakistan')
    os.makedirs(pdir, exist_ok=True)
    for f in os.listdir(_PDIR):
        if f.endswith('.xlsx'):
            shutil.copy(os.path.join(_PDIR, f), os.path.join(pdir, f))

    real_import = builtins.__import__

    def _no_openpyxl(name, *a, **k):
        if name == 'openpyxl':
            raise ImportError("no openpyxl")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, '__import__', _no_openpyxl)
    monkeypatch.setitem(sys.modules, 'openpyxl', None)
    sys.modules.pop('openpyxl', None)
    try:
        NamalValleyPakistan(path=str(tmp_path), verbosity=0)
        raise AssertionError("expected ImportError when openpyxl is missing")
    except ImportError as e:
        assert 'openpyxl' in str(e)


if __name__ == "__main__":
    import time
    t0 = time.time()
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            import inspect
            params = inspect.signature(fn).parameters
            if params:
                logger.info(f"skipping {name} (needs pytest fixtures) in __main__ run")
                continue
            print(f"running {name} ...", flush=True)
            fn()
    print(f"\nall no-fixture tests passed in {time.time()-t0:.1f}s")
