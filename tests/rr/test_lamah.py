"""
Integration tests for the LamaH rainfall-runoff datasets:

    * ``LamaHCE``  - LamaH-CE, Central Europe (mainly Austria)
    * ``LamaHIce`` - LamaH-Ice, Iceland

Both datasets are exercised at daily (``'D'``) and hourly (``'H'``) timestep and
for the three basin delineations (``total_upstrm`` / ``intermediate_all`` /
``intermediate_lowimp``). Every combination is run through the shared,
comprehensive :func:`utils.test_dataset` suite, which checks the fetching
fidelity, the static/dynamic feature counts, coordinates, boundaries, ``q_mm``
and that no re-download/re-extraction happens.

On top of that, the ``test_lamahce_*`` functions below compare what
:class:`aqua_fetch.LamaHCE` returns against an independent read of the raw csv
files, so that a change of the reading code that silently alters values, units
or timestamps is caught.

Run directly (``python test_lamah.py``); the data is expected to be already
downloaded under ``GSCAD_PATH``.
"""

import os
import site
import shutil
import logging
import tempfile
import warnings

import numpy as np
import pandas as pd

# add the repository root to the path so that ``aqua_fetch`` and the shared
# ``utils`` test helpers can be imported
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

if __name__ == "__main__":
    logging.basicConfig(filename='test_lamah.log', filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from aqua_fetch import LamaHCE, LamaHIce
from aqua_fetch.rr import _lamah
from aqua_fetch.rr._lamah import _ymd_index
from aqua_fetch.rr.utils import _make_boundary_2d
from aqua_fetch._backend import fiona, plt, netCDF4, xarray as xr

from utils import (test_dataset, test_stations, test_coords, test_boundary,
                   test_plot_catchment, test_q_mm)

# Root under which the (already downloaded) data lives. Each class appends its
# own sub-directory, so e.g. LamaH-CE daily is expected at
# ``<GSCAD_PATH>/LamaHCE_daily/LamaHCE/...``. Replace with the path on your
# machine; on another host the data is at
# ``/mnt/datawaha/hyex/atr/gscad_database/raw``.
GSCAD_PATH = '/mnt/storage1/atr/data/gscad_database/raw'

VERBOSITY = 4

# the three basin delineations, shared by both datasets and both timesteps
DATA_TYPES = ['total_upstrm', 'intermediate_all', 'intermediate_lowimp']


# ---------------------------------------------------------------------------
# LamaH-CE (Central Europe, mainly Austria)
# ---------------------------------------------------------------------------
# number of stations / static attributes per data_type (same for both timesteps)
LAMAHCE_NUM_STATIONS = [859, 859, 454]
# intermediate_all/lowimp carry one extra column: `area_km2` is the area
# upstream of the gauge and `area_km2_intermediate` the incremental catchment
LAMAHCE_NUM_STATIC = [84, 86, 86]

LAMAHCE_PATHS = {'D': os.path.join(GSCAD_PATH, 'LamaHCE_daily'),
                 'H': os.path.join(GSCAD_PATH, 'LamaHCE_hourly')}

# the LamaH-CE raw folders, used to read the source csvs independently of
# aqua_fetch in the fidelity checks below
RAW_ROOT = {ts: os.path.join(p, 'LamaHCE') for ts, p in LAMAHCE_PATHS.items()}
BASIN_DIR = {'total_upstrm': 'A_basins_total_upstrm',
             'intermediate_all': 'B_basins_intermediate_all',
             'intermediate_lowimp': 'C_basins_intermediate_lowimp'}
TS_DIR = {'D': 'daily', 'H': 'hourly'}

# instantiating the class is cheap (no data is read), but caching keeps the
# individual checks below independent of each other without paying for it twice
_CACHE = {}


def lamahce(timestep='D', data_type='total_upstrm', **kwargs):
    key = (timestep, data_type, tuple(sorted(kwargs.items())))
    if key not in _CACHE:
        _CACHE[key] = LamaHCE(path=LAMAHCE_PATHS[timestep], timestep=timestep,
                              data_type=data_type, verbosity=0, **kwargs)
    return _CACHE[key]


def raw_met_fpath(timestep, data_type, station):
    return os.path.join(RAW_ROOT[timestep], BASIN_DIR[data_type],
                        '2_timeseries', TS_DIR[timestep], f'ID_{station}.csv')


def raw_q_fpath(timestep, station):
    return os.path.join(RAW_ROOT[timestep], 'D_gauges', '2_timeseries',
                        TS_DIR[timestep], f'ID_{station}.csv')


def raw_index(df, timestep):
    """timestamps of a raw LamaH-CE csv, built with pandas' PeriodIndex"""
    if timestep == 'H':
        periods = pd.PeriodIndex.from_fields(
            year=df['YYYY'], month=df['MM'], day=df['DD'],
            hour=df['hh'], minute=df['mm'], freq='h')
    else:
        periods = pd.PeriodIndex.from_fields(
            year=df['YYYY'], month=df['MM'], day=df['DD'], freq='D')
    return periods.to_timestamp()


# ---------------------------------------------------------------------------
# a miniature but structurally faithful LamaH-CE product, used by the checks
# that must not touch the 84 GB real dataset (netCDF conversion, disk cleanup,
# missing/duplicate gauges)
# ---------------------------------------------------------------------------
SYNTH_MET_COLS = ['2m_temp_max', '2m_temp_mean', '2m_temp_min', '2m_dp_temp_max',
                  '2m_dp_temp_mean', '2m_dp_temp_min', '10m_wind_u', '10m_wind_v',
                  'fcst_alb', 'lai_high_veg', 'lai_low_veg', 'swe',
                  'surf_net_solar_rad_max', 'surf_net_solar_rad_mean',
                  'surf_net_therm_rad_max', 'surf_net_therm_rad_mean',
                  'surf_press', 'total_et', 'prec', 'volsw_123', 'volsw_4']


def make_synthetic_lamahce(root, stations=('1', '2', '3'), n_steps=40,
                           without_q=(), duplicate_coords=False, q_gaps=()):
    """
    Writes a daily LamaH-CE ``total_upstrm`` product with the real column names
    and separators under ``root`` (which must already end in ``LamaHCE``).
    """
    met_dir = os.path.join(root, 'A_basins_total_upstrm', '2_timeseries', 'daily')
    q_dir = os.path.join(root, 'D_gauges', '2_timeseries', 'daily')
    cat_dir = os.path.join(root, 'A_basins_total_upstrm', '1_attributes')
    gauge_dir = os.path.join(root, 'D_gauges', '1_attributes')
    for directory in (met_dir, q_dir, cat_dir, gauge_dir):
        os.makedirs(directory, exist_ok=True)

    dates = pd.date_range('1981-01-01', periods=n_steps, freq='D')
    rng = np.random.default_rng(0)

    for stn in stations:
        met = pd.DataFrame({'YYYY': dates.year, 'MM': dates.month,
                            'DD': dates.day, 'DOY': dates.dayofyear})
        for j, col in enumerate(SYNTH_MET_COLS):
            met[col] = np.round(rng.normal(10 * (j + 1), 1, n_steps), 3)
        met['surf_press'] = np.round(rng.uniform(90000, 101000, n_steps), 0)
        met.to_csv(os.path.join(met_dir, f'ID_{stn}.csv'), sep=';', index=False)

        if stn in without_q:
            continue
        q = pd.DataFrame({'YYYY': dates.year, 'MM': dates.month, 'DD': dates.day,
                          'qobs': np.round(rng.uniform(0.1, 5, n_steps), 3),
                          'ckhs': 1, 'qceq': 0, 'qcol': 0.0})
        q.loc[2:4, 'qobs'] = -999.0        # the documented no-data marker
        if stn in q_gaps:
            # rows simply absent -> not a contiguous regular series any more
            q = q.drop(index=[10, 11, 12]).reset_index(drop=True)
        q.to_csv(os.path.join(q_dir, f'ID_{stn}.csv'), sep=';', index=False)

    ids = list(stations)
    pd.DataFrame({'ID': ids,
                  'area_calc': [10.0 * (i + 1) for i in range(len(ids))],
                  'elev_mean': [100 * (i + 1) for i in range(len(ids))],
                  'slope_mean': [50 * (i + 1) for i in range(len(ids))],
                  }).to_csv(os.path.join(cat_dir, 'Catchment_attributes.csv'),
                            sep=';', index=False)

    lons = [4300000 + 1000 * i for i in range(len(ids))]
    lats = [2700000 + 1000 * i for i in range(len(ids))]
    if duplicate_coords:
        lons[1], lats[1] = lons[0], lats[0]
    pd.DataFrame({'ID': ids, 'name': [f'gauge{i}' for i in range(len(ids))],
                  'river': 'river', 'lon': lons, 'lat': lats,
                  'elev': [200 * (i + 1) for i in range(len(ids))],
                  }).to_csv(os.path.join(gauge_dir, 'Gauge_attributes.csv'),
                            sep=';', index=False)
    return root


class synthetic_lamahce:
    """context manager yielding a temp dir holding the miniature product"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        self.tmpdir = tempfile.mkdtemp(prefix='lamahce_synth_')
        make_synthetic_lamahce(os.path.join(self.tmpdir, 'LamaHCE'), **self.kwargs)
        # the data is on disk, so nothing may be downloaded or extracted
        self._orig = (_lamah.download, _lamah.unzip)

        def boom(*args, **kwargs):
            raise AssertionError('download/unzip was called for a local fixture')

        _lamah.download, _lamah.unzip = boom, boom
        return self.tmpdir

    def __exit__(self, *exc):
        _lamah.download, _lamah.unzip = self._orig
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        return False


def test_lamahce_daily():
    """LamaH-CE at daily timestep, all three basin delineations."""
    for idx, data_type in enumerate(DATA_TYPES):
        logger.info(f"testing LamaHCE {data_type} at daily timestep")

        dataset = LamaHCE(path=LAMAHCE_PATHS['D'],
                          timestep='D', data_type=data_type, verbosity=VERBOSITY)

        test_dataset(dataset,
                     num_stations=LAMAHCE_NUM_STATIONS[idx],
                     dyn_data_len=14244,
                     num_static_attrs=LAMAHCE_NUM_STATIC[idx],
                     num_dyn_attrs=22,
                     yearly_steps=366)
    return


def test_lamahce_hourly():
    """LamaH-CE at hourly timestep, all three basin delineations."""
    for idx, data_type in enumerate(DATA_TYPES):
        logger.info(f"testing LamaHCE {data_type} at hourly timestep")

        dataset = LamaHCE(path=LAMAHCE_PATHS['H'],
                          timestep='H', data_type=data_type, verbosity=VERBOSITY)

        test_dataset(dataset,
                     num_stations=LAMAHCE_NUM_STATIONS[idx],
                     dyn_data_len=341856,
                     num_static_attrs=LAMAHCE_NUM_STATIC[idx],
                     num_dyn_attrs=16,
                     yearly_steps=8761,
                     # one hourly station is ~35 MB of csv, so a 10% sample
                     # (85 stations, fetched twice) reads ~6 GB without testing
                     # anything that ~20 stations do not already test
                     dyn_fraction=0.023)
    return


def test_lamahce_methods():
    """
    The shared per-method helpers (``stations``, ``stn_coords``, ``get_boundary``,
    ``plot_catchment``, ``q_mm``) run over all six LamaH-CE configurations.

    These used to live in the cross-dataset matrix of ``test_methods.py``; they
    are here so that everything LamaH-CE is in one file. ``test_latlong_ranges``
    is on because the boundaries are reprojected to WGS84.
    """
    logger.info("testing LamaHCE per-method helpers")

    for timestep in ('D', 'H'):
        for idx, data_type in enumerate(DATA_TYPES):
            ds = lamahce(timestep, data_type)

            test_stations(ds, LAMAHCE_NUM_STATIONS[idx])
            test_coords(ds)

            if fiona is not None:
                test_boundary(ds, test_latlong_ranges=True)
                if plt is not None:
                    test_plot_catchment(ds)

            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                test_q_mm(ds)
    return


# ---------------------------------------------------------------------------
# LamaH-CE : fidelity of the values, units and timestamps
# ---------------------------------------------------------------------------

def test_lamahce_index_construction():
    """
    the fast numpy timestamp construction must agree with pandas' PeriodIndex
    bit for bit, at both timesteps
    """
    logger.info("testing LamaHCE index construction")

    for timestep in ('D', 'H'):
        fpath = raw_met_fpath(timestep, 'total_upstrm', '1')
        cols = ['YYYY', 'MM', 'DD'] + (['hh', 'mm'] if timestep == 'H' else [])
        raw = pd.read_csv(fpath, sep=';', usecols=cols)

        if timestep == 'H':
            fast = _ymd_index(raw['YYYY'], raw['MM'], raw['DD'], raw['hh'], raw['mm'])
        else:
            fast = _ymd_index(raw['YYYY'], raw['MM'], raw['DD'])

        reference = raw_index(raw, timestep)
        assert fast.equals(reference), f"{timestep}: index differs from PeriodIndex"
        assert np.array_equal(fast.values, reference.values), timestep
    return


def test_lamahce_temporal_extent_is_derived():
    """
    ``start``/``end`` must come from the files, not from a hardcoded literal
    """
    logger.info("testing LamaHCE temporal extent")

    for timestep, expected_len in (('D', 14244), ('H', 341856)):
        for data_type in DATA_TYPES:
            ds = lamahce(timestep, data_type)
            raw = pd.read_csv(raw_met_fpath(timestep, data_type, ds.stations()[0]),
                              sep=';',
                              usecols=['YYYY', 'MM', 'DD'] + (['hh', 'mm'] if timestep == 'H' else []))
            index = raw_index(raw, timestep)

            assert ds.start == index[0], f"{timestep} {data_type}: {ds.start} != {index[0]}"
            assert ds.end == index[-1], f"{timestep} {data_type}: {ds.end} != {index[-1]}"
            assert len(index) == expected_len, len(index)
    return


def test_lamahce_dynamic_matches_raw_files():
    """
    every value returned by ``fetch`` must be the value in the source csv, at
    the timestamp of the source csv. The only permitted change is the documented
    hPa conversion of the air pressure and the -999 -> NaN substitution.
    """
    logger.info("testing LamaHCE dynamic data against the raw csv files")

    for timestep in ('D', 'H'):
        for data_type in DATA_TYPES:
            _check_dynamic_against_source(timestep, data_type)
    return


def _check_dynamic_against_source(timestep, data_type):
    ds = lamahce(timestep, data_type)

    # '124' and '633' carry the -999 no-data marker, '1' does not. The two
    # intermediate products are checked on a single station: they differ only
    # in which basin the forcings were averaged over, not in how the files
    # are read, and one hourly station is ~35 MB of csv.
    available = set(ds.stations())
    stations = [s for s in ('1', '124', '633') if s in available]
    if data_type != 'total_upstrm':
        # intermediate_lowimp holds none of the three, fall back to any station
        stations = stations[:1] or sorted(available, key=int)[:1]
    assert stations, data_type

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _, dyn = ds.fetch(stations, as_dataframe=True)

    rename = ds.dyn_map[timestep]

    for stn in stations:
        got = dyn[stn]
        met = pd.read_csv(raw_met_fpath(timestep, data_type, stn), sep=';')
        met.index = raw_index(met, timestep)
        q = pd.read_csv(raw_q_fpath(timestep, stn), sep=';')
        q.index = raw_index(q, timestep)

        where = f"{timestep} {data_type} {stn}"
        assert got.index.equals(met.index), where

        for src, dest in rename.items():
            if src not in met.columns:
                continue
            expected = met[src].astype('float32')
            if dest == 'airpres_hpa':
                # Pa -> hPa is the one documented unit conversion
                expected = expected * 0.01
            assert np.array_equal(got[dest].to_numpy(), expected.to_numpy(),
                                  equal_nan=True), f"{where} {dest}"

        # columns that are passed through unrenamed
        for col in ['fcst_alb', 'lai_high_veg', 'lai_low_veg', 'total_et',
                    'volsw_123', 'volsw_4']:
            assert np.array_equal(got[col].to_numpy(),
                                  met[col].astype('float32').to_numpy(),
                                  equal_nan=True), f"{where} {col}"

        # streamflow: identical where the source has a value, NaN exactly
        # where the source carries the -999 no-data marker
        fetched_q = got['q_cms_obs'].reindex(q.index)
        nodata = q['qobs'] == -999
        assert fetched_q[nodata].isna().all(), f"{where}: -999 kept"
        assert np.array_equal(fetched_q[~nodata].to_numpy(),
                              q.loc[~nodata, 'qobs'].astype('float32').to_numpy()), \
            f"{where}: streamflow altered"
        # outside the runoff record the streamflow must be missing
        outside = got.index.difference(q.index)
        assert got.loc[outside, 'q_cms_obs'].isna().all(), where
    return


def test_lamahce_q_nodata_and_raw_accessor():
    """
    -999 must never reach the user as a discharge, and the raw accessor must
    still expose the source rows unchanged, flags included
    """
    logger.info("testing LamaHCE -999 handling")

    for timestep in ('D', 'H'):
        ds = lamahce(timestep)

        raw = ds.fetch_stn_q_raw('124')
        source = pd.read_csv(raw_q_fpath(timestep, '124'), sep=';')
        assert raw.columns.to_list() == ['qobs', 'ckhs', 'qceq', 'qcol'], raw.columns
        assert len(raw) == len(source)
        for col in raw.columns:
            assert np.array_equal(raw[col].to_numpy(), source[col].to_numpy(),
                                  equal_nan=True), col
        n_nodata = int((source['qobs'] == -999).sum())
        assert n_nodata > 0, "station 124 is expected to carry -999 values"
        assert int((raw['qobs'] == -999).sum()) == n_nodata

        # the analysis ready view replaces them and says so
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _, dyn = ds.fetch('124', dynamic_features='q_cms_obs', as_dataframe=True)
        messages = [str(w.message) for w in caught]
        assert any('-999' in m for m in messages), messages

        q = dyn['124']['q_cms_obs']
        assert (q == -999).sum() == 0
        assert q.min() >= 0, q.min()
        stamps = raw.index[source['qobs'].to_numpy() == -999]
        assert q.loc[stamps].isna().all()
    return


def test_lamahce_feature_subset_is_consistent():
    """
    fetching a subset of the dynamic features must return exactly the same
    numbers - and the same index - as fetching everything and slicing
    """
    logger.info("testing LamaHCE feature subsetting")

    for timestep in ('D', 'H'):
        ds = lamahce(timestep)
        stations = ['1', '124']

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, full = ds.fetch(stations, as_dataframe=True)

            for subset in (['pcp_mm', 'airtemp_C_mean'] if timestep == 'H'
                           else ['pcp_mm', 'airtemp_C_mean', 'airpres_hpa'],
                           ['q_cms_obs'],
                           ['q_cms_obs', 'swe_mm']):
                _, part = ds.fetch(stations, dynamic_features=subset,
                                   as_dataframe=True)
                for stn in stations:
                    assert part[stn].columns.to_list() == subset, part[stn].columns
                    assert part[stn].index.equals(full[stn].index), \
                        f"{timestep} {stn} {subset}: index differs"
                    assert np.array_equal(part[stn].to_numpy(),
                                          full[stn][subset].to_numpy(),
                                          equal_nan=True), f"{timestep} {stn} {subset}"
    return


def test_lamahce_parallel_equals_serial():
    """the process pool must not change what is read"""
    logger.info("testing LamaHCE serial vs parallel reads")

    stations = ['1', '124', '633', '826']

    for timestep in ('D', 'H'):
        serial = lamahce(timestep, processes=1)
        parallel = lamahce(timestep, processes=4)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, one = serial.fetch(stations, as_dataframe=True)
            _, many = parallel.fetch(stations, as_dataframe=True)

        for stn in stations:
            assert one[stn].index.equals(many[stn].index), stn
            assert one[stn].columns.equals(many[stn].columns), stn
            assert np.array_equal(one[stn].to_numpy(), many[stn].to_numpy(),
                                  equal_nan=True), stn
    return


def test_lamahce_q_mm():
    """q_mm must be q_cms converted with the catchment area of the dataset"""
    logger.info("testing LamaHCE q_mm")

    for timestep, seconds in (('D', 86400), ('H', 3600)):
        ds = lamahce(timestep)
        stations = ['1', '124']

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            q_mm = ds.q_mm(stations)
            _, dyn = ds.fetch(stations, dynamic_features='q_cms_obs',
                              as_dataframe=True)

        assert len(q_mm) == len(dyn[stations[0]]), (len(q_mm), len(dyn[stations[0]]))
        for stn in stations:
            expected = (dyn[stn]['q_cms_obs'] / (ds.area(stn).iloc[0] * 1e6)) * seconds * 1e3
            assert np.allclose(q_mm[stn].to_numpy(), expected.to_numpy(),
                               equal_nan=True, rtol=1e-6), stn
    return


def test_lamahce_coords_and_boundary():
    """
    gauge coordinates and catchment boundaries must both be in WGS84 and must
    agree with each other
    """
    logger.info("testing LamaHCE coordinates and boundaries")

    ds = lamahce('D')
    coords = ds.stn_coords()

    assert coords['lat'].between(46, 51).all(), coords['lat'].describe()
    assert coords['long'].between(7.5, 19).all(), coords['long'].describe()

    if fiona is None:
        return

    for stn in ds.stations()[:20]:
        rings = _make_boundary_2d(ds.get_boundary(stn))
        lons = np.concatenate([r[:, 0] for r in rings])
        lats = np.concatenate([r[:, 1] for r in rings])

        assert ((lons >= -180) & (lons <= 180)).all(), stn
        assert ((lats >= -90) & (lats <= 90)).all(), stn

        # the gauge is the outlet of its catchment, so it must lie within the
        # bounding box of the catchment polygon
        lat, lon = coords.loc[stn, 'lat'], coords.loc[stn, 'long']
        assert lons.min() - 0.01 <= lon <= lons.max() + 0.01, stn
        assert lats.min() - 0.01 <= lat <= lats.max() + 0.01, stn
    return


def test_lamahce_static_features():
    """static features are read from the two attribute files without change"""
    logger.info("testing LamaHCE static features")

    for idx, data_type in enumerate(DATA_TYPES):
        ds = lamahce('D', data_type)

        # the default of ``static_features`` must be usable
        one = ds.stations()[0]
        assert ds.fetch_static_features(one).shape == (1, LAMAHCE_NUM_STATIC[idx])

        static = ds.fetch_static_features('all', 'all')
        assert static.shape == (LAMAHCE_NUM_STATIONS[idx], LAMAHCE_NUM_STATIC[idx])
        assert not static.columns.duplicated().any()

        catch = pd.read_csv(
            os.path.join(RAW_ROOT['D'], BASIN_DIR[data_type], '1_attributes',
                         'Catchment_attributes.csv'), sep=';', index_col='ID')
        gauge = pd.read_csv(
            os.path.join(RAW_ROOT['D'], 'D_gauges', '1_attributes',
                         'Gauge_attributes.csv'), sep=';', index_col='ID')
        catch.index = catch.index.astype(str)
        gauge.index = gauge.index.astype(str)

        stn = ds.stations()[0]
        if data_type == 'total_upstrm':
            assert static.loc[stn, 'area_km2'] == catch.loc[stn, 'area_calc']
        else:
            # `area_calc` of an intermediate delineation is the incremental
            # catchment, so it is published under its own name while `area_km2`
            # keeps its cross-dataset meaning of "upstream of the gauge"
            assert static.loc[stn, 'area_km2_intermediate'] == catch.loc[stn, 'area_calc']
            total = pd.read_csv(
                os.path.join(RAW_ROOT['D'], 'A_basins_total_upstrm',
                             '1_attributes', 'Catchment_attributes.csv'),
                sep=';', index_col='ID')
            total.index = total.index.astype(str)
            assert static.loc[stn, 'area_km2'] == total.loc[stn, 'area_calc']
        assert static.loc[stn, 'slope_mkm-1'] == catch.loc[stn, 'slope_mean']
        # the coordinates are kept in the source (EPSG:3035) system in the
        # static table; only stn_coords() projects them
        assert static.loc[stn, 'lat'] == gauge.loc[stn, 'lat']
        assert static.loc[stn, 'long'] == gauge.loc[stn, 'lon']
        assert static.loc[stn, 'elev'] == gauge.loc[stn, 'elev']
    return


def test_lamahce_returns_copies():
    """cached internal state must not be reachable (and mutable) by callers"""
    logger.info("testing LamaHCE cache isolation")

    ds = lamahce('D')

    stations = ds.stations()
    stations.append('not-a-station')
    assert 'not-a-station' not in ds.stations()

    dyn = ds.dynamic_features
    dyn.clear()
    assert len(ds.dynamic_features) == 22

    static_features = ds.static_features
    static_features.clear()
    assert len(ds.static_features) == 84

    table = ds.static_data()
    table.iloc[0, 0] = -12345.0
    assert ds.static_data().iloc[0, 0] != -12345.0
    return


def test_lamahce_no_redownload():
    """
    instantiating the class when the data is on disk must neither download nor
    extract anything. The check does not use wall-clock time: ``download`` and
    ``unzip`` are replaced by functions that fail if they are ever called.
    """
    logger.info("testing that LamaHCE does not re-download")

    def boom(*args, **kwargs):
        raise AssertionError("download/extraction was triggered although the "
                             "data is already on disk")

    orig_download, orig_unzip = _lamah.download, _lamah.unzip
    _lamah.download, _lamah.unzip = boom, boom
    try:
        for timestep in ('D', 'H'):
            for data_type in DATA_TYPES:
                ds = LamaHCE(path=LAMAHCE_PATHS[timestep], timestep=timestep,
                             data_type=data_type, verbosity=0)
                assert len(ds.stations()) > 0
    finally:
        _lamah.download, _lamah.unzip = orig_download, orig_unzip
    return


def test_lamahce_download_and_overwrite_logic():
    """
    Offline check of the download/extract decision, driven by stubs so that no
    byte is fetched from the network:

        - the decision to skip is taken on the extracted folders, so deleting
          the archive (``remove_zip=True``) must not trigger a re-download
        - ``overwrite=True`` must delete the stale archive *and* the previously
          extracted folders before downloading again
    """
    logger.info("testing LamaHCE download/overwrite logic")

    archive = '2_LamaH-CE_daily.tar.gz'
    # what the daily archive is declared to leave behind (see dirs_to_check)
    folders = [os.path.join('A_basins_total_upstrm', '2_timeseries', 'daily'),
               os.path.join('D_gauges', '2_timeseries', 'daily')]
    calls = {'download': 0, 'unzip': 0}

    def fake_download(url, outdir, fname, **kwargs):
        calls['download'] += 1
        with open(os.path.join(outdir, fname), 'w') as fp:
            fp.write('archive')

    def fake_unzip(path, **kwargs):
        calls['unzip'] += 1
        for folder in folders:
            os.makedirs(os.path.join(path, folder), exist_ok=True)

    def make_dataset(tmpdir, **attrs):
        """a LamaHCE shell that only knows what _download_and_extract needs"""
        ds = object.__new__(LamaHCE)
        ds._path = tmpdir
        ds.name = 'LamaHCE'
        ds.data_type = 'total_upstrm'
        ds.verbosity = 0
        ds.overwrite = False
        ds.remove_zip = False
        ds.url = {archive: 'https://example.invalid/' + archive}
        for k, v in attrs.items():
            setattr(ds, k, v)
        return ds

    orig_download, orig_unzip = _lamah.download, _lamah.unzip
    _lamah.download, _lamah.unzip = fake_download, fake_unzip
    tmpdir = tempfile.mkdtemp(prefix='lamahce_dl_')
    try:
        # 1. nothing on disk -> download + extract
        make_dataset(tmpdir)._download_and_extract()
        assert calls == {'download': 1, 'unzip': 1}, calls
        assert all(os.path.isdir(os.path.join(tmpdir, f)) for f in folders)

        # 2. everything extracted -> no download, no extraction
        make_dataset(tmpdir)._download_and_extract()
        assert calls == {'download': 1, 'unzip': 1}, calls

        # 3. archive deleted but data extracted -> still no re-download
        os.remove(os.path.join(tmpdir, archive))
        make_dataset(tmpdir)._download_and_extract()
        assert calls == {'download': 1, 'unzip': 1}, calls

        # 4. remove_zip must delete the archive after extracting
        fake_download(None, tmpdir, archive)
        calls['download'] = 1
        make_dataset(tmpdir, remove_zip=True)._download_and_extract()
        assert not os.path.exists(os.path.join(tmpdir, archive))

        # 5. overwrite -> stale archive and stale folders are removed first and
        #    the data is fetched again
        fake_download(None, tmpdir, archive)
        calls.update(download=0, unzip=0)
        # the marker lives outside 2_timeseries: overwrite must wipe the whole
        # top level folder, not just the time series it checks for
        stale_dir = os.path.join(tmpdir, 'A_basins_total_upstrm', '1_attributes')
        os.makedirs(stale_dir, exist_ok=True)
        marker = os.path.join(stale_dir, 'stale.txt')
        with open(marker, 'w') as fp:
            fp.write('stale')
        make_dataset(tmpdir, overwrite=True)._download_and_extract()
        assert calls == {'download': 1, 'unzip': 1}, calls
        assert not os.path.exists(marker), "stale extracted data was not removed"
    finally:
        _lamah.download, _lamah.unzip = orig_download, orig_unzip
        shutil.rmtree(tmpdir, ignore_errors=True)
    return


def test_lamahce_hourly_archive_is_not_skipped():
    """
    The two LamaH-CE archives extract into the same top level folders and the
    default ``path`` does not depend on the timestep, so a daily-only
    installation used to satisfy the "already extracted?" test of the
    daily+hourly archive: ``LamaHCE(timestep='H')`` after ``LamaHCE()`` then
    never downloaded the hourly time series and died in ``stations()``.

    The decision is driven here by stubs, so no byte leaves the machine. Run
    against the pre-fix code this asserts False on the third case.
    """
    logger.info("testing that the LamaHCE hourly archive is not skipped")

    ARCHIVE = {'D': '2_LamaH-CE_daily.tar.gz', 'H': '1_LamaH-CE_daily_hourly.tar.gz'}
    # the daily archive fills only 'daily', the daily+hourly one fills both
    CARRIES = {'D': [os.path.join('A_basins_total_upstrm', '2_timeseries', 'daily'),
                     os.path.join('D_gauges', '2_timeseries', 'daily')]}
    CARRIES['H'] = CARRIES['D'] + [
        os.path.join('A_basins_total_upstrm', '2_timeseries', 'hourly'),
        os.path.join('D_gauges', '2_timeseries', 'hourly')]

    orig_download, orig_unzip = _lamah.download, _lamah.unzip
    tmpdir = tempfile.mkdtemp(prefix='lamahce_ts_')
    try:
        # (already installed, wanted timestep, must it download?)
        for installed, wanted, expected in ((None, 'D', True),
                                            (None, 'H', True),
                                            ('D', 'H', True),   # <-- the bug
                                            ('D', 'D', False),
                                            ('H', 'H', False),
                                            ('H', 'D', False)):
            root = os.path.join(tmpdir, f'{installed}_{wanted}', 'LamaHCE')
            os.makedirs(root)
            for folder in CARRIES.get(installed, []):
                os.makedirs(os.path.join(root, folder), exist_ok=True)

            calls = []

            def fake_download(url, outdir, fname, _w=wanted, **kwargs):
                calls.append('download')
                with open(os.path.join(outdir, fname), 'w') as fp:
                    fp.write('archive')

            def fake_unzip(path, _w=wanted, **kwargs):
                calls.append('unzip')
                for folder in CARRIES[_w]:
                    os.makedirs(os.path.join(path, folder), exist_ok=True)

            _lamah.download, _lamah.unzip = fake_download, fake_unzip

            ds = object.__new__(LamaHCE)
            ds._path = root
            ds.name = 'LamaHCE'
            ds.data_type = 'total_upstrm'
            ds.verbosity = 0
            ds.overwrite = False
            ds.remove_zip = False
            ds.url = {ARCHIVE[wanted]: 'https://example.invalid/' + ARCHIVE[wanted]}

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                ds._download_and_extract()

            assert bool(calls) is expected, (installed, wanted, calls)
            assert not [str(w.message) for w in caught], \
                (installed, wanted, [str(w.message) for w in caught])
            # whatever happened, the time series of the wanted timestep is there
            for folder in CARRIES[wanted]:
                assert os.path.isdir(os.path.join(root, folder)), (wanted, folder)
    finally:
        _lamah.download, _lamah.unzip = orig_download, orig_unzip
        shutil.rmtree(tmpdir, ignore_errors=True)
    return


def test_lamahce_q_mm_uses_total_upstream_area():
    """
    ``D_gauges`` holds the total discharge at the gauge for every ``data_type``,
    while the catchment attributes of the intermediate delineations describe
    only the incremental sub-catchment. ``q_mm`` must therefore divide by the
    total upstream area and be identical for all three delineations.

    Run against the pre-fix code, ``intermediate_all`` returns runoff heights up
    to three orders of magnitude too large and this fails.
    """
    logger.info("testing LamaHCE q_mm against the total upstream area")

    # intermediate_lowimp covers only 454 of the 859 gauges, so take the
    # stations from there; its ids are a subset of the other two delineations
    stations = lamahce('D', 'intermediate_lowimp').stations()[:3]

    reference = None
    for data_type in DATA_TYPES:
        ds = lamahce('D', data_type)
        assert set(stations).issubset(ds.stations()), (data_type, stations)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            q_mm = ds.q_mm(stations)
            total_area = ds.total_upstrm_area(stations)
            _, dyn = ds.fetch(stations, dynamic_features='q_cms_obs',
                              as_dataframe=True)

        for stn in stations:
            expected = (dyn[stn]['q_cms_obs'] / (total_area[stn] * 1e6)) * 86400 * 1e3
            assert np.allclose(q_mm[stn].to_numpy(), expected.to_numpy(),
                               equal_nan=True, rtol=1e-6), (data_type, stn)

        # area() means "upstream of the gauge" for every delineation ...
        assert np.allclose(ds.area(stations).to_numpy(), total_area.to_numpy(),
                           rtol=1e-6), data_type

        if data_type == 'total_upstrm':
            assert 'area_km2_intermediate' not in ds.static_features
            reference = q_mm
        else:
            # ... and the incremental area is still reachable, and smaller
            incremental = ds.fetch_static_features(
                stations, ['area_km2_intermediate']).iloc[:, 0]
            assert (incremental <= total_area + 1e-6).all(), data_type
            assert (incremental < total_area).any(), data_type
            # so the runoff heights must not depend on the delineation
            for stn in stations:
                assert np.allclose(q_mm[stn].to_numpy(),
                                   reference[stn].to_numpy(),
                                   equal_nan=True, rtol=1e-6), (data_type, stn)

    # and the published mean runoff of the dataset itself must be reproduced
    ds = lamahce('D')
    hydro_idx = pd.read_csv(
        os.path.join(RAW_ROOT['D'], 'D_gauges', '1_attributes',
                     'Hydro_indices_1981_2017.csv'), sep=';', index_col='ID')
    hydro_idx.index = hydro_idx.index.astype(str)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        got = ds.q_mm(['1', '7', '3']).loc['1981-01-01':'2017-12-31'].mean()
    published = hydro_idx.loc[got.index, 'q_mean']
    assert np.allclose(got.to_numpy(), published.to_numpy(), rtol=0.05), \
        (got.to_dict(), published.to_dict())
    return


def test_lamahce_missing_files_are_reported():
    """an incomplete extraction must warn instead of silently returning less"""
    logger.info("testing LamaHCE incomplete-extraction warning")

    archive = '2_LamaH-CE_daily.tar.gz'
    orig_download, orig_unzip = _lamah.download, _lamah.unzip
    tmpdir = tempfile.mkdtemp(prefix='lamahce_missing_')

    def fake_download(url, outdir, fname, **kwargs):
        with open(os.path.join(outdir, fname), 'w') as fp:
            fp.write('archive')

    def broken_unzip(path, **kwargs):
        # only one of the two expected folders shows up
        os.makedirs(os.path.join(path, 'A_basins_total_upstrm',
                                 '2_timeseries', 'daily'), exist_ok=True)

    _lamah.download, _lamah.unzip = fake_download, broken_unzip
    try:
        ds = object.__new__(LamaHCE)
        ds._path = tmpdir
        ds.name = 'LamaHCE'
        ds.data_type = 'total_upstrm'
        ds.verbosity = 0
        ds.overwrite = False
        ds.remove_zip = False
        ds.url = {archive: 'https://example.invalid/' + archive}

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            ds._download_and_extract()
        messages = [str(w.message) for w in caught]
        assert any('D_gauges' in m for m in messages), messages
    finally:
        _lamah.download, _lamah.unzip = orig_download, orig_unzip
        shutil.rmtree(tmpdir, ignore_errors=True)
    return


def test_lamahce_coords_against_pyproj():
    """
    the hand-rolled EPSG:3035 -> WGS84 inverse must agree with a full CRS
    library. pyproj is not a dependency of aqua_fetch, so the check is skipped
    when it is absent.
    """
    logger.info("testing LamaHCE coordinates against pyproj")

    try:
        from pyproj import Transformer
    except ImportError:
        logger.info("pyproj not installed - skipping")
        return

    ds = lamahce('D')
    gauge = pd.read_csv(
        os.path.join(RAW_ROOT['D'], 'D_gauges', '1_attributes',
                     'Gauge_attributes.csv'), sep=';', index_col='ID')
    gauge.index = gauge.index.astype(str)
    gauge = gauge.loc[ds.stations()]

    transformer = Transformer.from_crs('EPSG:3035', 'EPSG:4326', always_xy=True)
    lon_ref, lat_ref = transformer.transform(gauge['lon'].to_numpy(float),
                                             gauge['lat'].to_numpy(float))

    got = ds.stn_coords()
    # 1e-5 deg is ~1 m; the float32 that the base class casts the eastings to
    # is the only source of error left
    assert np.allclose(got['lat'].to_numpy(), lat_ref, atol=1e-5), \
        np.abs(got['lat'].to_numpy() - lat_ref).max()
    assert np.allclose(got['long'].to_numpy(), lon_ref, atol=1e-5), \
        np.abs(got['long'].to_numpy() - lon_ref).max()
    return


def test_lamahce_n_workers():
    """
    the pool is chosen by the size of the workload, not by the number of
    stations: a few hourly stations are worth parallelising, many daily
    single-feature ones are not
    """
    logger.info("testing LamaHCE worker-count heuristic")

    daily, hourly = lamahce('D'), lamahce('H')

    all_daily = daily._reader_spec(daily.dynamic_features)
    all_hourly = hourly._reader_spec(hourly.dynamic_features)
    q_hourly = hourly._reader_spec(['q_cms_obs'])

    # a single station is never worth a pool
    assert daily._n_workers(all_daily, 1) == 1
    assert hourly._n_workers(all_hourly, 1) == 1

    # ~1.4 MB per daily station -> serial for a handful, parallel for many
    assert daily._n_workers(all_daily, 3) == 1
    assert daily._n_workers(all_daily, 200) > 1

    # ~35 MB per hourly station -> parallel already at two stations
    assert hourly._n_workers(all_hourly, 2) > 1

    # runoff only skips the meteo file, so the same two stations stay serial
    assert q_hourly['met_usecols'] is None
    assert hourly._n_workers(q_hourly, 2) == 1

    # processes=1 must not spawn a pool at all, however large the workload.
    # Asserted on the observable behaviour rather than on the return value.
    serial = lamahce('H', processes=1)
    original = _lamah.cf.ProcessPoolExecutor

    def no_pool(*args, **kwargs):
        raise AssertionError("a process pool was started although processes=1")

    _lamah.cf.ProcessPoolExecutor = no_pool
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, dyn = serial.fetch(['1', '124', '633', '826'],
                                  dynamic_features=['pcp_mm'], as_dataframe=True)
        assert len(dyn) == 4
    finally:
        _lamah.cf.ProcessPoolExecutor = original
    return


def test_lamahce_plot_num_observations():
    """the inherited plotting method must work for LamaHCE"""
    logger.info("testing LamaHCE plot_num_observations")

    if plt is None:
        return

    ds = lamahce('D')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ax = ds.plot_num_observations(stations=ds.stations()[:5],
                                      dynamic_features=['q_cms_obs', 'pcp_mm'],
                                      show=False)
    assert isinstance(ax, plt.Axes)
    plt.close()
    return


# ---------------------------------------------------------------------------
# LamaH-CE : checks that run against a miniature synthetic product, so that
# they stay offline and take milliseconds instead of touching the 84 GB dataset
# ---------------------------------------------------------------------------

def test_lamahce_netcdf_roundtrip():
    """
    ``to_netcdf=True`` must write one .nc per dynamic feature and reading them
    back must return exactly the values that the csv path returns - the
    conversion may not alter the data or its units.
    """
    logger.info("testing LamaHCE netcdf conversion")

    if netCDF4 is None or xr is None:
        return

    with synthetic_lamahce() as tmpdir:
        ds = LamaHCE(path=tmpdir, timestep='D', to_netcdf=True, verbosity=0)

        ncdir = os.path.join(ds.path, 'total_upstrm_D')
        assert sorted(os.listdir(ncdir)) == sorted(ds.dynamic_fnames)
        assert ds.all_ncs_exist

        stations = sorted(ds.stations())
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # goes through _make_ds_from_ncs because all .nc files now exist
            _, from_nc = ds.fetch(stations, as_dataframe=True)
            # and the csv reader, for comparison
            from_csv = ds._read_dynamic(stations, ds.dynamic_features)

        for stn in stations:
            got = from_nc[stn][ds.dynamic_features]
            ref = from_csv[stn][ds.dynamic_features]
            assert got.index.equals(ref.index), stn
            assert np.array_equal(got.to_numpy(), ref.to_numpy(),
                                  equal_nan=True), f"{stn}: netcdf != csv"

        # a date-restricted read through the .nc path must slice, not shift
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, window = ds.fetch(stations[:1], st='19810105', en='19810110',
                                 as_dataframe=True)
        idx = window[stations[0]].index
        assert idx[0] == pd.Timestamp('1981-01-05') and idx[-1] == pd.Timestamp('1981-01-10')

        # a second instantiation must not rebuild the files
        mtimes = {f: os.path.getmtime(os.path.join(ncdir, f))
                  for f in os.listdir(ncdir)}
        LamaHCE(path=tmpdir, timestep='D', to_netcdf=True, verbosity=0)
        assert mtimes == {f: os.path.getmtime(os.path.join(ncdir, f))
                          for f in os.listdir(ncdir)}, "netcdf files were rewritten"
    return


def test_lamahce_duplicate_gauges_warn():
    """two gauges at the same location are reported but kept"""
    logger.info("testing LamaHCE duplicate-gauge warning")

    with synthetic_lamahce(duplicate_coords=True) as tmpdir:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            ds = LamaHCE(path=tmpdir, timestep='D', to_netcdf=False, verbosity=0)
        messages = [str(w.message) for w in caught]
        assert any('may therefore be duplicates' in m for m in messages), messages
        # warned, not excluded
        assert len(ds.stations()) == 3, ds.stations()

    # and the real product must not trigger it
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        LamaHCE(path=LAMAHCE_PATHS['D'], timestep='D', verbosity=0).static_data()
    assert not [w for w in caught if 'duplicates' in str(w.message)]
    return


def test_lamahce_missing_runoff_file():
    """a gauge without a runoff file yields NaN streamflow and a warning"""
    logger.info("testing LamaHCE missing runoff file")

    with synthetic_lamahce(without_q=('3',)) as tmpdir:
        ds = LamaHCE(path=tmpdir, timestep='D', to_netcdf=False, verbosity=0)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _, dyn = ds.fetch(['1', '3'], as_dataframe=True)

        assert any('no runoff file' in str(w.message) for w in caught), \
            [str(w.message) for w in caught]
        assert dyn['3']['q_cms_obs'].isna().all()
        # the station with a runoff file is unaffected
        assert dyn['1']['q_cms_obs'].notna().any()
        assert len(dyn['3']) == len(dyn['1'])
    return


def test_lamahce_runoff_with_gaps():
    """
    A runoff file whose rows are not a contiguous series must still be stamped
    from the dates written in each row: the missing steps become NaN and every
    remaining value keeps its own date. Rebuilding the index from the first and
    last row would shift the tail of such a file by three days.
    """
    logger.info("testing LamaHCE runoff file with missing rows")

    with synthetic_lamahce(q_gaps=('2',)) as tmpdir:
        ds = LamaHCE(path=tmpdir, timestep='D', to_netcdf=False, verbosity=0)

        source = pd.read_csv(ds.q_fname('2'), sep=';')
        stamps = pd.to_datetime(source[['YYYY', 'MM', 'DD']].rename(
            columns={'YYYY': 'year', 'MM': 'month', 'DD': 'day'}))
        assert len(stamps) < len(pd.date_range(stamps.iloc[0], stamps.iloc[-1],
                                               freq='D')), "fixture has no gap"

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, dyn = ds.fetch(['2'], as_dataframe=True)
        got = dyn['2']['q_cms_obs']

        missing = got.index.difference(stamps)
        assert len(missing) == 3, missing
        assert got.loc[missing].isna().all()

        present = source['qobs'] != -999
        assert np.array_equal(got.loc[stamps[present]].to_numpy(),
                              source.loc[present, 'qobs'].astype('float32').to_numpy())
    return


def test_lamahce_free_disk_space():
    """archives are removable once their content is extracted; data is kept"""
    logger.info("testing LamaHCE free_disk_space")

    with synthetic_lamahce() as tmpdir:
        ds = LamaHCE(path=tmpdir, timestep='D', to_netcdf=False, verbosity=0)

        archive = os.path.join(ds.path, '2_LamaH-CE_daily.tar.gz')
        with open(archive, 'wb') as fp:
            fp.write(b'x' * 2048)

        assert ds._archive_files() == [archive], ds._archive_files()

        report = ds.free_disk_space(dry_run=True)
        assert report == {archive: 2048}, report
        assert os.path.exists(archive), "dry_run must not delete anything"

        report = ds.free_disk_space(dry_run=False)
        assert report == {archive: 2048}, report
        assert not os.path.exists(archive)
        # the extracted data must survive, i.e. no re-download on next init
        assert os.path.isdir(os.path.join(ds.path, 'D_gauges'))
        assert len(LamaHCE(path=tmpdir, timestep='D', verbosity=0).stations()) == 3
    return


# ---------------------------------------------------------------------------
# LamaH-Ice (Iceland)
# ---------------------------------------------------------------------------
# number of stations is the same for both timesteps; the static-attribute count
# differs because the daily ``total_upstrm`` product ships extra water-balance
# attributes that the hourly / intermediate products do not.
LAMAHICE_NUM_STATIONS = [111, 107, 86]
LAMAHICE_DAILY_NUM_STATIC = [154, 115, 115]
LAMAHICE_HOURLY_NUM_STATIC = [138, 115, 115]


def test_lamahice_hourly():
    """LamaH-Ice at hourly timestep, all three basin delineations."""
    for idx, data_type in enumerate(DATA_TYPES):
        logger.info(f"testing LamaHIce {data_type} at hourly timestep")

        dataset = LamaHIce(path=os.path.join(GSCAD_PATH, 'LamaHIce_hourly'),
                           timestep='H', data_type=data_type, verbosity=VERBOSITY)

        test_dataset(dataset,
                     num_stations=LAMAHICE_NUM_STATIONS[idx],
                     dyn_data_len=412848,
                     num_static_attrs=LAMAHICE_HOURLY_NUM_STATIC[idx],
                     num_dyn_attrs=28,
                     yearly_steps=8761,
                     test_latlong_ranges=False)
    return


def test_lamahice_daily():
    """LamaH-Ice at daily timestep, all three basin delineations."""
    for idx, data_type in enumerate(DATA_TYPES):
        logger.info(f"testing LamaHIce {data_type} at daily timestep")

        dataset = LamaHIce(path=os.path.join(GSCAD_PATH, 'LamaHIce_daily'),
                           timestep='D', data_type=data_type, verbosity=VERBOSITY)

        test_dataset(dataset,
                     num_stations=LAMAHICE_NUM_STATIONS[idx],
                     dyn_data_len=26298,
                     num_static_attrs=LAMAHICE_DAILY_NUM_STATIC[idx],
                     num_dyn_attrs=36,
                     yearly_steps=366,
                     test_latlong_ranges=False)
    return


if __name__ == "__main__":
    # offline, synthetic-fixture checks first: they take milliseconds and a
    # failure here means the rest is not worth running
    test_lamahce_download_and_overwrite_logic()
    test_lamahce_hourly_archive_is_not_skipped()
    test_lamahce_missing_files_are_reported()
    test_lamahce_netcdf_roundtrip()
    test_lamahce_duplicate_gauges_warn()
    test_lamahce_missing_runoff_file()
    test_lamahce_runoff_with_gaps()
    test_lamahce_free_disk_space()

    # cheap checks against the real data
    test_lamahce_index_construction()
    test_lamahce_no_redownload()
    test_lamahce_returns_copies()
    test_lamahce_n_workers()
    test_lamahce_temporal_extent_is_derived()
    test_lamahce_static_features()
    test_lamahce_coords_and_boundary()
    test_lamahce_coords_against_pyproj()
    test_lamahce_plot_num_observations()
    test_lamahce_q_nodata_and_raw_accessor()
    test_lamahce_dynamic_matches_raw_files()
    test_lamahce_feature_subset_is_consistent()
    test_lamahce_parallel_equals_serial()
    test_lamahce_q_mm()
    test_lamahce_q_mm_uses_total_upstream_area()
    test_lamahce_methods()

    test_lamahce_daily()
    test_lamahce_hourly()
    test_lamahice_hourly()
    test_lamahice_daily()

    print("*** All LamaH tests passed ***")
