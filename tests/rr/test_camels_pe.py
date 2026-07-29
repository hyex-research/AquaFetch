"""
Tests for the CAMELS_PE (Peru, 136 catchments) dataset.

The tests verify that the class

    * exposes the correct number of stations (136), dynamic (9, i.e. the observed
      variables only, excluding the model-simulated streamflow) and static (78)
      features,
    * fetches the hydro-meteorological time series and static attributes
      **without changing the values or their units** (compared against a fresh,
      independent read of the raw csv files),
    * does not re-download or re-extract when the data is already on disk,
    * reports a temporal extent that matches the actual data (not a drifting
      hardcoded literal),
    * handles coordinates / boundaries (WGS84) and q_mm correctly.

The tests are efficient: with the consolidated ``camels_pe_D.nc`` cache built
once during the first initialisation, the whole file runs in well under a
minute.
"""

import os
import site
import shutil
import random
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
    logging.basicConfig(filename='test_camels_pe.log', filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

import aqua_fetch
from aqua_fetch import CAMELS_PE
from aqua_fetch.rr import _camels
from aqua_fetch._backend import xarray as xr

from utils import test_dataset

# Path under which the (already downloaded) data lives. The class appends
# ``CAMELS_PE`` to it, so the extracted data is expected at
# ``<CAMELS_PE_PATH>/CAMELS_PE/CAMELS-PE_v1.0.1/CAMELS-PE``. Replace with the
# location on your machine (or leave as ``None`` to download to the default
# aqua_fetch data directory).
CAMELS_PE_PATH = '/home/atr/AquaFetch/data_staging/test_data'

NUM_STATIONS = 136
NUM_STATIC = 78
NUM_DYNAMIC = 9   # the model-simulated streamflow (flow_sim) is excluded
DYN_LEN = 16436  # daily steps 1981-01-01 .. 2025-12-31

# the exact, order-preserving set of standardised dynamic feature names.
# NOTE: raw ``flow_sim`` (simulated streamflow) is deliberately NOT presented.
EXPECTED_DYN_FEATURES = [
    'pcp_mm', 'prec_var', 'q_mm_obs', 'pet_mm',
    'airtemp_C_min', 'airtemp_C_mean', 'airtemp_C_max', 'srad', 'vp_hpa',
]

# raw time-series column -> standardised dynamic feature name (``flow_sim`` is
# intentionally absent: simulated data is not presented)
RAW_TO_STD = {
    'prec': 'pcp_mm', 'prec_var': 'prec_var', 'flow_obs': 'q_mm_obs',
    'pet': 'pet_mm', 'tmin': 'airtemp_C_min',
    'tmean': 'airtemp_C_mean', 'tmax': 'airtemp_C_max', 'srad': 'srad',
    'vprp': 'vp_hpa',
}

dataset = CAMELS_PE(path=CAMELS_PE_PATH, verbosity=0)


# ---------------------------------------------------------------------------
# CAMELS_PE specific tests
# ---------------------------------------------------------------------------

def _raw_ts(dataset, station) -> pd.DataFrame:
    """independent read of one raw by_catchment time-series csv"""
    fpath = os.path.join(dataset.ts_dir, "by_catchment", f"{station}.csv")
    return pd.read_csv(fpath, index_col='date', parse_dates=True)


def test_registration():
    """the class is importable from the package and registered."""
    logger.info("test_registration")
    assert 'CAMELS_PE' in aqua_fetch.ALL_DATASETS
    # can be built through the RainfallRunoff factory too
    from aqua_fetch import RainfallRunoff
    rr = RainfallRunoff('CAMELS_PE', path=CAMELS_PE_PATH, verbosity=0)
    assert len(rr.stations()) == NUM_STATIONS
    return


def test_feature_names():
    """dynamic feature names are exactly the 9 expected ones and the
    unit-ambiguous ``srad``/``prec_var`` are kept with their raw names."""
    logger.info("test_feature_names")
    assert dataset.dynamic_features == EXPECTED_DYN_FEATURES, dataset.dynamic_features
    # srad is MJ m-2 day-1 (NOT W m-2) and prec_var is mm2 day-2: both must be
    # kept raw because no canonical name carries those units
    assert 'srad' in dataset.dynamic_features
    assert 'prec_var' in dataset.dynamic_features
    # canonical names that DO match units must be present
    for f in ('pcp_mm', 'q_mm_obs', 'pet_mm', 'vp_hpa',
              'airtemp_C_min', 'airtemp_C_mean', 'airtemp_C_max'):
        assert f in dataset.dynamic_features
    # the model-simulated streamflow must NOT be presented (observational-only)
    assert 'q_mm_sim' not in dataset.dynamic_features
    assert 'flow_sim' not in dataset.dynamic_features

    assert len(dataset.static_features) == NUM_STATIC
    assert len(set(dataset.static_features)) == NUM_STATIC, "duplicate static names"
    # the standardised names needed by area()/stn_coords() must be present
    for f in ('area_km2', 'lat', 'long', 'elev_gauge_m', 'elev_catch_m'):
        assert f in dataset.static_features, f
    return


def test_simulated_flow_excluded():
    """the raw files DO ship a model-simulated streamflow column (``flow_sim``),
    but the class must not present it as a dynamic feature (observational-data-
    only policy). This confirms the exclusion is real (we are dropping an
    existing column, not a no-op) and would catch a regression that re-adds it."""
    logger.info("test_simulated_flow_excluded")
    stn = dataset.stations()[0]
    raw = _raw_ts(dataset, stn)
    assert 'flow_sim' in raw.columns, "raw source unexpectedly has no flow_sim column"
    df = dataset._read_stn_dyn(stn)
    assert 'flow_sim' not in df.columns and 'q_mm_sim' not in df.columns
    assert df.shape[1] == NUM_DYNAMIC == 9, df.shape
    return


def test_returns_copies():
    """cached lists must be copies: mutating the returned list/df index must
    not corrupt the instance."""
    logger.info("test_returns_copies")
    stns = dataset.stations()
    stns.append('XXX')
    assert 'XXX' not in dataset.stations()

    feats = dataset.dynamic_features
    feats.append('YYY')
    assert 'YYY' not in dataset.dynamic_features
    return


def test_read_stn_dyn_transform_fidelity():
    """Directly exercise the raw-CSV -> standardized transform (``_read_stn_dyn``
    / ``dyn_map``), independent of the netCDF cache. This is the core read path;
    the cache-based ``test_dynamic_fidelity`` below would NOT catch a regression
    here (e.g. gap-filling, downcasting, unit scaling) when a stale ``.nc`` is
    present, so it must be tested on its own."""
    logger.info("test_read_stn_dyn_transform_fidelity")
    for stn in random.sample(dataset.stations(), 5):
        raw = _raw_ts(dataset, stn)
        df = dataset._read_stn_dyn(stn)   # <-- the actual read+rename, not the cache
        assert list(df.columns) == EXPECTED_DYN_FEATURES, df.columns.tolist()
        assert df.shape == (DYN_LEN, NUM_DYNAMIC), df.shape
        # values are cast to the configured float precision (self.fp), nothing else
        assert (df.dtypes == dataset.fp).all(), df.dtypes.to_dict()
        for raw_c, std_c in RAW_TO_STD.items():
            # NaN gaps preserved exactly (no gap-filling / dropping)
            assert (df[std_c].isna().values == raw[raw_c].isna().values).all(), \
                f"{stn}:{std_c} NaN pattern changed by _read_stn_dyn"
            # the ONLY transformation is the precision cast: values must equal the
            # raw values cast to self.fp (no unit conversion / rescale / gap fill)
            assert np.array_equal(df[std_c].values,
                                  raw[raw_c].to_numpy(dtype=dataset.fp),
                                  equal_nan=True), \
                f"{stn}:{std_c} values changed by _read_stn_dyn"
    return


def test_dynamic_fidelity():
    """fetched dynamic values (the ``fetch`` / netCDF-cache path) must be
    identical (value AND unit) to a fresh read of the raw csv for several
    stations, including the observed-streamflow gaps which must be preserved as
    NaN."""
    logger.info("test_dynamic_fidelity")
    stns = random.sample(dataset.stations(), 5)
    for stn in stns:
        raw = _raw_ts(dataset, stn)
        _, dyn = dataset.fetch(stn, as_dataframe=True)
        df = dyn[stn]
        assert df.shape == (DYN_LEN, NUM_DYNAMIC), df.shape
        for raw_c, std_c in RAW_TO_STD.items():
            # NaN mask must match exactly (no gap filling / dropping)
            assert (df[std_c].isna().values == raw[raw_c].isna().values).all(), \
                f"{stn}:{std_c} NaN pattern changed"
            # values must equal the raw values cast to self.fp (the cache is built
            # at self.fp precision); no unit conversion / rescale / gap fill
            assert np.array_equal(df[std_c].values,
                                  raw[raw_c].to_numpy(dtype=dataset.fp),
                                  equal_nan=True), \
                f"{stn}:{std_c} values changed"
    # srad is MJ m-2 day-1: daily values are O(1-30), NOT the O(100-400) of W m-2
    _, dyn = dataset.fetch(stns[0], as_dataframe=True)
    assert dyn[stns[0]]['srad'].max() < 50, "srad looks like W m-2, not MJ m-2 day-1"
    return


def test_static_fidelity():
    """static attribute values must match the raw attribute files (checking a
    renamed column, a raw-kept column and a categorical column)."""
    logger.info("test_static_fidelity")
    topo = pd.read_csv(os.path.join(dataset._attr_dir, "topographic_attributes.csv"),
                       dtype={'gauge_id': str}).set_index('gauge_id')
    soil = pd.read_csv(os.path.join(dataset._attr_dir, "soil_attributes.csv"),
                       dtype={'gauge_id': str}).set_index('gauge_id')
    sf = dataset.fetch_static_features('all', 'all')

    stn = random.choice(dataset.stations())
    # renamed: area -> area_km2
    assert np.isclose(sf.loc[stn, 'area_km2'], topo.loc[stn, 'area'])
    # renamed: elev_median -> elev_catch_med_m (canonical median-elevation name)
    assert 'elev_catch_med_m' in sf.columns
    assert 'elev_median' not in sf.columns
    assert np.isclose(sf.loc[stn, 'elev_catch_med_m'], topo.loc[stn, 'elev_median'])
    # categorical string attribute preserved verbatim
    assert sf.loc[stn, 'soil_dominant_class'] == soil.loc[stn, 'soil_dominant_class']
    return


def test_manifest_completeness():
    """every thematic attribute file must cover the same 136 gauge_ids as the
    authoritative stations.csv manifest (guards against a truncated/interrupted
    extraction silently reporting fewer catchments)."""
    logger.info("test_manifest_completeness")
    meta = pd.read_csv(os.path.join(dataset._meta_dir, "stations.csv"),
                       dtype={'gauge_id': str})
    manifest = set(meta['gauge_id'])
    assert len(manifest) == NUM_STATIONS

    for name in ("topographic_attributes", "climatic_indices",
                 "hydrological_signatures", "landcover_attributes",
                 "geologic_attributes", "soil_attributes",
                 "human_intervention_attributes"):
        df = pd.read_csv(os.path.join(dataset._attr_dir, f"{name}.csv"),
                         dtype={'gauge_id': str})
        assert set(df['gauge_id']) == manifest, f"{name} does not cover all gauges"

    # every station also has an individual time-series file
    bc = os.path.join(dataset.ts_dir, "by_catchment")
    files = {f[:-4] for f in os.listdir(bc) if f.endswith('.csv')}
    assert files == manifest, "by_catchment files do not match stations manifest"
    return


def test_start_end_match_data():
    """the (hardcoded) start/end must equal the actual temporal extent of the
    raw files for a random sample of stations (catches drift)."""
    logger.info("test_start_end_match_data")
    for stn in random.sample(dataset.stations(), 12):
        raw = _raw_ts(dataset, stn)
        assert raw.index.min() == dataset.start, f"{stn} start {raw.index.min()}"
        assert raw.index.max() == dataset.end, f"{stn} end {raw.index.max()}"
    return


def test_q_mm_is_observed_mm():
    """q_mm must return the observed streamflow (already in mm/day) unchanged,
    i.e. it must NOT go through the cms->mm area conversion."""
    logger.info("test_q_mm_is_observed_mm")
    assert dataset._mm_feature_name == 'q_mm_obs'
    stn = random.choice(dataset.stations())
    raw = _raw_ts(dataset, stn)
    q = dataset.q_mm(stn)
    assert q.shape[1] == 1
    # q_mm returns flow_obs unchanged (only cast to self.fp), NOT a cms->mm conversion
    assert np.array_equal(q[stn].to_numpy(dtype=dataset.fp),
                          raw['flow_obs'].to_numpy(dtype=dataset.fp),
                          equal_nan=True)
    return


def test_boundary_is_wgs84():
    """catchment boundary is a MultiPolygon in valid WGS84 ranges over Peru."""
    logger.info("test_boundary_is_wgs84")
    from aqua_fetch.rr.utils import _make_boundary_2d
    from aqua_fetch._backend import fiona
    if fiona is None:
        return
    geom = dataset.get_boundary(dataset.stations()[0])
    for ring in _make_boundary_2d(geom):
        lons, lats = ring[:, 0], ring[:, 1]
        # Peru: lon ~ -81..-68, lat ~ -19..0
        assert (lons > -82).all() and (lons < -67).all(), "lon out of Peru range"
        assert (lats > -19).all() and (lats < 1).all(), "lat out of Peru range"
    return


def test_no_duplicate_false_positive():
    """the gauge 'San Pedro' occurs twice but at very different coordinates, so
    the (name + rounded-coords) duplicate check must NOT flag it."""
    logger.info("test_no_duplicate_false_positive")
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        CAMELS_PE(path=CAMELS_PE_PATH, verbosity=0)
    dup = [str(x.message) for x in w if 'duplicate' in str(x.message).lower()]
    assert dup == [], f"unexpected duplicate warning: {dup}"
    return


def test_no_redownload_or_reextract():
    """re-instantiating with the data already present must NOT call the
    download/unzip machinery (guard is on the extracted timeseries folder), and
    must NOT rebuild the netcdf cache. Both are asserted by making the
    respective operation raise if invoked (not by wall-clock timing)."""
    logger.info("test_no_redownload_or_reextract")
    orig_dl = _camels.download_and_unzip
    orig_nc = xr.Dataset.to_netcdf if xr is not None else None

    def _boom_dl(*a, **k):
        raise AssertionError("download_and_unzip must not be called when data exists")

    def _boom_nc(*a, **k):
        raise AssertionError("netcdf cache must not be rebuilt when it already exists")

    _camels.download_and_unzip = _boom_dl
    if xr is not None:
        xr.Dataset.to_netcdf = _boom_nc
    try:
        ds2 = CAMELS_PE(path=CAMELS_PE_PATH, verbosity=0)
        assert len(ds2.stations()) == NUM_STATIONS
        assert ds2.dyn_fpath_exists
    finally:
        _camels.download_and_unzip = orig_dl
        if xr is not None:
            xr.Dataset.to_netcdf = orig_nc
    return


def test_overwrite_removes_stale_before_download():
    """overwrite=True must delete the previously extracted directory, the stale
    archive (if present) AND the derived netCDF cache, all BEFORE re-downloading,
    so that stale data is never silently kept (and the cache is rebuilt from the
    fresh csv, not the old cache). Verified by recording the removal/download
    calls on ``_download_camels_pe`` directly (the actual removal and network
    download are stubbed out so no data is destroyed)."""
    logger.info("test_overwrite_removes_stale_before_download")
    calls = []
    orig_rmtree, orig_remove = _camels.shutil.rmtree, _camels.os.remove
    orig_dl = _camels.download_and_unzip

    _camels.shutil.rmtree = lambda p, *a, **k: calls.append(('rmtree', p))
    _camels.os.remove = lambda p, *a, **k: calls.append(('remove', p))
    _camels.download_and_unzip = lambda *a, **k: calls.append(('download', None))
    try:
        # call the download helper directly to isolate its removal-ordering logic
        dataset._download_camels_pe(overwrite=True)
    finally:
        _camels.shutil.rmtree = orig_rmtree
        _camels.os.remove = orig_remove
        _camels.download_and_unzip = orig_dl

    ops = [c[0] for c in calls]
    assert 'download' in ops, "overwrite=True must (re)download"
    # the stale extracted directory must be removed
    assert any(op == 'rmtree' and str(p).endswith('CAMELS-PE_v1.0.1')
               for op, p in calls), f"stale extracted dir not removed: {calls}"
    # the stale netCDF cache must be removed too
    assert any(op == 'remove' and str(p).endswith('.nc')
               for op, p in calls), f"stale netcdf cache not removed: {calls}"
    # every removal must happen BEFORE the (re)download
    dl_idx = ops.index('download')
    assert all(i < dl_idx for i, op in enumerate(ops) if op in ('rmtree', 'remove')), \
        f"stale data must be removed before re-download: {ops}"
    return


def test_duplicate_warning_fires():
    """the duplicate check must actually WARN when a genuine (name + coords)
    duplicate exists. Injected via a doctored stations.csv so the warn branch is
    exercised (the real data has no such duplicate)."""
    logger.info("test_duplicate_warning_fires")
    tmp = tempfile.mkdtemp()
    try:
        meta = pd.read_csv(os.path.join(dataset._meta_dir, "stations.csv"),
                           dtype={'gauge_id': str})
        # append an exact copy of the first gauge -> same name + coords
        doctored = pd.concat([meta, meta.iloc[[0]]], ignore_index=True)
        doctored.to_csv(os.path.join(tmp, "stations.csv"), index=False)

        orig_prop = CAMELS_PE._meta_dir
        CAMELS_PE._meta_dir = property(lambda self: tmp)
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                dataset._warn_duplicate_gauges()
            dup = [x for x in w if 'duplicate' in str(x.message).lower()]
            assert len(dup) == 1, f"expected exactly one duplicate warning, got {len(dup)}"
        finally:
            CAMELS_PE._meta_dir = orig_prop
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_fetch_outside_data_range():
    """fetching a window entirely outside the 1981-2025 extent must return an
    empty frame (not error), while a window inside returns data."""
    logger.info("test_fetch_outside_data_range")
    stn = dataset.stations()[0]
    _, before = dataset.fetch(stn, st='1900-01-01', en='1970-12-31', as_dataframe=True)
    assert len(before[stn]) == 0, before[stn].shape
    _, inside = dataset.fetch(stn, st='2010-01-01', en='2010-12-31', as_dataframe=True)
    assert len(inside[stn]) == 365, inside[stn].shape
    return


if __name__ == "__main__":
    random.seed(313)

    # dataset-specific checks
    test_registration()
    test_feature_names()
    test_simulated_flow_excluded()
    test_returns_copies()
    test_read_stn_dyn_transform_fidelity()
    test_dynamic_fidelity()
    test_static_fidelity()
    test_manifest_completeness()
    test_start_end_match_data()
    test_fetch_outside_data_range()
    test_q_mm_is_observed_mm()
    test_boundary_is_wgs84()
    test_no_duplicate_false_positive()
    test_duplicate_warning_fires()
    test_no_redownload_or_reextract()
    test_overwrite_removes_stale_before_download()

    # the standard, comprehensive suite shared by all rainfall-runoff datasets
    test_dataset(dataset,
                 num_stations=NUM_STATIONS,
                 dyn_data_len=DYN_LEN,
                 num_static_attrs=NUM_STATIC,
                 num_dyn_attrs=NUM_DYNAMIC,
                 yearly_steps=366,
                 st="20040101", en="20041231")

    print("*** All CAMELS_PE tests passed ***")
