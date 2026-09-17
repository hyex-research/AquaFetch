"""
Tests for the harmonisation of radiation units across the rr datasets.

Every canonical radiation name ends in ``wm2`` and that is a promise. Several
source datasets distribute radiation in something else (MJ m-2 day-1,
J cm-2 day-1, J m-2 day-1) or, in Daymet's case, as a mean over the daylight
period rather than the 24-h day. These tests check that what the library
*ships* -- the factors declared on the classes -- actually discharges that
promise.

The physical test does not consult any dataset's metadata. It divides the
surface downward shortwave by the top-of-atmosphere irradiance for the same
latitude and day-of-year; the resulting clearness index Kt is bounded by physics
at 0 < Kt < ~0.8 regardless of who published the data. A wrong conversion factor
lands outside that band by exactly the factor error. Run against the pre-fix
code, the CAMELS_FR, CAMELS_AUS, CABra, HYSETS and CAMELS_US cases all fail.

The tests read the raw source files directly and use the classes' declared
factors *without instantiating them*, so nothing here downloads data, rebuilds a
netCDF cache or otherwise touches the user's data directory.
"""

import os
import re
import glob
import warnings

import numpy as np
import pandas as pd
import pytest

from aqua_fetch import (
    CAMELS_FR, CAMELS_AUS, CABra, HYSETS, CAMELS_US, CAMELS_GB, CAMELS_DE,
    CAMELS_IND, CAMELS_PL, EStreams, GSHA,
    Bull, GRDCCaravan, LamaHCE,
)
from aqua_fetch.rr._map import (
    solar_radiation,
    net_solar_radiation,
    max_net_solar_radiation,
    min_net_solar_radiation,
    max_solar_radiation,
    min_solar_radiation,
    solar_radiation_with_specifier,
    solar_radiation_with_spatial_stat,
    daylight_solar_radiation,
    downward_longwave_radiation,
    net_longwave_radiation,
    max_net_longwave_radiation,
    min_net_longwave_radiation,
    RAD_STATS,
    MJ_M2_DAY_TO_WM2,
    J_CM2_DAY_TO_WM2,
    J_M2_DAY_TO_WM2,
)
from aqua_fetch.rr.utils import (
    _RainfallRunoff,
    CACHE_VERSION,
    cache_name,
)

raw_data_path = '/path/to/raw/data'  # replace with actual path

GSC = 1367.0  # solar constant, W m-2

# a daily-mean clearness index outside this band is not physically achievable
# for a catchment-averaged, multi-year record
KT_LO, KT_HI = 0.25, 0.80


def toa_daily_mean(lat_deg, doy):
    """Extraterrestrial (top-of-atmosphere) irradiance averaged over the 24-h
    day, W m-2. Standard astronomical geometry (cf. FAO-56 Eq. 21); depends only
    on latitude and day-of-year, never on the dataset."""
    phi = np.radians(np.asarray(lat_deg, dtype='float64'))
    j = np.asarray(doy, dtype='float64')
    dr = 1 + 0.033 * np.cos(2 * np.pi * j / 365.0)
    dec = 0.409 * np.sin(2 * np.pi * j / 365.0 - 1.39)
    ws = np.arccos(np.clip(-np.tan(phi) * np.tan(dec), -1, 1))
    return (GSC / np.pi) * dr * (
        ws * np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.sin(ws))


def clearness_index(values, lat, index):
    """Mean Kt of a daily series, skipping polar night."""
    ra = toa_daily_mean(lat, index.dayofyear.values)
    ok = ra > 20
    return float(np.nanmean(np.asarray(values, dtype='float64')[ok] / ra[ok]))


def declared_factors(cls):
    """The ``dyn_factors`` a class ships, read without running ``__init__`` so
    that no download or cache rebuild is triggered."""
    return cls.dyn_factors.fget(object.__new__(cls))


def declared_dyn_map(cls, timestep='D'):
    obj = object.__new__(cls)
    # a few classes branch their mapping on the timestep; nothing else in these
    # property bodies touches state that ``__init__`` would have built
    obj.timestep = timestep
    return cls.dyn_map.fget(obj)


def _skip_if_absent(path):
    if not os.path.exists(path):
        pytest.skip(f"raw data not available at {path}")


# --------------------------------------------------------------------------
# the conversion constants themselves
# --------------------------------------------------------------------------

def test_conversion_constants():
    """Each constant converts a daily accumulation to a 24-h mean flux."""
    # 1 MJ m-2 spread over a day is 1e6 J / 86400 s = 11.574 W m-2
    assert MJ_M2_DAY_TO_WM2 == pytest.approx(11.5740740, rel=1e-6)
    # 1 J cm-2 == 1e4 J m-2
    assert J_CM2_DAY_TO_WM2 == pytest.approx(MJ_M2_DAY_TO_WM2 / 100.0, rel=1e-12)
    assert J_M2_DAY_TO_WM2 == pytest.approx(MJ_M2_DAY_TO_WM2 / 1e6, rel=1e-12)
    # a typical mid-latitude daily total of 10 MJ m-2 is ~116 W m-2
    assert 10 * MJ_M2_DAY_TO_WM2 == pytest.approx(115.74, abs=0.01)


# --------------------------------------------------------------------------
# _apply_dyn_factors
# --------------------------------------------------------------------------

def apply_dyn_factors(df, factors):
    """Calls the base-class method with ``factors`` as the dataset's
    ``dyn_factors``, without creating (and so downloading) a dataset."""
    holder = type('Holder', (), {'dyn_factors': factors})()
    return _RainfallRunoff._apply_dyn_factors(holder, df)


def test_apply_dyn_factors_scalar_and_callable():
    df = pd.DataFrame({'a': [1.0, 2.0], 'b': [10.0, 20.0], 'c': [5.0, 5.0]})
    apply_dyn_factors(df, {'a': 2.0, 'b': lambda x: x - 1, 'absent': 99.0})

    assert df['a'].tolist() == [2.0, 4.0]           # scalar multiplier
    assert df['b'].tolist() == [9.0, 19.0]          # callable
    assert df['c'].tolist() == [5.0, 5.0]           # untouched
    assert 'absent' not in df                       # missing column is skipped


def test_apply_dyn_factors_empty_is_noop():
    df = pd.DataFrame({'a': [1.0, 2.0]})
    before = df['a'].tolist()
    apply_dyn_factors(df, {})
    assert df['a'].tolist() == before


def test_hysets_converts_both_output_shapes():
    """HYSETS returns either a dict of DataFrames or an xarray Dataset, from
    three different read paths. The conversion sits at their shared exit so the
    paths cannot drift apart; check both shapes convert the same columns."""
    xr = pytest.importorskip('xarray')
    obj = object.__new__(HYSETS)

    raw = 1.2796e7          # a real daily J m-2 accumulation from the source file
    expected = raw / 86400.0

    dct = {'1': pd.DataFrame({solar_radiation(): [raw],
                              net_solar_radiation(): [raw],
                              'pcp_mm': [5.0]})}
    out = HYSETS._to_canonical_units(obj, dct)
    assert out['1'][solar_radiation()][0] == pytest.approx(expected)
    assert out['1'][net_solar_radiation()][0] == pytest.approx(expected)
    assert out['1']['pcp_mm'][0] == 5.0          # non-radiation left alone

    feats = [solar_radiation(), 'pcp_mm']
    ds = xr.Dataset(
        {'1': (['time', 'dynamic_features'], np.array([[raw, 5.0]]))},
        coords={'time': pd.date_range('2000-01-01', periods=1),
                'dynamic_features': feats})
    vals = HYSETS._to_canonical_units(obj, ds)['1'].values[0]
    assert vals[0] == pytest.approx(expected)
    assert vals[1] == 5.0


# --------------------------------------------------------------------------
# net vs downward shortwave
# --------------------------------------------------------------------------

@pytest.mark.parametrize('cls,key', [
    (Bull, 'D'), (GRDCCaravan, 'D'), (LamaHCE, 'D'), (LamaHCE, 'H'),
])
def test_net_shortwave_is_not_named_downward(cls, key):
    """A raw column that is net shortwave must not be served under the
    downward-shortwave name. net = downward x (1 - albedo), and over 120 random
    HYSETS watersheds that ratio has a median of 0.824 (albedo ~0.18), so
    conflating them is a silent ~18% bias."""
    dm = declared_dyn_map(cls)
    if key in dm and isinstance(dm[key], dict):
        dm = dm[key]

    downward_names = {solar_radiation(), max_solar_radiation(),
                      min_solar_radiation()}
    net_names = {net_solar_radiation(), max_net_solar_radiation(),
                 min_net_solar_radiation()}

    net_raw = [k for k in dm if 'net_solar' in k or 'net_solar_rad' in k]
    assert net_raw, f"no net-shortwave column found in {cls.__name__} dyn_map"

    for raw in net_raw:
        assert dm[raw] not in downward_names, (
            f"{cls.__name__} maps net shortwave '{raw}' to the downward name "
            f"'{dm[raw]}'")
        assert dm[raw] in net_names, (
            f"{cls.__name__} maps '{raw}' to unexpected '{dm[raw]}'")


def test_spatial_and_temporal_minima_no_longer_collide():
    """CAMELS_DE publishes the *spatial* min/max of downward shortwave across
    the catchment; BULL and GRDCCaravan publish the *temporal* min/max of net
    shortwave within the day. These used to produce the identical feature name
    ``solrad_wm2_min`` despite being different quantities."""
    de = set(declared_dyn_map(CAMELS_DE).values())
    bull = set(declared_dyn_map(Bull).values())
    caravan = set(declared_dyn_map(GRDCCaravan).values())

    assert solar_radiation_with_spatial_stat('min') in de     # swdownrad_wm2_spatmin
    assert min_net_solar_radiation() in bull                  # swnetrad_wm2_min
    assert min_net_solar_radiation() in caravan

    # no *radiation* name is shared between a spatial-stat publisher and a
    # temporal-stat one (sharing q_cms_obs etc. is of course fine)
    rad = lambda s: {n for n in s if 'rad_wm2' in n}
    assert not (rad(de) & rad(bull)), \
        f"CAMELS_DE and BULL still share radiation names: {rad(de) & rad(bull)}"

    # CAMELS_DE's spatial mean IS the default, so it carries no token at all
    assert solar_radiation() in de
    assert not any(n.endswith('_mean') for n in de if 'rad_wm2' in n)


def test_stat_token_cannot_be_passed_as_a_source():
    """The structural cause of the old collision: `_with_specifier` accepted an
    aggregation token, minting a name identical to the aggregation builder's."""
    for tok in ('min', 'max', 'mean', 'med', 'std', 'daylight'):
        with pytest.raises(ValueError, match='aggregation token'):
            solar_radiation_with_specifier(tok)
    # a genuine source is still fine
    assert solar_radiation_with_specifier('silo') == 'swdownrad_wm2_silo'


def test_every_radiation_name_obeys_the_convention():
    """<band><direction>rad_wm2[_<stat>][_<source>], fixed slot order, closed
    vocabularies. Parses every radiation feature the library serves."""
    pattern = re.compile(
        r'^(?P<band>sw|lw)(?P<dir>down|up|net)rad_wm2'
        r'(?:_(?P<rest>.+))?$')

    from aqua_fetch import Caravan_DK, CAMELS_FI, CAMELS_PE, CAMELSH, LamaHIce

    names = set()
    for cls in (CAMELS_US, CAMELS_GB, CAMELS_DE, CAMELS_FR, CAMELS_IND, CAMELS_PL,
                CAMELS_AUS, CABra, EStreams, GSHA, HYSETS, Bull, GRDCCaravan, LamaHCE,
                LamaHIce, Caravan_DK, CAMELS_FI, CAMELS_PE, CAMELSH):
        for ts in ('D', 'H'):
            dm = declared_dyn_map(cls, ts)
            for v in dm.values():
                if isinstance(v, dict):
                    names.update(x for x in v.values() if isinstance(x, str))
                elif isinstance(v, str):
                    names.add(v)
    rad = {n for n in names if 'rad_wm2' in n}
    assert len(rad) >= 12, f"expected the full radiation set, got {rad}"

    for n in sorted(rad):
        m = pattern.match(n)
        assert m, f"{n!r} does not obey <band><direction>rad_wm2[_stat][_source]"
        rest = m.group('rest')
        if rest:
            parts = rest.split('_')
            # stat, if present, must come first and be from the closed vocabulary
            if parts[0] in RAD_STATS:
                parts = parts[1:]
            # whatever remains is a source token and must NOT be a stat
            for p in parts:
                assert p not in RAD_STATS, (
                    f"{n!r}: {p!r} is a stat token sitting in the source slot")
        # the unit must stay at index 1, as it is for every other name in _map
        assert n.split('_')[1] == 'wm2', f"{n!r}: unit is not at field index 1"


def test_net_thermal_is_served_as_lwnet():
    """ERA5's "net thermal radiation" is net longwave. The old separate name
    `thermrad_wm2` is gone; the Caravan-style datasets serve it as lwnetrad_wm2."""
    from aqua_fetch import Caravan_DK
    expected = {'mean': net_longwave_radiation(),
                'max': max_net_longwave_radiation(),
                'min': min_net_longwave_radiation()}
    for cls, suffix in ((Bull, '_BULL'), (GRDCCaravan, ''), (Caravan_DK, '')):
        dm = declared_dyn_map(cls)
        for stat, name in expected.items():
            assert dm[f'surface_net_thermal_radiation_{stat}{suffix}'] == name, cls.__name__
        assert not any('thermrad' in str(v) for v in dm.values()), cls.__name__


# --------------------------------------------------------------------------
# the physical check: does the shipped factor produce a plausible Kt?
# --------------------------------------------------------------------------

def test_camels_fr_shortwave_kt():
    """CAMELS-FR distributes ``tsd_rad_ssi`` as J cm-2 accumulated over the day
    (``CAMELS-FR_description.ods``). Unconverted it implies more radiation at the
    ground than arrives at the top of the atmosphere."""
    ts_dir = os.path.join(raw_data_path, 'CAMELS', 'CAMELS_FR',
                          'CAMELS_FR_time_series', 'CAMELS_FR_time_series', 'daily')
    _skip_if_absent(ts_dir)

    factor = declared_factors(CAMELS_FR)[solar_radiation()]
    assert factor == pytest.approx(J_CM2_DAY_TO_WM2)

    f = sorted(glob.glob(os.path.join(ts_dir, '*.csv')))[0]
    d = pd.read_csv(f, sep=';', comment='#')
    d.index = pd.to_datetime(d['tsd_date'], format='%Y%m%d')
    s = pd.to_numeric(d['tsd_rad_ssi'], errors='coerce')

    lat = 47.6  # Loire basin; the first station file, from the outlet geopackage
    assert clearness_index(s.values, lat, d.index) > 1.0, (
        "raw J cm-2 values should be physically impossible as W m-2 -- if this "
        "fails the test is no longer detecting the bug it was written for")
    assert KT_LO < clearness_index(s.values * factor, lat, d.index) < KT_HI


def test_camels_aus_shortwave_kt():
    """The v2 Data Description gives ``radiation_silo.csv`` as MJ m-2."""
    f = os.path.join(raw_data_path, 'CAMELS', 'CAMELS_AUS', '05_hydrometeorology',
                     '05_hydrometeorology', '03_Other', 'SILO', 'radiation_SILO.csv')
    master = os.path.join(raw_data_path, 'CAMELS', 'CAMELS_AUS',
                          'CAMELS_AUS_Attributes&Indices_MasterTable.csv')
    _skip_if_absent(f)
    _skip_if_absent(master)

    factor = declared_factors(CAMELS_AUS)[solar_radiation_with_specifier('silo')]
    assert factor == pytest.approx(MJ_M2_DAY_TO_WM2)

    mt = pd.read_csv(master)
    mt = mt.set_index(mt.station_id.astype(str))
    rad = pd.read_csv(f)
    rad.index = pd.to_datetime(rad[['year', 'month', 'day']])

    stn = rad.columns[3]
    s = pd.to_numeric(rad[stn], errors='coerce').replace(-99.0, np.nan)
    lat = float(mt.loc[stn, 'lat_outlet'])

    assert clearness_index(s.values, lat, rad.index) < 0.1, (
        "raw MJ m-2 day-1 values should be absurdly small as W m-2")
    assert KT_LO < clearness_index(s.values * factor, lat, rad.index) < KT_HI


def test_cabra_shortwave_kt():
    """The units row inside each CABra climate file reads ``MJ m-2``."""
    f = os.path.join(raw_data_path, 'CABra', 'CABra_climate_daily_series',
                     'climate_daily', 'ens', 'CABra_100_climate_ENS.txt')
    _skip_if_absent(f)

    factor = declared_factors(CABra)[solar_radiation_with_specifier('ens')]
    assert factor == pytest.approx(MJ_M2_DAY_TO_WM2)

    with open(f) as fp:
        lat = float(fp.read().split('\n')[3].split()[1])

    d = pd.read_csv(f, sep=r'\s+', skiprows=13).iloc[1:]  # row 0 holds the units
    d = d.apply(pd.to_numeric, errors='coerce')
    d.index = pd.to_datetime(dict(year=d.Year, month=d.Month, day=d.Day))
    s = d['srad_ens'].replace(-999, np.nan)

    assert clearness_index(s.values, lat, d.index) < 0.1
    assert KT_LO < clearness_index(s.values * factor, lat, d.index) < KT_HI


def test_hysets_shortwave_kt():
    """The source netCDF variables carry ``units: 'J m-2'``."""
    nc_path = os.path.join(raw_data_path, 'HYSETS', 'HYSETS_2023_update_ERA5.nc')
    props_path = os.path.join(raw_data_path, 'HYSETS', 'HYSETS_watershed_properties.txt')
    _skip_if_absent(nc_path)
    _skip_if_absent(props_path)

    import netCDF4 as nc

    factor = declared_factors(HYSETS)[solar_radiation()]
    assert factor == pytest.approx(J_M2_DAY_TO_WM2)

    ds = nc.Dataset(nc_path)
    try:
        # the dataset states its own units -- assert we read them right
        assert ds['surface_downwards_solar_radiation'].units == 'J m-2'
        t = ds['time']
        index = pd.to_datetime(
            nc.num2date(t[:], t.units, only_use_cftime_datetimes=False))
        values = np.asarray(
            ds['surface_downwards_solar_radiation'][0, :], dtype='float64')
    finally:
        ds.close()

    lat = float(pd.read_csv(props_path)['Centroid_Lat_deg_N'].iloc[0])
    s = pd.Series(values, index=index).dropna()

    assert clearness_index(s.values, lat, s.index) > 100
    assert KT_LO < clearness_index(s.values * factor, lat, s.index) < KT_HI


def test_camels_us_srad_is_daylight_averaged():
    """Daymet reports srad as a mean over the daylight period only. The raw
    value exceeds the top-of-atmosphere daily mean; weighting by the daylight
    fraction recovers the 24-h mean that ``swdownrad_wm2`` promises."""
    forcing_dir = os.path.join(
        raw_data_path, 'CAMELS', 'CAMELS_US',
        'basin_timeseries_v1p2_metForcing_obsFlow',
        'basin_dataset_public_v1p2', 'basin_mean_forcing', 'daymet')
    if not os.path.exists(forcing_dir):
        pytest.skip(f"raw data not available at {forcing_dir}")

    f = sorted(glob.glob(os.path.join(forcing_dir, '*', '*_forcing_leap.txt')))[0]
    with open(f) as fp:
        lat = float(fp.read().split('\n')[0])

    d = pd.read_csv(f, sep=r'\s+', skiprows=3)
    d.columns = [c.lower() for c in d.columns]
    d.index = pd.to_datetime(dict(year=d.year, month=d.mnth, day=d.day))

    raw = d['srad(w/m2)'].values
    converted = raw * d['dayl(s)'].values / 86400.0

    # more radiation at the surface than at the top of the atmosphere
    assert clearness_index(raw, lat, d.index) > 1.0
    assert KT_LO < clearness_index(converted, lat, d.index) < KT_HI


def test_camels_us_publishes_both_the_native_and_the_24h_mean():
    """Deliberately needs no raw data. The assertions below used to sit inside
    test_camels_us_srad_is_daylight_averaged, behind its skip, so on a machine
    without the CAMELS_US forcing tree they never executed -- and a broken call
    in them went unnoticed. Anything checkable without data belongs in a test
    that cannot be skipped."""
    assert declared_dyn_map(CAMELS_US)['srad(W/m2)'] == daylight_solar_radiation()
    feats = CAMELS_US.dynamic_features.fget(object.__new__(CAMELS_US))
    assert solar_radiation() in feats            # derived 24-h mean
    assert daylight_solar_radiation() in feats   # Daymet's native value, kept
    assert 'dayl(s)' in feats                    # the weighting term stays visible
    assert len(feats) == len(set(feats)), 'duplicate feature name'


def test_no_data_free_assertion_hides_behind_a_skip():
    """Guard for the flaw above: every test that can skip must be reachable.
    Fails loudly if the data-free CAMELS_US checks are ever folded back into a
    skipping test."""
    import inspect
    import tests.rr.test_radiation_units as mod
    src = inspect.getsource(mod.test_camels_us_srad_is_daylight_averaged)
    assert 'declared_dyn_map' not in src, (
        'data-free assertions have been moved back inside a skipping test')


def test_camels_gb_shortwave_needs_no_conversion():
    """CAMELS-GB is already a daily-mean W m-2 (its supporting documentation says
    'catchment daily averaged downward short wave radiation W m-2'). A control:
    the test band must not reject data that was correct all along."""
    ts_dir = os.path.join(raw_data_path, 'CAMELS', 'CAMELS_GB', 'camels_gb',
                          'camels_gb', 'data', 'timeseries')
    topo = os.path.join(raw_data_path, 'CAMELS', 'CAMELS_GB', 'camels_gb',
                        'camels_gb', 'data', 'CAMELS_GB_topographic_attributes.csv')
    _skip_if_absent(ts_dir)
    _skip_if_absent(topo)

    assert solar_radiation() not in declared_factors(CAMELS_GB)

    t = pd.read_csv(topo)
    t = t.set_index(t.gauge_id.astype(str))
    f = sorted(glob.glob(os.path.join(ts_dir, '*.csv')))[0]
    sid = os.path.basename(f).split('_')[4]
    d = pd.read_csv(f, index_col='date', parse_dates=True)

    kt = clearness_index(d['shortwave_rad'].values, t.loc[sid, 'gauge_lat'], d.index)
    assert KT_LO < kt < KT_HI


# --------------------------------------------------------------------------
# longwave carried the same unit defect in the same files
# --------------------------------------------------------------------------

def test_published_values_are_never_modified_only_unit_converted():
    """The library converts units; it does not alter published values. Every
    declared factor must therefore be a pure unit conversion -- in particular
    no dataset may carry a -1 to normalise a sign convention."""
    for cls in (CAMELS_FR, CAMELS_AUS, CABra, HYSETS, CAMELS_US, LamaHCE, Bull,
                GRDCCaravan, CAMELS_GB, CAMELS_DE, GSHA):
        for factor in declared_factors(cls).values():
            if callable(factor):
                continue
            assert factor > 0, (
                f"{cls.__name__} declares a non-positive dyn_factor ({factor}); "
                f"negating published data is a modification, not a unit "
                f"conversion, and is not permitted")


def test_lamah_thermal_keeps_its_source_names_and_values():
    """LamaH-CE's net thermal radiation is positive-upward -- a different, valid
    convention. It is neither negated nor mapped onto ``lwnetrad_wm2`` (which
    this library defines positive-toward-the-surface); it keeps its source names
    so nothing is silently mixed, and the class warns."""
    dm = declared_dyn_map(LamaHCE, 'D')['D']
    for col in ('surf_net_therm_rad_mean', 'surf_net_therm_rad_max'):
        assert col not in dm, (
            f"{col} must stay unmapped: mapping it to a canonical name would "
            f"assert a sign convention the data does not follow")

    # net shortwave is positive under both conventions, so it IS harmonised
    assert dm['surf_net_solar_rad_mean'] == net_solar_radiation()
    assert dm['surf_net_solar_rad_max'] == max_net_solar_radiation()

    # and no factor touches the thermal columns
    factors = declared_factors(LamaHCE)
    assert net_longwave_radiation() not in factors
    assert all(v > 0 for v in factors.values() if not callable(v))


def test_lamah_warns_about_its_thermal_sign_convention():
    """The user must be told, since the values are served as published."""
    obj = object.__new__(LamaHCE)
    obj._dynamic_features = ['surf_net_therm_rad_mean', 'surf_net_therm_rad_max',
                             'q_cms_obs']
    with pytest.warns(UserWarning, match='POSITIVE-UPWARD'):
        LamaHCE._warn_thermal_sign(obj)

    # a dataset/timestep without those columns must not warn
    obj._dynamic_features = ['q_cms_obs', 'pcp_mm']
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        LamaHCE._warn_thermal_sign(obj)


def test_lwnetrad_means_positive_toward_surface_for_everyone_who_uses_it():
    """Only datasets that actually follow the convention are mapped to
    ``lwnetrad_wm2``; each must be predominantly negative."""
    checks = []
    bull = os.path.join(raw_data_path, 'Bull', 'timeseries', 'timeseries',
                        'csv', 'BULL', 'BULL_10004.csv')
    if os.path.exists(bull):
        assert net_longwave_radiation() in set(declared_dyn_map(Bull).values())
        checks.append(('Bull', pd.read_csv(bull)['surface_net_thermal_radiation_mean']
                       .astype(float).mean()))
    if not checks:
        pytest.skip('raw data not available')
    for name, mean in checks:
        assert mean < 0, f"{name}: {mean:.2f} W m-2 is not positive-toward-surface"


def test_the_datasets_that_were_missed_are_now_covered():
    """Caravan_DK, CAMELS_FI, CAMELS_PE, CAMELSH and CAMELS_SK all ship radiation
    and were all left unharmonised. Four are now mapped; CAMELS_SK cannot be."""
    from aqua_fetch import Caravan_DK, CAMELS_FI, CAMELS_PE, CAMELS_SK, CAMELSH
    from aqua_fetch.rr._map import KJ_M2_DAY_TO_WM2

    # Caravan extension: net, already W m-2, same six columns as BULL
    dk = declared_dyn_map(Caravan_DK)
    assert dk['surface_net_solar_radiation_mean'] == net_solar_radiation()
    assert dk['surface_net_thermal_radiation_min'] == min_net_longwave_radiation()
    assert not declared_factors(Caravan_DK), 'Caravan is already W m-2'

    # kJ m-2 day-1 (support document)
    fi = declared_dyn_map(CAMELS_FI)
    assert fi['radiation_global'] == solar_radiation()
    assert declared_factors(CAMELS_FI)[solar_radiation()] == pytest.approx(KJ_M2_DAY_TO_WM2)
    assert KJ_M2_DAY_TO_WM2 == pytest.approx(MJ_M2_DAY_TO_WM2 / 1000.0)

    # MJ m-2 day-1 (README)
    pe = declared_dyn_map(CAMELS_PE)
    assert pe['srad'] == solar_radiation()
    assert declared_factors(CAMELS_PE)[solar_radiation()] == pytest.approx(MJ_M2_DAY_TO_WM2)

    # CAMELS_SK's radiation is an ERA5-Land running accumulation on hourly rows.
    # No multiplicative factor can turn that into a flux, so it must stay
    # unmapped rather than be labelled W m-2.
    sk = declared_dyn_map(CAMELS_SK)
    assert 'surface_net_solar_radiation' not in sk
    assert 'surface_net_thermal_radiation' not in sk
    assert not any('rad_wm2' in str(v) for v in sk.values())

    # NLDAS-2 downward fluxes, already hourly W m-2: renamed, not converted
    h = declared_dyn_map(CAMELSH, 'H')
    assert h['SWdown'] == solar_radiation()
    assert h['LWdown'] == downward_longwave_radiation()
    assert not declared_factors(CAMELSH)


def test_longwave_conversions_declared():
    """CAMELS-FR's ``tsd_rad_dli`` and HYSETS' thermal variables are the same
    accumulation-vs-flux defect in the same dicts."""
    assert declared_factors(CAMELS_FR)[downward_longwave_radiation()] == \
        pytest.approx(J_CM2_DAY_TO_WM2)
    assert declared_factors(HYSETS)[downward_longwave_radiation()] == \
        pytest.approx(J_M2_DAY_TO_WM2)


# --------------------------------------------------------------------------
# cache files
# --------------------------------------------------------------------------

def test_cache_files_carry_the_version():
    """Cache files written before the radiation work hold the old feature names
    and units. Every cache name now includes CACHE_VERSION, so those old files
    are simply never found and fresh ones are built from the source files.
    This fails if any dataset that writes a cache stops using the version."""
    import inspect
    from aqua_fetch import LamaHCE
    from aqua_fetch.rr import _gsha, _lamah

    v = f"_v{CACHE_VERSION}"
    assert CACHE_VERSION >= 2, "version 1 is what the old, unversioned files were"
    assert cache_name('meteo_vars.nc') == f"meteo_vars{v}.nc"
    assert cache_name('total_upstrm_D') == f"total_upstrm_D{v}"      # folders too

    def fname(cls, **attrs):
        obj = object.__new__(cls)
        obj.name = cls.__name__
        for k, val in attrs.items():
            setattr(obj, k, val)
        return cls.dyn_fname.fget(obj)

    # the standard cache (used by 23 datasets) and the two datasets that name
    # theirs differently
    assert fname(CAMELS_GB, timestep='D') == f"camels_gb_D{v}.nc"
    assert fname(CABra, timestep='D', met_src='ens') == f"cabra_D_ens{v}.nc"
    assert fname(LamaHCE, timestep='D', data_type='total_upstrm').endswith(f"{v}.nc")

    # EStreams' meteorology cache
    e = object.__new__(EStreams)
    e._path = '/data/EStreams'
    assert EStreams.nc_path.fget(e).endswith(f"meteorology{v}.nc")

    # GSHA and LamaH build their cache paths in several places; none may be a
    # bare, unversioned name
    gsha = inspect.getsource(_gsha)
    bare = re.findall(r"""(?<!cache_name\()(['"])(lc_variables|reservoir_variables|"""
                      r"""streamflow_indices|lai|meteo_vars|storage|daily_q)\.nc\1""", gsha)
    assert not bare, f"unversioned GSHA cache names: {bare}"
    lamah = inspect.getsource(_lamah)
    assert not re.search(r'os\.path\.join\(self\.path, f"\{self\.data_type\}_\{self\.timestep\}"', lamah), \
        "LamaH per-feature cache folder is not versioned"
