
import os
import site   # so that aqua_fetch directory is in path
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

import warnings
import zipfile

import numpy as np
import pandas as pd

from aqua_fetch import CaravanQual
from aqua_fetch.wq.caravan_qual import _RECORD

raw_data_path = '/mnt/storage1/atr/data/gscad_database/raw'


# the constituent file used for most fidelity checks
_PARAM = 'Amoxicillin'
# a station and the Caravan gauge used for the deterministic assertions
_STN = 'wqms_01200003'
_GAUGE = 'grdc_4146360'


def _read_raw(ds, parameter):
    """reads a constituent csv directly (bypassing the class) for comparison."""
    return pd.read_csv(os.path.join(ds.csv_dir, f"{parameter}.csv"))


def test_metadata():
    ds = CaravanQual(path=raw_data_path, verbosity=0)

    assert len(ds.parameters) == 100
    assert len(ds.stations()) == 151859
    assert len(ds.gauges()) == 27073
    assert len(ds.countries()) == 134
    assert len(ds.linked_stations()) == 59724

    # units must be identical to the dataset's own data dictionary
    assert ds.parameter_units['TEMP'] == 'deg C'
    assert ds.parameter_units['EC'] == 'uS/cm'
    assert ds.parameter_units['pH'] == 'pH'
    assert ds.parameter_units['NO3N'] == 'mg/l'
    assert ds.parameter_units['As-Dis'] == 'ug/l'
    assert ds.parameter_units['EColi'] == 'cfu/100ml'

    assert ds.parameter_description['TEMP'] == 'Water Temperature'

    # every constituent has a unit
    assert set(ds.parameter_units) == set(ds.parameters)

    # start/end must describe the observations, not the narrower zarr grid
    assert ds.start == pd.Timestamp('1894-08-30')
    assert ds.end == pd.Timestamp('2025-12-15')
    assert ds.zarr_start == pd.Timestamp('1980-01-01')
    assert ds.zarr_end == pd.Timestamp('2025-09-30')
    assert ds.start < ds.zarr_start and ds.end > ds.zarr_end

    return


def test_start_end_do_not_drop_data():
    """
    ds.data(p, st=ds.start, en=ds.end) must be a no-op. When start/end were
    hardcoded to the zarr grid this silently discarded ~10% of the corpus.
    """
    ds = CaravanQual(path=raw_data_path, verbosity=0)

    for p in ['TEMP', 'EC', 'pH']:
        full = ds.data(p)
        bounded = ds.data(p, st=ds.start, en=ds.end)
        assert len(bounded) == len(full), (
            f"{p}: bounding by start/end dropped {len(full) - len(bounded)} rows")

    # and the pre-1980 observations really are measurements, not junk
    temp = ds.data('TEMP', en='1979-12-31')
    assert len(temp) > 900_000
    assert temp['flag'].isna().sum() > 800_000
    return


def test_no_redownload():
    """
    re-initialising must not re-download or re-extract the data. Rather than
    timing it (a proxy that passes for the wrong reasons), this replaces the
    download and extraction hooks with ones that fail loudly if reached.
    """
    from aqua_fetch.wq import caravan_qual as cq

    calls = []
    real_download = cq.download
    real_extractall = zipfile.ZipFile.extractall

    def spy_download(*args, **kwargs):
        calls.append(('download', args, kwargs))
        raise AssertionError("download() called although the data is present")

    def spy_extractall(self, *args, **kwargs):
        calls.append(('extractall', args, kwargs))
        raise AssertionError("extraction attempted although the data is present")

    cq.download = spy_download
    zipfile.ZipFile.extractall = spy_extractall
    try:
        ds = CaravanQual(path=raw_data_path, verbosity=0)
    finally:
        cq.download = real_download
        zipfile.ZipFile.extractall = real_extractall

    assert calls == [], calls
    assert os.path.isdir(ds.csv_dir)
    assert len(ds.parameters) == 100
    return


def test_data_fidelity():
    """the data() reader must not alter the values or units of the raw file."""
    ds = CaravanQual(path=raw_data_path, verbosity=0)

    raw = _read_raw(ds, _PARAM)
    d = ds.data(_PARAM)

    # same number of observations
    assert d.shape == (4013, 8)
    assert d.shape[0] == raw.shape[0]
    assert d.columns.tolist() == [
        'wqms_id', 'dates', 'obs', 'unit', 'flag',
        'detection_limit', 'imputation_method', 'streamflow']

    # values are preserved exactly (units unchanged)
    assert np.isclose(d['obs'].sum(), raw['obs'].sum())
    assert d['unit'].unique().tolist() == ['ug/l']
    assert d['wqms_id'].nunique() == 633

    # obs are read as float64 so large outliers are not corrupted to inf
    dosat = ds.data('DOSAT')
    assert not np.isinf(dosat['obs']).any()
    assert dosat['obs'].abs().max() > np.finfo(np.float32).max

    return


def test_data_station_and_time_filter():
    ds = CaravanQual(path=raw_data_path, verbosity=0)

    # single station
    d = ds.data('Atrazine', stations=_STN)
    assert d['wqms_id'].nunique() == 1
    assert len(d) == 68

    # time trimming
    d = ds.data('Atrazine', stations=_STN, st='2005-01-01', en='2005-12-31')
    assert d['dates'].min() >= pd.Timestamp('2005-01-01')
    assert d['dates'].max() <= pd.Timestamp('2005-12-31')
    assert len(d) == 22

    return


def test_fetch():
    ds = CaravanQual(path=raw_data_path, verbosity=0)

    # station-centric wide table
    data = ds.fetch(stations=[_STN], parameters=['Atrazine', 'NO3N'])
    assert list(data.keys()) == [_STN]
    df = data[_STN]
    assert df.columns.tolist() == ['Atrazine', 'NO3N']
    assert df.shape == (163, 2)
    assert isinstance(df.index, pd.DatetimeIndex)

    # single string arguments
    data = ds.fetch(stations=_STN, parameters='Atrazine')
    assert data[_STN].shape == (68, 1)

    # a station without the requested constituent is not returned
    data = ds.fetch(stations=['wqms_00100001'], parameters='Amoxicillin')
    assert data == {}

    # the vectorised implementation must match a reference per-station pivot,
    # including that each station only carries the constituents it has data for
    stns = ds.stations_with_parameter('Atrazine')[:30]
    got = ds.fetch(stations=stns, parameters=['Atrazine', 'NO3N'])
    raw = pd.concat([
        ds.data(p, stations=stns)[['wqms_id', 'dates', 'obs']].assign(variable=p)
        for p in ['Atrazine', 'NO3N']], ignore_index=True)
    # fetch drops NaN observations up front (they add nothing to the mean), so
    # the reference must too
    raw = raw[raw['obs'].notna()]
    for stn, g in raw.groupby('wqms_id'):
        ref = g.pivot_table(index='dates', columns='variable', values='obs',
                            aggfunc='mean')
        ref.columns.name = None
        ref = ref.reindex(columns=[c for c in ['Atrazine', 'NO3N']
                                   if c in ref.columns])
        assert stn in got
        assert got[stn].equals(ref), stn
    return


def test_fetch_duplicate_handling():
    """same-day replicate samples are averaged in the wide table."""
    ds = CaravanQual(path=raw_data_path, verbosity=0)

    raw = _read_raw(ds, _PARAM)
    dup = raw[(raw['wqms_id'] == 'wqms_01200311') & (raw['dates'] == '2019-07-02')]
    assert len(dup) == 2  # this station-date genuinely has two replicates

    data = ds.fetch(stations='wqms_01200311', parameters=_PARAM)
    val = data['wqms_01200311'].loc['2019-07-02', _PARAM]
    assert np.isclose(val, dup['obs'].mean())
    return


def test_gauge_helpers():
    ds = CaravanQual(path=raw_data_path, verbosity=0)

    assert len(ds.gauge_stations(_GAUGE)) == 206

    data = ds.fetch_by_gauge(_GAUGE, parameters='TEMP')
    assert list(data.keys()) == [_GAUGE]
    df = data[_GAUGE]
    # long-format, distinct wqms_id retained (no lossy aggregation across stations)
    assert 'wqms_id' in df.columns
    assert df.shape == (8635, 9)
    assert df['wqms_id'].nunique() == 164
    # streamflow is co-located gauge discharge attached to each sample
    assert 'streamflow' in df.columns

    return


def test_helpers():
    ds = CaravanQual(path=raw_data_path, verbosity=0)

    n = ds.num_obs(_PARAM)
    assert int(n.sum()) == 4013
    assert len(ds.stations_with_parameter(_PARAM)) == 633

    coords = ds.stn_coords()
    assert coords.shape == (151859, 2)
    assert coords.columns.tolist() == ['lat', 'long']
    single = ds.stn_coords('wqms_00100001')
    assert np.isclose(single['lat'].iloc[0], 17.09211)
    assert np.isclose(single['long'].iloc[0], -61.838)

    gc = ds.gauge_coords(_GAUGE)
    assert gc.columns.tolist() == ['lat', 'long']

    return


def test_attributes():
    """optional zarr-based catchment attributes (requires xarray + zarr)."""
    try:
        import xarray  # noqa: F401
        import zarr    # noqa: F401
    except (ImportError, ModuleNotFoundError):
        print("skipping attributes test (xarray/zarr not installed)")
        return

    ds = CaravanQual(path=raw_data_path, attributes=True, verbosity=0)

    sf = ds.static_features
    assert len(sf) == 205

    # attributes are indexed by river segment (LINKNO) and unchanged
    attrs = ds.attributes(['area', 'ele_mt_sav'])
    assert attrs.index.name == 'LINKNO'
    assert attrs.shape == (93544, 2)

    # the wqms -> LINKNO join must be faithful to a direct lookup
    info = ds.site_info()
    stn = 'wqms_00100001'
    linkno = int(info.loc[stn, 'LINKNO'])
    sa = ds.stn_attributes(stn, 'area')
    assert sa.index.tolist() == [stn]
    assert np.isclose(float(sa['area'].iloc[0]),
                      float(attrs.loc[linkno, 'area']))

    # every water quality station's LINKNO is covered by the attributes
    covered = info['LINKNO'].dropna().astype(int).isin(set(attrs.index)).mean()
    assert covered == 1.0

    return


def test_latest_zenodo_version():
    """
    the pinned record must still be the latest version of the concept record.
    Asserting _RECORD against its own literal would be a tautology, so this
    queries Zenodo and skips (rather than fails) when offline.
    """
    import json
    from urllib.request import urlopen
    from urllib.error import URLError

    try:
        with urlopen(
                f"https://zenodo.org/api/records/{_RECORD}/versions/latest",
                timeout=30) as resp:
            latest = json.load(resp)
    except (URLError, OSError, TimeoutError) as e:
        print(f"skipping Zenodo version check (no network): {e}")
        return

    assert str(latest['id']) == _RECORD, (
        f"pinned record {_RECORD} is outdated; latest is {latest['id']} "
        f"(v{latest['metadata'].get('version')})")
    # and it must really be the newest version of the concept record
    rel = latest['metadata'].get('relations', {}).get('version', [{}])[0]
    assert rel.get('is_last') is True
    return


def _fixture(tmp, ds, archive_constituents, stale_constituent=None,
             write_archive=True):
    """
    Builds an isolated dataset directory: symlinked metadata csvs, a
    ``wqms-csv.zip`` holding ``archive_constituents`` and (optionally) a stale
    extraction holding ``stale_constituent``.
    """
    import shutil
    import zipfile

    ds_dir = os.path.join(tmp, 'CaravanQual')
    os.makedirs(ds_dir, exist_ok=True)
    for f in ['wqms_site_info.csv', 'caravan_site_info.csv',
              'Caravan-Qual_zarr_variables.csv']:
        link = os.path.join(ds_dir, f)
        if not os.path.exists(link):
            os.symlink(os.path.join(ds.path, f), link)

    if write_archive:
        with zipfile.ZipFile(os.path.join(ds_dir, 'wqms-csv.zip'), 'w') as z:
            for c in archive_constituents:
                z.write(os.path.join(ds.csv_dir, f'{c}.csv'), f'wqms-csv/{c}.csv')

    if stale_constituent is not None:
        os.makedirs(os.path.join(ds_dir, 'wqms-csv'), exist_ok=True)
        shutil.copy(os.path.join(ds.csv_dir, f'{stale_constituent}.csv'),
                    os.path.join(ds_dir, 'wqms-csv', f'{stale_constituent}.csv'))
    return ds_dir


def test_overwrite_re_downloads_and_re_extracts():
    """
    overwrite=True must replace the local archive and re-extract from the newly
    fetched one.

    ``download()`` saves to ``<name>1`` when the target exists, so without an
    explicit remove the fresh archive lands in a file nothing reads and the
    STALE archive gets extracted. The download is stubbed here (no network), and
    the stub serves an archive whose contents differ from the stale one, so the
    assertion can only pass if the freshly downloaded archive was used.
    """
    import shutil
    import tempfile
    import zipfile
    from aqua_fetch.wq import caravan_qual as cq

    ds = CaravanQual(path=raw_data_path, verbosity=0)

    tmp = tempfile.mkdtemp(prefix='caravanqual_overwrite_')
    try:
        # on-disk (stale) archive holds Amoxicillin; the "remote" one holds two
        # different constituents
        ds_dir = _fixture(tmp, ds, archive_constituents=['Amoxicillin'],
                          stale_constituent='Amoxicillin')

        remote_zip = os.path.join(tmp, 'remote-wqms-csv.zip')
        with zipfile.ZipFile(remote_zip, 'w') as z:
            for c in ['As-Dis', 'Atrazine']:
                z.write(os.path.join(ds.csv_dir, f'{c}.csv'), f'wqms-csv/{c}.csv')

        attempted = []

        def fake_download(url, outdir=None, fname=None, verbosity=1):
            attempted.append(fname)
            dest = os.path.join(outdir, fname)
            assert not os.path.exists(dest), (
                f"download() called while {fname} still exists; it would be "
                f"saved as {fname}1 and the stale copy would be used")
            if fname == 'wqms-csv.zip':
                shutil.copy(remote_zip, dest)
            else:
                shutil.copy(os.path.join(ds.path, fname), dest)
            return dest

        real_download = cq.download
        cq.download = fake_download
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fresh = CaravanQual(path=tmp, overwrite=True, verbosity=0)
        finally:
            cq.download = real_download

        # every file was re-fetched ...
        assert 'wqms-csv.zip' in attempted, attempted
        # ... no stray "<name>1" file was created ...
        assert not any(f.endswith('.zip1') or f.endswith('.csv1')
                       for f in os.listdir(ds_dir)), os.listdir(ds_dir)
        # ... and the extraction came from the NEW archive, not the stale one
        on_disk = sorted(os.listdir(os.path.join(ds_dir, 'wqms-csv')))
        assert on_disk == ['As-Dis.csv', 'Atrazine.csv'], on_disk
        assert fresh.parameters == ['As-Dis', 'Atrazine']

        # an incomplete extraction must be reported, not silently truncated
        assert any('constituent files are missing' in str(c.message)
                   for c in caught), [str(c.message) for c in caught]
    finally:
        shutil.rmtree(tmp)
    return


def test_removed_zip_does_not_redownload():
    """
    With the archives deleted (remove_zip=True) but the data still extracted,
    instantiating again must not re-download the 641 MB archive.
    """
    import shutil
    import tempfile
    from aqua_fetch.wq import caravan_qual as cq

    ds = CaravanQual(path=raw_data_path, verbosity=0)

    tmp = tempfile.mkdtemp(prefix='caravanqual_nozip_')
    try:
        # extracted data present, archive absent - exactly the remove_zip state
        _fixture(tmp, ds, archive_constituents=[], stale_constituent='Amoxicillin',
                 write_archive=False)

        attempted = []

        def fake_download(url, outdir=None, fname=None, verbosity=1):
            attempted.append(fname)
            raise AssertionError(f"re-downloaded {fname} although it is extracted")

        real_download = cq.download
        cq.download = fake_download
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                again = CaravanQual(path=tmp, verbosity=0)
        finally:
            cq.download = real_download

        assert attempted == [], attempted
        assert again.parameters == ['Amoxicillin']
    finally:
        shutil.rmtree(tmp)
    return


def test_validate_ids_type_error():
    """scalar (non-iterable) input must give a clear message, not a TypeError
    from deep inside a comprehension."""
    ds = CaravanQual(path=raw_data_path, verbosity=0)
    try:
        ds.stn_coords(123)
        raise AssertionError("expected TypeError")
    except TypeError as e:
        assert 'unknown type' in str(e), str(e)
    return


def test_fetch_requires_parameters():
    """
    fetching all 100 constituents would need ~25 GB, so it must be explicit.
    Both None *and* 'all' must be refused: validate_attributes expands 'all'
    to the full list, so guarding only on None left the hole wide open.
    """
    ds = CaravanQual(path=raw_data_path, verbosity=0)

    for call in (lambda: ds.fetch(),
                 lambda: ds.fetch(stations=_STN),
                 lambda: ds.fetch(stations=_STN, parameters='all'),
                 lambda: ds.fetch(parameters='all'),
                 lambda: ds.fetch_by_gauge(_GAUGE),
                 lambda: ds.fetch_by_gauge(_GAUGE, parameters='all')):
        try:
            call()
            raise AssertionError("expected ValueError for None/'all' parameters")
        except ValueError as e:
            assert 'parameters' in str(e), str(e)
    return


def test_duplicate_gauges():
    """
    CLAUDE.md requires duplicates be detected by *gauge name*, reported and
    never excluded. Coordinate matching misses them: the same USGS station in
    two collections has coordinates differing at ~1e-6 degrees.
    """
    ds = CaravanQual(path=raw_data_path, verbosity=0)

    dups = ds.duplicate_gauges()
    assert len(dups) == 1258
    assert dups['duplicate_group'].nunique() == 628
    # the ALSEA RIVER pair: same name + trailing USGS id, different collection
    alsea = dups[dups['gauge_name'] == 'ALSEA RIVER NEAR TIDEWATER, OR']
    assert set(alsea.index) == {'camels_14306500', 'hysets_14306500'}
    assert alsea['duplicate_group'].nunique() == 1  # grouped together
    # a coordinate key would have missed them
    assert alsea['gauge_lat'].nunique() == 2

    # nothing is excluded: they are all still addressable gauges
    assert set(dups.index).issubset(set(ds.gauges()))

    # and the warning is emitted at verbosity>0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CaravanQual(path=raw_data_path, verbosity=1)
    assert any('share a gauge_name' in str(c.message) for c in caught)
    return


def test_iterable_inputs():
    """a one-shot iterable (generator) must not be silently emptied."""
    ds = CaravanQual(path=raw_data_path, verbosity=0)

    gen = (s for s in ['wqms_01200003'])
    from_gen = ds.data('Atrazine', stations=gen)
    from_list = ds.data('Atrazine', stations=['wqms_01200003'])
    assert len(from_gen) == len(from_list) == 68
    return


def test_internal_state_not_mutable():
    """callers must not be able to corrupt the cached lists."""
    ds = CaravanQual(path=raw_data_path, verbosity=0)

    stns = ds.stations()
    stns.append('bogus')
    assert 'bogus' not in ds.stations()

    params = ds.parameters
    params.clear()
    assert len(ds.parameters) == 100

    gauges = ds.gauges()
    gauges.clear()
    assert len(ds.gauges()) == 27073
    return


def test_attributes_available_when_on_disk():
    """
    if the zarr store is already extracted, attributes must be readable even
    when the instance was created with attributes=False - it is wrong to send
    the user off to re-download 1.5 GB they already have.
    """
    ds = CaravanQual(path=raw_data_path, attributes=False, verbosity=0)
    if not os.path.isdir(ds.zarr_dir):
        print("skipping: zarr store not extracted")
        return
    try:
        import xarray  # noqa: F401
        import zarr    # noqa: F401
    except (ImportError, ModuleNotFoundError):
        print("skipping: xarray/zarr not installed")
        return

    assert len(ds.static_features) == 205
    assert ds.attributes('area').shape == (93544, 1)
    return


def test_units_in_fetch():
    """
    9 constituents carry >1 unit string. Spelling variants of the same quantity
    must be kept; a genuinely different quantity must be dropped from the wide
    table (which has no column to carry a differing unit).
    """
    ds = CaravanQual(path=raw_data_path, verbosity=1)

    # TEMP: 'cel' is a spelling variant of 'deg C' -> kept, no warning
    temp = ds._read_constituent('TEMP')
    assert set(temp['unit'].dropna().unique()) == {'deg C', 'cel'}
    cel_stn = temp.loc[temp['unit'] == 'cel', 'wqms_id'].iloc[0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = ds.fetch(stations=cel_stn, parameters='TEMP')
    assert cel_stn in got and len(got[cel_stn]) > 0
    assert not any('dropped' in str(c.message) for c in caught)

    # POC: a single row is reported in '%' instead of 'mg/l' -> dropped + warned
    poc = ds._read_constituent('POC')
    pct_stn = poc.loc[poc['unit'] == '%', 'wqms_id'].iloc[0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds.fetch(stations=pct_stn, parameters='POC')
    assert any('dropped' in str(c.message) and 'POC' in str(c.message)
               for c in caught)

    # unit_counts exposes the composition
    counts = ds.unit_counts('TEMP')
    assert counts.sum() == len(temp)
    assert set(counts.index) == {'deg C', 'cel'}

    # POC is 187,793 mg/l + exactly one '%' row, and no row lacks a unit;
    # the missing-unit branch of _drop_foreign_units is purely defensive
    poc = ds._read_constituent('POC')
    assert poc['unit'].isna().sum() == 0
    assert poc['unit'].value_counts()['%'] == 1
    assert len(ds._drop_foreign_units(poc, 'POC')) == len(poc) - 1

    # 'ug' is a truncated spelling of 'ug/l', not a different quantity: its 48
    # values lie inside the range of the 'ug/l' ones, so they must be KEPT
    carb = ds._read_constituent('Carbamazepine')
    assert carb['unit'].value_counts()['ug'] == 48
    assert len(ds._drop_foreign_units(carb, 'Carbamazepine')) == len(carb)
    ug = carb.loc[carb['unit'] == 'ug', 'obs']
    ugl = carb.loc[carb['unit'] == 'ug/l', 'obs']
    assert ug.min() >= ugl.min() and ug.max() <= ugl.max()

    # 'no/100ml' is a different enumeration method from 'cfu/100ml', NOT a
    # spelling variant, so it must be dropped from the wide table (kept in data)
    ec = ds._read_constituent('EColi')
    assert ec['unit'].value_counts()['no/100ml'] == 79220
    assert len(ds._drop_foreign_units(ec, 'EColi')) == len(ec) - 79220

    # corpus-wide the wide table drops the 2 '%' rows plus 125,620 'no/100ml'
    dropped = 0
    for p in ds.parameters:
        d = ds._read_constituent(p)
        dropped += len(d) - len(ds._drop_foreign_units(d, p))
    assert dropped == 2 + 125620, dropped
    return


def test_drop_flagged():
    """drop_flagged removes the '*' outliers that drop_imputed does not."""
    ds = CaravanQual(path=raw_data_path, verbosity=0)

    # a station whose TEMP carries '*'-flagged values
    t = ds.data('TEMP', stations=None) if False else ds._read_constituent('TEMP')
    star_stn = t.loc[t['flag'] == '*', 'wqms_id'].iloc[0]
    raw = ds.data('TEMP', stations=star_stn)
    n_star = (raw['flag'] == '*').sum()
    assert n_star > 0
    assert raw['imputation_method'].notna().sum() == (raw['flag'] == '<').sum()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = ds.fetch(stations=star_stn, parameters='TEMP')[star_stn]
        # drop_imputed leaves the '*' rows in
        di = ds.fetch(stations=star_stn, parameters='TEMP',
                      drop_imputed=True)[star_stn]
        # drop_flagged removes them
        df_ = ds.fetch(stations=star_stn, parameters='TEMP',
                       drop_flagged=True)[star_stn]

    n_unflagged_dates = raw.loc[raw['flag'].isna(), 'dates'].nunique()
    assert len(df_) == n_unflagged_dates
    assert len(df_) < len(di)  # drop_flagged is stricter than drop_imputed
    return


def test_drop_imputed():
    """below-detection values substituted by LOD/2 or ROS can be excluded."""
    ds = CaravanQual(path=raw_data_path, verbosity=0)

    stn = 'wqms_01200311'
    raw = ds.data(_PARAM, stations=stn)
    assert raw['imputation_method'].notna().sum() == 1

    keep = ds.fetch(stations=stn, parameters=_PARAM)[stn]
    drop = ds.fetch(stations=stn, parameters=_PARAM, drop_imputed=True)[stn]

    measured = raw.loc[raw['imputation_method'].isna(), 'obs'].iloc[0]
    # default averages the imputed and the measured replicate ...
    assert np.isclose(keep[_PARAM].iloc[0], raw['obs'].mean())
    # ... drop_imputed keeps only the real measurement
    assert np.isclose(drop[_PARAM].iloc[0], measured)
    assert not np.isclose(keep[_PARAM].iloc[0], drop[_PARAM].iloc[0])
    return


def test_variables_and_helpers():
    ds = CaravanQual(path=raw_data_path, verbosity=0)

    v = ds.variables()
    assert v.columns.tolist() == ['variable', 'category', 'units', 'description']
    assert (v['category'] == 'water_quality').sum() == 100
    assert set(v['category']) >= {'water_quality', 'catchment_attributes',
                                  'weather', 'metadata', 'stream_attributes'}

    # fetch_by_gauge with an explicit gauge list
    data = ds.fetch_by_gauge([_GAUGE], parameters='Atrazine')
    assert list(data.keys()) == [_GAUGE]
    assert isinstance(data[_GAUGE], pd.DataFrame)
    return


def test_attributes_error_paths():
    """
    attribute methods must fail with a clear message when the zarr store is
    genuinely absent (as opposed to merely not requested - see
    test_attributes_available_when_on_disk).
    """
    import shutil
    import tempfile

    ds = CaravanQual(path=raw_data_path, verbosity=0)

    tmp = tempfile.mkdtemp(prefix='caravanqual_noattrs_')
    try:
        # csv data present, zarr store absent
        _fixture(tmp, ds, archive_constituents=[],
                 stale_constituent='Amoxicillin', write_archive=False)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bare = CaravanQual(path=tmp, attributes=False, verbosity=0)

        assert not os.path.isdir(bare.zarr_dir)

        for call in (lambda: bare.static_features,
                     lambda: bare.attributes('area'),
                     lambda: bare.stn_attributes('wqms_00100001', 'area')):
            try:
                call()
                raise AssertionError("expected ValueError without the zarr store")
            except ValueError as e:
                assert 'attributes=True' in str(e), str(e)
    finally:
        shutil.rmtree(tmp)
    return


def test_performance():
    """the documented one-liners must not be accidentally quadratic."""
    import time
    ds = CaravanQual(path=raw_data_path, verbosity=0)

    t0 = time.time()
    coords = ds.stn_coords()
    took = time.time() - t0
    assert coords.shape == (151859, 2)
    # was ~65 s before the 'all' short-circuit
    assert took < 5, f"stn_coords('all') took {took:.1f}s"
    return


test_metadata()
test_start_end_do_not_drop_data()
test_duplicate_gauges()
test_iterable_inputs()
test_internal_state_not_mutable()
test_attributes_available_when_on_disk()
test_no_redownload()
test_data_fidelity()
test_data_station_and_time_filter()
test_fetch()
test_fetch_duplicate_handling()
test_gauge_helpers()
test_helpers()
test_attributes()
test_latest_zenodo_version()
test_overwrite_re_downloads_and_re_extracts()
test_removed_zip_does_not_redownload()
test_validate_ids_type_error()
test_fetch_requires_parameters()
test_units_in_fetch()
test_drop_imputed()
test_drop_flagged()
test_variables_and_helpers()
test_attributes_error_paths()
test_performance()

print('All CaravanQual tests passed!')
