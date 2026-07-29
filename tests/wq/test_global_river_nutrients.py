"""
Tests for :obj:`aqua_fetch.GlobalRiverNutrients` (Peters et al., 2026 global
river nutrient concentrations & streamflow dataset).

The tests verify that fetching is faithful (does not change the observed values
or their units), efficient and that every implemented method works. To keep the
suite fast (< 3-4 min) and avoid pulling the ~1.5 GB streamflow archive, only
the small TN/TP archives are downloaded; the streamflow reader is exercised on a
tiny synthetic fixture and the streamflow metadata (downloaded at init) is
tested directly.
"""
import os
import site
import tempfile
import warnings
import unittest

import numpy as np
import pandas as pd

# add the project root to the path
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

from aqua_fetch import GlobalRiverNutrients, ALL_DATASETS
import aqua_fetch.wq._global_river_nutrients as M

# default to the shared data location, overridable so the suite can run anywhere
DATA_PATH = os.environ.get('AQUA_FETCH_DATA', '/mnt/datawaha/hyex/atr/data')

# instantiating downloads only the (~7 MB) metadata for the three variables
ds = GlobalRiverNutrients(path=DATA_PATH, verbosity=0)


class TestGlobalRiverNutrients(unittest.TestCase):

    def test_registered(self):
        assert 'GlobalRiverNutrients' in ALL_DATASETS

    def test_parameters(self):
        assert ds.parameters() == ['TN', 'TP', 'streamflow']

    def test_station_counts(self):
        # available (redistributable) sites -> exactly the sites with a data file
        assert len(ds.stations('TN')) == 19100, len(ds.stations('TN'))
        assert len(ds.stations('TP')) == 40279, len(ds.stations('TP'))
        assert len(ds.stations('streamflow')) == 27497, len(ds.stations('streamflow'))
        # available_only=False returns every metadata row (incl. licensing-excluded)
        assert len(ds.stations('TN', available_only=False)) == 19424
        assert len(ds.stations('TP', available_only=False)) == 41639
        assert len(ds.stations('streamflow', available_only=False)) == 32103

    def test_stations_returns_fresh_list(self):
        # mutating the returned list must not corrupt the cached state
        stns = ds.stations('TN')
        n = len(stns)
        stns.append('junk')
        assert len(ds.stations('TN')) == n

    def test_metadata(self):
        m = ds.metadata('TN')
        assert m.shape == (19424, 9), m.shape
        assert m.index.name == 'WQ.ID'
        for c in ['WQ.lat', 'WQ.lon', 'region', 'start', 'end', 'WQ.available']:
            assert c in m.columns, c
        # start/end parsed to datetime
        assert pd.api.types.is_datetime64_any_dtype(m['start'])
        # returns a copy (mutation must not affect the cache)
        m.loc[m.index[0], 'region'] = 'ZZZ'
        assert ds.metadata('TN')['region'].iloc[0] != 'ZZZ'

    def test_source_metadata(self):
        sm = ds.metadata('TN', source=True)
        assert sm.shape == (20327, 9), sm.shape
        assert 'WQ.source' in sm.columns

    def test_coords(self):
        coords = ds.stn_coords('TP')
        assert coords.shape == (40279, 2), coords.shape
        assert coords.columns.tolist() == ['lat', 'long']
        # coordinates are plausible WGS84
        assert coords['lat'].between(-90, 90).all()
        assert coords['long'].between(-180, 180).all()
        one = ds.stn_coords('TN', 'WQ000002')
        assert one.shape == (1, 2)
        assert abs(one['lat'].iloc[0] - 55.9254) < 1e-6
        assert abs(one['long'].iloc[0] - (-130.0355)) < 1e-6

    def test_start_end(self):
        assert isinstance(ds.start, pd.Timestamp)
        assert isinstance(ds.end, pd.Timestamp)
        assert str(ds.start.date()) == '1806-01-01'
        assert str(ds.end.date()) == '2025-01-07'

    def test_fetch_tn(self):
        data = ds.fetch('TN', ['WQ000002', 'WQ000004'])
        assert set(data) == {'WQ000002', 'WQ000004'}
        assert data['WQ000002'].shape == (248, 1), data['WQ000002'].shape
        assert data['WQ000004'].shape == (159, 1)
        df = data['WQ000002']
        assert df.columns.tolist() == ['tot_N_mg/l']
        assert isinstance(df.index, pd.DatetimeIndex)
        # measurement column kept at full float64 precision (never downcast)
        assert df['tot_N_mg/l'].dtype == np.float64

    def test_fetch_stn(self):
        df = ds.fetch_stn('TN', 'WQ000004')
        assert df.shape == (159, 1)
        assert df.columns.tolist() == ['tot_N_mg/l']

    def test_fetch_tp(self):
        tp_stns = ds.stations('TP')[:2]
        data = ds.fetch('TP', tp_stns)
        assert set(data) == set(tp_stns)
        for s in tp_stns:
            assert data[s].columns.tolist() == ['tot_P_mg/l']

    def test_fetch_is_faithful(self):
        # the fetched values/units/dates must equal an INDEPENDENT read of the
        # raw per-site csv -- fetching must not alter the observations.
        raw_dir = os.path.join(ds.path, 'TN.2_-_Harmonised')
        raw = pd.read_csv(os.path.join(raw_dir, 'WQ000002.csv'))
        got = ds.fetch_stn('TN', 'WQ000002')
        assert len(raw) == len(got)
        np.testing.assert_array_equal(
            pd.to_datetime(raw['date']).values, got.index.values)
        np.testing.assert_array_equal(
            raw['value'].to_numpy(dtype='float64'),
            got['tot_N_mg/l'].to_numpy())

    def test_fetch_parallel_matches_serial(self):
        # >= 32 stations triggers the process-pool path; it must agree with the
        # serial (processes=1) path exactly.
        stns = ds.stations('TN')[:40]
        par = ds.fetch('TN', stns)                 # default -> parallel
        ser = ds.fetch('TN', stns, processes=1)    # serial
        assert set(par) == set(ser)
        for s in stns[:5]:
            pd.testing.assert_frame_equal(par[s], ser[s])

    def test_source_fetch(self):
        # raw source files retain the provenance columns unchanged
        data = ds.fetch('TN', ['ECCC_AK08DC0001'], source=True)
        df = data['ECCC_AK08DC0001']
        assert df.shape == (248, 3), df.shape
        assert df.columns.tolist() == ['WQ.source', 'WQ.ID', 'value']
        assert (df['WQ.source'] == 'ECCC').all()

    def test_aux_tables(self):
        assert ds.unavailable_stations('TN').shape == (326, 3)
        assert list(ds.unavailable_stations('TN').columns) == ['WQ.source', 'WQ.ID', 'WQ.ID2']
        assert ds.measurement_years('TN').shape == (15409, 7)
        assert ds.duplicates('TN').shape == (63227, 13)

    def test_streamflow_reader(self):
        # streamflow shares the generic reader; verify the streamflow value name
        # on a tiny synthetic fixture (avoids the ~1.5 GB streamflow archive).
        # 123456.789 is NOT exactly representable in float32 (it rounds to
        # 123456.7890625). Comparing as python floats (float64) proves the point;
        # a float32 read would fail the value assertion below.
        assert float(np.float32(123456.789)) != 123456.789
        with tempfile.TemporaryDirectory() as d:
            fp = os.path.join(d, 'S000001.csv')
            pd.DataFrame({'date': ['1806-01-01', '1806-01-02'],
                          'value': [123456.789, 0.0]}).to_csv(fp, index=False)
            df = M._read_site_csv(fp, 'q_cms_obs', source=False)
        assert df.columns.tolist() == ['q_cms_obs']
        assert isinstance(df.index, pd.DatetimeIndex)
        # large discharge kept exactly at float64 (would be corrupted by float32);
        # float() forces a float64 comparison so a float32 read genuinely fails
        assert df['q_cms_obs'].dtype == np.float64
        assert float(df['q_cms_obs'].iloc[0]) == 123456.789

    def test_streamflow_metadata_only(self):
        # streamflow metadata accessors work from the (small) metadata downloaded
        # at init, without pulling the ~1.5 GB per-site archive.
        coords = ds.stn_coords('streamflow')
        assert coords.shape == (27497, 2)
        assert coords['lat'].between(-90, 90).all()
        m = ds.metadata('streamflow')
        assert m.shape == (32103, 9)
        assert m.index.name == 'S.ID'

    def test_errors(self):
        self.assertRaises(AssertionError, ds.stations, 'XX')           # bad param
        self.assertRaises(ValueError, ds.fetch, 'TN', 'NOPE')          # scalar id
        self.assertRaises(ValueError, ds.fetch, 'TN', ['NOPE'])        # list form
        self.assertRaises(TypeError, ds.fetch, 'TN', 123)              # wrong type
        self.assertRaises(AssertionError, ds.fetch_stn, 'TN', 'NOPE')  # unknown
        self.assertRaises(AssertionError, ds.fetch_stn, 'TN', 123)     # non-str

    def test_invalid_ids_error_is_concise(self):
        # an unknown id must NOT dump the whole (~19k-id) allowed set to stdout;
        # the error should name only the offending ids and be short.
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with self.assertRaises(ValueError) as cm:
                ds.fetch('TN', ['NOPE1', 'NOPE2'])
        assert len(buf.getvalue()) < 1000, len(buf.getvalue())  # no id dump
        msg = str(cm.exception)
        assert 'NOPE1' in msg and len(msg) < 500, msg
        # a generator input is materialised, not silently drained
        got = ds.fetch('TN', (s for s in ['WQ000002', 'WQ000004']))
        assert set(got) == {'WQ000002', 'WQ000004'}

    def test_stn_coords_list_form(self):
        c = ds.stn_coords('TN', ['WQ000002', 'WQ000004'])
        assert c.shape == (2, 2)
        assert c.columns.tolist() == ['lat', 'long']

    def test_all_shortcircuit_matches_list(self):
        # the 'all' sentinel (short-circuited, O(n)) must give exactly the same
        # result as passing the explicit list of ids
        a = ds.stn_coords('TN')
        b = ds.stn_coords('TN', ds.stations('TN'))
        pd.testing.assert_frame_equal(a, b)

    def test_fetch_tp_is_faithful(self):
        stn = ds.stations('TP')[0]
        raw = pd.read_csv(os.path.join(ds.path, 'TP.2_-_Harmonised', stn + '.csv'))
        got = ds.fetch_stn('TP', stn)
        np.testing.assert_array_equal(
            raw['value'].to_numpy(dtype='float64'), got['tot_P_mg/l'].to_numpy())
        np.testing.assert_array_equal(
            pd.to_datetime(raw['date']).values, got.index.values)

    def test_source_fetch_is_faithful(self):
        raw = pd.read_csv(os.path.join(
            ds.path, 'TN.1_-_Source_Data', 'ECCC_AK08DC0001.csv'))
        got = ds.fetch_stn('TN', 'ECCC_AK08DC0001', source=True)
        np.testing.assert_array_equal(
            raw['value'].to_numpy(dtype='float64'), got['value'].to_numpy())
        assert got['WQ.source'].tolist() == raw['WQ.source'].tolist()
        assert got['WQ.ID'].tolist() == raw['WQ.ID'].tolist()

    def test_num_obs(self):
        # observations available per site == n.days == per-site file row count
        n = ds.num_obs('TN')
        assert len(n) == 19100
        assert int(n.median()) == 35
        assert int(n.sum()) == 1232518
        # equals the row count of a site's file (faithful per-site count)
        stn = 'WQ000002'
        rows = len(pd.read_csv(os.path.join(ds.path, 'TN.2_-_Harmonised', stn + '.csv')))
        assert int(n.loc[stn]) == rows == 248
        # returns a copy (mutation must not corrupt the cache)
        n.iloc[0] = -1
        assert ds.num_obs('TN').iloc[0] != -1

    def test_download_data(self):
        # happy path: TN is already extracted, so this is a no-op that must not
        # raise and must leave the extracted directory in place
        ds.download_data('TN')
        assert os.path.isdir(os.path.join(ds.path, 'TN.2_-_Harmonised'))

    def test_duplicate_check_warns(self):
        # the coordinate-duplicate warning fires (verbosity>0) for a parameter
        # known to contain co-located sites
        ds_v = GlobalRiverNutrients(path=DATA_PATH, verbosity=1)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            ds_v.stations('TN')  # triggers _meta -> _check_duplicates
            msgs = [str(x.message) for x in w if 'coordinates' in str(x.message)]
        assert msgs, 'expected a coordinate-duplicate warning'

    def test_incomplete_extraction_warns(self):
        # a directory with fewer per-site files than the manifest must warn
        ds_v = GlobalRiverNutrients(path=DATA_PATH, verbosity=0)
        with tempfile.TemporaryDirectory() as d:
            sub = os.path.join(d, 'TN.2_-_Harmonised')
            os.makedirs(sub)
            for s in ds_v.stations('TN')[:3]:
                open(os.path.join(sub, s + '.csv'), 'w').write('date,value\n2000-01-01,1\n')
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                ds_v._check_complete('TN', False, sub)
                msgs = [str(x.message) for x in w if 'incomplete' in str(x.message)]
        assert msgs, 'expected an incomplete-extraction warning'

    def test_data_dir_overwrite_reextracts(self):
        # overwrite=True must remove the stale EXTRACTED directory and re-extract
        # (the CLAUDE.md trap), stubbed so no real network / no shared-data clobber
        orig = M.download
        calls = {'dl': [], 'extract': []}

        def fake_dl(url, outdir, fname, verbosity=1):
            calls['dl'].append(fname)
            open(os.path.join(outdir, fname), 'w').write('zip')

        with tempfile.TemporaryDirectory() as tmp:
            d = GlobalRiverNutrients.__new__(GlobalRiverNutrients)
            d.verbosity = 0
            d._path = os.path.join(tmp, 'GlobalRiverNutrients')
            os.makedirs(d._path)
            d.overwrite = True
            d.remove_zip = False
            d._refreshed = set()
            d._files_map = {'TN.2_-_Harmonised.zip': 'http://x'}
            d._warned_streamflow = False
            d._check_complete = lambda *a, **k: None  # needs meta; skip here
            # a stale extracted dir with a marker file
            stale = os.path.join(d._path, 'TN.2_-_Harmonised')
            os.makedirs(stale)
            open(os.path.join(stale, 'STALE.csv'), 'w').write('x')

            def fake_extract(zp):
                calls['extract'].append(zp)
                os.makedirs(stale, exist_ok=True)
                open(os.path.join(stale, 'FRESH.csv'), 'w').write('x')
            d._extract = fake_extract

            M.download = fake_dl
            try:
                out = d._data_dir('TN', source=False)
            finally:
                M.download = orig

            assert calls['dl'] == ['TN.2_-_Harmonised.zip'], calls['dl']
            assert len(calls['extract']) == 1
            assert not os.path.exists(os.path.join(stale, 'STALE.csv'))  # stale gone
            assert os.path.exists(os.path.join(out, 'FRESH.csv'))         # fresh present

    def test_no_redownload_on_reinit(self):
        # re-instantiating when the data is present must not re-download anything
        orig = M.download

        def boom(*a, **k):
            raise AssertionError('download() called although data is present')

        M.download = boom
        try:
            ds2 = GlobalRiverNutrients(path=DATA_PATH, verbosity=0)
            _ = ds2.stations('TN')
            _ = ds2.start
            _ = ds2.fetch_stn('TN', 'WQ000002')  # data already extracted
        finally:
            M.download = orig

    def _stub_ds(self, tmp, overwrite):
        ds_tmp = GlobalRiverNutrients.__new__(GlobalRiverNutrients)
        ds_tmp.verbosity = 0
        ds_tmp._path = os.path.join(tmp, 'GlobalRiverNutrients')  # bypass setter
        os.makedirs(ds_tmp.path)
        ds_tmp._files_map = {'X.csv': 'http://example/X'}
        ds_tmp._refreshed = set()
        ds_tmp.overwrite = overwrite
        open(os.path.join(ds_tmp.path, 'X.csv'), 'w').write('old')  # pre-existing
        return ds_tmp

    def test_no_overwrite_keeps_existing(self):
        # a present file is not re-downloaded when overwrite=False
        calls, orig = [], M.download
        M.download = lambda url, outdir, fname, verbosity=1: calls.append(fname)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                self._stub_ds(tmp, overwrite=False)._download_file('X.csv')
            assert calls == []
        finally:
            M.download = orig

    def test_overwrite_refreshes_once(self):
        # overwrite=True removes the stale file and re-downloads exactly once per
        # session (a second call within the session does not re-download)
        calls, orig = [], M.download

        def stub(url, outdir, fname, verbosity=1):
            calls.append(fname)
            open(os.path.join(outdir, fname), 'w').write('new')

        M.download = stub
        try:
            with tempfile.TemporaryDirectory() as tmp:
                d = self._stub_ds(tmp, overwrite=True)
                d._download_file('X.csv')   # stale removed -> re-download
                d._download_file('X.csv')   # already refreshed -> no re-download
            assert calls == ['X.csv'], calls
        finally:
            M.download = orig


if __name__ == '__main__':
    unittest.main()
