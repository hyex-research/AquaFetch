
import os
import site
# add the aqua_fetch directory in the path
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

import shutil
import tempfile
import unittest
import logging

if __name__ == "__main__":
    logging.basicConfig(filename='test_gsha.log', filemode='w', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

import pandas as pd
from aqua_fetch._backend import xarray as xr

from aqua_fetch import GSHA, Thailand, Japan, Arcticnet, Spain

from utils import (
    test_dataset as run_shared_tests,
    test_coords as check_coords,
    test_stations as check_stations,
    test_area as check_area,
    test_boundary as check_boundary,
    test_plot_stations as check_plot_stations,
    test_fetch_static_feature as check_fetch_static_feature,
    test_fetch_dynamic_features as check_fetch_dynamic_features,
    test_plot_catchment as check_plot_catchment,
    )

raw_data_path = '/path/to/your/raw/data'  # change this path accordingly

VERBOSITY = 3


class TestGSHA(unittest.TestCase):
    """
    Tests for the :class:`aqua_fetch.GSHA` dataset (21568 stations across 13
    agencies; 26 daily + 21 yearly dynamic features and 35 static features).

    A single instance is shared across all tests (built once in ``setUpClass``)
    so that GSHA's cached netcdf handles / csv tables are reused instead of
    being re-read for every test.
    """

    @classmethod
    def setUpClass(cls):
        cls.ds = GSHA(path=raw_data_path, verbosity=VERBOSITY)

    def test_dataset_attributes(self):
        assert self.ds.wsAll.shape == (21568, 4), self.ds.wsAll.shape
        assert len(self.ds.agencies) == 13, len(self.ds.agencies)
        return

    def test_stations(self):
        check_stations(self.ds, 21568)
        return

    def test_boundary(self):
        check_boundary(self.ds)
        return

    def test_plot_stations(self):
        check_plot_stations(self.ds)
        return

    def test_plot_catchment(self):
        check_plot_catchment(self.ds)
        return

    def test_atlas(self):
        assert self.ds.atlas().shape == (21568, 24), self.ds.atlas().shape
        assert self.ds.atlas(agency='arcticnet').shape == (106, 24), self.ds.atlas(agency='arcticnet').shape
        assert self.ds.atlas(stations='1001_arcticnet').shape == (1, 24), self.ds.atlas(stations='1001_arcticnet').shape
        return

    def test_uncertainty(self):
        assert self.ds.uncertainty().shape == (21568, 7), self.ds.uncertainty().shape
        assert self.ds.uncertainty(agency='arcticnet').shape == (106, 7), self.ds.uncertainty(agency='arcticnet').shape
        assert self.ds.uncertainty(stations='1001_arcticnet').shape == (1, 7), self.ds.uncertainty(stations='1001_arcticnet').shape
        return

    def test_area(self):
        check_area(self.ds)
        assert self.ds.area(agency='arcticnet').shape == (106,), self.ds.area(agency='arcticnet').shape
        assert self.ds.area(stations='1001_arcticnet').shape == (1,), self.ds.area(stations='1001_arcticnet').shape
        return

    def test_coords(self):
        check_coords(self.ds)
        assert self.ds.stn_coords(agency='arcticnet').shape == (106, 2), self.ds.stn_coords(agency='arcticnet').shape
        assert self.ds.stn_coords(stations='1001_arcticnet').shape == (1, 2), self.ds.stn_coords(stations='1001_arcticnet').shape
        return

    def test_lc_vars(self):
        assert self.ds.lc_variables_stn('1001_arcticnet').shape[1] == 3

        if xr is not None:
            out = self.ds.lc_variables()
            assert len(out) == 21568, len(out)
            assert out['1001_arcticnet'].shape[1] == 3

            out = self.ds.lc_variables(agency='arcticnet')
            assert len(out) == 106, len(out)
            assert out['1001_arcticnet'].shape[1] == 3
        return

    def test_streamflow_indices(self):
        # exercised with ``to_netcdf=False`` (reads the per-station csvs) so a
        # separate instance is used to avoid disturbing the shared one.
        ds1 = GSHA(path=raw_data_path, to_netcdf=False, verbosity=VERBOSITY)
        assert ds1.streamflow_indices_stn('1001_arcticnet').shape[1] == 16

        out = ds1.streamflow_indices()
        assert isinstance(out, dict), type(out)
        assert len(out) == 21568, len(out)
        assert out['1001_arcticnet'].shape[1] == 16

        out = ds1.streamflow_indices(agency='arcticnet')
        assert isinstance(out, dict), type(out)
        assert len(out) == 106, len(out)
        assert out['1001_arcticnet'].shape[1] == 16
        return

    def test_lai(self):
        assert isinstance(self.ds.lai_stn('1001_arcticnet'), pd.Series)

        lai = self.ds.fetch_lai()
        assert len(lai) == 21568, len(lai)
        assert lai.sizes['time'] == 14541, lai.sizes['time']

        out = self.ds.fetch_lai('1001_arcticnet')
        assert len(out) == 1, len(out)

        out = self.ds.fetch_lai(agency='arcticnet')
        assert len(out) == 106, len(out)
        return

    def test_storage(self):
        out = self.ds.storage_vars_stn('1001_arcticnet')
        assert isinstance(out, pd.DataFrame), type(out)
        assert out.shape[1] == 6, out.shape

        out = self.ds.fetch_storage_vars(agency='arcticnet')
        assert len(out) == 106, len(out)
        assert out['1001_arcticnet'].shape[1] == 6, out['1001_arcticnet'].shape

        out = self.ds.fetch_storage_vars()
        assert len(out) == 21568, len(out)
        return

    def test_meteo(self):
        out = self.ds.meteo_vars_stn('1001_arcticnet')
        assert isinstance(out, pd.DataFrame), type(out)
        assert out.shape[1] == 19, out.shape

        out = self.ds.fetch_meteo_vars(agency='arcticnet')
        assert len(out) == 106, len(out)

        out = self.ds.fetch_meteo_vars()
        assert len(out) == 21568, len(out)
        return

    def test_reservoir(self):
        out = self.ds.reservoir_variables_stn('1001_arcticnet')
        assert isinstance(out, pd.DataFrame), type(out)
        assert out.shape[1] == 2, out.shape

        out = self.ds.reservoir_variables(agency='arcticnet')
        assert len(out) == 106, len(out)
        assert out['1001_arcticnet'].shape[1] == 2, out['1001_arcticnet'].shape

        out = self.ds.reservoir_variables()
        assert len(out) == 21568, len(out)
        return

    def test_fetch_static_features(self):
        out = self.ds.fetch_static_features(agency='arcticnet')
        assert out.shape == (106, 35), out.shape

        check_fetch_static_feature(self.ds, '1001_arcticnet', 21568, 35)
        return

    def test_fetch_dynamic_features(self):
        out = self.ds.fetch_dynamic_features(agency='arcticnet')
        assert len(out) == 106, len(out)

        check_fetch_dynamic_features(self.ds, '1001_arcticnet', 16071)
        return

    def test_stn_dynamic_features(self):
        data = self.ds.fetch_stn_dynamic_features('1001_arcticnet')
        assert data.shape == (16071, 26), data.shape
        assert len(self.ds.dynamic_features) == 26

        data = self.ds.fetch_stn_dynamic_features(
            '1001_arcticnet',
            dynamic_features=['airtemp_C_mean_era5', 'pcp_mm_mswep'])
        assert data.shape == (16071, 2), data.shape
        return


class TestGSHADerivedDatasets(unittest.TestCase):
    """
    Datasets whose static/dynamic catchment attributes and boundaries are taken
    from GSHA (i.e. :class:`aqua_fetch.rr._gsha._GSHA` subclasses). These verify
    that GSHA correctly feeds the downstream classes.
    """

    def test_thailand(self):
        ds = Thailand(path=raw_data_path, verbosity=VERBOSITY)
        run_shared_tests(ds,
                     num_stations=73,
                     dyn_data_len=7305,
                     num_static_attrs=35,
                     num_dyn_attrs=27,
                     st="1992-01-01",
                     en="1992-12-31",
                     )
        return

    def test_japan(self):
        ds = Japan(path=raw_data_path, verbosity=VERBOSITY)
        run_shared_tests(ds,
                     num_stations=751,
                     dyn_data_len=16071,
                     num_static_attrs=35,
                     num_dyn_attrs=27,
                     )
        return

    def test_japan_hourly(self):
        ds = Japan(path=raw_data_path, timestep="h", verbosity=VERBOSITY)
        q = ds.fetch_q()
        assert pd.infer_freq(q.index) == 'h'
        return

    def test_arcticnet(self):
        ds = Arcticnet(path=raw_data_path, verbosity=VERBOSITY)
        run_shared_tests(ds,
                     num_stations=106,
                     dyn_data_len=9131,
                     num_static_attrs=35,
                     num_dyn_attrs=27,
                     st="1992-01-01",
                     en="1992-12-31",
                     )
        return

    def test_spain(self):
        ds = Spain(path=raw_data_path, verbosity=VERBOSITY)
        run_shared_tests(ds,
                     num_stations=889,
                     dyn_data_len=15249,
                     num_static_attrs=35,
                     num_dyn_attrs=27,
                     st="1992-01-01",
                     en="1992-12-31",
                     )
        return


class TestThailandSourceRemoved(unittest.TestCase):
    """
    The RID source files of Thailand are offline, so the class must raise an
    informative error instead of attempting a download. These tests need no data
    and no network: GSHA's own initialization is stubbed and ``_download`` raises
    if it is ever reached.
    """

    def setUp(self):
        from aqua_fetch.rr import _gsha

        def fake_gsha_init(ds, path=None, gsha_path=None, verbosity=1, **kwargs):
            # only what the check needs: the real path setter, which appends
            # the class name
            ds.verbosity = verbosity
            ds.path = path

        def no_download(ds, overwrite=False, **kwargs):
            raise AssertionError("a download was attempted")

        self._orig = (_gsha._GSHA.__init__, Thailand._download)
        _gsha._GSHA.__init__ = fake_gsha_init
        Thailand._download = no_download
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        from aqua_fetch.rr import _gsha
        _gsha._GSHA.__init__, Thailand._download = self._orig
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_raises_when_data_missing(self):
        with self.assertRaises(RuntimeError) as ctx:
            Thailand(path=self.tmp, verbosity=0)
        msg = str(ctx.exception)
        assert 'HTTP 404' in msg and 'disc_d_<year>_RIDall' in msg, msg
        assert 'daily_q.csv' in msg and 'web.archive.org' in msg, msg
        assert os.path.join(self.tmp, 'Thailand') in msg, msg
        return

    def test_overwrite_does_not_delete_local_data(self):
        fpath = os.path.join(self.tmp, 'Thailand', 'daily_q.csv')
        os.makedirs(os.path.dirname(fpath))
        with open(fpath, 'w') as fp:
            fp.write('time,C.1\n1980-01-01,1.0\n')
        with self.assertRaises(RuntimeError) as ctx:
            Thailand(path=self.tmp, overwrite=True, verbosity=0)
        assert 'cannot be downloaded again' in str(ctx.exception)
        assert os.path.exists(fpath), "overwrite=True deleted the local data"
        return

    def test_silent_when_data_present(self):
        os.makedirs(os.path.join(self.tmp, 'Thailand', 'disc_d_1980_RIDall'))
        # with the data on disk, the (real) download step only reports it is
        # already there, so a no-op stands in for it
        Thailand._download = lambda ds, overwrite=False, **kwargs: None
        ds = Thailand(path=self.tmp, verbosity=0)
        assert ds.path == os.path.join(self.tmp, 'Thailand')
        return


if __name__ == "__main__":
    unittest.main()
