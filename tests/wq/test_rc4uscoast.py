
import os
import site   # so that water_datasets directory is in path
wd_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(wd_dir)

import unittest

import numpy as np
import xarray as xr

from aqua_fetch import RC4USCoast


ds = RC4USCoast(path='/mnt/datawaha/hyex/atr/data')


class TestRC4USCoast(unittest.TestCase):

    def test_parameters(self):
        params = ds.parameters
        assert isinstance(params, list)
        assert len(params) == 28
        return

    def test_stations(self):
        stns = ds.stations
        assert isinstance(stns, list), type(stns)
        assert len(stns) == 140
        return

    def test_fetch_q(self):
        # get data of all stations as DataFrame
        q = ds.fetch_q("all")
        assert q.shape == (876, 140)

        # get data of only two stations
        q = ds.fetch_q([1, 10])
        assert q.shape == (876, 2)
        # get data as xarray Dataset
        q = ds.fetch_q("all", as_dataframe=False)
        assert isinstance(q, xr.Dataset)

        data = ds.fetch_q("all", as_dataframe=True, st="20000101", en="20181230")
        assert data.shape == (228, 140), data.shape
        return

    def test_fetch_chem(self):

        data = ds.fetch_chem(['temp', 'do'])
        assert isinstance(data, xr.Dataset)
        data = ds.fetch_chem(['temp', 'do'], as_dataframe=True)
        assert data.shape == (122640, 4)

        data = ds.fetch_chem(['temp', 'do'], st="19800101", en="20181230",
                             as_dataframe=True)
        assert data.shape == (65520, 4)
        return
    
    def test_coords(self):
        coords = ds.coords()
        assert coords.shape == (140, 2)
        assert np.all(coords.columns == ['lat', 'long'])
        return


if __name__ == "__main__":
    unittest.main()