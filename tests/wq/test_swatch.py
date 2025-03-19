
import os
import site   # so that aqua_fetch directory is in path
wd_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(wd_dir)

import unittest

import pandas as pd

from aqua_fetch import Swatch


ds = Swatch(path='/mnt/datawaha/hyex/atr/data')


class TestSwatch(unittest.TestCase):

    def test_sites(self):
        sites = ds.sites
        assert isinstance(sites, list)
        assert len(sites) == 26322
        return

    def test_fetch(self):

        df = ds.fetch()
        
        assert df.shape == (3901296, 6)

        st_name = "Jordan Lake"
        df1 = df[df['location'] == st_name]
        assert df1.shape == (4, 6)

        return
    
    def test_coords(self):
        coords = ds.coords
        assert isinstance(coords, pd.DataFrame)
        assert coords.shape == (26322, 2)
        return


if __name__ == "__main__":
    unittest.main()
