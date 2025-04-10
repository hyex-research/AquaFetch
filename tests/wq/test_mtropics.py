
import os
import site   # so that aqua_fetch directory is in path
wd_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(wd_dir)

import unittest

import pandas as pd

from aqua_fetch import MtropicsLaos, ecoli_mekong


laos = MtropicsLaos(path='/mnt/datawaha/hyex/atr/data/')


class TestMtropicsLaos(unittest.TestCase):

    def test_pcp(self):
        pcp = laos.fetch_pcp()
        assert isinstance(pcp.index, pd.DatetimeIndex)
        assert pcp.shape == (1665361, 1)
        assert pcp.index.freqstr == '6T'
        return

    def test_weather_station_data(self):
        w = laos.fetch_weather_station_data()
        assert isinstance(w.index, pd.DatetimeIndex)
        assert w.index.freq == 'H', f"{w.index.freq}"
        assert w.shape == (166536, 4)
        # assert int(w.isna().sum().sum()) == 82114
        return

    def test_fetch_hydro(self):
        wl, spm = laos.fetch_hydro()
        assert wl.shape == (454692, 1), f"{wl.shape}"
        assert isinstance(wl.index, pd.DatetimeIndex)
        assert spm.shape == (6428, 1)
        assert isinstance(spm.index, pd.DatetimeIndex)
        return

    def test_fetch_ecoli(self):
        ecoli = laos.fetch_ecoli()
        assert ecoli.shape == (409, 1)
        assert int(ecoli.isna().sum()) == 42, int(ecoli.isna().sum())
        assert isinstance(ecoli.index, pd.DatetimeIndex)
        ecoli_all = laos.fetch_ecoli(features='all')
        assert ecoli_all.shape == (409, 3)
        assert int(ecoli_all.isna().sum().sum()) == 162
        assert isinstance(ecoli_all.index, pd.DatetimeIndex)
        return

    def test_fetch_physio_chem(self):
        phy_chem = laos.fetch_physiochem('T_deg')
        assert phy_chem.shape == (411, 1)
        assert int(phy_chem.isna().sum()) == 63
        assert isinstance(phy_chem.index, pd.DatetimeIndex)
        phy_chem_all = laos.fetch_physiochem(features='all')
        assert phy_chem_all.shape == (411, 8)
        assert int(phy_chem_all.isna().sum().sum()) == 640
        assert isinstance(phy_chem_all.index, pd.DatetimeIndex)
        return

    def test_rain_gauges(self):
        rg = laos.fetch_rain_gauges()
        assert isinstance(rg.index, pd.DatetimeIndex)
        assert rg.index.freq == 'D'
        assert rg.shape == (6939, 7)
        assert int(rg.isna().sum().sum()) == 34510
        return

    def test_regression(self):

        df = laos.make_regression()
        assert isinstance(df.index, pd.DatetimeIndex)
        assert int(df.isna().sum().sum()) == 650483, int(df.isna().sum().sum())
        self.assertEqual(df.shape[-1], 9)

        return

    def test_regression_3min(self):

        df = laos.make_regression(freq="3min")
        assert isinstance(df.index, pd.DatetimeIndex)
        assert int(df.isna().sum().sum()) == 1301309, int(df.isna().sum().sum())
        self.assertEqual(df.shape[-1], 9)

        df = laos.make_regression(freq="3min", lookback_steps=1)
        assert int(df.isna().sum().sum()) == 0

        df = laos.make_regression(freq="3min", lookback_steps=1, en="20191231 00:00:00")

        return

    def test_regression_with_lookback(self):

        df = laos.make_regression(lookback_steps=30)
        assert df.shape == (5948, 9)
        assert int(df.isna().sum().sum()) == 5690, int(df.isna().sum().sum())

        return

    def test_regression_with_lookback_1(self):

        df = laos.make_regression(lookback_steps=1)
        assert int(df.isna().sum().sum()) == 0, int(df.isna().sum().sum())

        return

    def test_classification_with_lookback(self):

        df = laos.make_classification(lookback_steps=30)
        assert df.shape == (5948, 9)
        assert int(df.isna().sum().sum()) == 5690, int(df.isna().sum().sum())
        s = df['Ecoli_mpn100']
        assert (s == 0).sum() == 102
        self.assertEqual((s == 1).sum(), 156)

        return

    def test_classification_with_lookback_1(self):

        df = laos.make_classification(lookback_steps=1)
        assert int(df.isna().sum().sum()) == 0, int(df.isna().sum().sum())

        return

    def test_classification(self):

        df = laos.make_classification()
        assert isinstance(df.index, pd.DatetimeIndex)
        assert int(df.isna().sum().sum()) == 650483, int(df.isna().sum().sum())
        s = df['Ecoli_mpn100']
        assert (s == 0).sum() == 102
        self.assertEqual((s == 1).sum(), 156)

        return

    def test_ecoli_mekong(self):
        ecoli = ecoli_mekong()
        assert isinstance(ecoli, pd.DataFrame)
        return

    def test_ecoli_source(self):
        source = laos.fetch_source()
        assert source.shape == (252, 19)
        return


if __name__ == "__main__":
    unittest.main()
