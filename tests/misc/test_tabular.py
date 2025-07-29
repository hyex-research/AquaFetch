
import os
import site

# add the parent directory in the path
wd_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(wd_dir)

import unittest
from typing import Union

import pandas as pd

from aqua_fetch import busan_beach, gw_punjab
from aqua_fetch import WQJordan, WQJordan2, YamaguchiClimateJp, FlowBenin
from aqua_fetch import HydrometricParana, EtpPcpSamoylov, HydrocarbonsGabes
from aqua_fetch import Weisssee, RiverTempSpain, WQCantareira, RiverIsotope
from aqua_fetch import FlowSamoylov, FlowSedDenmark, StreamTempSpain
from aqua_fetch import HoloceneTemp, FlowTetRiver, SedimentAmersee
from aqua_fetch import PrecipBerlin, RiverTempEroo
from aqua_fetch import WaterChemEcuador, WaterChemVictoriaLakes, HydroChemJava
from aqua_fetch import GeoChemMatane, WeatherJena, SWECanada



def check_data(dataset, num_datasets=1,
               min_len_data=1, index_col: Union[None, str] = 'index'):
    data = dataset.fetch(index_col=index_col)
    assert len(data) == num_datasets, f'data is of length {len(data)}'

    for k, v in data.items():
        assert len(v) >= min_len_data, f'{v} if length {len(v)}'
        if index_col is not None:
            assert isinstance(v.index, pd.DatetimeIndex), f"""
            for {k} index is of type {type(v.index)}"""
    return


def test_jena_weather():
    wj = WeatherJena(path='/mnt/datawaha/hyex/atr/data')
    df = wj.fetch()

    assert df.shape[0] >= 919551
    assert df.shape[1] >= 21

    wj = WeatherJena(path='/mnt/datawaha/hyex/atr/data', obs_loc='soil')
    df = wj.fetch()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] >= 895770
    assert df.shape[1] >= 33

    wj = WeatherJena(path='/mnt/datawaha/hyex/atr/data',
                    obs_loc='saale')
    df = wj.fetch()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] >= 1102993
    assert df.shape[1] >= 30
    return


def test_swe_canada():
    swe = SWECanada(path='/mnt/datawaha/hyex/atr/data')

    stns = swe.stations()

    df = swe.fetch(1)
    assert len(df) == 1
    print('finished checking single stations')

    df = swe.fetch(10, st='20110101')
    assert len(df) == 10
    print('finished checking random 10 stations')

    df = swe.fetch(0.001, st='20110101')
    assert len(df) == 2
    print('finished checking 0.001 % stations')

    df = swe.fetch('ALE-05AE810', st='20110101')
    assert df['ALE-05AE810'].shape == (3500, 3)
    print('finished checking station ALE-05AE810')

    df = swe.fetch(stns[0:10], st='20110101')
    assert len(df) == 10
    print('finished checking first stations')

    return


class TestPangaea(unittest.TestCase):

    def test_Weisssee(self):
        dataset = Weisssee(path='/mnt/datawaha/hyex/atr/data')
        check_data(dataset, 21, 29)
        return

    def test_jordanwq(self):
        dataset = WQJordan(path='/mnt/datawaha/hyex/atr/data')
        check_data(dataset, 1, 428)
        return

    def test_jordanwq2(self):
        dataset = WQJordan2(path='/mnt/datawaha/hyex/atr/data')
        check_data(dataset, 1, 189)
        return

    def test_YamaguchiClimateJp(self):
        dataset = YamaguchiClimateJp(path='/mnt/datawaha/hyex/atr/data')
        check_data(dataset, 1, 877)
        return

    def test_FlowBenin(self):
        dataset = FlowBenin(path='/mnt/datawaha/hyex/atr/data')
        check_data(dataset, 4, 600)
        return

    def test_HydrometricParana(self):
        dataset = HydrometricParana(path='/mnt/datawaha/hyex/atr/data')
        check_data(dataset, 2, 1700)
        return

    def test_RiverTempSpain(self):
        dataset = RiverTempSpain(path='/mnt/datawaha/hyex/atr/data')
        check_data(dataset, 21, 400)
        return

    def test_WQCantareira(self):
        dataset = WQCantareira(path='/mnt/datawaha/hyex/atr/data')
        check_data(dataset, 1, 67)
        return
    
    def test_RiverIsotope(self):
        dataset = RiverIsotope(path='/mnt/datawaha/hyex/atr/data')
        check_data(dataset, 1, 398, index_col=None)
        return

    def test_EtpPcpSamoylov(self):
        dataset = EtpPcpSamoylov(path='/mnt/datawaha/hyex/atr/data')
        check_data(dataset, 1, 4214)
        return

    def test_FlowSamoylov(self):
        dataset = FlowSamoylov(path='/mnt/datawaha/hyex/atr/data')
        check_data(dataset, 1, 3292)
        return

    def test_FlowSedDenmark(self):
        dataset = FlowSedDenmark(path='/mnt/datawaha/hyex/atr/data')
        check_data(dataset, 1, 29663)
        return

    def test_StreamTempSpain(self):
        dataset = StreamTempSpain(path='/mnt/datawaha/hyex/atr/data')
        check_data(dataset, 1, 1)
        return

    def test_RiverTempEroo(self):
        dataset = RiverTempEroo(path='/mnt/datawaha/hyex/atr/data')
        check_data(dataset, 1, 138442)
        return

    def test_HoloceneTemp(self):
        dataset = HoloceneTemp(path='/mnt/datawaha/hyex/atr/data')
        check_data(dataset, 1, 1030, index_col=None)
        return

    def test_FlowTetRiver(self):
        dataset = FlowTetRiver(path='/mnt/datawaha/hyex/atr/data')
        check_data(dataset, 1, 7649)
        return

    def test_SedimentAmersee(self):
        dataset = SedimentAmersee(path='/mnt/datawaha/hyex/atr/data')
        check_data(dataset, 1, 455, index_col=None)
        return

    def test_HydrocarbonsGabes(self):
        dataset = HydrocarbonsGabes(path='/mnt/datawaha/hyex/atr/data')
        check_data(dataset, 1, 14, index_col=None)
        return

    def test_WaterChemEcuador(self):
        dataset = WaterChemEcuador(path='/mnt/datawaha/hyex/atr/data')
        check_data(dataset, 1, 10, index_col=None)
        return

    def test_WaterChemVictoriaLakes(self):
        dataset = WaterChemVictoriaLakes(path='/mnt/datawaha/hyex/atr/data')
        check_data(dataset, 1, 4, index_col=None)
        return

    def test_HydroChemJava(self):
        dataset = HydroChemJava(path='/mnt/datawaha/hyex/atr/data')
        check_data(dataset, 1, 40)
        return

    def test_PrecipBerlin(self):
        dataset = PrecipBerlin(path='/mnt/datawaha/hyex/atr/data')
        check_data(dataset, 1, 113952)
        return

    def test_GeoChemMatane(self):
        dataset = GeoChemMatane(path='/mnt/datawaha/hyex/atr/data')
        check_data(dataset, 1, 166)
        return

    def test_swe_canada(self):
        #test_swe_canada()
        return

    def test_jena_weather(self):
        test_jena_weather()
        return

    def test_gw_punjab(self):
        df = gw_punjab()
        assert df.shape == (68782, 5)
        assert isinstance(df.index, pd.DatetimeIndex)
        df_lts = gw_punjab("LTS")
        assert df_lts.shape == (7546, 4), df_lts.shape
        assert isinstance(df_lts.index, pd.DatetimeIndex)

        df = gw_punjab(country="IND")
        assert df.shape == (29172, 5)
        df = gw_punjab(country="PAK")
        assert df.shape == (39610, 5)
        return

    def test_busan(self):
        data = busan_beach()
        assert data.shape == (1446, 14)
        return


if __name__=="__main__":
    unittest.main()
