
import os
import site   # so that aqua_fetch directory is in path
import logging
import unittest

# add the parent directory in the path
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

import pandas as pd

from aqua_fetch import CCAM
from aqua_fetch import CAMELS_CH
from aqua_fetch import CAMELS_GB, CAMELS_AUS
from aqua_fetch import CAMELS_CL, CAMELS_US, HYPE
from aqua_fetch import WaterBenchIowa
from aqua_fetch import CAMELS_DE
from aqua_fetch import CAMELS_SE
from aqua_fetch import CAMELS_IND
from aqua_fetch import CAMELS_FR
from aqua_fetch import RainfallRunoff
from aqua_fetch import RRLuleaSweden
from aqua_fetch import CAMELS_NZ
from aqua_fetch import CAMELS_LUX
from aqua_fetch import CAMELS_COL
from aqua_fetch import CAMELS_SK
from aqua_fetch.rr import CAMELS_FI


gscad_path = '/mnt/datawaha/hyex/atr/gscad_database/raw'

if __name__ == "__main__":
    logging.basicConfig(filename='test_camels.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from utils import test_dataset


class TestCamels(unittest.TestCase):

    def test_gb(self):
        ds_gb = CAMELS_GB(path=os.path.join(gscad_path, 'CAMELS'))
        test_dataset(ds_gb, 671, 16436, 145, 10)
        return

    def test_aus(self):
        ds_aus = CAMELS_AUS(path=os.path.join(gscad_path, 'CAMELS_AUS_V1'), version=1)
        test_dataset(ds_aus, 222, 23376, 166, 28)

        ds_aus = CAMELS_AUS(path=os.path.join(gscad_path, 'CAMELS'), version=2, verbosity=4)
        test_dataset(ds_aus, 561, 26388, 187, 28)
        return

    def test_hype(self):
        ds_hype = HYPE(path=gscad_path)
        test_dataset(ds_hype, 564, 12783, 0, 9)
        return

    def test_cl(self):
        ds_cl = CAMELS_CL(os.path.join(gscad_path, 'CAMELS'))
        test_dataset(ds_cl, num_stations=516, dyn_data_len=38374,
                     num_static_attrs=104, num_dyn_attrs=12)
        return

    def test_us(self):
        ds_us = CAMELS_US(path=os.path.join(gscad_path, 'CAMELS'), verbosity=4)
        test_dataset(ds_us, 671, 12784, 59, 8)
        return

    def test_ccam(self):
        ccam = CCAM(path=gscad_path)
        test_dataset(ccam, 102, 8035, 124, 16)
        return

    def test_ccam_meteo(self):
        dataset = CCAM(path=gscad_path)

        stations = os.listdir(dataset.meteo_path)

        for idx, stn in enumerate(stations):

            if stn not in ['35616.txt']:

                station = stn.split('.')[0]

                df = dataset._read_meteo_from_csv(station)

                assert df.shape == (11413, 9)

                if idx % 100 == 0:
                    logger.info(idx)
        return

    def test_waterbenchiowa(self):

        dataset = WaterBenchIowa(path=gscad_path)

        _, data = dataset.fetch(static_features=None)
        assert len(data) == 125
        for k, v in data.items():
            assert v.shape == (61344, 3)

        _, data = dataset.fetch(5, as_dataframe=True)
        assert data.shape == (184032, 5)

        _, data = dataset.fetch(5, static_features="all", as_dataframe=True)
        assert data['static'].shape == (5, 7)
        data = dataset.fetch_dynamic_features('644', as_dataframe=True)
        assert data.unstack().shape == (61344, 3)

        stns = dataset.stations()
        assert len(stns) == 125

        static_data = dataset.fetch_static_features(stns)
        assert static_data.shape == (125, 7)

        static_data = dataset.fetch_static_features('592')
        assert static_data.shape == (1, 7)

        static_data = dataset.fetch_static_features(stns, ['slope', 'area'])
        assert static_data.shape == (125, 2)

        data = dataset.fetch_static_features('592', features=['slope', 'area'])
        assert data.shape == (1,2)
        return

    def test_camels_ch(self):
        ds_swiss = CAMELS_CH(path=os.path.join(gscad_path, 'CAMELS'))
        test_dataset(ds_swiss, 331, 14610, 209, 9)

        ds_swiss = CAMELS_CH(path=os.path.join(gscad_path, 'CAMELS'), timestep='H')
        q = ds_swiss.read_hourly_q_ch(ds_swiss.hourly_stations()[0])
        assert pd.infer_freq(q.index) == 'H'

        return

    def test_camels_de(self):
        dataset = CAMELS_DE(path=os.path.join(gscad_path, 'CAMELS'))
        test_dataset(dataset, 1555, 25568, 111, 21)
        return

    def test_camels_se(self):
        dataset = CAMELS_SE(path=os.path.join(gscad_path, 'CAMELS'))
        test_dataset(dataset, 50, 21915, 76, 4)
        return

    def test_india(self):

        dataset = CAMELS_IND(path=os.path.join(gscad_path, 'CAMELS'))
        test_dataset(dataset, 472, 14976, 210, 20)
        return

    def test_fr(self):
        ds = CAMELS_FR(gscad_path, verbosity=4)
        test_dataset(
            ds,
            num_stations=654,
            dyn_data_len=18993,
            num_static_attrs=344,
            num_dyn_attrs=22,
        )
        return

    def test_rainfallrunoff(self):
        ds_aus = RainfallRunoff('CAMELS_AUS', path=os.path.join(gscad_path, 'CAMELS'),
                                 overwrite=True)
        test_dataset(ds_aus, 561, 26388, 187, 28)
        return

    def test_camels_nz(self):
        dataset = CAMELS_NZ(path=os.path.join(gscad_path, 'CAMELS'))
        test_dataset(dataset, 369, 460928, 39, 5, 
                     yearly_steps=8760  # this number might not be correct
                     )
        return

    def test_camels_lux(self):
        for ts, num_vals, yearly_steps in zip(
            ['D', 'H', '15Min'],
            [6209, 149016, 596061],
            [366, 8761, 35041],
        ):
            dataset = CAMELS_LUX(path=os.path.join(gscad_path, 'CAMELS'),
                                 timestep=ts)
            test_dataset(dataset, 56, num_vals, 61, 25, st="20120101", en="20121231", yearly_steps=yearly_steps)
        return

    def test_camels_col(self):
        dataset = CAMELS_COL(path=os.path.join(gscad_path, 'CAMELS'))
        test_dataset(dataset, 347, 15340, 255, 6)
        return

    def test_camels_sk(self):
        dataset = CAMELS_SK(path=os.path.join(gscad_path, 'CAMELS'))
        test_dataset(dataset, 178, 175320, 215, 17,
                     st="20120101", en="20121231", 
                     yearly_steps=8761)
        return

    def test_camels_fi(self):

        dataset = CAMELS_FI(path=os.path.join(gscad_path, 'CAMELS'), verbosity=4)
        test_dataset(dataset, 320, 23010, 106, 16)

        return


if __name__=="__main__":
    unittest.main()
