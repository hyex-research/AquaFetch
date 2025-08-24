
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
from aqua_fetch import CAMELS_FI
from aqua_fetch import CAMELSH


gscad_path = '/mnt/datawaha/hyex/atr/gscad_database/raw'

if __name__ == "__main__":
    logging.basicConfig(filename='test_camels.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from utils import test_dataset, test_dynamic_data
from utils import (
    test_static_data, test_all_data, test_attributes,
    test_fetch_dynamic_features, test_fetch_dynamic_multiple_stations,
    test_fetch_static_feature, test_st_en_with_static_and_dynamic,
    test_selected_dynamic_features, 
    test_fetch_station_features,
    test_area
)


class TestCamels(unittest.TestCase):

    def test_gb(self):
        dataset = CAMELS_GB(path=os.path.join(gscad_path, 'CAMELS'))
        test_dataset(dataset, 671, 16436, 145, 10)
        return

    def test_aus(self):
        dataset = CAMELS_AUS(path=os.path.join(gscad_path, 'CAMELS_AUS_V1'), version=1)
        test_dataset(dataset, 222, 23376, 166, 28)

        dataset = CAMELS_AUS(path=os.path.join(gscad_path, 'CAMELS'), version=2, verbosity=4)
        test_dataset(dataset, 561, 26388, 187, 28)
        return

    def test_hype(self):
        dataset = HYPE(path=gscad_path)
        test_dataset(dataset, 564, 12783, 0, 9)
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
        dataset = CCAM(path=gscad_path)
        test_dataset(dataset, 102, 8035, 124, 16)
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

        test_dynamic_data(dataset, 'all', 125, 61344)
        test_static_data(dataset, 'all', 125)
        test_all_data(dataset, 3, 61344, True)
        test_attributes(dataset, 7, 3, 125)
        test_fetch_dynamic_features(dataset, '592', 61344, True)
        test_fetch_dynamic_multiple_stations(dataset, 3, 61344, True)
        test_fetch_static_feature(dataset, '592', 125, 7)
        test_st_en_with_static_and_dynamic(dataset, '592', True, yearly_steps=8737, st='20130101', en='20131231')
        test_selected_dynamic_features(dataset, 61344, as_dataframe=True)
        test_fetch_station_features(dataset, 7, 3, 61344)
        test_area(dataset)
        return

    def test_camels_ch(self):
        dataset = CAMELS_CH(path=os.path.join(gscad_path, 'CAMELS'))
        test_dataset(dataset, 331, 14610, 209, 9)

        dataset = CAMELS_CH(path=os.path.join(gscad_path, 'CAMELS'), timestep='H')
        q = dataset.read_hourly_q_ch(dataset.hourly_stations()[0])
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
        dataset = RainfallRunoff('CAMELS_AUS', path=os.path.join(gscad_path, 'CAMELS'),
                                 overwrite=True)
        test_dataset(dataset, 561, 26388, 187, 28)
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

    def test_camelsh(self):
        dataset = CAMELSH(path=os.path.join(gscad_path, 'CAMELS'), verbosity=4)

        test_dataset(dataset, 5767, 394488, 779, 13)
        return


if __name__=="__main__":
    unittest.main()
