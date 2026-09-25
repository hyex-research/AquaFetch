
import os
import site   # so that aqua_fetch directory is in path
import logging
import unittest

# add the parent directory in the path
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

import pandas as pd

from aqua_fetch import CCAM
from aqua_fetch import CAMELS_AUS
from aqua_fetch import CAMELS_US, HYPE
from aqua_fetch import WaterBenchIowa
from aqua_fetch import CAMELS_DE
from aqua_fetch import CAMELS_SE
from aqua_fetch import RainfallRunoff
from aqua_fetch import RRLuleaSweden
from aqua_fetch import CAMELS_COL
from aqua_fetch import CAMELS_SK
from aqua_fetch import CAMELS_PL
from aqua_fetch import CAMELSH


raw_data_path = '/path/to/raw/data'  # replace with actual path

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

    def test_aus(self):
        dataset = CAMELS_AUS(path=os.path.join(raw_data_path, 'CAMELS', 'CAMELS_AUS_V1'), version=1)
        test_dataset(dataset, 222, 23376, 166, 28)

        dataset = CAMELS_AUS(path=os.path.join(raw_data_path, 'CAMELS'), version=2, verbosity=4)
        test_dataset(dataset, 561, 26388, 187, 28)
        return

    def test_hype(self):
        dataset = HYPE(path=raw_data_path)
        test_dataset(dataset, 564, 12783, 0, 9)
        return

    def test_us(self):
        ds_us = CAMELS_US(path=os.path.join(raw_data_path, 'CAMELS'), verbosity=4)
        # 9 dynamic features: the 8 raw Daymet/USGS columns plus the derived
        # 24-h mean ``swdownrad_wm2``. Daymet's own srad is a daylight-period mean
        # and is kept, unaltered, as ``swdownrad_wm2_daylight``.
        test_dataset(ds_us, 671, 12784, 59, 9)
        return

    def test_ccam(self):
        dataset = CCAM(path=raw_data_path)
        test_dataset(dataset, 102, 8035, 124, 16)
        return

    def test_ccam_meteo(self):
        dataset = CCAM(path=raw_data_path)

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

        dataset = WaterBenchIowa(path=raw_data_path)

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

    # CAMELS_CH is tested in tests/rr/test_camels_ch.py, which also holds the
    # shared suite and the EPSG:2056 -> WGS84 checks that used to be here

    def test_camels_de(self):
        dataset = CAMELS_DE(path=os.path.join(raw_data_path, 'CAMELS'))
        test_dataset(dataset, 1582, 25568, 111, 21, test_latlong_ranges=False)
        return

    def test_camels_de_h(self):
        # hourly (CAMELS-DE-1h) data. The 109 static attributes come from the 7
        # attribute files (the modelled simulation_benchmark file is excluded);
        # all 26 observed + meteo-forcing timeseries columns are kept.
        # test_latlong_ranges=True here (unlike daily) validates that the hourly
        # boundaries are reprojected from EPSG:3035 to WGS84.
        dataset = CAMELS_DE(path=os.path.join(raw_data_path, 'CAMELS'),
                            timestep='H', verbosity=4)
        test_dataset(dataset, 1611, 210383, 109, 26,
                     yearly_steps=8760,
                     test_latlong_ranges=True)

        # the hourly index must be a regular hourly series
        _, dyn = dataset.fetch(stations='DE110000', dynamic_features='q_cms_obs',
                               as_dataframe=True)
        assert pd.infer_freq(dyn['DE110000'].index) in ['H', 'h'], \
            pd.infer_freq(dyn['DE110000'].index)
        return

    def test_camels_se(self):
        dataset = CAMELS_SE(path=os.path.join(raw_data_path, 'CAMELS'))
        test_dataset(dataset, 50, 21915, 76, 4)
        return

    def test_rainfallrunoff(self):
        dataset = RainfallRunoff('CAMELS_AUS', path=os.path.join(raw_data_path, 'CAMELS'),
                                 overwrite=True)
        test_dataset(dataset, 561, 26388, 187, 28)
        return

    def test_camels_col(self):
        # the February 2026 release: 346 gauges, 5 dynamic and 79 static
        # features. test_latlong_ranges=True validates that the boundaries are
        # reprojected from EPSG:3395 (World Mercator, meters) to WGS84
        dataset = CAMELS_COL(path=os.path.join(raw_data_path, 'CAMELS'),
                             remove_zip=False)
        test_dataset(dataset, 346, 15340, 79, 5, test_latlong_ranges=True)
        return

    def test_camels_sk(self):
        dataset = CAMELS_SK(path=os.path.join(raw_data_path, 'CAMELS'))
        test_dataset(dataset, 178, 175320, 215, 17,
                     st="20120101", en="20121231", 
                     yearly_steps=8761)
        return

    def test_camelsh(self):
        dataset = CAMELSH(path=os.path.join(raw_data_path, 'CAMELS'), verbosity=4)

        test_dataset(dataset, 5767, 394488, 779, 13)
        return

    def test_pl(self):
        dataset = CAMELS_PL(path=os.path.join(raw_data_path, 'CAMELS'), verbosity=4)
        # ~51 sec
        test_dataset(dataset, 354, 27029, 74, 13,
                     yearly_steps=366, st="20040101", en="20041231")
        return


if __name__=="__main__":
    unittest.main()
