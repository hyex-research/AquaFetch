
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

    def test_gb(self):
        dataset = CAMELS_GB(path=os.path.join(raw_data_path, 'CAMELS'))
        test_dataset(dataset, 671, 16436, 145, 10, test_latlong_ranges=False)
        return

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

    def test_cl(self):
        ds_cl = CAMELS_CL(os.path.join(raw_data_path, 'CAMELS'))
        test_dataset(ds_cl, num_stations=516, dyn_data_len=38374,
                     num_static_attrs=104, num_dyn_attrs=12)
        return

    def test_us(self):
        ds_us = CAMELS_US(path=os.path.join(raw_data_path, 'CAMELS'), verbosity=4)
        test_dataset(ds_us, 671, 12784, 59, 8)
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

    def test_camels_ch(self):
        dataset = CAMELS_CH(path=os.path.join(raw_data_path, 'CAMELS'))
        test_dataset(dataset, 331, 14610, 209, 9)

        dataset = CAMELS_CH(path=os.path.join(raw_data_path, 'CAMELS'), timestep='H')
        q = dataset.read_hourly_q_ch(dataset.hourly_stations()[0])
        assert pd.infer_freq(q.index) in ['H', 'h'], pd.infer_freq(q.index)

        return

    def test_camels_ch_coords(self):
        # Regression test for the CH1903+/LV95 (EPSG:2056) -> WGS84 conversion of
        # both gauge coordinates (stn_coords) and catchment boundaries
        # (get_boundary). Before the fix, stn_coords returned the dataset's
        # 2-decimal (~500 m) gauge_lat/gauge_lon and get_boundary returned raw
        # EPSG:2056 easting/northing (values in the millions, not lon/lat).
        import numpy as np
        from utils import _make_boundary_2d
        from aqua_fetch._backend import fiona
        from aqua_fetch._geom_utils import epsg2056_point_to_wgs84

        dataset = CAMELS_CH(path=os.path.join(raw_data_path, 'CAMELS'))

        # Switzerland's WGS84 bounding box (with a small margin); every value
        # returned by the class must fall inside it.
        LON = (5.5, 11.0)
        LAT = (45.5, 48.2)

        # --- station coordinates ---
        coords = dataset.stn_coords()
        assert coords.shape == (331, 2), coords.shape
        assert list(coords.columns) == ['lat', 'long'], list(coords.columns)
        assert coords['lat'].between(*LAT).all(), "gauge lat out of CH range"
        assert coords['long'].between(*LON).all(), "gauge lon out of CH range"

        # station 2004 is 'Murten' (outlet of Lake Murten) at ~46.93 N, 7.12 E
        c = dataset.stn_coords('2004')
        assert abs(c['lat'].iloc[0] - 46.9308) < 1e-3, c['lat'].iloc[0]
        assert abs(c['long'].iloc[0] - 7.1169) < 1e-3, c['long'].iloc[0]

        # the precise E/N-derived coordinates must agree with the dataset's own
        # 2-decimal gauge_lat/gauge_lon (i.e. differ only by the ~0.005 deg
        # rounding), proving the conversion is correct and not an arbitrary shift
        topo = dataset.topo_attrs()
        lat_c, lon_c = epsg2056_point_to_wgs84(
            topo['gauge_easting'].values.astype(float),
            topo['gauge_northing'].values.astype(float))
        assert np.abs(lat_c - topo['gauge_lat'].values).max() < 6e-3
        assert np.abs(lon_c - topo['gauge_lon'].values).max() < 6e-3

        if fiona is None:
            return

        # --- catchment boundaries: WGS84, geometry type/structure preserved ---
        # 2004 = simple Polygon, 2022 = Polygon with an interior ring (hole),
        # 3028 = MultiPolygon. All must convert without raising and land in CH.
        for stn, gtype in [('2004', 'Polygon'), ('2022', 'Polygon'),
                           ('3028', 'MultiPolygon')]:
            geom = dataset.get_boundary(stn)
            assert geom.type == gtype, (stn, geom.type)
            pts = np.vstack(_make_boundary_2d(geom))
            assert pts[:, 0].min() >= LON[0] and pts[:, 0].max() <= LON[1], stn
            assert pts[:, 1].min() >= LAT[0] and pts[:, 1].max() <= LAT[1], stn
        return

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

    def test_india(self):

        dataset = CAMELS_IND(path=os.path.join(raw_data_path, 'CAMELS'))
        test_dataset(dataset, 472, 14976, 210, 20)
        return

    def test_fr(self):
        ds = CAMELS_FR(os.path.join(raw_data_path, 'CAMELS'), verbosity=4)
        test_dataset(
            ds,
            num_stations=654,
            dyn_data_len=18993,
            num_static_attrs=344,
            num_dyn_attrs=22,
            test_latlong_ranges=False
        )
        return

    def test_rainfallrunoff(self):
        dataset = RainfallRunoff('CAMELS_AUS', path=os.path.join(raw_data_path, 'CAMELS'),
                                 overwrite=True)
        test_dataset(dataset, 561, 26388, 187, 28)
        return

    def test_camels_nz(self):

        dataset = CAMELS_NZ(path=os.path.join(raw_data_path, 'CAMELS'), verbosity=3)
        test_dataset(dataset, 369, 19208, 40, 5, test_latlong_ranges=False)

        dataset = CAMELS_NZ(path=os.path.join(raw_data_path, 'CAMELS'), timestep='H', verbosity=4)
        test_dataset(dataset, 369, 460978, 40, 5, 
                        yearly_steps=8760,  # this number might not be correct
                        test_latlong_ranges=False
                        )
        return

    def test_camels_lux(self):
        for ts, num_vals, yearly_steps in zip(
            ['D', 'H', '15Min'],
            [6209, 149016, 596061],
            [366, 8761, 35041],
        ):
            dataset = CAMELS_LUX(path=os.path.join(raw_data_path, 'CAMELS'),
                                 timestep=ts)
            test_dataset(dataset, 56, num_vals, 61, 25, st="20120101", en="20121231", yearly_steps=yearly_steps)
        return

    def test_camels_col(self):
        dataset = CAMELS_COL(path=os.path.join(raw_data_path, 'CAMELS'))
        test_dataset(dataset, 347, 15340, 255, 6, test_latlong_ranges=False)
        return

    def test_camels_sk(self):
        dataset = CAMELS_SK(path=os.path.join(raw_data_path, 'CAMELS'))
        test_dataset(dataset, 178, 175320, 215, 17,
                     st="20120101", en="20121231", 
                     yearly_steps=8761)
        return

    def test_camels_fi(self):

        dataset = CAMELS_FI(path=os.path.join(raw_data_path, 'CAMELS'), verbosity=4)
        test_dataset(dataset, 320, 23010, 106, 16, test_latlong_ranges=False)

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
