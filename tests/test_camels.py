
import os
import site   # so that water_datasets directory is in path
import random
import logging
import unittest

# add the parent directory in the path
wd_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(wd_dir)

import pandas as pd
import xarray as xr

from water_datasets import CABra
from water_datasets import CCAM
from water_datasets import CAMELS_DK
from water_datasets.rr import CAMELS_DK0
from water_datasets import CAMELS_CH
from water_datasets import CAMELS_GB, CAMELS_BR, CAMELS_AUS
from water_datasets import CAMELS_CL, CAMELS_US, LamaHCE, HYSETS, HYPE
from water_datasets import WaterBenchIowa
from water_datasets import CAMELS_DE
from water_datasets import LamaHIce
from water_datasets import GRDCCaravan
from water_datasets import CAMELS_SE
from water_datasets import Simbi
from water_datasets import Bull
from water_datasets import CAMELS_IND
from water_datasets import RainfallRunoff
from water_datasets import Arcticnet
from water_datasets import USGS


gscad_path = '/mnt/datawaha/hyex/atr/gscad_database/raw'

if __name__ == "__main__":
    logging.basicConfig(filename='test_camels.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from utils import (
    test_dataset, 
    test_dynamic_data,
    test_attributes
    )


class TestCamels(unittest.TestCase):

    def test_gb(self):
        ds_gb = CAMELS_GB(path=os.path.join(gscad_path, 'CAMELS'))
        test_dataset(ds_gb, 671, 16436, 145, 10)
        return

    def test_aus(self):
        ds_aus = CAMELS_AUS(path=os.path.join(gscad_path, 'CAMELS_AUS_V1'), version=1)
        test_dataset(ds_aus, 222, 23376, 166, 26)

        ds_aus = CAMELS_AUS(path=os.path.join(gscad_path, 'CAMELS'), version=2)
        test_dataset(ds_aus, 561, 26388, 187, 26)
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

    def test_lamah(self):
        stations = {'D': [859, 859, 454], 'H': [859, 859, 454]}
        static = {'D': [80, 81, 80], 'H': [84, 81, 80]}
        num_dyn_attrs = {'D': 22, 'H': 16}
        len_dyn_data = {'D': 14244, 'H': 341856}
        test_df = True
        yearly_steps = {'D': 366, 'H': 8784}

        for idx, dt in enumerate(LamaHCE._data_types):

            for ts in ['H', 'D']:

                if ts =='H':
                    test_df=False

                #if ts in ['D']:

                logger.info(f'checking for {dt} at {ts} time step')

                ds_eu = LamaHCE(timestep=ts, data_type=dt, path=gscad_path)

                test_dataset(ds_eu, stations[ts][idx],
                                len_dyn_data[ts], static[ts][idx], 
                                num_dyn_attrs=num_dyn_attrs[ts],
                                test_df=test_df, 
                                yearly_steps=yearly_steps[ts])
        return

    def test_br(self):
        ds_br = CAMELS_BR(path=os.path.join(gscad_path, 'CAMELS'))
        test_dataset(ds_br, 897, 14245, 67, 10)
        return

    def test_cabra(self):
        for source in ['era5', 'ref', 'ens']:
            dataset = CABra(path=gscad_path, met_src=source)
            test_dataset(dataset, 735, 10957, 97, 12)
        return

    def test_us(self):
        ds_us = CAMELS_US(path=os.path.join(gscad_path, 'CAMELS'))
        test_dataset(ds_us, 671, 12784, 59, 8)
        return

    def test_dk(self):
        ds_dk = CAMELS_DK0(path=os.path.join(gscad_path, 'CAMELS'))
        test_dataset(ds_dk, 308, 14609, 211, 39)
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

                stn_id = stn.split('.')[0]

                df = dataset._read_meteo_from_csv(stn_id)

                assert df.shape == (11413, 9)

                if idx % 100 == 0:
                    logger.info(idx)
        return


    def test_waterbenchiowa(self):

        dataset = WaterBenchIowa(path=gscad_path)

        data = dataset.fetch(static_features=None)
        assert len(data) == 125
        for k, v in data.items():
            assert v.shape == (61344, 3)

        data = dataset.fetch(5, as_dataframe=True)
        assert data.shape == (184032, 5)

        data = dataset.fetch(5, static_features="all", as_dataframe=True)
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

    def test_camels_dk_docs(self):

        dataset = CAMELS_DK0(path= os.path.join(gscad_path, 'CAMELS'))

        assert len(dataset.stations()) == 308
        assert dataset.fetch_static_features(dataset.stations()).shape == (308, 211)
        assert dataset.fetch_static_features('80001').shape == (1, 211)
        assert dataset.fetch_static_features(features=['gauge_lat', 'area']).shape == (308, 2)
        assert dataset.fetch_static_features('80001', features=['gauge_lat', 'area']).shape == (1, 2)

        df = dataset.fetch(stations=0.1, as_dataframe=True)
        assert df.index.names == ['time', 'dynamic_features']
        df = dataset.fetch(stations=1, as_dataframe=True)
        assert df.unstack().shape == (14609, 39)
        assert dataset.fetch(stations='80001', as_dataframe=True).unstack().shape == (14609, 39)

        df = dataset.fetch(1, as_dataframe=True,
                           dynamic_features=['snow_depth_water_equivalent_mean', 'temperature_2m_mean',
                                             'potential_evaporation_sum', 'total_precipitation_sum',
                                             'streamflow']).unstack()
        assert df.shape == (14609, 5)
        df = dataset.fetch(10, as_dataframe=True)
        assert df.shape == (569751, 10)

        data = dataset.fetch(stations='80001', static_features="all", as_dataframe=True)
        assert data['static'].shape == (1, 211)
        assert data['dynamic'].shape == (569751, 1)
        return

    def test_camels_de(self):
        dataset = CAMELS_DE(path=os.path.join(gscad_path, 'CAMELS'))
        test_dataset(dataset, 1555, 25568, 111, 21)
        return

    def test_lamahice(self):
        
        stations = {
            'D': [111, 107,  107],
            'H': [76]
            }
        length = {'D': 26298, 'H': 412825}
        num_dynamic = {'D': 36, 'H': 28}
        yr_steps = {'D': 366, 'H': 8784}

        for idx, data_type in enumerate(['total_upstrm', #'intermediate_all', 'intermediate_lowimp'
                          ]):

            for timestep in ['H', 'D']:
                
                logger.info(f'checking for {data_type}, {timestep}')

                dataset = LamaHIce(path=gscad_path, timestep=timestep, data_type=data_type)

                test_dataset(dataset, 
                             stations[timestep][idx], 
                             length[timestep], 
                             154, 
                             num_dynamic[timestep],
                             yearly_steps=yr_steps[timestep]
                             )
        return

    def test_grdccaravan(self):
        dataset = GRDCCaravan(path=gscad_path)
        test_dataset(dataset, 5357, 26801, 211, 39)
        return

    def test_camels_se(self):
        dataset = CAMELS_SE(path=os.path.join(gscad_path, 'CAMELS'))
        test_dataset(dataset, 50, 21915, 76, 4)
        return

    def test_simbi(self):
        dataset = Simbi(path=gscad_path)
        #test_dataset(dataset, 70, 17167, 232, 3, raise_len_error=False)
            # check that dynamic attribues from all data can be retrieved.
        test_dynamic_data(dataset, None, 70, 17167)
        test_dynamic_data(dataset, None, 70, 17167, as_dataframe=True)

        # check that dynamic data of 10% of stations can be retrieved
        test_dynamic_data(dataset, 0.1, int(7), 17167, 
                            raise_len_error=False)
        test_dynamic_data(dataset, 0.1, int(70*0.1), 17167, True,
                            raise_len_error=False)
        test_attributes(dataset, 232, 3, 70)

        dataset.area(dataset.static_data_stations())

        return

    def test_camels_dk():
        dataset = CAMELS_DK(path=os.path.join(gscad_path, 'CAMELS'))
        test_dataset(dataset, 304, 12782, 119, 13)
        return

    def test_bull(self):
        dataset = Bull(path=gscad_path)
        test_dataset(dataset, 484, 25932, 214, 55)
        return

    def test_india(self):

        dataset = CAMELS_IND(path=os.path.join(gscad_path, 'CAMELS'))
        test_dataset(dataset, 472, 14976, 210, 20)
        return

    def test_rainfallrunoff(self):
        ds_aus = RainfallRunoff('CAMELS_AUS', path=os.path.join(gscad_path, 'CAMELS'),
                                 overwrite=True)
        test_dataset(ds_aus, 561, 26388, 187, 26)
        return

    def test_arcticnet(self):
        dataset = Arcticnet(path=gscad_path)
        return

    def test_usgs(self):
        dataset = USGS(path=gscad_path, verbosity=3)
        test_dataset(dataset, 12004, 25202, 27, 5, test_df=False)
        return


if __name__=="__main__":
    unittest.main()
