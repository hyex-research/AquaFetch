
import os
import site
# add the aqua_fetch directory in the path
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

import unittest

import logging

if __name__ == "__main__":
    logging.basicConfig(filename='test_usgs.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import numpy as np 
 
from dataretrieval import nwis

from aqua_fetch import USGS
from aqua_fetch import CAMELSH
from aqua_fetch.rr._usgs import download_daily_q_nwis, _download_metadata
from aqua_fetch.rr._usgs import download_hourly_q_nwis, download_hourly_record

from utils import test_dataset


raw_data_path = '/path/to/raw/data'  # replace with actual path


class TestUSGS(unittest.TestCase):

    def test_daily_q(self):
        site = "14105700"
        end = "2024-05-30"
        df_nwis = nwis.get_record(site, 
                        parameterCd="00060", 
                        start="1820-01-01",  # DAILY_START
                        end=end,    # DAILY_END
                        service="dv",
                        )

        df_usgs = download_daily_q_nwis(site, end=end)

        np.testing.assert_array_equal(df_nwis.index, df_usgs.index)

        np.testing.assert_array_equal(df_nwis.columns, df_usgs.columns)

        np.testing.assert_array_equal(df_nwis.values, df_usgs.values)

        return

    def test_metadata(self):
        site = "14105700"
        metadata_nwis = nwis.get_record(site, 
                        parameterCd="00060", 
                        service="site",
                        )

        metadata_usgs = _download_metadata(site)

        np.testing.assert_array_equal(metadata_nwis.index, metadata_usgs.index)

        np.testing.assert_array_equal(metadata_nwis.columns, metadata_usgs.columns)

        for i,j in zip(metadata_nwis.values.reshape(-1,), metadata_usgs.values.reshape(-1,)):
            np.testing.assert_array_equal([i], [j])
        return

    def test_hourly_q(self):
        site = "09246200"
        df_nwis = nwis.get_record(site, 
                        parameterCd="00060", 
                        start='1995-08-11',  
                        end='1995-08-12',    
                        service="iv",
                        )

        df_usgs = download_hourly_q_nwis(site)

        np.testing.assert_array_equal(df_nwis.index, df_usgs.index)

        np.testing.assert_array_equal(df_nwis.columns, df_usgs.columns)

        np.testing.assert_array_equal(df_nwis.values, df_usgs.values)

        return

    def test_with_camelsh(self):

        dataset = CAMELSH(path=os.path.join(raw_data_path, 'CAMELS'), verbosity=4)

        start = "1995-08-11"
        end = "1996-12-31"

        site = "09246200"
        q_camelsh = dataset.q([site])[site].sel(dynamic_features=['q_cms_obs']).to_pandas()


        path = os.path.join(raw_data_path, 'USGS', 'hourly_files')
        q_af = download_hourly_record(site, end=end, path=path, start=start)


        np.testing.assert_array_almost_equal(
            q_camelsh.loc[q_af.index[0]: q_af.index[-1], 'q_cms_obs'].values, 
            q_af.values,
            decimal=4,
            )
        return

    def test_dataset_class(self):
        dataset = USGS(path=raw_data_path, verbosity=2)
        test_dataset(dataset, 12004, 27028, 29, 20)
        return


if __name__ == "__main__":
    unittest.main()
