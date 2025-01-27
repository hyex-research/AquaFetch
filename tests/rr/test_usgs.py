
import os
import site
# add the aqua_fetch directory in the path
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

import logging

if __name__ == "__main__":
    logging.basicConfig(filename='test_usgs.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import numpy as np 
 
from dataretrieval import nwis

from aqua_fetch import USGS
from aqua_fetch.rr._usgs import download_daily_q_nwis, _download_metadata

from utils import test_dataset

site = "14105700"


def test_daily_q():
    df_nwis = nwis.get_record(site, 
                    parameterCd="00060", 
                    start="1820-01-01",  # DAILY_START
                    end="2024-05-30",    # DAILY_END
                    service="dv",
                    )

    df_usgs = download_daily_q_nwis(site)

    np.testing.assert_array_equal(df_nwis.index, df_usgs.index)

    np.testing.assert_array_equal(df_nwis.columns, df_usgs.columns)

    np.testing.assert_array_equal(df_nwis.values, df_usgs.values)

    return


def test_metadata():
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


gscad_path = '/mnt/datawaha/hyex/atr/gscad_database/raw'

dataset = USGS(path=gscad_path, verbosity=3)
test_dataset(dataset, 12004, 25202, 27, 5, test_df=False)