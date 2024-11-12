
import os
import site
import re
from io import StringIO
# add the parent directory in the path
wd_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(wd_dir)
 

import numpy as np 
 
from dataretrieval import nwis

from water_datasets.rr._usgs import download_daily_q_nwis, _download_metadata

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