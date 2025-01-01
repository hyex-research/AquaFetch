
import os
import site
wd_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(wd_dir)

import logging

if __name__ == "__main__":
    logging.basicConfig(filename='test_portugal.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import pandas as pd

from water_datasets import Portugal

from utils import test_dataset

gscad_path = '/mnt/datawaha/hyex/atr/gscad_database/raw'

ds = Portugal(path=gscad_path, verbosity=3)

q = ds.get_q()

assert q.shape == (18628, 280)

assert isinstance(q.index, pd.DatetimeIndex)

# find columns in q dataframe which are all NaN
nan_cols = q.columns[q.isna().all()]

test_dataset(ds, 
             num_stations=280, 
             dyn_data_len=18628, 
             num_static_attrs=208,
              num_dyn_attrs=10,
              test_df=False,
              )

