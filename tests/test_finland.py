
import os
import site
wd_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(wd_dir)

import logging

if __name__ == "__main__":
    logging.basicConfig(filename='test_finland.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from water_datasets import Finland

from utils import test_dataset

gscad_path = '/mnt/datawaha/hyex/atr/gscad_database/raw'


ds = Finland(path=gscad_path, processes=1, 
             verbosity=3)

test_dataset(ds, 
             num_stations=669, 
             dyn_data_len=4199, 
             num_static_attrs=208,
              num_dyn_attrs=10,
              test_df=False,
              st="20200101",
              en="20201231"
              )
