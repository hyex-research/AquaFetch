
import os
import site
wd_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(wd_dir)

import logging

if __name__ == "__main__":
    logging.basicConfig(filename='test_poland.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from water_datasets import Poland

from utils import test_dataset

gscad_path = '/mnt/datawaha/hyex/atr/gscad_database/raw'


ds = Poland(path=gscad_path, verbosity=3)

test_dataset(ds, 
             num_stations=1287, 
             dyn_data_len=26844, 
             num_static_attrs=208,
              num_dyn_attrs=10,
              test_df=False,
              )
