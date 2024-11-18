
import os
import site
wd_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(wd_dir)

import logging

if __name__ == "__main__":
    logging.basicConfig(filename='test_japan.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from utils import test_dataset

from water_datasets import Japan

gscad_path = '/mnt/datawaha/hyex/atr/gscad_database/raw'


ds = Japan(path=gscad_path, verbosity=3)

test_dataset(ds, 
             num_stations=751, 
             dyn_data_len=16071, 
             num_static_attrs=35,
              num_dyn_attrs=27,
              test_df=False,
              )