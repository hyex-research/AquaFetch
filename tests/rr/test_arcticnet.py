
import os
import site
wd_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(wd_dir)

import logging

if __name__ == "__main__":
    logging.basicConfig(filename='test_arcticnet.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from water_datasets import Arcticnet

from utils import (
    test_dataset,
    )

gscad_path = '/mnt/datawaha/hyex/atr/gscad_database/raw'


ds = Arcticnet(path=gscad_path, verbosity=3)

test_dataset(ds, 
             num_stations=106, 
             dyn_data_len=16071, 
             num_static_attrs=35,
              num_dyn_attrs=27,
              test_df=False,
              st="1992-01-01",
              en="1992-12-31",
              )