
import os
import site
wd_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(wd_dir)

import logging

if __name__ == "__main__":
    logging.basicConfig(filename='test_ireland.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from water_datasets import Ireland

from utils import test_dataset

gscad_path = '/mnt/datawaha/hyex/atr/gscad_database/raw'


ds = Ireland(path=gscad_path, verbosity=3)

ds.basin_id_gauge_id_map()['IEOP0126'] == '20002'
ds.gauge_id_basin_id_map()['20002'] == 'IEOP0126'

test_dataset(ds, 
             num_stations=464, 
             dyn_data_len=26844, 
             num_static_attrs=208,
              num_dyn_attrs=10,
              test_df=False,
              )
