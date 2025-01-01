
import os
import site
wd_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(wd_dir)

import logging

if __name__ == "__main__":
    logging.basicConfig(filename='test_camels_fr.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


from water_datasets import RainfallRunoff, CAMELS_FR
from utils import test_dataset

gscad_path = '/mnt/datawaha/hyex/atr/gscad_database/raw'



ds = CAMELS_FR(gscad_path, verbosity=4)
#static_attrs = ds.static_attrs()
# out = ds.ts_attrs()
# static_data = ds.static_data()
# ds._read_dyn_stn(ds.stations()[0])

#ds = RainfallRunoff('CAMELS_FR', gscad_path)
test_dataset(
    ds,
    num_stations=654,
    dyn_data_len=18993,
    num_static_attrs=344,
    num_dyn_attrs=22,
)