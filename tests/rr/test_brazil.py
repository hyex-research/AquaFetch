
import os
import site   # so that aqua_fetch directory is in path
import logging

# add the parent directory in the path
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

gscad_path = '/mnt/datawaha/hyex/atr/gscad_database/raw'

if __name__ == "__main__":
    logging.basicConfig(filename='test_brazil.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from aqua_fetch import CAMELS_BR, CABra

from utils import (
    test_dataset
    )


for source in ['era5', 'ref', 'ens']:
    dataset = CABra(path=gscad_path, met_src='era5', verbosity=4)
    test_dataset(dataset, 735, 10957, 87, 13)


ds_br = CAMELS_BR(path=os.path.join(gscad_path, 'CAMELS'), verbosity=3)
test_dataset(ds_br, 897, 14245, 67, 11)