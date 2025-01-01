
import os
import site   # so that water_quality directory is in path
import logging

# add the parent directory in the path
wd_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(wd_dir)

gscad_path = '/mnt/datawaha/hyex/atr/gscad_database/raw'

if __name__ == "__main__":
    logging.basicConfig(filename='test_denmark.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from water_datasets import CAMELS_DK

from utils import (
    test_dataset,
    test_dynamic_data,
    test_attributes
    )

dataset = CAMELS_DK(path=os.path.join(gscad_path, 'CAMELS'), verbosity=3)
test_dataset(dataset, 304, 12782, 119, 13)
