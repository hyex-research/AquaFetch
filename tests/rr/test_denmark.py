
import os
import site   # so that aqua_fetch directory is in path
import logging

# add the parent directory in the path
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

gscad_path = '/mnt/datawaha/hyex/atr/gscad_database/raw'

if __name__ == "__main__":
    logging.basicConfig(filename='test_denmark.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from aqua_fetch import CAMELS_DK, Caravan_DK

from utils import test_dataset

dataset = CAMELS_DK(path=os.path.join(gscad_path, 'CAMELS'), verbosity=3)
test_dataset(dataset, 304, 12782, 119, 13)


ds_dk = Caravan_DK(path= gscad_path)
test_dataset(ds_dk, 308, 14609, 211, 39)
