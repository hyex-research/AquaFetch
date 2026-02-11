
import os
import site   # so that aqua_fetch directory is in path
import logging

# add the parent directory in the path
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

raw_data_path = '/path/to/raw/data'

if __name__ == "__main__":
    logging.basicConfig(filename='test_grdccaravan.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from aqua_fetch import GRDCCaravan

from utils import test_dataset

dataset = GRDCCaravan(path=raw_data_path, verbosity=4)

test_dataset(dataset, 5356, 26800, 215, 41)
