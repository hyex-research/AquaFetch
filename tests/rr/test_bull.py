
import os
import site   # so that aqua_fetch directory is in path
import logging

# add the parent directory in the path
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

raw_data_path = '/path/to/raw/data'  # replace with actual path

if __name__ == "__main__":
    logging.basicConfig(filename='test_bull.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from aqua_fetch import Bull

from utils import test_dataset


dataset = Bull(path=raw_data_path, verbosity=3)
test_dataset(dataset, 484, 25932, 214, 55)
