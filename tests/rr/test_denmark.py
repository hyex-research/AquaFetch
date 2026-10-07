
import os
import site   # so that aqua_fetch directory is in path
import logging

# add the parent directory in the path
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)


if __name__ == "__main__":
    logging.basicConfig(filename='test_denmark.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from aqua_fetch import CAMELS_DK, Caravan_DK

from utils import test_dataset as run_shared_tests
from utils import test_boundaries_around_gauges as check_boundaries_around_gauges


gscad_path = ''

dataset = CAMELS_DK(path=os.path.join(gscad_path, 'CAMELS'), verbosity=3)
run_shared_tests(dataset, 304, 12782, 119, 13)
# reprojected from EPSG:25832, including the 29 catchments with holes
check_boundaries_around_gauges(dataset)
# gauges and boundaries share their projection parameters, so check one gauge
# against pyproj (EPSG:25832 -> EPSG:4326 of 559057.7232 E, 6131379.493 N)
lat, lon = dataset.stn_coords('54130033').iloc[0]
assert abs(lat - 55.3252417540135) < 1e-5 and abs(lon - 9.930789450196023) < 1e-5, (lat, lon)


ds_dk = Caravan_DK(path= gscad_path)
run_shared_tests(ds_dk, 308, 14609, 211, 39)
