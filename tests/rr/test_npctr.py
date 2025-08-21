
import os
import site   # so that AquaFetch directory is in path
import logging

# add the parent directory in the path
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

if __name__ == "__main__":
    logging.basicConfig(filename='test_npctr.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from utils import test_dataset

import pandas as pd

from aqua_fetch.rr import NPCTRCatchments


ds = NPCTRCatchments(path='/mnt/datawaha/hyex/atr/data', verbosity=4)

test_dataset(ds, 7, 53072, 14, 14, st="20140101", en="20141231")


sh_q = ds.read_5min_q()

for k,v in sh_q.items():
    assert k in ds.stations()
    assert isinstance(v, pd.DataFrame)
    assert isinstance(v.index, pd.DatetimeIndex)


ds = NPCTRCatchments(path='/mnt/datawaha/hyex/atr/data', timestep='5min')

pcp_5m = ds.read_pcp()
rh_5m = ds.read_rel_hum()
#temp_5m = ds.read_temp()
solrad_5m = ds.read_sol_rad()
ws_5m = ds.read_wind_speed()
snowdepth_5m = ds.read_snow_depth()
winddir_5m = ds.read_wind_dir()


