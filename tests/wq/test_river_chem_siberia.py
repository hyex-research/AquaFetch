
import os
import site   # so that aqua_fetch directory is in path
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

from aqua_fetch import RiverChemSiberia


ds = RiverChemSiberia(path='/mnt/datawaha/hyex/atr/data')

assert len(ds.stations()) == 7

assert len(ds.parameters) == 34

coords = ds.stn_coords()
assert coords.shape == (7, 2)

