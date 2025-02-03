
import os
import site   # so that aqua_fetch directory is in path
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

from aqua_fetch import SyltRoads


ds = SyltRoads(
    path='/mnt/datawaha/hyex/atr/data',
    verbosity=4
    )

df = ds.fetch()
assert df.shape == df.shape
assert len(ds.parameters) == 16
assert ds.fetch(['Sal', 'Temp [°C]', 'pH']).shape == (5710, 3)

