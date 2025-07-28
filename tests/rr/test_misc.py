
import os
import site
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

import logging

if __name__ == "__main__":
    logging.basicConfig(filename='test_estreams.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from aqua_fetch.rr import DraixBleone
from aqua_fetch.rr import JialingRiverChina

from utils import test_fetch_dynamic_features


def test_jilingriverchina():
    dataset = JialingRiverChina()

    assert len(dataset.stations()) == 11

    for stn in dataset.stations():
        df = dataset._read_stn_dyn(stn)
        assert len(df.shape) == 2

    assert len(dataset.dynamic_features) == 43

    test_fetch_dynamic_features(dataset, dataset.stations()[0], as_dataframe=True)

    return


ds = DraixBleone(path='/mnt/datawaha/hyex/atr/data', verbosity=4)

assert len(ds.stations()) == 4

assert len(ds.static_features) == 5

for stn in ds.stations():
    df = ds._read_stn_dyn(stn)
    assert len(df.shape) == 2

# df = ds._read_q_stn(ds.stations()[0])