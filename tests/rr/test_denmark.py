
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

from aqua_fetch import CAMELS_DK, Caravan_DK

from utils import (
    test_dataset,
    test_dynamic_data,
    test_attributes
    )

dataset = CAMELS_DK(path=os.path.join(gscad_path, 'CAMELS'), verbosity=3)
test_dataset(dataset, 304, 12782, 119, 13)


ds_dk = Caravan_DK(path=os.path.join(gscad_path, 'CAMELS'))
test_dataset(ds_dk, 308, 14609, 211, 39)


def test_camels_dk_docs():

    dataset = Caravan_DK(path= os.path.join(gscad_path, 'CAMELS'))

    assert len(dataset.stations()) == 308
    assert dataset.fetch_static_features(dataset.stations()).shape == (308, 211)
    assert dataset.fetch_static_features('80001').shape == (1, 211)
    assert dataset.fetch_static_features(features=['gauge_lat', 'area']).shape == (308, 2)
    assert dataset.fetch_static_features('80001', features=['gauge_lat', 'area']).shape == (1, 2)

    df = dataset.fetch(stations=0.1, as_dataframe=True)
    assert df.index.names == ['time', 'dynamic_features']
    df = dataset.fetch(stations=1, as_dataframe=True)
    assert df.unstack().shape == (14609, 39)
    assert dataset.fetch(stations='80001', as_dataframe=True).unstack().shape == (14609, 39)

    df = dataset.fetch(1, as_dataframe=True,
                        dynamic_features=['snow_depth_water_equivalent_mean', 'temperature_2m_mean',
                                            'potential_evaporation_sum', 'total_precipitation_sum',
                                            'streamflow']).unstack()
    assert df.shape == (14609, 5)
    df = dataset.fetch(10, as_dataframe=True)
    assert df.shape == (569751, 10)

    data = dataset.fetch(stations='80001', static_features="all", as_dataframe=True)
    assert data['static'].shape == (1, 211)
    assert data['dynamic'].shape == (569751, 1)
    return