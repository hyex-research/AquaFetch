
import math
import os
import site
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

import logging

if __name__ == "__main__":
    logging.basicConfig(filename='test_estreams.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ignore all warnings
import warnings
warnings.filterwarnings("ignore")

import pandas as pd

from aqua_fetch import EStreams
from aqua_fetch import Ireland
from aqua_fetch import Finland
from aqua_fetch import Italy
from aqua_fetch import Poland
from aqua_fetch import Portugal
from aqua_fetch import Slovenia

from utils import (
    test_dataset,
    test_coords,
    test_stations,
    test_area,
    test_boundary,
    test_plot_stations,
    test_fetch_static_feature,
    test_fetch_dynamic_features,
    test_plot_catchment,
    )

gscad_path = '/mnt/datawaha/hyex/atr/gscad_database/raw'

ds = EStreams(path=gscad_path, verbosity=3)

assert ds.md.shape == (17130, 29), ds.md.shape

assert len(ds.countries) == 39, len(ds.agencies)

assert len(ds.country_stations('IT')) == 767, len(ds.country_stations('IT'))

assert len(ds.country_stations('ES')) == 1440, len(ds.country_stations('ES'))

assert len(ds.country_stations('IE')) == 464, len(ds.country_stations('IE'))

assert len(ds.country_stations('PL')) == 1287, len(ds.country_stations('PL'))


def test_coords_():

    test_coords(ds)

    assert ds.stn_coords(countries='IE').shape == (464, 2), ds.stn_coords(countries='IE').shape

    assert ds.stn_coords(stations='IEEP0281').shape == (1, 2), ds.stn_coords(stations='IEEP0281').shape

    return


def test_area_():

    test_area(ds)

    assert ds.area(countries='IE').shape == (464, ), ds.area(countries='IE').shape

    assert ds.area(stations='IEEP0281').shape == (1, ), ds.area(stations='IEEP0281').shape

    return


def test_fetch_static_features():
    # out = ds.fetch_static_features(countries='IE')  # todo : why countries argument is not working
    # assert out.shape == (464, 214), out.shape

    test_fetch_static_feature(ds, 'IEEP0281', 17130, 214)
    return


def test_meteo():
    out = ds.meteo_data_station('IEEP0281')

    assert isinstance(out, pd.DataFrame), type(out)

    assert out.shape[1] == 9, out.shape

    out = ds.meteo_data(countries='IE')
    assert len(out) == 464, len(out)

    out = ds.meteo_data()

    assert len(out) == 17130, len(out)

    return


def test_fetch_dynamic_features_():
    out = ds.fetch_dynamic_features(countries='IE')

    assert len(out) == 464, len(out)

    test_fetch_dynamic_features(ds, 'IEEP0281', 17130)

    return


test_coords_()

test_area_()

test_fetch_static_features()

test_meteo()

test_fetch_dynamic_features_()

test_boundary(ds)

test_plot_stations(ds)

test_stations(ds, 17130)

test_plot_catchment(ds)

print('All tests passed!')


ds = Ireland(path=gscad_path, verbosity=3)

ds.basin_id_gauge_id_map()['IEOP0126'] == '20002'
ds.gauge_id_basin_id_map()['20002'] == 'IEOP0126'

test_dataset(ds, 
             num_stations=464, 
             dyn_data_len=26844, 
             num_static_attrs=214,
              num_dyn_attrs=10
              )

_, dynamic = ds.fetch('all', dynamic_features='q_cms_obs', as_dataframe=True)
pd.concat(list(dynamic.values()), axis=1).count().sum() >= 3303345


ds = Finland(path=gscad_path, processes=1, 
             verbosity=3)

test_dataset(ds, 
             num_stations=669, 
             dyn_data_len=4199, 
             num_static_attrs=214,
              num_dyn_attrs=10,
              st="20200101",
              en="20201231"
              )
_, dynamic = ds.fetch('all', dynamic_features='q_cms_obs', as_dataframe=True)
pd.concat(list(dynamic.values()), axis=1).count().sum() >= 814277

ds = Italy(path=gscad_path, verbosity=3)

test_dataset(ds, 
             num_stations=294, 
             dyn_data_len=26844, 
             num_static_attrs=214,
              num_dyn_attrs=10
              )



ds = Poland(path=gscad_path, verbosity=3)

test_dataset(ds, 
             num_stations=1287, 
             dyn_data_len=26844, 
             num_static_attrs=214,
              num_dyn_attrs=10
              )

_, dynamic = ds.fetch('all', dynamic_features='q_cms_obs', as_dataframe=True)
pd.concat(list(dynamic.values()), axis=1).count().sum() >= 16319627

q = ds.get_q()

assert q.shape[1]>1287

ds = Portugal(path=gscad_path, verbosity=3)

q = ds.get_q()

assert q.shape == (18628, 280)

assert isinstance(q.index, pd.DatetimeIndex)

# find columns in q dataframe which are all NaN
nan_cols = q.columns[q.isna().all()]

test_dataset(ds, 
             num_stations=280, 
             dyn_data_len=18628, 
             num_static_attrs=214,
              num_dyn_attrs=10
              )


## Slovenia

ds = Slovenia(path=gscad_path, verbosity=3)

q = ds.get_q()

assert q.shape == (27028, 117)

assert len(q.columns[q.isna().all()]) == 0, len(q.columns[q.isna().all()])

test_dataset(ds, 
             num_stations=117, 
             dyn_data_len=27028, 
             num_static_attrs=214,
              num_dyn_attrs=10
              )
