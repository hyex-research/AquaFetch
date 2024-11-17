
import os
import site
wd_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(wd_dir)

import logging

if __name__ == "__main__":
    logging.basicConfig(filename='test_estreams.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import pandas as pd
from water_datasets._backend import xarray as xr

from water_datasets.rr import EStreams

from utils import (
    test_coords, 
    test_stations, 
    test_area, 
    test_boundary, 
    test_plot_stations,
    test_fetch_static_feature,
    test_fetch_dynamic_features
    )

gscad_path = '/mnt/datawaha/hyex/atr/gscad_database/raw'

ds = EStreams(path=gscad_path, verbosity=3)

assert ds.md.shape == (15047, 24), ds.md.shape

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
    out = ds.fetch_static_features(countries='IE')
    assert out.shape == (464, 184), out.shape

    test_fetch_static_feature(ds, 'IEEP0281', 15047, 184)
    return


def test_meteo():
    out = ds.meteo_data_station('IEEP0281')

    assert isinstance(out, pd.DataFrame), type(out)

    assert out.shape[1] == 9, out.shape

    out = ds.meteo_data(countries='IE')
    assert len(out) == 464, len(out)

    out = ds.meteo_data()

    assert len(out) == 15047, len(out)

    return


def test_fetch_dynamic_features_():
    out = ds.fetch_dynamic_features(countries='IE')

    assert len(out) == 464, len(out)

    test_fetch_dynamic_features(ds, 'IEEP0281')

    return


test_coords_()

test_area_()

test_fetch_static_features()

test_meteo()

test_fetch_dynamic_features_()

test_boundary(ds)

test_plot_stations(ds)

test_stations(ds, 15047)