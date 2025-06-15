
import os
import site
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

import logging

if __name__ == "__main__":
    logging.basicConfig(filename='test_gsha.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import pandas as pd
from aqua_fetch._backend import xarray as xr

from aqua_fetch import Thailand
from aqua_fetch import GSHA
from aqua_fetch import Japan
from aqua_fetch import Arcticnet
from aqua_fetch import Spain
from utils import (
    test_dataset,
    test_coords, 
    test_stations, 
    test_area, 
    test_boundary, 
    test_plot_stations,
    test_fetch_static_feature,
    test_fetch_dynamic_features
    )

gscad_path = '/mnt/datawaha/hyex/atr/gscad_database/raw'

ds = GSHA(path=gscad_path, verbosity=3)

assert ds.wsAll.shape == (21568, 4), ds.wsAll.shape

assert len(ds.agencies) == 13, len(ds.agencies)


def test_atlas():
    assert ds.atlas().shape == (21568, 24), ds.atlas().shape

    assert ds.atlas(agency='arcticnet').shape == (106, 24), ds.atlas(agency='arcticnet').shape

    assert ds.atlas(stations='1001_arcticnet').shape == (1, 24), ds.atlas(stations='1001_arcticnet').shape

    return


def test_uncertainty():
    
    assert ds.uncertainty().shape == (21568, 7), ds.uncertainty().shape

    assert ds.uncertainty(agency='arcticnet').shape == (106, 7), ds.uncertainty(agency='arcticnet').shape

    assert ds.uncertainty(stations='1001_arcticnet').shape == (1, 7), ds.uncertainty(stations='1001_arcticnet').shape

    return


def test_area_():

    test_area(ds)

    assert ds.area(agency='arcticnet').shape == (106, ), ds.area(agency='arcticnet').shape

    assert ds.area(stations='1001_arcticnet').shape == (1, ), ds.area(stations='1001_arcticnet').shape

    return


def test_coords_():

    test_coords(ds)

    assert ds.stn_coords(agency='arcticnet').shape == (106, 2), ds.stn_coords(agency='arcticnet').shape

    assert ds.stn_coords(stations='1001_arcticnet').shape == (1, 2), ds.stn_coords(stations='1001_arcticnet').shape

    return


def test_lc_vars():

    assert ds.lc_variables_stn('1001_arcticnet').shape[1] == 3

    if xr is not None:
        out = ds.lc_variables()

        # get number of variables in out 
        assert len(out) == 21568, len(out)

        assert out['1001_arcticnet'].shape[1] == 3

        out = ds.lc_variables(agency='arcticnet')

        assert len(out) == 106, len(out)

        assert out['1001_arcticnet'].shape[1] == 3
    
    return


def test_streamflow_indices():
    ds1 = GSHA(path=gscad_path, to_netcdf=False, verbosity=3)
    assert ds1.streamflow_indices_stn('1001_arcticnet').shape[1] == 16
    out = ds1.streamflow_indices()
    assert isinstance(out, dict), type(out)
    assert len(out) == 21568, len(out)
    assert out['1001_arcticnet'].shape[1] == 16

    out = ds1.streamflow_indices(agency='arcticnet')

    assert isinstance(out, dict), type(out)
    assert len(out) == 106, len(out)
    assert out['1001_arcticnet'].shape[1] == 16
    return


def test_lai():

    assert isinstance(ds.lai_stn('1001_arcticnet'), pd.Series)

    lai = ds.fetch_lai()
    assert len(lai) == 21568, len(lai)

    lai.dims['time'] == 14541

    out = ds.fetch_lai('1001_arcticnet')
    assert len(out) == 1, len(out)

    out = ds.fetch_lai(agency='arcticnet')

    assert len(out) == 106, len(out)

    return


def test_storage():
    out = ds.storage_vars_stn('1001_arcticnet')
    assert isinstance(out, pd.DataFrame), type(out)
    assert out.shape[1] == 6, out.shape

    out = ds.fetch_storage_vars(agency='arcticnet')

    assert len(out) == 106, len(out)

    out['1001_arcticnet'].shape[1] == 6

    out = ds.fetch_storage_vars()

    assert len(out) == 21568, len(out)

    return


def test_meteo():
    out = ds.meteo_vars_stn('1001_arcticnet')

    assert isinstance(out, pd.DataFrame), type(out)

    assert out.shape[1] == 19, out.shape

    out = ds.fetch_meteo_vars(agency='arcticnet')
    assert len(out) == 106, len(out)

    out = ds.fetch_meteo_vars()

    assert len(out) == 21568, len(out)

    return


def test_reservoir():
    out = ds.reservoir_variables_stn('1001_arcticnet')

    assert isinstance(out, pd.DataFrame), type(out)

    assert out.shape[1] == 2, out.shape

    out = ds.reservoir_variables(agency='arcticnet')

    assert len(out) == 106, len(out)

    assert out['1001_arcticnet'].shape[1] == 2, out['1001_arcticnet'].shape

    out = ds.reservoir_variables()

    assert len(out) == 21568, len(out)

    return


def test_fetch_static_features():
    out = ds.fetch_static_features(agency='arcticnet')
    assert out.shape == (106, 35), out.shape

    test_fetch_static_feature(ds, '1001_arcticnet', 21568, 35)
    return


def test_fetch_dynamic_features_():

    out = ds.fetch_dynamic_features(agency='arcticnet')

    assert len(out) == 106, len(out)

    test_fetch_dynamic_features(ds, '1001_arcticnet')

    return


def test_stn_dynamic_features():
    data = ds.fetch_stn_dynamic_features('1001_arcticnet')
    data.shape == (16071, 26)
    assert len(ds.dynamic_features) == 26
    data = ds.fetch_stn_dynamic_features('1001_arcticnet',
    dynamic_features=['airtemp_C_mean_era5', 'pcp_mm_mswep'])
    data.shape == (16071, 2)
    return


test_stations(ds, 21568)

test_boundary(ds)

test_plot_stations(ds)

test_atlas()

test_uncertainty()

test_area_()

test_coords_()

test_lc_vars()

test_streamflow_indices()

test_lai()

test_storage()

test_meteo()

test_fetch_static_features()

test_fetch_dynamic_features_()

test_stn_dynamic_features()

print('All tests passed!')


ds = Thailand(path=gscad_path, verbosity=3)

test_dataset(ds, 
             num_stations=73, 
             dyn_data_len=7305, 
             num_static_attrs=35,
              num_dyn_attrs=27,
              test_df=False,
              st="1992-01-01",
              en="1992-12-31",
              )


ds = Japan(path=gscad_path, verbosity=3)

test_dataset(ds, 
             num_stations=751, 
             dyn_data_len=16071, 
             num_static_attrs=35,
              num_dyn_attrs=27,
              test_df=False,
              )


ds = Arcticnet(path=gscad_path, verbosity=3)

test_dataset(ds, 
             num_stations=106, 
             dyn_data_len=9131, 
             num_static_attrs=35,
              num_dyn_attrs=27,
              test_df=False,
              st="1992-01-01",
              en="1992-12-31",
              )


ds = Spain(path=gscad_path, verbosity=3)

test_dataset(ds,
             num_stations=889,
             dyn_data_len=15249,
             num_static_attrs=35,
              num_dyn_attrs=27,
              test_df=False,
              st="1992-01-01",
              en="1992-12-31",
              )
