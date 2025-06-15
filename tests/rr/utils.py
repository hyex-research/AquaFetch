import os
import logging

import random

import numpy as np
import pandas as pd
from aqua_fetch._backend import xarray as xr

from aqua_fetch._backend import plt, netCDF4, shapefile

logger = logging.getLogger(__name__)


def test_stations(dataset, stations_len):
    logger.info(f"test_stations for {dataset.name}")
    stations = dataset.stations()
    assert len(stations) == stations_len, f'number of stations for {dataset.name} are {len(stations)}'

    for stn in stations:
        assert isinstance(stn, str)

    assert all([isinstance(i, str) for i in stations])
    return


def test_coords(dataset):
    logger.info(f"testing coords for {dataset.name}")

    stations = dataset.stations()
    df = dataset.stn_coords()  # returns coordinates of all stations
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(stations)
    assert 'lat' in df and 'long' in df
    df = dataset.stn_coords(stations[0])  # returns coordinates of station
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1, len(df)
    assert 'lat' in df and 'long' in df
    df = dataset.stn_coords(stations[0:2])  # returns coordinates of two stations
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert 'lat' in df and 'long' in df
    return


def test_plot_stations(dataset):
    logger.info(f"testing plot_stations for {dataset.name}")

    stations = dataset.stations()
    dataset.plot_stations(show=False)
    plt.close()
    dataset.plot_stations(stations[0:3], show=False)
    plt.close()
    dataset.plot_stations(marker='o', ms=0.3, show=False)
    plt.close()
    ax = dataset.plot_stations(marker='o', ms=0.3, show=False)
    ax.set_title("Stations")
    assert isinstance(ax, plt.Axes)
    plt.close()
    return


def test_area(dataset):
    logger.info(f"testing area for {dataset.name}")

    stations = dataset.stations()
    s = dataset.area()  # returns area of all stations
    assert isinstance(s, pd.Series)
    assert len(s) == len(stations)
    assert s.name == "area_km2", s.name
    s = dataset.area(stations[0])  # returns area of station
    assert isinstance(s, pd.Series)
    assert len(s) == 1, len(s)
    assert s.name == "area_km2"
    s = dataset.area(stations[0:2])  # returns area of two stations
    assert isinstance(s, pd.Series)
    assert len(s) == 2
    assert s.name == "area_km2"
    return


def test_q_mmd(dataset):
    logger.info(f"testing q_mmd for {dataset.name}")
    stations = dataset.stations()

    df = dataset.q_mmd(stations[0])  # returns q of station
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == 1, df.shape

    df = dataset.q_mmd(stations[0:2])  # returns q of two stations
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == 2, df.shape

    return


def test_boundary(dataset):
    logger.info(f"testing get_boundary for {dataset.name}")

    boundary = dataset.get_boundary(dataset.stations()[0])

    assert isinstance(boundary, np.ndarray)

    return


def test_fetch_static_feature(dataset, station, num_stations, num_static_features):
    logger.info(f"testing fetch_static_features method for {dataset.name}")
    if len(dataset.static_features) > 0:
        df, _ = dataset.fetch(station, dynamic_features=None, static_features='all')
        assert isinstance(df, pd.DataFrame)
        assert len(df.loc[station, :]) == len(dataset.static_features), f'shape is: {df.loc[station].shape}'

        df = dataset.fetch_static_features(station, static_features='all')

        assert isinstance(df,
                          pd.DataFrame), f'fetch_static_features for {dataset.name} returned of type {df.__class__.__name__}'
        assert len(df.loc[station, :]) == len(dataset.static_features), f'shape is: {df.loc[station].shape}'

        df = dataset.fetch_static_features("all", static_features='all')

        assert_dataframe(df, dataset)

        assert df.shape == (
        num_stations, num_static_features), f"{df.shape} Expected {(num_stations, num_static_features)}"
    return


def assert_dataframe(df, dataset):
    assert isinstance(df, pd.DataFrame), f"""
    fetch_static_features for {dataset.name} returned of type {df.__class__.__name__}"""
    return


def test_fetch_dynamic_features(dataset, station, as_dataframe=False):
    logger.info(f"test_fetch_dynamic_features for {dataset.name} and {station} stations")
    df = dataset.fetch_dynamic_features(station, as_dataframe=as_dataframe)
    if as_dataframe:
        assert df.unstack().shape[1] == len(
            dataset.dynamic_features), f'for {dataset.name}, num_dyn_attributes are {df.shape[1]}'
    else:
        assert isinstance(df, xr.Dataset), f'data is of type {df.__class__.__name__}'
        assert len(df.data_vars) == 1, f'{len(df.data_vars)}'

    dataset.fetch_dynamic_features(station, dynamic_features=dataset.dynamic_features[0])

    logger.info(f"Finished test_fetch_dynamic_features for {dataset.name} and {station} stations")
    return


def test_dynamic_data(dataset, stations, num_stations, stn_data_len,
                      as_dataframe=False, raise_len_error=True):
    logger.info(f"test_dynamic_data for {dataset.name}")

    if stations == 'all' and len(dataset.stations()) > 500:
        if dataset.timestep == 'D':
            # randomly select 500 stations
            stations = random.sample(dataset.stations(), 500)
        else:
            stations = random.sample(dataset.stations(), 20)

        logger.info(f"randomly selected {len(stations)} stations for {dataset.name}")
        num_stations = len(stations)

    _, df = dataset.fetch(stations=stations, static_features=None, as_dataframe=as_dataframe)

    logger.info(f"fetched data for {stations} stations for {dataset.name}")

    if as_dataframe:
        check_dataframe(dataset, df, num_stations, stn_data_len, raise_len_error=raise_len_error)
    else:
        check_dataset(dataset, df, num_stations, stn_data_len, raise_len_error=raise_len_error)

    return


def test_selected_dynamic_features(dataset, as_dataframe=False):
    logger.info(f"test_selected_dynamic_features for {dataset.name}")
    features = dataset.dynamic_features[0:2]
    _, data = dataset.fetch(dataset.stations()[0], dynamic_features=features, as_dataframe=as_dataframe)

    if as_dataframe:
        data = data.unstack()
        assert data.shape[1] == 2, f"data.shape is {data.shape}, expected 2 dynamic features"
    else:
        assert len(data.dynamic_features) == 2, len(data.dynamic_features)

    # checking for multiple stations
    features = dataset.dynamic_features[0:2]
    _, data = dataset.fetch(dataset.stations()[0:3], dynamic_features=features, as_dataframe=as_dataframe)

    if as_dataframe:
        data = data.iloc[:, 0].unstack()
        assert data.shape[1] == 2, data.shape
    else:
        assert len(data.dynamic_features) == 2, len(data.dynamic_features)
    return


def test_fetch_station_features(dataset, num_static_attrs, num_dyn_attrs, dyn_length):
    logger.info(f"testing fetch_station_features for {dataset.name}")

    station = random.choice(dataset.stations())

    static, dynamic = dataset.fetch_station_features(station)

    assert static.shape == (1, num_static_attrs), f"shape is {static.shape}"
    assert len(dynamic.columns) == num_dyn_attrs, f"num_dyn_attrs is {len(dynamic.data_vars)}"
    assert len(dynamic) == dyn_length, f"length is {len(dynamic)}"

    # test for single static feature

    # test for single dynamic feature

    return

def test_all_data(dataset, stations, stn_data_len, as_dataframe=False,
                  raise_len_error=True):
    if as_dataframe:
        logger.info(f"test_all_data for {dataset.name} with as_dataframe=True")
    else:
        logger.info(f"test_all_data for {dataset.name}")

    if len(dataset.static_features) > 0:
        static, dynamic = dataset.fetch(stations, static_features='all', as_dataframe=as_dataframe)
        assert static.shape == (stations, len(dataset.static_features)), f"shape is {static.shape}"
    else:
        _, dynamic = dataset.fetch(stations, static_features=None, as_dataframe=as_dataframe)

    if as_dataframe:
        check_dataframe(dataset, dynamic, stations, stn_data_len, raise_len_error=raise_len_error)
    else:
        check_dataset(dataset, dynamic, stations, stn_data_len, raise_len_error=raise_len_error)

    return


def check_dataset(dataset, xds, num_stations, data_len,
                  raise_len_error=True):
    assert isinstance(xds, xr.Dataset), f'xds is of type {xds.__class__.__name__}'
    assert len(xds.data_vars) == num_stations, f'for {dataset.name}, {len(xds.data_vars)} data_vars are present'
    for var in xds.data_vars:
        msg = f"""shape of data is {xds[var].data.shape} and not {data_len, len(dataset.dynamic_features)}"""
        if raise_len_error:
            assert xds[var].data.shape == (data_len, len(dataset.dynamic_features)), msg
        else:
            logger.warning(msg)

    for dyn_attr in xds.coords['dynamic_features'].data:
        assert dyn_attr in dataset.dynamic_features, f'{dyn_attr} not in dataset.dynamic_features'
    return


def check_dataframe(
        dataset,
        df: pd.DataFrame,
        num_stations: int,
        data_len: int,
        raise_len_error=True
):
    logger.info(f"checking sanity of dataframe of shape {df.shape}")
    assert isinstance(df, pd.DataFrame)
    assert df.shape[1] == num_stations, f'dataset lenth is {df.shape[1]} while target is {num_stations}'
    for col in df.columns:
        #     for dyn_attr in dataset.dynamic_features:
        #         stn_data = df[col]  # (stn_data_len*dynamic_features, )
        #         _stn_data_len = len(stn_data.iloc[stn_data.index.get_level_values('dynamic_features') == dyn_attr])
        #         assert _stn_data_len>=stn_data_len, f"{col} for {dataset.name} is not of length {stn_data_len}"
        stn_data = df[col].unstack()
        desired_shape = (data_len, len(dataset.dynamic_features))
        # data for each station must minimum be of this shape
        msg = f"""for {col} station of {dataset.name} the shape is {stn_data.shape} and not {desired_shape}"""
        if raise_len_error:
            assert stn_data.shape == desired_shape, msg
        else:
            logger.warning(msg)

    logger.info(f"Finished checking sanity of dataframe of shape {df.shape}")
    return


def test_static_data(dataset, stations, target):
    if stations == 'all':
        logger.info(f"test_static_data for {dataset.name} for all stations expected {target}")
    else:
        logger.info(f"test_static_data for {dataset.name} for {stations} stations expected {target}")

    if len(dataset.static_features) > 0:
        df, _ = dataset.fetch(stations=stations, dynamic_features=None, static_features='all')
        assert isinstance(df, pd.DataFrame)
        assert len(df) == target, f'length of static df is {len(df)} Expected {target}'
        exp_shape = (target, len(dataset.static_features))
        assert df.shape == exp_shape, f'for {dataset.name}, actual shape {df.shape} and exp shape {exp_shape}'

    return


def test_attributes(dataset, static_attr_len, dyn_attr_len, stations):
    logger.info(f"test_attributes for {dataset.name}")
    static_features = dataset.static_features
    assert len(
        set(static_features)) == static_attr_len, f'for {dataset.name} static_features are {len(static_features)} and not {static_attr_len}'
    assert isinstance(static_features, list)
    assert all([isinstance(i, str) for i in static_features])

    assert os.path.exists(dataset.path)

    dynamic_features = dataset.dynamic_features
    assert len(
        dynamic_features) == dyn_attr_len, f'Obtained dynamic attributes: {len(dynamic_features)} Expected: {dyn_attr_len}'
    assert isinstance(dynamic_features, list)
    assert all([isinstance(i, str) for i in dynamic_features])

    test_stations(dataset, stations)

    return


def test_fetch_dynamic_multiple_stations(dataset, n_stns, stn_data_len, as_dataframe=False):
    logger.info(f"testing fetch_dynamic_multiple_stations for {dataset.name} for {n_stns} stations")
    stations = dataset.stations()
    _, data = dataset.fetch(stations[0:n_stns], as_dataframe=as_dataframe)

    if as_dataframe:
        check_dataframe(dataset, data, n_stns, stn_data_len)
    else:
        check_dataset(dataset, data, n_stns, stn_data_len)

    return


def test_st_en_with_static_and_dynamic(
        dataset, station,
        as_dataframe=False,
        yearly_steps=366,
        st='19880101',
        en='19881231',
):
    logger.info(f"testing {dataset.name} with st and en with both static and dynamic")

    if len(dataset.static_features) > 0:
        static, dynamic = dataset.fetch([station], static_features='all',
                             st=st,
                             en=en, as_dataframe=as_dataframe)
        if as_dataframe:
            check_dataframe(dataset, dynamic, 1, yearly_steps)
        else:
            check_dataset(dataset, dynamic, 1, yearly_steps)

        assert static.shape == (1, len(dataset.static_features))

        data = dataset.fetch_dynamic_features(station, st=st, en=en,
                                              as_dataframe=as_dataframe)
        if as_dataframe:
            check_dataframe(dataset, data, 1, yearly_steps)
        else:
            check_dataset(dataset, data, 1, yearly_steps)
    return


def test_dataset(dataset, num_stations, dyn_data_len, num_static_attrs, num_dyn_attrs,
                 test_df=True, yearly_steps=366,
                 raise_len_error=True,
                 st="20040101", en="20041231",
                 has_q: bool = True,
                 ):
    
    # if netCDF4 is not None:
    #     # check that dynamic attribues from all data can be retrieved.
    #     test_dynamic_data(dataset, 'all', num_stations, dyn_data_len)
    # if test_df:
    #     test_dynamic_data(dataset, 'all', num_stations, dyn_data_len, as_dataframe=True)

    # if netCDF4 is not None:
    #     # check that dynamic data of 10% of stations can be retrieved
    #     test_dynamic_data(dataset, 0.1, int(num_stations * 0.1), dyn_data_len,
    #                   raise_len_error=raise_len_error)
    # if test_df:
    #     test_dynamic_data(dataset, 0.1, int(num_stations * 0.1), dyn_data_len, True,
    #                       raise_len_error=raise_len_error)

    # test_static_data(dataset, 'all', num_stations)  # check that static data of all stations can be retrieved

    # test_static_data(dataset, 0.1,
    #                  int(num_stations * 0.1))  # check that static data of 10% of stations can be retrieved

    # if netCDF4 is not None:
    #     test_all_data(dataset, 3, dyn_data_len, raise_len_error=raise_len_error)

    if test_df:
        test_all_data(dataset, 3, dyn_data_len, True, raise_len_error=raise_len_error)

    # check length of static attribute categories
    test_attributes(dataset, num_static_attrs, num_dyn_attrs, num_stations)

    if netCDF4 is not None:
        # make sure dynamic data from one station have num_dyn_attrs attributes
        test_fetch_dynamic_features(dataset, random.choice(dataset.stations()))
    if test_df:
        test_fetch_dynamic_features(dataset, random.choice(dataset.stations()), True)

    if netCDF4 is not None:
        # make sure that dynamic data from 3 stations each have correct length/shape
        test_fetch_dynamic_multiple_stations(dataset, 3, dyn_data_len)
    if test_df:
        test_fetch_dynamic_multiple_stations(dataset, 3, dyn_data_len, True)

    # make sure that static data from one station can be retrieved
    test_fetch_static_feature(dataset, random.choice(dataset.stations()),
                              num_stations, num_static_attrs)
    if netCDF4 is not None:
        test_st_en_with_static_and_dynamic(dataset, random.choice(dataset.stations()),
                                       yearly_steps=yearly_steps,
                                       st=st, en=en)
    if test_df:
        test_st_en_with_static_and_dynamic(dataset, random.choice(dataset.stations()), True,
                                           yearly_steps=yearly_steps,
                                           st=st, en=en)

    # test that selected dynamic features can be retrieved successfully
    test_selected_dynamic_features(dataset, as_dataframe=test_df)

    # test_fetch_station_features(dataset, num_static_attrs, num_dyn_attrs, dyn_data_len)

    test_coords(dataset)

    if plt is not None:
        test_plot_stations(dataset)

    test_area(dataset)

    if has_q:
        test_q_mmd(dataset)

    if shapefile is not None:
        test_boundary(dataset)

    logger.info(f"** Finished testing {dataset.name} **")

    return
