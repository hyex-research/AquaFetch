
import logging

import numpy as np
import pandas as pd
from water_datasets._backend import xarray as xr

import matplotlib.pyplot as plt

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
    assert s.name == "area", s.name
    s = dataset.area(stations[0])  # returns area of station
    assert isinstance(s, pd.Series)
    assert len(s) == 1, len(s)
    assert s.name == "area"
    s = dataset.area(stations[0:2])  # returns area of two stations
    assert isinstance(s, pd.Series)
    assert len(s) == 2
    assert s.name == "area"
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


def test_fetch_static_feature(dataset, stn_id, num_stations, num_static_features):
    logger.info(f"testing fetch_static_features method for {dataset.name}")
    if len(dataset.static_features)>0:
        df = dataset.fetch(stn_id, dynamic_features=None, static_features='all')
        assert isinstance(df, pd.DataFrame)
        assert len(df.loc[stn_id, :]) == len(dataset.static_features), f'shape is: {df.loc[stn_id].shape}'

        df = dataset.fetch_static_features(stn_id, features='all')

        assert isinstance(df, pd.DataFrame), f'fetch_static_features for {dataset.name} returned of type {df.__class__.__name__}'
        assert len(df.loc[stn_id, :]) == len(dataset.static_features), f'shape is: {df.loc[stn_id].shape}'

        df = dataset.fetch_static_features("all", features='all')

        assert_dataframe(df, dataset)

        assert df.shape == (num_stations, num_static_features), f"{df.shape} Expected {(num_stations, num_static_features)}"
    return


def assert_dataframe(df, dataset):
    assert isinstance(df, pd.DataFrame), f"""
    fetch_static_features for {dataset.name} returned of type {df.__class__.__name__}"""
    return


def test_fetch_dynamic_features(dataset, stn_id, as_dataframe=False):
    logger.info(f"test_fetch_dynamic_features for {dataset.name} and {stn_id} stations")
    df = dataset.fetch_dynamic_features(stn_id, as_dataframe=as_dataframe)
    if as_dataframe:
        assert df.unstack().shape[1] == len(dataset.dynamic_features), f'for {dataset.name}, num_dyn_attributes are {df.shape[1]}'
    else:
        assert isinstance(df, xr.Dataset), f'data is of type {df.__class__.__name__}'
        assert len(df.data_vars) == 1, f'{len(df.data_vars)}'
    logger.info(f"Finished test_fetch_dynamic_features for {dataset.name} and {stn_id} stations")
    return