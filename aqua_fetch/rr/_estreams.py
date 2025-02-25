
__all__ = [
    "EStreams", 
    "_EStreams", 
    "Finland", 
    "Ireland", 
    "Italy", 
    "Poland",
    "Portugal",
    ]

import os
import time
import warnings
from io import StringIO
import concurrent.futures as cf
from urllib.error import HTTPError
from typing import Union, List, Dict
from concurrent.futures import ProcessPoolExecutor

import requests

import numpy as np
import pandas as pd

try:
    import xml.etree.ElementTree as ET
except (ModuleNotFoundError, ImportError):
    ET = None

from .._backend import xarray as xr
from ..utils import get_cpus
from ..utils import check_attributes
from .utils import _RainfallRunoff
 
 
from ._map import (
    catchment_area,
    gauge_latitude,
    gauge_longitude,
    slope
    )

from ._map import (
    min_air_pressure,
    total_precipitation,
    mean_potential_evapotranspiration,
    mean_rel_hum,
    mean_air_temp,
    mean_windspeed,
    max_air_temp,
    min_air_temp,
    solar_radiation,
    observed_streamflow_cms,
    )


class EStreams(_RainfallRunoff):
    """
    Handles EStreams data following the work of
    `Nascimento et al., 2024 <https://doi.org/10.1038/s41597-024-03706-1>`_ .
    The data is available at its `zenodo repository <https://zenodo.org/records/13961394>`_ .
    It should be noted that this dataset does not contain observed streamflow data.
    It has 17130 stations, 9 dynamic (meteorological) features with daily timestep, 27 dynamic
    features with yearly timestep and 214 static features. The dynamic features
    are from 1950-01-01 to 2023-06-30.

    Examples
    --------
    >>> from aqua_fetch import EStreams
    >>> dataset = EStreams()
    """

    url = "https://zenodo.org/records/13961394"

    def __init__(self, path=None, **kwargs):
        super().__init__(path, **kwargs)

        self._download()

        self.md = self.gauge_stations()
        self.md.rename(columns={'area_estreams': catchment_area()}, inplace=True)
        self._stations = self.__stations()
        self._dynamic_features = self.meteo_data_station('IEEP0281').columns.tolist()
        self._static_features = self._static_data().columns.tolist()

        self.boundary_file = os.path.join(self.path2,
                                          "shapefiles", "estreams_catchments.shp")
        self._create_boundary_id_map(self.boundary_file, 0)

    @property
    def path2(self):
        return os.path.join(self.path, 'EStreams', 'EStreams')

    @property
    def dynamic_features(self) -> List[str]:
        return self._dynamic_features

    @property
    def static_features(self):
        return self._static_features

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp('1950-01-01')

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp('2023-06-30')

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'area_estreams': catchment_area(),
                'lat': gauge_latitude(),
                'slope_sawicz': slope('no_unit'),
                'lon': gauge_longitude(),
        }
    
    @property
    def dyn_map(self):
        return {
            't_min': min_air_temp(),
            't_max': max_air_temp(),
            't_mean': mean_air_temp(),
            'p_mean': total_precipitation(),
            'pet_mean': mean_potential_evapotranspiration(),
            'rh_mean': mean_rel_hum(),
            'sp_min': min_air_pressure(),
            'swr_mean': solar_radiation(),
            'ws_mean': mean_windspeed()
        }

    def _static_data(self) -> pd.DataFrame:
        """
        Returns a dataframe with static attributes of catchments
        """
        static_path = os.path.join(self.path2, 'attributes', 'static_attributes')

        dfs = [self.hydro_clim_sigs(), self.md.copy()]
        for f in os.listdir(static_path):
            if f.endswith('.csv'):
                df = pd.read_csv(os.path.join(static_path, f), index_col='basin_id', dtype={'basin_id': str})
                dfs.append(df)

        df = pd.concat(dfs, axis=1)

        df.rename(columns=self.static_map, inplace=True)

        df.columns.name = 'static_features'
        df.index.name = 'station_id'
        return df

    def gauge_stations(self) -> pd.DataFrame:
        """
        reads the file estreams_gauging_stations.csv as dataframe
        """
        df = pd.read_csv(
            os.path.join(self.path2, 'streamflow_gauges', 'estreams_gauging_stations.csv'),
            index_col='basin_id',
            dtype={'basin_id': str}
        )
        return df

    def stations(self) -> List[str]:
        """
        Returns a list of all station names. Note that the `basin_id` column is 
        used as the station name.
        """
        return self._stations

    def __stations(self) -> List[str]:
        df = pd.read_csv(
            os.path.join(self.path2, 'streamflow_gauges', 'estreams_gauging_stations.csv'),
            usecols=['basin_id', 'lat'],
            dtype={'basin_id': str}
        )
        df.set_index('basin_id', inplace=True)
        return df.index.tolist()

    @property
    def countries(self) -> List[str]:
        """
        returns the names of 39 countries covered by EStreams as list
        """
        return self.md.loc[:, 'gauge_country'].unique().tolist()

    def country_of_stn(self, stn: str) -> str:
        """find the agency to which a station belongs """
        return self.md.loc[stn, 'gauge_country']

    def country_stations(self, country: str) -> List[str]:
        """returns the station ids from a particular country"""
        return self.md[self.md['gauge_country'] == country].index.tolist()

    def stn_coords(self, stations: List[str] = "all", countries: List[str] = "all") -> pd.DataFrame:
        """
        Returns the coordinates of one or more stations

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of shape (stations, 2)

        Examples
        --------
        >>> from aqua_fetch import EStreams
        >>> dataset = EStreams()
        >>> dataset.stn_coords('IEEP0281')
        >>> dataset.stn_coords(['IEEP0281', 'IEEP0282'])
        >>> dataset.stn_coords(countries='IE')
        """
        stations = self._get_stations(countries, stations)

        df = pd.read_csv(
            os.path.join(self.path2, 'streamflow_gauges', 'estreams_gauging_stations.csv'),
            usecols=['basin_id', 'lat', 'lon'],
            dtype={'basin_id': str}
        )
        df.set_index('basin_id', inplace=True)
        df.rename(columns={'lon': 'long'}, inplace=True)
        return df.loc[stations]

    def _get_stations(self, countries: List[str] = "all", stations: List[str] = "all") -> List[str]:
        if countries != "all" and stations != 'all':
            raise ValueError("Either provide countries or stations not both")

        if countries != "all":
            countries = check_attributes(countries, self.countries, 'countries')
            stations = self.md[self.md['gauge_country'].isin(countries)].index.tolist()
        else:
            stations = check_attributes(stations, self.stations(), 'stations')

        return stations

    def area(self, stations: List[str] = "all", countries: List[str] = "all") -> pd.Series:
        """area of catchments im km2"""

        stations = self._get_stations(countries, stations)
        return self.md.loc[stations, catchment_area()]

    def meteo_data_station(self, station: str) -> pd.DataFrame:
        """
        Returns the meteorological data of a station

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of meteorological data of shape (time, 9)
        """
        df = pd.read_csv(
            os.path.join(self.path2, 'meteorology', f'estreams_meteorology_{station}.csv'),
            index_col='date',
            parse_dates=True
        )
        df.columns.name = 'dynamic_features'
        df.index.name = 'time'

        df.rename(columns=self.dyn_map, inplace=True)

        return df

    def meteo_data(
            self,
            stations: Union[str, List[str]] = "all",
            countries: Union[List[str], str] = "all"
    ):
        """
        Returns the meteorological data of one or more stations
        either as dictionary of dataframes or xarray Dataset
        """
        stations = self._get_stations(countries, stations)
        out = self._metedo_data_all_stations()

        if isinstance(out, dict):
            return {stn: out[stn] for stn in stations}

        return out[stations]

    def _metedo_data_all_stations(self):
        """
        Returns the meteorological data of all stations
        """
        nc_path = os.path.join(self.path, 'EStreams', 'meteorology.nc')
        if self.to_netcdf and os.path.exists(nc_path):
            if self.verbosity > 1:
                print(f"Reading from {nc_path}")
            return xr.open_dataset(nc_path)

        cpus = self.processes or max(get_cpus() - 2, 1)
        stations = self.stations()
        meteo_vars = {}

        if self.verbosity:
            print(f"Fetching meteorological data of {len(stations)} stations using {cpus} cpus")

        start = time.time()
        with cf.ProcessPoolExecutor(cpus) as exe:  # takes ~500 secs with 110 cpus
            dfs = exe.map(self.meteo_data_station, stations)

        if self.verbosity:
            print(f"Fetching meteorological data took {time.time() - start:.2f} seconds")

        if self.to_netcdf:
            for stn, df in zip(self.stations(), dfs):
                meteo_vars[stn] = df

            encoding = {stn: {'dtype': 'float32', 'zlib': True, 'complevel': 3} for stn in meteo_vars.keys()}

            meteo_vars = xr.Dataset(meteo_vars)

            if self.verbosity: print(f"Saving to {nc_path}")
            meteo_vars.to_netcdf(nc_path, encoding=encoding)

        return meteo_vars

    def hydro_clim_sigs(
            self,
            stations: List[str] = "all",
            countries: List[str] = "all"
    ) -> pd.DataFrame:
        """
        Returns the hydro-climatic signatures of one or more stations

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of hydro-climatic signatures of shape (stations, 31)
        """
        stations = self._get_stations(countries, stations)

        df = pd.read_csv(
            os.path.join(
                self.path2, 
                'hydroclimatic_signatures',
                'estreams_hydrometeo_signatures.csv'),
            index_col='basin_id',
            dtype={'basin_id': str}
        )
        return df.loc[stations, :]

    def fetch_stn_dynamic_features(
            self,
            station: str,
            dynamic_features='all',
    ) -> pd.DataFrame:
        """
        Fetches all or selected dynamic features of one station.

        Parameters
        ----------
            station : str
                name/id of station of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                dynamic features are returned.

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of shape (n, features) where n is the number of days

        Examples
        --------
        >>> from aqua_fetch import EStreams
        >>> camels = EStreams()
        >>> camels.fetch_stn_dynamic_features('IEEP0281').unstack()
        >>> camels.dynamic_features
        >>> camels.fetch_stn_dynamic_features('IEEP0281',
        ... features=['p_mean', 't_mean', 'pet_mean']).unstack()
        """
        features = check_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')

        return self.meteo_data_station(station).loc[:, features]

    def fetch_dynamic_features(
            self,
            stations: Union[List[str], str] = "all",
            dynamic_features='all',
            st=None,
            en=None,
            as_dataframe=False,
            countries: Union[str, List[str]] = "all",
    ):
        """Fetches all or selected dynamic features of one station.

        Parameters
        ----------
            stations : str
                name/id of station of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                dynamic features are returned.
            st : Optional (default=None)
                start time from where to fetch the data.
            en : Optional (default=None)
                end time untill where to fetch the data
            as_dataframe : bool, optional (default=False)
                if true, the returned data is :obj:`pandas.DataFrame` otherwise it
                is :obj:`xarray.Dataset`

        Examples
        --------
        >>> from aqua_fetch import EStreams
        >>> camels = EStreams()
        >>> camels.fetch_dynamic_features('IEEP0281', as_dataframe=True).unstack()
        >>> camels.dynamic_features
        >>> camels.fetch_dynamic_features('IEEP0281',
        ... features=['p_mean', 't_mean', 'pet_mean'],
        ... as_dataframe=True).unstack()
        """

        stations = self._get_stations(countries, stations)

        features = check_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')

        if len(stations) == 1:
            if as_dataframe:
                return self.fetch_stn_dynamic_features(stations[0], features)
            else:
                return xr.Dataset({stations[0]: xr.DataArray(self.fetch_stn_dynamic_features(stations[0], features))})

        if as_dataframe:
            raise NotImplementedError("as_dataframe=True is not implemented yet")

        return self.meteo_data(stations).sel(dynamic_features=features)


class _EStreams(_RainfallRunoff):
    """
    Parent class for those datasets which use static and dynamic data from EStreams.
    """

    def __init__(
            self,
            path: Union[str, os.PathLike] = None,
            estreams_path: Union[str, os.PathLike] = None,
            overwrite: bool = False,
            verbosity: int = 1,
            **kwargs):
        super().__init__(path, verbosity=verbosity, overwrite=overwrite, **kwargs)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if estreams_path is None:
            if self.verbosity:
                print(f"estreams_path is not provided, using {os.path.dirname(self.path)} as default")
            self.estreams_path = os.path.dirname(self.path)
        else:
            self.estreams_path = estreams_path

        self.estreams = EStreams(path=self.estreams_path, overwrite=overwrite, verbosity=verbosity)

        self.md = self.estreams.md.loc[self.estreams.md['gauge_country'] == self.country_name]
        self._stations = self.estreams.country_stations(self.country_name)
        self.boundary_file = self.estreams.boundary_file

        self.bndry_id_map = self.estreams.bndry_id_map.copy()

    @property
    def dynamic_features(self) -> List[str]:
        return [observed_streamflow_cms()] + self.estreams.dynamic_features

    @property
    def static_features(self) -> List[str]:
        return self.estreams.static_features

    @property
    def country_name(self) -> str:
        return NotImplementedError

    @property
    def _coords_name(self) -> List[str]:
        return ['lat', 'lon']

    @property
    def _area_name(self) -> str:
        # area_official is the area given by data provides but it contains nans
        # supposing that estreams people's method is better/updated one
        return catchment_area()

    @property
    def _q_name(self) -> str:
        return observed_streamflow_cms()

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp('1950-01-01')

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp('2023-06-30')

    def _fetch_dynamic_features(
            self,
            stations: list,
            dynamic_features='all',
            st=None,
            en=None,
            as_dataframe=False,
    ):
        """Fetches dynamic features of station."""
        st, en = self._check_length(st, en)
        features = check_attributes(dynamic_features, self.dynamic_features.copy(), 'dynamic_features')

        daily_q = None

        if observed_streamflow_cms() in features:
            daily_q = self.get_q(as_dataframe)
            if isinstance(daily_q, xr.Dataset):
                daily_q = daily_q.sel(time=slice(st, en))[stations]
            else:
                daily_q = daily_q.loc[st:en, stations]

            features.remove(observed_streamflow_cms())

        if len(features) == 0:
            return daily_q

        # stations_ = [f"{stn}_{self.agency_name}" for stn in stations]
        data = self.estreams.fetch_dynamic_features(stations, features, st, en, as_dataframe)

        if daily_q is not None:
            if isinstance(daily_q, xr.Dataset):
                assert isinstance(data, xr.Dataset), "xarray dataset not supported"
                data = data.rename({stn: stn.split('_')[0] for stn in data.data_vars})

                # first create a new dimension in daily_q named dynamic_features
                daily_q = daily_q.expand_dims({'dynamic_features': [observed_streamflow_cms()]})
                data = xr.concat([data, daily_q], dim='dynamic_features').sel(time=slice(st, en))
            else:
                # -1 because the data in .nc files hysets starts with 0
                data.rename(columns={stn: stn.split('_')[0] for stn in stations}, inplace=True)
                assert isinstance(data.index, pd.MultiIndex)
                # data is multiindex dataframe but daily_q is not
                # first make daily_q multiindex
                daily_q['dynamic_features'] = 'daily_q'
                daily_q.set_index('dynamic_features', append=True, inplace=True)
                daily_q = daily_q.reorder_levels(['time', 'dynamic_features'])
                data = pd.concat([data, daily_q], axis=0).sort_index()

        return data

    def _fetch_static_features(
            self,
            station="all",
            static_features: Union[str, list] = 'all',
            st=None,
            en=None,
    ) -> pd.DataFrame:
        """Fetches static features of station."""
        if self.verbosity > 1:
            print('fetching static features')
        stations = check_attributes(station, self.stations(), 'stations')
        # stations_ = [f"{stn}_{self.agency_name}" for stn in stations]
        static_feats = self.estreams.fetch_static_features(stations, static_features).copy()
        # static_feats.index = [stn.split('_')[0] for stn in static_feats.index]
        return static_feats

    def fetch_stations_features(
            self,
            stations: list,
            dynamic_features: Union[str, list, None] = 'all',
            static_features: Union[str, list, None] = None,
            st=None,
            en=None,
            as_dataframe: bool = False,
            **kwargs
    ):
        """
        returns features of multiple stations

        Returns
        -------
        tuple
            A tuple of static and dynamic features. Static features are always
            returned as :obj:`pandas.DataFrame` with shape (stations, static features).
            The index of static features' DataFrame is the station/gauge ids while the columns 
            are names of the static features. Dynamic features are returned either as
            :obj:`xarray.Dataset` or :obj:`pandas.DataFrame` depending upon whether `as_dataframe`
            is True or False and whether the :obj:`xarray` library is installed or not.
            If dynamic features are :obj:`xarray.Dataset`, then this dataset consists of `data_vars`
            equal to the number of stations and station names as :obj:`xarray.Dataset.variables`  
            and `time` and `dynamic_features` as dimensions and coordinates. If 
            dynamic features are returned as :obj:`pandas.DataFrame`, then
            the first index is `time` and the second index is `dynamic_features`.

        Examples
        --------
        >>> from aqua_fetch import Arcticnet
        >>> dataset = Arcticnet()
        >>> stations = dataset.stations()
        >>> features = dataset.fetch_stations_features(stations)
        """
        stations = check_attributes(stations, self.stations(), 'stations')
        static, dynamic = None, None

        if dynamic_features is not None:

            dynamic = self._fetch_dynamic_features(stations=stations,
                                               dynamic_features=dynamic_features,
                                               as_dataframe=as_dataframe,
                                               st=st,
                                               en=en,
                                               **kwargs
                                               )

            if static_features is not None:  # we want both static and dynamic
                static = self._fetch_static_features(station=stations,
                                                     static_features=static_features,
                                                     st=st,
                                                     en=en,
                                                     **kwargs
                                                     )


        elif static_features is not None:
            # we want only static
            static = self._fetch_static_features(
                station=stations,
                static_features=static_features,
                **kwargs
            )
        else:
            raise ValueError(f"""
            static features are {static_features} and dynamic features are also {dynamic_features}""")

        return static, dynamic

    def fetch_static_features(
            self,
            stations: Union[str, List[str]] = "all",
            static_features: Union[str, List[str]] = "all",
            st=None,
            en=None,
    ) -> pd.DataFrame:
        """
        returns static atttributes of one or multiple stations

        Parameters
        ----------
            stations : str
                name/id of station of which to extract the data
            static_features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.
            st :
            en :

        Examples
        ---------
        >>> from aqua_fetch import Japan
        >>> dataset = Japan()
        get the names of stations
        >>> stns = dataset.stations()
        >>> len(stns)
            12004
        get all static data of all stations
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (12004, 27)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('01010070')
        >>> static_data.shape
           (1, 27)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['Drainage_Area_km2', 'Elevation_m'])
        >>> static_data.shape
           (12004, 2)
        """
        return self._fetch_static_features(stations, static_features, st, en)


START_YEAR = 2012
# todo : why q for only 239 stations is downloaded and others return HTTPError, it is 
# due to wrong fromatting error in pd.read_csv?
# better to save all the data downloaded i.e. water level and temperature as well

class Finland(_EStreams):
    """
    Data of 669 catchments of Finland. 
    The observed streamflow data is downloaded from 
    https://wwwi3.ymparisto.fi .
    The meteorological data, static catchment 
    features and catchment boundaries are
    taken from :py:class:`aqua_fetch.EStreams` follwoing the works
    of `Nascimento et al., 2024 <https://doi.org/10.5194/hess-25-471-2021>`_ . Therefore,
    the number of staic features are 35 and dynamic features are 27 and the
    data is available from 2012-01-01 to 2023-06-30.
    """
    def __init__(
            self, 
            path:Union[str, os.PathLike] = None,
            estreams_path:Union[str, os.PathLike] = None,
            verbosity:int=1,
            **kwargs):

        super().__init__(path=path, estreams_path=estreams_path, verbosity=verbosity, **kwargs)

    @property
    def country_name(self)->str:
        return 'FI'

    @property
    def start(self)->pd.Timestamp:
        return pd.Timestamp('2012-01-01')

    def stations(self)->List[str]:
        """returns the `basin_id` of the stations"""
        return self._stations

    def gauge_id_basin_id_map(self)->dict:
        # guage_id '5902650'
        # basin_id 'FI000001'
        # '5902650' -> 'FI000001'
        return {k:v for v,k in self.md['gauge_id'].to_dict().items()}

    def basin_id_gauge_id_map(self)->dict:
        # guage_id '5902650'
        # basin_id 'FI000001'
        # 'FI000001' -> '5902650'
        return self.md['gauge_id'].to_dict()
    
    def get_q(self, as_dataframe:bool=True, overwrite:bool=False):
        """
        downloads (if not already downloaded) and returns the daily streamflow data of Finland.
        either as :obj:`pandas.DataFrame` or as xarray dataset.
        """
        fpath = os.path.join(self.path, 'daily_q.csv')

        if not os.path.exists(fpath) or overwrite:

            if self.verbosity: print("Downloading discharge data For Finland")

            df_2001_2023 = self.download_2001_2023()
            df_2024 = self.download_2024()
        
            data = pd.concat([df_2001_2023, df_2024])
            data.index.name = 'time'
            data.to_csv(fpath, index_label="index")

        else:
            if self.verbosity>1: 
                print(f"Reading from pre-existing {fpath} file")
            data = pd.read_csv(fpath, index_col="index",
                                na_values=['-'])
            data.index = pd.to_datetime(data.index)
            data.index.name = 'time'

        if as_dataframe:
            return data
        
        return xr.Dataset({stn: xr.DataArray(data.loc[:, stn]) for stn in data.columns})

    def download_2024(self):

        if self.verbosity: 
            print("Downloading 2024 year data")

        dfs = []
        failures = 0
        for idx, bsn_id in enumerate(self.stations()):

            gauge_id = self.basin_id_gauge_id_map()[bsn_id]

            url = f"https://wwwi3.ymparisto.fi/i3/tilanne/ENG/discharge/image/bigimage/Q{gauge_id}.txt"

            try:
                df = pd.read_csv(url, 
                                #delim_whitespace=True,
                                sep='\s+',
                                skiprows=10, 
                                encoding="ISO-8859-1",
                                decimal=',',
                                names=['date', bsn_id, 'avg', 'min', 'max'],
                                index_col='date',
                                parse_dates=True,
                                dayfirst=True,
                                na_values=['-']
                                )
            except HTTPError:
                failures += 1
                warnings.warn(f" {idx} Failed to download {bsn_id} {failures}", UserWarning)
                df = pd.DataFrame(columns=['date', bsn_id, 'avg', 'min', 'max'])

            if self.verbosity>2:
                print(f"{idx}: for {bsn_id} {df.shape}")

            dfs.append(df[bsn_id].astype('float32'))

        df_2024 = pd.concat(dfs, axis=1)

        if self.verbosity: 
            print(f"Downloaded data of shape {df_2024.shape} for 2024")

        return df_2024

    def download_2001_2023(self):

        if self.verbosity: print("Downloading 2012-2023 year data")

        cpus = self.processes or max(get_cpus()-2, 1)

        if self.verbosity:
            print(f"downloading daily data for {len(self.stations())} stations from {2012} to {2023} with {cpus} cpus")

        if cpus == 1:
            all_data = self.download_data_seq()
        else:
            all_data = self.download_data_parallel(cpus)
        
        if self.verbosity>2:
            print(f"total number of stations: {len(all_data)} each with shape {all_data[0].shape}")
        
        df_2012_2023 = pd.concat(all_data, axis=1)

        if self.verbosity: print(f"Downloaded data of shape {df_2012_2023.shape} for 2001-2023")

        return df_2012_2023

    def download_data_parallel(self, cpus:int=None):
        # todo : taking forever to download the data
        start = time.time()

        stations = self.stations()

        _map = self.basin_id_gauge_id_map()
        basin_ids = [_map[stn] for stn in stations]
        years = range(START_YEAR, 2024)            
        stations_ = [[stn]*len(years) for stn in stations]
        # flatten the list
        stations_ = [item for sublist in stations_ for item in sublist]
        basin_ids_ = [[bsn_id]*len(years) for bsn_id in basin_ids]
        basin_ids_ = [item for sublist in basin_ids_ for item in sublist]
        years_ = list(years) * len(stations)

        if self.verbosity>1:
            print(f"Total function calls: {len(stations_)} with {cpus} cpus")

        with cf.ProcessPoolExecutor(cpus) as executor:
            results = executor.map(download_daily_stn_yr, basin_ids_, stations_, 
                                   years_)        

        if self.verbosity:
            print(f"total time taken to download data: {time.time() - start}")

        all_data = []
        for bsn_id, stn in zip(basin_ids, stations):
            stn_data = []
            for yr in years:
                stn_yr_data = next(results)
                stn_data.append(stn_yr_data[bsn_id])
            
            stn_data = pd.concat(stn_data, axis=0)
            stn_data.name = stn

            if self.verbosity>2:
                print(f"for {stn} with shape {stn_data.shape}")

            all_data.append(stn_data)
        return all_data

    def download_data_seq(self):
        # takes around 1 hour to download all the data
        failures = 0
        dfs = []
        stations = self.stations()
        for idx, bsn_id in enumerate(stations):

            gauge_id = self.basin_id_gauge_id_map()[bsn_id]

            stn_dfs = []
            for year in range(2012, 2024):

                url = f"https://wwwi3.ymparisto.fi/i3/kktiedote/ENG/{year}/discharge/image/bigimage/Q{gauge_id}.txt"

                if year == 2012: skiprows = 7 
                elif year in [2013, 2014]: skiprows = 9
                else: skiprows = 10

                try:
                    yr_df = pd.read_csv(url, 
                                    #delim_whitespace=True,
                                    sep='\s+',
                                    skiprows=skiprows, 
                                    encoding="ISO-8859-1",
                                    decimal=',',
                                    names=['date', bsn_id, 'avg', 'min', 'max'],
                                    index_col='date',
                                    parse_dates=True,
                                    dayfirst=True,
                                    na_values=['-'],
                                    )
                except HTTPError:
                    failures += 1
                    warnings.warn(f" {idx} Failed to download {bsn_id} {year} {failures}", UserWarning)
                    yr_df = pd.DataFrame(
                        columns=['date', bsn_id, 'avg', 'min', 'max'],
                    )

                stn_dfs.append(yr_df)
            
            if len(stn_dfs) > 0:
                stn_df = pd.concat(stn_dfs, axis=0)
                if self.verbosity:
                    print(f"{idx}/{len(stations)}: for {bsn_id} {stn_df.shape} {len(stn_dfs)}")

                dfs.append(stn_df[bsn_id].astype('float32'))

        return dfs

def download_daily_stn_yr(
        gauge_id:str,
        bsn_id:str,
        year:int
        )->pd.DataFrame:

    url = f"https://wwwi3.ymparisto.fi/i3/kktiedote/ENG/{year}/discharge/image/bigimage/Q{gauge_id}.txt"

    if year == 2012: skiprows = 7 
    elif year in [2013, 2014]: skiprows = 9
    else: skiprows = 10

    try:
        yr_df = pd.read_csv(url, 
                        #delim_whitespace=True,
                        sep='\s+',
                        skiprows=skiprows, 
                        encoding="ISO-8859-1",
                        decimal=',',
                        names=['date', bsn_id, 'avg', 'min', 'max'],
                        index_col='date',
                        parse_dates=True,
                        dayfirst=True,
                        na_values=['-'],
                        )
    except HTTPError:
        yr_df = pd.DataFrame(
            columns=['date', bsn_id, 'avg', 'min', 'max'],
        )
    
    return yr_df



class Ireland(_EStreams):
    """
    Data of 464 catchments of Ireland. Out of these 464 catchments, 
    280 are from OPW and 184 are from EPA.
    The observed streamflow data for EPA stations is downloaded from 
    https://epawebapp.epa.ie/Hydronet/#Flow while the observed streamflow for OPW 
    stations is downloaded from https://waterlevel.ie/hydro-data/#/overview/Waterlevel.
    It should be that out of 280 OPW stations, streamflow data is available for only 129
    stations. The meteorological data, static catchment 
    features and catchment boundaries are
    taken from :py:class:`aqua_fetch.EStreams` follwoing the works
    of `Nascimento et al., 2024 <https://doi.org/10.5194/hess-25-471-2021>`_ project. Therefore,
    the number of staic features are 35 and dynamic features are 27 and the
    data is available from 1992-01-01 to 2020-06-31.
    """
    def __init__(
            self, 
            path:Union[str, os.PathLike] = None,
            estreams_path:Union[str, os.PathLike] = None,
            verbosity:int=1,
            **kwargs):

        super().__init__(path=path, estreams_path=estreams_path, verbosity=verbosity, **kwargs)

    @property
    def country_name(self)->str:
        return 'IE'
   
    @property
    def epa_stations(self)->List[str]:
        md = self.estreams.md
        stns = md.loc[(md['gauge_country']=='IE') & (md['gauge_provider']=='IE_EPA')]['gauge_id']
        epa_stns =  stns.tolist()
        if self.timestep in ('H', 'hourly'): epa_stns.remove('38004')  # todo: 
        return epa_stns

    @property
    def opw_stations(self)->List[str]:
        md = self.estreams.md
        stns = md.loc[(md['gauge_country']=='IE') & (md['gauge_provider']=='IE_OPW')]['gauge_id']
        return stns.tolist()
    
    def is_opw_station(self, stn)->bool:
        return stn in self.opw_stations

    def is_epa_station(self, stn)->bool:
        return stn in self.epa_stations

    def stations(self)->List[str]:
        """The `basin_id` EStreams dataset is used as station names"""
        return self._stations
    
    def gauge_id_basin_id_map(self)->dict:
        """
        A dictionary whose keys are gauge_id and values are basin_id. 
        Supposing guage_id is '18118' and basin_id is 'IEEP0281'
        then '18118' -> 'IEEP0281'
        """
        return {k:v for v,k in self.md['gauge_id'].to_dict().items()}

    def basin_id_gauge_id_map(self)->dict:
        # guage_id '18118'
        # basin_id 'IEEP0281'
        # 'IEEP0281' -> '18118'
        return self.md['gauge_id'].to_dict()

    def get_q(
            self, 
            as_dataframe:bool=True,
            overwrite:bool=False, 
            ):
        fname = 'daily_q' if self.timestep in ["D", 'daily'] else 'hourly_q'
        ext = '.csv'
    
        fpath = os.path.join(self.path, fname + ext)

        if not os.path.exists(fpath) or overwrite:

            cpus = self.processes or max(get_cpus() - 2, 1)

            if cpus > 1:
                epa_df = self.download_epa_data_parallel(cpus=cpus)
                opw_df = self.download_opw_data_parallel(cpus=cpus)
            else:
                epa_df = self.download_epa_data_seq()
                opw_df = self.download_opw_data_seq()

            data = pd.concat([epa_df, opw_df], axis=1)
            data.index.name = 'time'
            data.rename(columns=self.gauge_id_basin_id_map(), inplace=True)

            if ext == '.csv':
                data.to_csv(fpath, index_label="index")
            else:
                data = xr.Dataset({stn: xr.DataArray(data.loc[:, stn]) for stn in data.columns})
                data.to_netcdf(fpath)

        else:
            if self.verbosity > 1: print(f"Reading from pre-exising {fpath}")
            if ext == '.csv':
                data = pd.read_csv(fpath, index_col="index")
                data.index = pd.to_datetime(data.index)
                data.index.name = 'time'
                data.rename(columns=self.gauge_id_basin_id_map(), inplace=True)
            else:
                data = xr.open_dataset(fpath)
                if self.verbosity > 1: print(f"opened {fpath}")

        if isinstance(data, pd.DataFrame):
            if as_dataframe:
                return data
            return xr.Dataset({stn: xr.DataArray(data.loc[:, stn]) for stn in data.columns})
        else:
            if as_dataframe:
                data = data.to_dataframe()
                data.index.name = 'time'
                return data
            return data

    def download_epa_data_seq(self):
        """
        Examples
        ---------
        >>> epa_df = download_epa_data()
        """
        folder = {'D': 'daily', 'H': 'hourly'}[self.timestep]

        all_epa_data_file = os.path.join(self.path, f"epa_{folder}.csv")
        if os.path.exists(all_epa_data_file):
            if self.verbosity>1: print(f"{all_epa_data_file} already exists")
            df = pd.read_csv(all_epa_data_file, index_col=0, parse_dates=True)
            print(f"{all_epa_data_file} already exists")  
            return df

        print("Downloading EPA data Sequentially")

        epa_failiures = 0
        epa_dfs = []

        for idx, stn in enumerate(self.epa_stations):

            fpath = os.path.join(self.path, "EPA", folder, f"{stn}.csv")

            print(f"{idx}/{len(self.epa_stations)} Downloading {stn}")

            df, epa_failiures = _download_epa_stn_data(fpath, self.timestep)

            epa_dfs.append(df) 

        print(f'total epa failiures: {epa_failiures}')
        print(f'total epa dfs: {len(epa_dfs)}')

        df = pd.concat(epa_dfs, axis=1).astype('float32')

        if self.verbosity>1: print(f"Downloaded total epa dfs: {len(epa_dfs)} saving to {all_epa_data_file}")
        df.to_csv(all_epa_data_file)
        return df

    def download_epa_data_parallel(self, cpus=None):

        if cpus is None:
            cpus = self.processes or max(get_cpus() - 2, 1)

        folder = {'D': 'daily', 'H': 'hourly'}[self.timestep]

        all_epa_data_file = os.path.join(self.path, f"epa_{folder}.csv")
        if os.path.exists(all_epa_data_file):
            df = pd.read_csv(all_epa_data_file, index_col=0, parse_dates=True)
            print(f"{all_epa_data_file} already exists")  
            return df

        timesteps = [self.timestep] * len(self.epa_stations)
        fpaths = [os.path.join(self.path, "EPA", folder, f"{stn}.csv") for stn in self.epa_stations]

        print(f"Downloading {len(fpaths)} EPA stations using {cpus} cpus at {os.path.join(self.path, 'EPA', folder)}")

        with ProcessPoolExecutor(cpus) as executor:
            epa_dfs = list(executor.map(_download_epa_stn_data, fpaths, timesteps))

        df = pd.concat([val[0] for val in epa_dfs], axis=1).astype('float32')

        if self.timestep in ["D", 'daily']:
            # remove hourly information from the index
            # 2000-01-01 01:00:00 -> 2000-01-01
            df.index = df.index.normalize()

        print(f'Downloaded total epa dfs: {len(epa_dfs)}')

        df.to_csv(all_epa_data_file)
        return df
    
    def download_opw_data_parallel(self, cpus=None):

        folder = {'D': 'daily', 'H': 'hourly'}[self.timestep]

        all_opw_data_file = os.path.join(self.path, f"opw_{folder}.csv")
        if os.path.exists(all_opw_data_file):
            df = pd.read_csv(all_opw_data_file, index_col=0, parse_dates=True)
            print(f"{all_opw_data_file} already exists")  
            return df

        fpaths = [os.path.join(self.path, "OPW", folder, f"{stn}.csv") for stn in self.opw_stations]

        if self.verbosity:
            print(f"Downloading {len(fpaths)} OPW stations using {cpus} cpus at {os.path.join(self.path, 'OPW', folder)}")

        with ProcessPoolExecutor(cpus) as executor:
            opw_dfs = list(executor.map(_download_opw_stn_data, fpaths, [self.timestep]*len(self.opw_stations)))

        opw_df = [df for df in opw_dfs]
        opw_df = pd.concat(opw_df, axis=1).astype('float32')

        if self.timestep in ("D", "daily"):
            opw_df.index = opw_df.index.tz_localize(None)

        if self.verbosity:
            print(f"Downloaded total opw dfs: {len(opw_dfs)}")
            print(f"Saving opw data {opw_df.shape} to {all_opw_data_file}")

        failures = [df for df in opw_dfs if len(df)==0]
        if len(failures)>0:
            self.opw_failures = [s.name for s in failures]
            warnings.warn(f"Failed to download {len(failures)} OPW stations")
    
        opw_df.to_csv(all_opw_data_file)

        return opw_df

    def download_opw_data_seq(self):
        """
        Examples
        ---------
        >>> opw_df = download_opw_data()
        """

        folder = {'D': 'daily', 'H': 'hourly'}[self.timestep]

        all_opw_data_file = os.path.join(self.path, f"opw_{folder}.csv")
        if os.path.exists(all_opw_data_file):
            df = pd.read_csv(all_opw_data_file, index_col=0, parse_dates=True)
            if self.verbosity: print(f"{all_opw_data_file} already exists")  
            return df

        if self.verbosity: print("Downloading OPW data")

        failiures = 0
        opw_dfs = []
        for idx, stn in enumerate(self.opw_stations):

            fpath = os.path.join(self.path, "OPW", folder, f"{stn}.csv")

            print(f"{idx}/{len(self.opw_stations)} Downloading {stn}")

            df = _download_opw_stn_data(fpath, self.timestep)
 
            opw_dfs.append(df)
            if len(df)==0:
                failiures += 1

        if self.verbosity:
            print(f"total failiures: {failiures}")
            print(f"total opw dfs: {len(opw_dfs)}")

        opw_dfs1 = [df for df in opw_dfs if len(df)>0]
        opw_df = pd.concat(opw_dfs1, axis=1).astype('float32')

        #if self.timestep in ("D", "daily"):
        opw_df.index = opw_df.index.tz_localize(None)

        if self.verbosity:
            print(f"Saving opw data {opw_df.shape} to {all_opw_data_file}")
        opw_df.to_csv(all_opw_data_file)

        return opw_df


def _download_epa_stn_data(fpath, timestep="D")->pd.Series:
    stn = os.path.basename(fpath).split('.')[0]
    if timestep in ("D", 'daily'):
        fname = "daymean.zip"
    else:
        fname = "15min.zip"

    epa_failiures = 0

    url = f"https://epawebapp.epa.ie/Hydronet/output/internet/stations/DUB/{stn}/Q/complete_{fname}"
    url2 = f"https://epawebapp.epa.ie/Hydronet/output/internet/stations/MON/{stn}/Q/complete_{fname}"
    url3 = f"https://epawebapp.epa.ie/Hydronet/output/internet/stations/ATH/{stn}/Q/complete_{fname}"
    url4 = f"https://epawebapp.epa.ie/Hydronet/output/internet/stations/COR/{stn}/Q/complete_{fname}"
    url5 = f"https://epawebapp.epa.ie/Hydronet/output/internet/stations/KIK/{stn}/Q/complete_{fname}"
    url6 = f"https://epawebapp.epa.ie/Hydronet/output/internet/stations/CAS/{stn}/Q/complete_{fname}"

    try:
        df = pd.read_csv(url, 
                    comment='#', 
                    sep=';',
                    names=["timestamp", stn, "qflag"]
                    )
    except HTTPError:
        try:
            df = pd.read_csv(url2, 
            comment='#', 
            sep=';',
            names=["timestamp", stn, "qflag"])
        except HTTPError:
            try:
                df = pd.read_csv(url3, 
                comment='#', 
                sep=';',
                names=["timestamp", stn, "qflag"])
            except HTTPError:
                try:
                    df = pd.read_csv(url4, 
                    comment='#', 
                    sep=';',
                    names=["timestamp", stn, "qflag"])
                except HTTPError:
                    try:
                        df = pd.read_csv(url5, 
                        comment='#', 
                        sep=';',
                        names=["timestamp", stn, "qflag"])
                    except HTTPError:
                        try:
                            df = pd.read_csv(url6, 
                            comment='#', 
                            sep=';',
                            names=["timestamp", stn, "qflag"])
                        except HTTPError:
                            print(f"Failed to download {stn}")
                            epa_failiures += 1
                            pass


    df.index = pd.to_datetime(df.pop('timestamp'))

    # considering quality codes https://epawebapp.epa.ie/Hydronet/#FAQ
    # Quality codes: Good, nan, Suspect, Extrapolated, Unchecked, Excellent, Estimated

    df = df.loc[~df['qflag'].isin(['Unchecked'])]
    
    if timestep == "D":
        return df[stn], epa_failiures

    return df[stn].resample(timestep).mean(), epa_failiures


def _download_opw_stn_data(fpath, timestep="D")->pd.Series:
    stn = os.path.basename(fpath).split('.')[0]
    # we don't/can't download daily data 
    if timestep == "daily":
        timestep = "D"
    elif timestep == "hourly":
        timestep = "H"
    elif timestep == "D":
        pass
    else:
        raise ValueError(f"timestep should be either 'D' or 'H' but it is {timestep}")

    url = f"https://waterlevel.ie/hydro-data/data/internet/stations/0/{stn}/Q/Discharge_complete.zip"

    try:
        df = pd.read_csv(url,
                        comment='#',
                        sep=';',
                        names=["timestamp", stn, "q_code"]
                        )
    except HTTPError:
        warnings.warn(f"Failed to download {stn}", UserWarning)
        df = pd.Series(name=stn)

    df.index = pd.to_datetime(df.pop('timestamp'))
    df.index = df.index.tz_localize(None)  

    # considering quality codes as given here https://waterlevel.ie/hydro-data/#/html/qualitycodes
    # df['q_code'] has following values : 36, 46, 31, 56, 96, 225, 101, 32, 99, 254
    # 96 and 254 are provisional values and can be changed!
    # 101 is erronous while 36 and 46 contain fair and siginificant errors respectively.

    # get rows where q_code is not 96 or 254
    df = df.loc[~df['q_code'].isin([96, 254])]
    
    stn_data = df[stn]
    #stn_data = stn_data.resample(timestep).apply(lambda subdata: tw_resampler(subdata, stn_data.sort_index(), timestep))    
    stn_data = stn_data.resample(timestep).mean()
    return stn_data


class Italy(_EStreams):
    """
    Data of 294 catchments of Italy. 
    The observed streamflow data is downloaded from 
    http://www.hiscentral.isprambiente.gov.it/hiscentral/hydromap.aspx?map=obsclient .
    The meteorological data, static catchment 
    features and catchment boundaries are
    taken from :py:class:`aqua_fetch.EStreams` follwoing the works
    of `Nascimento et al., 2024 <https://doi.org/10.5194/hess-25-471-2021>`_ . Therefore,
    the number of staic features are 35 and dynamic features are 27 and the
    data is available from 1992-01-01 to 2020-06-31.
    """
    def __init__(
            self, 
            path:Union[str, os.PathLike] = None,
            estreams_path:Union[str, os.PathLike] = None,
            verbosity:int=1,
            **kwargs):

        super().__init__(path=path, estreams_path=estreams_path, verbosity=verbosity, **kwargs)

        self._stations = self.ispra_stations()

    @property
    def country_name(self)->str:
        return 'IT'

    def gauge_id_basin_id_map(self)->dict:
        # guage_id 'hsl-abr:5010'
        # basin_id 'ITIS0001'
        # 'hsl-abr:5010' -> 'ITIS0001'
        return {k:v for v,k in self.md['gauge_id'].to_dict().items()}
    
    def stations(self)->List[str]:
        """returns the basin_id of the stations"""
        return self._stations

    def ispra_stations_gauge_ids(self)->List[str]:
        return self.md.loc[self.md['gauge_provider']=='IT_ISPRA']['gauge_id'].to_list()

    def ispra_stations(self)->List[str]:
        return self.md.loc[self.md['gauge_provider']=='IT_ISPRA'].index.to_list()
        
    def all_stations(self)->List[str]:
        return self.estreams.country_stations("IT")

    def get_q(self, as_dataframe:bool=True):
        fpath = os.path.join(self.path, 'daily_q.csv')

        if not os.path.exists(fpath) or self.overwrite:

            data = self.download_ispra_data()
        
            data.to_csv(fpath, index_label="time")

        else:
            if self.verbosity > 1: 
                print(f"Reading q from pre-existing {fpath} file")
            data = pd.read_csv(fpath, index_col="time")
            data.index = pd.to_datetime(data.index)
            data.index.name = "time"

        # replace 'hsl-abr:5010' with 'ITIS0001'
        data.rename(columns=self.gauge_id_basin_id_map(), inplace=True)

        if as_dataframe:
            return data
        
        return xr.Dataset({stn: xr.DataArray(data.loc[:, stn]) for stn in data.columns})

    def download_ispra_data(self):      

        if self.verbosity > 1:
            print("Downloading ISPRA data")

        dfs = []

        for idx, station in enumerate(self.ispra_stations_gauge_ids()):

            initial = station.split(":")[0]
            response = requests.get(
                f"http://hydroserver.ddns.net/italia/{initial}/index.php/default/services/cuahsi_1_1.asmx/GetValuesObject?authToken=&location={station}&variable={initial}:Discharge&startDate=1900-01-01&endDate=2020-12-31")
            root = ET.fromstring(response.content)

            namespace = {'ns': 'http://www.cuahsi.org/waterML/1.1/'}
            # Extract the time series data
            timeseries = []
            for value in root.findall('.//ns:value', namespace):
                date_time = value.attrib['dateTime']
                data_value = value.text
                timeseries.append({'dateTime': date_time, 'value': data_value})

            df = pd.DataFrame(timeseries)

            df.index = pd.to_datetime(df.pop('dateTime'))
            df.columns = [station]
            print(idx, station, df.shape)

            dfs.append(df)    

        df = pd.concat(dfs, axis=1)

        return df


def download_ispra_stn(station):
    initial = station.split(":")[0]
    response = requests.get(
        f"http://hydroserver.ddns.net/italia/{initial}/index.php/default/services/cuahsi_1_1.asmx/GetValuesObject?authToken=&location={station}&variable={initial}:Discharge&startDate=1900-01-01&endDate=2020-12-31")
    root = ET.fromstring(response.content)

    namespace = {'ns': 'http://www.cuahsi.org/waterML/1.1/'}
    # Extract the time series data
    timeseries = []
    for value in root.findall('.//ns:value', namespace):
        date_time = value.attrib['dateTime']
        data_value = value.text
        timeseries.append({'dateTime': date_time, 'value': data_value})

    df = pd.DataFrame(timeseries)

    df.index = pd.to_datetime(df.pop('dateTime'))
    df.columns = [station]


# todo: why concatenating the 1077 stations in prior to 2023 and 833 
# stations from 2023 become 1181? Like a lot of new stations come in 2023?

class Poland(_EStreams):
    """
    Data of 1287 catchments of Poland. 
    The observed streamflow data is downloaded from 
    https://danepubliczne.imgw.pl .
    The meteorological data, static catchment 
    features and catchment boundaries are
    taken from :py:class:`aqua_fetch.EStreams` follwoing the works
    of `Nascimento et al., 2024 <https://doi.org/10.5194/hess-25-471-2021>`_ . Therefore,
    the number of staic features are 35 and dynamic features are 27 and the
    data is available from 1992-01-01 to 2020-06-31.
    """
    def __init__(
            self, 
            path:Union[str, os.PathLike] = None,
            estreams_path:Union[str, os.PathLike] = None,
            verbosity:int=1,
            **kwargs):

        super().__init__(path=path, estreams_path=estreams_path, verbosity=verbosity, **kwargs)

    @property
    def country_name(self)->str:
        return 'PL'

    def gauge_id_basin_id_map(self)->dict:
        # guage_id '149180020'
        # basin_id 'PL000001'
        # '149180020' -> 'PL000001'
        return {k:v for v,k in self.md['gauge_id'].to_dict().items()}

    def basin_id_gauge_id_map(self)->dict:
        # guage_id '149180020'
        # basin_id 'PL000001'
        # 'PL000001' -> '149180020'
        return self.md['gauge_id'].to_dict()

    @property
    def zip_files_dir(self)->str:
        """path where zip files will be stored"""
        return os.path.join(self.path, 'zip_files')

    @property
    def csv_files_dir(self)->str:
        """path where csv (obtained after extracting zip files) files will be stored"""
        return os.path.join(self.path, 'csv_files')

    def stations(self)->List[str]:
        """returns the basin_id of the stations"""
        return self._stations

    def get_q(self, as_dataframe:bool=True):

        fpath = os.path.join(self.path, 'daily_q.csv')

        if not os.path.exists(fpath):
            data = self._make_csv()
        else:
            if self.verbosity: print(f"Reading from existing {fpath} file")
            data = pd.read_csv(fpath, index_col="time")
            data.index = pd.to_datetime(data.index)
            data.index.name = 'time'

        # replace '149180020' with 'PL000001'
        data.rename(columns=self.gauge_id_basin_id_map(), inplace=True)

        # todo: make sure that the following stations actually not have any data
        if data.shape[1]<len(self.stations()):
            warnings.warn(f"{len(self.stations())-data.shape[1]} stations are missing in the downloaded data")
            for stn in self.stations():
                if stn not in data.columns:
                    data[stn] = np.nan

        if as_dataframe:
            return data
        
        return xr.Dataset({stn: xr.DataArray(data.loc[:, stn]) for stn in data.columns})

    def _make_csv(self):

        years = []
        months = []
        for yr in range(1951, 2023):
            for month in range(1, 13):
                month = str(month).zfill(2)
                years.append(yr)
                months.append(month)

        cpus = self.processes or max(get_cpus()-2, 1)

        if self.verbosity:
            print(f"Downloading zip files using {cpus} cpus")

        start = time.time()
        with cf.ProcessPoolExecutor(max_workers=cpus) as executor:
            data = list(executor.map(download_single_file, years, months))

        if self.verbosity: 
            print(f"Downloaded all files in {time.time()-start} seconds")

        data = pd.concat([df for df in data], axis=0)

        if self.verbosity>1:
            print(f"Data until 2022 has shape: {data.shape}")

        data23 =  download_data_2023(2023)

        if self.verbosity>1:
            print(f"Data for 2023 has shape: {data23.shape}")
    
        data = pd.concat([data, data23], axis=0)
        data.sort_index(inplace=True)

        data.index.name = 'time'

        if self.verbosity:
            print(f"Saving daily discharge data {data.shape} to {self.path}")
        data.to_csv(os.path.join(self.path, "daily_q.csv"), index_label="time")

        return data


def download_single_file(year, month:str):
    url = f"https://danepubliczne.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_hydrologiczne/dobowe/{year}/codz_{year}_{month}.zip"

    try:
        df = pd.read_csv(
                        url,
                        compression='zip',
                        encoding="ISO-8859-1",
                        engine='python',
                        on_bad_lines="skip",
                        names=['stn_id', 'year', 'day', 'q_cms', 'month'],
                        usecols=[0, 3, 5, 7, 9],
                        # sometimes casting month to int fails
                        dtype={'stn_id': str, 'year': 'int', 'day': 'int', 'q_cms': np.float32, #'month': 'int'
                               },
                        #parse_dates={'date': ['year', 'month', 'day']},
                        #index_col='date',
                        na_values=[99999.999]
                        )
    except HTTPError:
        raise Exception(f"Failed to download {url}")

    # sometimes (such as 1992-07) month column has missing values
    month = df.loc[:, 'month']
    month = month.ffill()

    yr, month, day = df.loc[:, 'year'].astype(int), month.astype(int), df.loc[:, 'day'].astype(int)    
    df.index = pd.to_datetime(pd.DataFrame({
            'year': yr,
            'month': month,
            'day': day,
        }))

    df = df.pivot_table(index=df.index, columns="stn_id", values="q_cms")
    df.columns = [col.replace(' ', '') for col in df.columns]

    df.rename(columns={"ï»¿149180020": "149180020"}, inplace=True)

    # as per documentation, 99999.999 is missing value
    df.replace(99999.999, np.nan, inplace=True)


    df.index = df.index.tz_localize(None)
    df.sort_index(inplace=True)
    return df


def download_data_2023(year):
    df = pd.read_csv(
        f"https://danepubliczne.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_hydrologiczne/dobowe/{year}/codz_{year}.zip", 
        compression='zip', 
        encoding="ISO-8859-1",  
        engine='python',
        on_bad_lines="skip",
        sep=';',
        names=['stn_id', 'year', 'day', 'q_cms', 'month'],
        usecols=[0, 3, 5, 7, 9],
        dtype={'stn_id': str, 'year': 'int', 'day': 'int', 'q_cms': np.float32, 'month': 'int'},
        parse_dates={'date': ['year', 'month', 'day']},
        index_col='date',
        na_values=[99999.999]
        )

    df.replace("ï»¿149180020", "149180020", inplace=True)

    df = df.pivot_table(index=df.index, columns="stn_id", values="q_cms")

    # replace empty splace in column names
    df.columns = [col.replace(' ', '') for col in df.columns]

    # as per documentation, 99999.999 is missing value
    df.replace(99999.999, np.nan, inplace=True)

    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        raise Exception(f"Failed to convert index to datetime for {year}")

    df.index = df.index.tz_localize(None)
    return df


class Portugal(_EStreams):
    """
    Data of 280 catchments of Portugal.
    The observed streamflow data is downloaded from 
    https://snirh.apambiente.pt .
    The meteorological data, static catchment 
    features and catchment boundaries for the 280 catchments are
    taken from :py:class:`aqua_fetch.EStreams` follwoing the works
    of `Nascimento et al., 2024 <https://doi.org/10.5194/hess-25-471-2021>`_ project. Therefore,
    the number of staic features are 35 and dynamic features are 27 and the
    data is available from 1972-01-01 to 2022-12-31 .
    """
    def __init__(
            self, 
            path:Union[str, os.PathLike] = None,
            estreams_path:Union[str, os.PathLike] = None,
            verbosity:int=1,
            **kwargs):

        super().__init__(path=path, estreams_path=estreams_path, verbosity=verbosity, **kwargs)

        fpath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'portugal_stn_codes.csv')
        self.codes = pd.read_csv(fpath, index_col=0)

    @property
    def country_name(self)->str:
        return 'PT'

    @property
    def start(self)->pd.Timestamp:
        return pd.Timestamp('1972-01-01')

    @property
    def end(self)->pd.Timestamp:
        return pd.Timestamp('2022-12-31')

    def gauge_id_basin_id_map(self)->dict:
        # guage_id '03J/02H'
        # basin_id 'PT000001'
        # '03J/02H' -> 'PT000001'
        return {k:v for v,k in self.md['gauge_id'].to_dict().items()}

    def stations(self)->List[str]:
        return self._stations
    
    def download_q_data_seq(self):
        """downloads q data sequentially"""
        if self.verbosity: print("Downloading q data sequentially")

        start = time.time()

        data = []

        for i in range(len(self.codes)):

            g_code = self.codes.iloc[i, 1]

            stn_data = download_stn_data(g_code)
            stn_data.name = self.codes.index[i]

            data.append(stn_data)

            if self.verbosity and i%5 == 0:
                print(i, "q files downloaded")

        tot_time = round ((time.time() - start) / 60, 2)

        if self.verbosity: print(f"Total Time taken {tot_time} minutes")

        return pd.concat(data, axis=1)
    
    def download_q_data_parallel(self, cpus:int=4):
        """downloads q data in parallel"""
        start = time.time()

        data = []

        with ProcessPoolExecutor(max_workers=cpus) as executor:

            futures = [executor.submit(download_stn_data, g_code) for g_code in self.codes.iloc[:, 1]]

            for i, future in enumerate(futures):

                stn_data = future.result()
                stn_data.name = self.codes.index[i]

                data.append(stn_data)

                if i%10 == 0:
                    print(i, "Done")

        tot_time = round ((time.time() - start) / 60, 2)

        if self.verbosity: print(f"Total Time taken {tot_time} minutes")

        return pd.concat(data, axis=1)

    def get_q(
            self, 
            as_dataframe:bool=True,
            ):
        """
        returns the streamflow data of Portugal as xarray.Dataset or pandas.DataFrame

        Returns
        -------
        xarray.Dataset or pandas.DataFrame. If as_dataframe is True, returns pandas.DataFrame
        with columns as station codes and index as time. If as_dataframe is False, returns
        xarray.Dataset with station codes as variables and time as dimension.
        """
        fname = 'daily_q.csv' 

        fpath = os.path.join(self.path, fname)

        if not os.path.exists(fpath) or self.overwrite:

            if self.verbosity>1: print(f"Downloading q data at {self.path}")

            cpus = self.processes or max(get_cpus() - 2, 1)

            if cpus > 1:
                q_df = self.download_q_data_parallel(cpus=cpus)
            else:
                q_df = self.download_q_data_seq()
        else:
            if self.verbosity: print(f"Reading q data from pre-existing file {fpath}")
            q_df = pd.read_csv(fpath, index_col=0)
            q_df.index = pd.to_datetime(q_df.index, dayfirst=True)
        
        q_df.index.name = 'time'

        # q_df columns are 03J/02H	15G/02H	11H/02H which needs to be mapped to PT000001	PT000002	PT000003
        # because stations are identified by basin_id
        q_df = q_df.rename(columns=self.gauge_id_basin_id_map())

        if as_dataframe:
            return q_df
        return xr.Dataset({stn: xr.DataArray(q_df.loc[:, stn]) for stn in q_df.columns})


def download_stn_data(gauge_code:int)->pd.Series:

    url = f'https://snirh.apambiente.pt/snirh/_dadosbase/site/paraCSV/dados_csv.php?sites={gauge_code}&pars=1850&tmin=01/01/1972&tmax=31/12/2022&formato=csv'

    # Add headers if needed (you may need to adjust these)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }

    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        data = StringIO(response.text)
        # head = pd.read_csv(data, skiprows=1,
        #                 nrows=1)
        df = pd.read_csv(data, skiprows=3, index_col=0, parse_dates=True, dayfirst=True)
        assert df.columns[0] == 'Caudal médio diário (m3/s)'
        s = df.iloc[0:-1, 0]
        s.name = str(gauge_code)
    else:
        warnings.warn(f"Failed to retrieve data: {response.status_code}")
        s = pd.Series(name=str(gauge_code))
    
    return s
