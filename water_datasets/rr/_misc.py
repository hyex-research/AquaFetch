
import os
import time
import concurrent.futures as cf
from typing import List, Union

import pandas as pd

from .._backend import xarray as xr
from ..utils import get_cpus
from ..utils import check_attributes
from .camels import Camels


class DraixBleone(Camels):
    """
    A high-frequency, long-term data set of hydrology and sediment yield: the alpine
    badland catchments of Draix-Bléone Observatory

    """
    url = {
        "spatial": "https://doi.org/10.57745/RUQLJL",
        "hydro_sediment": "https://doi.org/10.17180/obs.draix",
        "climate": "https://doi.org/10.57745/BEYQFQ"
        }


class EStreams(Camels):
    """
    Handles EStreams data following the work of 
    `Nascimento et al., 2024 <https://doi.org/10.1038/s41597-024-03706-1>`_ .
    The data is available at `zenodo <https://zenodo.org/records/13961394>`_ .
    It should be noted that this dataset does not contain observed streamflow data.
    It has 15047 stations, 9 dynamic features with daily timestep, 27 dynamic 
    features with yearly timestep and 208 static features. The dynamic features
    are from 1950-01-01 to 2023-06-30.
    """

    def __init__(self, path = None,  **kwargs):
        super().__init__(path, **kwargs)

        self.md = self.gauge_stations()
        self._stations = self.__stations()
        self._dynamic_features = self.meteo_data_station('IEEP0281').columns.tolist()
        self._static_features = self.static_data().columns.tolist()

        self.boundary_file = os.path.join(self.path,
                                "EStreams",
                                "shapefiles", "estreams_catchments.shp")
        self._create_boundary_id_map(self.boundary_file, 0)

    @property
    def dynamic_features(self)->List[str]:
        return self._dynamic_features

    @property
    def static_features(self):
        return self._static_features

    @property
    def start(self)->pd.Timestamp:
        return pd.Timestamp('1950-01-01')

    @property
    def end(self)->pd.Timestamp:
        return pd.Timestamp('2023-06-30')

    def static_data(self)->pd.DataFrame:
        """
        Returns a dataframe with static attributes of catchments
        """
        static_path = os.path.join(self.path, 'EStreams', 'attributes', 'static_attributes')

        dfs = [self.hydro_clim_sigs(), self.md.copy()]
        for f in os.listdir(static_path):
            if f.endswith('.csv'):
                df = pd.read_csv(os.path.join(static_path, f), index_col='basin_id', dtype={'basin_id': str})
                dfs.append(df)
        
        df = pd.concat(dfs, axis=1)
        df.columns.name = 'static_features'
        df.index.name = 'station_id'
        return df

    def gauge_stations(self)->pd.DataFrame:
        """
        reads the file estreams_gauging_stations.csv as dataframe
        """
        df = pd.read_csv(
            os.path.join(self.path, 'EStreams', 'streamflow_gauges', 'estreams_gauging_stations.csv'),
            index_col='basin_id',
            dtype={'basin_id': str}
            )
        return df

    def stations(self)->List[str]:
        """
        Returns a list of all station names
        """
        return self._stations

    def __stations(self)->List[str]:
        df = pd.read_csv(
            os.path.join(self.path, 'EStreams', 'streamflow_gauges', 'estreams_gauging_stations.csv'),
            usecols=['basin_id', 'lat'],
            dtype={'basin_id': str}
        )
        df.set_index('basin_id', inplace=True)
        return df.index.tolist()

    @property
    def countries(self)->List[str]:
        """
        returns the names of 39 countries covered by EStreams as list
        """
        return self.md.loc[:, 'gauge_country'].unique().tolist()

    def country_of_stn(self, stn:str)->str:
        """find the agency to which a station belongs """
        return self.md.loc[stn, 'gauge_country']

    def country_stations(self, country:str)->List[str]:
        """returns the station ids from a particular country"""
        return self.md[self.md['gauge_country'] == country].index.tolist()

    def stn_coords(self, stations:List[str] = "all", countries:List[str] = "all")->pd.DataFrame:
        """
        Returns the coordinates of one or more stations

        Returns
        -------
        pd.DataFrame
            a pandas dataframe of shape (stations, 2)
        
        Examples
        --------
        >>> from water_datasets import EStreams
        >>> dataset = EStreams()
        >>> dataset.stn_coords('IEEP0281')
        >>> dataset.stn_coords(['IEEP0281', 'IEEP0282'])
        >>> dataset.stn_coords(countries='IE')
        """
        stations = self._get_stations(countries, stations)

        df = pd.read_csv(
            os.path.join(self.path, 'EStreams', 'streamflow_gauges', 'estreams_gauging_stations.csv'),
            usecols=['basin_id', 'lat', 'lon'],
            dtype={'basin_id': str}
        )
        df.set_index('basin_id', inplace=True)
        df.rename(columns={'lon': 'long'}, inplace=True)
        return df.loc[stations]

    def _get_stations(self, countries:List[str] = "all", stations:List[str] = "all")->List[str]:
        if countries != "all" and stations != 'all':
            raise ValueError("Either provide countries or stations not both")
        
        if countries != "all":
            countries = check_attributes(countries, self.countries, 'countries')
            stations = self.md[self.md['gauge_country'].isin(countries)].index.tolist()
        else:
            stations = check_attributes(stations, self.stations(), 'stations')
        
        return stations

    def area(self, stations:List[str]="all", countries:List[str] = "all")->pd.Series:
        """area of catchments im km2"""

        stations = self._get_stations(countries, stations)
        return self.md.loc[stations, 'area']    

    def fetch_static_features(
            self,
            stations: Union[str, List[str]] = "all",
            features:Union[str, List[str]] = "all",
            countries:List[str] = "all",
    ) -> pd.DataFrame:
        """
        Returns static features of one or more stations.

        Parameters
        ----------
            stn_id : str
                name/id of station/stations of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Returns
        -------
        pd.DataFrame
            a pandas dataframe of shape (stations, features)

        Examples
        ---------
        >>> from water_datasets import EStreams
        >>> dataset = EStreams()
        get the names of stations
        >>> stns = dataset.stations()
        >>> len(stns)
            15047
        get all static data of all stations
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (15047, 153)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('IEEP0281')
        >>> static_data.shape
           (1, 153)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['slp_dg_mean', 'ele_mt_mean'])
        >>> static_data.shape
           (15047, 2)
        >>> data = dataset.fetch_static_features('IEEP0281', features=['slp_dg_mean', 'ele_mt_mean'])
        >>> data.shape
           (1, 2)
        >>> out = ds.fetch_static_features(countries='IE')
        >>> out.shape
        (464, 153
        """
        stations = self._get_stations(countries, stations)
        features = check_attributes(features, self.static_features, 'features')

        return self.static_data().loc[stations, features]

    def meteo_data_station(self, stn_id:str)->pd.DataFrame:
        """
        Returns the meteorological data of a station

        Returns
        -------
        pd.DataFrame
            a pandas dataframe of meteorological data of shape (time, 9)
        """
        df = pd.read_csv(
            os.path.join(self.path, 'EStreams', 'meteorology', f'estreams_meteorology_{stn_id}.csv'),
            index_col='date',
            parse_dates=True
        )
        df.columns.name = 'dynamic_features'
        df.index.name = 'time'
        return df

    def meteo_data(
            self, 
            stations:Union[str, List[str]]="all", 
            countries:Union[List[str], str]="all"
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
            return xr.open_dataset(nc_path)

        cpus = self.processes or get_cpus()-2
        stations = self.stations()
        meteo_vars = {}

        if self.verbosity:
            print(f"Fetching meteorological data of {len(stations)} stations using {cpus} cpus")

        start = time.time()
        with cf.ProcessPoolExecutor(cpus) as exe:  # takes ~500 secs with 110 cpus
            dfs = exe.map(self.meteo_data_station, stations)

        if self.verbosity:
            print(f"Fetching meteorological data took {time.time()-start:.2f} seconds")

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
            stations:List[str] = "all", 
            countries:List[str] = "all"
            )->pd.DataFrame:
        """
        Returns the hydro-climatic signatures of one or more stations

        Returns
        -------
        pd.DataFrame
            a pandas dataframe of hydro-climatic signatures of shape (stations, 31)
        """
        stations = self._get_stations(countries, stations)
        
        df = pd.read_csv(
            os.path.join(
                self.path, 
                'EStreams', 
                'hydroclimatic_signatures', 
                'estreams_hydrometeo_signatures.csv'),
            index_col='basin_id',
            dtype={'basin_id': str}
        )
        return df.loc[stations, :]

    def fetch_stn_dynamic_features(
            self,
            stn_id:str,
            dynamic_features = 'all',
    )->pd.DataFrame:
        """
        Fetches all or selected dynamic features of one station.

        Parameters
        ----------
            stn_id : str
                name/id of station of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                dynamic features are returned.

        Returns
        -------
        pd.DataFrame
            a pandas dataframe of shape (n, features) where n is the number of days

        Examples
        --------
        >>> from water_datasets import EStreams
        >>> camels = EStreams()
        >>> camels.fetch_stn_dynamic_features('IEEP0281').unstack()
        >>> camels.dynamic_features
        >>> camels.fetch_stn_dynamic_features('IEEP0281',
        ... features=['p_mean', 't_mean', 'pet_mean']).unstack()
        """
        features = check_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')

        return self.meteo_data_station(stn_id).loc[:, features]


    def fetch_dynamic_features(
            self,
            stations: Union[List[str], str] = "all",
            dynamic_features = 'all',
            st=None,
            en=None,
            as_dataframe=False,
            countries:Union[str, List[str]] = "all",
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
                if true, the returned data is pandas DataFrame otherwise it
                is xarray dataset

        Examples
        --------
            >>> from water_datasets import EStreams
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
        
        return self.meteo_data(stations)

class _EStreams(Camels):
    """
    Parent class for those datasets which use static and dynamic data from EStreams.
    """

    def __init__(
            self, 
            path:Union[str, os.PathLike] = None,
            estreams_path:Union[str, os.PathLike] = None,
            verbosity:int=1,
            **kwargs):
        super().__init__(path, verbosity=verbosity, **kwargs)

        if estreams_path is None:
            self.estreams_path = os.path.dirname(self.path)
        else:
            self.estreams_path = estreams_path
        
        self.estreams = EStreams(path=self.estreams_path, verbosity=verbosity)     

        self.md = self.estreams.md.loc[self.estreams.md['gauge_country']==self.country_name]
        self._stations = self.estreams.country_stations(self.country_name)
        self.boundary_file = self.estreams.boundary_file

        self.bndry_id_map = self.estreams.bndry_id_map.copy()

    @property
    def dynamic_features(self)->List[str]:
        return ['obs_q_cms'] + self.estreams.dynamic_features

    @property
    def static_features(self)->List[str]:
        return self.estreams.static_features

    @property
    def country_name(self)->str:
        return NotImplementedError

    @property
    def _coords_name(self)->List[str]:
        return ['lat', 'lon']

    @property
    def _area_name(self)->str:
        return 'area'

    @property
    def _q_name(self)->str:
        return "obs_q_cms"

    @property
    def start(self)->pd.Timestamp:
        return pd.Timestamp('1950-01-01')

    @property
    def end(self)->pd.Timestamp:
        return pd.Timestamp('2023-06-30')

    def _fetch_dynamic_features(
            self,
            stations: list,
            dynamic_features = 'all',
            st=None,
            en=None,
            as_dataframe=False,
            as_ts=False
    ):
        """Fetches dynamic features of station."""
        st, en = self._check_length(st, en)
        features = check_attributes(dynamic_features, self.dynamic_features.copy(), 'dynamic_features')

        daily_q = None

        if 'obs_q_cms' in features:
            daily_q = self.get_q(as_dataframe)
            if isinstance(daily_q, xr.Dataset):
                daily_q = daily_q.sel(time=slice(st, en))[stations]
            else:
                daily_q = daily_q.loc[st:en, stations]
            
            features.remove('obs_q_cms')

        if len(features) == 0:
            return daily_q

        #stations_ = [f"{stn}_{self.agency_name}" for stn in stations]
        data = self.estreams.fetch_dynamic_features(stations, features, st, en, as_dataframe)

        if daily_q is not None:
            if isinstance(daily_q, xr.Dataset):
                assert isinstance(data, xr.Dataset), "xarray dataset not supported"
                data = data.rename({stn:stn.split('_')[0] for stn in data.data_vars})

                # first create a new dimension in daily_q named dynamic_features
                daily_q = daily_q.expand_dims({'dynamic_features': ['obs_q_cms']})
                data = xr.concat([data, daily_q], dim='dynamic_features').sel(time=slice(st, en))
            else:
                # -1 because the data in .nc files hysets starts with 0
                data.rename(columns={stn:stn.split('_')[0] for stn in stations}, inplace=True)
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
            as_ts=False
    )->pd.DataFrame:
        """Fetches static features of station."""
        if self.verbosity>1:
            print('fetching static features')
        stations = check_attributes(station, self.stations(), 'stations')
        #stations_ = [f"{stn}_{self.agency_name}" for stn in stations]
        static_feats = self.estreams.fetch_static_features(stations, static_features).copy()
        #static_feats.index = [stn.split('_')[0] for stn in static_feats.index]
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

        Examples
        --------
        >>> from water_datasets import Arcticnet
        >>> dataset = Arcticnet()
        >>> stations = dataset.stations()
        >>> features = dataset.fetch_stations_features(stations)
        """
        stations = check_attributes(stations, self.stations(), 'stations')

        if dynamic_features is not None:

            dyn = self._fetch_dynamic_features(stations=stations,
                                               dynamic_features=dynamic_features,
                                               as_dataframe=as_dataframe,
                                               st=st,
                                               en=en,
                                               **kwargs
                                               )

            if static_features is not None:  # we want both static and dynamic
                to_return = {}
                static = self._fetch_static_features(station=stations,
                                                     static_features=static_features,
                                                     st=st,
                                                     en=en,
                                                     **kwargs
                                                     )
                to_return['static'] = static
                to_return['dynamic'] = dyn
            else:
                to_return = dyn

        elif static_features is not None:
            # we want only static
            to_return = self._fetch_static_features(
                station=stations,
                static_features=static_features,
                **kwargs
            )
        else:
            raise ValueError(f"""
            static features are {static_features} and dynamic features are also {dynamic_features}""")

        return to_return

    def fetch_static_features(
            self,
            stations: Union[str, List[str]] = "all",
            features:Union[str, List[str]] = "all",
            st=None,
            en=None,
            as_ts=False
    ) -> pd.DataFrame:
        """
        returns static atttributes of one or multiple stations

        Parameters
        ----------
            stations : str
                name/id of station of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.
            st :
            en :
            as_ts :

        Examples
        ---------
        >>> from water_datasets import Japan
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
        return self._fetch_static_features(stations, features, st, en, as_ts)
