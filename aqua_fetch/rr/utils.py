import os
import time
import random
import warnings
import concurrent.futures as cf
from typing import Union, List, Dict, Tuple

import numpy as np
import pandas as pd

from .._datasets import Datasets
from .._backend import netCDF4
from .._backend import fiona
from .._backend import xarray as xr, plt, easy_mpl, plt_Axes
from ..utils import check_attributes, get_cpus

from ._map import (
    catchment_area,
    gauge_latitude,
    gauge_longitude,
)

# directory separator
SEP = os.sep


def gb_message():
    link = "https://doi.org/10.5285/8344e4f3-d2ea-44f5-8afa-86d2987543a9"
    raise ValueError(f"Dwonlaoad the data from {link} and provide the directory "
                     f"path as dataset=Camels(data=data)")


class _RainfallRunoff(Datasets):
    """
    This is the parent class for invidual rainfall-runoff datasets like CAMELS-GB etc.
    This class is not meant to be for direct use. It is inherited by the child classes
    which are specific to a dataset like CAMELS-GB, CAMELS-AUS etc.
    This class first downloads the dataset if it is not already downloaded.
    Then the selected features for a selected catchment/station are fetched and provided to the
    user using the method `fetch`.

    Attributes
    -----------
    - path str/path: diretory of the dataset
    - dynamic_features list: tells which dynamic features are available in
      this dataset
    - static_features list: a list of static features.
    - static_attribute_categories list: tells which kinds of static features
      are present in this category.

    Methods
    ---------
    - stations : returns name/id of stations for which the data (dynamic features)
        exists as list of strings.
    - fetch : fetches all features (both static and dynamic type) of all
            station/gauge_ids or a speficified station. It can also be used to
            fetch all features of a number of stations ids either by providing
            their guage_id or  by just saying that we need data of 20 stations
            which will then be chosen randomly.
    - fetch_dynamic_features :
            fetches speficied dynamic features of one specified station. If the
            dynamic attribute is not specified, all dynamic features will be
            fetched for the specified station. If station is not specified, the
            specified dynamic features will be fetched for all stations.
    - fetch_static_features :
            works same as `fetch_dynamic_features` but for `static` features.
            Here if the `category` is not specified then static features of
            the specified station for all categories are returned.
        stations : returns list of stations
    """

    DATASETS = {
        'CAMELS_BR': {'url': "https://zenodo.org/record/3964745#.YA6rUxZS-Uk",
                      },
        'CAMELS-GB': {'url': gb_message},
    }

    def __init__(
            self,
            path: str = None,
            timestep: str = "D",
            to_netcdf: bool = True,
            overwrite: bool = False,
            verbosity: int = 1,
            **kwargs
    ):
        """

        parameters
        -----------
            path : str
                if provided and the directory exists, then the data will be read
                from this directory. If provided and the directory does not exist,
                then the data will be downloaded in this directory. If not provided,
                then the data will be downloaded in the default directory.
            timestep : str

            verbosity : int
                0: no message will be printed
            kwargs : dict
                Any other keyword arguments for the Datasets class
        """
        super(_RainfallRunoff, self).__init__(path=path, verbosity=verbosity, overwrite=overwrite, **kwargs)

        self.bndry_id_map = {}
        self.timestep = timestep

        if netCDF4 is None:
            if to_netcdf:
                msg = "netCDF4 module is not installed. Please install it to save data in netcdf format"
                warnings.warn(msg, UserWarning)
            to_netcdf = False
        self.to_netcdf = to_netcdf

    @property
    def dyn_map(self) -> Dict[str, str]:
        """A dictionary that maps dynamic features to their names in the dataset."""
        return {}

    @property
    def static_map(self) -> Dict[str, str]:
        """A dictionary that maps static features to their names in the dataset."""
        return {}

    @property
    def static_factors(self) -> Dict[str, str]:
        """A dictionary that maps static features to the factors with they needs
        to be multiplied to get the actual value"""
        return {}
        
    @property
    def dyn_factors(self) -> Dict[str, float]:
        return {}

    @property
    def boundary_id_map(self) -> str:
        """
        Name of the attribute in the boundary (shapefile/.gpkg) file that
        will be used to map the catchment/station id to the geometry of the
        catchment/station. This is used to create the boundary id map.
        if not given, then the first attribute in the boundary file will be used.
        """
        return None

    def mmd_to_cms(self, q_mmd: pd.Series) -> pd.Series:
        """converts discharge from mmd to cms"""
        area_m2 = self.area(q_mmd.name) * 1e6
        q_md = q_mmd * 0.001  # convert mm/day to m/day
        return q_md * area_m2.iloc[0] / 86400

    def cms_to_mmd(self, q_cms:pd.Series)->pd.Series:
        """convert streamflow from cms to mmd"""
        area_m2 = self.area(q_cms.name) * 1e6  # area in m2
        return ((q_cms * 86400)/area_m2.iloc[0]) * 1e3  # cms to m/day

    @staticmethod
    def mean_temp(tmin:pd.Series, tmax:pd.Series)->pd.Series:
        """calculates mean temperature from tmin and tmax"""
        assert len(tmin) == len(tmax), f"length of tmin {len(tmin)} and tmax {len(tmax)} must be same"
        return (tmin + tmax)/2

    def _create_boundary_id_map(self):

        if fiona is None:
            raise ModuleNotFoundError("fiona module is not installed. Please install it to use boundary file")

        # Dictionary to hold {CatchID: geometry}
        self.bndry_id_map = {}

        assert os.path.exists(self.boundary_file), \
            f"Boundary file {self.boundary_file} does not exist."

        with fiona.open(self.boundary_file, "r") as src:

            boundary_id_map = self.boundary_id_map
            if boundary_id_map is None:
                schema = src.schema
                properties = schema['properties']
                boundary_id_map = list(properties.keys())[0]  # use the first property as default
                if self.verbosity:
                    print(f"Using attribute '{boundary_id_map}' as default for boundary ID mapping.")
            
            for feature in src:

                if self.name in ['CAMELS_CH', 'CAMELS_IND', 'CABra']:
                    # from '2004.0' -> '2004' for CAMELS_CH
                    # from '03001' -> '3001' for CAMELS_IND
                    catch_id = str(int(feature["properties"][boundary_id_map]))
                elif self.name == 'CAMELS_LUX':
                    idx = int(feature["properties"][boundary_id_map])
                    if idx < 10:
                        catch_id = f"ID_{str(idx).zfill(2)}"
                    else:
                        catch_id = f"ID_{idx}"
                elif self.name == 'Simbi':
                    catch_id = feature['properties'][boundary_id_map]
                    catch_id = catch_id.split('-')[1]
                elif self.name == 'Caravan_DK':
                    catch_id = str(feature["properties"][boundary_id_map])
                    catch_id = str(catch_id).split('_')[1]
                else:
                    # since we are treating catchment/station id as string
                    catch_id = str(feature["properties"][boundary_id_map])
                geometry = feature["geometry"]

                self.bndry_id_map[catch_id] = geometry

        return self.bndry_id_map

    def stations(self) -> List[str]:
        """
        Names/ids of stations/catchment/gauges or whatever that would
        be used to index each station in the dataset. Since this is a method,
        it is called multiple times, it is better to cache the result
        and return the cached result instead of reading the data again and again
        The user is recommended to implement this method in the child class in a more efficient way.
        """
        return self._static_data().index.tolist()

    def _read_dynamic(
            self, 
            stations, 
            dynamic_features, 
            st:Union[str, pd.Timestamp] = None, 
            en:Union[str, pd.Timestamp] = None
            ) -> Dict[str, pd.DataFrame]:
        
        st, en = self._check_length(st, en)
        dyn_feats = check_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')
        stations = check_attributes(stations, self.stations(), 'stations')

        cpus = self.processes or min(get_cpus(), 16)
        start = time.time()

        if cpus == 1:
            dyn = {}
            for idx, stn in enumerate(stations):
            
                stn_df = self._read_stn_dyn(stn).loc[st:en]
                
                stn_df.columns.name = 'dynamic_features'
                stn_df.index.name = 'time'

                dyn[stn] = stn_df

                if self.verbosity and idx % 100 == 0:
                    print(f"Read {idx+1}/{len(stations)} stations.")
                elif self.verbosity>1 and idx % 50 == 0:
                    print(f"Read {idx+1}/{len(stations)} stations.")
                elif self.verbosity>2 and idx % 10 == 0:
                    print(f"Read {idx+1}/{len(stations)} stations.")
        else:
            with cf.ProcessPoolExecutor(cpus) as executor:
                results = executor.map(self._read_stn_dyn, stations)
            
            dyn = {}
            for stn, stn_df in zip(stations, results):
                stn_df.columns.name = 'dynamic_features'
                stn_df.index.name = 'time'

                dyn[stn] = stn_df.loc[st:en]

        total = time.time() -  start
        if self.verbosity:
            print(f"Read {len(dyn)} stations for {len(dyn_feats)} in {total:.2f} seconds with {cpus} cpus.")
    
        return dyn

    def _read_stn_dyn(self, stn: str) -> pd.DataFrame:
        """
        reads dynamic data of one station

        parameters
        ----------
            stn : str
                name/id of the station for which to read the dynamic data. This
                must be one of the station names returned by
                :meth:`stations`.
        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` with index as time and columns as dynamic features.
            The index is a :obj:`pandas.DatetimeIndex` and the columns are the names of
            dynamic features.
    
        """
        raise NotImplementedError(f"Must be implemented in the child class")

    def fetch_static_features(
            self,
            stations: Union[str, list] = "all",
            static_features: Union[str, list] = "all"
    ) -> pd.DataFrame:
        """Fetches all or selected static features of one or more stations.

        Parameters
        ----------
            stations : str/list
                name/id of station of which to extract the data
            static_features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame`

        Examples
        --------
        >>> from aqua_fetch import CAMELS_AUS
        >>> camels = CAMELS_AUS()
        >>> camels.fetch_static_features('912101A')
        >>> camels.static_features
        >>> camels.fetch_static_features('912101A',
        ... static_features=['elev_mean', 'relief', 'ksat', 'pop_mean'])
        for CAMELS_FR
        >>> from aqua_fetch import CAMELS_FR
        >>> dataset = CAMELS_FR()
        get the names of stations
        >>> stns = dataset.stations()
        >>> len(stns)
            654
        get all static data of all stations
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (472, 210)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('42600042')
        >>> static_data.shape
           (1, 210)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['slope_mean', 'aridity'])
        >>> static_data.shape
           (472, 2)
        >>> data = dataset.fetch_static_features('42600042', static_features=['slope_mean', 'aridity'])
        >>> data.shape
           (1, 2)
        """
        stations = check_attributes(stations, self.stations(), 'stations')
        features = check_attributes(static_features, self.static_features, 'static_features')
        df:pd.DataFrame = self._static_data()
        return df.loc[stations, features]

    def _static_data(self) -> pd.DataFrame:
        """returns all static data as DataFrame. The index of the DataFrame
        is the station ids and the columns are the static features.
        This method must be implemented in the child class.

        Returns
        -------
        pd.DataFrame
        a :obj:`pandas.DataFrame` with index as station ids and columns as static features.
        The index is a :obj:`pandas.Index` and the columns are the names of
        static features.
        """
        raise NotImplementedError(f"Must be implemented in the child class")

    @property
    def start(self) -> pd.Timestamp:  # start of data
        return pd.Timestamp("1800-01-01")

    @property
    def end(self) -> pd.Timestamp:  
        """end of data"""
        return pd.Timestamp.today().strftime("%Y-%m-%d")

    @property
    def static_features(self) -> List[str]:
        """
        Returns a list of static features that are available in the dataset.
        Since this is a method is called multiple times, it is better to cache the result
        and return the cached result instead of reading the data again and again
        or the user implementing this method in the child class in a more efficient way.

        Returns
        -------
        List[str]
            a list of static features that are available in the dataset.
            The names of the features are the same as the names used in the
            dataset. The names can be used to fetch the data using
            :meth:`fetch_static_features`.
        """
        return self._static_data().columns.tolist()

    @property
    def dynamic_features(self) -> List[str]:
        """
        Returns a list of dynamic features that are available in the dataset.
        Since this is a method is called multiple times, it is better to cache the result
        and return the cached result instead of reading the data again and again
        or the user implementing this method in the child class in a more efficient way.

        Returns
        -------
        List[str]
            a list of dynamic features that are available in the dataset.
            The names of the features are the same as the names used in the
            dataset. The names can be used to fetch the data using
            :meth:`fetch_dynamic_features`.
        """
        return self._read_stn_dyn(self.stations()[0]).columns.tolist()

    @property
    def _area_name(self) -> str:
        """name of feature from static_features to be used as area"""
        raise NotImplementedError

    @property
    def _mmd_feature_name(self) -> str:
        return None

    @property
    def _q_name(self) -> str:
        return None

    @property
    def _coords_name(self) -> List[str]:
        """
        names of features from static_features to be used as station
        coordinates (lat, long)
        """
        raise NotImplementedError

    def area(
            self,
            stations: Union[str, List[str]] = 'all'
    ) -> pd.Series:
        """
        Returns area (Km2) of all/selected catchments as :obj:`pandas.Series`

        parameters
        ----------
        stations : str/list (default=None)
            name/names of stations. Default is ``all``, which will return
            area of all stations

        Returns
        --------
        pd.Series
            a :obj:`pandas.Series` whose indices are catchment ids and values
            are areas of corresponding catchments.

        Examples
        ---------
        >>> from aqua_fetch import CAMELS_CH
        >>> dataset = CAMELS_CH()
        >>> dataset.area()  # returns area of all stations
        >>> dataset.area('2004')  # returns area of station whose id is 2004
        >>> dataset.area(['2004', '6004'])  # returns area of two stations
        """

        stations = check_attributes(stations, self.stations(), 'stations')

        df = self.fetch_static_features(static_features=[catchment_area()])
        #df.columns = [catchment_area()]

        return df.loc[stations, catchment_area()]

    def _check_length(self, st, en):
        if st is None:
            st = self.start
        else:
            st = pd.Timestamp(st)
        if en is None:
            en = self.end
        else:
            en = pd.Timestamp(en)
        return st, en

    @property
    def camels_dir(self):
        """Directory where all camels datasets will be saved. This will under
         datasets directory"""
        return os.path.join(self.base_ds_dir, "CAMELS")

    def fetch(self,
              stations: Union[str, list, int, float] = "all",
              dynamic_features: Union[List[str], str, None] = 'all',
              static_features: Union[str, List[str], None] = None,
              st: Union[None, str] = None,
              en: Union[None, str] = None,
              as_dataframe: bool = False,
              **kwargs
              ) -> Tuple[pd.DataFrame, Union[pd.DataFrame, "Dataset"]]:
        """
        Fetches the features of one or more stations.

        Parameters
        ----------
            stations :
                It can have following values:
                    - int : number of (randomly selected) stations to fetch
                    - float : fraction of (randomly selected) stations to fetch
                    - str : name/id of station to fetch. However, if ``all`` is
                        provided, then all stations will be fetched.
                    - list : list of names/ids of stations to fetch
            dynamic_features : If not None, then it is the features to be
                fetched. If None, then all available features are fetched
            static_features : list of static features to be fetches. None
                means no static attribute will be fetched.
            st : starting date of data to be returned. If None, the data will be
                returned from where it is available.
            en : end date of data to be returned. If None, then the data will be
                returned till the date data is available.
            as_dataframe : whether to return dynamic features as :obj:`pandas.DataFrame` 
                or as :obj:`xarray.Dataset`.
            kwargs : keyword arguments to read the files

        Returns
        -------
        tuple
            A tuple of static and dynamic features. Static features are always
            returned as pandas DataFrame with shape (stations, staticfeatures).
            The index of static features is the station/gauge ids while the columns 
            are the static features. Dynamic features are returned as either
            xarray Dataset or pandas DataFrame depending upon whether `as_dataframe`
            is True or False and whether the xarray module is installed or not.
            If dynamic features are xarray Dataset, then it consists of `data_vars`
            equal to the number of stations and `time` adn `dynamic_features` as
            dimensions. If dynamic features are returned as pandas DataFrame, then
            the first index is `time` and the second index is `dynamic_features`.

        Examples
        --------
        >>> from aqua_fetch import CAMELS_AUS
        >>> dataset = CAMELS_AUS()
        >>> # get data of 10% of stations
        >>> df = dataset.fetch(stations=0.1, as_dataframe=True)  # returns a multiindex dataframe
        ...  # fetch data of 5 (randomly selected) stations
        >>> five_random_stn_data = dataset.fetch(stations=5, as_dataframe=True)
        ... # fetch data of 3 selected stations
        >>> three_selec_stn_data = dataset.fetch(stations=['912101A','912105A','915011A'], as_dataframe=True)
        ... # fetch data of a single stations
        >>> single_stn_data = dataset.fetch(stations='318076', as_dataframe=True)
        ... # get both static and dynamic features as dictionary
        >>> static, dynamic = dataset.fetch(1, static_features="all", as_dataframe=True)  # -> dict
        >>> dynamic
        ... # get only selected dynamic features
        >>> sel_dyn_features = dataset.fetch(stations='318076',
        ...     dynamic_features=['q_mmd_obs', 'solrad_wm2_silo'], as_dataframe=True)
        ... # fetch data between selected periods
        >>> data = dataset.fetch(stations='318076', st="20010101", en="20101231", as_dataframe=True)

        """
        if isinstance(stations, int):
            # the user has asked to randomly provide data for some specified number of stations
            stations = random.sample(self.stations(), stations)
        elif isinstance(stations, list):
            pass
        elif isinstance(stations, str):
            if stations == 'all':
                stations = self.stations()
            else:
                stations = [stations]
        elif isinstance(stations, float):
            num_stations = int(len(self.stations()) * stations)
            stations = random.sample(self.stations(), num_stations)
        elif stations is None:
            # fetch for all stations
            stations = self.stations()
        else:
            raise TypeError(f"Unknown value provided for stations {stations}")

        return self.fetch_stations_features(
            stations,
            dynamic_features,
            static_features,
            st=st,
            en=en,
            as_dataframe=as_dataframe,
            **kwargs
        )

    def _maybe_to_netcdf(self, fname: str):

        # todo : we should save the dynamic data with default names of dynamic features
        # and then convert them to the standard names using the dyn_map
        # otherwise everytime we change dyn_map, we will have to convert the data again
        # and more importantly, all the users who used a previous version of the dataset
        # will have to download the data again, which is not good
        self.dyn_fname = os.path.join(self.path, f'{fname}.nc')
        if self.to_netcdf:
            if not os.path.exists(self.dyn_fname) or self.overwrite:
                # saving all the data in netCDF file using xarray
                print(f'converting data to netcdf format for faster io operations')
                _, data = self.fetch(static_features=None)

                data.to_netcdf(self.dyn_fname)
            else:
                if self.verbosity:
                    print(f"dynamic data already exists in {self.dyn_fname}. "
                          f"To overwrite, set `overwrite=True`")
        return

    def fetch_stations_features(
            self,
            stations: Union[str, List[str]],
            dynamic_features: Union[str, List[str]] = 'all',
            static_features: Union[str, List[str]] = None,
            st: Union[str, pd.Timestamp] = None,
            en: Union[str, pd.Timestamp] = None,
            as_dataframe: bool = False,
            **kwargs
              ) -> Tuple[pd.DataFrame, Union[pd.DataFrame, "Dataset"]]:
        """
        Reads features of more than one stations.

        parameters
        ----------
        stations :
            list of stations for which data is to be fetched.
        dynamic_features :
            list of dynamic features to be fetched.
            if ``all``, then all dynamic features will be fetched.
        static_features : list of static features to be fetched.
            If ``all``, then all static features will be fetched. If None,
            `then no static attribute will be fetched.
        st :
            start of data to be fetched.
        en :
            end of data to be fetched.
        as_dataframe :
            whether to return the dynamic data as pandas dataframe. default
            is :obj:`xarray.Dataset` object
        kwargs dict:
            additional keyword arguments

        Returns
        -------
        tuple
            A tuple of static and dynamic features. Static features are always
            returned as :obj:`pandas.DataFrame` with shape (stations, staticfeatures).
            The index of static features is the station/gauge ids while the columns 
            are the static features. Dynamic features are returned as either
            :obj:`xarray.Dataset` or :obj:`pandas.DataFrame` depending upon whether `as_dataframe`
            is True or False and whether the xarray module is installed or not.
            If dynamic features are xarray Dataset, then it consists of `data_vars`
            equal to the number of stations and `time` adn `dynamic_features` as
            dimensions. If dynamic features are returned as pandas DataFrame, then
            the first index is `time` and the second index is `dynamic_features`.

        Raises:
            ValueError, if both dynamic_features and static_features are None

        Examples
        --------
        >>> from aqua_fetch import CAMELS_AUS
        >>> dataset = CAMELS_AUS()
        ... # find out station ids
        >>> dataset.stations()
        ... # get data of selected stations as xarray Dataset
        >>> dataset.fetch_stations_features(['912101A', '912105A', '915011A'])
        ... # get data of selected stations as pandas DataFrame
        >>> dataset.fetch_stations_features(['912101A', '912105A', '915011A'],
        ...  as_dataframe=True)
        ... # get both dynamic and static features of selected stations
        >>> dataset.fetch_stations_features(['912101A', '912105A', '915011A'],
        ... dynamic_features=['q_mmd_obs', 'airtemp_C_mean_silo'], static_features=['elev_mean'])
        """

        if xr is None:
            if not as_dataframe:
                warnings.warn("xarray module is not installed so as_dataframe will have no effect. "
                              "Dynamic features will be returned as pandas DataFrame")
                as_dataframe = True

        st, en = self._check_length(st, en)
        static, dynamic = None, None

        stations = check_attributes(stations, self.stations(), 'stations')

        if dynamic_features is not None:

            dynamic_features = check_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')

            if netCDF4 is None or not os.path.exists(self.dyn_fname):
                # read from csv files
                # following code will run only once when fetch is called inside init method
                dyn = self._read_dynamic(stations, dynamic_features, st=st, en=en)

            else:
                dyn = xr.open_dataset(self.dyn_fname)  # daataset
                dyn = dyn[stations].sel(dynamic_features=dynamic_features, time=slice(st, en))
                if as_dataframe:
                    dyn = dyn.to_dataframe(['time', 'dynamic_features'])

            if static_features is not None:
                static = self.fetch_static_features(stations, static_features)
                dynamic = _handle_dynamic(dyn, as_dataframe)
            else:
                dynamic = _handle_dynamic(dyn, as_dataframe)

        elif static_features is not None:

            return self.fetch_static_features(stations, static_features), dynamic

        else:
            raise ValueError(f"static features are {static_features} and dynamic features are {dynamic_features}")

        return static, dynamic

    def fetch_dynamic_features(
            self,
            station: str,
            dynamic_features='all',
            st=None,
            en=None,
            as_dataframe=False
    )-> Union[pd.DataFrame, "Dataset"]:
        """Fetches all or selected dynamic features of one station.

        Parameters
        ----------
            station : str
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
                is :obj:`xarray.Dataset`
        
        Returns
        -------
        pd.DataFrame/xr.Dataset
            a pandas dataframe or xarray dataset of dynamic features
            If as_dataframe is True, then the returned data is a pandas
            DataFrame with multiindex. The first index is `time` and the second
            index is `dynamic_features`. If as_dataframe is False, and xarray
            module is installed, then the returned data is xarray dataset with
            `data_vars` equal to the number of stations and `time` and `dynamic_features`
            as dimensions.

        Examples
        --------
        >>> from aqua_fetch import CAMELS_AUS
        >>> camels = CAMELS_AUS()
        >>> camels.fetch_dynamic_features('912101A', as_dataframe=True).unstack()
        >>> camels.dynamic_features
        >>> camels.fetch_dynamic_features('912101A',
        ... features=['airtemp_C_awap_max', 'vp_hpa_awap', 'q_cms_obs'],
        ... as_dataframe=True).unstack()
        """

        assert isinstance(station, str), f"station id must be string is is of type {type(station)}"
        station = [station]
        return self.fetch_stations_features(
            stations=station,
            dynamic_features=dynamic_features,
            static_features=None,
            st=st,
            en=en,
            as_dataframe=as_dataframe
        )[1]

    def fetch_station_features(
            self,
            station: str,
            dynamic_features: Union[str, list, None] = 'all',
            static_features: Union[str, list, None] = None,
            st: Union[str, None] = None,
            en: Union[str, None] = None,
            **kwargs
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetches features for one station.

        Parameters
        -----------
            station :
                station id/gauge id for which the data is to be fetched.
            dynamic_features : str/list, optional
                names of dynamic features/attributes to fetch
            static_features :
                names of static features/attributes to be fetches
            st : str,optional
                starting point from which the data to be fetched. By default,
                the data will be fetched from where it is available.
            en : str, optional
                end point of data to be fetched. By default the dat will be fetched

        Returns
        -------
        tuple
            A tuple of static and dynamic features, both as :obj:`pandas.DataFrame`.
            The dataframe of static features will be of single row while the dynamic
            features will be of shape (time, dynamic features).

        Examples
        --------
        >>> from aqua_fetch import CAMELS_AUS
        >>> dataset = CAMELS_AUS()
        >>> dataset.fetch_station_features('912101A')

        """
        st, en = self._check_length(st, en)

        static, dynamic = None, None

        if dynamic_features:
            dynamic = self.fetch_dynamic_features(station, dynamic_features, st=st,
                                                  en=en, **kwargs)
            
            if xr is not None and isinstance(dynamic, xr.Dataset):
                dynamic = dynamic[station].to_pandas()

            if static_features is not None:
                static = self.fetch_static_features(station, static_features)

        elif static_features is not None:
            static = self.fetch_static_features(station, static_features)

        return static, dynamic

    def plot_stations(
            self,
            stations: List[str] = 'all',
            marker='.',
            color:str=None,
            ax: plt_Axes = None,
            show: bool = True,
            **kwargs
    ) -> plt_Axes:
        """
        plots coordinates of stations

        Parameters
        ----------
        stations :
            name/names of stations. If not given, all stations will be plotted
        marker :
            marker to use.
        color : str, optional
            name of static feature to use as color. 
        ax : plt.Axes
            matplotlib axes to draw the plot. If not given, then
            new axes will be created.
        show : bool
        **kwargs

        Returns
        -------
        plt.Axes

        Examples
        --------
        >>> from aqua_fetch import CAMELS_AUS
        >>> dataset = CAMELS_AUS()
        >>> dataset.plot_stations()
        >>> dataset.plot_stations(['1', '2', '3'])
        >>> dataset.plot_stations(marker='o', ms=0.3)
        >>> ax = dataset.plot_stations(marker='o', ms=0.3, show=False)
        >>> ax.set_title("Stations")
        >>> plt.show()
        using area as color
        >>> ds.plot_stations(color='area_km2')

        """
        from easy_mpl.utils import add_cbar, map_array_to_cmap

        xy = self.stn_coords(stations)

        _kws = dict(
            ax_kws=dict(xlabel="Longitude",
            ylabel="Latitude",
            title=f"{self.name} Stations (n={len(xy)})",
        ))

        _kws.update(kwargs)

        if color is not None:
            assert color in self.static_features, f"color {color} is not in static features {self.static_features}"
            c = self.fetch_static_features(stations, color)
            c = c.astype('float32')
            ul = round(c[color].quantile([0.99]).item(), 2)

            if self.verbosity > 0:
                print(f"Setting upper limit to {ul} for color scale")

            c[c>ul] = ul

            colorbar = _kws.pop('colorbar', True)
            _kws['cmap'] = _kws.get('cmap', 'viridis')

            ax, _= easy_mpl.scatter(
                xy.loc[:, 'long'].values,
                    xy.loc[:, 'lat'].values,
                    ax=ax,
                    c=c.values.reshape(-1,),
                    show=False, 
                    **_kws)
            
            if colorbar:
                c, mapper = map_array_to_cmap(c.values.reshape(-1,), _kws['cmap'])
                add_cbar(ax, mappable=mapper, pad=0.3,
                     border=False,
                     title=color, title_kws=dict(fontsize=12))
        else:
            ax = easy_mpl.plot(xy.loc[:, 'long'].values,
                           xy.loc[:, 'lat'].values,
                           marker, ax=ax, 
                           show=False, 
                           **_kws)

        if show:
            plt.show()

        return ax

    def q_mmd(
            self,
            stations: Union[str, List[str]] = "all"
    ) -> pd.DataFrame:
        """
        returns streamflow in the units of milimeter per day. This is obtained
        by diving ``q``/area

        parameters
        ----------
        stations : str/list
            name/names of stations. Default is ``all``, which will return
            area of all stations

        Returns
        --------
        pd.DataFrame
            a :obj:`pandas.DataFrame` whose indices are time-steps and columns
            are catchment/station ids.

        """

        stations = check_attributes(stations, self.stations(), 'stations')

        if self._mmd_feature_name is None:
            _, q = self.fetch_stations_features(
                stations,
                dynamic_features="q_cms_obs", 
                as_dataframe=True)
            q.index = q.index.get_level_values(0)
            area_m2 = self.area(stations) * 1e6  # area in m2
            q = (q / area_m2) * 86400  # cms to m/day
            return q * 1e3  # to mm/day

        else:

            _, q = self.fetch_stations_features(
                stations,
                dynamic_features=self._mmd_feature_name,
                as_dataframe=True)
            q.index = q.index.get_level_values(0)
            return q

    def stn_coords(
            self,
            stations: Union[str, List[str]] = 'all'
    ) -> pd.DataFrame:
        """
        returns coordinates of stations as DataFrame
        with ``long`` and ``lat`` as columns.

        Parameters
        ----------
        stations :
            name/names of stations. If not given, coordinates
            of all stations will be returned.

        Returns
        -------
        pd.DataFrame
            :obj:`pandas.DataFrame` with ``long`` and ``lat`` columns.
            The length of dataframe will be equal to number of stations
            wholse coordinates are to be fetched.

        Examples
        --------
        >>> from aqua_fetch import CAMELS_CH
        >>> dataset = CAMELS_CH()
        >>> dataset.stn_coords() # returns coordinates of all stations
        >>> dataset.stn_coords('2004')  # returns coordinates of station whose id is 2004
        >>> dataset.stn_coords(['2004', '6004'])  # returns coordinates of two stations

        >>> from aqua_fetch import CAMELS_AUS
        >>> dataset = CAMELS_AUS()
        >>> dataset.stn_coords() # returns coordinates of all stations
        >>> dataset.stn_coords('912101A')  # returns coordinates of station whose id is 912101A
        >>> dataset.stn_coords(['G0050115', '912101A'])  # returns coordinates of two stations

        """
        df = self.fetch_static_features(static_features=[gauge_latitude(), gauge_longitude()])
        #df.columns = ['lat', 'long']
        stations = check_attributes(stations, self.stations(), 'stations')

        df = df.loc[stations, :].astype(self.fp)

        return self.transform_stn_coords(df)

    def transform_stn_coords(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        transforms coordinates from geographic to projected

        must be implemented in base classes
        """
        return df


    def transform_coords(self, xyz: np.ndarray) -> np.ndarray:
        """
        transforms coordinates from projected to geographic

        must be implemented in base classes
        """
        return xyz

    def get_boundary(
            self,
            catchment_id: str,
    )-> List[np.ndarray]:
        """
        returns boundary of a catchment in a required format

        Parameters
        ----------
        catchment_id : str
            name/id of catchment
        as_type : str
            'numpy' or 'geopandas'

        Examples
        --------
        >>> from aqua_fetch import CAMELS_SE
        >>> dataset = CAMELS_SE()
        >>> dataset.get_boundary(dataset.stations()[0])
        """

        # todo : by default it should return fiona Geometry
        # we can do the conversion to numpy when plotting

        assert isinstance(catchment_id, str), f"catchment_id must be string but is of type {type(catchment_id)}"

        # todo : when we repeatedly call get_boundary, we should not create the 
        # boundary_id_map for all catchments again
        if self.name in ['Thailand', 'Japan', 'Arcticnet', 'Spain']:
            bndry_id_map = self.gsha._create_boundary_id_map()
        # elif self.name in ['HYSETS']:
        #     bndry_id_map = self._create_boundary_id_map()
        elif self.name in ['USGS']:
            bndry_id_map = self.hysets._create_boundary_id_map()            
        else:
            bndry_id_map = self._create_boundary_id_map()

        def make_polygon_2d(polygon):
            """
            converts a 3D polygon to 2D polygon by removing the z coordinate
            """
            if polygon.ndim == 3:
                assert polygon.shape[0] == 1, "Only one polygon is expected for a catchment"
                polygon = polygon[0]
            
            if polygon.shape[1] == 3:
                # if the polygon has 3 coordinates, then we will remove the z coordinate
                assert polygon[:, -1].sum() == 0, "Z coordinate is not zero for the polygon"
                polygon = polygon[:, :-1]

            return polygon

        if self.name in ['HYSETS']:
            catchment_id = self.WatershedID_OfficialID_map[catchment_id]
        # elif self.name == 'USGS':
        #     catchment_id = self.hysets.OfficialID_WatershedID_map[catchment_id]
        elif self.name == 'Thailand':
            catchment_id = catchment_id.replace('.', '_')
        
        if self.name in ['Thailand', 'Japan', 'Arcticnet', 'Spain']:
            catchment_id = f"{catchment_id}_{self.agency_name}"

        rings = []
        geometry = bndry_id_map[catchment_id]
        if geometry.type == 'MultiPolygon':
            for polygon in geometry.coordinates:
                if len(polygon) == 1:
                    polygon = np.array(polygon)
                    rings.append(make_polygon_2d(polygon))
                else:
                    for p in polygon:  # for GRDC_1159100, # there are multiple polygons
                        p = np.array(p)
                        rings.append(make_polygon_2d(p))
        else:
            if len(geometry.coordinates) > 1:
                for polygon in geometry.coordinates:
                    polygon = np.array(polygon)
                    rings.append(make_polygon_2d(polygon))
            else:
                polygon = np.array(geometry.coordinates)
                rings.append(make_polygon_2d(polygon))

        rings = self.transform_coords(rings)

        return rings

    def plot_catchment(
            self,
            catchment_id: str,
            show_outlet:bool = False,
            ax: plt_Axes = None,
            show: bool = True,
            **kwargs
    ):
        """
        plots catchment boundaries

        Parameters
        ----------
        catchment_id : str
            name/id of catchment to plot
        show_outlet : bool, optional (default=False)
            if True, then outlet of the catchment will be plotted as a red dot
        ax : plt.Axes
            matplotlib axes to draw the plot. If not given, then
            new axes will be created.
        show : bool
        **kwargs

        Returns
        -------
        plt.Axes

        Examples
        --------
        >>> from aqua_fetch import CAMELS_AUS
        >>> dataset = CAMELS_AUS()
        >>> dataset.plot_catchment('912101A')
        >>> dataset.plot_catchment('912101A', marker='o', ms=0.3)
        >>> ax = dataset.plot_catchment('912101A', marker='o', ms=0.3, show=False)
        >>> ax.set_title("Catchment Boundary")
        >>> plt.show()
        # show the outlet as well
        >>> CAMELS_AUS.plot_catchment('912101A', show_outlet=True)

        """
        rings = self.get_boundary(catchment_id)

        _kws = dict(
            ax_kws=dict(xlabel="Longitude", ylabel="Latitude")
        )

        _kws.update(kwargs)

        for ring in rings:
            ax = easy_mpl.plot(ring[:, 0], ring[:, 1],
                                show=False, ax=ax, **_kws)
        
        if show_outlet:
            coords = self.stn_coords(catchment_id)
            ax.scatter(coords['long'], coords['lat'], marker='o', 
                       color='red', 
                       s=10, 
                       label='Outlet')

        if show:
            plt.show()
        return ax


def _handle_dynamic(dyn, as_dataframe: bool):
    if as_dataframe and isinstance(dyn, dict) and isinstance(list(dyn.values())[0], pd.DataFrame):
        # if the dyn is a dictionary of station, DataFames pairs, and each DataFrame's index is 'time'
        # 'columns' are 'dynamic_features', we will return a MultiIndex
        # dataframe instead of a dictionary whose first index is time and second index is dynamic_feature
        # Step 1: Convert each DataFrame to long format (melt)
        long_dfs = []
        for station, df in dyn.items():
            long_df = df.reset_index().melt(id_vars='time', var_name='dynamic_feature', value_name='value')
            long_df['station'] = station
            long_dfs.append(long_df)
        # Step 2: Concatenate all long DataFrames into one
        combined_df = pd.concat(long_dfs)
        # Step 3: Pivot to get desired format: rows = (time, dynamic_feature), columns = stations
        dyn = combined_df.pivot(index=['time', 'dynamic_feature'], columns='station', values='value')
    elif isinstance(dyn, dict) and isinstance(list(dyn.values())[0], pd.DataFrame):
        # dyn is a dictionary of key, DataFames and we have to return xr Dataset
        # dyn = pd.concat(dyn, axis=0, keys=dyn.keys())
        dyn = xr.Dataset(dyn)
    return dyn
