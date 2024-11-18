
import os
import random
import warnings
from typing import Union, List, Dict

import numpy as np
import pandas as pd

from .._datasets import Datasets
from .._backend import netCDF4
from .._backend import shapefile
from .._backend import xarray as xr, plt, easy_mpl, plt_Axes
from ..utils import check_attributes


# directory separator
SEP = os.sep


def gb_message():
    link = "https://doi.org/10.5285/8344e4f3-d2ea-44f5-8afa-86d2987543a9"
    raise ValueError(f"Dwonlaoad the data from {link} and provide the directory "
                     f"path as dataset=Camels(data=data)")


class Camels(Datasets):
    """
    Get CAMELS dataset.
    This class first downloads the CAMELS dataset if it is not already downloaded.
    Then the selected features for a selected id are fetched and provided to the
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
            path:str = None,
            timestep:str = "D",
            id_idx_in_bndry_shape:int = None,
            to_netcdf:bool = True,
            overwrite:bool = False,
            verbosity:int = 1,
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
        super(Camels, self).__init__(path=path, verbosity=verbosity, overwrite=overwrite, **kwargs)

        self.bndry_id_map = {}
        self.timestep = timestep

        if netCDF4 is None:
            if to_netcdf:
                warnings.warn("netCDF4 module is not installed. Please install it to save data in netcdf format")
            to_netcdf = False
        self.to_netcdf = to_netcdf

    @property
    def dyn_map(self)->Dict[str, str]:
        return {}

    @property
    def dyn_factors(self)->Dict[str, float]:
        return {}

    def _create_boundary_id_map(self, boundary_file, id_idx_in_bndry_shape):

        if boundary_file is None:
            return
        
        if shapefile is None:
            warnings.warn("shapefile module is not installed. Please install it to use boundary file")
            return

        from shapefile import Reader

        if self.verbosity>1:
            print(f"loading boundary file {boundary_file}")

        assert os.path.exists(boundary_file), f"{boundary_file} does not exist"
        bndry_sf = Reader(boundary_file)

        # shapefile of chille contains spanish characters which can not be
        # decoded with utf-8
        if os.path.basename(bndry_sf.shapeName) in [
            'catchments_camels_cl_v1_3',
            "WKMSBSN",
            'estreams_catchments',
            'CAMELS_DE_catchments',
        ]:
            bndry_sf.encoding = 'ISO-8859-1'

        self.bndry_id_map = self._get_map(bndry_sf,
                                        id_index=id_idx_in_bndry_shape,
                                        name="bndry_shape")
    
        bndry_sf.close()
        return

    @staticmethod
    def _get_map(sf_reader, id_index=None, name:str='')->Dict[str, int]:

        fieldnames = [f[0] for f in sf_reader.fields[1:]]

        if len(fieldnames) > 1:
            if id_index is None:
                raise ValueError(f"""
                more than one fileds are present in {name} shapefile 
                i.e: {fieldnames}. 
                Please provide a value for id_idx_in_{name} that must be
                less than {len(fieldnames)}
                """)
        else:
            id_index = 0

        if os.path.basename(sf_reader.shapeName) in ["CAMELS_CH_catchments"]:
            catch_ids_map = {
            str(int(rec[id_index])): idx for idx, rec in enumerate(sf_reader.iterRecords())
        }
        else:
            catch_ids_map = {
            str(rec[id_index]): idx for idx, rec in enumerate(sf_reader.iterRecords())
        }

        return catch_ids_map
            
    def stations(self)->List[str]:
        raise NotImplementedError

    def _read_dynamic_from_csv(self, stations, dynamic_features, st=None,
                               en=None)->dict:
        raise NotImplementedError

    def fetch_static_features(
            self,
            stn_id: Union[str, list] = None,
            features: Union[str, list] = None
    )->pd.DataFrame:
        """Fetches all or selected static features of one or more stations.

        Parameters
        ----------
            stn_id : str
                name/id of station of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Returns
        -------
        pd.DataFrame
            a pandas dataframe

        Examples
        --------
            >>> from water_datasets import CAMELS_AUS
            >>> camels = CAMELS_AUS()
            >>> camels.fetch_static_features('224214A')
            >>> camels.static_features
            >>> camels.fetch_static_features('224214A',
            ... features=['elev_mean', 'relief', 'ksat', 'pop_mean'])
        """
        raise NotImplementedError

    @property
    def start(self):  # start of data
        raise NotImplementedError

    @property
    def end(self):  # end of data
        raise NotImplementedError

    @property
    def dynamic_features(self) -> list:
        raise NotImplementedError

    @property
    def _area_name(self)->str:
        """name of feature from static_features to be used as area"""
        raise NotImplementedError

    @property
    def _mmd_feature_name(self)->str:
        return None

    @property
    def _q_name(self)->str:
        return None
    
    @property
    def _coords_name(self)->List[str]:
        """
        names of features from static_features to be used as station 
        coordinates (lat, long)
        """
        raise NotImplementedError

    def area(
            self,
            stations: Union[str, List[str]] = 'all'
    ) ->pd.Series:
        """
        Returns area (Km2) of all/selected catchments as pandas series

        parameters
        ----------
        stations : str/list (default=None)
            name/names of stations. Default is ``all``, which will return
            area of all stations

        Returns
        --------
        pd.Series
            a pandas series whose indices are catchment ids and values
            are areas of corresponding catchments.

        Examples
        ---------
        >>> from water_datasets import CAMELS_CH
        >>> dataset = CAMELS_CH()
        >>> dataset.area()  # returns area of all stations
        >>> dataset.area('2004')  # returns area of station whose id is 2004
        >>> dataset.area(['2004', '6004'])  # returns area of two stations
        """

        stations = check_attributes(stations, self.stations(), 'stations')

        df = self.fetch_static_features(features=[self._area_name])
        df.columns = ['area']

        return df.loc[stations, 'area']

    def _check_length(self, st, en):
        if st is None:
            st = self.start
        if en is None:
            en = self.end
        return st, en

    def to_ts(self, static, st, en, as_ts=False, freq='D'):

        st, en = self._check_length(st, en)

        if as_ts:
            idx = pd.date_range(st, en, freq=freq)
            static = pd.DataFrame(np.repeat(static.values, len(idx), axis=0), index=idx,
                                  columns=static.columns)
            return static
        else:
            return static

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
              ) -> Union[dict, pd.DataFrame]:
        """
        Fetches the features of one or more stations.

        Arguments:
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
            as_dataframe : whether to return dynamic features as pandas
                dataframe or as xarray dataset.
            kwargs : keyword arguments to read the files

        returns:
            If both static  and dynamic features are obtained then it returns a
            dictionary whose keys are station/gauge_ids and values are the
            features and dataframes.
            Otherwise either dynamic or static features are returned.

        Examples
        --------
        >>> from water_datasets import CAMELS_AUS
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
        >>> data = dataset.fetch(1, static_features="all", as_dataframe=True)  # -> dict
        >>> data['dynamic']
        ... # get only selected dynamic features
        >>> sel_dyn_features = dataset.fetch(stations='318076',
        ...     dynamic_features=['streamflow_MLd', 'solarrad_AWAP'], as_dataframe=True)
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
        self.dyn_fname = os.path.join(self.path, f'{fname}.nc')
        if self.to_netcdf:
            if not os.path.exists(self.dyn_fname) or self.overwrite:
                # saving all the data in netCDF file using xarray
                print(f'converting data to netcdf format for faster io operations')
                data = self.fetch(static_features=None)

                data.to_netcdf(self.dyn_fname)
        return

    def fetch_stations_features(
            self,
            stations: Union[str, List[str]],
            dynamic_features:Union[str, List[str]] = 'all',
            static_features:Union[str, List[str]] = None,
            st: Union[str, pd.Timestamp] = None,
            en:Union[str, pd.Timestamp] = None,
            as_dataframe: bool = False,
            **kwargs
    ):
        """Reads features of more than one stations.

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
            is xr.Dataset object
        kwargs dict: 
            additional keyword arguments

        Returns
        -------
            Dynamic and static features of multiple stations. Dynamic features
            are by default returned as xr.Dataset unless `as_dataframe` is True, in
            such a case, it is a pandas dataframe with multiindex. If xr.Dataset,
            it consists of `data_vars` equal to number of stations and for each
            station, the `DataArray` is of dimensions (time, dynamic_features).
            where `time` is defined by `st` and `en` i.e. length of `DataArray`.
            In case, when the returned object is pandas DataFrame, the first index
            is `time` and second index is `dyanamic_features`. Static features
            are always returned as pandas DataFrame and have shape
            `(stations, static_features)`. If `dynamic_features` is None,
            then they are not returned and the returned value only consists of
            static features. Same holds true for `static_features`.
            If both are not None, then the returned type is a dictionary with
            `static` and `dynamic` keys.

        Raises:
            ValueError, if both dynamic_features and static_features are None

        Examples
        --------
            >>> from water_datasets import CAMELS_AUS
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
            ... dynamic_features=['streamflow_mmd', 'tmax_AWAP'], static_features=['elev_mean'])
        """
        st, en = self._check_length(st, en)

        stations = check_attributes(stations, self.stations(), 'stations')

        if dynamic_features is not None:

            dynamic_features = check_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')

            if netCDF4 is None or not os.path.exists(self.dyn_fname):
                # read from csv files
                # following code will run only once when fetch is called inside init method
                dyn = self._read_dynamic_from_csv(stations, dynamic_features, st=st, en=en)

            else:
                dyn = xr.open_dataset(self.dyn_fname)  # daataset
                dyn = dyn[stations].sel(dynamic_features=dynamic_features, time=slice(st, en))
                if as_dataframe:
                    dyn = dyn.to_dataframe(['time', 'dynamic_features'])

            if static_features is not None:
                static = self.fetch_static_features(stations, static_features)
                dyn = _handle_dynamic(dyn, as_dataframe)
                stns = {'dynamic': dyn, 'static': static}
            else:
                dyn = _handle_dynamic(dyn, as_dataframe)
                stns = dyn

        elif static_features is not None:

            return self.fetch_static_features(stations, static_features)

        else:
            raise ValueError

        return stns

    def fetch_dynamic_features(
            self,
            stn_id: str,
            dynamic_features = 'all',
            st=None,
            en=None,
            as_dataframe=False
    ):
        """Fetches all or selected dynamic features of one station.

        Parameters
        ----------
            stn_id : str
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
            >>> from water_datasets import CAMELS_AUS
            >>> camels = CAMELS_AUS()
            >>> camels.fetch_dynamic_features('224214A', as_dataframe=True).unstack()
            >>> camels.dynamic_features
            >>> camels.fetch_dynamic_features('224214A',
            ... features=['tmax_AWAP', 'vprp_AWAP', 'streamflow_mmd'],
            ... as_dataframe=True).unstack()
        """

        assert isinstance(stn_id, str), f"station id must be string is is of type {type(stn_id)}"
        station = [stn_id]
        return self.fetch_stations_features(
            station,
            dynamic_features,
            None,
            st=st,
            en=en,
            as_dataframe=as_dataframe
        )

    def fetch_station_features(
            self,
            station: str,
            dynamic_features: Union[str, list, None] = 'all',
            static_features: Union[str, list, None] = None,
            as_ts: bool = False,
            st: Union[str, None] = None,
            en: Union[str, None] = None,
            **kwargs
    ) -> pd.DataFrame:
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
            as_ts : bool
                whether static features are to be converted into a time
                series or not. If yes then the returned time series will be of
                same length as that of dynamic attribtues.
            st : str,optional
                starting point from which the data to be fetched. By default,
                the data will be fetched from where it is available.
            en : str, optional
                end point of data to be fetched. By default the dat will be fetched

        Returns
        -------
        pd.DataFrame
            dataframe if as_ts is True else it returns a dictionary of static and
            dynamic features for a station/gauge_id

        Examples
        --------
            >>> from water_datasets import CAMELS_AUS
            >>> dataset = CAMELS_AUS()
            >>> dataset.fetch_station_features('912101A')

        """
        st, en = self._check_length(st, en)

        station_df = pd.DataFrame()
        if dynamic_features:
            dynamic = self.fetch_dynamic_features(station, dynamic_features, st=st,
                                                  en=en, **kwargs)
            station_df = pd.concat([station_df, dynamic])

            if static_features is not None:
                static = self.fetch_static_features(station, static_features)

                if as_ts:
                    station_df = pd.concat([station_df, static], axis=1)
                else:
                    station_df = {'dynamic': station_df, 'static': static}

        elif static_features is not None:
            station_df = self.fetch_static_features(station, static_features)

        return station_df

    def plot_stations(
            self,
            stations:List[str] = 'all',
            marker='.',
            ax:plt_Axes = None,
            show:bool = True,
            **kwargs
    )->plt_Axes:
        """
        plots coordinates of stations

        Parameters
        ----------
        stations :
            name/names of stations. If not given, all stations will be plotted
        marker :
            marker to use.
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
        >>> from water_datasets import CAMELS_AUS
        >>> dataset = CAMELS_AUS()
        >>> dataset.plot_stations()
        >>> dataset.plot_stations(['1', '2', '3'])
        >>> dataset.plot_stations(marker='o', ms=0.3)
        >>> ax = dataset.plot_stations(marker='o', ms=0.3, show=False)
        >>> ax.set_title("Stations")
        >>> plt.show()

        """
        xy = self.stn_coords(stations)

        ax = easy_mpl.plot(xy.loc[:, 'long'].values,
                  xy.loc[:, 'lat'].values,
                  marker, ax=ax, show=False, **kwargs)

        if show:
            plt.show()

        return ax

    def q_mmd(
            self,
            stations: Union[str, List[str]] = "all"
    )->pd.DataFrame:
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
            a pandas DataFrame whose indices are time-steps and columns
            are catchment/station ids.

        """

        stations = check_attributes(stations, self.stations())
        
        if self._mmd_feature_name is None:
            q = self.fetch_stations_features(stations,
                                            dynamic_features=self._q_name,
                                            as_dataframe=True)
            q.index = q.index.get_level_values(0)
            area_m2 = self.area(stations) * 1e6  # area in m2
            q = (q / area_m2) * 86400  # cms to m/day
            return q  * 1e3  # to mm/day
        
        else:

            q = self.fetch_stations_features(
                stations,
                dynamic_features=self._mmd_feature_name,
                as_dataframe=True)
            q.index = q.index.get_level_values(0)
            return q

    def stn_coords(
            self,
            stations:Union[str, List[str]] = 'all'
    ) ->pd.DataFrame:
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
        coords :
            pandas DataFrame with ``long`` and ``lat`` columns.
            The length of dataframe will be equal to number of stations
            wholse coordinates are to be fetched.

        Examples
        --------
        >>> from water_datasets import CAMELS_CH
        >>> dataset = CAMELS_CH()
        >>> dataset.stn_coords() # returns coordinates of all stations
        >>> dataset.stn_coords('2004')  # returns coordinates of station whose id is 2004
        >>> dataset.stn_coords(['2004', '6004'])  # returns coordinates of two stations

        >>> from water_datasets import CAMELS_AUS
        >>> dataset = CAMELS_AUS()
        >>> dataset.stn_coords() # returns coordinates of all stations
        >>> dataset.stn_coords('912101A')  # returns coordinates of station whose id is 912101A
        >>> dataset.stn_coords(['G0050115', '912101A'])  # returns coordinates of two stations

        """
        df = self.fetch_static_features(features=self._coords_name)
        df.columns = ['lat', 'long']
        stations = check_attributes(stations, self.stations())

        return df.loc[stations, :]

    def transform_coords(self, xyz:np.ndarray)->np.ndarray:
        """
        transforms coordinates from projected to geographic

        must be implemented in base classes
        """
        return xyz

    def get_boundary(
            self,
            catchment_id: str,
            as_type: str = 'numpy'
    ):
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
        >>> from water_datasets import CAMELS_SE
        >>> dataset = CAMELS_SE()
        >>> dataset.get_boundary(dataset.stations()[0])
        """

        if shapefile is None:
            raise ModuleNotFoundError("shapefile module is not installed. Please install it to use boundary file")

        from shapefile import Reader

        bndry_sf = Reader(self.boundary_file)
        bndry_shp = bndry_sf.shape(self.bndry_id_map[catchment_id])

        bndry_sf.close()

        xyz = np.array(bndry_shp.points)

        xyz = self.transform_coords(xyz)

        return xyz

    def plot_catchment(
            self,
            catchment_id: str,
            ax: plt_Axes = None,
            show: bool = True,
            **kwargs
    )->plt.Axes:
        """
        plots catchment boundaries

        Parameters
        ----------
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
        >>> from water_datasets import CAMELS_AUS
        >>> dataset = CAMELS_AUS()
        >>> dataset.plot_catchment()
        >>> dataset.plot_catchment(marker='o', ms=0.3)
        >>> ax = dataset.plot_catchment(marker='o', ms=0.3, show=False)
        >>> ax.set_title("Catchment Boundaries")
        >>> plt.show()

        """
        catchment = self.get_boundary(catchment_id)

        if isinstance(catchment, np.ndarray):
            if catchment.ndim == 2:
                ax = easy_mpl.plot(catchment[:, 0], catchment[:, 1],
                        show=False, ax=ax, **kwargs)
            else:
                raise NotImplementedError
        # elif isinstance(catchment, geojson.geometry.Polygon):
        #     coords = catchment['coordinates']
        #     x = [i for i, j in coords[0]]
        #     y = [j for i, j in coords[0]]
        #     ax = plot(x, y, show=False, ax=ax, **kwargs)
        # elif isinstance(catchment, SPolygon):
        #     x, y = catchment.exterior.xy
        #     ax = plot(x, y, show=False, ax=ax, **kwargs)
        # elif isinstance(catchment, SMultiPolygon):
        #     raise NotImplementedError
        else:
            raise NotImplementedError

        if show:
            plt.show()
        return ax


def _handle_dynamic(dyn, as_dataframe:bool):
    if as_dataframe and isinstance(dyn, dict) and isinstance(list(dyn.values())[0], pd.DataFrame):
        # if the dyn is a dictionary of key, DataFames, we will return a MultiIndex
        # dataframe instead of a dictionary        
        dyn = xr.Dataset(dyn).to_dataframe(['time', 'dynamic_features'])  # todo wiered that we have to first convert to xr.Dataset and then to DataFrame
    elif isinstance(dyn, dict) and isinstance(list(dyn.values())[0], pd.DataFrame):
        # dyn is a dictionary of key, DataFames and we have to return xr Dataset
        #dyn = pd.concat(dyn, axis=0, keys=dyn.keys())
        dyn = xr.Dataset(dyn)
    return dyn
