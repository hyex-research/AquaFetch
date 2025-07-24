import os
from typing import Union, List, Dict

import pandas as pd

from .utils import _RainfallRunoff
from ..utils import check_attributes

from ._map import (
    total_precipitation,
    mean_air_temp,
    total_potential_evapotranspiration_with_specifier,
    actual_evapotranspiration,
    observed_streamflow_cms,
)

from ._map import (
    catchment_area,
    gauge_latitude,
    gauge_longitude,
    slope
    )


class Caravan_DK(_RainfallRunoff):
    """
    Reads Caravan extension Denmark - Danish dataset for large-sample hydrology
    following the works of `Koch and Schneider 2022 <https://doi.org/10.34194/geusb.v49.829>`_ .
    The dataset is downloaded from `zenodo <https://zenodo.org/record/7962379>`_ . This dataset
    consists of static and dynamic features from 308 danish catchments. There are 38
    dynamic (time series) features from 1981-01-02 to 2020-12-31 with daily timestep
    and 211 static features for each of 308 catchments.

    Please note that there is an updated version of this dataset following the works
    of `Liu et al., 2024 <https://doi.org/10.5194/essd-2024-292>`_ . This dataset
    is associated with the :py:class:`aqua_fetch.CAMELS_DK` class which can be imported as follows:

    >>> from aqua_fetch import CAMELS_DK

    Examples
    ---------
    >>> from aqua_fetch import Caravan_DK
    >>> dataset = Caravan_DK()
    >>> _, data = dataset.fetch(0.1, as_dataframe=True)
    >>> data.shape
    (569751, 30)  # 30 represents number of stations
    >>> data.index.names == ['time', 'dynamic_features']
    True
    >>> _, df = dataset.fetch(stations=1, as_dataframe=True)
    >>> df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
    >>> df.shape
    (14609, 39)
    # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
    308
    # get data by station id
    >>> _, df = dataset.fetch(stations='80001', as_dataframe=True)
    >>> df.unstack().shape
    (14609, 39)
    # get names of available dynamic features
    >>> dataset.dynamic_features
    # get only selected dynamic features
    >>> _, df = dataset.fetch(1, as_dataframe=True,
    ... dynamic_features=['snow_depth_water_equivalent_mean', 'temperature_2m_mean',
    ... 'potential_evaporation_sum', 'total_precipitation_sum', 'q_cms_obs'])
    >>> df.unstack().shape
    (14609, 5)
    # get names of available static features
    >>> dataset.static_features
    # get data of 10 random stations
    >>> _, df = dataset.fetch(10, as_dataframe=True)
    >>> df.shape
    (569751, 10)  # remember this is multi-indexed DataFrame
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='80001', static_features="all", as_dataframe=True)
    >>> static.shape, dynamic.unstack().shape
    ((1, 211), (14609, 39))
    """

    url = "https://zenodo.org/record/7962379"

    def __init__(self,
                 path=None,
                 overwrite=False,
                 to_netcdf: bool = True,
                 **kwargs):
        """
        Parameters
        ----------
        path : str
            If the data is alredy downloaded then provide the complete
            path to it. If None, then the data will be downloaded.
            The data is downloaded once and therefore susbsequent
            calls to this class will not download the data unless
            ``overwrite`` is set to True.
        overwrite : bool
            If the data is already down then you can set it to True,
            to make a fresh download.
        to_netcdf : bool
            whether to convert all the data into one netcdf file or not.
            This will fasten repeated calls to fetch etc but will
            require netcdf5 package as well as xarry.
        """
        super(Caravan_DK, self).__init__(path=path, **kwargs)
        self.path = path
        self._download(overwrite=overwrite)

        self._static_features = self._static_data().columns.to_list()
        self._dynamic_features = self.__dynamic_features()

        self.dyn_fname = os.path.join(self.path, 'caravandk_dyn.nc')

        #if to_netcdf:
        self._maybe_to_netcdf('caravandk_dyn')

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(
            self.path,
            #"Caravan_DK",
            "Caravan_extension_DK",
            "Caravan_extension_DK",
            "Caravan_extension_DK",
            "shapefiles",
            "camelsdk",
            "camelsdk_basin_shapes.shp"
        )

    @property
    def boundary_id_map(self) -> str:
        return "gauge_id"

    
    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'area': catchment_area(),
                'gauge_lat': gauge_latitude(),
                'slope_mean': slope('mkm-1'),
                'gauge_lon': gauge_longitude(),
        }

    @property
    def csv_path(self):
        return os.path.join(self.path, "Caravan_extension_DK",
                            "Caravan_extension_DK", "Caravan_extension_DK",
                            "timeseries", "csv", "camelsdk")

    @property
    def nc_path(self):
        return os.path.join(self.path, "Caravan_extension_DK",
                            "Caravan_extension_DK", "Caravan_extension_DK",
                            "timeseries", "netcdf", "camelsdk")

    @property
    def other_attr_fpath(self):
        """returns path to attributes_other_camelsdk.csv file
        """
        return os.path.join(self.path, "Caravan_extension_DK",
                            "Caravan_extension_DK", "Caravan_extension_DK",
                            "attributes", "camelsdk", "attributes_other_camelsdk.csv")

    @property
    def caravan_attr_fpath(self):
        """returns path to attributes_caravan_camelsdk.csv file
        """
        return os.path.join(self.path, "Caravan_extension_DK",
                            "Caravan_extension_DK", "Caravan_extension_DK",
                            "attributes", "camelsdk",
                            "attributes_caravan_camelsdk.csv")

    def stations(self) -> List[str]:
        return [fname.split(".csv")[0][9:] for fname in os.listdir(self.csv_path)]

    def _read_csv(self, stn: str) -> pd.DataFrame:
        fpath = os.path.join(self.csv_path, f"camelsdk_{stn}.csv")
        df = pd.read_csv(os.path.join(fpath))
        df.index = pd.to_datetime(df.pop('date'))

        df.rename(columns=self.dyn_map, inplace=True)

        df.index.name = 'time'
        df.columns.name = 'dynamic_features'

        return df

    @property
    def dynamic_features(self) -> List[str]:
        """returns names of dynamic features"""
        return self._dynamic_features

    def __dynamic_features(self) -> List[str]:
        return self._read_csv('100006').columns.to_list()

    @property
    def static_features(self) -> List[str]:
        """returns static features for Denmark catchments"""
        return self._static_features

    @property
    def start(self):  # start of data
        return pd.Timestamp('1981-01-02 00:00:00')

    @property
    def end(self) -> pd.Timestamp:  # end of data
        return pd.Timestamp('2020-12-31 00:00:00')

    @property
    def dyn_map(self) -> Dict[str, str]:
        return {
            'streamflow': observed_streamflow_cms(),
        }

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
        >>> dataset = Caravan_DK()
        >>> dataset.stn_coords() # returns coordinates of all stations
        >>> dataset.stn_coords('100010')  # returns coordinates of station whose id is 912101A
        >>> dataset.stn_coords(['100010', '210062'])  # returns coordinates of two stations

        """
        df = pd.read_csv(self.other_attr_fpath,
                         dtype={"gauge_id": str,
                                'gauge_lat': float,
                                'gauge_lon': float,
                                'area': float,
                                'gauge_name': str,
                                'country': str
                                })
        df.index = [name.split('camelsdk_')[1] for name in df['gauge_id']]
        df = df[['gauge_lat', 'gauge_lon']]
        df.columns = ['lat', 'long']

        stations = check_attributes(stations, self.stations())

        return df.loc[stations, :]

    def _read_dynamic(
            self,
            stations,
            dynamic_features,
            st=None,
            en=None) -> dict:
        
        st, en = self._check_length(st, en)
        features = check_attributes(dynamic_features, self.dynamic_features)

        dyn = {stn: self._read_csv(stn).loc[st:en, features] for stn in stations}

        return dyn
    
    def _static_data(self)->pd.DataFrame:
        df = pd.concat([self.hyd_atlas_attributes(),
                        self.other_static_attributes(),
                        self.caravan_static_attributes()], axis=1)
        df.rename(columns=self.static_map, inplace=True)
        return df

    @property
    def hyd_atlas_fpath(self):
        return os.path.join(self.path,
                            "Caravan_extension_DK",
                            "Caravan_extension_DK",
                            "Caravan_extension_DK",
                            "attributes", "camelsdk",
                            "attributes_hydroatlas_camelsdk.csv")

    def hyd_atlas_attributes(self, stations='all') -> pd.DataFrame:
        """
        Returns
        --------
            a :obj:`pandas.DataFrame` of shape (308, 196)
        """
        stations = check_attributes(stations, self.stations())
        df = pd.read_csv(self.hyd_atlas_fpath)

        indices = df.pop('gauge_id')
        df.index = [idx[9:] for idx in indices]
        return df

    def other_static_attributes(self, stations='all') -> pd.DataFrame:
        """
        Returns
        --------
            a :obj:`pandas.DataFrame` of shape (308, 5)
        """
        stations = check_attributes(stations, self.stations())
        df = pd.read_csv(self.other_attr_fpath)
        indices = df.pop('gauge_id')
        df.index = [idx[9:] for idx in indices]
        return df

    def caravan_static_attributes(self, stations='all') -> pd.DataFrame:
        """
        Returns
        --------
            a :obj:`pandas.DataFrame` of shape (308, 10)
        """
        stations = check_attributes(stations, self.stations())
        df = pd.read_csv(self.caravan_attr_fpath)
        indices = df.pop('gauge_id')
        df.index = [idx[9:] for idx in indices]
        return df
