import os
import concurrent.futures as cf
from typing import Union, List, Dict

import pandas as pd

from .utils import _RainfallRunoff
from ..utils import get_cpus
from ..utils import check_st_en
from ..utils import check_attributes, download, _unzip

from .._backend import netCDF4, xarray as xr


from ._map import (
    observed_streamflow_cms,
    observed_streamflow_mmd,
    mean_air_temp,
    min_air_temp_with_specifier,
    max_air_temp_with_specifier,
    max_air_temp,
    min_air_temp,
    mean_air_temp_with_specifier,
    total_precipitation,
    total_precipitation_with_specifier,
    total_potential_evapotranspiration,
    total_potential_evapotranspiration_with_specifier,
    simulated_streamflow_cms,
    actual_evapotranspiration,
    actual_evapotranspiration_with_specifier,
    solar_radiation_with_specifier,
    mean_vapor_pressure,
    mean_vapor_pressure_with_specifier,
    mean_rel_hum,
    mean_rel_hum_with_specifier,
    rel_hum_with_specifier,
    mean_windspeed,
    u_component_of_wind,
    v_component_of_wind,
    solar_radiation,
    downward_longwave_radiation,
    snow_water_equivalent,
    mean_specific_humidity,
)

from ._map import (
    catchment_area,
    gauge_latitude,
    gauge_longitude,
    slope
    )

class GRDCCaravan(_RainfallRunoff):
    """
    This is a dataset of 5357 catchments from around the globe following the works of
    `Faerber et al., 2023 <https://zenodo.org/records/10074416>`_ . The dataset consists of 39
    dynamic (timeseries) features and 211 static features. The dynamic (timeseries) data
    spands from 1950-01-02 to 2019-05-19.

    if xarray + netCDF4 packages are installed then netcdf files will be downloaded
    otherwise csv files will be downloaded and used.

    Examples
    --------
    >>> from aqua_fetch import GRDCCaravan
    >>> dataset = GRDCCaravan()
    >>> _, df = dataset.fetch(stations=1, as_dataframe=True)
    >>> df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
    >>> df.shape
       (26801, 39)
    get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       5357
    get data of 10 % of stations as dataframe
    >>> _, df = dataset.fetch(0.1, as_dataframe=True)
    >>> df.shape
       (1045239, 535)
    The returned dataframe is a multi-indexed data
    >>> df.index.names == ['time', 'dynamic_features']
        True
    get data by station id
    >>> _, df = dataset.fetch(stations='GRDC_3664802', as_dataframe=True).unstack()
    >>> df.shape
         (26800, 39)
    get names of available dynamic features
    >>> dataset.dynamic_features
    get only selected dynamic features
    >>> _, data = dataset.fetch(1, as_dataframe=True,
    ...  dynamic_features=['total_precipitation_sum', 'potential_evaporation_sum', 'temperature_2m_mean', 'streamflow']).unstack()
    >>> data.shape
        (26800, 4)
    get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, df = dataset.fetch(10, as_dataframe=True)
    >>> df.shape  # remember this is a multiindexed dataframe
        (1045239, 10)
    If we want to get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='GRDC_3664802', static_features="all", as_dataframe=True)
    >>> static.shape, dynamic.shape
        ((1, 211), (1045200, 1))
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (5357, 2)
    >>> dataset.stn_coords('GRDC_3664802')  # returns coordinates of station whose id is GRDC_3664802
        -26.2271	-51.0771
    >>> dataset.stn_coords(['GRDC_3664802', 'GRDC_1159337'])  # returns coordinates of two stations

    """

    url = {
        'caravan-grdc-extension-nc.tar.gz':
            "https://zenodo.org/records/10074416/files/caravan-grdc-extension-nc.tar.gz?download=1",
        'caravan-grdc-extension-csv.tar.gz':
            "https://zenodo.org/records/10074416/files/caravan-grdc-extension-csv.tar.gz?download=1"
    }

    def __init__(
            self,
            path=None,
            overwrite: bool = False,
            verbosity: int = 1,
            **kwargs
    ):

        if xr is None:
            self.ftype == 'csv'
            if "caravan-grdc-extension-nc.tar.gz" in self.url:
                self.url.pop("caravan-grdc-extension-nc.tar.gz")
        else:
            self.ftype = 'netcdf'
            if "caravan-grdc-extension-csv.tar.gz" in self.url:
                self.url.pop("caravan-grdc-extension-csv.tar.gz")

        super().__init__(path=path, verbosity=verbosity, **kwargs)

        for _file, url in self.url.items():
            fpath = os.path.join(self.path, _file)
            if not os.path.exists(fpath) and not overwrite:
                if self.verbosity > 0:
                    print(f"Downloading {_file} from {url + _file}")
                download(url + _file, outdir=self.path, fname=_file, )
                _unzip(self.path)
            elif self.verbosity > 0:
                print(f"{_file} at {self.path} already exists")

        self.boundary_file = os.path.join(
            self.shapefiles_path,
            'grdc_basin_shapes.shp'
        )
        self._create_boundary_id_map(self.boundary_file, 0)

        # so that we dont have to read the files again and again
        self._stations = self.other_attributes().index.to_list()
        self._static_attributes = self.static_data().columns.tolist()
        self._dynamic_attributes = self._read_dynamic_for_stn(self.stations()[0]).columns.tolist()

        self.dyn_fname = ''

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'area': catchment_area(),
                'gauge_lat': gauge_latitude(),
                'gauge_lon': gauge_longitude(),
        }

    @property
    def dyn_map(self):
        return {
            'streamflow': observed_streamflow_cms(),
            'temperature_2m_mean': mean_air_temp_with_specifier('2m'),
            'temperature_2m_min': min_air_temp_with_specifier('2m'),
            'temperature_2m_max': max_air_temp_with_specifier('2m'),
            'total_precipitation_sum': total_precipitation(),
        }

    @property
    def static_features(self):
        return self._static_attributes

    @property
    def dynamic_features(self):
        return self._dynamic_attributes

    @property
    def shapefiles_path(self):
        if self.ftype == 'csv':
            return os.path.join(self.path, 'GRDC-Caravan-extension-csv',
                                'shapefiles', 'grdc')
        return os.path.join(self.path, 'GRDC-Caravan-extension-nc',
                            'shapefiles', 'grdc')

    @property
    def attrs_path(self):
        if self.ftype == 'csv':
            return os.path.join(self.path, 'GRDC-Caravan-extension-csv',
                                'attributes', 'grdc')
        return os.path.join(self.path, 'GRDC-Caravan-extension-nc',
                            'attributes', 'grdc')

    @property
    def ts_path(self) -> os.PathLike:
        if self.ftype == 'csv':
            return os.path.join(self.path, 'GRDC-Caravan-extension-csv',
                                'timeseries', 'grdc')

        return os.path.join(self.path, 'GRDC-Caravan-extension-nc',
                            'timeseries', self.ftype, 'grdc')

    def stations(self) -> List[str]:
        return self._stations

    # @property
    # def _coords_name(self) -> List[str]:
    #     return ['gauge_lat', 'gauge_lon']

    # @property
    # def _area_name(self) -> str:
    #     return 'area'

    @property
    def start(self):
        return pd.Timestamp("19500102")

    @property
    def end(self):
        return pd.Timestamp("20230519")

    # @property
    # def _q_name(self) -> str:
    #     return observed_streamflow_cms()

    def other_attributes(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.attrs_path, 'attributes_other_grdc.csv'), index_col='gauge_id')

    def hydroatlas_attributes(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.attrs_path, 'attributes_hydroatlas_grdc.csv'), index_col='gauge_id')

    def caravan_attributes(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.attrs_path, 'attributes_caravan_grdc.csv'), index_col='gauge_id')

    def static_data(self) -> pd.DataFrame:
        df = pd.concat([
            self.other_attributes(),
            self.hydroatlas_attributes(),
            self.caravan_attributes(),
        ], axis=1)

        df.rename(columns=self.static_map, inplace=True)

        return df

    def fetch_station_features(
            self,
            station: str,
            dynamic_features: Union[str, list, None] = 'all',
            static_features: Union[str, list, None] = None,
            as_ts: bool = False,
            st: Union[str, None] = None,
            en: Union[str, None] = None,
            **kwargs
    ) -> Dict[str, pd.DataFrame]:
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
        Dict
            dataframe if as_ts is True else it returns a dictionary of static and
            dynamic features for a station/gauge_id

        Examples
        --------
            >>> from aqua_fetch import GRDCCaravan
            >>> dataset = GRDCCaravan()
            >>> dataset.fetch_station_features('912101A')
        """
        dynamic_features = check_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')

        st, en = check_st_en(st, en, self.start, self.end)

        if self.ftype == "netcdf":
            fpath = os.path.join(self.ts_path, f'{station}.nc')
            df = xr.open_dataset(fpath).to_dataframe()
        else:
            fpath = os.path.join(self.ts_path, f'{station}.csv')
            df = pd.read_csv(fpath, index_col='date', parse_dates=True)
        
        df = df.loc[st:en, :]

        df.rename(columns=self.dyn_map, inplace=True)

        if static_features is not None:
            static = self.fetch_static_features(station, static_features)

        return {'static': static, 'dynamic': df[dynamic_features]}

    def fetch_static_features(
            self,
            stn_id: Union[str, list] = "all",
            static_features: Union[str, list] = "all"
    ) -> pd.DataFrame:
        """

        Returns static features of one or more stations.

        Parameters
        ----------
            stn_id : str
                name/id of station/stations of which to extract the data
            static_features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Returns
        -------
        pd.DataFrame
            a pandas dataframe of shape (stations, features)

        Examples
        ---------
        >>> from aqua_fetch import GRDCCaravan
        >>> dataset = GRDCCaravan()
        get all static data of all stations
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (1555, 111)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('DE110010')
        >>> static_data.shape
           (1, 111)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['p_mean', 'p_seasonality', 'frac_snow'])
        >>> static_data.shape
           (1555, 3)
        >>> data = dataset.fetch_static_features('DE110000', static_features=['p_mean', 'p_seasonality', 'frac_snow'])
        >>> data.shape
           (1, 3)
        """
        stations = check_attributes(stn_id, self.stations(), 'stations')

        df = self.static_data()
        features = check_attributes(static_features, df.columns.tolist(),
                                    "static_features")
        return df.loc[stations, features]

    def _read_dynamic_from_csv(
            self,
            stations,
            dynamic_features,
            st=None,
            en=None) -> dict:

        dynamic_features = check_attributes(dynamic_features, self.dynamic_features)
        stations = check_attributes(stations, self.stations())

        if len(stations) > 10:
            cpus = self.processes or min(get_cpus(), 64)
            with  cf.ProcessPoolExecutor(max_workers=cpus) as executor:
                results = executor.map(
                    self._read_dynamic_for_stn,
                    stations,
                )
            dyn = {stn: data.loc[st:en, dynamic_features] for stn, data in zip(stations, results)}
        else:
            dyn = {
                stn: self._read_dynamic_for_stn(stn).loc[st: en, dynamic_features] for stn in stations
            }

        return dyn

    def _read_dynamic_for_stn(self, stn_id) -> pd.DataFrame:
        if self.ftype == "netcdf":
            fpath = os.path.join(self.ts_path, f'{stn_id}.nc')
            df = xr.load_dataset(fpath).to_dataframe()
        else:
            fpath = os.path.join(self.ts_path, f'{stn_id}.csv')
            df = pd.read_csv(fpath, index_col='date', parse_dates=True)

        df.rename(columns=self.dyn_map, inplace=True)

        df.index.name = 'time'
        df.columns.name = 'dynamic_features'
        return df
