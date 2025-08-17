
import os
import concurrent.futures as cf
from typing import Union, List, Dict, Tuple, Callable, Any

import pandas as pd

from .utils import _RainfallRunoff
from ..utils import get_cpus
from ..utils import check_st_en  # todo check difference with self.check_length
from ..utils import check_attributes, download, unzip

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
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='GRDC_3664802', as_dataframe=True)
    >>> df = dynamic['GRDC_3664802'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (26801, 39)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       5357
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (535 out of 5357)
       535
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(26801, 39), (26801, 39), (26801, 39),... (26801, 39), (26801, 39)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('GRDC_3664802', as_dataframe=True,
    ...  dynamic_features=['total_precipitation_sum', 'potential_evaporation_sum', 'temperature_2m_mean', 'q_cms_obs'])
    >>> dynamic['GRDC_3664802'].shape
       (26801, 4)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='GRDC_3664802', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['GRDC_3664802'].shape
    ((1, 211), 1, (26801, 39))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 26801, 'dynamic_features': 39})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (5357, 2)
    >>> dataset.stn_coords('GRDC_3664802')  # returns coordinates of station whose id is GRDC_3664802
        -26.2271        -51.0771
    >>> dataset.stn_coords(['GRDC_3664802', 'GRDC_1159337'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('GRDC_3664802')
    # get coordinates of two stations
    >>> dataset.area(['GRDC_3664802', 'GRDC_1159337'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('GRDC_3664802')
    ...
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
            self.ftype = 'csv'
            if "caravan-grdc-extension-nc.tar.gz" in self.url:
                self.url.pop("caravan-grdc-extension-nc.tar.gz")
        else:
            self.ftype = 'netcdf'
            if "caravan-grdc-extension-csv.tar.gz" in self.url:
                self.url.pop("caravan-grdc-extension-csv.tar.gz")

        super().__init__(path=path, verbosity=verbosity, **kwargs)

        if not os.path.exists(self.path):
            if self.verbosity>1:
                print(f"Creating directory {self.path}")
            os.makedirs(self.path)

        for _file, url in self.url.items():
            fpath = os.path.join(self.path, _file)
            if not os.path.exists(fpath) and not overwrite:
                if self.verbosity > 0:
                    print(f"Downloading {_file} from {url + _file}")
                download(url + _file, outdir=self.path, fname=_file, )
                unzip(self.path)
            elif self.verbosity > 0:
                print(f"{_file} at {self.path} already exists")

        # so that we dont have to read the files again and again
        self._stations = self.other_attributes().index.to_list()
        self._static_attributes = self._static_data().columns.tolist()
        self._dynamic_attributes = self._read_stn_dyn(self.stations()[0]).columns.tolist()

        self.dyn_fname = ''

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(
            self.shapefiles_path,
            'grdc_basin_shapes.shp'
        )

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
            'streamflow': observed_streamflow_mmd(),
            'temperature_2m_mean': mean_air_temp_with_specifier('2m'),
            'temperature_2m_min': min_air_temp_with_specifier('2m'),
            'temperature_2m_max': max_air_temp_with_specifier('2m'),
            'total_precipitation_sum': total_precipitation(),
        }

    @property
    def dyn_generator(self)->Dict[str, Tuple[Callable, Any]]:
        return {
            observed_streamflow_cms(): (self.mmd_to_cms, observed_streamflow_mmd()),
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

    @property
    def start(self):
        return pd.Timestamp("19500102")

    @property
    def end(self):
        return pd.Timestamp("20230518")

    def other_attributes(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.attrs_path, 'attributes_other_grdc.csv'), index_col='gauge_id')

    def hydroatlas_attributes(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.attrs_path, 'attributes_hydroatlas_grdc.csv'), index_col='gauge_id')

    def caravan_attributes(self) -> pd.DataFrame:
        return pd.read_csv(os.path.join(self.attrs_path, 'attributes_caravan_grdc.csv'), index_col='gauge_id')

    def _static_data(self) -> pd.DataFrame:
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
            >>> from aqua_fetch import GRDCCaravan
            >>> dataset = GRDCCaravan()
            >>> dataset.fetch_station_features('912101A')
        """
        static, dynamic = None, None

        st, en = self._check_length(st, en)

        if dynamic_features is not None:
            
            dynamic = self._read_dynamic(station, dynamic_features, st, en)[station]

        if static_features is not None:
            static = self.fetch_static_features(station, static_features)

        return static, dynamic

    def _read_dynamic(
            self,
            stations,
            dynamic_features,
            st=None,
            en=None) -> dict:

        dynamic_features = check_attributes(dynamic_features, self.dynamic_features)
        stations = check_attributes(stations, self.stations())
        st, en = self._check_length(st, en)

        cpus = self.processes or min(get_cpus(), 64)

        if len(stations) > 10 and cpus>1:
            
            if self.verbosity > 0:
                print(f"Using {cpus} cpus to read dynamic features for {len(stations)} stations")
            with  cf.ProcessPoolExecutor(max_workers=cpus) as executor:
                results = executor.map(
                    self._read_stn_dyn,
                    stations,
                )
            dyn = {stn: data.loc[st:en, dynamic_features] for stn, data in zip(stations, results)}
        else:
            if self.verbosity > 0:
                print(f"Using single cpu to read dynamic features for {len(stations)} stations")
            dyn = {}
            for idx, stn in enumerate(stations):
                dyn[stn] = self._read_stn_dyn(stn).loc[st: en, dynamic_features]

                if self.verbosity>0 and idx % 100 == 0:
                    print(f"Read data for {idx} stations")

        return dyn

    def _read_stn_dyn(self, station) -> pd.DataFrame:
        if self.ftype == "netcdf":
            fpath = os.path.join(self.ts_path, f'{station}.nc')
            df = xr.load_dataset(fpath).to_dataframe()
        else:
            fpath = os.path.join(self.ts_path, f'{station}.csv')
            df = pd.read_csv(fpath, index_col='date', parse_dates=True)

        df.rename(columns=self.dyn_map, inplace=True)

        for new_feature, (func, old_feature) in self.dyn_generator.items():
            if new_feature not in df.columns:
                df[new_feature] = func(pd.Series(df[old_feature], name=station))

        df = df.sort_index()
        # Ensure df always extends to self.end
        if df.index[-1] < self.end:
            # Create complete date range from start of existing data to self.end
            complete_range = pd.date_range(start=self.start, end=self.end, freq='D')
            # Reindex to fill missing dates with NaN
            df = df.reindex(complete_range)

        df.index.name = 'time'
        df.columns.name = 'dynamic_features'
        return df
