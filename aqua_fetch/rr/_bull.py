import os
from typing import List, Union, Dict
import concurrent.futures as cf

import pandas as pd

from .utils import _RainfallRunoff
from .._backend import netCDF4
from .._backend import xarray as xr
from ..utils import check_attributes, get_cpus
from ._map import (
    total_potential_evapotranspiration_with_specifier,
    solar_radiation,
    max_solar_radiation,
    min_solar_radiation,
    mean_thermal_radiation,
    max_thermal_radiation,
    min_themal_radiation,
    max_air_temp_with_specifier,
    min_air_temp_with_specifier,
    mean_air_temp_with_specifier,
    total_precipitation_with_specifier,
    max_dewpoint_temperature,
    min_dewpoint_temperature,
    mean_dewpoint_temperature,
    mean_dewpoint_temperature_with_specifier,
    mean_potential_evaporation,
    observed_streamflow_cms,
    max_dewpoint_temperature,
    snow_water_equivalent,
    min_snow_water_equivalent,
    max_snow_water_equivalent,
    u_component_of_wind_with_specifier,
    v_component_of_wind_with_specifier
)

from ._map import (
    catchment_area,
    gauge_latitude,
    gauge_longitude,
    slope
    )

BUL_COLUMNS = [
    'snow_depth_water_equivalent_mean_BULL', 
    'surface_net_solar_radiation_mean_BULL',
    'surface_net_thermal_radiation_mean_BULL',
    'surface_pressure_mean_BULL', 'temperature_2m_mean_BULL', 'dewpoint_temperature_2m_mean_BULL',
    'u_component_of_wind_10m_mean_BULL',
    'v_component_of_wind_10m_mean_BULL', 'volumetric_soil_water_layer_1_mean_BULL',
    'volumetric_soil_water_layer_2_mean_BULL',
    'volumetric_soil_water_layer_3_mean_BULL', 'volumetric_soil_water_layer_4_mean_BULL',
    'snow_depth_water_equivalent_min_BULL',
    'surface_net_solar_radiation_min_BULL', 'surface_net_thermal_radiation_min_BULL', 'surface_pressure_min_BULL',
    'temperature_2m_min_BULL', 'dewpoint_temperature_2m_min_BULL', 'u_component_of_wind_10m_min_BULL',
    'v_component_of_wind_10m_min_BULL',
    'volumetric_soil_water_layer_1_min_BULL', 'volumetric_soil_water_layer_2_min_BULL',
    'volumetric_soil_water_layer_3_min_BULL',
    'volumetric_soil_water_layer_4_min_BULL', 'snow_depth_water_equivalent_max_BULL',
    'surface_net_solar_radiation_max_BULL',
    'surface_net_thermal_radiation_max_BULL', 'surface_pressure_max_BULL', 'temperature_2m_max_BULL',
    'dewpoint_temperature_2m_max_BULL',
    'u_component_of_wind_10m_max_BULL', 'v_component_of_wind_10m_max_BULL', 'volumetric_soil_water_layer_1_max_BULL',
    'volumetric_soil_water_layer_2_max_BULL', 'volumetric_soil_water_layer_3_max_BULL',
    'volumetric_soil_water_layer_4_max_BULL',
    'total_precipitation_sum_BULL', 'potential_evaporation_sum_BULL', 
    'streamflow_BULL'
]


class Bull(_RainfallRunoff):
    """
    Following the works of `Aparicio et al., 2024 <https://doi.org/10.1038/s41597-024-03594-5>`_.
    The data is taken from the `Zenodo repository <https://zenodo.org/records/10629809>`_.
    This dataset contains 484 stations with 55 dynamic (time series) features and
    214 static features. The dynamic features span from 1951 to 2021.

    Examples
    ---------
    >>> from aqua_fetch import Bull
    >>> dataset = Bull()
    >>> _, data = dataset.fetch(0.1, as_dataframe=True)
    >>> data.shape
    (1426260, 48)  # 40 represents number of stations
    Since data is a multi-index dataframe, we can get data of one station as below
    >>> data['BULL_9007'].unstack().shape  # the name of station could be different
    (25932, 13)
    If we don't set as_dataframe=True, then the returned data will be a xarray Dataset
    >>> data = dataset.fetch(0.1)
    >>> type(data)
        xarray.core.dataset.Dataset
    >>> data.dims
    FrozenMappingWarningOnValuesAccess({'time': 25932, 'dynamic_features': 55})
    >>> len(data.data_vars)
        48
    >>> _, df = dataset.fetch(stations=1, as_dataframe=True)  # get data of only one random station
    >>> df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
    >>> df.shape
    (25932, 55)
    # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
    484
    # get data by station id
    >>> df = dataset.fetch(stations='BULL_9007', as_dataframe=True).unstack()
    >>> df.shape
    (25932, 55)
    # get names of available dynamic features
    >>> dataset.dynamic_features
    # get only selected dynamic features
    >>> _, df = dataset.fetch(1, as_dataframe=True,
    ... dynamic_features=['pet_mm_AEMET',  'airtemp_C_mean_AEMET', 'pcp_mm_ERA5Land', 'q_obs_cms']).unstack()
    >>> df.shape
    (25932, 4)
    # get names of available static features
    >>> dataset.static_features
    # get data of 10 random stations
    >>> _, df = dataset.fetch(10, as_dataframe=True)
    >>> df.shape
    (166166, 10)  # remember this is multi-indexed DataFrame
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='BULL_9007', static_features="all", as_dataframe=True)
    >>> static.shape, dynamic.shape
    ((1, 214), (1426260, 1))
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (484, 2)
    >>> dataset.stn_coords('BULL_9007')  # returns coordinates of station whose id is GRDC_3664802
        41.298	-1.967
    >>> dataset.stn_coords(['BULL_9007', 'BULL_8083'])  # returns coordinates of two stations

    """

    url = "https://zenodo.org/records/10629809"

    def __init__(
            self,
            path,
            overwrite=False,
            **kwargs
    ):
        super().__init__(path, **kwargs)

        self._download(overwrite=overwrite)

        self._unzip_7z_files()

        if netCDF4 is None:
            self.ftype = "csv"
        else:
            self.ftype = "netcdf"

        self._dynamic_features = self._read_dynamic_for_stn(self.stations()[0]).columns.tolist()
        self._static_features = list(set(self._static_data().columns.tolist()))

        self.boundary_file = os.path.join(self.shapefiles_path, "BULL_basin_shapes.shp")

        self._create_boundary_id_map(self.boundary_file, 0)

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
            'dewpoint_temperature_2m_max_BULL': max_dewpoint_temperature(),
            'dewpoint_temperature_2m_mean_BULL': mean_dewpoint_temperature(),
            'dewpoint_temperature_2m_min_BULL': min_dewpoint_temperature(),  # todo: are we considering height
            'potential_evaporation_sum_BULL': mean_potential_evaporation(),  # todo: is it mean or total?
            'streamflow_BULL': observed_streamflow_cms(),
            'potential_evapotranspiration_AEMET': total_potential_evapotranspiration_with_specifier('AEMET'),
            'potential_evapotranspiration_EMO1_arc': total_potential_evapotranspiration_with_specifier('EMO1arc'),
            'potential_evapotranspiration_ERA5_Land': total_potential_evapotranspiration_with_specifier('ERA5Land'),
            'surface_net_solar_radiation_mean_BULL': solar_radiation(),
            'surface_net_solar_radiation_max_BULL': max_solar_radiation(),
            'surface_net_solar_radiation_min_BULL': min_solar_radiation(),
            'surface_net_thermal_radiation_max_BULL': max_thermal_radiation(),
            'surface_net_thermal_radiation_mean_BULL': mean_thermal_radiation(),
            'surface_net_thermal_radiation_min_BULL': min_themal_radiation(),
            'temperature_max_AEMET': max_air_temp_with_specifier('AEMET'),
            'temperature_max_EMO1_arc': max_air_temp_with_specifier('EMO1arc'),
            'temperature_max_ERA5_Land': max_air_temp_with_specifier('ERA5Land'),
            'temperature_mean_AEMET': mean_air_temp_with_specifier('AEMET'),
            'temperature_mean_EMO1_arc': mean_air_temp_with_specifier('EMO1arc'),
            'temperature_mean_ERA5_Land': mean_air_temp_with_specifier('ERA5Land'),
            'temperature_min_AEMET': min_air_temp_with_specifier('AEMET'),
            'temperature_min_EMO1_arc': min_air_temp_with_specifier('EMO1arc'),
            'temperature_min_ERA5_Land': min_air_temp_with_specifier('ERA5Land'),
            'total_precipitation_AEMET': total_precipitation_with_specifier('AEMET'),
            'total_precipitation_EMO1_arc': total_precipitation_with_specifier('EMO1arc'),
            'total_precipitation_ERA5_Land': total_precipitation_with_specifier('ERA5Land'),
            'total_precipitation_sum_BULL': total_precipitation_with_specifier('BULL'),
            'snow_depth_water_equivalent_max_BULL': max_snow_water_equivalent(),
            'snow_depth_water_equivalent_mean_BULL': snow_water_equivalent(),
            'snow_depth_water_equivalent_min_BULL': min_snow_water_equivalent(),
            #'surface_pressure_max_BULL':   # todo: is it same as air pressure
            'temperature_2m_max_BULL': max_air_temp_with_specifier('2m'),
            'temperature_2m_mean_BULL': mean_air_temp_with_specifier('2m'),
            'temperature_2m_min_BULL': min_air_temp_with_specifier('2m'),
            'u_component_of_wind_10m_max_BULL': u_component_of_wind_with_specifier('max_10m'),
            'u_component_of_wind_10m_mean_BULL': u_component_of_wind_with_specifier('mean_10m'),
            'u_component_of_wind_10m_min_BULL': u_component_of_wind_with_specifier('min_10m'),
            'v_component_of_wind_10m_max_BULL': v_component_of_wind_with_specifier('max_10m'),
            'v_component_of_wind_10m_mean_BULL': v_component_of_wind_with_specifier('mean_10m'),
            'v_component_of_wind_10m_min_BULL': v_component_of_wind_with_specifier('min_10m'),
        }

    @property
    def attributes_path(self):
        return os.path.join(self.path, "attributes")

    @property
    def shapefiles_path(self):
        return os.path.join(self.path, "shapefiles", "shapefiles")

    @property
    def ts_path(self):
        return os.path.join(self.path, "timeseries", "timeseries")

    @property
    def q_path(self):
        return os.path.join(self.ts_path, self.ftype, "streamflow")

    @property
    def aemet_path(self):
        return os.path.join(self.ts_path, self.ftype, "AEMET")

    @property
    def bull_path(self):
        return os.path.join(self.ts_path, self.ftype, "BULL")

    @property
    def era5_land_path(self):
        return os.path.join(self.ts_path, self.ftype, "ERA5_Land")

    @property
    def emo1_arc_path(self):
        return os.path.join(self.ts_path, self.ftype, "EMO1_arc")

    @property
    def start(self):
        return pd.Timestamp("19510102")

    @property
    def end(self):
        return pd.Timestamp("20211231")

    def stations(self) -> List[str]:
        return ["BULL_" + f.split('.')[0].split('_')[1] for f in os.listdir(self.q_path)]

    @property
    def dynamic_features(self) -> List[str]:
        return self._dynamic_features

    @property
    def static_features(self) -> List[str]:
        return self._static_features

    def _unzip_7z_files(self):
        # The attributes file is .7z file
        try:
            import py7zr
        except (ModuleNotFoundError, ImportError):
            raise ImportError('py7zr is required to extract the .7z files. Please install it using `pip install py7zr`')

        # get all .7z files in self.path
        files = [f for f in os.listdir(self.path) if f.endswith('.7z')]

        for file in files:
            fpath = os.path.join(self.path, file)
            with py7zr.SevenZipFile(fpath, mode='r') as z:
                z.extractall(path = self.path)
                print(f'Extracted {file}')

    def caravan_attributes(self) -> pd.DataFrame:
        """a dataframe of shape (484, 10)"""
        return pd.read_csv(
            os.path.join(self.attributes_path, "attributes_caravan_.csv"),
            index_col=0)

    def hydroatlas_attributes(self) -> pd.DataFrame:
        """a dataframe of shape (484, 197)"""
        df = pd.read_csv(
            os.path.join(self.attributes_path, "attributes_hydroatlas_.csv"),
            index_col=0)
        # because self.other_attributes() has a column named 'area'
        df.rename(columns={'area': 'area_hydroatlas'}, inplace=True)
        return df

    def other_attributes(self) -> pd.DataFrame:
        """a dataframe of shape (484, 7)"""
        return pd.read_csv(
            os.path.join(self.attributes_path, "attributes_other_ss.csv"),
            index_col=0)

    def _static_data(self) -> pd.DataFrame:
        df = pd.concat([
            self.caravan_attributes(),
            self.hydroatlas_attributes(),
            self.other_attributes()
        ], axis=1)
        df.rename(columns=self.static_map, inplace=True)
        return df

    def _read_dynamic_for_stn(self, station: str) -> pd.DataFrame:

        station = station.split('_')[1]

        df = pd.concat([
            self._read_q_for_stn(station),
            self._read_aemet_for_stn(station),
            self._read_bull_for_stn(station),
            self._read_era5_land_for_stn(station),
            self._read_emo1_arc_for_stn(station)
        ], axis=1)
        df.index.name = 'time'
        df.columns.name = 'dynamic_features'

        df.rename(columns=self.dyn_map, inplace=True)

        return df

    def _read_dynamic(
            self,
            stations,
            dynamic_features,
            st=None,
            en=None) -> dict:

        dynamic_features = check_attributes(dynamic_features, self.dynamic_features)
        stations = check_attributes(stations, self.stations())

        if st is None:
            st = self.start
        if en is None:
            en = self.end

        cpus = self.processes or min(get_cpus(), 64)

        if len(stations) > 10 and cpus > 1:

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

    def _read_q_for_stn(self, station) -> pd.DataFrame:
        """a dataframe of shape (time, 1)"""
        if self.ftype == "netcdf":
            fpath = os.path.join(self.q_path, f'streamflow_{station}.nc')
            df = xr.load_dataset(fpath).to_dataframe()
        else:
            fpath = os.path.join(self.q_path, f'streamflow_{station}.csv')
            df = pd.read_csv(fpath, index_col='date', parse_dates=True)
        df.index.name = 'time'
        df.columns.name = 'dynamic_features'
        return df

    def _read_aemet_for_stn(self, station) -> pd.DataFrame:
        """
        reads a dataframe of shape (time, 5)

        'temperature_max_AEMET',
        'temperature_min_AEMET',
        'temperature_mean_AEMET',
        'total_precipitation_AEMET',
        'potential_evapotranspiration_AEMET'
        """
        if self.ftype == "netcdf":
            fpath = os.path.join(self.aemet_path, f'AEMET_{station}.nc')
            df = xr.load_dataset(fpath).to_dataframe()
        else:
            fpath = os.path.join(self.aemet_path, f'AEMET_{station}.csv')
            df = pd.read_csv(fpath, index_col='date', parse_dates=True)
        df.index.name = 'time'
        df.columns.name = 'dynamic_features'
        df.columns = [col + '_AEMET' for col in df.columns]
        return df

    def _read_bull_for_stn(self, station) -> pd.DataFrame:
        """a dataframe of shape (time, 39) except for stn 3163"""
        if self.ftype == "netcdf":
            fpath = os.path.join(self.bull_path, f'BULL_{station}.nc')
            df = xr.load_dataset(fpath).to_dataframe()
        else:
            fpath = os.path.join(self.bull_path, f'BULL_{station}.csv')
            df = pd.read_csv(fpath, index_col='date', parse_dates=True)
        df.index.name = 'time'
        df.columns.name = 'dynamic_features'
        df.columns = [col + '_BULL' for col in df.columns]  # todo: why are we adding _BULL to the columns
        if len(df.columns) == 15:
            # add missing columns
            for col in BUL_COLUMNS:
                if col not in df.columns:
                    df[col] = None
        return df

    def _read_era5_land_for_stn(self, station) -> pd.DataFrame:
        """a dataframe of shape (time, 5) with following columns
            - 'temperature_max_ERA5_Land',
            - 'temperature_min_ERA5_Land',
            - 'temperature_mean_ERA5_Land',
            - 'total_precipitation_ERA5_Land',
            - 'potential_evapotranspiration_ERA5_Land'
        """
        if self.ftype == "netcdf":
            fpath = os.path.join(self.era5_land_path, f'ERA5_Land_{station}.nc')
            df = xr.load_dataset(fpath).to_dataframe()
        else:
            fpath = os.path.join(self.era5_land_path, f'ERA5_Land_{station}.csv')
            df = pd.read_csv(fpath, index_col='date', parse_dates=True)
        df.index.name = 'time'
        df.columns.name = 'dynamic_features'
        df.columns = [col + '_ERA5_Land' for col in df.columns]
        return df

    def _read_emo1_arc_for_stn(self, station) -> pd.DataFrame:
        """a dataframe of shape (time, 5) with following columns
            - 'temperature_max_EMO1_arc'
            - 'temperature_min_EMO1_arc'
            - 'temperature_mean_EMO1_arc'
            - 'total_precipitation_EMO1_arc'
            - 'potential_evapotranspiration_EMO1_arc'
        """
        if self.ftype == "netcdf":
            fpath = os.path.join(self.emo1_arc_path, f'EMO1_{station}.nc')
            df = xr.load_dataset(fpath).to_dataframe()
        else:
            fpath = os.path.join(self.emo1_arc_path, f'EMO1_{station}.csv')
            df = pd.read_csv(fpath, index_col='date', parse_dates=True)
        df.index.name = 'time'
        df.columns.name = 'dynamic_features'
        df.columns = [col + '_EMO1_arc' for col in df.columns]
        return df
