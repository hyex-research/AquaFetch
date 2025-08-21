import os
import json
import glob
import time
import shutil
import zipfile
import warnings
import concurrent.futures as cf
from typing import Union, List, Dict

import numpy as np
import pandas as pd

from .utils import _RainfallRunoff
from .._geom_utils import utm_to_lat_lon
from ..utils import get_cpus, download_and_unzip
from ..utils import check_attributes, download, unzip

from .._backend import netCDF4, xarray as xr

from ._map import (
    observed_streamflow_cms,
    observed_streamflow_mm,
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
    soil_moisture_layer1,
    soil_moisture_layer2,
    soil_moisture_layer3,
    soil_moisture_layer4,
    mean_dewpoint_temperature_at_2m,
    catchment_area_with_specifier,
    snow_water_equivalent_with_specifier,
    snow_depth,
)

from ._map import (
    catchment_area,
    gauge_latitude,
    gauge_longitude,
    slope,
    gauge_elevation_meters,
    catchment_elevation_meters,
    urban_fraction,
    urban_fraction_with_specifier,
    grass_fraction,
    grass_fraction_with_specifier,
    crop_fraction,
    crop_fraction_with_specifier,
    catchment_perimeter,
    aridity_index,
    med_catchment_elevation_meters,
    soil_depth,
    population_density,
    )

# directory separator
SEP = os.sep


class CAMELS_US(_RainfallRunoff):
    """
    This is a dataset of 671 US catchments with 59 static features
    and 8 dyanmic features for each catchment. The dyanmic features are
    timeseries from 1980-01-01 to 2014-12-31. This class
    downloads and processes CAMELS dataset of 671 catchments named as CAMELS
    from `ucar.edu <https://ral.ucar.edu/solutions/products/camels>`_
    following `Newman et al., 2015 <https://doi.org/10.5194/hess-19-209-2015>`_ ,
    `Newman et al., 2022 <https://gdex.ucar.edu/dataset/camels.html.>`_ and
    `Addor et al., 2017 <https://hess.copernicus.org/articles/21/5293/2017/>`_.

    Please note this data is also known as "CAMELS" however, we have named it CAMELS_US
    to differentiate it from other CAMELS like datasts from other parts of the world.

    Examples
    --------
    >>> from aqua_fetch import CAMELS_US
    >>> dataset = CAMELS_US()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='11478500', as_dataframe=True)
    >>> df = dynamic['11478500'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (12784, 8)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       671
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (67 out of 671)
       67
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(12784, 8), (12784, 8), (12784, 8),... (12784, 8), (12784, 8)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('11478500', as_dataframe=True,
    ...  dynamic_features=['pcp_mm', 'solrad_wm2', 'airtemp_C_max', 'airtemp_C_min', 'q_cms_obs'])
    >>> dynamic['11478500'].shape
       (12784, 5)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='11478500', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['11478500'].shape
    ((1, 59), 1, (12784, 8))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 12784, 'dynamic_features': 8})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (671, 2)
    >>> dataset.stn_coords('11478500')  # returns coordinates of station whose id is 11478500
        40.480419	-123.890877
    >>> dataset.stn_coords(['11478500', '14020000'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('11478500')
    # get coordinates of two stations
    >>> dataset.area(['11478500', '14020000'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('11478500')

    """
    DATASETS = ['CAMELS_US']

    url = {
        'camels_attributes_v2.0.pdf': 'https://gdex.ucar.edu/dataset/camels/file/',
        'camels_attributes_v2.0.xlsx': 'https://gdex.ucar.edu/dataset/camels/file/',
        'camels_clim.txt': 'https://gdex.ucar.edu/dataset/camels/file/',
        'camels_geol.txt': 'https://gdex.ucar.edu/dataset/camels/file/',
        'camels_hydro.txt': 'https://gdex.ucar.edu/dataset/camels/file/',
        'camels_name.txt': 'https://gdex.ucar.edu/dataset/camels/file/',
        'camels_soil.txt': 'https://gdex.ucar.edu/dataset/camels/file/',
        'camels_topo.txt': 'https://gdex.ucar.edu/dataset/camels/file/',
        'camels_vege.txt': 'https://gdex.ucar.edu/dataset/camels/file/',
        'readme.txt': 'https://gdex.ucar.edu/dataset/camels/file/',
        'basin_timeseries_v1p2_metForcing_obsFlow.zip': 'https://gdex.ucar.edu/dataset/camels/file/',
        'basin_set_full_res.zip': 'https://gdex.ucar.edu/dataset/camels/file/',
    }

    folders = {'basin_mean_daymet': f'basin_mean_forcing{SEP}daymet',
               'basin_mean_maurer': f'basin_mean_forcing{SEP}maurer',
               'basin_mean_nldas': f'basin_mean_forcing{SEP}nldas',
               'basin_mean_v1p15_daymet': f'basin_mean_forcing{SEP}v1p15{SEP}daymet',
               'basin_mean_v1p15_nldas': f'basin_mean_forcing{SEP}v1p15{SEP}nldas',
               'elev_bands': f'elev{SEP}daymet',
               'hru': f'hru_forcing{SEP}daymet'}

    dynamic_features_ = ['dayl(s)', 'prcp(mm/day)', 'srad(W/m2)',
                         'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)', 'Flow']

    def __init__(
            self,
            path:Union[str, os.PathLike]=None,
            data_source: str = 'basin_mean_daymet',
            **kwargs
    ):

        """
        parameters
        ----------
        path : str
            If the data is alredy downloaded then provide the complete
            path to it. If None, then the data will be downloaded.
            The data is downloaded once and therefore susbsequent
            calls to this class will not download the data unless
            ``overwrite`` is set to True.
        data_source : str
            allowed values are
                - basin_mean_daymet
                - basin_mean_maurer
                - basin_mean_nldas
                - basin_mean_v1p15_daymet
                - basin_mean_v1p15_nldas
                - elev_bands
                - hru
        """
        assert data_source in self.folders, f'allwed data sources are {self.folders.keys()}'
        self.data_source = data_source

        super().__init__(path=path, name="CAMELS_US", **kwargs)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        for fname, url in self.url.items():

            fpath = os.path.join(self.path, fname)
            furl = f"{url}{fname}"

            if not os.path.exists(fpath) or self.overwrite:

                if self.verbosity:
                    print(f"downloading {fname} from {url}")

                download(furl, self.path, fname, verbosity=self.verbosity)

                unzip(self.path, verbosity=self.verbosity)

        self.dataset_dir = os.path.join(
            self.path, 
            f'basin_timeseries_v1p2_metForcing_obsFlow{SEP}basin_dataset_public_v1p2')

        self._static_features = self._static_data().columns.tolist()
        self._maybe_to_netcdf()

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(
            self.path,
            "basin_set_full_res",
            "HCDN_nhru_final_671.shp"
        )

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'area_gages2': catchment_area(),
                'slope_mean': slope('mkm-1'),
                'gauge_lat': gauge_latitude(),
                'gauge_lon': gauge_longitude(),
        }

    @property
    def dyn_map(self):
        return {
            'Flow': observed_streamflow_cms(),  # todo : check units
            'tmin(C)': min_air_temp(),
            'tmax(C)': max_air_temp(),
            'prcp(mm/day)': total_precipitation(),
            'swe(mm)': snow_water_equivalent(),
            'pet_mean': total_potential_evapotranspiration(),
            'vp(Pa)': mean_vapor_pressure(),  # todo: convert frmo Pa to hpa
            'srad(W/m2)': solar_radiation(),
        }

    @property
    def dyn_factors(self) -> Dict[str, float]:
        return {
            observed_streamflow_cms(): 0.0283168,
        }

    @property
    def start(self):
        return "19800101"

    @property
    def end(self):
        return "20141231"

    @property
    def static_features(self)->List[str]:
        return self._static_features

    @property
    def dynamic_features(self) -> List[str]:
        return [self.dyn_map.get(feat, feat) for feat in self.dynamic_features_]

    def stations(self) -> list:
        stns = []
        for _dir in os.listdir(os.path.join(self.dataset_dir, 'usgs_streamflow')):
            cat = os.path.join(self.dataset_dir, f'usgs_streamflow{SEP}{_dir}')
            stns += [fname.split('_')[0] for fname in os.listdir(cat)]

        # remove stations for which static values are not available
        for stn in ['06775500', '06846500', '09535100']:
            stns.remove(stn)

        return stns

    def _read_stn_dyn(
            self,
            stn:str,
    ):

        assert isinstance(stn, str)
        df = None
        dir_name = self.folders[self.data_source]
        for cat in os.listdir(os.path.join(self.dataset_dir, dir_name)):
            cat_dirs = os.listdir(os.path.join(self.dataset_dir, f'{dir_name}{SEP}{cat}'))
            stn_file = f'{stn}_lump_cida_forcing_leap.txt'
            if stn_file in cat_dirs:
                df = pd.read_csv(os.path.join(self.dataset_dir,
                                                f'{dir_name}{SEP}{cat}{SEP}{stn_file}'),
                                    sep="\s+|;|:",
                                    skiprows=4,
                                    engine='python',
                                    names=['Year', 'Mnth', 'Day', 'Hr', 'dayl(s)', 'prcp(mm/day)', 'srad(W/m2)',
                                        'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)'],
                                    )
                df.index = pd.to_datetime(
                    df['Year'].map(str) + '-' + df['Mnth'].map(str) + '-' + df['Day'].map(str))

        flow_dir = os.path.join(self.dataset_dir, 'usgs_streamflow')
        for cat in os.listdir(flow_dir):
            cat_dirs = os.listdir(os.path.join(flow_dir, cat))
            stn_file = f'{stn}_streamflow_qc.txt'
            if stn_file in cat_dirs:
                fpath = os.path.join(flow_dir, f'{cat}{SEP}{stn_file}')
                q_df = pd.read_csv(fpath,
                                    sep=r"\s+",
                                    names=['station', 'Year', 'Month', 'Day', 'Flow', 'Flag'],
                                    engine='python')
                q_df.index = pd.to_datetime(
                    q_df['Year'].map(str) + '-' + q_df['Month'].map(str) + '-' + q_df['Day'].map(str))

        stn_df = pd.concat([
            df[['dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)']],
            q_df['Flow']],
            axis=1)

        stn_df.rename(columns=self.dyn_map, inplace=True)

        for col, fact in self.dyn_factors.items():
            if col in stn_df.columns:
                stn_df[col] *= fact

        return stn_df

    def _static_data(self)->pd.DataFrame:
        static_fpath = os.path.join(self.path, 'static_features.csv')
        if not os.path.exists(static_fpath):
            files = glob.glob(f"{self.path}/*.txt")

            static_dfs = []
            for f in files:
                if not f.endswith('readme.txt'):
                    if self.verbosity>2:
                        print(f"reading {f}")
                    # index should be read as string

                    _df = pd.read_csv(f, sep=';', index_col='gauge_id', dtype={'gauge_id': str})

                    static_dfs.append(_df)
            static_df = pd.concat(static_dfs, axis=1)
            static_df.to_csv(static_fpath, index_label='gauge_id')
        else:  # index should be read as string bcs it has 0s at the start
            static_df = pd.read_csv(static_fpath, index_col='gauge_id', dtype={'gauge_id': str})

        static_df.index = static_df.index.astype(str)

        static_df = static_df#.loc[stn_id][features]
        if isinstance(static_df, pd.Series):
            static_df = pd.DataFrame(static_df).transpose()

        static_df.rename(columns=self.static_map, inplace=True)
        
        return static_df


class CAMELS_GB(_RainfallRunoff):
    """
    This is a dataset of 671 catchments with 145 static features
    and 10 dyanmic features for each catchment following the work of
    `Coxon et al., 2020 <https://doi.org/10.5194/essd-12-2459-2020>`__.
    The dyanmic features are timeseries from 1970-10-01 to 2015-09-30.
    The data is downloaded from `ceh website <https://data-package.ceh.ac.uk/data/8344e4f3-d2ea-44f5-8afa-86d2987543a9.zip>`_

    Examples
    --------
    >>> from aqua_fetch import CAMELS_GB
    >>> dataset = CAMELS_GB()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='38017', as_dataframe=True)
    >>> df = dynamic['38017'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (26388, 28)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       671
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (67 out of 671)
       67
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(26388, 28), (26388, 28), (26388, 28),... (26388, 28), (26388, 28)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('38017', as_dataframe=True,
    ...  dynamic_features=['windspeed_mps', 'airtemp_C_mean', 'pet_mm', 'pcp_mm', 'q_cms_obs'])
    >>> dynamic['38017'].shape
       (26388, 4)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='38017', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['38017'].shape
    ((1, 145), 1, (26388, 28))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 26388, 'dynamic_features': 28})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (671, 2)
    >>> dataset.stn_coords('38017')  # returns coordinates of station whose id is 38017
        51.880001	-0.28
    >>> dataset.stn_coords(['38017', '42001'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('38017')
    # get coordinates of two stations
    >>> dataset.area(['38017', '42001'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('38017')
    """
    dynamic_features_ = ["precipitation", "pet", "temperature", "discharge_spec",
                         "discharge_vol", "peti",
                         "humidity", "shortwave_rad", "longwave_rad", "windspeed"]

    def __init__(self, path=None, **kwargs):
        """
        parameters
        ------------
        path : str
            If the data is alredy downloaded then provide the complete
            path to it. If None, then the data will be downloaded.
            The data is downloaded once and therefore susbsequent
            calls to this class will not download the data unless
            ``overwrite`` is set to True.
        """
        super().__init__(name="CAMELS_GB", path=path, **kwargs)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if not os.path.exists(os.path.join(self.path, 'camels_gb')):
            download(
                outdir=self.path,
                url="https://data-package.ceh.ac.uk/data/8344e4f3-d2ea-44f5-8afa-86d2987543a9.zip",
                fname="camels_gb.zip"
            )
            if self.verbosity > 0:
                print("unzipping the downloaded file")
            unzip(self.path, verbosity=self.verbosity)

            # rename the folder camels_gb/8344e4f3-d2ea-44f5-8afa-86d2987543a9 to camels_gb/caemls_gb
            shutil.move(
                os.path.join(self.path, 'camels_gb', '8344e4f3-d2ea-44f5-8afa-86d2987543a9'),
                os.path.join(self.path, 'camels_gb', 'camels_gb')
            )
        else:
            if self.verbosity > 0:
                print(f"dataset is already available at {self.path}")

        self._static_features = self._static_data().columns.tolist()

        self._maybe_to_netcdf()

        if not os.path.exists(self.boundary_file):
            unzip(self.data_path, verbosity=self.verbosity)

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(
            self.data_path,
            "CAMELS_GB_catchment_boundaries",
            "CAMELS_GB_catchment_boundaries.shp"
        )

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'area': catchment_area(),
                'slope_fdc': slope(''),
                'gauge_lat': gauge_latitude(),
                'gauge_lon': gauge_longitude(),
        }

    @property
    def dyn_map(self):
        # table 1 in https://essd.copernicus.org/articles/12/2459/2020/#&gid=1&pid=1
        return {
            'discharge_vol': observed_streamflow_cms(),
            'discharge_spec': observed_streamflow_mm(),
            'temperature': mean_air_temp(),
            'humidity': mean_rel_hum(),  # todo: convert from g/kg to %
            'windspeed': mean_windspeed(),
            'precipitation': total_precipitation(),
            'pet': total_potential_evapotranspiration(),
            'peti': total_potential_evapotranspiration_with_specifier('intercep'),
            'shortwave_rad': solar_radiation(),
            'longwave_rad': downward_longwave_radiation(),
        }

    @property
    def data_path(self):
        return os.path.join(self.path, 'camels_gb', 'camels_gb', 'data')

    @property
    def static_attribute_categories(self) -> list:
        features = []
        for f in os.listdir(self.data_path):
            if os.path.isfile(os.path.join(self.data_path, f)) and f.endswith('csv'):
                features.append(f.split('_')[2])

        return features

    @property
    def start(self):
        return pd.Timestamp("19701001")

    @property
    def end(self):
        return pd.Timestamp("20150930")

    @property
    def static_features(self)->List[str]:
        return self._static_features

    @property
    def dynamic_features(self) -> List[str]:
        return [self.dyn_map.get(feat, feat) for feat in self.dynamic_features_]

    def stations(self, to_exclude=None):
        # CAMELS_GB_hydromet_timeseries_StationID_number
        path = os.path.join(self.data_path, 'timeseries')
        gauge_ids = []
        for f in os.listdir(path):
            gauge_ids.append(f.split('_')[4])

        return gauge_ids

    @property
    def _mm_feature_name(self) -> str:
        return observed_streamflow_mm()

    @property
    def _area_name(self) -> str:
        return 'area'

    @property
    def _coords_name(self) -> List[str]:
        return ['gauge_lat', 'gauge_lon']

    def _read_stn_dyn(
        self,
        stn:str
    )->pd.DataFrame:
        # making one separate dataframe for one station
        path = os.path.join(self.data_path, f"timeseries")
        fname = f"CAMELS_GB_hydromet_timeseries_{stn}_19701001-20150930.csv"

        df = pd.read_csv(os.path.join(path, fname), index_col='date')
        df.index = pd.to_datetime(df.index)
        df.index.freq = pd.infer_freq(df.index)

        df.rename(columns=self.dyn_map, inplace=True)

        return df

    def _static_data(self)->pd.DataFrame:
        static_fpath = os.path.join(self.data_path, 'static_features.csv')
        if os.path.exists(static_fpath):
            static_df = pd.read_csv(static_fpath, index_col='gauge_id')
        else:
            files = glob.glob(f"{self.data_path}/*.csv")
            static_dfs = []
            for f in files:
                _df = pd.read_csv(f, index_col='gauge_id')
                static_dfs.append(_df)
            static_df = pd.concat(static_dfs, axis=1)
            static_df.to_csv(static_fpath)

        static_df.index = static_df.index.astype(str)

        static_df.rename(columns=self.static_map, inplace=True)

        return static_df


class CAMELS_AUS(_RainfallRunoff):
    """
    This is a dataset of 561 Australian catchments with 187 static features and
    28 dyanmic features for each catchment. The dyanmic features are timeseries
    from 1950-01-01 to 2022-03-31. By default this class reads version 2 of CAMELS-AUS dataset
    following `Fowler et al., 2024 <https://doi.org/10.5194/essd-2024-263>`_ .

    If ``version`` is 1 then this class reads data following `Fowler et al., 2021 <https://doi.org/10.5194/essd-13-3847-2021>`_
    which is a dataset of 222 Australian catchments with 161 static features
    and 26 dyanmic features for each catchment. The dyanmic features are
    timeseries from 1957-01-01 to 2018-12-31.

    Examples
    --------
    >>> from aqua_fetch import CAMELS_AUS
    >>> dataset = CAMELS_AUS()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='912101A', as_dataframe=True)
    >>> df = dynamic['912101A'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (26388, 28)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       561
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (56 out of 561)
       56
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(26388, 28), (26388, 28), (26388, 28),... (26388, 28), (26388, 28)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('912101A', as_dataframe=True,
    ...  dynamic_features=['airtemp_C_awap_max', 'pcp_mm_awap', 'et_morton_actual_SILO', 'q_cms_obs'])
    >>> dynamic['912101A'].shape
       (26388, 4)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='912101A', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['912101A'].shape
    ((1, 187), 1, (26388, 28))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 26388, 'dynamic_features': 28})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (561, 2)
    >>> dataset.stn_coords('912101A')  # returns coordinates of station whose id is 912101A
        -38.214199	-71.8283
    >>> dataset.stn_coords(['912101A', '912105A'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('912101A')
    # get coordinates of two stations
    >>> dataset.area(['912101A', '912105A'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('912101A')
    ...
    # The version 1 can be of CAMELS_AUS can be accessed as below
    >>> dataset = CAMELS_AUS(version=1)
    >>> len(dataset.stations())
    222
    >>> _, dynamic = dataset.fetch(stations='912101A', as_dataframe=True)
    >>> dynamic['912101A'].shape
    (23376, 26)
    """

    url = 'https://doi.pangaea.de/10.1594/PANGAEA.921850'
    url_v2 = "https://zenodo.org/records/13350616"
    urls = {1: {
        "01_id_name_metadata.zip": "https://download.pangaea.de/dataset/921850/files/",
        "02_location_boundary_area.zip": "https://download.pangaea.de/dataset/921850/files/",
        "03_streamflow.zip": "https://download.pangaea.de/dataset/921850/files/",
        "04_attributes.zip": "https://download.pangaea.de/dataset/921850/files/",
        "05_hydrometeorology.zip": "https://download.pangaea.de/dataset/921850/files/",
        "CAMELS_AUS_Attributes&Indices_MasterTable.csv": "https://download.pangaea.de/dataset/921850/files/",
        # "Units_01_TimeseriesData.pdf": "https://download.pangaea.de/dataset/921850/files/",
        # "Units_02_AttributeMasterTable.pdf": "https://download.pangaea.de/dataset/921850/files/",
    },
        2: {
            "01_id_name_metadata.zip": "https://zenodo.org/records/13350616/files/",
            "02_location_boundary_area.zip": "https://zenodo.org/records/13350616/files/",
            "03_streamflow.zip": "https://zenodo.org/records/13350616/files/",
            "04_attributes.zip": "https://zenodo.org/records/13350616/files/",
            "05_hydrometeorology.zip": "https://zenodo.org/records/13350616/files/",
            "CAMELS_AUS_Attributes&Indices_MasterTable.csv": "https://zenodo.org/records/13350616/files/",
        }
    }

    folders = {1: {
        'streamflow_MLd': f'03_streamflow{SEP}03_streamflow',
        'streamflow_MLd_inclInfilled': f'03_streamflow{SEP}03_streamflow',
        'streamflow_mmd': f'03_streamflow{SEP}03_streamflow',

        'et_morton_actual_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
        'et_morton_point_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
        'et_morton_wet_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
        'et_short_crop_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
        'et_tall_crop_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
        'evap_morton_lake_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
        'evap_pan_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
        'evap_syn_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',

        'precipitation_AWAP': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}01_precipitation_timeseries',
        'precipitation_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}01_precipitation_timeseries',
        'precipitation_var_AWAP': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}01_precipitation_timeseries',

        'solarrad_AWAP': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AWAP',
        'tmax_AWAP': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AWAP',
        'tmin_AWAP': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AWAP',
        'vprp_AWAP': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AWAP',

        'mslp_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        'radiation_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        'rh_tmax_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        'rh_tmin_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        'tmax_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        'tmin_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        'vp_deficit_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        'vp_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
    },
        2: {
            'streamflow_MLd': f'03_streamflow{SEP}03_streamflow',
            'streamflow_MLd_inclInfilled': f'03_streamflow{SEP}03_streamflow',
            'streamflow_mmd': f'03_streamflow{SEP}03_streamflow',

            'et_morton_actual_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
            'et_morton_point_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
            'et_morton_wet_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
            'et_short_crop_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
            'et_tall_crop_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
            'evap_morton_lake_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
            'evap_pan_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
            'evap_syn_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',

            'precipitation_AGCD': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}01_precipitation_timeseries',
            'precipitation_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}01_precipitation_timeseries',
            'precipitation_var_AGCD': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}01_precipitation_timeseries',

            # 'solarrad_AWAP': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AGCD{SEP}solarrad_AWAP',
            'tmax_AGCD': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AGCD',
            'tmin_AGCD': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AGCD',
            'vapourpres_h09_AGCD': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AGCD',
            'vapourpres_h15_AGCD': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AGCD',

            'mslp_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
            'radiation_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
            'rh_tmax_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
            'rh_tmin_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
            'tmax_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
            'tmin_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
            'vp_deficit_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
            'vp_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        }
    }

    def __init__(
            self,
            path: str = None,
            version: int = 2,
            to_netcdf: bool = True,
            overwrite: bool = False,
            verbosity: int = 1,
            **kwargs
    ):
        """
        Arguments:
            path: path where the CAMELS_AUS dataset has been downloaded. This path
                must contain five zip files and one xlsx file. If None, then the
                data will be downloaded.
            version: version of the dataset to download. Allowed values are 1 and 2.
            to_netcdf :
        """
        self.version = version

        super().__init__(path=path, verbosity=verbosity, **kwargs)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        for _file, url in self.urls[version].items():
            fpath = os.path.join(self.path, _file)

            if os.path.exists(fpath) and overwrite:
                os.remove(fpath)
                if verbosity > 0: print(f"Re-downloading {_file} from {url + _file} at {fpath}")
                download(url + _file, outdir=self.path, fname=_file, verbosity=self.verbosity)

            elif not os.path.exists(fpath):
                if verbosity > 0:
                    print(f"Downloading {_file} from {url + _file} at {fpath}")
                download(url + _file, outdir=self.path, fname=_file, verbosity=self.verbosity)
            elif verbosity > 0:
                print(f"{_file} already exists at {self.path}")

        # maybe the .zip file has been downloaded previously but not unzipped
        unzip(self.path, verbosity=verbosity, overwrite=overwrite)

        if netCDF4 is None:
            to_netcdf = False

        # if to_netcdf:
        self._maybe_to_netcdf()

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(
            self.path,
            "02_location_boundary_area",
            "02_location_boundary_area",
            "shp",
            "CAMELS_AUS_Boundaries_adopted.shp" if self.version == 1 else "CAMELS_AUS_v2_Boundaries_adopted.shp"
        )

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'catchment_area': catchment_area(),
                'maen_slope_pct': slope('%'),
                'lat_outlet': gauge_latitude(),
                'long_outlet': gauge_longitude(),
        }

    @property
    def dyn_map(self):
        # table 2 in https://essd.copernicus.org/articles/13/3847/2021/#&gid=1&pid=1
        return {
            'streamflow_MLd': observed_streamflow_cms(),
            'streamflow_mmd': observed_streamflow_mm(),
            'tmin_SILO': min_air_temp_with_specifier('silo'),
            'tmax_SILO': max_air_temp_with_specifier('silo'),
            'tmin_AWAP': min_air_temp_with_specifier('awap'),
            'tmax_AWAP': max_air_temp_with_specifier('awap'),
            'et_morton_actual_SILO': actual_evapotranspiration_with_specifier('silo_morton'),
            'et_morton_point_SILO': actual_evapotranspiration_with_specifier('silo_morton_point'),
            'et_short_crop_SILO': actual_evapotranspiration_with_specifier('silo_short_crop'),
            'et_tall_crop_SILO': actual_evapotranspiration_with_specifier('silo_tall_crop'),
            'precipitation_AWAP': total_precipitation_with_specifier('awap'),
            'precipitation_SILO': total_precipitation_with_specifier('silo'),
            'solarrad_AWAP': solar_radiation_with_specifier('awap'),  # convert MJ/m2/day to W/m2
            'radiation_SILO': solar_radiation_with_specifier('silo'),  # convert MJ/m2/day to W/m2
            'vp_SILO': mean_vapor_pressure_with_specifier('silo'),
            'vprp_AWAP': mean_vapor_pressure_with_specifier('awap'),
            'rh_tmax_SILO': mean_rel_hum_with_specifier('silo_tmax'),
            'rh_tmin_SILO': mean_rel_hum_with_specifier('silo_tmin'),
            'vapourpres_h09_AGCD': mean_vapor_pressure_with_specifier('agcd_h09'),
            'vapourpres_h15_AGCD': mean_vapor_pressure_with_specifier('agcd_h15'),
            'tmin_AGCD': min_air_temp_with_specifier('agcd'),
            'tmax_AGCD': max_air_temp_with_specifier('agcd'),
            'precipitation_AGCD': total_precipitation_with_specifier('agcd'),
            #'mslp_SILO': mean_sea_level_pressure_with_specifier('silo'),
        }

    @property
    def dyn_factors(self):
        return {
            observed_streamflow_cms(): 0.01157,
        }

    @property
    def dyn_generators(self):
        if self.version == 1:
            return {
    # new column to be created : function to be applied, inputs
    mean_air_temp_with_specifier('silo'): (self.mean_temp, (min_air_temp_with_specifier('silo'), max_air_temp_with_specifier('silo'))), 
    mean_air_temp_with_specifier('awap'): (self.mean_temp, (min_air_temp_with_specifier('awap'), max_air_temp_with_specifier('awap'))),
        }
        else:
            return {
    # new column to be created : function to be applied, inputs
    mean_air_temp_with_specifier('silo'): (self.mean_temp, (min_air_temp_with_specifier('silo'), max_air_temp_with_specifier('silo'))),
    mean_air_temp_with_specifier('agcd'): (self.mean_temp, (min_air_temp_with_specifier('agcd'), max_air_temp_with_specifier('agcd'))),
        }

    @property
    def start(self):
        return "19500101"

    @property
    def end(self):
        return "20181231" if self.version == 1 else "20220331"

    @property
    def location(self):
        return "Australia"

    def stations(self, as_list=True) -> list:
        fname = os.path.join(self.path, f"01_id_name_metadata{SEP}01_id_name_metadata{SEP}id_name_metadata.csv")
        df = pd.read_csv(fname)
        if as_list:
            return df['station_id'].to_list()
        else:
            return df

    @property
    def static_attribute_categories(self):
        features = []
        path = os.path.join(self.path, f'04_attributes{SEP}04_attributes')
        for f in os.listdir(path):
            if os.path.isfile(os.path.join(path, f)) and f.endswith('csv'):
                f = str(f.split('.csv')[0])
                features.append(''.join(f.split('_')[2:]))
        return features

    @property
    def static_features(self) -> List[str]:

        return self._static_data().columns.tolist()

    @property
    def dynamic_features(self) -> list:
        return [self.dyn_map.get(feat, feat) for feat in list(self.folders[self.version].keys())] + list(self.dyn_generators.keys())

    def _static_data(self, #stations, #features,
                     st=None, en=None):

        #features = check_attributes(features, self.static_features, 'static_features')
        static_fname = 'CAMELS_AUS_Attributes&Indices_MasterTable.csv'
        static_fpath = os.path.join(self.path, static_fname)
        static_df = pd.read_csv(static_fpath, index_col='station_id')

        static_df.index = static_df.index.astype(str)
        #static_df = static_df.loc[stations]#[features]
        if isinstance(static_df, pd.Series):
            static_df = pd.DataFrame(static_df).transpose()
        
        static_df.rename(columns=self.static_map, inplace=True)

        return static_df

    def _read_dynamic(
            self, 
            stations, 
            dynamic_features, 
            st=None,
            en=None,
            ):

        st, en = self._check_length(st, en)
        dynamic_features = check_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')

        dyn_attrs = {}
        dyn = {}
        dtype = {stn: np.float32 for stn in stations}
        dtype.update({'year': str, 'month': str, 'day': str})

        for _attr in list(self.folders[self.version].keys()):

            if self.dyn_map.get(_attr, _attr) not in dynamic_features:
                continue

            _path = os.path.join(self.path, f'{self.folders[self.version][_attr]}{SEP}{_attr}.csv')
            
            attr_df = pd.read_csv(_path, na_values=['-99.99'], usecols=['year', 'month', 'day'] + stations,
                                  dtype=dtype)
            attr_df.index = pd.to_datetime(attr_df[['year', 'month', 'day']])

            dyn_attrs[_attr] = attr_df[stations]

        # making one separate dataframe for one station
        for idx, stn in enumerate(stations):
            stn_df = pd.DataFrame()
            for attr, attr_df in dyn_attrs.items():
                # if attr in dynamic_features:
                stn_df[attr] = attr_df[stn]

            stn_df.rename(columns=self.dyn_map, inplace=True)

            for col, fact in self.dyn_factors.items():
                if col in stn_df.columns:
                    stn_df[col] = stn_df[col] * fact

            for new_col, (func, old_col) in self.dyn_generators.items():
                if isinstance(old_col, str):
                    if old_col in stn_df.columns:
                        # name of Series to func should be same as station id
                        stn_df[new_col] = func(pd.Series(stn_df[old_col], name=stn))
                else:
                    assert isinstance(old_col, tuple)
                    if all([col in stn_df.columns for col in old_col]):
                        # feed all old_cols to the function
                        stn_df[new_col] = func(*[pd.Series(stn_df[col], name=stn) for col in old_col])
            
            if self.verbosity>1 and idx % 100 == 0:
                print(f"processed {idx} stations")

            stn_df.index.name = 'time'
            stn_df.columns.name = 'dynamic_features'
            dyn[stn] = stn_df.loc[st:en, dynamic_features]

        return dyn


class CAMELS_CL(_RainfallRunoff):
    """
    This is a dataset of 516 Chilean catchments with
    104 static features and 12 dyanmic features for each catchment.
    The dyanmic features are timeseries from 1913-02-15 to 2018-03-09.
    This class downloads and processes CAMELS dataset of Chile following the work of
    `Alvarez-Garreton et al., 2018 <https://doi.org/10.5194/hess-22-5817-2018>`_ .

    Examples
    ---------
    >>> from aqua_fetch import CAMELS_CL
    >>> dataset = CAMELS_CL()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='8350001', as_dataframe=True)
    >>> df = dynamic['8350001'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (38374, 12)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       516
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (51 out of 516)
       51
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(38374, 12), (38374, 12), (38374, 12),... (38374, 12), (38374, 12)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('8350001', as_dataframe=True,
    ...  dynamic_features=['pet_mm_hargreaves', 'pcp_mm_mswep', 'airtemp_C_mean', 'q_cms_obs'])
    >>> dynamic['8350001'].shape
       (38374, 4)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='8350001', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['8350001'].shape
    ((1, 104), 1, (38374, 12))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 38374, 'dynamic_features': 12})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (516, 2)
    >>> dataset.stn_coords('8350001')  # returns coordinates of station whose id is 8350001
        -38.214199	-71.8283
    >>> dataset.stn_coords(['8350001', '3820003'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('8350001')
    # get coordinates of two stations
    >>> dataset.area(['8350001', '3820003'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('8350001')
    """

    urls = {
        "1_CAMELScl_attributes.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "2_CAMELScl_streamflow_m3s.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "3_CAMELScl_streamflow_mm.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "4_CAMELScl_precip_cr2met.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "5_CAMELScl_precip_chirps.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "6_CAMELScl_precip_mswep.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "7_CAMELScl_precip_tmpa.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "8_CAMELScl_tmin_cr2met.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "9_CAMELScl_tmax_cr2met.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "10_CAMELScl_tmean_cr2met.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "11_CAMELScl_pet_8d_modis.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "12_CAMELScl_pet_hargreaves.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "13_CAMELScl_swe.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "14_CAMELScl_catch_hierarchy.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "CAMELScl_catchment_boundaries.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
    }

    dynamic_features_ = ['streamflow_m3s', 'streamflow_mm',
                         'precip_cr2met', 'precip_chirps', 'precip_mswep', 'precip_tmpa',
                         'tmin_cr2met', 'tmax_cr2met', 'tmean_cr2met',
                         'pet_8d_modis', 'pet_hargreaves',
                         'swe'
                         ]

    def __init__(self,
                 path: str = None,
                 **kwargs,
                 ):
        """
        Arguments:
            path: path where the CAMELS-CL dataset has been downloaded. This path must
                  contain five zip files and one xlsx file.
        """

        super().__init__(path=path, **kwargs)
        self.path = path

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        for _file, url in self.urls.items():
            fpath = os.path.join(self.path, _file)
            if not os.path.exists(fpath) or (os.path.exists(fpath) and self.overwrite):
                if self.verbosity:
                    print(f"Downloading {_file} from {url + _file} at {fpath}")
                download(url + _file, self.path, verbosity=self.verbosity)
                unzip(self.path, verbosity=self.verbosity)
            
        self._static_features = self._static_data().columns.tolist()

        # self.dyn_fname = os.path.join(self.path, 'camels_cl_dyn.nc')
        self._maybe_to_netcdf()

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(
            self.path,
            "CAMELScl_catchment_boundaries",
            "CAMELScl_catchment_boundaries",
            "catchments_camels_cl_v1.3.shp"
        )

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'area': catchment_area(),
                'slope_mean': slope('mkm-1'),
                'gauge_lat': gauge_latitude(),
                'gauge_lon': gauge_longitude(),
        }

    @property
    def dyn_map(self):
        return {
            'streamflow_m3s': observed_streamflow_cms(),
            'streamflow_mm': observed_streamflow_mm(),
            'tmin_cr2met': min_air_temp(),
            'tmax_cr2met': max_air_temp(),
            'tmean_cr2met': mean_air_temp(),
            'precip_mswep': total_precipitation_with_specifier('mswep'),
            'precip_tmpa': total_precipitation_with_specifier('tmpa'),
            'precip_cr2met': total_precipitation_with_specifier('cr2met'),
            'precip_chirps': total_precipitation_with_specifier('chirps'),
            'pet_hargreaves': total_potential_evapotranspiration_with_specifier('hargreaves'),
            'pet_8d_modis': total_potential_evapotranspiration_with_specifier('modis'),
        }

    @property
    def _all_dirs(self):
        """All the folders in the dataset_directory"""
        return [f for f in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, f))]

    @property
    def start(self):
        return "19130215"

    @property
    def end(self):
        return "20180309"

    @property
    def location(self):
        return "Chile"

    @property
    def static_features(self) -> List[str]:
        return self._static_features

    def _static_data(self)->pd.DataFrame:
        path = os.path.join(self.path, f"1_CAMELScl_attributes{SEP}1_CAMELScl_attributes.txt")
        df = pd.read_csv(path, sep='\t', index_col='gauge_id')
        df = df.transpose()

        df.rename(columns=self.static_map, inplace=True)
                
        # remove empty space in index of df
        df.index = df.index.str.strip()

        return df

    @property
    def dynamic_features(self) -> List[str]:
        return [self.dyn_map.get(feat, feat) for feat in self.dynamic_features_]

    @property
    def _mm_feature_name(self) -> str:
        return observed_streamflow_mm()

    def stn_coords(
            self,
            stations: Union[str, List[str]] = "all"
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
        >>> dataset = CAMELS_CL()
        >>> dataset.stn_coords() # returns coordinates of all stations
        >>> dataset.stn_coords('12872001')  # returns coordinates of station whose id is 912101A
        >>> dataset.stn_coords(['12872001', '12876004'])  # returns coordinates of two stations
        """
        fpath = os.path.join(self.path,
                             '1_CAMELScl_attributes',
                             '1_CAMELScl_attributes.txt')
        df = pd.read_csv(fpath, sep='\t', index_col='gauge_id')
        df = df.loc[['gauge_lat', 'gauge_lon'], :].transpose()
        df.columns = ['lat', 'long']
        stations = check_attributes(stations, self.stations(), 'stations')
        df.index = [index.strip() for index in df.index]
        return df.loc[stations, :].astype(self.fp)

    def stations(self) -> list:
        """
        Tells all station ids for which a data of a specific attribute is available.
        """
        stn_fname = os.path.join(self.path, 'stations.json')
        if not os.path.exists(stn_fname):
            _stations = {}
            for dyn_attr in self.dynamic_features_:
                for _dir in self._all_dirs:
                    if dyn_attr in _dir:
                        fname = os.path.join(self.path, f"{_dir}{SEP}{_dir}.txt")
                        df = pd.read_csv(fname, sep='\t', nrows=2, index_col='gauge_id')
                        _stations[dyn_attr] = list(df.columns)

            stns = list(set.intersection(*map(set, list(_stations.values()))))
            with open(stn_fname, 'w') as fp:
                json.dump(stns, fp)
        else:
            with open(stn_fname, 'r') as fp:
                stns = json.load(fp)
        return stns

    def _read_dynamic(self, stations, dynamic_features, st=None, en=None):

        dyn = {}
        st, en = self._check_length(st, en)
        dynamic_features = check_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')

        assert all(stn in self.stations() for stn in stations)

        dynamic_features = check_attributes(dynamic_features, self.dynamic_features)

        # reading all dynnamic features
        dyn_attrs = {}
        for attr in self.dynamic_features_:
            fname = [f for f in self._all_dirs if '_' + attr in f][0]
            fname = os.path.join(self.path, f'{fname}{SEP}{fname}.txt')
            df = pd.read_csv(fname, sep='\t', index_col=['gauge_id'], na_values=" ")
            df.index = pd.to_datetime(df.index)

            dyn_attrs[attr] = df[st:en]

        # making one separate dataframe for one station
        for stn in stations:
            stn_df = pd.DataFrame()
            for attr, attr_df in dyn_attrs.items():
                # if attr in dynamic_features:
                stn_df[attr] = attr_df[stn]

            stn_df.rename(columns=self.dyn_map, inplace=True)
            stn_df.index.name = 'time'
            stn_df.columns.name = 'dynamic_features'
            dyn[stn] = stn_df.loc[st:en, dynamic_features]

        return dyn


class CAMELS_CH(_RainfallRunoff):
    """
    Data of 331 Swiss catchments from
    `Hoege et al., 2023 <https://doi.org/10.5194/essd-15-5755-2023>`_ .
    The dataset consists of 209 static catchment features and 9 dynamic features.
    The dynamic features span from 19810101 to 20201231 with daily timestep.
    For daily (``D``) ``timestep``, only streamflow is available for 170 swiss catchments.
    The hourly (``H``) streamflow data is obtained from `Kauzlaric et al., 2023 <https://zenodo.org/records/7691294>`_ .

    Examples
    ---------
    >>> from aqua_fetch import CAMELS_CH
    >>> dataset = CAMELS_CH()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='2004', as_dataframe=True)
    >>> df = dynamic['2004'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (14610, 9)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       331
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (33 out of 331)
       33
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(14610, 9), (14610, 9), (14610, 9),... (14610, 9), (14610, 9)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('2004', as_dataframe=True,
    ...  dynamic_features=['pcp_mm', 'airtemp_C_mean', 'q_cms_obs'])
    >>> dynamic['2004'].shape
       (14610, 3)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='2004', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['2004'].shape
    ((1, 209), 1, (14610, 9))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 14610, 'dynamic_features': 9})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (331, 2)
    >>> dataset.stn_coords('2004')  # returns coordinates of station whose id is 2004
        47.925221       8.191595
    >>> dataset.stn_coords(['2004', '2007'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('2004')
    # get coordinates of two stations
    >>> dataset.area(['2004', '2007'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('2004')

    """
    url = {
        'camels_ch.zip': "https://zenodo.org/record/7957061",
        'DischargeDBHydroCH.zip': 'https://zenodo.org/records/7691294'
    }

    def __init__(
            self,
            path=None,
            overwrite: bool = False,
            to_netcdf: bool = True,
            timestep: str = 'D',
            **kwargs
    ):
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
            This will fasten repeated calls to fetch etc. but will
            require netCDF4 package as well as xarry.
        """
        super().__init__(path=path, **kwargs)

        self.timestep = timestep

        if timestep == 'D' and 'DischargeDBHydroCH.zip' in self.url:
            self.url.pop('DischargeDBHydroCH.zip')

        self._download(overwrite=overwrite)

        self._dynamic_features = self._read_stn_dyn(self.stations()[0]).columns.tolist()

        # if to_netcdf:
        self._maybe_to_netcdf()

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(
            self.path,
            'camels_ch',
            'camels_ch',
            'catchment_delineations',
            'CAMELS_CH_catchments.shp'
        )

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'area': catchment_area(),
                'slope_mean': slope('degrees'),
                'gauge_lat': gauge_latitude(),
                'gauge_lon': gauge_longitude(),
        }

    @property
    def dyn_map(self):
        # table 1 in https://essd.copernicus.org/articles/15/5755/2023/
        return {
            'discharge_vol(m3/s)': observed_streamflow_cms(),
            # 'discharge_vol(m3/s)': 'sim_q_cms',
            'discharge_spec(mm/d)': observed_streamflow_mm(),
            'temperature_min(°C)': min_air_temp(),
            'temperature_max(°C)': max_air_temp(),
            'temperature_mean(°C)': mean_air_temp(),
            'precipitation(mm/d)': total_precipitation(),
            'swe(mm)': snow_water_equivalent(),
        }

    @property
    def camels_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.path, 'camels_ch', 'camels_ch')

    @property
    def static_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.camels_path, 'static_attributes')

    @property
    def dynamic_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.camels_path, 'time_series', 'observation_based')

    @property
    def glacier_attr_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_glacier_attributes.csv')

    @property
    def clim_attr_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_climate_attributes_obs.csv')

    @property
    def geol_attr_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_geology_attributes.csv')

    @property
    def supp_geol_attr_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_geology_attributes_supplement.csv')

    @property
    def hum_inf_attr_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_humaninfluence_attributes.csv')

    @property
    def hydrogeol_attr_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_hydrogeology_attributes.csv')

    @property
    def hydrol_attr_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_hydrology_attributes_obs.csv')

    @property
    def lc_attr_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_landcover_attributes.csv')

    @property
    def soil_attr_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_soil_attributes.csv')

    @property
    def topo_attr_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_topographic_attributes.csv')

    @property
    def static_features(self):
        return self._static_data().columns.tolist()

    @property
    def dynamic_features(self) -> List[str]:
        return self._dynamic_features

    def all_hourly_stations(self) -> List[str]:
        """Names of all stations which have hourly data"""
        return pd.read_excel(
            os.path.join(self.path, 'Inventory_discharge_hydroCH.xlsx'), dtype={'ID': str}
        )['ID'].values.tolist()

    def hourly_stations(self) -> List[str]:
        """
        IDs of those stations which have hourly data and which are also part of
        CAMELS-CH dataset
        """
        return [stn for stn in self.all_hourly_stations() if stn in self.stations()]

    @property
    def start(self):  # start of data
        return pd.Timestamp('1981-01-01')

    @property
    def end(self):  # end of data
        return pd.Timestamp('2020-12-31')

    def stations(self) -> List[str]:
        """Returns station ids for catchments"""
        stns = pd.read_csv(
            self.glacier_attr_path,
            sep=';',
            skiprows=1
        )['gauge_id'].values.tolist()
        return [str(stn) for stn in stns]

    @property
    def foen_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.path, 'DischargeDBHydroCH', 'DischargeDBHydroCH', 'CH', 'FOEN')

    def foen_stations(self) -> List[str]:
        """Returns all the stations in the FOEN folder"""
        return os.listdir(self.foen_path)

    def read_hourly_q_ch(self, stn: str) -> pd.DataFrame:
        stn = f"Q_{stn}_hourly.asc"
        fname = [fname for fname in self.foen_stations() if stn in fname][0]
        fpath = os.path.join(self.foen_path, fname)

        q = pd.read_csv(fpath,
                        sep="\t",
                        parse_dates=[['YYYY', 'MM', 'DD', 'HH']],
                        index_col='YYYY_MM_DD_HH',
                        )
        q.index = pd.to_datetime(q.index)
        q.columns = ['q_cms']
        q.index.name = "time"
        return q

    def glacier_attrs(self) -> pd.DataFrame:
        """
        returns a dataframe with four columns
            - 'glac_area'
            - 'glac_vol'
            - 'glac_mass'
            - 'glac_area_neighbours'
        """
        df = pd.read_csv(
            self.glacier_attr_path,
            sep=';',
            skiprows=1,
            index_col='gauge_id',
            dtype=np.float32
        )
        df.index = df.index.astype(int).astype(str)
        return df

    def climate_attrs(self) -> pd.DataFrame:
        """returns 14 climate attributes of catchments.
        """
        df = pd.read_csv(
            self.clim_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id',
            dtype={
                'gauge_id': str,
                'p_mean': float,
                'aridity': float,
                'pet_mean': float,
                'p_seasonality': float,
                'frac_snow': float,
                'high_prec_freq': float,
                'high_prec_dur': float,
                'high_prec_timing': str,
                'low_prec_timing': str
            }
        )
        return df

    def geol_attrs(self) -> pd.DataFrame:
        """15 geological features"""
        df = pd.read_csv(
            self.geol_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id',
            dtype=np.float32
        )
        df.index = df.index.astype(int).astype(str)
        return df

    def supp_geol_attrs(self) -> pd.DataFrame:
        """supplimentary geological features"""
        df = pd.read_csv(
            self.supp_geol_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id',
            dtype=np.float32
        )

        df.index = df.index.astype(int).astype(str)
        return df

    def human_inf_attrs(self) -> pd.DataFrame:
        """
        14 athropogenic factors
        """
        df = pd.read_csv(
            self.hum_inf_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id',
            dtype={
                'gauge_id': str,
                'n_inhabitants': int,
                'dens_inhabitants': float,
                'hp_count': int,
                'hp_qturb': float,
                'hp_inst_turb': float,
                'hp_max_power': float,
                'num_reservoir': int,
                'reservoir_cap': float,
                'reservoir_he': float,
                'reservoir_fs': float,
                'reservoir_irr': float,
                'reservoir_nousedata': float,
                # 'reservoir_year_first': int,
                # 'reservoir_year_last': int
            }
        )
        return df

    def hydrogeol_attrs(self) -> pd.DataFrame:
        """10 hydrogeological factors"""
        df = pd.read_csv(
            self.hydrogeol_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id',
            dtype=float
        )
        df.index = df.index.astype(int).astype(str)
        return df

    def hydrol_attrs(self) -> pd.DataFrame:
        """14 hydrological parameters + 2 useful infos"""
        df = pd.read_csv(
            self.hydrol_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id',
            dtype={
                'gauge_id': str,
                'sign_number_of_years': int,
                'q_mean': float,
                'runoff_ratio': float, 'stream_elas': float, 'slope_fdc': float,
                'baseflow_index_landson': float,
                'hfd_mean': float,
                'Q5': float, 'Q95': float, 'high_q_freq': float, 'high_q_dur': float,
                'low_q_freq': float
            }
        )
        return df

    def landcolover_attrs(self) -> pd.DataFrame:
        """13 landcover parameters"""
        return pd.read_csv(
            self.lc_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id',
            dtype={
                'gauge_id': str,
                'crop_perc': float,
                'grass_perc': float,
                'scrub_perc': float,
                'dwood_perc': float,
                'mixed_wood_perc': float,
                'ewood_perc': float,
                'wetlands_perc': float,
                'inwater_perc': float,
                'ice_perc': float,
                'loose_rock_perc': float,
                'rock_perc': float,
                'urban_perc': float,
                'dom_land_cover': str
            }
        )

    def soil_attrs(self) -> pd.DataFrame:
        """80 soil parameters"""
        df = pd.read_csv(
            self.soil_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id'
        )
        df.index = df.index.astype(int).astype(str)
        return df

    def topo_attrs(self) -> pd.DataFrame:
        """topographic parameters"""
        df = pd.read_csv(
            self.topo_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id',
            encoding="unicode_escape"
        )

        df.index = df.index.astype(int).astype(str)
        return df

    def _static_data(self)->pd.DataFrame:
        df = pd.concat(
            [
                self.climate_attrs(),
                self.geol_attrs(),
                self.supp_geol_attrs(),
                self.glacier_attrs(),
                self.human_inf_attrs(),
                self.hydrogeol_attrs(),
                self.hydrol_attrs(),
                self.landcolover_attrs(),
                self.soil_attrs(),
                self.topo_attrs(),
            ],
            axis=1)
        df.index = df.index.astype(str)

        df.rename(columns=self.static_map, inplace=True)

        return df

    def _read_stn_dyn(self, station: str) -> pd.DataFrame:
        """
        Reads daily dynamic (meteorological + streamflow) data for one catchment
        and returns as DataFrame
        """

        df = pd.read_csv(
            os.path.join(self.dynamic_path, f"CAMELS_CH_obs_based_{station}.csv"),
            sep=';',
            index_col='date',
            parse_dates=True,
            dtype=np.float32
        )

        df.rename(columns=self.dyn_map, inplace=True)

        return df


class CAMELS_DE(_RainfallRunoff):
    """
    This is the data from 1555 German catchments following the work of
    `Loritz et al., 2024 <https://doi.org/10.5194/essd-16-5625-2024>`_ .
    The data is downloaded from `zenodo <https://zenodo.org/record/12733968>`_ .
    This data consists of 111 static and 21 dynamic features. The dynamic features
    span from 1951-01-01 to 2020-12-31 with daily timestep.

    Examples
    --------
    >>> from aqua_fetch import CAMELS_DK
    >>> dataset = CAMELS_DK()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='DE110260', as_dataframe=True)
    >>> df = dynamic['DE110260'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (25568, 21)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       1555
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (155 out of 1555)
       155
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(25568, 21), (25568, 21), (25568, 21),... (25568, 21), (25568, 21)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('DE110260', as_dataframe=True,
    ...  dynamic_features=['airtemp_C_mean', 'rh_%', 'pcp_mm_mean', 'q_cms_obs'])
    >>> dynamic['DE110260'].shape
       (25568, 4)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='DE110260', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['DE110260'].shape
    ((1, 111), 1, (25568, 21))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 25568, 'dynamic_features': 21})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (1555, 2)
    >>> dataset.stn_coords('DE110260')  # returns coordinates of station whose id is DE110260
        47.925221       8.191595
    >>> dataset.stn_coords(['DE110260', 'DE110250'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('DE110260')
    # get coordinates of two stations
    >>> dataset.area(['DE110260', 'DE110250'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('DE110260')
    """
    url = "https://zenodo.org/record/12733968"

    def __init__(
            self,
            path=None,
            overwrite: bool = False,
            to_netcdf: bool = True,
            verbosity: int = 1,
            **kwargs
    ):
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
            This will fasten repeated calls to fetch etc. but will
            require netCDF4 package as well as xarray.
        """
        super().__init__(path=path, verbosity=verbosity, **kwargs)

        self._download(overwrite=overwrite)

        if to_netcdf and netCDF4 is None:
            warnings.warn("netCDF4 is not installed. Therefore, the data will not be converted to netcdf format.")
            to_netcdf = False

        # if to_netcdf:
        self._maybe_to_netcdf()

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(self.path,
                            "camels_de",
                            "CAMELS_DE_catchment_boundaries",
                            "catchments", 
                            "CAMELS_DE_catchments.shp")

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'area': catchment_area(),
                'slope_fdc': slope(''),
                'gauge_lat': gauge_latitude(),
                'gauge_lon': gauge_longitude(),
        }

    @property
    def dyn_map(self):
        # table 1 in https://essd.copernicus.org/articles/16/5625/2024/#&gid=1&pid=1
        return {
            'discharge_vol': observed_streamflow_cms(),
            'discharge_spec': observed_streamflow_mm(),
            'temperature_min': min_air_temp(),
            'temperature_max': max_air_temp(),
            'temperature_mean': mean_air_temp(),
            # 'precipitation_mean': 'pcp_mm',
            'precipitation_mean': total_precipitation_with_specifier('mean'), # todo: is it mean or total?
            'precipitation_median': total_precipitation_with_specifier('median'),
            'precipitation_stdev': total_precipitation_with_specifier('std'),
            'precipitation_min': total_precipitation_with_specifier('min'),
            'precipitation_max': total_precipitation_with_specifier('max'),
            'humidity_mean': mean_rel_hum(),
            'humidity_median': rel_hum_with_specifier('med'),
            'humidity_stdev': rel_hum_with_specifier('std'),
            'humidity_min': rel_hum_with_specifier('min'),
            'humidity_max': rel_hum_with_specifier('max'),
            # 'water_level':  # observed daily water level,
            'radiation_global_stdev': solar_radiation_with_specifier('std'),
            'radiation_global_min': solar_radiation_with_specifier('min'),
            'radiation_global_median': solar_radiation_with_specifier('med'),
            'radiation_global_mean': solar_radiation_with_specifier('mean'),
            'radiation_global_max': solar_radiation_with_specifier('max'),
        }

    @property
    def ts_dir(self) -> str:
        return os.path.join(self.path, 'camels_de', 'timeseries')

    @property
    def clim_attr_path(self) -> str:
        return os.path.join(self.path, 'camels_de', 'CAMELS_DE_climatic_attributes.csv')

    @property
    def hum_infl_path(self) -> str:
        return os.path.join(self.path, 'camels_de', 'CAMELS_DE_humaninfluence_attributes.csv')

    @property
    def hydrogeol_attr_path(self) -> str:
        return os.path.join(self.path, 'camels_de', 'CAMELS_DE_hydrogeology_attributes.csv')

    @property
    def hydrol_attr_path(self) -> str:
        return os.path.join(self.path, 'camels_de', 'CAMELS_DE_hydrologic_attributes.csv')

    @property
    def lc_attr_path(self) -> str:
        return os.path.join(self.path, 'camels_de', 'CAMELS_DE_landcover_attributes.csv')

    @property
    def sim_attr_path(self) -> str:
        return os.path.join(self.path, 'camels_de', 'CAMELS_DE_simulation_benchmark.csv')

    @property
    def soil_attr_path(self) -> str:
        return os.path.join(self.path, 'camels_de', 'CAMELS_DE_soil_attributes.csv')

    @property
    def topo_attr_path(self) -> str:
        return os.path.join(self.path, 'camels_de', 'CAMELS_DE_topographic_attributes.csv')

    def stations(self) -> List[str]:
        return [f.split('_')[4].split('.')[0] for f in os.listdir(self.ts_dir)]

    def clim_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self.clim_attr_path, index_col='gauge_id',
                           # dtype=np.float32
                           )

    def hum_infl_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self.hum_infl_path, index_col='gauge_id',
                           # dtype=np.float32
                           )

    def hydrogeol_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self.hydrogeol_attr_path, index_col='gauge_id',
                           # dtype=np.float32
                           )

    def hydrol_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self.hydrol_attr_path, index_col='gauge_id',  # dtype=np.float32
                           )

    def lc_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self.lc_attr_path, index_col='gauge_id',  # dtype=np.float32
                           )

    def sim_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self.sim_attr_path, index_col='gauge_id',  # dtype=np.float32
                           )

    def soil_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self.soil_attr_path, index_col='gauge_id',  # dtype=np.float32
                           )

    def topo_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self.topo_attr_path, index_col='gauge_id',  # dtype=np.float32
                           )

    def _static_data(self) -> pd.DataFrame:
        df = pd.concat([
            self.clim_attrs(),
            self.hum_infl_attrs(),
            self.hydrogeol_attrs(),
            self.hydrol_attrs(),
            self.lc_attrs(),
            self.sim_attrs(),
            self.soil_attrs(),
            self.topo_attrs()
        ], axis=1)

        df.rename(columns=self.static_map, inplace=True)

        return df

    def _read_stn_dyn(self, station) -> pd.DataFrame:
        """
        Reads daily dynamic (meteorological + streamflow) data for one catchment
        and returns as DataFrame
        """

        df = pd.read_csv(
            os.path.join(self.ts_dir, f"CAMELS_DE_hydromet_timeseries_{station}.csv"),
            # sep=';',
            index_col='date',
            parse_dates=True,
            # dtype=np.float32
        )

        df.rename(columns=self.dyn_map, inplace=True)
        return df

    @property
    def start(self):
        return pd.Timestamp('1951-01-01')

    @property
    def end(self):
        return pd.Timestamp('2020-12-31')

    @property
    def dynamic_features(self) -> List[str]:
        return self._read_stn_dyn(self.stations()[0]).columns.tolist()

    @property
    def static_features(self) -> List[str]:
        return self._static_data().columns.tolist()

    @property
    def _coords_name(self) -> List[str]:
        return ['gauge_lat', 'gauge_lon']

    @property
    def _area_name(self) -> str:
        return 'area'

    @property
    def _mm_feature_name(self) -> str:
        """Observed catchment-specific discharge (converted to millimetres per day
        using catchment areas"""
        return observed_streamflow_mm()


class CAMELS_SE(_RainfallRunoff):
    """
    Dataset of 50 Swedish catchments following the works of
    `Teutschbein et al., 2024 <https://doi.org/10.1002/gdj3.239>`_ . The data is downloaded
    from Swedish National Data Service `website <https://snd.se/en/catalogue/dataset/2023-173>`_ .
    The dataset consists of 76 static catchment features and 4 dynamic features.
    The dynamic features span from 19610101 to 20201231 with daily timestep.

    Examples
    --------
    >>> from aqua_fetch import CAMELS_DK
    >>> dataset = CAMELS_DK()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='5', as_dataframe=True)
    >>> df = dynamic['5'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (21915, 4)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       50
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (5 out of 50)
       5
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(21915, 4), (21915, 4), (21915, 4),... (21915, 4), (21915, 4)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('5', as_dataframe=True,
    ...  dynamic_features=['q_cms_obs', 'q_mm_obs', 'pcp_mm', 'airtemp_C_mean'])
    >>> dynamic['5'].shape
       (21915, 5)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='5', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['5'].shape
    ((1, 76), 1, (21915, 4))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 21915, 'dynamic_features': 4})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (50, 2)
    >>> dataset.stn_coords('5')  # returns coordinates of station whose id is 5
        68.0356 21.9758
    >>> dataset.stn_coords(['5', '200'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('5')
    # get coordinates of two stations
    >>> dataset.area(['5', '200'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('5')

    """

    url = {
 'catchment properties.zip': 'https://api.researchdata.se/dataset/2023-173/1/file/data?filePath=catchment+properties.zip',
 'catchment time series.zip': 'https://api.researchdata.se/dataset/2023-173/1/file/data?filePath=catchment+time+series.zip',
 'catchment_GIS_shapefiles.zip': 'https://api.researchdata.se/dataset/2023-173/1/file/data?filePath=catchment_GIS_shapefiles.zip',
 'Documentation_2024-01-02.pdf': 'https://api.researchdata.se/dataset/2023-173/1/file/documentation?filePath=Documentation_2024-01-02.pdf'
    }

    def __init__(
            self,
            path: str = None,
            to_netcdf: bool = True,
            overwrite: bool = False,
            verbosity: int = 1,
            **kwargs
    ):
        """
        Arguments:
            path: path where the CAMELS_SE dataset has been downloaded. This path
                must contain five zip files and one xlsx file. If None, then the
                data will be downloaded.
            to_netcdf :
        """
        super().__init__(path=path, verbosity=verbosity, **kwargs)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        for _file, url in self.url.items():
            fpath = os.path.join(self.path, _file)
            if not os.path.exists(fpath) and not overwrite:
                if verbosity > 0:
                    print(f"Downloading {_file} from {url}")
                download(url, outdir=self.path, fname=_file, verbosity=self.verbosity)
                unzip(self.path, verbosity=self.verbosity)
            else:
                if self.verbosity > 0: print(f"{_file} at {self.path} already exists")

        self._static_features = list(set(self._static_data().columns.tolist()))
        self._stations = self.physical_properties().index.to_list()
        self._dynamic_features = self._read_stn_dyn(self.stations()[0], nrows=2).columns.tolist()

        if to_netcdf and netCDF4 is None:
            warnings.warn("netCDF4 is not installed. Therefore, the data will not be converted to netcdf format.")
            to_netcdf = False

        # if to_netcdf:
        self._maybe_to_netcdf()

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(self.path,
                            'catchment_GIS_shapefiles',
                            'catchment_GIS_shapefiles',
                            'Sweden_catchments_50_boundaries_WGS84.shp')

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'Area_km2': catchment_area(),
                'slope_mean_degree': slope('degrees'),
                'Latitude_WGS84': gauge_latitude(),
                'Longitude_WGS84': gauge_longitude(),
        }

    @property
    def dyn_map(self):
        return {
            'Qobs_m3s': observed_streamflow_cms(),
            'Qobs_mm': observed_streamflow_mm(),
            'Tobs_C': mean_air_temp(),
            'Pobs_mm': total_precipitation(),
        }

    @property
    def static_features(self):
        return self._static_features

    @property
    def dynamic_features(self) -> List[str]:
        return self._dynamic_features

    @property
    def properties_path(self):
        return os.path.join(self.path, 'catchment properties', 'catchment properties')

    @property
    def ts_dir(self) -> os.PathLike:
        return os.path.join(self.path, 'catchment time series', 'catchment time series')

    @property
    def _mm_feature_name(self) -> str:
        return observed_streamflow_cms()

    @property
    def start(self):
        return pd.Timestamp("19610101")

    @property
    def end(self):
        return pd.Timestamp("20201231")

    def stations(self) -> List[str]:
        return self._stations

    def landcover(self) -> pd.DataFrame:
        return pd.read_csv(
            os.path.join(self.properties_path, 'catchments_landcover.csv'),
            index_col='ID', dtype={'ID': str})

    def physical_properties(self) -> pd.DataFrame:
        return pd.read_csv(
            os.path.join(self.properties_path, 'catchments_physical_properties.csv'),
            index_col='ID', dtype={'ID': str})

    def soil_classes(self) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(self.properties_path, 'catchments_soil_classes.csv'),
            index_col='ID', dtype={'ID': str})
        df.columns = [f"{c}_sc" for c in df.columns]
        return df

    def hydro_signatures_1961_2020(self) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(self.properties_path, 'catchments_hydrological_signatures_1961_2020.csv'),
            index_col='ID', dtype={'ID': str})
        df.columns = [f"{c}_hs" for c in df.columns]
        return df

    def hydro_signatures_CNP_1961_1990(self) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(self.properties_path, 'catchments_hydrological_signatures_CNP1_1961_1990.csv'),
            index_col='ID', dtype={'ID': str})
        df.columns = [f"{c}_CNP_61_90" for c in df.columns]
        return df

    def hydro_signatures_CNP_1990_2020(self) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(self.properties_path, 'catchments_hydrological_signatures_CNP2_1991_2020.csv'),
            index_col='ID', dtype={'ID': str})
        df.columns = [f"{c}_CNP_91_20" for c in df.columns]
        return df

    def _static_data(self) -> pd.DataFrame:
        df = pd.concat([
            self.landcover(),
            self.physical_properties(),
            self.soil_classes(),
            self.hydro_signatures_1961_2020(),
            self.hydro_signatures_CNP_1961_1990(),
            self.hydro_signatures_CNP_1990_2020()
        ], axis=1)

        df.rename(columns=self.static_map, inplace=True)

        return df

    def _read_stn_dyn(self, station, nrows=None) -> pd.DataFrame:
        """
        Reads daily dynamic (meteorological + streamflow) data for one catchment
        and returns as DataFrame
        """
        # find file starting with 'catchment_id_stn_id_' in self.path
        stn_id = f'catchment_id_{station}_'
        fname = [f for f in os.listdir(self.ts_dir) if f.startswith(stn_id)]
        assert len(fname) == 1
        fname = fname[0]

        df = pd.read_csv(
            os.path.join(self.ts_dir, fname),
            index_col='Year_Month_Day',
            parse_dates=[['Year', 'Month', 'Day']],
            dtype={'Qobs_m3s': np.float32, 'Qobs_mm': np.float32, 'Pobs_mm': np.float32, 'Tobs_C': np.float32},
            nrows=nrows,
        )

        for old_name, new_name in self.dyn_map.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)

        return df


class CAMELS_DK(_RainfallRunoff):
    """
    This is an updated version of :py class:`aqua_fetch.rr.Caravan_DK`
    dataset . This dataset was presented
    by `Liu et al., 2024 <https://doi.org/10.5194/essd-2024-292>`_ and is
    available at `dataverse <https://dataverse.geus.dk/dataset.xhtml?persistentId=doi:10.22008/FK2/AZXSYP>`_ .
    This dataset consists of 119 static and 13 dynamic features from 3330 Danish catchments.
    The dynamic (time series) features span from 1989-01-02 to 2023-12-31 with daily timestep.
    However, the streamflow observations are available for only 304 catchments.

    Examples
    ---------
    >>> from aqua_fetch import CAMELS_DK
    >>> dataset = CAMELS_DK()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='54130033', as_dataframe=True)
    >>> df = dynamic['54130033'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (12782, 13)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       304
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (30 out of 304)
       30
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(12782, 13), (12782, 13), (12782, 13),... (12782, 13), (12782, 13)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('54130033', as_dataframe=True,
    ...  dynamic_features=['Abstraction', 'pet_mm', 'airtemp_C_mean', 'pcp_mm', 'q_cms_obs'])
    >>> dynamic['54130033'].shape
       (12782, 5)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='54130033', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['54130033'].shape
    ((1, 119), 1, (12782, 13))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 12782, 'dynamic_features': 13})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (304, 2)
    >>> dataset.stn_coords('54130033')  # returns coordinates of station whose id is 54130033
        55.325242	9.93079
    >>> dataset.stn_coords(['54130033', '13210113'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('54130033')
    # get coordinates of two stations
    >>> dataset.area(['54130033', '13210113'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('54130033')
    """

    url = {
        'CAMELS_DK_304_gauging_catchment_boundaries.cpg': 'https://dataverse.geus.dk/api/access/datafile/83017',
        'CAMELS_DK_304_gauging_catchment_boundaries.prj': 'https://dataverse.geus.dk/api/access/datafile/83019',
        'CAMELS_DK_304_gauging_catchment_boundaries.shp': 'https://dataverse.geus.dk/api/access/datafile/83021',
        'CAMELS_DK_304_gauging_catchment_boundaries.dbf': 'https://dataverse.geus.dk/api/access/datafile/83020',
        'CAMELS_DK_304_gauging_catchment_boundaries.shx': 'https://dataverse.geus.dk/api/access/datafile/83018',
        'CAMELS_DK_304_gauging_stations.cpg': 'https://dataverse.geus.dk/api/access/datafile/83008',
        'CAMELS_DK_304_gauging_stations.dbf': 'https://dataverse.geus.dk/api/access/datafile/83010',
        'CAMELS_DK_304_gauging_stations.prj': 'https://dataverse.geus.dk/api/access/datafile/83009',
        'CAMELS_DK_304_gauging_stations.shp': 'https://dataverse.geus.dk/api/access/datafile/83011',
        'CAMELS_DK_304_gauging_stations.shx': 'https://dataverse.geus.dk/api/access/datafile/83007',
        'CAMELS_DK_climate.csv': 'https://dataverse.geus.dk/api/access/datafile/83123',
        'CAMELS_DK_geology.csv': 'https://dataverse.geus.dk/api/access/datafile/83124',
        'CAMELS_DK_georegion.dbf': 'https://dataverse.geus.dk/api/access/datafile/83030',
        'CAMELS_DK_georegion.prj': 'https://dataverse.geus.dk/api/access/datafile/83026',
        'CAMELS_DK_georegion.sbn': 'https://dataverse.geus.dk/api/access/datafile/83027',
        'CAMELS_DK_georegion.sbx': 'https://dataverse.geus.dk/api/access/datafile/83028',
        'CAMELS_DK_georegion.shp': 'https://dataverse.geus.dk/api/access/datafile/83029',
        'CAMELS_DK_georegion.shx': 'https://dataverse.geus.dk/api/access/datafile/83031',
        'CAMELS_DK_landuse.csv': 'https://dataverse.geus.dk/api/access/datafile/83125',
        'CAMELS_DK_script.py': 'https://dataverse.geus.dk/api/access/datafile/83135',
        'CAMELS_DK_signature_obs_based.csv': 'https://dataverse.geus.dk/api/access/datafile/83131',
        'CAMELS_DK_signature_sim_based.csv': 'https://dataverse.geus.dk/api/access/datafile/83132',
        'CAMELS_DK_soil.csv': 'https://dataverse.geus.dk/api/access/datafile/83126',
        'CAMELS_DK_topography.csv': 'https://dataverse.geus.dk/api/access/datafile/83127',
        'Data_description.pdf': 'https://dataverse.geus.dk/api/access/datafile/83138',
        'Gauged_catchments.zip': 'https://dataverse.geus.dk/api/access/datafile/83022',
        'Ungauged_catchments.zip': 'https://dataverse.geus.dk/api/access/datafile/83025',
    }

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
            require netCDF4 package as well as xarray.
        """
        super(CAMELS_DK, self).__init__(path=path, **kwargs)
        self._download(overwrite=overwrite)

        # self.dyn_fname = os.path.join(self.path, 'camelsdk_dyn.nc')
        self._static_features = self._static_data().columns.to_list()
        self._dynamic_features = self._read_csv(self.stations()[0]).columns.to_list()

        # if to_netcdf:
        self._maybe_to_netcdf()

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(
            self.path,
            "CAMELS_DK_304_gauging_catchment_boundaries.shp"
        )

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'catch_area': catchment_area(),
                'slope_mean': slope('mkm-1'),
                'catch_outlet_lat': gauge_latitude(),
                'catch_outlet_lon': gauge_longitude(),
        }

    @property
    def dyn_map(self):
        # table 1 in https://essd.copernicus.org/preprints/essd-2024-292/essd-2024-292.pdf
        return {
            'Qobs': observed_streamflow_cms(),
            'temperature': mean_air_temp(),
            'precipitation': total_precipitation(),
            'pet': total_potential_evapotranspiration(),  # todo: should we write method (makkink)
            'Qsim': simulated_streamflow_cms(),
            "DKM_eta": actual_evapotranspiration()
        }

    @property
    def gaug_catch_path(self):
        return os.path.join(self.path, "Gauged_catchments", "Gauged_catchments")

    @property
    def climate_fpath(self):
        return os.path.join(self.path, "CAMELS_DK_climate.csv")

    @property
    def geology_fpath(self):
        return os.path.join(self.path, "CAMELS_DK_geology.csv")

    @property
    def landuse_fpath(self):
        return os.path.join(self.path, "CAMELS_DK_landuse.csv")

    @property
    def soil_fpath(self):
        return os.path.join(self.path, "CAMELS_DK_soil.csv")

    @property
    def topography_fpath(self):
        return os.path.join(self.path, "CAMELS_DK_topography.csv")

    @property
    def signature_obs_fpath(self):
        return os.path.join(self.path, "CAMELS_DK_signature_obs_based.csv")

    @property
    def signature_sim_fpath(self):
        return os.path.join(self.path, "CAMELS_DK_signature_sim_based.csv")

    def climate_data(self):
        df = pd.read_csv(self.climate_fpath, index_col=0)
        df.index = df.index.astype(str)
        return df

    def geology_data(self):
        df = pd.read_csv(self.geology_fpath, index_col=0)
        df.index = df.index.astype(str)
        return df

    def landuse_data(self):
        df = pd.read_csv(self.landuse_fpath, index_col=0)
        df.index = df.index.astype(str)
        return df

    def soil_data(self):
        df = pd.read_csv(self.soil_fpath, index_col=0)
        df.index = df.index.astype(str)
        return df

    def topography_data(self):
        df = pd.read_csv(self.topography_fpath, index_col=0)
        df.index = df.index.astype(str)
        return df

    def signature_obs_data(self):
        df = pd.read_csv(self.signature_obs_fpath, index_col=0)
        df.index = df.index.astype(str)
        return df

    def signature_sim_data(self):
        df = pd.read_csv(self.signature_sim_fpath, index_col=0)
        df.index = df.index.astype(str)
        return df

    def _static_data(self) -> pd.DataFrame:
        """combination of topographic + soil + landuse + geology + climate features

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of static features of all catchments of shape (3330, 119)
        """
        df = pd.concat([self.climate_data(),
                          self.geology_data(),
                          self.landuse_data(),
                          self.soil_data(),
                          self.topography_data()
                          ], axis=1)
        
        df.rename(columns=self.static_map, inplace=True)

        return df

    def stations(self) -> List[str]:
        return [fname.split(".csv")[0].split('_')[4] for fname in os.listdir(self.gaug_catch_path)]

    def _read_csv(self, stn: str) -> pd.DataFrame:
        fpath = os.path.join(self.gaug_catch_path, f"CAMELS_DK_obs_based_{stn}.csv")
        df = pd.read_csv(os.path.join(fpath), parse_dates=True, index_col='time')
        df.columns.name = 'dynamic_features'
        df.pop('catch_id')
        df = df.astype(np.float32)

        df.rename(columns=self.dyn_map, inplace=True)
        # for old_name, new_name in self.dyn_map.items():
        #     if old_name in df.columns:
        #         df.rename(columns={old_name: new_name}, inplace=True)
        return df

    @property
    def dynamic_features(self) -> List[str]:
        """returns names of dynamic features"""
        return self._dynamic_features

    @property
    def static_features(self) -> List[str]:
        """returns static features for Denmark catchments"""
        return self._static_features

    @property
    def _coords_name(self) -> List[str]:
        return ['catch_outlet_lat', 'catch_outlet_lon']

    @property
    def _area_name(self) -> str:
        return 'catch_area'

    @property
    def _q_name(self) -> str:
        return observed_streamflow_cms()

    @property
    def start(self) -> pd.Timestamp:  # start of data
        return pd.Timestamp('1989-01-02 00:00:00')

    @property
    def end(self) -> pd.Timestamp:  # end of data
        return pd.Timestamp('2023-12-31 00:00:00')

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

    def transform_stn_coords(self, df:pd.DataFrame)->pd.DataFrame:

        ct_m = pd.DataFrame(columns=['lat', 'long'], index=df.index)
        # Test the function using lat, long in c DataFrame
        for i in range(0, len(df)):
            lat, lon = utm_to_lat_lon(df.iloc[i, 1], df.iloc[i, 0], 32)
            ct_m.iloc[i] = [lat, lon]
        
        return ct_m

    def transform_coords(self, coords):
        """
        Transforms the coordinates to the required format.
        """
        # from EPSG:25832 - ETRS89 / UTM zone 32N to WGS84
        return coords


class CAMELS_IND(_RainfallRunoff):
    """
    Dataset of 472 catchments from Republic of India following the works of
    `Mangukiya et al., 2024 <https://doi.org/10.5194/essd-2024-379>`_.
    The dataset consists of 210 static catchment features and 20 dynamic features.
    The dynamic features span from 19800101 to 20201231 with daily timestep.

    Examples
    ---------
    >>> from aqua_fetch import CAMELS_IND
    >>> dataset = CAMELS_IND()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='3001', as_dataframe=True)
    >>> df = dynamic['3001'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (14976, 20)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       472
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (47 out of 472)
       47
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(14976, 20), (14976, 20), (14976, 20),... (14976, 20), (14976, 20)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('3001', as_dataframe=True,
    ...  dynamic_features=['pcp_mm', 'rh_%', 'airtemp_C_mean', 'pet_mm', 'q_cms_obs'])
    >>> dynamic['3001'].shape
       (14976, 5)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10

    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='3001', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['3001'].shape
    ((1, 210), 1, (14976, 20))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 14976, 'dynamic_features': 20})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (472, 2)
    >>> dataset.stn_coords('3001')  # returns coordinates of station whose id is 3001
        48.006298   -4.063848
    >>> dataset.stn_coords(['3001', '17021'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('3001')
    # get coordinates of two stations
    >>> dataset.area(['3001', '17021'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('3001')
    """
    url = "https://zenodo.org/records/13221214"

    def __init__(self,
                 path=None,
                 overwrite=False,
                 to_netcdf: bool = True,
                 **kwargs):
        super(CAMELS_IND, self).__init__(path=path, **kwargs)
        self._download(overwrite=overwrite)

        names = pd.read_csv(
            os.path.join(self.static_path, "camels_India_name.txt"),
            sep=";",
            index_col=0,
            dtype={0: str}
        )
        id_str = names.index.to_list()
        id_int = names.index.astype(int).to_list()
        self.id_map = {str(k): v for k, v in zip(id_int, id_str)}

        self._static_features = self._static_data().columns.to_list()
        self._dynamic_features = self._read_stn_dyn(self.stations()[0]).columns.to_list()

        # if to_netcdf:
        self._maybe_to_netcdf()

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(
            self.path,
            "shapefiles_catchment",
            "Merged",
            "all_catchments.shp"
        )

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'cwc_area': catchment_area(),
                'slope_mean': slope('degrees'),
                'cwc_lat': gauge_latitude(),
                'cwc_lon': gauge_longitude(),
        }

    @property
    def dyn_map(self):
        # Table A1
        return {
            # 'streamflow_cms': 'obs_q_cms',
            'tmin(C)': min_air_temp(),
            'tmax(C)': max_air_temp(),
            'tavg(C)': mean_air_temp(),
            'prcp(mm/day)': total_precipitation(),
            'rel_hum(%)': mean_rel_hum(),
            'wind(m/s)': mean_windspeed(),
            'wind_u(m/s)': u_component_of_wind(), 
            'wind_v(m/s)': v_component_of_wind(),
            # surface downward short-wave radiation flux
            'srad_sw(w/m2)': solar_radiation(),
            # surface downward long-wave radiation flux
            'srad_lw(w/m2)': downward_longwave_radiation(),
            #'sm_lvl2(kg/m2)',   # soil moisture of layer 1 (0-0.1 m below ground)
            #'sm_lvl2(kg/m2)', 
            #'sm_lvl3(kg/m2)', 
            #'sm_lvl4(kg/m2)': ,
            'pet_gleam(mm/day)': total_potential_evapotranspiration_with_specifier('gleam'),
            'pet(mm/day)': total_potential_evapotranspiration(),
            'aet_gleam(mm/day)': actual_evapotranspiration_with_specifier('gleam'),
            #'evap_canopy(kg/m2/s)': evaporation
        }

    @property
    def static_path(self) -> os.PathLike:
        return os.path.join(self.path, "attributes_txt")

    @property
    def q_path(self) -> os.PathLike:
        return os.path.join(self.path, "streamflow_timeseries")

    @property
    def forcings_path(self) -> os.PathLike:
        return os.path.join(self.path, "catchment_mean_forcings")

    @property
    def dynamic_features(self) -> List[str]:
        """returns names of dynamic features"""
        return self._dynamic_features

    @property
    def static_features(self) -> List[str]:
        """returns static features for Denmark catchments"""
        return self._static_features

    @property
    def start(self) -> pd.Timestamp:  # start of data
        return pd.Timestamp('1980-01-01')

    @property
    def end(self) -> pd.Timestamp:  # end of data
        return pd.Timestamp('2020-12-31')

    def stn_forcing_path(self, stn: str) -> os.PathLike:
        return os.path.join(
            self.forcings_path,
            self.id_map.get(stn)[0:2],
            f"{self.id_map.get(stn)}.csv"
        )

    def stations(self) -> List[str]:
        """
        returns names of stations a list

        **Node:** 0s are omitted from the start of the station names
        which means 03001 is returned as 3001
        """
        stns = pd.read_csv(
            os.path.join(self.static_path, "camels_India_name.txt"),
            sep=";",
            index_col=0,
            dtype={0: str}
        ).index.to_list()

        return [str(int(stn)) for stn in stns]

    def _static_data(self) -> pd.DataFrame:
        """
        combination of topographic + soil + landuse + geology + climate + hydro
        + climate + anthropogenic features

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of static features of all catchments of shape (3330, 119)
        """
        files = glob.glob(f"{self.static_path}/*.txt")

        dfs = []
        for f in files:
            df = pd.read_csv(f, sep=";", index_col=0)
            df.index = df.index.astype(str)
            dfs.append(df)

        df = pd.concat(dfs, axis=1)

        df.rename(columns=self.static_map, inplace=True)

        return df

    def _read_q(self, stn: str = None, ) -> pd.DataFrame:
        """reads observed streamflow data"""
        fpath = os.path.join(self.q_path, f"streamflow_observed.csv")

        cols = ['year', 'month', 'day']
        if stn is not None:
            cols.append(stn)
        else:
            cols = None

        df = pd.read_csv(os.path.join(fpath),
                         index_col='year_month_day',
                         parse_dates=[['year', 'month', 'day']],
                         usecols=cols,
                         )

        return df.astype(np.float32)

    def _read_forcings(self, stn: str) -> pd.DataFrame:
        """reads the foring data for a given station"""
        fpath = self.stn_forcing_path(stn)
        df = pd.read_csv(fpath,
                         index_col='year_month_day',
                         parse_dates=[['year', 'month', 'day']],
                         )
        return df.astype(np.float32)

    def _read_stn_dyn(self, stn: str) -> pd.DataFrame:
        """reads dynamic data for a given station"""
        q = self._read_q(stn)[stn]
        q.name = observed_streamflow_cms()
        df = pd.concat([self._read_forcings(stn), pd.DataFrame(q)], axis=1)

        for old_name, new_name in self.dyn_map.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)

        return df


class CAMELS_FR(_RainfallRunoff):
    """
    Dataset of 654 catchments from France following the works of
    `Delaigue et al., 2024 <https://doi.org/10.5194/essd-2024-415>`_.
    The dataset consists of 344 static catchment features and 22 dynamic features.
    The dynamic features span from 1970101 to 20211231 with daily timestep.

    Examples
    ---------
    >>> from aqua_fetch import CAMELS_FR
    >>> dataset = CAMELS_FR()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='J421191001', as_dataframe=True)
    >>> df = dynamic['J421191001'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (12782, 22)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       654
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (65 out of 654)
       65
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(12782, 22), (12782, 22), (12782, 22),... (12782, 22), (12782, 22)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('J421191001', as_dataframe=True,
    ...  dynamic_features=['pcp_mm', 'spechum_gkg', 'airtemp_C_mean', 'pet_mm_pm', 'q_cms_obs'])
    >>> dynamic['J421191001'].shape
       (12782, 5)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='J421191001', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['J421191001'].shape
    ((1, 344), 1, (12782, 22))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 12782, 'dynamic_features': 22})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (654, 2)
    >>> dataset.stn_coords('J421191001')  # returns coordinates of station whose id is J421191001
        48.006298   -4.063848
    >>> dataset.stn_coords(['J421191001', '802'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('J421191001')
    # get coordinates of two stations
    >>> dataset.area(['J421191001', '802'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('J421191001')
    """
    url = {
        "ADDITIONAL_LICENSES.zip": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/343463",
        "CAMELS_FR_attributes.zip": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/343464",
        'CAMELS_FR_geography.zip': 'https://entrepot.recherche.data.gouv.fr/api/access/datafile/343465',
        'CAMELS_FR_time_series.zip': 'https://entrepot.recherche.data.gouv.fr/api/access/datafile/343470',
        'README.md': 'https://entrepot.recherche.data.gouv.fr/api/access/datafile/431300',
        'CAMELS-FR_description.ods': 'https://entrepot.recherche.data.gouv.fr/api/access/datafile/348740',
    }

    def __init__(self,
                 path=None,
                 overwrite=False,
                 **kwargs):
        super().__init__(path=path, **kwargs)

        self._download(overwrite=overwrite)

        self._stations = self.__stations()

        self._static_features = list(set(self._static_data().columns.to_list()))

        self._dynamic_features = self._read_stn_dyn(self.stations()[0]).columns.to_list()

        # if self.to_netcdf:
        self._maybe_to_netcdf()

    @property
    def boundary_file(self) -> os.PathLike:        
        return os.path.join(
            self.path, 
            'CAMELS_FR_geography', 
            'CAMELS_FR_geography', 
            'CAMELS_FR_catchment_boundaries.gpkg')

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'hyd_slope_fdc': slope(''),
                # 'sit_latitude', 'sit_longitude', todo : what is difference between site and guage lat/lon?
                # gauge latitude/longitude in WGS84 (Hydroportail coordinates)
                'sta_x_w84': gauge_longitude(),
                'sta_y_w84': gauge_latitude(),
                # todo: should we use sta_x_w84_snap and sta_y_w84_snap which are 
                # gauge longitude in WGS84 (INRAE's own estimation, snapped on thorical river network)
                'sit_area_topo': catchment_area(),
        }

    @property
    def dyn_map(self)->Dict[str, str]:
        return {
            # streamflow in liters per second
            'tsd_q_l': observed_streamflow_cms(),
            # streamflow in milimeters per day
            'tsd_q_mm': observed_streamflow_mm(),
            'tsd_wind': mean_windspeed(),
            'tsd_temp_min': min_air_temp(),  # minimum air temperature over the period (18h day-1, 18h day]
            'tsd_temp_max': max_air_temp(),  # maximum air temperature over the period (18h day-1, 18h day]
            'tsd_temp': mean_air_temp(),  # mean air temperature over the period (18h day-1, 18h day]
            # short wave visible radiation over the period (0h day, 0h day+1]
            'tsd_rad_ssi': solar_radiation(),  # todo: convert from J cm⁻² to W m⁻²
            # long wave atmospheric radiation over the period (0h day, 0h day+1]
            'tsd_rad_dli': downward_longwave_radiation(), # todo : convert from J cm⁻² to W m⁻²
            # specific air humidity over the period (0h day, 0h day+1]
            'tsd_humid': mean_specific_humidity(),
            # PET over the period (0h day, 0h day+1] (Penman-Monteith method with a modified albedo when snow lies on the ground)
            'tsd_pet_pm': total_potential_evapotranspiration_with_specifier('pm'),
            'tsd_pet_pe': total_potential_evapotranspiration_with_specifier('pe'),
            'tsd_pet_ou': total_potential_evapotranspiration_with_specifier('ou'),
            # total precipitation (liquid + solid) over the period (6h day, 6h day+1]
            'tsd_prec': total_precipitation(),
            # solid fraction of precipitation over the period (6h day, 6h day+1]
            'tsd_prec_solid_frac': total_precipitation_with_specifier('solfrac'),  # todo : check its units?
        }

    @property
    def daily_ts_path(self) -> os.PathLike:
        return os.path.join(self.path, "CAMELS_FR_time_series", "CAMELS_FR_time_series", "daily")

    @property
    def attr_path(self) -> os.PathLike:
        return os.path.join(self.path, "CAMELS_FR_attributes", "CAMELS_FR_attributes")

    @property
    def static_attr_path(self) -> os.PathLike:
        return os.path.join(self.attr_path, "static_attributes")

    @property
    def ts_stat_path(self) -> os.PathLike:
        return os.path.join(self.attr_path, "time_series_statistics")

    @property
    def static_features(self) -> List[str]:
        """returns static features for Denmark catchments"""
        return self._static_features

    @property
    def dynamic_features(self) -> List[str]:
        """returns names of dynamic features"""
        return self._dynamic_features

    @property
    def start(self) -> pd.Timestamp:  # start of data
        return pd.Timestamp('1970-01-01')

    @property
    def end(self) -> pd.Timestamp:  # end of data
        return pd.Timestamp('2021-12-31')

    def __stations(self) -> List[str]:
        return pd.read_csv(os.path.join(
            self.static_attr_path,
            "CAMELS_FR_human_influences_dams.csv"),
            sep=";",
            index_col=0).index.to_list()

    def stations(self) -> List[str]:
        return self._stations

    @property
    def geog_path(self) -> os.PathLike:
        return os.path.join(self.path, "CAMELS_FR_geography", "CAMELS_FR_geography")

    def static_attrs(self) -> pd.DataFrame:
        """
        combination of topographic + soil + landuse + geology + climate + hydro
        + climate + anthropogenic features

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of static features of all catchments of shape (654, xxxx)
        """
        files = glob.glob(f"{self.static_attr_path}/*.csv")
        dfs = []
        for f in files:
            df = pd.read_csv(f, sep=";", index_col=0)
            df.index = df.index.astype(str)
            if len(df) == 654:
                dfs.append(df)
            elif self.verbosity > 1:
                print(f"skipping {os.path.basename(f)} as it has {len(df)} rows")

        static_attrs = pd.concat(dfs, axis=1)

        gen_attrs = pd.read_csv(
            os.path.join(self.static_attr_path, "CAMELS_FR_site_general_attributes.csv"),
            sep=";",
            index_col=0,
        )

        # in gen_attrs the stn_id has lenght of 8 while in static_attrs it is 10
        # so adding the last two digits to the gen_attrs
        _map = {stn[0:-2]: stn for stn in static_attrs.index}
        gen_attrs = gen_attrs.rename(index=_map)

        if self.verbosity:
            for stn in static_attrs.index:
                if stn not in gen_attrs.index:
                    print(stn, " not found in site_general_attributes.csv")

        static_attrs = pd.concat([gen_attrs, static_attrs], axis=1)
        return static_attrs

    def ts_attrs(self) -> pd.DataFrame:
        """
        daily_timeseries statistics of all catchments

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of static features of all catchments of shape (654, xxxx)
        """
        files = glob.glob(f"{self.ts_stat_path}/*.csv")
        dfs = []
        for f in files:
            df = pd.read_csv(f, sep=";", index_col=0)
            df.index = df.index.astype(str)
            if len(df) == 654:
                dfs.append(df)
            elif self.verbosity > 1:
                print(f"skipping {os.path.basename(f)} as it has {len(df)} rows")

        return pd.concat(dfs, axis=1)

    def _static_data(self) -> pd.DataFrame:
        """
        static attributes plus timeseries statistics

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of static features of all catchments of shape (654, xxxx)
        """

        static_data = pd.concat([
            self.static_attrs(),
            self.ts_attrs()
        ], axis=1)
        # remove duplicated columns
        df = static_data.loc[:, ~static_data.columns.duplicated()].copy()

        df.rename(columns=self.static_map, inplace=True)

        return df

    def _read_stn_dyn(self, station: str) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(self.daily_ts_path, f"CAMELS_FR_tsd_{station}.csv"),
            sep=";",
            index_col=0,
            parse_dates=True,
            comment="#",
        )

        df.rename(columns=self.dyn_map, inplace=True)

        return df


class CAMELS_SPAT(_RainfallRunoff):
    """
    Dataset of 1426 catchments from North America (USA and Canada) following the works of
    `Knoben et al., 2025 <https://doi.org/10.5194/egusphere-2025-893>`_.
    """
    stn_name = 'USA_14141500'
    time_step = 'obs-hourly'  # or obs_daily
    scale = 'macro-scale' # or headwater or meso-scale
    data_type = 'observations'  # or 
    url = {
        f'https://www.frdr-dfdr.ca/repo/files/1/published/publication_1211/submitted_data/{data_type}/{scale}/{time_step}/{stn_name}_hourly_flow_observations.nc',

        }


class CAMELS_NZ(_RainfallRunoff):
    """
    Dataset of 369 catchments from New Zealand following the works of
    `Harrigan et al., 2025 <https://doi.org/10.5194/essd-2025-244>`_.
    The dataset consists of 39 static catchment features and 5 dynamic features.
    The dynamic features span from 19720101 to 20240802 with hourly timestep.
    The data is downloaded from `figshare <https://doi.org/10.26021/canterburynz.28827644>`_.
    
    Examples
    ---------
    >>> from aqua_fetch import CAMELS_NZ
    >>> dataset = CAMELS_NZ()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='74321', as_dataframe=True)
    >>> df = dynamic['74321'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (460928, 5)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       347
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (34 out of 347)
       34
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(460928, 5), (460928, 5), (460928, 5),... (460928, 5), (460928, 5)]

    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('74321', as_dataframe=True,
    ...  dynamic_features=['pcp_mm', 'rh_%', 'airtemp_C_mean', 'pet_mm', 'q_cms_obs'])
    >>> dynamic['74321'].shape
       (460928, 4)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='74321', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['74321'].shape
    ((1, 39), 1, (460928, 5))

    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 460928, 'dynamic_features': 5})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (347, 2)
    >>> dataset.stn_coords('74321')  # returns coordinates of station whose id is 74321
        -45.945599      170.101486
    >>> dataset.stn_coords(['74321', '802'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('74321')
    # get coordinates of two stations
    >>> dataset.area(['74321', '802'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('74321')
    """
    url = "https://figshare.canterbury.ac.nz/ndownloader/articles/28827644/versions/1"

    def __init__(self,
                 path:Union[str, os.PathLike]=None,
                 timestep = 'H',
                 **kwargs):

        super().__init__(name="CAMELS_NZ", path=path, timestep=timestep, **kwargs)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if not os.path.exists(os.path.join(self.path, 'camels_nz.zip')) and not self.overwrite:
            download(
            outdir=self.path,
            url=self.url,
            fname="camels_nz.zip",
            verbosity=self.verbosity,
        )

        unzip(self.path, verbosity=self.verbosity)

        # unzip the .zip files which are inside the camels_nz folder
        unzip(os.path.join(self.path, 'camels_nz'), verbosity=self.verbosity)

        # if self.to_netcdf:
        self._maybe_to_netcdf()

    @property
    def boundary_file(self)-> os.PathLike:
        return os.path.join(
            self.shapefile_path,
            "catnz_SpatialJoin.shp"
        )

    @property
    def dyn_map(self)->Dict[str, str]:
        return {
            'flow': observed_streamflow_cms(),
            'temperature': mean_air_temp(),
            'Relative_humidity': mean_rel_hum(),
            'precipitation': total_precipitation(),
            'PET': total_potential_evapotranspiration(),
        }

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'latitude': gauge_latitude(),
                'longitude': gauge_longitude(),
                'uparea': catchment_area(),
                'elevation': gauge_elevation_meters(),
                'usAveSlope': slope('degrees')
        }
   
    @property
    def start(self) -> pd.Timestamp:
        return  pd.Timestamp('1972-01-01 00:00:00')
    
    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp('2024-08-02 09:00:00')

    @property
    def dynamic_features(self) -> List[str]:
        """returns names of dynamic features"""
        return [self.dyn_map[feature] for feature in self._path_map]
    
    @property
    def static_features(self) -> List[str]:
        """returns static features for New Zealand catchments"""
        return self._static_data(nrows=2).columns.to_list()

    def stations(self)->List[str]:
        fpath = os.path.join(self.static_path, '4.CAMELS_NZ_Geology.csv')
        df = pd.read_csv(fpath, index_col=0, usecols=[0, 1])
        return df.index.astype(str).tolist()

    @property
    def temp_path(self) -> os.PathLike:
        return os.path.join(self.path, 'camels_nz', 'CAMELS_NZ_Temperature')
    
    @property
    def precip_path(self) -> os.PathLike:
        return os.path.join(self.path, 'camels_nz', 'CAMELS_NZ_Precipitation')
    
    @property
    def q_path(self) -> os.PathLike:
        return os.path.join(self.path, 'camels_nz', 'CAMELS_NZ_Streamflow')
    
    @property
    def shapefile_path(self) -> os.PathLike:
        return os.path.join(self.path, 'camels_nz', 'CAMELS_NZ_Catchment_Boundaries')
    
    @property
    def pet_path(self) -> os.PathLike:
        return os.path.join(self.path, 'camels_nz', 'CAMELS_NZ_PET')

    @property
    def rh_path(self) -> os.PathLike:
        return os.path.join(self.path, 'camels_nz', 'CAMELS_NZ_Relative_Humidity')

    @property
    def static_path(self) -> os.PathLike:
        return os.path.join(self.path, 'camels_nz', 'CAMELS_NZ_Catchment_Atrributes')

    def _static_data(self, nrows:int = None) -> pd.DataFrame:
        """
        static attributes of catchments

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of static features of all catchments of shape (369, 39)
        """

        dfs = []
        idx = 0

        # read all .csv files in static_path
        for csv_file in glob.glob(os.path.join(self.static_path, '*.csv')):
            df = pd.read_csv(csv_file, index_col=0, nrows=nrows)

            df.index = df.index.astype(str)

            if idx > 0:
                df.drop(columns=['RID', 'StationName', 'latitude', 'longitude'], inplace=True, errors='ignore')

            dfs.append(df)

            idx += 1
        
        static_data = pd.concat(dfs, axis=1)

        static_data.rename(columns=self.static_map, inplace=True)

        return static_data
    
    @property
    def _nodata_stns(self):
        """data from following stations is not available. The corresponding files are empty."""
        return ['75253', "75261", "75265", "75276", "75294", "15408", 
                "15410", "15453", "33356", "52916", "74318", "74321", "1114629"]

    def _read_dynamic_para(
            self, 
            stations:Union[str, List[str]] = "all",
            para_name:str = "PET",
            )-> pd.DataFrame:
        """
        reads dynamic data for a given parameter for given stations.
        """
        assert para_name in list(self._path_map.keys())
        cpus = self.processes or min(get_cpus(), 32)

        stations = check_attributes(stations, self.stations(), 'stations')

        start = time.time()

        if cpus == 1:
            q_dfs = []

            for _, stn in enumerate(stations):

                stn_q = self._read_stn_dyn_para(stn, para_name)
                q_dfs.append(stn_q)    

                if self.verbosity and _ % 100 == 0:
                    print(f"Read {len(q_dfs)} stations so far...")
        else:
            with cf.ProcessPoolExecutor(cpus) as executor:
                results = executor.map(
                    self._read_stn_dyn_para, 
                    stations, 
                    (para_name for _ in range(len(stations)))
                    )
            
            q_dfs = [stn_q for stn_q in results]

        total = time.time() -  start
        if self.verbosity:
            print(f"Read {len(q_dfs)} stations for {para_name} in {total:.2f} seconds with {cpus} cpus.")

        q_df = pd.concat(q_dfs, axis=1)
        return q_df
    
    @property
    def _path_map(self) -> Dict[str, os.PathLike]:
         return {
            'PET': self.pet_path,
            'precipitation': self.precip_path,
            'Relative_humidity': self.rh_path,
            'temperature': self.temp_path,
            'flow': self.q_path,
        }

    def _read_stn_dyn_para(self, stn:str, para_name:str) -> pd.Series:
        """
        read dynamic data for a given station and parameter.
        """
        stn_q = pd.Series(dtype=np.float32, name=stn)

        fname = {
            'Relative_humidity': 'RH'
        }
        fpath = os.path.join(
            self._path_map[para_name], f'{fname.get(para_name, para_name)}_station_id_{stn}.csv')

        if os.path.exists(fpath):
            if para_name == 'flow' and stn in self._nodata_stns:
                return stn_q
                        
            try:
                stn_q = pd.read_csv(fpath, index_col=0, parse_dates=True, na_values=['NA  '])
            except pd.errors.EmptyDataError:
                print(f"Warning: {para_name}_station_id_{stn}.csv is empty. Skipping station {stn}.")
                return stn_q

            format = '%m/%d/%Y %H:%M'
            if para_name == 'flow' and stn == '57521':
                format = '%d/%m/%Y %H:%M'            
            stn_q.index = pd.to_datetime(stn_q.index, format=format)

            stn_q = stn_q[para_name].astype(np.float32).rename(stn)
        else:
            if self.verbosity>1: 
                print(f"Warning: {para_name}_station_id_{stn}.csv does not exist. Skipping station {stn}.")
            stn_q = pd.Series(dtype=np.float32, name=stn)
        
        # remove rows with duplicated index, ideally there should not be any
        stn_q = stn_q[~stn_q.index.duplicated(keep='first')]

        return stn_q

    def _read_stn_dyn(self, stn:str)->pd.DataFrame:
        """
        reads dynamic data for a given station
        """
        stn_dfs = []
        for para in self._path_map.keys():

            stn_para = self._read_stn_dyn_para(stn, para)
            stn_dfs.append(stn_para.rename(para, inplace=True))
        stn_df = pd.concat(stn_dfs, axis=1)
        stn_df.index = pd.to_datetime(stn_df.index)

        # convert the temperature to Celcius from Kelvin
        stn_df['temperature'] = stn_df['temperature'] - 273.15

        stn_df.rename(columns=self.dyn_map, inplace=True)
        
        return stn_df
    

class CAMELS_COL(_RainfallRunoff):
    """
    Dataset of 347 catchments from Colombia following the works of
    `Jimenez et al., 2025 <https://doi.org/10.5194/essd-2025-200>`_.
    The dataset consists of 255 static catchment features and 6 dynamic features.
    The dynamic features span from 19810101 to 20221231 with daily timestep.
    The data is downloaded from `Zenodo <https://zenodo.org/records/15554735>`_.

    Examples
    ---------
    >>> from aqua_fetch import CAMELS_COL
    >>> dataset = CAMELS_COL()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='35067040', as_dataframe=True)
    >>> df = dynamic['35067040'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (15340, 6)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       347
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (34 out of 347)
       34
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(15340, 6), (15340, 6), (15340, 6),... (15340, 6), (15340, 6)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('35067040', as_dataframe=True,
    ...  dynamic_features=['pcp_mm', 'airtemp_C_mean', 'pet_mm', 'q_cms_obs'])
    >>> dynamic['35067040'].shape
       (15340, 4)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='35067040', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['35067040'].shape
    ((1, 255), 1, (15340, 6))

    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 15340, 'dynamic_features': 6})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (347, 2)
    >>> dataset.stn_coords('35067040')  # returns coordinates of station whose id is 35067040
        4.746433        -73.587807
    >>> dataset.stn_coords(['35067040', '21187030'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('35067040')
    # get coordinates of two stations
    >>> dataset.area(['35067040', '21187030'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('35067040')

    """
    url = "https://zenodo.org/records/15554735"

    def __init__(self,
                 path=None,
                 overwrite=False,
                 to_netcdf: bool = True,
                 **kwargs):

        super(CAMELS_COL, self).__init__(
            path=path, 
            to_netcdf=to_netcdf, 
            **kwargs)

        self._download(overwrite=overwrite)

        # if self.to_netcdf:
        self._maybe_to_netcdf()
        
    @property
    def bbox(self) -> Dict[str, float]:
        return dict(llcrnrlon=-80, llcrnrlat=-5, urcrnrlon=-65, urcrnrlat=12)

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(
            self.path,
            "03_CAMELS_COL_Basin_boundary",
            "03_CAMELS_COL_Basin_boundary",
            "CAMELS_COL_catchments_boundaries.shp"
        )

    @property
    def dyn_map(self) -> Dict[str, str]:
        """
        dynamic features map for CAMELS-LUX catchments
        """
        return {
            'streamflow': observed_streamflow_cms(),
            'pr': total_precipitation(),  # CHIRPS V2
            't_mean': mean_air_temp(),
            't_min': min_air_temp(),
            't_max': max_air_temp(),
            'poten_evapo': total_potential_evapotranspiration(),
        }

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'gauge_lat': gauge_latitude(),
                'gauge_lon': gauge_longitude(),
                'area': catchment_area(),
                'gauge_elev': gauge_elevation_meters(),
                'perimeter': catchment_perimeter(),
        }

    @property
    def ts_path(self) -> os.PathLike:
        return os.path.join(
            self.path,
            "04_CAMELS_COL_Hydrometeorological_data",
            "04_CAMELS_COL_Hydrometeorological_data",
        )

    def stations(self) -> List[str]:
        return [fname[14:22] for fname in os.listdir(self.ts_path)]

    @property
    def dynamic_features(self) -> List[str]:
        df = self._read_stn_dyn(self.stations()[0], nrows=2)
        return df.columns.to_list()

    @property
    def static_features(self) -> List[str]:  # todo : calling this method again and again can be slow
        """
        returns static features for Colombia catchments
        """
        df = self._static_data()
        return df.columns.to_list()

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp('1981-01-01')
    
    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp('2022-12-31')

    def _read_stn_dyn(self, stn:str, nrows=None)->pd.DataFrame:
        """
        reads dynamic data for a given station
        """
        stn_df = pd.read_csv(
            os.path.join(self.ts_path, f"Hydromet_data_{stn}.txt.txt"), 
            sep='\t',
            index_col=0, 
            parse_dates=True,
            nrows=nrows,
            )
        
        stn_df.index = pd.to_datetime(stn_df.index)

        if stn_df.index.has_duplicates:
            print(f"Warning: {stn} has duplicated index. Removing duplicates.")
        
        stn_df.rename(columns=self.dyn_map, inplace=True)
        
        return stn_df

    def _soil_data(self) -> pd.DataFrame:
        """
        reads 07_CAMELS_COL_Soil_characteristics.xlsx file
        """
        df = pd.read_excel(
            os.path.join(self.path, "07_CAMELS_COL_Soil_characteristics.xlsx"),
            index_col=0,
            dtype={0: str},
        ).T

        df.index = [name.split('_')[1] for name in df.index]

        return df

    def _lc_data(self) -> pd.DataFrame:
        """
        reads 06_CAMELS_COL_Land_cover_characteristics.xlsx file
        """
        df = pd.read_excel(
            os.path.join(self.path, "06_CAMELS_COL_Land_cover_characteristics.xlsx"),
            index_col=0,
            dtype={0: str},
        ).T

        df.index = [name.split('_')[1] for name in df.index]

        df = df.dropna(axis=1, how='all')

        return df
    
    def _geol_data(self) -> pd.DataFrame:
        """
        reads 05_CAMELS_COL_Geology_characteristics.xlsx file
        """
        df = pd.read_excel(
            os.path.join(self.path, "05_CAMELS_COL_Geologic_characteristics.xlsx"),
            index_col=0,
            dtype={0: str},
            usecols="D:MM",
        ).T

        df.index = [name.split('_')[1] for name in df.index]

        df = df.dropna(axis=1, how='all')
        return df

    def _static_data(self) -> pd.DataFrame:
        """
        static attributes of catchments

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of static features of all catchments of shape (347, 255)
        """

        dfs = []
        idx = 0

        # read all .csv files
        for xlsx_file in [
            '02_CAMELS_COL_Catchment_information',
            '08_CAMELS_COL_Climatic_indices', 
            '09_CAMELS_COL_Hydrological_signatures', 
            '10_CAMELS_COL_Physiograpic_characteristics']:

            #if not  csv_file.endswith('basin_id.csv'):
            df = pd.read_excel(os.path.join(self.path, f"{xlsx_file}.xlsx"), 
                               index_col=0, dtype={0: str})

            df.index = df.index.astype(str)

            dfs.append(df)

            idx += 1
        
        static_data = pd.concat(dfs, axis=1)

        soil = self._soil_data()
        lc = self._lc_data()
        geol = self._geol_data()

        static_data = pd.concat([static_data, soil, lc, geol], axis=1)

        static_data.rename(columns=self.static_map, inplace=True)

        for col, fac in  self.static_factors.items():
            if col in static_data.columns:
                static_data[col] *= fac

        # convert latitude and longitude from EPSG:3395 to EPSG:4326
        R = 6378137.0   # Earth's radius in meters for EPSG:3395
        static_data[gauge_longitude()] = np.degrees(static_data[gauge_longitude()] / R)
        static_data[gauge_latitude()] = np.degrees(2 * np.arctan(np.exp(static_data[gauge_latitude()] / R)) - np.pi / 2)

        return static_data    


class CAMELS_SK(_RainfallRunoff):
    """
    Dataset of 178 catchments from South Korea following the work of 
    `Kim et al., 2025 <https://doi.org/10.5281/zenodo.15073263>`_.
    The dataset consists of 215 static catchment features and 17 dynamic features.
    The dynamic features span from 20000101 to 20191231 with hourly timestep.

    Examples
    ---------
    >>> from aqua_fetch import CAMELS_SK
    >>> dataset = CAMELS_SK()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='2013615', as_dataframe=True)
    >>> df = dynamic['2013615'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (175320, 17)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       178
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (17 out of 178)
       17
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(175320, 17), (175320, 17), (175320, 17),... (175320, 17), (175320, 17)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('2013615', as_dataframe=True,
    ...  dynamic_features=['total_precipitation', 'snow_depth', 'air_temp_obs', 'potential_evaporation', 'q_cms_obs'])
    >>> dynamic['2013615'].shape
       (175320, 17)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='2013615', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['2013615'].shape
    ((1, 215), 1, (175320, 17))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 175320, 'dynamic_features': 17})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (178, 2)
    >>> dataset.stn_coords('2013615')  # returns coordinates of station whose id is 2013615
        35.880798       128.173096
    >>> dataset.stn_coords(['2013615', '2017620'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('2013615')
    # get coordinates of two stations
    >>> dataset.area(['2013615', '2017620'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('2013615')
    
    """

    url = "https://zenodo.org/records/15073264"

    def __init__(self,
                 path=None,
                 timestep:str = "H",
                 to_netcdf: bool = True,
                 **kwargs):
 
        super(CAMELS_SK, self).__init__(
            path=path,  
            timestep=timestep,
            to_netcdf=to_netcdf, 
            **kwargs)
        
        self._download(overwrite=self.overwrite)

        # if self.to_netcdf:
        self._maybe_to_netcdf()
        
        self._unzip_7z_files()

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(
            self.path,
            "shp",
            "KFM_bas.shp"
        )

    @property
    def start(self) -> pd.Timestamp:
        """
        start of data
        """
        return pd.Timestamp('2000-01-01')
    
    @property
    def end(self) -> pd.Timestamp:
        """
        end of data
        """
        return pd.Timestamp('2019-12-31 23:59:59')

    @property
    def ts_path(self) -> os.PathLike:
        return os.path.join(self.path, "timeseries", "timeseries")

    @property
    def static_features(self) -> List[str]:
        return self._static_data(nrows=2).columns.to_list()
    
    @property
    def dynamic_features(self) -> List[str]:
        return self._read_stn_dyn(self.stations()[0], nrows=2).columns.to_list()

    def stations(self) -> List[str]:
        """
        returns names of stations as a list
        """
        return [fname.split('.')[0] for fname in os.listdir(self.ts_path)]

    @property
    def static_map(self) -> Dict[str, str]:
        return {
            'Area': catchment_area(),
            'Area_HydroATLAS': catchment_area_with_specifier('hydroatlas'),
            'Lon_gage': gauge_longitude(),
            'Lat_gage': gauge_latitude(),
            }

    @property
    def dyn_map(self) -> Dict[str, str]:
        """
        dynamic features map for CAMELS-SK catchments
        """
        return {
            # todo not sure about the units of these features
            # 'total_precipitation': total_precipitation(),
            # 'temperature_2m': mean_air_temp_with_specifier('2m'),
            # 'dewpoint_temperature_2m': mean_dewpoint_temperature_at_2m(),
            # 'potential_evaporation': total_potential_evapotranspiration(),
            # 'u_component_of_wind_10m': u_component_of_wind(),
            # 'v_component_of_wind_10m': v_component_of_wind(),
            # 'surface_net_solar_radiation': solar_radiation(),
            # 'air_temp_obs': mean_air_temp(),
            # 'precip_obs': total_precipitation(),
            # 'wind_sp_obs': mean_windspeed(),
            'streamflow': observed_streamflow_cms(),
        }

    def _unzip_7z_files(self):
        # The attributes file is .7z file
        try:
            import py7zr
        except (ModuleNotFoundError, ImportError):
            raise ImportError('py7zr is required to extract the .7z files. Please install it using `pip install py7zr`')

        if not os.path.exists(self.boundary_file):

            fpath = os.path.join(self.path, 'shp.7z')
            with py7zr.SevenZipFile(fpath, mode='r') as z:
                z.extractall(path = self.path)
                print(f'Extracted {fpath}')
        return

    def _read_stn_dyn(self, stn:str, nrows=None)->pd.DataFrame:
        """
        reads dynamic data for a given station
        """
        stn_df = pd.read_csv(
            os.path.join(self.ts_path, f"{stn}.csv"),
            index_col=0, 
            parse_dates=True,
            nrows=nrows,
            )
        
        stn_df.index = pd.to_datetime(stn_df.index)

        if stn_df.index.has_duplicates:
            print(f"Warning: {stn} has duplicated index. Removing duplicates.")
        
        stn_df.rename(columns=self.dyn_map, inplace=True)
        
        return stn_df

    def _static_data(self, nrows:int = None) -> pd.DataFrame:
        """
        static attributes of catchments

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of static features of all catchments of shape (178, 239)
        """

        dfs = []
        idx = 0

        # read all .csv files in static_path
        for csv_file in glob.glob(os.path.join(self.path, '*.csv')):
            df = pd.read_csv(csv_file, index_col=0, nrows=nrows)

            df.index = df.index.astype(str)

            if df.index.duplicated().any():
                if self.verbosity > 1:
                    print(f"Warning: Duplicated indices found in {csv_file}. Dropping duplicates.")
                df = df.drop_duplicates()

            if csv_file.endswith('HydroATLAS.csv'):
                val_stns = [stn for stn in df.index if stn in self.stations()]
                df = df.loc[val_stns, :]

            dfs.append(df)

            idx += 1
        
        static_data = pd.concat(dfs, axis=1)

        # remove duplicated columns
        static_data = static_data.loc[:, ~static_data.columns.duplicated()].copy()

        static_data.rename(columns=self.static_map, inplace=True)

        return static_data    


class CAMELS_LUX(_RainfallRunoff):
    """
    Dataset of 56 catchments from Luxembourg following the work of
    `Nijzink et al., 2025 <https://doi.org/10.5194/essd-2024-482>`_.
    The dataset consists of 61 static catchment features and 25 dynamic features.
    The dynamic features span from 20040101 to 20211231 with daily, hourly, and 15-minute timesteps.
    The data is downloaded from `Zenodo <https://zenodo.org/records/14910359>`_.

    Examples
    ---------
    >>> from aqua_fetch import CAMELS_LUX
    >>> dataset = CAMELS_LUX()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='ID_02', as_dataframe=True)
    >>> df = dynamic['ID_02'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (6209, 25)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       56
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (5)
       5
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(6209, 25), (6209, 25), (6209, 25),... (6209, 25), (6209, 25)]

    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('ID_02', as_dataframe=True,
    ...  dynamic_features=['pcp_mm_station', 'rh_%', 'airtemp_C_mean', 'pet_mm_pm', 'q_cms_obs'])
    >>> dynamic['ID_02'].shape
       (6209, 5)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='ID_02', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['ID_02'].shape
    ((1, 61), 1, (6209, 25))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 6209, 'dynamic_features': 25})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (56, 2)
    >>> dataset.stn_coords('ID_02')  # returns coordinates of station whose id is ID_02
        49.586288       6.14908
    >>> dataset.stn_coords(['ID_02', 'ID_01'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('ID_02')
    # get coordinates of two stations
    >>> dataset.area(['ID_02', 'ID_01'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('ID_02')
    ...
    # if we want to get hourly data we can do as below
    >>> dataset = CAMELS_LUX(timestep='H')
    >>> _, dynamic = dataset.fetch(stations='ID_02', as_dataframe=True)
    >>> df.shape
    (149016, 25)   
    ...
    # if we want to get 15Min data we can do as below
    >>> dataset = CAMELS_LUX(timestep='15Min')
    >>> _, dynamic = dataset.fetch(stations='ID_02', as_dataframe=True)
    >>> df.shape
    (596061, 25) 
    """

    url = "https://zenodo.org/records/14910359"

    def __init__(self,
                 path=None,
                 timestep:str = 'D',
                 overwrite=False,
                 to_netcdf: bool = True,
                 **kwargs):

        assert timestep in ['D', 'H', '15Min'], "timestep must be one of ['D', 'H', '15Min']"

        super(CAMELS_LUX, self).__init__(
            path=path, 
            timestep=timestep,
            to_netcdf=to_netcdf, 
            **kwargs)
        
        self._download(overwrite=overwrite)

        # if self.to_netcdf:
        self._maybe_to_netcdf()

    @property
    def static_features(self) -> List[str]:
        return self._static_data().columns.to_list()

    @property
    def dynamic_features(self) -> List[str]:
        df = self._read_stn_dyn(self.stations()[0], nrows=2)
        return df.columns.to_list()

    def stations(self) -> List[str]:
        """
        returns names of stations a list
        """
        return pd.read_csv(
            os.path.join(self.path, "CAMELS-LUX", "basin_id.csv"),
            header=None,
            index_col=0,
            dtype={0: str}
        ).index.to_list()

    @property
    def boundary_file(self):
        return os.path.join(
            self.path,
            "CAMELS-LUX_shapefiles",
            "catchments_CAMELS-LUX.shp"
        )
    
    @property
    def static_map(self) -> Dict[str, str]:
        return {
            'Lat': gauge_latitude(),
            'Lon': gauge_longitude(),
            'area_km2': catchment_area(),
            'SLOPE_MEAN': slope('degree'),
            'Z_MEAN': catchment_elevation_meters(),
            'grassland': grass_fraction(),
            'agricultural_land': crop_fraction(),
            'urban': urban_fraction(),
            'perimeter_km': catchment_perimeter(),
        }
    
    @property
    def static_factors(self) -> Dict[str, float]:
        """
        static factors for CAMELS-LUX catchments
        """
        return {
            urban_fraction(): 0.01,
            grass_fraction(): 0.01,
            crop_fraction(): 0.01,
        }

    @property
    def dyn_map(self) -> Dict[str, str]:
        """
        dynamic features map for CAMELS-LUX catchments
        """
        return {
            'Q': observed_streamflow_cms(),
            'Qspec': observed_streamflow_mm(),
            'RR_rad': total_precipitation_with_specifier('radar'),
            'RR_stn': total_precipitation_with_specifier('station'),
            'tp': total_precipitation_with_specifier('era5'),
            't2m': mean_air_temp(),
            'PET_Oudin': total_potential_evapotranspiration_with_specifier('oudin'),
            'PET_PM': total_potential_evapotranspiration_with_specifier('pm'),
            'q': mean_specific_humidity(),  # todo : convert from kg/kg -> g/kg
            'rh': mean_rel_hum(),
            'ws10500': mean_windspeed(),
            'swvl1': soil_moisture_layer1(),
            'swvl2': soil_moisture_layer2(),
            'swvl3': soil_moisture_layer3(),
            'swvl4': soil_moisture_layer4(),
        }

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp('2004-01-01')
    
    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp('2021-12-31')
        
    @property
    def ts_path(self) -> os.PathLike:
        return os.path.join(self.path, "CAMELS-LUX", "timeseries")
    
    @property
    def topo_fpath(self) -> os.PathLike:
        return os.path.join(self.path, "CAMELS-LUX", "CAMELS_LUX_topographic_attributes.csv")
    
    @property
    def daily_ts_path(self) -> os.PathLike:
        return os.path.join(self.ts_path, "daily")
    
    @property
    def hourly_ts_path(self) -> os.PathLike:
        return os.path.join(self.ts_path, "hourly")
    
    @property
    def subhourly_ts_path(self) -> os.PathLike:
        return os.path.join(self.ts_path, "15Min")

    def _static_data(self) -> pd.DataFrame:
        """
        static attributes of catchments

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of static features of all catchments of shape (56, 61)
        """

        dfs = []
        idx = 0

        # read all .csv files
        for csv_file in glob.glob(os.path.join(self.path, 'CAMELS-LUX', '*.csv')):

            if not  csv_file.endswith('basin_id.csv'):
                df = pd.read_csv(csv_file, index_col=0, dtype={0: str})

                df.index = df.index.astype(str)

                dfs.append(df)

                idx += 1
        
        static_data = pd.concat(dfs, axis=1)

        static_data.rename(columns=self.static_map, inplace=True)

        # static_factors should be called after renaming the columns
        for col, fac in  self.static_factors.items():
            if col in static_data.columns:
                static_data[col] *= fac

        return static_data

    def _read_stn_dyn(self, stn:str, nrows=None)->pd.DataFrame:
        """
        reads dynamic data for a given station
        """
        ts_path = {
            'D': self.daily_ts_path,
            'H': self.hourly_ts_path,
            '15Min': self.subhourly_ts_path
        }

        stn_df = pd.read_csv(
            os.path.join(ts_path[self.timestep], f"CAMELS_LUX_hydromet_timeseries_{stn}.csv"), 
            index_col=0, 
            parse_dates=True,
            nrows=nrows,
            )
        
        stn_df.index = pd.to_datetime(stn_df.index)

        if stn_df.index.has_duplicates:
            print(f"Warning: {stn} has duplicated index. Removing duplicates.")
        
        # drop rows with duplicated index, ideally there should not be any
        if self.timestep == '15Min':
            stn_df = stn_df[~stn_df.index.duplicated(keep='first')]

        stn_df.rename(columns=self.dyn_map, inplace=True)
        
        return stn_df


class CAMELS_DEBY(_RainfallRunoff):
    """
    lumped and gridded data at hourly and daily timestep for 210
    Bavarian (Germany) catchments following the work of
    `Anwar et al., 2025 <https://doi.org/10.5281/zenodo.14893685>`_.
    """


class HYD_Responses(_RainfallRunoff):
    """
    `von Matt et al., 2025 <https://doi.org/10.5281/zenodo.14713274>
    """


class CAMELS_ES(_RainfallRunoff):
    """
    """
    url = "https://zenodo.org/records/15040948"


class CAMELS_FI(_RainfallRunoff):
    """
    Dataset of 320 Finnish catchments with 16 dynamic features and 106 static features.
    The dynamic features span from 19610101 to 20231231 with daily timestep.
    The data is downloaded from `Zenodo <https://zenodo.org/records/16257216>`_.

    
    Examples
    ---------
    >>> from aqua_fetch import CAMELS_FI
    >>> dataset = CAMELS_FI()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='1156', as_dataframe=True)
    >>> df = dynamic['1156'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (23010, 16)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       320
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (32)
       32
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(23010, 16), (23010, 16), (23010, 16),... (23010, 16), (23010, 16)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('1156', as_dataframe=True,
    ...  dynamic_features=['pcp_mm', 'snowdepth_m', 'airtemp_C_mean', 'pet_mm', 'q_cms_obs'])
    >>> dynamic['1156'].shape
       (23010, 5)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='1156', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['1156'].shape
    ((1, 106), 1, (23010, 5))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 23010, 'dynamic_features': 16})
    ...
    >>> len(dynamic.data_vars)   # -> 10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (320, 2)
    >>> dataset.stn_coords('1156')  # returns coordinates of station whose id is 1156
        62.253101       24.444099
    >>> dataset.stn_coords(['1156', '1116'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('1156')
    # get coordinates of two stations
    >>> dataset.area(['1156', '1116'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('1156')
    """

    url = "https://zenodo.org/records/16257216"

    def __init__(self,
                 path=None,
                 overwrite=False,
                 to_netcdf: bool = True,
                 **kwargs):

        super(CAMELS_FI, self).__init__(
            path=path, 
            to_netcdf=to_netcdf, 
            **kwargs)
        
        self._download(overwrite=overwrite)

        self._unzip_boundaries()

        self._maybe_to_netcdf()

    @property
    def data_path(self) -> os.PathLike:
        return os.path.join(self.path, 
                            "CAMELS-FI", 
                            "CAMELS-FI",
                            "data")
    
    @property
    def boundary_path(self) -> os.PathLike:
        return os.path.join(self.data_path, "CAMELS_FI_catchment_boundaries")

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(
            self.boundary_path,
            "CAMELS_FI_catchment_boundaries.shp"
        )
    
    @property
    def ts_path(self) -> os.PathLike:
        return os.path.join(
            self.data_path,
            "timeseries",
        )

    def stations(self) -> List[str]:
        return [
            fname.split('.')[0].split('_')[4] for fname in os.listdir(self.ts_path) if fname.endswith('.csv')
            ]

    @property
    def dyn_map(self) -> Dict[str, str]:
        """
        dynamic features map for CAMELS-FI catchments
        """
        return {
            'discharge_vol': observed_streamflow_cms(),
            'discharge_spec': observed_streamflow_mm(),
            'precipitation': total_precipitation(),
            'pet': total_potential_evapotranspiration(),
            'temperature_min': min_air_temp(),
            'temperature_mean': mean_air_temp(),
            'temperature_max': max_air_temp(),
            'humidity_rel': mean_rel_hum(),
            'snow_depth': snow_depth(),  # change from cm to m
            'swe': snow_water_equivalent_with_specifier('era5'),
            'swe_cci3-1': snow_water_equivalent_with_specifier('cci3-1'),
        }

    @property
    def static_map(self) -> Dict[str, str]:
        return {
            'gauge_lat': gauge_latitude(),
            'gauge_lon': gauge_longitude(),
            'area': catchment_area(),   
            'slope': slope('percent'),
            #'slope_fdc': catchment_elevation_meters(),
            'aridity': aridity_index(),
            'grass_perc_2000': grass_fraction_with_specifier('2000'),
            'urban_perc_2000': urban_fraction_with_specifier('2000'),
            'crop_perc_2000': crop_fraction_with_specifier('2000'),
            'grass_perc_2006': grass_fraction_with_specifier('2006'),
            'urban_perc_2006': urban_fraction_with_specifier('2006'),
            'crop_perc_2006': crop_fraction_with_specifier('2006'),
            'grass_perc_2012': grass_fraction_with_specifier('2012'),
            'urban_perc_2012': urban_fraction_with_specifier('2012'),
            'crop_perc_2012': crop_fraction_with_specifier('2012'),
            'grass_perc_2018': grass_fraction_with_specifier('2018'),
            'urban_perc_2018': urban_fraction_with_specifier('2018'),
            'crop_perc_2018': crop_fraction_with_specifier('2018'), 
            'elev_gauge': gauge_elevation_meters(),
            'elev_50': med_catchment_elevation_meters(),
            'soil_depth': soil_depth(),
            'dens_inhabitants': population_density(),
        }

    @property
    def static_factors(self) -> Dict[str, float]:
        """
        static factors for CAMELS-LUX catchments
        """
        return {
            grass_fraction_with_specifier('2000'): 0.01,
            urban_fraction_with_specifier('2000'): 0.01,
            crop_fraction_with_specifier('2000'): 0.01,
            grass_fraction_with_specifier('2006'): 0.01,
            urban_fraction_with_specifier('2006'): 0.01,
            crop_fraction_with_specifier('2006'): 0.01,
            grass_fraction_with_specifier('2012'): 0.01,
            urban_fraction_with_specifier('2012'): 0.01,
            crop_fraction_with_specifier('2012'): 0.01,
            grass_fraction_with_specifier('2018'): 0.01,
            urban_fraction_with_specifier('2018'): 0.01,
            crop_fraction_with_specifier('2018'): 0.01,
        }

    @property
    def start(self) -> pd.Timestamp:
        """
        start of data
        """
        return pd.Timestamp('1961-01-01')
    
    @property
    def end(self) -> pd.Timestamp:
        """
        end of data
        """
        return pd.Timestamp('2023-12-31')

    def _static_data(self) -> pd.DataFrame:
        """
        static attributes of catchments

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of static features of all catchments of shape (320, 106)
        """

        csv_files = glob.glob(os.path.join(self.data_path, '*.csv'))

        dfs = []
        for csv_file in csv_files:

            df = pd.read_csv(
                csv_file, 
                index_col=0, 
                dtype={0: str}
            )

            df.index = df.index.astype(str)

            dfs.append(df)
        
        static_data = pd.concat(dfs, axis=1)

        static_data.rename(columns=self.static_map, inplace=True)

        # static_factors should be called after renaming the columns
        for col, fac in  self.static_factors.items():
            if col in static_data.columns:
                static_data[col] *= fac
        
        return static_data

    def _read_stn_dyn(self, stn:str, nrows=None)->pd.DataFrame:
        """
        reads dynamic data for a given station
        """

        fpath = os.path.join(
            self.ts_path, 
            f"CAMELS_FI_hydromet_timeseries_{stn}_19610101-20231231.csv")
        
        df = pd.read_csv(fpath, index_col=0, parse_dates=True, nrows=nrows)

        df.index = pd.to_datetime(df.index)
        if df.index.has_duplicates:
            print(f"Warning: {stn} has duplicated index. Removing duplicates.")
          
        df.rename(columns=self.dyn_map, inplace=True)

        return df
    
    def _unzip_boundaries(self):
        if not os.path.exists(self.boundary_path):
            zip_file = os.path.join(self.data_path, "CAMELS_FI_catchment_boundaries.zip")
            if os.path.exists(zip_file):
                if self.verbosity:
                    print(f"Unzipping boundary file {os.path.basename(zip_file)} to {self.data_path}")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(self.boundary_path)
            else:
                raise FileNotFoundError(f"Boundary file {zip_file} not found.")
        return


class CAMELSH(_RainfallRunoff):
    """
    Dataset of 5,767 catchments from united states of america following the work of
    `Tran et al., (2025) <https://doi.org/10.1038/s41597-025-05612-6>`_ . It consists
    of hourly data with 13 dynamic features and 779 static features. The dynamic features
    span from 19800101 to 20241231 with hourly timestep. The data is downloaded from
    `Zenodo <https://zenodo.org/records/16729675>`_.

    Please note that usage of this dataset requires xarray and netCDF4 libraries.

    Examples
    --------
    >>> from aqua_fetch import CAMELSH
    >>> dataset = CAMELSH()
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       5767
    ... # get data by station id/name
    >>> _, dynamic = dataset.fetch(stations='02342070', as_dataframe=True)
    >>> df = dynamic['02342070'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (394488, 13)
    ...
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (67 out of 5767)
       67
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(394488, 13), (394488, 8), (394488, 13),... (394488, 13), (394488, 13)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('02342070', as_dataframe=True,
    ...  dynamic_features=['SWdown', 'pcp_mm', 'pet_mm', 'airtemp_C_mean', 'q_cms_obs'])
    >>> dynamic['02342070'].shape
       (394488, 5)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='02342070', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['02342070'].shape
    ((1, 779), 1, (394488, 13))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 394488, 'dynamic_features': 8})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (5767, 2)
    >>> dataset.stn_coords('02342070')  # returns coordinates of station whose id is 02342070
        32.37431	-84.957993
    >>> dataset.stn_coords(['02342070', '14316700'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('02342070')
    # get coordinates of two stations
    >>> dataset.area(['02342070', '14316700'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('02342070')
  
    
    """
    url = {
        "Hourly2.zip": "https://zenodo.org/records/16729675",  # contains observed q and water level
        "timeseries_nonobs.7z": "https://zenodo.org/records/15070091",  # contains NLDAS forcing data
        "timeseries.7z": "https://zenodo.org/records/15066778",
        "attributes.7z": "https://zenodo.org/records/15066778",
        "info.csv": "https://zenodo.org/records/15066778",
        "shapefiles.7z": "https://zenodo.org/records/15066778"
    }


    def __init__(self,
                 path=None,
                 overwrite=False,
                 **kwargs,
    ):
        super(CAMELSH, self).__init__(
        path=path,
        timestep="H",
        **kwargs)
            
        for fname, url in self.url.items():
            fpath = os.path.join(self.path, fname)

            if not os.path.exists(fpath) or overwrite:
                download_and_unzip(self.path, url, include=[fname], verbosity=self.verbosity)

            uzipped_dir_path = os.path.join(self.path, fname.split('.')[0])
            if not os.path.exists(uzipped_dir_path):
                unzip(self.path, keep_parent_dir=True, verbosity=self.verbosity)

        self.__stations = [fname.split('_')[0] for fname in os.listdir(self.h2_path)]

    def stations(self) -> List[str]:
        return self.__stations

    @property
    def static_map(self) -> Dict[str, str]:
        return {
            'LAT_GAGE': gauge_latitude(),
            'LNG_GAGE': gauge_longitude(),
            #'LAT_CENT': centroid_latitude(),
            #'LONG_CENT': centroid_longitude(),
            'ELEV_MEAN_M_BASIN': catchment_elevation_meters(),
            'DRAIN_SQKM': catchment_area(),
            'ELEV_SITE_M': gauge_elevation_meters(),
            'SLOPE_PCT': slope('percent'),
            'PDEN_2000_BLOCK': population_density(2000),
            'PDEN_DAY_LANDSCAN_2007': population_density(2007),
            #'PDEN_NIGHT_LANDSCAN_2007': population_density(2007),
            # 'CLAYAVE': clay_content(),
            # 'SILTAVE': silt_content(),
            # 'SANDAVE': sand_content(),
        }

    @property
    def dyn_map(self) -> Dict[str, str]:
        return {
            #'water_level': water_level('m'),
            'Tair': mean_air_temp(),
            'PotEvap': total_potential_evapotranspiration(),
            'Rainf': total_precipitation(),
            'streamflow': observed_streamflow_cms()
        }

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(
            self.path,
            "shapefiles",
            "CAMELSH_shapefile.shp"
        )

    @property
    def boundary_id_map(self) -> str:
        return "GAGE_ID"

    @property
    def h2_path(self) -> os.PathLike:
        return os.path.join(self.path, "Hourly2", "Hourly2")
    
    @property
    def attr_path(self) -> os.PathLike:
        return os.path.join(self.path, "attributes")
    
    @property
    def sf_path(self) -> os.PathLike:
        return os.path.join(self.path, "shapefiles")
    
    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(
            self.sf_path,
            "CAMELSH_shapefile.shp"
        )
    
    @property
    def nonobs_path(self) -> os.PathLike:
        return os.path.join(self.path, "timeseries_nonobs", "Data", "CAMELSH", "timeseries_nonobs")

    @property
    def timeseries_path(self) -> os.PathLike:
        return os.path.join(self.path, "timeseries", "Data", "CAMELSH", "timeseries")

    def _read_stn_q(self, stn):
        fpath = os.path.join(self.h2_path, f"{stn}_hourly.nc")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"q data for station {stn} not found in {self.h2_path}")
        
        ds = xr.open_dataset(fpath, engine='netcdf4')
        return ds

    def _read_stn_forcing(self, stn):
        fpath = os.path.join(self.nonobs_path, f"{stn}.nc")
        if not os.path.exists(fpath):
            fpath = os.path.join(self.timeseries_path, f"{stn}.nc")
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"forcing data for station {stn} not found in {self.nonobs_path}")
        
        ds = xr.open_dataset(fpath, engine='netcdf4')

        # todo : what is difference between Streamflow in forcing and streamflow in Hourly2 path?
        ds = ds.drop_vars("Streamflow", errors="ignore")

        ds = ds.rename({'DateTime': 'time'})
        return ds    
    
    def _read_stn_dyn(self, stn:str, nrows=None) -> pd.DataFrame:
        q = self._read_stn_q(stn)
        forcing = self._read_stn_forcing(stn)
        # todo : converting to pandas will make the code extremely slow
        # conversion should be done after we have fetched data from all stations and if required
        ds = xr.merge([q, forcing]).to_pandas()
        
        ds.rename(columns=self.dyn_map, inplace=True)
        return ds

    def _static_data(self) -> pd.DataFrame:
        """
        reads static data for all stations
        """
        csv_files = glob.glob(os.path.join(self.attr_path, '*.csv'))

        dfs = []
        for csv_file in csv_files:

            if 'attributes_hydroATLAS.csv' in csv_file:
                df = pd.read_csv(
                csv_file, 
                index_col=0, 
                sep='\t',
                dtype={0: str})
            else:
                df = pd.read_csv(
                csv_file, 
                index_col=0, 
                dtype={0: str})
            df.index = df.index.astype(str)
            dfs.append(df)

        df = pd.concat(dfs, axis=1)

        # drop duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]

        # rename columns using self.static_map
        df = df.rename(columns=self.static_map)
        return df
