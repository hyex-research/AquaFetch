import os
import warnings
from typing import List, Dict

import numpy as np
import pandas as pd

from ..utils import _RainfallRunoff
from ...utils import download, unzip
from ..._backend import netCDF4
from .._map import (
    observed_streamflow_cms,
    observed_streamflow_mm,
    mean_air_temp,
    total_precipitation,
)
from .._map import catchment_area, gauge_latitude, gauge_longitude, slope


class CAMELS_SE(_RainfallRunoff):
    """
    Dataset of 50 Swedish catchments following the works of
    `Teutschbein et al., 2024 <https://doi.org/10.1002/gdj3.239>`_ . The data is downloaded
    from Swedish National Data Service `website <https://snd.se/en/catalogue/dataset/2023-173>`_ .
    The dataset consists of 76 static catchment features and 4 dynamic features.
    The dynamic features span from 19610101 to 20201231 with daily timestep.

    Examples
    --------
    >>> from aqua_fetch import CAMELS_SE
    >>> dataset = CAMELS_SE()
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
