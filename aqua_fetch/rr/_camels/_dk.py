import os
from typing import List, Dict

import numpy as np
import pandas as pd

from ..utils import _RainfallRunoff
from ..._geom_utils import tmerc_to_wgs84, transform_geometry
from ...utils import validate_attributes
from .._map import (
    observed_streamflow_cms,
    mean_air_temp,
    total_precipitation,
    total_potential_evapotranspiration,
    simulated_streamflow_cms,
    actual_evapotranspiration,
)
from .._map import catchment_area, gauge_latitude, gauge_longitude, slope


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
        features = validate_attributes(dynamic_features, self.dynamic_features)

        dyn = {stn: self._read_csv(stn).loc[st:en, features] for stn in stations}

        return dyn

    # ETRS89 / UTM zone 32N (EPSG:25832), the CRS of the gauges and the boundaries;
    # tmerc_to_wgs84 reproduces pyproj to within 6 cm on all boundary vertices
    _UTM32N = dict(lon_0=9.0, k0=0.9996, false_easting=500000.0, false_northing=0.0)

    def transform_stn_coords(self, df:pd.DataFrame)->pd.DataFrame:
        # the projected easting is in the 'long' column, the northing in 'lat'
        lat, lon = tmerc_to_wgs84(df['long'].to_numpy(float), df['lat'].to_numpy(float),
                                  **self._UTM32N)
        return pd.DataFrame({'lat': lat, 'long': lon}, index=df.index)

    def transform_boundary(self, boundary):
        """
        Transforms a catchment boundary, holes and all, from ETRS89 / UTM zone
        32N (EPSG:25832) to WGS84 lon/lat, so that it matches the gauge
        coordinates.
        """
        return transform_geometry(boundary, lambda x, y: tmerc_to_wgs84(x, y, **self._UTM32N))
