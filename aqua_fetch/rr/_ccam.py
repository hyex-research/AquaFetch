
import os
import json
from typing import Union, List, Dict

import numpy as np
import pandas as pd

from .utils import _RainfallRunoff
from .._backend import fiona
from .._backend import xarray as xr
from ..utils import merge_shapefiles_fiona
from ..utils import check_attributes, dateandtime_now

from ._map import (
    observed_streamflow_cms,
    observed_streamflow_mm,
    min_air_temp,
    max_air_temp,
    mean_air_temp,
    mean_rel_hum,
    mean_daily_evaporation,
    max_daily_ground_surface_temp,
    min_daily_ground_surface_temp,
    mean_daily_ground_surface_temp,
    total_precipitation,
    sunshine_duration,
    max_windspeed,
    min_windspeed,
    mean_windspeed,
)

from ._map import (
    catchment_area,
    gauge_latitude,
    gauge_longitude,
    slope
    )


class CCAM(_RainfallRunoff):
    """
    Dataset for Yellow River (China) catchments. The CCAM dataset was published by
    `Hao et al., 2021 <https://doi.org/10.5194/essd-13-5591-2021>`_ and has two sets.
    One set consists of catchment attributes, meteorological data, catchment boundaries
    of over 4000 catchments. However this data does not have streamflow data. The second
    set consists of streamflow, catchment attributes, catchment boundaries and meteorological
    data for 102 catchments of Yellow River. Since this second set conforms to the norms
    of CAMELS, this class uses this second set. Therefore, the ``fetch``, ``stations`` and other
    methods/attributes of this class return data of only Yellow River catchments
    and not for whole china. However, the first set of data is can
    also be fetched using `fetch_meteo` method of this class. The temporal extent of both
    sets is from 1999 to 2020. However, the streamflow time series in first set has very
    large number of missing values. The data of Yellow river consists fo 16 dynamic
    features (time series) and 124 static features (catchment attributes).

    Examples
    ---------
    >>> from aqua_fetch import CCAM
    >>> dataset = CCAM()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='0010', as_dataframe=True)
    >>> df = dynamic['0010'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (8035, 16)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       102
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (10 out of 102)
       10
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(8035, 16), (8035, 16), (8035, 16),... (8035, 16), (8035, 16)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('0010', as_dataframe=True,
    ...  dynamic_features=['pcp_mm', 'airtemp_C_mean', 'evap_mm', 'rh_%', 'q_cms_obs'])
    >>> dynamic['0010'].shape
       (8035, 5)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='0010', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['0010'].shape
    ((1, 124), 1, (8035, 8))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 8035, 'dynamic_features': 16})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (102, 2)
    >>> dataset.stn_coords('0010')  # returns coordinates of station whose id is 0010
        36.059803	112.3638
    >>> dataset.stn_coords(['0010', '0104'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('0010')
    # get coordinates of two stations
    >>> dataset.area(['0010', '0104'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('0010')

    """
    url = "https://zenodo.org/record/5729444"

    def __init__(self,
                 path=None,
                 overwrite:bool=False,
                 to_netcdf:bool = True,
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
            require netCDF4 package as well as xarry.
        """
        super(CCAM, self).__init__(path=path, **kwargs)
        self.path = path
        self._download(overwrite=overwrite)

        if to_netcdf:
            self._maybe_to_netcdf()
            self._maybe_meteo_to_nc()
        
        shp_path = os.path.join(self.path,
                                "7_HydroMLYR",
                                "7_HydroMLYR",
                                "0_basin_boundary")

        files = [file for file in os.listdir(shp_path) if file.endswith('.shp')]
        shp_files = [os.path.join(shp_path, shp_file) for shp_file in files]
        boundaries = os.path.join(shp_path, "boundaries")

        if fiona is not None:
            merge_shapefiles_fiona(shp_files, boundaries)

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(
            self.path, 
            "7_HydroMLYR", 
            "7_HydroMLYR", 
            "0_basin_boundary",
            "boundaries",
            "boundaries.shp")

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'area': catchment_area(),
                'slope': slope('mkm-1'),
                'lat': gauge_latitude(),
                'lon': gauge_longitude(),
        }

    @property
    def dyn_map(self):
        return {
        'q': observed_streamflow_cms(),  # todo: check the units
        'tem_min': min_air_temp(),
        'tem_max': max_air_temp(),
        'tem_mean': mean_air_temp(),
        'rhu': mean_rel_hum(),
        'evp': mean_daily_evaporation(),
        'gst_max': max_daily_ground_surface_temp(),
        'gst_min': min_daily_ground_surface_temp(),
        'gst_mean': mean_daily_ground_surface_temp(),
        'pre': total_precipitation(),
        'ssd': sunshine_duration(),
        'win_max': max_windspeed(),
        'win_mean': mean_windspeed(),
        }

    @property
    def meteo_path(self):
        """path where daily meteorological data of stations is present"""
        return os.path.join(self.path, "1_meteorological", '1_meteorological')

    @property
    def meteo_nc_path(self):
        return os.path.join(self.path, "meteo_data.nc")

    @property
    def meteo_stations(self)->List[str]:
        stations = [fpath.split('.')[0] for fpath in os.listdir(self.meteo_path)]
        stations.remove('35616')
        return stations

    @property
    def yr_data_path(self):
        return os.path.join(self.path, "7_HydroMLYR", "7_HydroMLYR", '1_data')

    def stations(self):
        """Returns station ids for catchments on Yellow River"""
        return os.listdir(self.yr_data_path)

    @property
    def dynamic_features(self)->List[str]:
        """names of hydro-meteorological time series data for Yellow River catchments"""

        return [self.dyn_map.get(feat, feat) for feat in ['pre', 'evp', 'gst_mean', 'prs_mean', 'tem_mean', 'rhu', 'win_mean',
       'gst_min', 'prs_min', 'tem_min', 'gst_max', 'prs_max', 'tem_max', 'ssd',
       'win_max', 'q']]

    @property
    def static_features(self)->List[str]:
        """names of static features for Yellow River catchments"""
        attr_fpath = os.path.join(self.yr_data_path, self.stations()[0], 'attributes.json')
        with open(attr_fpath, 'r') as fp:
            data = json.load(fp)
        return [self.static_map.get(feat, feat) for feat in data.keys()]

    @property
    def start(self):  # start of data
        return pd.Timestamp('1999-01-02 00:00:00')

    @property
    def end(self):  # end of data
        return pd.Timestamp('2020-12-31 00:00:00')

    def _read_meteo_from_csv(
            self,
            station:str
    )->pd.DataFrame:
        """returns daily meteorological data of one station as DataFrame after reading it
        from csv file. This data is from 1990-01-01 to 2021-03-31. The returned
        dataframe has following columns
            - 'PRE'
            - 'TEM': temperature
            - 'PRS': pressure
            - 'RHU',
            - 'EVP',
            - 'WIN',
            - 'SSD': sunshine duration
            - 'GST': ground surface temperature
            - 'PET'

        """
        fpath = os.path.join(self.meteo_path, f"{station}.txt")

        df = pd.read_csv(fpath)
        df.index = pd.to_datetime(df.pop("Date"))

        if 'PET' not in df:
            df['PET'] = None

        # following two stations have multiple enteries
        if station in ['17456', '18161']:
            df = drop_duplicate_indices(df)

        return df

    def _maybe_meteo_to_nc(self):
        if os.path.exists(self.meteo_nc_path):
            if self.verbosity>1:
                print(f"meteo data already converted to netcdf file at {self.meteo_nc_path}")
            return
        
        if self.verbosity:
            print(f"converting meteo data to netcdf file")

        stations = os.listdir(self.meteo_path)
        dyn = {}
        for idx, stn in enumerate(stations):

            if stn not in ['35616.txt']:
                stn_id = stn.split('.')[0]

                dyn[stn_id] = self._read_meteo_from_csv(stn_id).astype(np.float32)
            
            if self.verbosity and idx % 500 == 0:
                print(f"{idx}/{len(stations)} stations processed")
            
            elif self.verbosity>1 and idx % 200 == 0:
                print(f"{idx}/{len(stations)} stations processed")

        data_vars = {}
        coords = {}
        for k, v in dyn.items():
            data_vars[k] = (['time', 'dynamic_features'], v)
            index = v.index
            index.name = 'time'
            coords = {
                'dynamic_features': list(v.columns),
                'time': index
            }

        if self.verbosity>1:
            print(f"Creating xarray dataset")

        xds = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs={'date': f"create on {dateandtime_now()}"}
        )

        if self.verbosity>1:
            print(f"creating netcdf file at {self.meteo_nc_path}")
        xds.to_netcdf(self.meteo_nc_path)
        return

    def fetch_meteo(
            self,
            station:Union[str, List[str]] = "all",
            features:Union[str, List[str]] = "all",
            st = '1990-01-01',
            en = '2021-03-31',
            as_dataframe:bool = True
    ):
        """
        fetches meteorological data of 4902 chinese catchments

        Examples
        ---------
        >>> from aqua_fetch import CCAM
        >>> dataset = CCAM()
        >>> dynamic_features = ['PRE', 'TEM', 'PRS', 'RHU', 'EVP', 'WIN', 'PET']
        >>> st = '1999-01-01'
        >>> en = '2020-03-31'
        >>> xds = dataset.fetch_meteo(features=features, st=st, en=en)
        """
        def_features = ['PRE', 'TEM', 'PRS', 'RHU', 'EVP', 'WIN', 'SSD', 'GST', 'PET']
        features = check_attributes(features, def_features)
        stations = check_attributes(station, self.meteo_stations)
        if xr is None:
            raise ModuleNotFoundError(f"xarray must be installed")
        else:

            dyn = xr.open_dataset(self.meteo_nc_path)
            dyn = dyn[stations].sel(dynamic_features=features, time=slice(st, en))
            if as_dataframe:
                dyn = dyn.to_dataframe(['time', 'dynamic_features'])

        return dyn

    def _read_yr_dynamic_from_csv(
            self,
            station:str
        )->pd.DataFrame:
        """
        Reads daily dynamic (meteorological + streamflow) data for one catchment of
        yellow river and returns as DataFrame
        """
        meteo_fpath = os.path.join(self.yr_data_path, station, 'meteorological.txt')
        q_fpath = os.path.join(self.yr_data_path, station, 'streamflow_raw.txt')

        meteo = pd.read_csv(meteo_fpath)
        meteo.index = pd.to_datetime(meteo.pop('date'))
        q = pd.read_csv(q_fpath)
        q.index = pd.to_datetime(q.pop('date'))

        df = pd.concat([meteo, q], axis=1).astype(np.float32)

        df.rename(columns=self.dyn_map, inplace=True)

        df.index.name = 'time'
        df.columns.name = 'dynamic_features'

        return df

    def _read_dynamic(
            self,
            stations,
            dynamic_features,
            st=None,
            en=None)->dict:
        """reads dynamic data of one or more catchments located along Yellow River basin
        """
        attributes = check_attributes(dynamic_features, self.dynamic_features)

        dyn = {stn: self._read_yr_dynamic_from_csv(stn).loc["19990101": "20201231", attributes] for stn in stations}

        # making sure that data for all stations has same dimensions by inserting nans
        # and removign duplicates
        dyn = {stn:drop_duplicate_indices(data) for stn, data in dyn.items()}
        dummy = pd.DataFrame(index=pd.date_range("19990101", "20201231", freq="D"))

        dummy.index.name = 'time'
        dummy.columns.name = 'dynamic_features'

        dyn = {stn: pd.concat([v, dummy], axis=1) for stn, v in dyn.items()}
        return dyn

    def _static_data(self)->pd.DataFrame:
        ds = []
        for stn in self.stations():
            d = self._read_yr_static(stn)
            ds.append(d)
        df = pd.concat(ds, axis=1).transpose()

        df.rename(columns=self.static_map, inplace=True)

        return df

    def _read_yr_static(
            self,
            station:str
        )->pd.Series:
        """
        Reads catchment attributes data for Yellow River catchments
        """
        fpath = os.path.join(self.yr_data_path, station, 'attributes.json')

        with open(fpath, 'r') as fp:
            data = json.load(fp)

        return pd.Series(data, name=station)


def drop_duplicate_indices(df):
    return df[~df.index.duplicated(keep='first')]