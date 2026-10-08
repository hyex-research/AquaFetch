import os
import glob
import warnings
from typing import Union, List, Dict

import pandas as pd

from ..utils import _RainfallRunoff
from .._map import observed_streamflow_cms, catchment_area_with_specifier
from .._map import catchment_area, gauge_latitude, gauge_longitude


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
    def boundary_file(self) -> Union[str, os.PathLike]:
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
    def ts_path(self) -> Union[str, os.PathLike]:
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
            #
            # surface_net_solar_radiation / surface_net_thermal_radiation are
            # deliberately NOT mapped, and no dyn_factor can fix them. They are
            # ERA5-Land RUNNING ACCUMULATIONS in J m-2 carried on hourly rows:
            # the value climbs through the day, resets mid-series, and then sits
            # flat overnight holding the previous total. For 2010-06-21 the
            # series runs 14.9e6 (00:00-05:00, flat) -> 17.6e6 (09:00) ->
            # 1.96e6 (10:00, reset) -> 18.4e6 (20:00) -> flat to midnight.
            # Recovering a per-timestep flux needs differencing consecutive
            # steps and handling the reset -- a derivation, not a unit
            # conversion -- so it is left to the user rather than guessed at.
            # Mapping it to swnetrad_wm2 would assert W m-2 for a J m-2
            # accumulation, which is simply false.
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
                if self.verbosity:
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
            warnings.warn(f"{stn} has duplicated index. Removing duplicates.")
        
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
