
import os
from typing import List, Union

import numpy as np
import pandas as pd

from .camels import Camels
from ..utils import check_attributes
from .._backend import shapefile, xarray as xr
from ._gsha import GSHA


class Arcticnet(Camels):
    """
    Data of 106 catchments of arctic region from 
    `r-arcticnet project <https://www.r-arcticnet.sr.unh.edu/v4.0/AllData/index.html>`_ .
    The meteorological data static catchment features and catchment boundaries 
    taken from `GSHA <https://doi.org/10.5194/essd-16-1559-2024>`_ project. Therefore,
    the number of staic features are 35 and dynamic features are 27 and the
    data is available from 1979-01-01 to 2022-12-31.
    """
    #url = "https://zenodo.org/record/7563600"

    def __init__(
            self, 
            path:Union[str, os.PathLike] = None,
            gsha_path:Union[str, os.PathLike] = None,
            verbosity:int=1,
            **kwargs):
        super().__init__(path=path, verbosity=verbosity, **kwargs)
    
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.metadata = self.get_metadata()
        self._get_q()

        if gsha_path is None:
            self.gsha_path = os.path.dirname(self.path)
        else:
            self.gsha_path = gsha_path
        
        self.gsha = GSHA(path=self.gsha_path, verbosity=verbosity)
        self._stations = [stn for stn in self.all_stations() if stn in self.gsha_arctic_stns()]

        self._static_features = self.gsha.static_features
        self.boundary_file = self.gsha.boundary_file

    @property
    def start(self)->pd.Timestamp:
        return pd.Timestamp('1979-01-01')

    @property
    def end(self)->pd.Timestamp:
        return pd.Timestamp('2022-12-31')

    @property
    def dynamic_features(self)->List[str]:
        return ['obs_q_cms'] + self.gsha.dynamic_features
    
    @property
    def static_features(self)->List[str]:
        return self.gsha.static_features
    
    def stations(self)->List[str]:
        return self._stations

    def all_stations(self):
        return self.metadata.index.astype(str).tolist()

    @property
    def _coords_name(self)->List[str]:
        return ['lat', 'long']

    @property
    def _area_name(self)->str:
        return 'area'

    def get_metadata(self):
        metadata_path = os.path.join(self.path, "metadata.csv")
        if not os.path.exists(metadata_path):
            df = pd.read_csv(
                "https://www.r-arcticnet.sr.unh.edu/v4.0/russia-arcticnet/Daily_SiteAttributes.txt", 
                #"https://www.r-arcticnet.sr.unh.edu/v4.0/data/SiteAttributes.txt",
                sep="\t",
                encoding_errors='ignore',
                )
            df.index = df.pop('Code')
            df.to_csv(metadata_path, index=True)
        else:
            df = pd.read_csv(metadata_path, index_col=0)
        return df

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
        >>> from water_datasets import Arcticnet
        >>> dataset = Arcticnet()
        >>> dataset.get_boundary(dataset.stations()[0])
        """

        if shapefile is None:
            raise ModuleNotFoundError("shapefile module is not installed. Please install it to use boundary file")

        from shapefile import Reader

        bndry_sf = Reader(self.boundary_file)
        bndry_shp = bndry_sf.shape(self.gsha.bndry_id_map[f"{catchment_id}_arcticnet"])

        bndry_sf.close()

        xyz = np.array(bndry_shp.points)

        xyz = self.transform_coords(xyz)

        return xyz
    
    def get_q(self, as_dataframe:bool=True):
        nc_path = os.path.join(self.path, "daily_q.nc")

        if os.path.exists(nc_path):
            if self.verbosity:
                print(f"Reading {nc_path}")
            q_ds = xr.open_dataset(nc_path)
        else:
            q_ds = xr.Dataset({stn:self.get_stn_q(stn) for stn in self.stations()})
            # rename dimension/coordinate to 'time' in q_ds
            q_ds = q_ds.rename({'dim_0':'time'})

            encoding = {stn: {'dtype': 'float32', 'zlib': True, 'complevel': 3} for stn in self.stations()}
            if self.verbosity:
                print(f"Writing {nc_path}")
            q_ds.to_netcdf(nc_path, encoding=encoding)

        if as_dataframe:
            q_ds = q_ds.to_dataframe()
        return q_ds

    def _get_q(self, as_dataframe:bool=True):
        q_path = os.path.join(self.path, "daily_q.csv")
        if not os.path.exists(q_path):
            df = pd.read_csv(
                "https://www.r-arcticnet.sr.unh.edu/v4.0/russia-arcticnet/discharge_m3_s_UNH-UCLA.txt", 
                #"https://www.r-arcticnet.sr.unh.edu/v4.0/data/Discharge_ms.txt",
                sep="\t")
            df.to_csv(q_path)
        else:
            df = pd.read_csv(q_path, index_col=0)
        return df

    def get_stn_q(self, stn: str):
        q = self._get_q()
        stn_q = q.loc[q['Code']==int(stn), :].copy()

        stn_q.drop('Code', axis=1, inplace=True)

        # Function to check leap year and adjust February days
        def adjust_feb_days(year):
            if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                return 29
            return 28

        month_days1 = {
            'Jan': 31, 'Feb': 28, # February will be adjusted dynamically
            'Mar': 31, 'Apr': 30, 'May': 31, 'Jun': 30,
            'Jul': 31, 'Aug': 31, 'Sep': 30, 'Oct': 31,
            'Nov': 30, 'Dec': 31
        }

        # Prepare a mapping of month names to the number of days
        month_days2 = {
            1: 31, 2: 28, # February will be adjusted dynamically
            3: 31, 4: 30, 5: 31, 6: 30,
            7: 31, 8: 31, 9: 30, 10: 31,
            11: 30, 12: 31
        }

        # Melt the DataFrame to convert it from wide format to long format
        df_long = stn_q.melt(id_vars=['Year', 'Day'], value_vars=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                        var_name='Month', value_name='Value')

        month_to_int = {month: i + 1 for i, month in enumerate(month_days1.keys())}
        df_long['Month'] = df_long['Month'].map(month_to_int).astype(int)

        def get_days_in_month(row):
            try:
                if int(row.Month) == 2:
                    return adjust_feb_days(row.Year)
                return month_days2[int(row.Month)]
            except KeyError as e:
                print(f"KeyError encountered: {e} with row details: {row}")
                return None  # Or handle the error as needed

        df_long['DaysInMonth'] = df_long.apply(get_days_in_month, axis=1)

        # Filter out invalid days (e.g., February 30)
        df_long = df_long[df_long['Day'] <= df_long['DaysInMonth']]

        # Create a datetime column from the Year, Month, Day
        df_long.index = pd.to_datetime(df_long[['Year', 'Month', 'Day']])

        return pd.Series(df_long['Value'], name=str(stn))        

    def gsha_arctic_stns(self)->List[str]:

        df = self.gsha.wsAll.copy()
        return [stn.split('_')[0] for stn in df[df.index.str.endswith('_arcticnet')].index]

    def _fetch_dynamic_features(
            self,
            stations: list,
            dynamic_features = 'all',
            st=None,
            en=None,
            as_dataframe=False,
            as_ts=False
    ):
        """Fetches dynamic features of station."""
        st, en = self._check_length(st, en)
        features = check_attributes(dynamic_features, self.dynamic_features.copy(), 'dynamic_features')

        daily_q = None

        if 'obs_q_cms' in features:
            daily_q = self.get_q(as_dataframe)
            if isinstance(daily_q, xr.Dataset):
                daily_q = daily_q.sel(time=slice(st, en))[stations]
            else:
                daily_q = daily_q.loc[st:en, stations]
            
            features.remove('obs_q_cms')

        if len(features) == 0:
            return daily_q

        stations_ = [f"{stn}_arcticnet" for stn in stations]
        data = self.gsha.fetch_dynamic_features(stations_, features, st, en, as_dataframe)

        if daily_q is not None:
            if isinstance(daily_q, xr.Dataset):
                assert isinstance(data, xr.Dataset), "xarray dataset not supported"
                data = data.rename({stn:stn.split('_')[0] for stn in data.data_vars})

                # first create a new dimension in daily_q named dynamic_features
                daily_q = daily_q.expand_dims({'dynamic_features': ['obs_q_cms']})
                data = xr.concat([data, daily_q], dim='dynamic_features').sel(time=slice(st, en))
            else:
                # -1 because the data in .nc files hysets starts with 0
                data.rename(columns={stn:stn.split('_')[0] for stn in stations}, inplace=True)
                assert isinstance(data.index, pd.MultiIndex)
                # data is multiindex dataframe but daily_q is not
                # first make daily_q multiindex
                daily_q['dynamic_features'] = 'daily_q'
                daily_q.set_index('dynamic_features', append=True, inplace=True)
                daily_q = daily_q.reorder_levels(['time', 'dynamic_features'])
                data = pd.concat([data, daily_q], axis=0).sort_index()

        return data

    def _fetch_static_features(
            self,
            station="all",
            static_features: Union[str, list] = 'all',
            st=None,
            en=None,
            as_ts=False
    )->pd.DataFrame:
        """Fetches static features of station."""
        if self.verbosity>1:
            print('fetching static features')
        stations = check_attributes(station, self.stations(), 'stations')
        stations_ = [f"{stn}_arcticnet" for stn in stations]
        static_feats = self.gsha.fetch_static_features(stations_, static_features).copy()
        static_feats.index = [stn.split('_')[0] for stn in static_feats.index]
        return static_feats

    def fetch_stations_features(
            self,
            stations: list,
            dynamic_features: Union[str, list, None] = 'all',
            static_features: Union[str, list, None] = None,
            st=None,
            en=None,
            as_dataframe: bool = False,
            **kwargs
    ):
        """
        returns features of multiple stations

        Examples
        --------
        >>> from water_datasets import Arcticnet
        >>> dataset = Arcticnet()
        >>> stations = dataset.stations()
        >>> features = dataset.fetch_stations_features(stations)
        """
        stations = check_attributes(stations, self.stations(), 'stations')

        if dynamic_features is not None:

            dyn = self._fetch_dynamic_features(stations=stations,
                                               dynamic_features=dynamic_features,
                                               as_dataframe=as_dataframe,
                                               st=st,
                                               en=en,
                                               **kwargs
                                               )

            if static_features is not None:  # we want both static and dynamic
                to_return = {}
                static = self._fetch_static_features(station=stations,
                                                     static_features=static_features,
                                                     st=st,
                                                     en=en,
                                                     **kwargs
                                                     )
                to_return['static'] = static
                to_return['dynamic'] = dyn
            else:
                to_return = dyn

        elif static_features is not None:
            # we want only static
            to_return = self._fetch_static_features(
                station=stations,
                static_features=static_features,
                **kwargs
            )
        else:
            raise ValueError(f"""
            static features are {static_features} and dynamic features are also {dynamic_features}""")

        return to_return

    def fetch_static_features(
            self,
            stations: Union[str, List[str]] = "all",
            features:Union[str, List[str]] = "all",
            st=None,
            en=None,
            as_ts=False
    ) -> pd.DataFrame:
        """
        returns static atttributes of one or multiple stations

        Parameters
        ----------
            stations : str
                name/id of station of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.
            st :
            en :
            as_ts :

        Examples
        ---------
        >>> from water_datasets import Arcticnet
        >>> dataset = Arcticnet()
        get the names of stations
        >>> stns = dataset.stations()
        >>> len(stns)
            12004
        get all static data of all stations
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (12004, 27)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('01010070')
        >>> static_data.shape
           (1, 27)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['Drainage_Area_km2', 'Elevation_m'])
        >>> static_data.shape
           (12004, 2)
        """
        return self._fetch_static_features(stations, features, st, en, as_ts)

    @property
    def _q_name(self)->str:
        raise ValueError("This dataset does not have a (observed) discharge attribute")