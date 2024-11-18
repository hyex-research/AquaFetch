
import os
from typing import List, Union

import pandas as pd

from .._backend import xarray as xr
from ._gsha import _GSHA


class Arcticnet(_GSHA):
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
        super().__init__(path=path, gsha_path=gsha_path, verbosity=verbosity, **kwargs)
    
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.metadata = self.get_metadata()
        self._get_q()

        self._stations = [stn for stn in self.all_stations() if stn in self.gsha_arctic_stns()]

        self._static_features = self.gsha.static_features

    def stations(self)->List[str]:
        return self._stations

    def all_stations(self):
        return self.metadata.index.astype(str).tolist()

    @property
    def agency_name(self)->str:
        return 'arcticnet'

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
