
import os
import re
import time
import concurrent.futures as cf
from typing import List, Union

import numpy as np
import pandas as pd

from ..utils import get_cpus
from ..utils import check_attributes
from ..utils import merge_shapefiles
from .._backend import xarray as xr
from ._gsha import _GSHA

# the dates for data to be downloaded 
START_YEAR = 1979
END_YEAR = 2023


class Japan(_GSHA):
    """
    Data of 694 catchments of Japan from 
    `river.go.jp website <http://www1.river.go.jp>`_ .
    The meteorological data static catchment features and catchment boundaries 
    taken from `GSHA <https://doi.org/10.5194/essd-16-1559-2024>`_ project. Therefore,
    the number of staic features are 35 and dynamic features are 27 and the
    data is available from 1979-01-01 to 2022-12-31.
    """

    def __init__(
            self, 
            path:Union[str, os.PathLike] = None,
            gsha_path:Union[str, os.PathLike] = None,
            verbosity:int=1,
            **kwargs):
        super().__init__(path=path, verbosity=verbosity, 
                         gsha_path=gsha_path, **kwargs)
    
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self._stations = self.__stations()

    @property
    def agency_name(self)->str:
        return 'MLIT'

    def _maybe_move_and_merge_shpfiles(self):

        out_shp_file = os.path.join(self.path, "boundaries.shp")

        if not os.path.exists(out_shp_file):
            df = self.gsha._coords()
            jpn_stns = df.loc[df['agency'] == 'MLIT']
            shp_path = os.path.join(self.gsha.path, "WatershedPolygons", 
                                    "WatershedPolygons")
            shp_files = [os.path.join(shp_path, f"{filename}.shp") for filename in jpn_stns['station_id'].values]
            for f in shp_files:
                assert os.path.exists(f)

            merge_shapefiles(shp_files, out_shp_file, add_new_field=True)
        return

    # def stations(self)->List[str]:
    #     return self._stations

    # def __stations(self)->List[str]:
    #     """
    #     returns names of only those stations which are also documented
    #     by GSHA.
    #     """
    #     return [stn.split('_')[0] for stn in self.gsha.agency_stations('MLIT')]    

    def get_q(self, as_dataframe:bool=True)->pd.DataFrame:
        """reads daily streamflow for all stations and puts them in a single
        file named data.csv. If data.csv is already present, then it is read
        and its contents are returned as dataframe.
        """

        if self.timestep in ('daily', 'D'):
            df = download_daily_data(
                self.stations(), 
                self.path, 
                verbosity=self.verbosity,
                cpus=self.processes
                )
        else:
            df = self.get_hourly_data(cpus=self.processes)

        df.index.name = 'time'

        if as_dataframe:
            return df
        
        df = xr.Dataset({stn: xr.DataArray(df.loc[:, stn]) for stn in df.columns})
        return df

    def get_hourly_data(self, cpus=None):

        hourly_file = os.path.join(self.path, 'hourly_data.csv')

        if os.path.exists(hourly_file):
            print(f"reading hourly data from {hourly_file}")
            return pd.read_csv(hourly_file, index_col=0)

        path = os.path.join(self.path, 'hourly_files')
        
        if self.verbosity>0: print(f"preparing hourly data using {cpus} cpus")

        stn_qs = []
        for idx, stn in enumerate(self.stations()):
            stn_q = download_hourly_stn(path, stn=stn, cpus=cpus, verbosity=self.verbosity)

            if self.verbosity>0: print(f"{idx} {stn}, {len(stn_q)}, {stn_q.index[0]}")
            
            stn_qs.append(stn_q)
        
        q = pd.concat(stn_qs, axis=1)

        q.to_csv(hourly_file)
        return q


def download_daily_stn_yr(
        stn:str="309191289913130",
        yr:int=1979,
)->pd.Series:
    """downloads daily data for a year for a station"""

    url = f'http://www1.river.go.jp/cgi-bin/DspWaterData.exe?KIND=7&ID={stn}&BGNDATE={yr}0131&ENDDATE={yr}1231&KAWABOU=NO'
    df = pd.read_html(url, encoding='EUC-JP')[1].loc[2:, 1:].reset_index(drop=True)

    # make a dictionary with months as keys and number of days as values
    days_in_month = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

    # if it is a leap year, change the number of days in February
    if yr % 4 == 0:
        days_in_month[2] = 29

    assert len(df)<13, len(df)

    yearly_data = []
    for i in range(0, len(df)):
        row = df.iloc[i, 0:days_in_month[i+1]]
        yearly_data.append(row)

    stn_data = pd.concat(yearly_data).reset_index(drop=True)
    stn_data.index = pd.date_range(start=f'{yr}-01-01', end=f'{yr}-12-31', freq='D')
    stn_data.name = stn

    return stn_data


def download_daily_data(
        stations:List[str], 
        path:Union[str, os.PathLike], 
        verbosity:int=1,
        cpus:int=None
        ):
    """downloads daily data for all stations"""
    csv_path = os.path.join(path, 'daily_q.csv')

    if os.path.exists(csv_path):
        if verbosity:
            print(f"reading daily data from {csv_path}")
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)

    years = range(START_YEAR, END_YEAR+1)
    stations_ = [[stn]*len(years) for stn in stations]
    # flatten the list
    stations_ = [item for sublist in stations_ for item in sublist]
    years_ = list(years) * len(stations)

    cpus = cpus or get_cpus()-2

    if verbosity:
        print(f"downloading daily data for {len(stations)} stations from {years[0]} to {years[-1]}")
    
    if verbosity>1:
        print(f"Total function calls: {len(stations_)} with {cpus} cpus")

    start = time.time()
    with cf.ProcessPoolExecutor(cpus) as executor:
        results = executor.map(download_daily_stn_yr, stations_, years_)
    
    if verbosity:
        print(f"total time taken to download data: {time.time() - start}")

    all_data = []
    for stn in stations:
        stn_data = []
        for yr in years:
            stn_yr_data = next(results)
            stn_data.append(stn_yr_data)
        
        stn_data = pd.concat(stn_data, axis=0)
        stn_data.name = stn

        if verbosity>2:
            print(f"total number of years: {yr} for {stn} with shape {stn_data.shape}")

        all_data.append(stn_data)
    
    if verbosity>2:
        print(f"total number of stations: {len(all_data)} each with shape {all_data[0].shape}")
    
    all_data = pd.concat(all_data, axis=1)

    all_data = all_data.replace({'−': np.nan, '欠測': np.nan})
    all_data = all_data.astype(np.float32)    
    if verbosity:
        print(f"saving daily data to {csv_path} with shape {all_data.shape}")
    all_data.to_csv(csv_path)
    return


def download_hourly_stn_day(
        stn:str="309191289913130", 
        st:str="20211227",
        en:str="20211227"
        ):
    """download hourly data for a single day for a single station"""
    url = f"http://www1.river.go.jp/cgi-bin/SrchSiteSuiData2.exe?SUIKEI=90336000&BGNDATE={st}&ENDDATE={en}&ID={stn}:0202;"
    
    data = pd.read_html(url)[0]       
    df = data.iloc[7:]
    df.columns = ['date', 'time', stn]

    if len(df)>0:
        # make sure that we have data for all 24 hours
        assert len(df) == 24, len(df)
    else:
        df.pop(stn)
        df.insert(2, stn, [np.nan for _ in range(24)])

    df.index = pd.date_range(pd.Timestamp(st), periods=24, freq="H")

    return df[stn]


def download_hourly_stn(
        path:Union[str, os.PathLike],
        stn:str="301031281101030", 
        st_yr:int=1980, 
        en_yr:int=2024,
        cpus:int=64,
        verbosity:int = 1
        )->pd.Series:
    
    fpath = os.path.join(path, f"{stn}.csv")
    if os.path.exists(fpath):
        if verbosity>0: print(f"{stn} already exists")
        return pd.read_csv(fpath, index_col=0)
    
    starts, ends = [], [] 
    for yr in range(st_yr, en_yr):

        if yr % 4 == 0:
            n_days = 366
        else:
            n_days = 365

        for j_day in range(0, n_days):
            # convert jday to date
            date = pd.Timestamp(f"{yr}-01-01") + pd.Timedelta(days=j_day)
            st = date.strftime("%Y%m%d")
            en = date.strftime("%Y%m%d")

            starts.append(st)
            ends.append(en)

    stations = [stn for _ in range(len(starts))]
    with cf.ProcessPoolExecutor(cpus) as executor:
        results = executor.map(download_hourly_stn_day, stations, starts, ends)

    yr_df = []
    for res in results:
        yr_df.append(res)

    q = pd.concat(yr_df)

    # replace "欠測" with np.nan
    q = q.replace("欠測", np.nan)
    q = q.replace('-', np.nan)

    q = q.dropna().astype(np.float32).sort_index()

    # drop duplicated index
    q = q[~q.index.duplicated(keep='first')]

    q.to_csv(fpath)
    return q
