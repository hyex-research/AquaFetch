import os
import time
import warnings
from typing import Union, List
import concurrent.futures as cf
from urllib.error import HTTPError

import pandas as pd

from ._misc import _EStreams
from ..utils import get_cpus
from .._backend import xarray as xr

START_YEAR = 2012
# todo : why q for only 239 stations is downloaded and others return HTTPError, it is 
# due to wrong fromatting error in pd.read_csv?
# better to save all the data downloaded i.e. water level and temperature as well

class Finland(_EStreams):
    """
    Data of 669 catchments of Finland. 
    The observed streamflow data is downloaded from 
    https://wwwi3.ymparisto.fi .
    The meteorological data, static catchment 
    features and catchment boundaries are
    taken from :py:class:`water_datasets.EStreams` follwoing the works 
    of `Nascimento et al., 2024 <https://doi.org/10.5194/hess-25-471-2021>`_ . Therefore,
    the number of staic features are 35 and dynamic features are 27 and the
    data is available from 2012-01-01 to 2023-06-30.
    """
    def __init__(
            self, 
            path:Union[str, os.PathLike] = None,
            estreams_path:Union[str, os.PathLike] = None,
            verbosity:int=1,
            **kwargs):

        super().__init__(path=path, estreams_path=estreams_path, verbosity=verbosity, **kwargs)

    @property
    def country_name(self)->str:
        return 'FI'

    @property
    def start(self)->pd.Timestamp:
        return pd.Timestamp('2012-01-01')

    def stations(self)->List[str]:
        """returns the basin_id of the stations"""
        return self._stations

    def gauge_id_basin_id_map(self)->dict:
        # guage_id '5902650'
        # basin_id 'FI000001'
        # '5902650' -> 'FI000001'
        return {k:v for v,k in self.md['gauge_id'].to_dict().items()}

    def basin_id_gauge_id_map(self)->dict:
        # guage_id '5902650'
        # basin_id 'FI000001'
        # 'FI000001' -> '5902650'
        return self.md['gauge_id'].to_dict()
    
    def get_q(self, as_dataframe:bool=True, overwrite:bool=False):
        """
        downloads (if not already downloaded) and returns the daily streamflow data of Finland.
        either as pandas dataframe or as xarray dataset.
        """
        fpath = os.path.join(self.path, 'daily_q.csv')

        if not os.path.exists(fpath) or overwrite:

            if self.verbosity: print("Downloading discharge data For Finland")

            df_2001_2023 = self.download_2001_2023()
            df_2024 = self.download_2024()
        
            data = pd.concat([df_2001_2023, df_2024])
            data.index.name = 'time'
            data.to_csv(fpath, index_label="index")

        else:
            if self.verbosity>1: 
                print(f"Reading from pre-existing {fpath} file")
            data = pd.read_csv(fpath, index_col="index",
                                na_values=['-'])
            data.index = pd.to_datetime(data.index)
            data.index.name = 'time'

        if as_dataframe:
            return data
        
        return xr.Dataset({stn: xr.DataArray(data.loc[:, stn]) for stn in data.columns})

    def download_2024(self):

        if self.verbosity: 
            print("Downloading 2024 year data")

        dfs = []
        failures = 0
        for idx, bsn_id in enumerate(self.stations()):

            gauge_id = self.basin_id_gauge_id_map()[bsn_id]

            url = f"https://wwwi3.ymparisto.fi/i3/tilanne/ENG/discharge/image/bigimage/Q{gauge_id}.txt"

            try:
                df = pd.read_csv(url, 
                                #delim_whitespace=True,
                                sep='\s+',
                                skiprows=10, 
                                encoding="ISO-8859-1",
                                decimal=',',
                                names=['date', bsn_id, 'avg', 'min', 'max'],
                                index_col='date',
                                parse_dates=True,
                                dayfirst=True,
                                na_values=['-']
                                )
            except HTTPError:
                failures += 1
                warnings.warn(f" {idx} Failed to download {bsn_id} {failures}", UserWarning)
                df = pd.DataFrame(columns=['date', bsn_id, 'avg', 'min', 'max'])

            if self.verbosity>2:
                print(f"{idx}: for {bsn_id} {df.shape}")

            dfs.append(df[bsn_id].astype('float32'))

        df_2024 = pd.concat(dfs, axis=1)

        if self.verbosity: 
            print(f"Downloaded data of shape {df_2024.shape} for 2024")

        return df_2024

    def download_2001_2023(self):

        if self.verbosity: print("Downloading 2012-2023 year data")

        cpus = self.processes or get_cpus()-2

        if self.verbosity:
            print(f"downloading daily data for {len(self.stations())} stations from {2012} to {2023}")

        if cpus == 1:
            all_data = self.download_data_seq()
        else:
            all_data = self.download_data_parallel(cpus)
        
        if self.verbosity>2:
            print(f"total number of stations: {len(all_data)} each with shape {all_data[0].shape}")
        
        df_2012_2023 = pd.concat(all_data, axis=1)

        if self.verbosity: print(f"Downloaded data of shape {df_2012_2023.shape} for 2001-2023")

        return df_2012_2023

    def download_data_parallel(self, cpus:int=None):
        # todo : taking forever to download the data
        start = time.time()

        stations = self.stations()

        _map = self.basin_id_gauge_id_map()
        basin_ids = [_map[stn] for stn in stations]
        years = range(START_YEAR, 2024)            
        stations_ = [[stn]*len(years) for stn in stations]
        # flatten the list
        stations_ = [item for sublist in stations_ for item in sublist]
        basin_ids_ = [[bsn_id]*len(years) for bsn_id in basin_ids]
        basin_ids_ = [item for sublist in basin_ids_ for item in sublist]
        years_ = list(years) * len(stations)

        if self.verbosity>1:
            print(f"Total function calls: {len(stations_)} with {cpus} cpus")

        with cf.ProcessPoolExecutor(cpus) as executor:
            results = executor.map(download_daily_stn_yr, basin_ids_, stations_, 
                                   years_)        

        if self.verbosity:
            print(f"total time taken to download data: {time.time() - start}")

        all_data = []
        for bsn_id, stn in zip(basin_ids, stations):
            stn_data = []
            for yr in years:
                stn_yr_data = next(results)
                stn_data.append(stn_yr_data[bsn_id])
            
            stn_data = pd.concat(stn_data, axis=0)
            stn_data.name = stn

            if self.verbosity>2:
                print(f"for {stn} with shape {stn_data.shape}")

            all_data.append(stn_data)
        return all_data

    def download_data_seq(self):
        # takes around 1 hour to download all the data
        failures = 0
        dfs = []
        for idx, bsn_id in enumerate(self.stations()):

            gauge_id = self.basin_id_gauge_id_map()[bsn_id]

            stn_dfs = []
            for year in range(2012, 2024):

                url = f"https://wwwi3.ymparisto.fi/i3/kktiedote/ENG/{year}/discharge/image/bigimage/Q{gauge_id}.txt"

                if year == 2012: skiprows = 7 
                elif year in [2013, 2014]: skiprows = 9
                else: skiprows = 10

                try:
                    yr_df = pd.read_csv(url, 
                                    #delim_whitespace=True,
                                    sep='\s+',
                                    skiprows=skiprows, 
                                    encoding="ISO-8859-1",
                                    decimal=',',
                                    names=['date', bsn_id, 'avg', 'min', 'max'],
                                    index_col='date',
                                    parse_dates=True,
                                    dayfirst=True,
                                    na_values=['-'],
                                    )
                except HTTPError:
                    failures += 1
                    warnings.warn(f" {idx} Failed to download {bsn_id} {year} {failures}", UserWarning)
                    yr_df = pd.DataFrame(
                        columns=['date', bsn_id, 'avg', 'min', 'max'],
                    )

                stn_dfs.append(yr_df)
            
            if len(stn_dfs) > 0:
                stn_df = pd.concat(stn_dfs, axis=0)
                if self.verbosity:
                    print(f"{idx}: for {bsn_id} {stn_df.shape} {len(stn_dfs)}")

                dfs.append(stn_df[bsn_id].astype('float32'))

        return dfs

def download_daily_stn_yr(
        gauge_id:str,
        bsn_id:str,
        year:int
        )->pd.DataFrame:

    url = f"https://wwwi3.ymparisto.fi/i3/kktiedote/ENG/{year}/discharge/image/bigimage/Q{gauge_id}.txt"

    if year == 2012: skiprows = 7 
    elif year in [2013, 2014]: skiprows = 9
    else: skiprows = 10

    try:
        yr_df = pd.read_csv(url, 
                        #delim_whitespace=True,
                        sep='\s+',
                        skiprows=skiprows, 
                        encoding="ISO-8859-1",
                        decimal=',',
                        names=['date', bsn_id, 'avg', 'min', 'max'],
                        index_col='date',
                        parse_dates=True,
                        dayfirst=True,
                        na_values=['-'],
                        )
    except HTTPError:
        yr_df = pd.DataFrame(
            columns=['date', bsn_id, 'avg', 'min', 'max'],
        )
    
    return yr_df
