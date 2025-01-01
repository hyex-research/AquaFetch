import os
import time
import zipfile
import warnings
from pathlib import Path
from typing import Union, List
import concurrent.futures as cf
from urllib.error import HTTPError

import numpy as np
import pandas as pd

from ._misc import _EStreams
from ..utils import get_cpus
from .._backend import xarray as xr

# todo: why concatenating the 1077 stations in prior to 2023 and 833 
# stations from 2023 become 1181? Like a lot of new stations come in 2023?

class Poland(_EStreams):
    """
    Data of 1287 catchments of Poland. 
    The observed streamflow data is downloaded from 
    https://danepubliczne.imgw.pl .
    The meteorological data, static catchment 
    features and catchment boundaries are
    taken from :py:class:`water_quality.EStreams` follwoing the works
    of `Nascimento et al., 2024 <https://doi.org/10.5194/hess-25-471-2021>`_ . Therefore,
    the number of staic features are 35 and dynamic features are 27 and the
    data is available from 1992-01-01 to 2020-06-31.
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
        return 'PL'

    def gauge_id_basin_id_map(self)->dict:
        # guage_id '149180020'
        # basin_id 'PL000001'
        # '149180020' -> 'PL000001'
        return {k:v for v,k in self.md['gauge_id'].to_dict().items()}

    def basin_id_gauge_id_map(self)->dict:
        # guage_id '149180020'
        # basin_id 'PL000001'
        # 'PL000001' -> '149180020'
        return self.md['gauge_id'].to_dict()

    @property
    def zip_files_dir(self)->str:
        """path where zip files will be stored"""
        return os.path.join(self.path, 'zip_files')

    @property
    def csv_files_dir(self)->str:
        """path where csv (obtained after extracting zip files) files will be stored"""
        return os.path.join(self.path, 'csv_files')

    def stations(self)->List[str]:
        """returns the basin_id of the stations"""
        return self._stations

    def get_q(self, as_dataframe:bool=True):

        fpath = os.path.join(self.path, 'daily_q.csv')

        if not os.path.exists(fpath):
            data = self._make_csv()
        else:
            if self.verbosity: print(f"Reading from existing {fpath} file")
            data = pd.read_csv(fpath, index_col="time")
            data.index = pd.to_datetime(data.index)
            data.index.name = 'time'

        # replace '149180020' with 'PL000001'
        data.rename(columns=self.gauge_id_basin_id_map(), inplace=True)

        # todo: make sure that the following stations actually not have any data
        if data.shape[1]<len(self.stations()):
            warnings.warn(f"{len(self.stations())-data.shape[1]} stations are missing in the downloaded data")
            for stn in self.stations():
                if stn not in data.columns:
                    data[stn] = np.nan

        if as_dataframe:
            return data
        
        return xr.Dataset({stn: xr.DataArray(data.loc[:, stn]) for stn in data.columns})

    def _make_csv(self):

        years = []
        months = []
        for yr in range(1951, 2023):
            for month in range(1, 13):
                month = str(month).zfill(2)
                years.append(yr)
                months.append(month)

        cpus = self.processes or get_cpus()-2

        if self.verbosity:
            print(f"Downloading zip files using {cpus} cpus")

        start = time.time()
        with cf.ProcessPoolExecutor(max_workers=cpus) as executor:
            data = list(executor.map(download_single_file, years, months))

        if self.verbosity: 
            print(f"Downloaded all files in {time.time()-start} seconds")

        data = pd.concat([df for df in data], axis=0)

        if self.verbosity>1:
            print(f"Data until 2022 has shape: {data.shape}")

        data23 =  download_data_2023(2023)

        if self.verbosity>1:
            print(f"Data for 2023 has shape: {data23.shape}")
    
        data = pd.concat([data, data23], axis=0)
        data.sort_index(inplace=True)

        data.index.name = 'time'

        if self.verbosity:
            print(f"Saving daily discharge data {data.shape} to {self.path}")
        data.to_csv(os.path.join(self.path, "daily_q.csv"), index_label="time")

        return data


def download_single_file(year, month:str):
    url = f"https://danepubliczne.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_hydrologiczne/dobowe/{year}/codz_{year}_{month}.zip"

    try:
        df = pd.read_csv(
                        url,
                        compression='zip',
                        encoding="ISO-8859-1",
                        engine='python',
                        on_bad_lines="skip",
                        names=['stn_id', 'year', 'day', 'q_cms', 'month'],
                        usecols=[0, 3, 5, 7, 9],
                        # sometimes casting month to int fails
                        dtype={'stn_id': str, 'year': 'int', 'day': 'int', 'q_cms': np.float32, #'month': 'int'
                               },
                        #parse_dates={'date': ['year', 'month', 'day']},
                        #index_col='date',
                        na_values=[99999.999]
                        )
    except HTTPError:
        raise Exception(f"Failed to download {url}")

    # sometimes (such as 1992-07) month column has missing values
    month = df.loc[:, 'month']
    month = month.ffill()

    yr, month, day = df.loc[:, 'year'].astype(int), month.astype(int), df.loc[:, 'day'].astype(int)    
    df.index = pd.to_datetime(pd.DataFrame({
            'year': yr,
            'month': month,
            'day': day,
        }))

    df = df.pivot_table(index=df.index, columns="stn_id", values="q_cms")
    df.columns = [col.replace(' ', '') for col in df.columns]

    df.rename(columns={"ï»¿149180020": "149180020"}, inplace=True)

    # as per documentation, 99999.999 is missing value
    df.replace(99999.999, np.nan, inplace=True)


    df.index = df.index.tz_localize(None)
    df.sort_index(inplace=True)
    return df


def download_data_2023(year):
    df = pd.read_csv(
        f"https://danepubliczne.imgw.pl/data/dane_pomiarowo_obserwacyjne/dane_hydrologiczne/dobowe/{year}/codz_{year}.zip", 
        compression='zip', 
        encoding="ISO-8859-1",  
        engine='python',
        on_bad_lines="skip",
        sep=';',
        names=['stn_id', 'year', 'day', 'q_cms', 'month'],
        usecols=[0, 3, 5, 7, 9],
        dtype={'stn_id': str, 'year': 'int', 'day': 'int', 'q_cms': np.float32, 'month': 'int'},
        parse_dates={'date': ['year', 'month', 'day']},
        index_col='date',
        na_values=[99999.999]
        )

    df.replace("ï»¿149180020", "149180020", inplace=True)

    df = df.pivot_table(index=df.index, columns="stn_id", values="q_cms")

    # replace empty splace in column names
    df.columns = [col.replace(' ', '') for col in df.columns]

    # as per documentation, 99999.999 is missing value
    df.replace(99999.999, np.nan, inplace=True)

    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        raise Exception(f"Failed to convert index to datetime for {year}")

    df.index = df.index.tz_localize(None)
    return df
