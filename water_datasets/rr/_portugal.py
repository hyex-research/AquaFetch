
import os
import time
import warnings
import requests
from io import StringIO
from typing import Union, List
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from ._misc import _EStreams
from ..utils import get_cpus
from .._backend import xarray as xr


class Portugal(_EStreams):
    """
    Data of 280 catchments of Portugal.
    The observed streamflow data stations is downloaded from 
    https://snirh.apambiente.pt .
    The meteorological data, static catchment 
    features and catchment boundaries are
    taken from :py:class:`water_quality.EStreams` follwoing the works
    of `Nascimento et al., 2024 <https://doi.org/10.5194/hess-25-471-2021>`_ project. Therefore,
    the number of staic features are 35 and dynamic features are 27 and the
    data is available from 1972-01-01 to 2022-12-31 .
    """
    def __init__(
            self, 
            path:Union[str, os.PathLike] = None,
            estreams_path:Union[str, os.PathLike] = None,
            verbosity:int=1,
            **kwargs):

        super().__init__(path=path, estreams_path=estreams_path, verbosity=verbosity, **kwargs)

        fpath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'portugal_stn_codes.csv')
        self.codes = pd.read_csv(fpath, index_col=0)

    @property
    def country_name(self)->str:
        return 'PT'

    @property
    def start(self)->pd.Timestamp:
        return pd.Timestamp('1972-01-01')

    @property
    def end(self)->pd.Timestamp:
        return pd.Timestamp('2022-12-31')

    def gauge_id_basin_id_map(self)->dict:
        # guage_id '03J/02H'
        # basin_id 'PT000001'
        # '03J/02H' -> 'PT000001'
        return {k:v for v,k in self.md['gauge_id'].to_dict().items()}

    def stations(self)->List[str]:
        return self._stations
    
    def download_q_data_seq(self):
        """downloads q data sequentially"""
        if self.verbosity: print("Downloading q data sequentially")

        start = time.time()

        data = []

        for i in range(len(self.codes)):

            g_code = self.codes.iloc[i, 1]

            stn_data = download_stn_data(g_code)
            stn_data.name = self.codes.index[i]

            data.append(stn_data)

            if self.verbosity and i%5 == 0:
                print(i, "q files downloaded")

        tot_time = round ((time.time() - start) / 60, 2)

        if self.verbosity: print(f"Total Time taken {tot_time} minutes")

        return pd.concat(data, axis=1)
    
    def download_q_data_parallel(self, cpus:int=4):
        """downloads q data in parallel"""
        start = time.time()

        data = []

        with ProcessPoolExecutor(max_workers=cpus) as executor:

            futures = [executor.submit(download_stn_data, g_code) for g_code in self.codes.iloc[:, 1]]

            for i, future in enumerate(futures):

                stn_data = future.result()
                stn_data.name = self.codes.index[i]

                data.append(stn_data)

                if i%10 == 0:
                    print(i, "Done")

        tot_time = round ((time.time() - start) / 60, 2)

        if self.verbosity: print(f"Total Time taken {tot_time} minutes")

        return pd.concat(data, axis=1)

    def get_q(
            self, 
            as_dataframe:bool=True,
            ):
        """
        returns the streamflow data of Portugal as xarray.Dataset or pandas.DataFrame

        Returns
        -------
        xarray.Dataset or pandas.DataFrame. If as_dataframe is True, returns pandas.DataFrame
        with columns as station codes and index as time. If as_dataframe is False, returns
        xarray.Dataset with station codes as variables and time as dimension.
        """
        fname = 'daily_q.csv' 

        fpath = os.path.join(self.path, fname)

        if not os.path.exists(fpath) or self.overwrite:

            if self.verbosity>1: print(f"Downloading q data at {self.path}")

            cpus = self.processes or min(get_cpus() - 2, 16)

            if cpus > 1:
                q_df = self.download_q_data_parallel(cpus=cpus)
            else:
                q_df = self.download_q_data_seq()
        else:
            if self.verbosity: print(f"Reading q data from pre-existing file {fpath}")
            q_df = pd.read_csv(fpath, index_col=0)
            q_df.index = pd.to_datetime(q_df.index, dayfirst=True)
        
        q_df.index.name = 'time'

        # q_df columns are 03J/02H	15G/02H	11H/02H which needs to be mapped to PT000001	PT000002	PT000003
        # because stations are identified by basin_id
        q_df = q_df.rename(columns=self.gauge_id_basin_id_map())

        if as_dataframe:
            return q_df
        return xr.Dataset({stn: xr.DataArray(q_df.loc[:, stn]) for stn in q_df.columns})


def download_stn_data(gauge_code:int)->pd.Series:

    url = f'https://snirh.apambiente.pt/snirh/_dadosbase/site/paraCSV/dados_csv.php?sites={gauge_code}&pars=1850&tmin=01/01/1972&tmax=31/12/2022&formato=csv'

    # Add headers if needed (you may need to adjust these)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }

    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code == 200:
        data = StringIO(response.text)
        # head = pd.read_csv(data, skiprows=1,
        #                 nrows=1)
        df = pd.read_csv(data, skiprows=3, index_col=0, parse_dates=True, dayfirst=True)
        assert df.columns[0] == 'Caudal médio diário (m3/s)'
        s = df.iloc[0:-1, 0]
        s.name = str(gauge_code)
    else:
        warnings.warn(f"Failed to retrieve data: {response.status_code}")
        s = pd.Series(name=str(gauge_code))
    
    return s
