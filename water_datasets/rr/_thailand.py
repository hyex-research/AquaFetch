
import os
from typing import List, Union

import pandas as pd

from .._backend import xarray as xr
from ._gsha import _GSHA


class Thailand(_GSHA):
    """
    Data of 73 catchments of Thailand from 
    `RID project <https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/rid-river/disc_d.html>`_ .
    The meteorological data static catchment features and catchment boundaries 
    taken from `GSHA <https://doi.org/10.5194/essd-16-1559-2024>`_ project. Therefore,
    the number of staic features are 35 and dynamic features are 27 and the
    data is available from 1980-01-01 to 1999-12-31.
    """
    url = {
'disc_d_1980_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1980_RIDall.zip',
'disc_d_1981_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1981_RIDall.zip',
'disc_d_1982_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1982_RIDall.zip',
'disc_d_1983_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1983_RIDall.zip',
'disc_d_1984_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1984_RIDall.zip',
'disc_d_1985_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1985_RIDall.zip',
'disc_d_1986_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1986_RIDall.zip',
'disc_d_1987_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1987_RIDall.zip',
'disc_d_1988_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1988_RIDall.zip',
'disc_d_1989_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1989_RIDall.zip',
'disc_d_1990_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1990_RIDall.zip',
'disc_d_1991_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1991_RIDall.zip',
'disc_d_1992_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1992_RIDall.zip',
'disc_d_1993_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1993_RIDall.zip',
'disc_d_1994_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1994_RIDall.zip',
'disc_d_1995_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1995_RIDall.zip',
'disc_d_1996_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1996_RIDall.zip',
'disc_d_1997_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1997_RIDall.zip',
'disc_d_1998_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1998_RIDall.zip',
'disc_d_1999_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1999_RIDall.zip',
    }

    def __init__(
            self, 
            path:Union[str, os.PathLike] = None,
            gsha_path:Union[str, os.PathLike] = None,
            overwrite:bool=False,
            verbosity:int=1,
            **kwargs):
        super().__init__(
            path=path, 
            gsha_path=gsha_path, 
            overwrite=overwrite,
            verbosity=verbosity,
            **kwargs)

        self._download(overwrite=overwrite)

    @property
    def agency_name(self)->str:
        return 'RID'

    @property
    def start(self)->pd.Timestamp:
        return pd.Timestamp('1980-01-01')

    @property
    def end(self)->pd.Timestamp:
        return pd.Timestamp('1999-12-31')

    def get_q(self, as_dataframe:bool=True):
        """reads q"""

        fpath = os.path.join(self.path, 'daily_q.csv')

        if os.path.exists(fpath):
            if self.verbosity:
                print(f"Reading {fpath}")
            data = pd.read_csv(fpath, index_col=0, parse_dates=True)
            if as_dataframe:
                return data
            return xr.Dataset({stn: xr.DataArray(data.loc[:, stn]) for stn in data.columns})

        datas = []
        for year in range(1980, 2000):
            data = self._read_year(year)
            datas.append(data)

        data = pd.concat(datas)
        #data.columns = [column.replace('.', '_') for column in data.columns.tolist()]
        data.index.name = 'time'

        if self.verbosity:
            print(f"Writing {fpath}")
        data.to_csv(fpath)

        if as_dataframe:
            return data

        return xr.Dataset({stn: xr.DataArray(data.loc[:, stn]) for stn in data.columns})

    def _read_year(self, year:int):
        year_path = os.path.join(self.path, f'disc_d_{year}_RIDall')
        yr_dfs = []
        stn_ids = []

        ndays = 365
        if year%4==0:
            ndays = 366

        for file in os.listdir(year_path):
            fpath = os.path.join(year_path, file)

            df = pd.read_csv(
                fpath,
                sep='\t',
                names=['index', 'q_cms'],
                nrows=ndays,
                na_values=-9999.0,
            )

            df.index = pd.to_datetime(df.pop('index'))

            yr_dfs.append(df)

            stn_id = file.split('RID')[1].split('_m3s')[0]
            stn_ids.append(stn_id)

        df = pd.concat(yr_dfs, axis=1)
        df.columns = stn_ids
        return df