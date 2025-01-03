import os
from typing import Union, Dict

import numpy as np
import pandas as pd

from .._backend import xarray as xr
from ._gsha import _GSHA

from ._map import (
    catchment_area,
    gauge_latitude,
    gauge_longitude,
    slope
    )


class Spain(_GSHA):
    """
    Data of 889 catchments of Spain from 
    `ceh-es <https://ceh-flumen64.cedex.es>`_ website.
    The meteorological data static catchment features and catchment boundaries 
    taken from `GSHA <https://doi.org/10.5194/essd-16-1559-2024>`_ project. Therefore,
    the number of staic features are 35 and dynamic features are 27 and the
    data is available from 1979-01-01 to 2020-09-30.
    """

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

        self.areas = [
            "CANTABRICO", "DUERO", "EBRO", "GALICIA COSTA",
            "GUADALQUIVIR", "GUADIANA", "JUCAR", "MIÑO-SIL",
            "SEGURA", "TAJO"
        ]

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'area': catchment_area(),
                'lat': gauge_latitude(),
                'long': gauge_longitude(),

        }

    @property
    def end(self)->pd.Timestamp:
        return pd.Timestamp('2020-09-30')
    
    @property
    def agency_name(self)->str:
        return 'AFD'
    
    def daily_q_all_areas(self)->pd.DataFrame:
        """Daily data of gauging stations in river from all areas

        Retuns
        ------
        16_806_305 rows x 3
        """
        dfs = []
        for area in self.areas:
            df = self.daily_q_area(area)
            dfs.append(df)

        return pd.concat(dfs)

    def daily_q_area(self, area:str)->pd.DataFrame:
        """Reads Daily data of gauging stations in river which is in afliq.csv file"""

        url = f"https://ceh-flumen64.cedex.es/anuarioaforos//anuario-2019-2020/{area}/afliq.csv"

        df = pd.read_csv(url, #os.path.join(self.path, area, "afliq.csv"),
                         sep=';')

        idx = pd.to_datetime(df.pop('fecha'), dayfirst=True)
        df.index = idx
        df.index.name = "date"
        df.columns = ['stations', 'height_m', "q_cms"]
        return df

    def get_q(self, as_dataframe:bool=True):
        """
        returns daily q of all stations

        Returns
        -------
        pd.DataFrame
            a pandas dataframe of shape (39721, 1447)
        """

        fpath = os.path.join(self.path, 'daily_q.csv')

        if os.path.exists(fpath):
            data= pd.read_csv(fpath, index_col='Unnamed: 0')
            data.index = pd.to_datetime(data.index)
            data.index.name = 'time'
            if as_dataframe:
                return data
            return xr.Dataset({stn: xr.DataArray(data.loc[:, stn]) for stn in data.columns})

        q = self.daily_q_all_areas()

        st = []
        en = []
        for g_name, grp in q.groupby('stations'):
            st.append(grp.sort_index().index[0])
            en.append(grp.sort_index().index[-1])

        start = pd.to_datetime(st).sort_values()[0]
        end = pd.to_datetime(en).sort_values()[-1]

        daily_qs = []
        for stn, stn_df in q.groupby('stations'):
            q_ts = pd.Series(name=stn,
                             index=pd.date_range(start, end=end, freq="d"),
                             dtype=np.float32)
            q_ts[stn_df.index] = stn_df['q_cms']
            daily_qs.append(q_ts)

        data = pd.concat(daily_qs, axis=1)

        data.to_csv(fpath)

        data.index.name = 'time'

        if as_dataframe:
            return data
    
        return xr.Dataset({stn: xr.DataArray(data.loc[:, stn]) for stn in data.columns})
