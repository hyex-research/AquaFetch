import os
import time
import zipfile
import warnings
import requests
from pathlib import Path
from typing import Union, List
import concurrent.futures as cf

import numpy as np
import pandas as pd

try:
    import xml.etree.ElementTree as ET
except (ModuleNotFoundError, ImportError):
    ET = None

from ._misc import _EStreams
from ..utils import get_cpus
from .._backend import xarray as xr


class Italy(_EStreams):
    """
    Data of 294 catchments of Italy. 
    The observed streamflow data is downloaded from 
    http://www.hiscentral.isprambiente.gov.it/hiscentral/hydromap.aspx?map=obsclient .
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

        self._stations = self.ispra_stations()
    @property
    def country_name(self)->str:
        return 'IT'

    def gauge_id_basin_id_map(self)->dict:
        # guage_id 'hsl-abr:5010'
        # basin_id 'ITIS0001'
        # 'hsl-abr:5010' -> 'ITIS0001'
        return {k:v for v,k in self.md['gauge_id'].to_dict().items()}
    
    def stations(self)->List[str]:
        """returns the basin_id of the stations"""
        return self._stations

    def ispra_stations_gauge_ids(self)->List[str]:
        return self.md.loc[self.md['gauge_provider']=='IT_ISPRA']['gauge_id'].to_list()

    def ispra_stations(self)->List[str]:
        return self.md.loc[self.md['gauge_provider']=='IT_ISPRA'].index.to_list()
        
    def all_stations(self)->List[str]:
        return self.estreams.country_stations("IT")

    def get_q(self, as_dataframe:bool=True):
        fpath = os.path.join(self.path, 'daily_q.csv')

        if not os.path.exists(fpath) or self.overwrite:

            data = self.download_ispra_data()
        
            data.to_csv(fpath, index_label="time")

        else:
            if self.verbosity > 1: 
                print(f"Reading q from pre-existing {fpath} file")
            data = pd.read_csv(fpath, index_col="time")
            data.index = pd.to_datetime(data.index)
            data.index.name = "time"

        # replace 'hsl-abr:5010' with 'ITIS0001'
        data.rename(columns=self.gauge_id_basin_id_map(), inplace=True)

        if as_dataframe:
            return data
        
        return xr.Dataset({stn: xr.DataArray(data.loc[:, stn]) for stn in data.columns})

    def download_ispra_data(self):      

        if self.verbosity > 1:
            print("Downloading ISPRA data")

        dfs = []

        for idx, station in enumerate(self.ispra_stations_gauge_ids()):

            initial = station.split(":")[0]
            response = requests.get(
                f"http://hydroserver.ddns.net/italia/{initial}/index.php/default/services/cuahsi_1_1.asmx/GetValuesObject?authToken=&location={station}&variable={initial}:Discharge&startDate=1900-01-01&endDate=2020-12-31")
            root = ET.fromstring(response.content)

            namespace = {'ns': 'http://www.cuahsi.org/waterML/1.1/'}
            # Extract the time series data
            timeseries = []
            for value in root.findall('.//ns:value', namespace):
                date_time = value.attrib['dateTime']
                data_value = value.text
                timeseries.append({'dateTime': date_time, 'value': data_value})

            df = pd.DataFrame(timeseries)

            df.index = pd.to_datetime(df.pop('dateTime'))
            df.columns = [station]
            print(idx, station, df.shape)

            dfs.append(df)    

        df = pd.concat(dfs, axis=1)

        return df


def download_ispra_stn(station):
    initial = station.split(":")[0]
    response = requests.get(
        f"http://hydroserver.ddns.net/italia/{initial}/index.php/default/services/cuahsi_1_1.asmx/GetValuesObject?authToken=&location={station}&variable={initial}:Discharge&startDate=1900-01-01&endDate=2020-12-31")
    root = ET.fromstring(response.content)

    namespace = {'ns': 'http://www.cuahsi.org/waterML/1.1/'}
    # Extract the time series data
    timeseries = []
    for value in root.findall('.//ns:value', namespace):
        date_time = value.attrib['dateTime']
        data_value = value.text
        timeseries.append({'dateTime': date_time, 'value': data_value})

    df = pd.DataFrame(timeseries)

    df.index = pd.to_datetime(df.pop('dateTime'))
    df.columns = [station]