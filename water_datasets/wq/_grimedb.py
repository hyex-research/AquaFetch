__all__ = ["GRiMeDB"]

import os
from typing import Union, List

import pandas as pd
import numpy as np

from .._datasets import Datasets
from ..utils import check_attributes


class GRiMeDB(Datasets):
    """
    Global river database of methan concentrations and fluxes following
    `Stanley et al., 2023 <https://doi.org/10.5194/essd-15-2879-2023>`_
    """
    url = {
        "concentrations.csv": "https://portal.edirepository.org/nis/dataviewer?packageid=knb-lter-ntl.420.1&entityid=ba3e270bcab8ace5d157c995e4b791e4",
        "fluxes.csv": "https://portal.edirepository.org/nis/dataviewer?packageid=knb-lter-ntl.420.1&entityid=1a559f00566ed9f9f33ccb0daab0bef5",
        "sites.csv": "https://portal.edirepository.org/nis/dataviewer?packageid=knb-lter-ntl.420.1&entityid=3faa64303d5f5bcd043bb88f6768e603",
        "sources.csv": "https://portal.edirepository.org/nis/dataviewer?packageid=knb-lter-ntl.420.1&entityid=3615386d27a2d148be09e70ac22799e4"
    }

    def __init__(self, path=None, **kwargs):
        super().__init__(path=path, **kwargs)
        self.ds_dir = path
        self._download()

        self._stations = self.sites()['Site_ID'].unique().tolist()
    
    def stations(self)->List[str]:
        self._stations
    
    @property
    def streams(self)->List[str]:
        return self.sites()['Stream_Name'].unique().tolist()
    
    def concentrations(
            self,
            stations: Union[str, List[str]] = "all",
            streams: Union[str, List[str]] = "all",
            parameters: Union[str, List[str]] = "all"
            ):

        """
        Get concentrations data.

        Parameters
        ----------
        stations : Union[str, List[str]], optional
            station ID or list of station IDs, by default "all".
            If given, then ``streams`` must not be given. Check `.stations()` method
            for available stations.
        streams : Union[str, List[str]], optional
            stream name or list of stream names, by default "all".
            If given, then ``stations`` must not be given. Check `.streams` attribute
            for available streams.
        parameters : Union[str, List[str]], optional
            parameters to return, by default "all". Check `.parameters` attribute 
            for available parameters.
        """

        if stations != "all" and streams != "all":
            raise ValueError("Either stations or streams must be provided, not both.")

        fpath = os.path.join(self.path, 'concentrations.csv')

        df = pd.read_csv(fpath,
                         dtype={
                             'pH': np.float32, 
                             #'Site_ID': int,
                             #'Aggregated_Space': bool,
                             #'Aggregated_Time': bool,
                             #'FluxYesNo': bool
                                },
                                # converters are taking time
                        converters={'Date_start': pd.to_datetime,
                                    'Date_end': pd.to_datetime},
                        na_values={'pH': '.'})

        if stations != "all":
            stations = check_attributes(stations, self._stations, 'stations')
            df = df[df['Site_ID'].isin(stations)]
        elif streams != "all":
            streams = check_attributes(streams, self.streams, 'streams')
            sites = self.sites()
            stations = sites.loc[sites['Stream_Name'].isin(streams), 'Site_ID'].values.tolist()
            df = df[df['Site_ID'].isin(stations)]
        
        if parameters != "all":
            df = df[parameters]
    
        return df

    def sites(self):
        fpath = os.path.join(self.path, 'sites.csv')
        df = pd.read_csv(
            fpath,
            dtype={'Site_Name': str, 'Stream_Name': str, 'Basin_Region': str}
            )
        return df

    def fluxes(
            self,
            stations: Union[str, List[str]] = "all",
            ):
        fpath = os.path.join(self.path, 'fluxes.csv')
        df = pd.read_csv(fpath)

        if stations != "all":
            stations = check_attributes(stations, self._stations, 'stations')
            df = df[df['Site_ID'].isin(stations)]
        return df
    
    def sources(self):
        fpath = os.path.join(self.path, 'sources.csv')
        df = pd.read_csv(fpath)
        return df
