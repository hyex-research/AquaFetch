
__all__ = ['SanFranciscoBay', 'white_clay_creek', 'BuzzardsBay']

import os
from typing import Union, List

import requests
import pandas as pd

from .._datasets import Datasets
from ..utils import _unzip, check_attributes


class SanFranciscoBay(Datasets):
    """
    Time series of water quality parameters from 59 stations in San-Francisco from 1969 - 2015.
    For details on data see `Cloern et al.., 2017 <https://doi.org/10.1002/lno.10537>`_ 
    and `Schraga et al., 2017 <https://doi.org/10.1038/sdata.2017.98>`_.
    Following parameters are available:
    
        - ``Depth``
        - ``Discrete_Chlorophyll``
        - ``Ratio_DiscreteChlorophyll_Pheopigment``
        - ``Calculated_Chlorophyll``
        - ``Discrete_Oxygen``
        - ``Calculated_Oxygen``
        - ``Oxygen_Percent_Saturation``
        - ``Discrete_SPM``
        - ``Calculated_SPM``
        - ``Extinction_Coefficient``
        - ``Salinity``
        - ``Temperature``
        - ``Sigma_t``
        - ``Nitrite``
        - ``Nitrate_Nitrite``
        - ``Ammonium``
        - ``Phosphate``
        - ``Silicate``
    
    Examples
    --------
    >>> from water_datasets import SanFranciscoBay
    >>> ds = SanFranciscoBay()
    >>> data = ds.data()
    >>> data.shape
    (212472, 19)
    >>> stations = ds.stations()
    >>> len(stations)
    59
    >>> parameters = ds.parameters()
    >>> len(parameters)
    18
    ... # fetch data for station 18
    >>> stn18 = ds.fetch(stations='18')
    >>> stn18.shape
    (13944, 18)

    """
    url = {
"SanFranciscoBay.zip": "https://www.sciencebase.gov/catalog/file/get/64248ee5d34e370832fe343d"
}

    def __init__(self, path=None, **kwargs):
        super().__init__(path=path, **kwargs)
        self._download()

        self._stations = self.data()['Station_Number'].unique().tolist()
        self._parameters = self.data().columns.tolist()[1:]

    def stations(self)->List[str]:
        return self._stations
    
    def parameters(self)->List[str]:
        return self._parameters

    # def _download(self):

    #     fpath = os.path.join(self.path, 'SanFranciscoBay.zip')

    #     if not os.path.exists(fpath):
    #         # Send a GET request to the URL
    #         response = requests.get(self.url)

    #         # Check if the request was successful
    #         if response.status_code == 200:
                
    #             with open(fpath, 'wb') as file:
    #                 file.write(response.content)
    #             if self.verbosity: print("The file was downloaded successfully.")
    #         else:
    #             if self.verbosity: print("Failed to download the file. Status code:", response.status_code)
    #     _unzip(self.path)
    #     return

    def data(self)->pd.DataFrame:

        fpath = os.path.join(self.path, 'SanFranciscoBay', 'SanFranciscoBayWaterQualityData1969-2015v4.csv')

        df = pd.read_csv(fpath,
                         dtype={'Station_Number': str})

        # join Date and Time columns to create a datetime column
        # specify the format for Date/Month/YY
        df.index = pd.to_datetime(df.pop('Date') + ' ' + df.pop('Time'), format='%m/%d/%y %H:%M')
        df.pop('Julian_Date')

        return df

    def stn_data(
            self,
            stations:Union[str, List[str]]='all',
            )->pd.DataFrame:
        """
        Get station metadata.
        """
        fpath = os.path.join(self.path, 'SanFranciscoBay', 'SFBstation_locations19692015.csv')
        df = pd.read_csv(fpath, dtype={'Station_Number': str})
        df.index = df.pop('Station_Number')
        df =  df.dropna()

        stations = check_attributes(stations, self.stations(), 'stations')
        df = df.loc[stations, :]
        return df

    def fetch(
            self,
            stations:Union[str, List[str]]='all',
            parameters:Union[str, List[str]]='all',
    )->pd.DataFrame:
        """

        Parameters
        ----------
        parameters : Union[str, List[str]], optional
            The parameters to return. The default is 'all'.

        Returns
        -------
        pd.DataFrame
            DESCRIPTION.

        """
        parameters = check_attributes(parameters, self.parameters(), 'parameters')
        stations = check_attributes(stations, self.stations(), 'stations')

        data = self.data()

        data = data.loc[ data['Station_Number'].isin(stations), :]

        return data.loc[:, parameters]


def white_clay_creek(
        parameters:Union[str, List[str]]='all',
):
    """
    Time series of water quality parameters from 2001 - 2012.
        
        - chl-a
        - Dissolved Organic Carbon
    """
    url = {
        "chla": "https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/",
        "doc": "https://portal.edirepository.org/nis/dataviewer?packageid=edi.386.1&entityid=3f802081eda955b2b0b405b55b85d11c"
        }

    raise NotImplementedError


class BuzzardsBay(Datasets):
    """
    Water quality measurements in Buzzards Bay from 1992 - 2018. For more details on data
    see `Jakuba et al., <https://doi.org/10.1038/s41597-021-00856-4>`_
    data is downloaded from `MBLWHOI Library <https://darchive.mblwhoilibrary.org/entities/publication/f31123f1-2097-5742-8ce9-69010ea36460>`_
    """
    url = {
"buzzards_bay.xlsx": "https://darchive.mblwhoilibrary.org/bitstreams/87c25cf4-21b5-551c-bb7d-4604806109b4/download"}

    def __init__(self, path=None, **kwargs):
        super().__init__(path=path, **kwargs)
        self._download()

        self._stations = self.read_stations()['STN_ID'].unique().tolist()

        self._parameters = self.data().columns.tolist()

    @property
    def fpath(self):
        return os.path.join(self.path, 'buzzards_bay.xlsx')

    def stations(self)->List[str]:
        return self._stations
    
    @property
    def parameters(self)->List[str]:
        return self._parameters

    def fetch(
            self,
            parameters:Union[str, List[str]]='all',
    )->pd.DataFrame:
        """
        Fetch data for the specified parameters.
        """
        parameters = check_attributes(parameters, self.parameters(), 'parameters')
        data = self.data()
        return data.loc[:, parameters]
   
    def data(self):
        data = pd.read_excel(
            self.fpath, 
            sheet_name='all',
            dtype={
                'STN_ID': str,
                'STN_EQUIV': str,
                'SOURCE': str,
                'GEN_QC': self.fp,
                'PREC': self.fp,
                'WHTR': self.fp,
                #'TIME_QC': self.ip,
                'SAMPDEP_QC': self.fp,
                'SECCHI_M': self.fp,
                'SECC_QC': self.fp,
                #'TOTDEP_QC': self.ip,
                'TEMP_C': self.fp,
                #'TEMP_QC': self.ip
            }
            )
        
        if 'Unnamed: 0' in data.columns: 
            data.pop('Unnamed: 0')
        
        return data

    def metadata(self):

        meta = pd.read_excel(self.fpath, sheet_name='META')

        return meta

    def read_stations(self)->pd.DataFrame:
        stations = pd.read_excel(
            self.fpath, 
            sheet_name='Stations',
            skiprows=1,
            dtype={
                'STN_ID': str,
                'LATITUDE': self.fp,
                'LONGITUDE': self.fp,
                'Town': str,
                'EMBAYMENT': str,
                'WQI_Area': str,
                }
            )

        return stations
