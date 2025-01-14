
__all__ = ['SanFranciscoBay', 'WhiteClayCreek', 'BuzzardsBay']

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


class WhiteClayCreek(Datasets):
    """
    Time series of water quality parameters from White Clay Creek.
        
        - chl-a : 2001 - 2012
        - Dissolved Organic Carbon : 1977 - 2017
    """

    url = {
"WCC_CHLA_2001_1.csv": "https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/data/contents/WCC_CHLA_2001_1.csv",
"WCC_CHLA_2001.csv": "https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/data/contents/WCC_CHLA_2001.csv",
"WCC_CHLA_2002_1.csv": "https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/data/contents/WCC_CHLA_2002_1.csv",
"WCC_CHLA_2002.csv": "https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/data/contents/WCC_CHLA_2002.csv",
"WCC_CHLA_2003_1.csv": "https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/data/contents/WCC_CHLA_2003_1.csv",
"WCC_CHLA_2003.csv": "https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/data/contents/WCC_CHLA_2003.csv",
"WCC_CHLA_2004_1.csv": "https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/data/contents/WCC_CHLA_2004_1.csv",
"WCC_CHLA_2004.csv": "https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/data/contents/WCC_CHLA_2004.csv",
"WCC_CHLA_2005_1.csv": "https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/data/contents/WCC_CHLA_2005_1.csv",
"WCC_CHLA_2005.csv": "https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/data/contents/WCC_CHLA_2005.csv",
"WCC_CHLA_2006_1.csv": "https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/data/contents/WCC_CHLA_2006_1.csv",
"WCC_CHLA_2006.csv": "https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/data/contents/WCC_CHLA_2006.csv",
"WCC_CHLA_2007_1.csv": "https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/data/contents/WCC_CHLA_2007_1.csv",
"WCC_CHLA_2007.csv": "https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/data/contents/WCC_CHLA_2007.csv",
"WCC_CHLA_2008_1.csv": "https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/data/contents/WCC_CHLA_2008_1.csv",
"WCC_CHLA_2008.csv": "https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/data/contents/WCC_CHLA_2008.csv",
"WCC_CHLA_2009_1.csv": "https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/data/contents/WCC_CHLA_2009_1.csv",
"WCC_CHLA_2009.csv": "https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/data/contents/WCC_CHLA_2009.csv",
"WCC_CHLA_2010_1.csv": "https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/data/contents/WCC_CHLA_2010_1.csv",
"WCC_CHLA_2010.csv": "https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/data/contents/WCC_CHLA_2010.csv",
"WCC_CHLA_2011_1.csv": "https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/data/contents/WCC_CHLA_2011_1.csv",
"WCC_CHLA_2011.csv": "https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/data/contents/WCC_CHLA_2011.csv",
"WCC_CHLA_2012_1.csv": "https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/data/contents/WCC_CHLA_2012_1.csv",
"WCC_CHLA_2012.csv": "https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/data/contents/WCC_CHLA_2012.csv",
"doc.csv": "https://portal.edirepository.org/nis/dataviewer?packageid=edi.386.1&entityid=3f802081eda955b2b0b405b55b85d11c"
        }


    def __init__(self, path=None, **kwargs):
        super().__init__(path=path, **kwargs)
        self._download()

    def fetch(
            self,
            stations:Union[str, List[str]]='all',
            parameters:Union[str, List[str]]='all',
        ):
    
        raise NotImplementedError
    
    def doc(self)->pd.DataFrame:
        """
        Dissolved Organic Carbon data
        """
        fpath = os.path.join(self.path, 'doc.csv')
        import pandas as pd
        df = pd.read_csv(fpath, index_col=0, parse_dates=True,
                        dtype={'site': str})
        return df
    
    def chla(self)->pd.DataFrame:
        """
        Chlorophyll-a data
        """
        files = [f for f in os.listdir(self.path) if f.startswith("WCC_CHLA")]

        # start reading file when line starts with "\data"

        dfs = []
        for f in files:
            with open(os.path.join(self.path, f), 'r') as f:
                for line in f:
                    if line.startswith("\data"):
                        break
                
                # read the header
                df = pd.read_csv(f, sep=',', header=None)

            df.insert(0, 'date', pd.to_datetime(df.iloc[:, 1]))

            df.columns = ['date', 'site', 'junk',
                          'chla_chlaspec', 'chlafluor1', 'chlafluor2', 'chlafluor3',
                          'pheophytin_pheospec', 'Pheophytinfluor1', 'Pheophytinfluor2', 'Pheophytinfluor3',
                          ]
            
            df = df.drop(columns=['junk'])

            dfs.append(df)
    
        df = pd.concat(dfs, axis=0)
        return df


class BuzzardsBay(Datasets):
    """
    Water quality measurements in Buzzards Bay from 1992 - 2018. For more details on data
    see `Jakuba et al., <https://doi.org/10.1038/s41597-021-00856-4>`_
    data is downloaded from `MBLWHOI Library <https://darchive.mblwhoilibrary.org/entities/publication/f31123f1-2097-5742-8ce9-69010ea36460>`_

    Examples
    --------
    >>> from water_datasets import BuzzardsBay
    >>> ds = BuzzardsBay()
    >>> doc = ds.doc()
    >>> doc.shape
    (11092, 4)
    >>> chla = ds.chla()
    >>> chla.shape
    (1028, 10)
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
