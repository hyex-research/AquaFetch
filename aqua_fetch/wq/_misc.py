
__all__ = [
    'SanFranciscoBay', 
    'WhiteClayCreek', 
    'BuzzardsBay',
    'SeluneRiver'
           ]


import os
from typing import Union, List

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



class SeluneRiver(Datasets):
    """
    Dataset of physico-chemical variables measured at different levels, 
    for a 2021 and 2022 for characterization
    of Hyporheic zone of Selune River, Manche, Normandie, France following
    `Moustapha Ba et al., 2023 <https://doi.org/10.1016/j.dib.2022.108837>`_ .
    The data is available at `data.gouv.fr <https://doi.org/10.57745/SBXWUC>`_ .
    The following variables are available:
       
        - water level
        - temperature 
        - conductivity 
        - oxygen  
        - pressure
    """
    url = {
    "data_downstream_signy-zh.tab": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/150676",
    "data_baro_upstream-virey.tab": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/151002",
    "data_conduc_upstream-virey-zh.tab": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/150783",
    "data_mini-lomos_downstream-signy.tab": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/150678",
    "data_mini-lomos_upstream-virey.tab": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/150780",
    "data_oxygen_downstream-signy-river.tab": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/150771",
    "data_oxygen_upstream-virey-river.tab": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/150782",
    "data_oxygen_upstream-virey-zh.tab": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/150781",
    "data_station_downstream-signy-river.tab": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/150868",
    "data_station_oxygen_upstream-virey-river.tab": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/150865",
    "data_station_upstream-virey-river.tab": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/150866",
    "data_water-level_upstream-virey-river.tab": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/150779",
    "readme.txt":"https://entrepot.recherche.data.gouv.fr/api/access/datafile/151001",
    "readme1.0.txt":"https://entrepot.recherche.data.gouv.fr/api/access/datafile/156508",
    }

    def __init__(self, path=None, **kwargs):
        super().__init__(path=path, **kwargs)
        self._download()

    def data(self)->pd.DataFrame:
        """
        Return a DataFrame of the data
        """

        fpath = os.path.join(self.path, 'data_downstream_signy-zh.tab')
        downstream_signy_zh = pd.read_csv(fpath, sep='\t', index_col=0, parse_dates=True,
                        dtype={'id': str})
        downstream_signy_zh.columns = [col + '_dwnstr_signyzh' for col in downstream_signy_zh.columns]
        downstream_signy_zh.index = pd.to_datetime(downstream_signy_zh.index)
        downstream_signy_zh.index.name = 'date'

        fpath = os.path.join(self.path, 'data_baro_upstream-virey.tab')
        baro_upstream_virey = pd.read_csv(fpath, sep='\t', index_col=0, parse_dates=True,
                        dtype={'barometric': float})
        baro_upstream_virey.columns = [col + '_baro_upstr_virey' for col in baro_upstream_virey.columns]
        #assert baro_upstream_virey.shape == (31927, 1)
        baro_upstream_virey.index = pd.to_datetime(baro_upstream_virey.index)
        baro_upstream_virey.index.name = 'date'

        fpath = os.path.join(self.path, 'data_conduc_upstream-virey-zh.tab')
        cond_upstream_virey = pd.read_csv(fpath, sep='\t', index_col=0, parse_dates=True,
                        dtype={'cond_30cm': float, 't_30cm_sensor_cond': float})
        cond_upstream_virey.columns = [col + '_cond_upstream_virey' for col in cond_upstream_virey.columns]
        #assert cond_upstream_virey.shape == (31927, 2)
        cond_upstream_virey.index = pd.to_datetime(cond_upstream_virey.index)
        cond_upstream_virey.index.name = 'date'

        fpath = os.path.join(self.path, 'data_mini-lomos_downstream-signy.tab')
        mini_lomos_downstream_signy = pd.read_csv(fpath, sep='\t', index_col=0, parse_dates=True,
                        dtype=float)  # diff_press, t_river, t at 10,20,30,40 cm
        #assert mini_lomos_downstream_signy.shape == (14843, 6)
        mini_lomos_downstream_signy.columns = [col + '_mini_lomos_dwnstr_signy' for col in mini_lomos_downstream_signy.columns]
        mini_lomos_downstream_signy.index = pd.to_datetime(mini_lomos_downstream_signy.index)
        mini_lomos_downstream_signy.index.name = 'date'

        fpath = os.path.join(self.path, 'data_oxygen_downstream-signy-river.tab')
        oxy_downstream_signy = pd.read_csv(fpath, sep='\t', index_col=0, parse_dates=True,
                        dtype=float
                        )  # temp, oxy_sat, oxy_conc
        #assert oxy_downstream_signy.shape == (31947, 3)
        oxy_downstream_signy.columns = [col + '_oxy_dwnstr_signy' for col in oxy_downstream_signy.columns]
        oxy_downstream_signy.index = pd.to_datetime(oxy_downstream_signy.index)
        oxy_downstream_signy.index.name = 'date'

        fpath = os.path.join(self.path, 'data_station_downstream-signy-river.tab')
        downstream_signy = pd.read_csv(fpath, sep='\t', index_col=0, parse_dates=True,
                            dtype=float
                            ) # cond, turb, wl
        #assert downstream_signy.shape == (31947, 3)
        downstream_signy.columns = [col + '_dwnstr_signy' for col in downstream_signy.columns]
        downstream_signy.index = pd.to_datetime(downstream_signy.index, format="mixed")
        downstream_signy.index.name = 'date'

        fpath = os.path.join(self.path, 'data_station_oxygen_upstream-virey-river.tab')
        oxy_upstream_virey = pd.read_csv(fpath, sep='\t', index_col=0, parse_dates=True,
                        dtype=float
                        ) # con_oxy
        #assert oxy_upstream_virey.shape == (31927, 1)
        oxy_upstream_virey.columns = [col + '_oxy_upstr_virey_stn' for col in oxy_upstream_virey.columns]
        oxy_upstream_virey.index = pd.to_datetime(oxy_upstream_virey.index, format="mixed")
        oxy_upstream_virey.index.name = 'date'

        fpath = os.path.join(self.path, 'data_station_upstream-virey-river.tab')
        upstream_virey = pd.read_csv(fpath, sep='\t', index_col=0, parse_dates=True,
                        dtype=float
                        )  # cond, turb, wl
        #assert upstream_virey.shape == (31947, 3)
        upstream_virey.columns = [col + '_upstr_virey_stn' for col in upstream_virey.columns]
        upstream_virey.index = pd.to_datetime(upstream_virey.index, format="mixed")
        upstream_virey.index.name = 'date'

        fpath = os.path.join(self.path, 'data_water-level_upstream-virey-river.tab')
        wl_upstream_virey = pd.read_csv(fpath, sep='\t', index_col=0, parse_dates=True,
                        dtype=float
                        )  # wl, temp
        #assert wl_upstream_virey.shape == (31927, 2)
        wl_upstream_virey.columns = [col + '_upstr_virey' for col in wl_upstream_virey.columns]
        wl_upstream_virey.index = pd.to_datetime(wl_upstream_virey.index)
        wl_upstream_virey.index.name = 'date'

        fpath = os.path.join(self.path, 'data_mini-lomos_upstream-virey.tab')
        mini_lomos_upstream_virey = pd.read_csv(fpath, sep='\t', index_col=0, parse_dates=True,
                        dtype=float
                        )  # diff_press, t_river, t at 10,20,30,40 cm
        #assert mini_lomos_upstream_virey.shape == (8621, 6)
        mini_lomos_upstream_virey.columns = [col + '_mini_lomos_upstr_virey' for col in mini_lomos_upstream_virey.columns]
        mini_lomos_upstream_virey.index = pd.to_datetime(mini_lomos_upstream_virey.index)
        mini_lomos_upstream_virey.index.name = 'date'

        fpath = os.path.join(self.path, 'data_oxygen_upstream-virey-river.tab')
        oxy_upstream_virey = pd.read_csv(fpath, sep='\t', index_col=0, parse_dates=True,
                        dtype=float
                        )  # temp, oxy_sat, oxy_conc
        #assert oxy_upstream_virey.shape == (25699, 2)
        oxy_upstream_virey.columns = [col + '_oxy_upstr_virey' for col in oxy_upstream_virey.columns]
        oxy_upstream_virey.index = pd.to_datetime(oxy_upstream_virey.index)
        oxy_upstream_virey.index.name = 'date'

        fpath = os.path.join(self.path, 'data_oxygen_upstream-virey-zh.tab')
        oxy_upstream_virey_zh = pd.read_csv(fpath, sep='\t', index_col=0, parse_dates=True,
                        dtype=float
                        )  # oxy_conc and t_ at 15 and 30 cm
        #assert oxy_upstream_virey_zh.shape == (31927, 4)
        oxy_upstream_virey_zh.columns = [col + '_oxy_upstr_virey_zh' for col in oxy_upstream_virey_zh.columns]
        oxy_upstream_virey_zh.index = pd.to_datetime(oxy_upstream_virey_zh.index)
        oxy_upstream_virey_zh.index.name = 'date'

        # concatenate all dataframes

        df = pd.concat([downstream_signy_zh, baro_upstream_virey, cond_upstream_virey,
                        mini_lomos_downstream_signy, oxy_downstream_signy, downstream_signy,
                        oxy_upstream_virey, upstream_virey, wl_upstream_virey, mini_lomos_upstream_virey,
                        oxy_upstream_virey, oxy_upstream_virey_zh], axis=1)
        
        return df
