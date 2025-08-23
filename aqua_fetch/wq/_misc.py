
__all__ = [
    'SanFranciscoBay', 
    'WhiteClayCreek', 
    'BuzzardsBay',
    'SeluneRiver'
           ]


import os
from typing import Union, List, Dict

import pandas as pd

from .._datasets import Datasets
from ..utils import check_attributes


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
    >>> from aqua_fetch import SanFranciscoBay
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
    >>> from aqua_fetch import BuzzardsBay
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


class WQNaizin(Datasets):
    """
    see `Dupas et al., 2024 <https://doi.org/10.1016/j.watres.2024.122108>`_ for details on data.
    """
    url = {
        # High spatial resolution synoptic water chemistry sampling during streamflow intermittence in the Naizin catchment, northwest France
        "Discharge_Naizin.tab": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/602075",
        "metadata_files_Naizin.pdf": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/602072",
        "NaizinCatchment.zip" : "https://entrepot.recherche.data.gouv.fr/api/access/datafile/602081",
        "PARAFACLoadings_Naizin.tab": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/602080",
        "StreamWaterChemistry_Naizin.tab": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/602079",
        # High-frequency measurements of 7 major ions concentrations during discharge events in the Kervidy-Naizin and Strengbach catchments (2018-2022)
        "Brekenfeld_etal_2024_discharge_Kervidy-Naizin.tab": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/192124",
        "Brekenfeld_etal_2024_discharge_Strengbach.tab": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/192126",
        "Brekenfeld_etal_2024_IonConcentrations_Kervidy-Naizin.tab": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/192128",
        "Brekenfeld_etal_2024_IonConcentrations_Strengbach.tab":"https://entrepot.recherche.data.gouv.fr/api/access/datafile/192129",
        # High-frequency measurement of Nitrate and Dissolved Organic Carbon in the Kervidy-Naizin catchment (2010-2023)
        "DataVerseSpectro.pdf":"https://entrepot.recherche.data.gouv.fr/api/access/datafile/188240",
        "spectroDataverse.txt":"https://entrepot.recherche.data.gouv.fr/api/access/datafile/188198",
        # High-frequency measurement of Total Phosphorus and Total Reactive Phosphorus in the Kervidy-Naizin catchment (2016-2023)
        "phosphax30.txt":"https://entrepot.recherche.data.gouv.fr/api/access/datafile/188174",
        "DataVersePhosphax.pdf": "https://entrepot.recherche.data.gouv.fr/api/access/datafile/188175",
        }


class NinAfrica(Datasets):
    """
    Data of N compounts in sub-saharan African surface waters following the work of
    `Jacobs and Breuer, 2024 <https://doi.org/10.1016/j.scitotenv.2024.176611>`_ .
    The data is from 34 African countries.
    """
    url = "https://zenodo.org/records/13252417"
    
    def __init__(
            self, 
            path=None, 
            **kwargs):

        super().__init__(path=path, **kwargs)

        self._download()

    @property
    def countries(self)->List[str]:
        """
        List of countries in the dataset.
        """
        return self.data()['country'].unique().tolist()

    def data(self)->pd.DataFrame:
        """
        Return a DataFrame of the data
        """
        fpath = os.path.join(self.path, 'Dataset.xlsx')
        df = pd.read_excel(fpath, index_col=0, skiprows=1)
        # make a dictionary where columns are keys and first row values are values
        info = df.iloc[0].to_dict()

        df = df.iloc[2:].copy()

        df.rename(columns=self._col_map(), inplace=True)
        return df

    def _col_map(self)->Dict[str, str]:

        return {
            'Elevation range [m a.s.l.]': 'elev_min',
            'Unnamed: 15': 'elev_max',
            'Precipitation [mm yr-1]': 'mean_pcp_mm', 
            'Unnamed: 18': 'min_pcp_mm', 
            'Unnamed: 19': 'max_pcp_mm',
            'Air temperature [°C]': 'mean_airtemp_C',
            'Unnamed: 21': 'sd_airtemp_C',
            'Unnamed: 22': 'min_airtemp_C',
            'Unnamed: 23': 'max_airtemp_C',
            'Total N concentration [mg L-1]': 'mean_totN_mg/L', 
            'Unnamed: 42': 'sd_totN_mg/L',
            'Total N export [kg ha-1 y-1]': 'mean_totNexport_kg/ha/yr', 
            'Unnamed: 44': 'sd_totNexport_kg/ha/yr',
            'Particulate N concentration [mg L-1]': 'mean_particulateN_mg/L',
            'Unnamed: 46': 'sd_particulateN_mg/L',
            'Particulate N export [kg ha-1 y-1]': 'mean_particulateNexport_kg/ha/yr',
            'Unnamed: 48': 'sd_particulateNexport_kg/ha/yr',
            'TON concentration [mg L-1]': 'mean_TON_mg/L',  # total organic N
            'Unnamed: 50': 'sd_TON_mg/L',
            'TON export [kg ha-1 y-1]': 'mean_TONexport_kg/ha/yr',
            'Unnamed: 52': 'sd_TONexport_kg/ha/yr',
            'PON concentration [mg L-1]': 'mean_PON_mg/L',
            'Unnamed: 54': 'sd_PON_mg/L',
            'PON export [kg ha-1 y-1]': 'mean_PONexport_kg/ha/yr',
            'Unnamed: 56': 'sd_PONexport_kg/ha/yr',
            'TDN concentration [mg L-1]': 'mean_TDN_mg/L',
            'Unnamed: 58': 'sd_TDN_mg/L',
            'TDN export [kg ha-1 y-1]': 'mean_TDNexport_kg/ha/yr',
            'Unnamed: 60': 'sd_TDNexport_kg/ha/yr',
            'DON concentration [mg L-1]': 'mean_DON_mg/L',
            'Unnamed: 62': 'sd_DON_mg/L',
            'DON export [kg ha-1 y-1]': 'mean_DONexport_kg/ha/yr',
            'Unnamed: 64': 'sd_DONexport_kg/ha/yr',
            'DIN concentration [mg L-1]': 'mean_DIN_mg/L',
            'Unnamed: 66': 'sd_DIN_mg/L',
            'DIN export [kg ha-1 y-1]': 'mean_DINexport_kg/ha/yr',
            'Unnamed: 68': 'sd_DINexport_kg/ha/yr',
            'NO3-N+NO2-N concentration [mg L-1]': 'mean_NO3-N+NO2-N_mg/L',
            'Unnamed: 70': 'sd_NO3-N+NO2-N_mg/L',
            'NO3-N+NO2-N export [kg ha-1 y-1]': 'mean_NO3-N+NO2-Nexport_kg/ha/yr',
            'Unnamed: 72': 'sd_NO3-N+NO2-Nexport_kg/ha/yr',
            'NO3-N concentration [mg L-1]': 'mean_NO3-N_mg/L',
            'Unnamed: 74': 'sd_NO3-N_mg/L',
            'NO3-N export [kg ha-1 y-1]': 'mean_NO3-Nexport_kg/ha/yr',
            'Unnamed: 76': 'sd_NO3-Nexport_kg/ha/yr',
            'NO2-N concentration [mg L-1]': 'mean_NO2-N_mg/L',
            'Unnamed: 78': 'sd_NO2-N_mg/L',
            'NO2-N export [kg ha-1 y-1]': 'mean_NO2-Nexport_kg/ha/yr',
            'Unnamed: 80': 'sd_NO2-Nexport_kg/ha/yr',
            'NH4-N concentration [mg L-1]': 'mean_NH4-N_mg/L',
            'Unnamed: 82': 'sd_NH4-N_mg/L',
            'NH4-N export [kg ha-1 y-1]': 'mean_NH4-Nexport_kg/ha/yr',
            'Unnamed: 84': 'sd_NH4-Nexport_kg/ha/yr',
            'NH3-N concentration [mg L-1]': 'mean_NH3-N_mg/L',
            'Unnamed: 86': 'sd_NH3-N_mg/L',
            'NH3-N export [kg ha-1 y-1]': 'mean_NH3-Nexport_kg/ha/yr',
            'Unnamed: 88': 'sd_NH3-Nexport_kg/ha/yr',
            'N2O concentration [µg L-1]': 'mean_N2O_mg/L',
            'Unnamed: 90': 'sd_N2O_mg/L',
            'N2O export [kg ha-1 y-1]': 'mean_N2Oexport_kg/ha/yr',
            'Unnamed: 92': 'sd_N2Oexport_kg/ha/yr',
            'N2O flux [µg m-2 h-1]': 'mean_N2Oflux_µg/m2/h',
            'Unnamed: 94': 'sd_N2Oflux_µg/m2/h',
            'Water temperature [°C]': 'mean_water_temp_C',
            'Unnamed: 96': 'sd_water_temp_C',
            'EC [µS cm-1]': 'mean_EC_µS/cm',
            'Unnamed: 98': 'sd_EC_µS/cm',
            'DO [mg L-1]': 'mean_DO_mg/L',
            'Unnamed: 100': 'sd_DO_mg/L',
            'BOD [mg L-1]': 'mean_BOD_mg/L',
            'Unnamed: 102': 'sd_BOD_mg/L',
            'pH [units]': 'mean_pH',
            'Unnamed: 104': 'sd_pH',
            'TDS [mg L-1]': 'mean_TDS_mg/L',
            'Unnamed: 106': 'sd_TDS_mg/L',
            'Turbidity [NTU]': 'mean_turbidity_NTU',
            'Unnamed: 108': 'sd_turbidity_NTU',
            'TSS [mg L-1]': 'mean_TSS_mg/L',
            'Unnamed: 110': 'sd_TSS_mg/L',
            'TSS export [t ha-1 y-1]': 'mean_TSSexport_t/ha/yr',
            'Unnamed: 112': 'sd_TSSexport_t/ha/yr',
            'Chlorophyll a [µg L-1]': 'mean_Chlorophyll_a_µg/L',
            'Unnamed: 114': 'sd_Chlorophyll_a_µg/L',
            'TP concentration [mg L-1]': 'mean_TP_mg/L',
            'Unnamed: 116': 'sd_TP_mg/L',
            'TP export [kg ha-1 y-1]': 'mean_TPexport_kg/ha/yr',
            'Unnamed: 118': 'sd_TPexport_kg/ha/yr',
            'PO4-P concentration [mg L-1]': 'mean_PO4-P_mg/L',
            'Unnamed: 120': 'sd_PO4-P_mg/L',
            'PO4-P export [kg ha-1 y-1]': 'mean_PO4-Pexport_kg/ha/yr',
            'Unnamed: 122': 'sd_PO4-Pexport_kg/ha/yr',
            'DOC concentration [mg L-1]': 'mean_DOC_mg/L',
            'Unnamed: 124': 'sd_DOC_mg/L',
            'Discharge [m³ s-1]': 'mean_Discharge_m3/s',
            'Unnamed: 126': 'sd_Discharge_m3/s',
        }


class OhioTurbidity(Datasets):
    """
    Turbidity data and storm event characters (runoff, precipitation and antecedent 
    characteristics) of three urban watersheds in Cuyahoga County, Ohio, USA from 
    2018 to 2021 at 10 minutes frequency. For more details on data see 
    `Safdar et al., 2024 <https://doi.org/10.1021/acsestwater.4c00214>`_.

    """
    url = 'https://www.hydroshare.org/resource/a249f3100f924ad09600c9d3de2183b6/'

