
__all__ = ["DraixBleone", "JialingRiverChina"]

import os
from typing import List

import pandas as pd

from .utils import check_attributes
from .utils import _RainfallRunoff

from ._map import (
    observed_streamflow_cms, 
    snow_depth, 
    snow_water_equivalent,
    mean_windspeed,
    total_precipitation,
    mean_dewpoint_temperature,
    )


class DraixBleone(_RainfallRunoff):
    """
    A high-frequency, long-term data set of hydrology and sediment yield: the alpine
    badland catchments of Draix-Bléone Observatory following the work of `Klotz et al., 2023 <https://doi.org/10.5194/essd-15-4371-2023>`_.

    """
    url = {
        # "spatial": "https://doi.org/10.57745/RUQLJL",
        # "hydro_sediment": "https://doi.org/10.17180/obs.draix",
        # "climate": "https://doi.org/10.57745/BEYQFQ"
"README.txt": 
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/158242",
"DRAIXBLEONE_DRAIX_BRU_DISCH.txt":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/158223",
"DRAIXBLEONE_DRAIX_BRU_SEDTRAP.txt":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/158225",
"DRAIXBLEONE_DRAIX_BRU_SSC.txt":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/158222",
"DRAIXBLEONE_DRAIX_LAV_DISCH.txt":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/158229",
"DRAIXBLEONE_DRAIX_LAV_SEDTRAP.txt":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/158224",
"DRAIXBLEONE_DRAIX_MOU_DISCH.txt":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/158226",
"DRAIXBLEONE_DRAIX_ROU_DISCH.txt":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/158238",

"Draix_Bleone_instruments.shp":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168716",
"Draix_Bleone_instruments.prj":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168720",
"Draix_Bleone_instruments.dbf":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168715",
"Draix_Bleone_instruments.cpg":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168718",
"Draix_Bleone_instruments.shx":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168717",
"Draix_Bleone_instruments.qpj":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168719",

"Draix_Bleone_catchment_contours.shp":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168844",
"Draix_Bleone_catchment_contours.prj":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168839",
"Draix_Bleone_catchment_contours.dbf":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168843",
"Draix_Bleone_catchment_contours.cpg":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168840",
"Draix_Bleone_catchment_contours.shx":  
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168841",
"Draix_Bleone_catchment_contours.qpj":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168842",

# "DEM_Draix.tif":
#         "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168727",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._download()

    def stations(self)->List[str]:
        return ['BRU', 'LAV', 'MOU', 'ROU']

    def _read_q_stn(self, stn:str):

        fpath = os.path.join(self.path, f"DRAIXBLEONE_DRAIX_{stn}_DISCH.txt")
        stn_df = pd.read_csv(fpath, sep=';', index_col=0, parse_dates=False,
                             header=2, usecols=[0, 1, 2],
                             )
        
        stn_df.index = pd.to_datetime(stn_df.index, format="%d/%m/%Y %H:%M:%S")

        stn_df.rename(columns={'Valeur': observed_streamflow_cms()}, inplace=True)
        stn_df.columns.name = 'dynamic_features'
        stn_df.index.name = 'time'

        # convert L/s to m3/s
        # stn_df['runoff'] = stn_df['runoff'] * 1000.0

        return stn_df

    def _read_dynamic(self, stations, dynamic_features, st=None, en=None):
        stations = check_attributes(stations, self.stations(), 'stations')

        dyn = {}
        for stn in stations:
            
            dyn[stn] = self._read_q_stn(stn)
        return dyn

    def _static_coords(self, stations = 'all'):
        # from README.txt file
        coords = {'BRU': (965694, 6345789, 801, 1.07, 87), 
                  'LAV': (968818, 6343668, 850, 0.86, 32), 
                  'MOU': (968688, 6343610, 847, 0.086, 46), 
                  'ROU': (968828, 6343644, 852, 0.0013, 21)
                  }
        coords = pd.DataFrame.from_dict(coords, orient='index', 
                                        columns=['long', 'lat', 'altitude', 'area', 'veg_cover_%'])
        return coords
    
    def area(self, stations = 'all'):
        area = {'BRU': 0.5, 'LAV': 0.5, 'MOU': 0.5, 'ROU': 0.5}
        area = pd.DataFrame.from_dict(area, orient='index', columns=['area'])
        return area


class JialingRiverChina(_RainfallRunoff):
    """
    Dataset of 11 catchments in the upper, middle and lower reaches of the Jialing 
    River mainstream basin, China . For more infromation on data see `Wang et al., 2024 <https://doi.org/10.1016/j.envsoft.2024.106091>`_.
    The data consists of daily observations of weather variables and runoff from 
    2010 to 2022. 

    The dataset is available at `github link <https://github.com/AtrCheema/CVTGR-model>`_.
    """
    url = {
        'Beibei.csv': 'https://raw.githubusercontent.com/AtrCheema/CVTGR-model/refs/heads/main/Data/OriginalData/Beibei.csv',
        'Ciba.csv': 'https://raw.githubusercontent.com/AtrCheema/CVTGR-model/refs/heads/main/Data/OriginalData/Ciba.csv',
        'Dongjintuo.csv': 'https://raw.githubusercontent.com/AtrCheema/CVTGR-model/refs/heads/main/Data/OriginalData/Dongjintuo.csv',
        'Fengzhou.csv': 'https://raw.githubusercontent.com/AtrCheema/CVTGR-model/refs/heads/main/Data/OriginalData/Fengzhou.csv',
        'Guangyuan.csv': 'https://raw.githubusercontent.com/AtrCheema/CVTGR-model/refs/heads/main/Data/OriginalData/Guangyuan.csv',
        'Jinxi.csv': 'https://raw.githubusercontent.com/AtrCheema/CVTGR-model/refs/heads/main/Data/OriginalData/Jinxi.csv',
        'Langzhong.csv': 'https://raw.githubusercontent.com/AtrCheema/CVTGR-model/refs/heads/main/Data/OriginalData/Langzhong.csv',
        'Lueyang.csv': 'https://raw.githubusercontent.com/AtrCheema/CVTGR-model/refs/heads/main/Data/OriginalData/Lueyang.csv',
        'Tanjiazhuang.csv': 'https://raw.githubusercontent.com/AtrCheema/CVTGR-model/refs/heads/main/Data/OriginalData/Tanjiazhuang.csv',
        'Tingzikou.csv': 'https://raw.githubusercontent.com/AtrCheema/CVTGR-model/refs/heads/main/Data/OriginalData/Tingzikou.csv',
        'Wusheng.csv': 'https://raw.githubusercontent.com/AtrCheema/CVTGR-model/refs/heads/main/Data/OriginalData/Wusheng.csv',
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._download()    

    @property
    def dyn_map(self):
        return {
            'runoff': observed_streamflow_cms(),  # as per Table 3 in paper, this is streamflow
            'sd': snow_depth(),
            'sdwe': snow_water_equivalent(),
            'Pre': total_precipitation(),
            'dpt': mean_dewpoint_temperature(),
            'aws': mean_windspeed(),
        }

    def stations(self)->List[str]:
        return [f.split('.')[0] for f in os.listdir(self.path)]
    
    def _read_dynamic_stn(self, stn:str):

        fpath = os.path.join(self.path, f"{stn}.csv")
        stn_df = pd.read_csv(fpath, index_col=0, parse_dates=True)

        stn_df.rename(columns=self.dyn_map, inplace=True)

        stn_df.columns.name = 'dynamic_features'
        stn_df.index.name = 'time'

        return stn_df

    def _read_dynamic(self, stations, dynamic_features, st=None, en=None):
        
        stations = check_attributes(stations, self.stations(), 'stations')

        dyn = {}
        for stn in stations:
            
            stn_df = self._read_dynamic_stn(stn)

            dyn[stn] = stn_df
        
        return dyn


class HeiheRiverChina:
    """
     Data of the precipitation, stream discharge, air temperature, dissolved organic 
     carbon concentrations and dissolved inorganic carbon concentrations and DOM 
     optical indices of water at different locations in the Hulugou catchment, upper 
     reaches of Heihe River, Northeastern Tibet Plateau, China. For for on this data
     see `Liu et al., 2025 <https://doi.org/10.1016/j.envsoft.2025.106567>`_ and 
     `Hu et al., 2023 < https://doi.org/10.1029/2022WR032426>`_.
    """
    url = "https://zenodo.org/records/7067158"
