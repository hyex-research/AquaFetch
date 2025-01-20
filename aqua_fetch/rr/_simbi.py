
import os
from typing import List, Union, Dict

import pandas as pd

from .utils import _RainfallRunoff
from ..utils import check_attributes

from ._map import (
    catchment_area,
    gauge_latitude,
    gauge_longitude,
    slope
    )


class Simbi(_RainfallRunoff):
    """
    monthly rainfall from 1905 - 2005, daily rainfall from 1920-1940, 70 daily
    streamflow series, and 23 monthly temperature series for 24 catchments of Haiti

    `Bathelemy et al., 2023 <https://doi.org/10.23708/02POK6>`_
    `Bathelemy et al., 2024 <doi: 10.5194/essd-16-2073-2024>`_


    Examples
    ---------
    >>> from aqua_fetch import Simbi
    >>> simbi = Simbi()
    """

    url = {
        '00_SIMBI_OBSERVED_DATA.zip': "https://dataverse.ird.fr/api/access/datafile/44141",
        '01_SIMBI_CATCHMENT.zip': "https://dataverse.ird.fr/api/access/datafile/43638",
        '02_SIMBI_SIMULATED_STREAMFLOW.zip': "https://dataverse.ird.fr/api/access/datafile/43639",
        '03_SIMBI_ATTRIBUTE.zip': "https://dataverse.ird.fr/api/access/datafile/43640",
        "04_SIMBI_MAP.zip": "https://dataverse.ird.fr/api/access/datafile/43646",
        "08_SIMBI_METADATA.zip": "https://dataverse.ird.fr/api/access/datafile/43644",
        'SIMBI_README.txt': 'https://dataverse.ird.fr/api/access/datafile/43644'

    }

    def __init__(
            self,
            path: str = None,
            overwrite:bool = False,
            verbosity:int = 1,
            **kwargs
    ):
        """
        Arguments:
            path: path where the Simbi dataset has been downloaded. This path
                must contain five zip files and one xlsx file. If None, then the
                data will be downloaded.
            to_netcdf :
        """
        super().__init__(path=path, verbosity=verbosity, **kwargs)    

        self._download(overwrite=overwrite)

        self._static_features = self.static_data().columns.tolist()
        self._dynamic_features = ['q', 'pcp', 'temp']

        self.boundary_file = os.path.join(self.path, '01_SIMBI_CATCHMENT', 'Haitian_Catchment.shp')

        self._create_boundary_id_map(self.boundary_file, 1)

        self.dyn_fname = ''

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'Area': catchment_area(),
                'Lat_Cent': gauge_latitude(),
                'Slope': slope('degrees'),
                'Lon_Cent': gauge_longitude(),
        }
    
    @property
    def static_features(self):
        return self._static_features
    
    @property
    def dynamic_features(self):
        return self._dynamic_features

    @property
    def _coords_name(self)->List[str]:
        return ['Lat_Exu', 'Lon_Exu']

    @property
    def _area_name(self) ->str:
        return 'Area'  

    @property
    def start(self):
        return pd.Timestamp("19200101")

    @property
    def end(self):
        return pd.Timestamp("20051231")

    @property
    def daily_q_path(self):
        return os.path.join(self.path, '00_SIMBI_OBSERVED_DATA', '02_DAILY_STREAMFLOW')

    @property
    def daily_pcp_path(self):
        return os.path.join(self.path, '00_SIMBI_OBSERVED_DATA', '01_DAILY_RAINFALL') 
    
    @property
    def daily_pcp_20_40_path(self):
        return os.path.join(self.daily_pcp_path, '1920_1940')

    @property
    def daily_pcp_48_60_path(self):
        return os.path.join(self.daily_pcp_path, '1948_1966')

    @property
    def attributes_path(self):
        return os.path.join(self.path, '03_SIMBI_ATTRIBUTE')

    @property
    def clim_sig_path(self):
        return os.path.join(self.attributes_path, '01_CLIMATIC_SIGNATURE')   
    
    @property
    def daily_clim_sig_path(self):
        return os.path.join(self.clim_sig_path, '02_DAILY')
    
    @property
    def monthly_clim_sig_path(self):
        return os.path.join(self.clim_sig_path, '01_MONTHLY')
    
    @property
    def other_attrs_path(self):
        return os.path.join(self.attributes_path, '02_OTHERS')

    @property
    def temp_path(self):
        return os.path.join(self.path, '00_SIMBI_OBSERVED_DATA', '05_DAILY_LONG_TERM_AVERAGE_TEMPERATURE')

    def stations(self)->List[str]:
        """Returns names/IDs of 24 stations which have all (boundary, streamflow, 
        static features) data. Although there are 70 stations which have daily 
        streamflow data, only 24 of them have static + boundary data.
        """
        return self.boundary_stations()

    def all_stations(self)->List[str]:
        """
        Not all stations have all data.
        """
        return [f"0{str(i).zfill(2)}" for i in range(1, 71)]
    
    def q_stations(self)->List[str]:
        """
        Returns names/IDs of 70 stations with daily streamflow data.
        """
        return [f"0{str(i).zfill(2)}" for i in range(1, 71)]
    
    def pcp_stations(self)->List[str]:
        """
        Returns IDs of 74 stations with daily rainfall data.
        """
        s1 = [stn.split('.')[0].split('_')[1] for stn in os.listdir(self.daily_pcp_20_40_path)]
        s2 = [stn.split('.')[0].split('_')[1] for stn in os.listdir(self.daily_pcp_48_60_path)]
        return list(set(s1 + s2))
    
    def temp_stations(self)->List[str]:
        """
        Returns names/IDs of 21 stations with daily temperature data.
        """
        return [stn.split('.')[0].split('_')[1] for stn in os.listdir(self.temp_path)]
    
    def boundary_stations(self)->List[str]:
        """
        Returns names/IDs of 24 stations with boundary data.
        """
        return [stn.split('-')[1] for stn in self.bndry_id_map.keys()]
    
    def static_data_stations(self)->List[str]:
        """
        Returns names/IDs of 24 stations with static data.
        """
        return self.static_data().index.tolist()

    def daily_bsi(self)->pd.DataFrame:
        """
        Read the daily BSI values.
        """
        fpath = os.path.join(self.daily_clim_sig_path, 'baseflow_index.csv')
        df = pd.read_csv(fpath, index_col=0)
        df.index = [i.split('-')[1] for i in df.index]
        df.columns = [f"{i}_d" for i in df.columns]
        return df
    
    def daily_high_q_dur(self)->pd.DataFrame:
        """
        Read the daily high flow values.
        """
        fpath = os.path.join(self.daily_clim_sig_path, 'high_q_dur.csv')
        df = pd.read_csv(fpath, index_col=0)
        df.index = [i.split('-')[1] for i in df.index]
        df.columns = [f"{i}_d_hq_dur" for i in df.columns]
        return df
    
    def daily_high_q_freq(self)->pd.DataFrame:
        """
        Read the daily flow frequency values.
        """
        fpath = os.path.join(self.daily_clim_sig_path, 'high_q_freq.csv')
        df = pd.read_csv(fpath, index_col=0)
        df.index = [i.split('-')[1] for i in df.index]
        df.columns = [f"{i}_d_hq_freq" for i in df.columns]
        return df
    
    def daily_low_q_dur(self)->pd.DataFrame:
        """
        Read the daily low flow values.
        """
        fpath = os.path.join(self.daily_clim_sig_path, 'low_q_dur.csv')
        df = pd.read_csv(fpath, index_col=0)
        df.index = [i.split('-')[1] for i in df.index]
        df.columns = [f"{i}_d_lq_dur" for i in df.columns]
        return df
    
    def daily_low_q_freq(self)->pd.DataFrame:
        """
        Read the daily low flow frequency values.
        """
        fpath = os.path.join(self.daily_clim_sig_path, 'low_q_freq.csv')
        df = pd.read_csv(fpath, index_col=0)
        df.index = [i.split('-')[1] for i in df.index]
        df.columns = [f"{i}_d_lq_freq" for i in df.columns]
        return df
    
    def daily_q_mean(self)->pd.DataFrame:
        """
        Read the daily mean flow values.
        """
        fpath = os.path.join(self.daily_clim_sig_path, 'q_mean.csv')
        df = pd.read_csv(fpath, index_col=0)
        df.index = [i.split('-')[1] for i in df.index]
        df.columns = [f"{i}_d_mean" for i in df.columns]
        return df
    
    def daily_quantile_5(self)->pd.DataFrame:
        """
        Read the daily 5th quantile flow values.
        """
        fpath = os.path.join(self.daily_clim_sig_path, 'quantile_5.csv')
        df = pd.read_csv(fpath, index_col=0)
        df.index = [i.split('-')[1] for i in df.index]
        df.columns = [f"{i}_d_q5" for i in df.columns]
        return df

    def daily_quantile_95(self)->pd.DataFrame:
        """
        Read the daily 95th quantile flow values.
        """
        fpath = os.path.join(self.daily_clim_sig_path, 'quantile_95.csv')
        df = pd.read_csv(fpath, index_col=0)
        df.index = [i.split('-')[1] for i in df.index]
        df.columns = [f"{i}_d_q95" for i in df.columns]
        return df
    
    def daily_clim_sigs(self)->pd.DataFrame:
        """
        Read the daily climate signatures.
        """
        return pd.concat([
            self.daily_bsi(),
            self.daily_high_q_dur(),
            self.daily_high_q_freq(),
            self.daily_low_q_dur(),
            self.daily_low_q_freq(),
            self.daily_q_mean(),
            self.daily_quantile_5(),
            self.daily_quantile_95()
        ], axis=1)
    
    def monthly_aridity_runoff(self)->pd.DataFrame:
        """
        Read the monthly aridity runoff values.
        """
        fpath = os.path.join(self.monthly_clim_sig_path, 'aridity_runoff.csv')
        df = pd.read_csv(fpath, index_col=0)
        df.index = [i.split('-')[1] for i in df.index]
        df.columns = [f"{i}_mon_arid" for i in df.columns]
        return df
    
    def monthly_average(self)->pd.DataFrame:
        """
        Read the monthly average flow values.
        """
        fpath = os.path.join(self.monthly_clim_sig_path, 'average.csv')
        df = pd.read_csv(fpath, parse_dates=True, index_col=0)
        df.index = [i.split('-')[1] for i in df.index]
        df.columns = [f"{i}_mon_avg" for i in df.columns]
        return df
    
    def monthly_QMNA5(self)->pd.DataFrame:
        """
        Read the monthly QMNA5 flow values.
        """
        fpath = os.path.join(self.monthly_clim_sig_path, 'QMNA5.csv')
        df = pd.read_csv(fpath, index_col=0)
        df.index = [i.split('-')[1] for i in df.index]
        df.columns = [f"{i}_mon_QMNA5" for i in df.columns]
        return df
    
    def monthly_QMXA10(self)->pd.DataFrame:
        """
        Read the monthly QMNA10 flow values.
        """
        fpath = os.path.join(self.monthly_clim_sig_path, 'QMXA10.csv')
        df = pd.read_csv(fpath, index_col=0)
        df.index = [i.split('-')[1] for i in df.index]
        df.columns = [f"{i}_mon_QMXA10" for i in df.columns]
        return df
    
    def monthly_quantile_5(self)->pd.DataFrame:
        """
        Read the monthly 5th quantile flow values.
        """
        fpath = os.path.join(self.monthly_clim_sig_path, 'quantile_5.csv')
        df = pd.read_csv(fpath, index_col=0)
        df.index = [i.split('-')[1] for i in df.index]
        df.columns = [f"{i}_mon_q5" for i in df.columns]
        return df

    def monthly_quantile_95(self)->pd.DataFrame:
        """
        Read the monthly 95th quantile flow values.
        """
        fpath = os.path.join(self.monthly_clim_sig_path, 'quantile_95.csv')
        df = pd.read_csv(fpath, index_col=0)
        df.index = [i.split('-')[1] for i in df.index]
        df.columns = [f"{i}_mon_q95" for i in df.columns]
        return df
    
    def monthly_clim_sigs(self)->pd.DataFrame:
        """
        Read the monthly climate signatures.
        """
        return pd.concat([
            self.monthly_aridity_runoff(),
            self.monthly_average(),
            self.monthly_QMNA5(),
            self.monthly_QMXA10(),
            self.monthly_quantile_5(),
            self.monthly_quantile_95()
        ], axis=1)

    def stream_density(self)->pd.DataFrame:
        """
        Read the stream density values.
        """
        fpath = os.path.join(self.other_attrs_path, 'stream_density.csv')
        df = pd.read_csv(fpath, index_col=0)
        df.index = [i.split('-')[1] for i in df.index]
        return df
    
    def percent_lc_98(self)->pd.DataFrame:
        """
        Read the land cover percentage values.
        """
        fpath = os.path.join(self.other_attrs_path, 'Percent_land_cover_98.csv')
        df = pd.read_csv(fpath, index_col=0)
        df.index = [i.split('-')[1] for i in df.index]
        df.columns = [f"{i}_lc_98" for i in df.columns]
        return df
    
    def percent_lc_95(self)->pd.DataFrame:
        """
        Read the 95th land cover percentage values.
        """
        fpath = os.path.join(self.other_attrs_path, 'Percent_land_cover_95.csv')
        df = pd.read_csv(fpath, index_col=0)
        df.index = [i.split('-')[1] for i in df.index]
        df.columns = [f"{i}_lc_95" for i in df.columns]
        return df
    
    def percent_geology(self)->pd.DataFrame:
        """
        Read the geology percentage values.
        """
        fpath = os.path.join(self.other_attrs_path, 'Percent_geologic_class.csv')
        df = pd.read_csv(fpath, index_col=0)
        df.index = [i.split('-')[1] for i in df.index]
        df.columns = [f"{i}_geol" for i in df.columns]
        return df
    
    def topography(self)->pd.DataFrame:
        """
        Read the topography values.
        """
        fpath = os.path.join(self.other_attrs_path, 'location_and_topography.csv')
        df = pd.read_csv(fpath, index_col=0)
        df.index = [i.split('-')[1] for i in df.index]
        return df
    
    def hypsometric_curve(self)->pd.DataFrame:
        """
        Read the hyposometric curve values.
        """
        fpath = os.path.join(self.other_attrs_path, 'hypsometric_curve.csv')
        df = pd.read_csv(fpath, index_col=0)
        df.index = [i.split('-')[1] for i in df.index]
        return df
    
    def aquifer_class(self)->pd.DataFrame:
        """
        Read the aquifer class values.
        """
        fpath = os.path.join(self.other_attrs_path, 'Percent_aquifer_class.csv')
        df = pd.read_csv(fpath, index_col=0)
        df.index = [i.split('-')[1] for i in df.index]
        return df
    
    def carb_sed_magma(self)->pd.DataFrame:
        """
        Read the carbonated sedimentary and magmatic values.
        """
        fpath = os.path.join(self.other_attrs_path, 'Percent_carb_sediment_magma.csv')
        df = pd.read_csv(fpath, index_col=0)
        df.index = [i.split('-')[1] for i in df.index]
        return df
    
    def other_attributes(self)->pd.DataFrame:
        """
        Read the other attributes.
        """
        return pd.concat([
            self.stream_density(),
            self.percent_lc_98(),
            self.percent_lc_95(),
            self.percent_geology(),
            self.topography(),
            self.hypsometric_curve(),
            self.aquifer_class(),
            self.carb_sed_magma()
        ], axis=1)

    def clim_sigs(self)->pd.DataFrame:
        """
        Read the climate signatures.
        """
        return pd.concat([
            self.daily_clim_sigs(),
            self.monthly_clim_sigs()
        ], axis=1)
    
    def static_data(self)->pd.DataFrame:
        """
        Read the static data.
        """
        return pd.concat([
            self.other_attributes(),
            self.clim_sigs()
        ], axis=1)

    def fetch_static_features(
            self,
            stn_id: Union[str, list] = 'all',
            static_features: Union[str, list] = 'all'
    )->pd.DataFrame:
        """

        Returns static features of one or more stations.

        Parameters
        ----------
            stn_id : str
                name/id of station/stations of which to extract the data
            static_features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Returns
        -------
        pd.DataFrame
            a pandas dataframe of shape (stations, features)

        Examples
        ---------
        >>> from water_quality import Simbi
        >>> dataset = Simbi()
        get all static data of all stations
        >>> stns = dataset.static_data_stations()
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (24, 232)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('001')
        >>> static_data.shape
           (1, 232)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['stream_density', 'pcp', 'Forest_lc_98'])
        >>> static_data.shape
           (24, 3)
        >>> data = dataset.fetch_static_features('001', static_features=['stream_density', 'pcp', 'Forest_lc_98'])
        >>> data.shape
           (1, 3)
        """
        stations = check_attributes(stn_id, self.static_data_stations())

        df = self.static_data().copy()
        features = check_attributes(static_features, self.static_features,
                                    "static_features")
        return df.loc[stations, features]            
 
    @property
    def dyn_map(self):
        return {
        'q': 'obs_q_cms',
        'temp': 'mean_temp_C',
        'pcp': 'pcp_mm',
        }

    def _read_dynamic_from_csv(
        self,
        stations,
        dynamic_features,
        st="1919-01-01",
        en="2005-12-31"
) ->dict:
        """
        reads dynamic data of one or more catchments
        """

        attributes = check_attributes(dynamic_features, self.dynamic_features)
        stations = check_attributes(stations, self.stations())

        dyn = {}

        for stn in stations:
            df = self._read_dynamic_for_stn(stn).loc[st:en, attributes]
            dyn[stn] = df

        return dyn

    def read_stn_q(self, stn:str)->pd.DataFrame:
        """
        Read the daily streamflow data for a station.
        """
        fpath = os.path.join(self.daily_q_path, f'Q_{stn}.csv')
        df = pd.read_csv(fpath, parse_dates=True, index_col=0)
        return df
    
    def read_stn_pcp(self, stn:str)->pd.DataFrame:
        """
        Read the daily rainfall data for a station.
        """
        df1, df2 = pd.DataFrame(columns=['P']), pd.DataFrame(columns=['P'])
        fpath = os.path.join(self.daily_pcp_20_40_path, f'P_{stn}.csv')
        if os.path.exists(fpath):
            df1 = pd.read_csv(fpath, parse_dates=True, index_col=0)
            #df1.columns = ['pcp']
        
        fpath = os.path.join(self.daily_pcp_48_60_path, f'P_{stn}.csv')
        if os.path.exists(fpath):
            df2 = pd.read_csv(fpath, parse_dates=True, index_col=0)
            #df2.columns = ['pcp2']
        
        df = pd.concat([df1, df2])

        return df
    
    def read_stn_temp(self, stn:str)->pd.DataFrame:
        """
        Read the daily temperature data for a station.
        """
        df = pd.DataFrame(columns=['temp'])
        fpath = os.path.join(self.temp_path, f'P_{stn}.csv')
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, parse_dates=True, index_col=0)
        return df
    
    def _read_dynamic_for_stn(self, stn:str)->pd.DataFrame:
        """
        Read the daily streamflow, rainfall, and temperature data for a station.
        """
        df1 = self.read_stn_q(stn)
        df2 = self.read_stn_pcp(stn)
        df3 = self.read_stn_temp(stn)
        df = pd.concat([df1, df2, df3], axis=1)
        df.columns = ['q', 'pcp', 'temp']
        df.index = pd.to_datetime(df.index)
        df.columns.name = 'dynamic_features'
        df.index.name = 'time'
        return df.sort_index()
