
__all__ = ["NPCTRCatchments"]

import os
from typing import Union, List, Dict

import numpy as np
import pandas as pd

from .._backend import shapefile
from ..utils import check_attributes
from .utils import _RainfallRunoff
from ._map import(
    observed_streamflow_cms,
    mean_air_temp,
    mean_rel_hum,
    mean_windspeed,
    solar_radiation,
    total_precipitation,
    snow_depth,
)

from ._map import (
    catchment_area,
    gauge_longitude,
    gauge_latitude,
    slope,
    catchment_elevation_meters,
    max_catchment_elevation_meters,
)

# SSN : Stream Sensor Node and WSN : Weather Sensor Node

class NPCTRCatchments(_RainfallRunoff):
    """
    High-resolution streamflow and weather data (2013–2019) for seven small coastal
    watersheds in the northeast Pacific coastal temperate rainforest, Canada following
    `Korver et al., 2022 <https://doi.org/10.5194/essd-14-4231-2022>`_ . The data
    include 8 dynamic features at hourly and 5 min timestep and 14 static features.
    The dynamic features include streamflow, precipitation, temperature, relative humidity,
    wind speed, wind direction, and solar radiation.

    Examples
    --------
    >>> from aqua_fetch import NPCTRCatchments
    >>> ds = NPCTRCatchments()
    >>> ds.stations
    ['626', '693', '703', '708', '819', '844', '1015']
    >>> len(ds.static_features)
    12
    >>> area = ds.area()
    >>> area.shape
    (7,)
    >>> coords = ds.stn_coords()
    >>> coords.shape
    (7, 2)
    """
    url = {
        "2013-2019_Discharge1015_5min.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_Discharge1015_5min.csv",
        "2013-2019_Discharge626_5min.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_Discharge626_5min.csv",
        "2013-2019_Discharge693_5min.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_Discharge693_5min.csv",
        "2013-2019_Discharge703_5min.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_Discharge703_5min.csv",
        "2013-2019_Discharge708_5min.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_Discharge708_5min.csv",
        "2013-2019_Discharge819_5min.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_Discharge819_5min.csv",
        "2013-2019_Discharge844_5min.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_Discharge844_5min.csv",
        "2013-2019_Discharge_Hourly.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_Discharge_Hourly.csv",
        "2013-2019_RH_5min.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_RH_5min.csv",
        "2013-2019_RH_Hourly.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_RH_Hourly.csv",
        "2013-2019_Rad_5min.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_Rad_5min.csv",
        "2013-2019_Rad_Hourly.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_Rad_Hourly.csv",
        "2013-2019_Rain_5min.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_Rain_5min.csv",
        "2013-2019_Rain_Hourly.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_Rain_Hourly.csv",
        "2013-2019_SnowDepth_Hourly.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_SnowDepth_Hourly.csv",
        "2013-2019_Ta_5min.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_Ta_5min.csv",
        "2013-2019_Ta_Hourly.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_Ta_Hourly.csv",
        "2013-2019_WindDir_5min.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_WindDir_5min.csv",
        "2013-2019_WindDir_Hourly.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_WindDir_Hourly.csv",
        "2013-2019_WindSpd_5min.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_WindSpd_5min.csv",
        "2013-2019_WindSpd_Hourly.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/2013-2019_WindSpd_Hourly.csv",
        "Data-Dictonary.csv":
            "https://media.githubusercontent.com/media/HakaiInstitute/essd2021-hydromet-datapackage/main/Data-Dictonary.csv",
        "Watersheds_lidar_derived.zip":
            "https://drive.google.com/file/d/1TzrDVZszFI0Y44WbRJCAuReYUpR8_wcx/view?usp=drive_link",
        "Focal_watersheds_lidar_derived.zip":
            "https://drive.google.com/file/d/1W6qP_9wqintvWlsVrR6FBByaelmJAIBp/view?usp=drive_link",       
        "watershedsmedata.pdf":
            "https://drive.google.com/file/d/1knjwA7EuI5L8rlzMkdsUvfN5CyxwMfL5/view?usp=drive_link",
    }

    def __init__(self, path=None,
                 timestep:str = "Hourly",
                 **kwargs):
        super().__init__(path=path, **kwargs)

        self._download()

        self.timestep = _verify_timestep(timestep)

        self.boundary_file = os.path.join(
            self.path, 
            "Focal_watersheds_lidar_derived",
            "focal_watersheds_lidar_derived.shp"
            )
        
        self._create_boundary_id_map(self.boundary_file, 0)

        self._static_features = self._get_static().columns.tolist()

    def stations(self)->List[str]:
        return ["626", "693", "703", "708", "819", "844", "1015"]

    @property
    def dynamic_features(self)->List[str]:
        return []

    @property
    def static_features(self)->List[str]:
        return self._static_features

    @property
    def q_attributes(self):
        return ["Qrate", "Qrate_min", "Qrate_max", "Qvol", "Qvol_min", "Qvol_max",
                "Qmm", "Qmm_min", "Qmm_max"]

    def stn_coords(self, stations = 'all', sensor='SSN')->pd.DataFrame:
        """
        By default uses coordinate information of Stream Sensor Nodes, assuming that
        stream sensors would be closer to the stream gauge. The values are taken
        from `Table A1 of paper <https://essd.copernicus.org/articles/14/4231/2022/essd-14-4231-2022.html#&gid=1&pid=1>`_
        """        
        coords = self.all_stn_coords()

        if sensor == 'SSN':
            coords.index = coords.index.astype(str).str.strip('SSN')
        elif sensor == 'WSN':
            coords.index = coords.index.astype(str).str.strip('WSN')

        stations = check_attributes(stations, self.stations(), 'stations')
        
        return coords.loc[stations, :]

    def all_stn_coords(self)->pd.DataFrame:
        """
        Using coordinate information of Stream Sensor Nodes, assuming that
        stream sensors would be closer to the stream gauge. The values are taken
        from `Table A1 of paper <https://essd.copernicus.org/articles/14/4231/2022/essd-14-4231-2022.html#&gid=1&pid=1>`_
        """
        stations = {
            "RefStn": {gauge_latitude: 51.6520, gauge_longitude(): -128.1287},
            "SSN626": {gauge_latitude: 51.6408, gauge_longitude(): -128.1219},
            "WSN626": {gauge_latitude: 51.6262, gauge_longitude(): -128.1018},
            "SSN693": {gauge_latitude: 51.6442, gauge_longitude(): -127.9978},
            "WSN693_703": {gauge_latitude: 51.6106, gauge_longitude(): -127.9871},
            "SSN703": {gauge_latitude: 51.6166, gauge_longitude(): -128.0257},
            "WSN703": {gauge_latitude: 51.6433, gauge_longitude(): -128.0228},
            "WSN703_708": {gauge_latitude: 51.6222, gauge_longitude(): -128.0507},
            "SSN708": {gauge_latitude: 51.6486, gauge_longitude(): -128.0684},
            "SSN819": {gauge_latitude: 51.6619, gauge_longitude(): -128.0419},
            "WSN819_1015": {gauge_latitude: 51.6827, gauge_longitude(): -128.0433},
            "SSN844": {gauge_latitude: 51.6608, gauge_longitude(): -128.0025},
            "WSN844": {gauge_latitude: 51.6614, gauge_longitude(): -127.9975},
            "SSN1015": {gauge_latitude: 51.6906, gauge_longitude(): -128.0653},
            "East Buxton": {gauge_latitude: 51.5899, gauge_longitude(): -128.9752},
            "Hecate": {gauge_latitude: 51.6826, gauge_longitude():-128.0228}
        }

        return pd.DataFrame(stations).T

    def read_hourly_q(self)->Dict[str, pd.DataFrame]:
        fpath = os.path.join(self.path, '2013-2019_Discharge_Hourly.csv')
        df = pd.read_csv(fpath, 
                         dtype={
                             'Watershed': str,
                             'Qrate': self.fp,
                             })
        df.index = pd.to_datetime(df.pop('Datetime'))

        df.rename(columns={'Qrate': observed_streamflow_cms()}, inplace=True)

        dfs = {name[3:]: grp for name, grp in df.groupby('Watershed')}

        # remove Watershed column in each dataframe
        dfs = {k: v.drop(columns='Watershed') for k, v in dfs.items()}
        return dfs
    
    def read_5min_q(self)->Dict[str, pd.DataFrame]:
        dfs = {}
        for stn in self.stations():
            fpath = os.path.join(self.path, f'2013-2019_Discharge{stn}_5min.csv')
            df = pd.read_csv(fpath, 
                             dtype={
                                 'Watershed': str,
                                 'Qrate': self.fp,
                             })
            df.index = pd.to_datetime(df.pop('Datetime'))
            df.pop("Watershed")
            df.rename(columns={'Qrate': observed_streamflow_cms()}, inplace=True)
            dfs[stn] = df
        return dfs

    def get_pcp(self, sensor="SSN"):

        df = self.read_pcp()
        if sensor == "SSN":
            df['Site'] = df['Site'].str.strip('SSN')
        elif sensor == "WSN":
            df['Site'] = df['Site'].str.strip('WSN')
        
        data = {n: g for n, g in df.groupby('Site')}

        # remove Site column
        data = {k: v.drop(columns='Site') for k, v in data.items()}
        return data

    def read_pcp(
            self,
    ):
        """

        Examples
        --------
        >>> ds = NPCTRCatchments()
        >>> pcp = ds.read_pcp()
        >>> pcp.shape
        (849472, 5)
        >>> pcp['Site'].nunique()
        15
        pcp.index[0], pcp.index[-1]
        (Timestamp('2013-09-09 21:00:00'), Timestamp('2019-10-01 00:00:00'))
        # A is accepted and E is estimated
        >>> pcp['Qflags'].unique()
        [nan, 'AV', 'EV', 'EV: Sensor malfunction due to wolf bite']
        >>> ds = NPCTRCatchments(timestep='5min')
        >>> pcp = ds.read_pcp()
        >>> pcp.shape
        (8712098, 5)
        >>> pcp['Site'].nunique()
        14
        >>> pcp.index[0], pcp.index[-1]
        (Timestamp('2013-09-05 00:00:00'), Timestamp('2019-10-01 00:00:00'))
        """
        fpath = os.path.join(self.path, f'2013-2019_Rain_{self.timestep}.csv')
        df = pd.read_csv(fpath, 
                         usecols=['Date', 'Rain', 'Qflags', 'Site', 'TotalP'],
                         dtype={
                             'Qflags': str, 
                             'Rain': self.fp, 
                             'Site': str, 'TotalP': self.fp
                             })
        df.index = pd.to_datetime(df.pop('Date'))

        df.rename(columns={'Rain': total_precipitation()}, inplace=True)

        return df

    def get_temp(self, sensor="SSN"):
        df = self.read_temp()
        if sensor == "SSN":
            df['Site'] = df['Site'].str.strip('SSN')
        elif sensor == "WSN":
            df['Site'] = df['Site'].str.strip('WSN')
        
        data = {n: g for n, g in df.groupby('Site')}

        # remove Site column
        data = {k: v.drop(columns='Site') for k, v in data.items()}

    def read_temp(
            self,
    ):
        """

        Examples
        --------
        >>> from aqua_fetch import NPCTRCatchments
        >>> ds = NPCTRCatchments()
        >>> temp = ds.read_temp()
        >>> temp.shape
        (745836, 4)
        >>> temp['Site'].nunique()
        14
        >>> temp['Qflag'].unique()
        [nan, 'AV', 'EV']
        >>> temp['Qlevel'].unique()
        [nan,  2.,  3.,  1.]
        >>> ds = NPCTRCatchments(timestep='5min')
        >>> temp_5m = ds.read_temp()
        >>> temp_5m.shape
        (8957388, 3)
        >>> temp_5m['Site'].nunique()
        14
        >>> temp_5m['Qlevel'].unique()
        [1, 2]
        >>> temp_5m['Qflags'].nunique()
        5344
        """
        fpath = os.path.join(self.path, f'2013-2019_Ta_{self.timestep}.csv')

        if self.timestep == 'Hourly':
            usecols = ['Date', 'Qlevel', 'Qflag', 'TAir', 'Site']
        else:
            usecols = ['Date', 'Qlevel', 'Qflags', 'TAir', 'Site']

        df = pd.read_csv(fpath,
                         usecols=usecols,
                         index_col='Date',
                         )
        
        df.rename(columns={'TAir': mean_air_temp()}, inplace=True)
        return df

    def read_rel_hum(
            self,
    ):
        """

        Examples
        --------
        >>> ds = NPCTRCatchments()
        >>> rh = ds.read_rel_hum()
        >>> rh.shape
        (849472, 4)
        >>> rh['Site'].nunique()
        15
        >>> rh.index[0], rh.index[-1]
        (Timestamp('2013-09-10 00:00:00'), NaT)
        ... getting data for 5min timestep
        >>> ds = NPCTRCatchments(timestep='5min')
        >>> rh_5m = ds.read_rel_hum()
        >>> rh_5m.shape
        (8281767, 3)
        >>> rh_5m['Site'].nunique()
        13
        >>> rh_5m.index[0], rh.index[-1]
        (Timestamp('2013-09-10 00:00:00'), NaT)
        >>> rh_5m['Qlevel'].unique()
        ['1', '2', '3', nan]
        """
        fpath = os.path.join(self.path, f'2013-2019_RH_{self.timestep}.csv')
        df = pd.read_csv(fpath, 
                        usecols=['Date', 'RH', 'Site', #'Qflag', 
                                 'Qlevel'],
                         dtype={
                             'RH': self.fp, 
                             'Site': str,
                             'Qflag': str,
                             'Qlevel': str
                         })
        df.index = pd.to_datetime(df.pop('Date'))

        df.rename(columns={'RH': mean_rel_hum()}, inplace=True)
 
        return df
    
    def read_wind_speed(
            self,
    ):
        """
        
        Examples
        --------
        >>> from aqua_fetch import NPCTRCatchments
        >>> ds = NPCTRCatchments()
        >>> ws = ds.read_wind_speed()
        >>> ws.shape
        (424744, 4)
        >>> ws['Site'].nunique()
        8
        >>> ws['Site'].unique()
        ['WSN626', 'SSN693', 'WSN693703', 'WSN703708', 'WSN8191015', 'BuxtonEast', 'Hecate', 'RefStn']
        >>> ws.index[0], ws.index[-1]
        (Timestamp('2013-09-09 20:00:00'), Timestamp('2019-10-01 00:00:00'))
        ... getting data for 5min timestep
        >>> ds = NPCTRCatchments(timestep='5min')
        >>> ws = ds.read_wind_speed()
        >>> ws.shape
        (5096864, 4)
        >>> ws['Site'].nunique()
        8
        """
        fpath = os.path.join(self.path, f'2013-2019_WindSpd_{self.timestep}.csv')
        df = pd.read_csv(fpath, 
                         usecols=['Date', 'Qlevel', 'Qflags', 'WindSpd', 'Site'],
                         index_col='Date',
                         dtype={
                             'WindSpd': self.fp,
                             'Site': str,
                             'Qflags': str,
                         },
                         parse_dates=True
                         )
        df.rename(columns={'WindSpd': mean_windspeed()}, inplace=True)
        return df
    
    def read_sol_rad(
            self,
    ):
        """
        Solar radiation is common among all stations so no 'Site' column is 
        present in the dataframe.

        Examples
        --------
        >>> from aqua_fetch import NPCTRCatchments
        >>> ds = NPCTRCatchments()
        >>> solrad = ds.read_sol_rad()
        >>> solrad.shape
        (53072, 3)
        >>> solrad['Qflags_SolarRad'].unique()
        ['AV', 'EV']
        >>> ds = NPCTRCatchments(timestep='5min')
        >>> solrad = ds.read_sol_rad()
        >>> solrad.shape
        (637108, 3)
        >>> solrad['SolarRadQ_flags'].nunique()
        4
        """
        fpath = os.path.join(self.path, f'2013-2019_Rad_{self.timestep}.csv')

        if self.timestep == '5min':
            usecols = ['Measurement time', 'SolarRadQ_level', 'SolarRadQ_flags', 'SolarRadAvg']
        else:
            usecols = ['Date', 'Qlevel_SolarRad', 'Qflags_SolarRad', 'SolarRadAvg']
        df = pd.read_csv(fpath, 
                         usecols=usecols,
                         index_col=usecols[0],
                         dtype={
                             'SolarRadAvg': self.fp,
                             'Qflags_SolarRad': str,
                             'SolarRadQ_flags': str,
                             'SolarRadQ_level': np.int32
                         },
                         parse_dates=True
                         )
        df.rename(columns={'SolarRadAvg': solar_radiation()}, inplace=True)
        return df

    def read_snow_depth(
            self,
    ):
        """

        Examples
        --------
        >>> from aqua_fetch import NPCTRCatchments
        >>> ds = NPCTRCatchments()
        >>> snowdepth = ds.read_snow_depth()
        >>> snowdepth.shape
        (105016, 15)
        ... get 5min timestep data
        >>> ds = NPCTRCatchments(timestep='5min')
        >>> snowdepth = ds.read_snow_depth()
        >>> snowdepth.shape
        (105016, 15)
        """
        fpath = os.path.join(self.path, f'2013-2019_SnowDepth_Hourly.csv')
        df = pd.read_csv(fpath, 
                        #usecols=['Date', 'Qlevel', 'Qflags', 'SnowDepth', 'Site'],
                         index_col='Date',
                         comment="<",
                         )
        
        df.rename(columns={'SnowDepth': snow_depth()}, inplace=True)

        return df.iloc[0:-1]
    
    def read_wind_dir(
            self,
    ):
        """

        Examples
        --------
        >>> from aqua_fetch import NPCTRCatchments
        >>> ds = NPCTRCatchments()
        >>> winddir = ds.read_wind_dir()
        >>> winddir.shape
        (371651, 4)
        >>> winddir['Site'].nunique()
        7
        >>> winddir['Site'].unique()
        ['WSN626', 'SSN693', 'WSN693703', 'WSN703708', 'WSN8191015',
       'BuxtonEast', 'RefStn']
        ... getting data for 5min timestep
        >>> ds = NPCTRCatchments(timestep='5min')
        >>> winddir = ds.read_wind_dir()
        >>> winddir.shape
        (5096864, 4)
        >>> winddir['Site'].nunique()
        8
        >>> winddir['Site'].unique()
        ['WSN626', 'SSN693', 'WSN693703', 'WSN703708', 'WSN8191015',
       'BuxtonEast', 'Hecate', 'RefStn']
        """
        fpath = os.path.join(self.path, f'2013-2019_WindDir_{self.timestep}.csv')
        df = pd.read_csv(fpath, 
                         usecols=['Date', 'Qlevel', 'Qflags', 'WindDir', 'Site'],
                         index_col='Date',
                         )
        return df

    def _read_dynamic_from_csv(
            self,
            stations,
            dynamic_features,
            st=None,
            en=None) -> dict:

        features = check_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')
        stations = check_attributes(stations, self.stations(), 'stations')

        pcp = self.read_pcp()
        temp = self.read_temp()
        rh = self.read_rel_hum()
        ws = self.read_wind_speed()
        q = self.read_5min_q()
        solrad = self.read_sol_rad()
        snow = self.read_snow_depth()
        winddir = self.read_wind_dir()

        return

    def _get_static(self)->pd.DataFrame:
        sf = shapefile.Reader(self.boundary_file)

        # get all records
        ss = []
        for rec in sf.records():
            s = pd.Series(rec.as_dict())
            ss.append(s)

        static = pd.concat(ss, axis=1).transpose().set_index('WTS_ID_A')

        sf.close()

        # drop WTS_ID_F column
        static.drop(columns='WTS_ID_F', inplace=True)

        static.rename(columns={
            'Wts_area': catchment_area(),
            'Lnd_area': 'land_area_km2',  # watershed area minus waterbody area
            'Wtb_area': 'waterbody_area_km2',
            'Lm': 'channel_length_km',  # main channel length
            'Lt': 'total_stream_network_length_km',
            'DD': 'drainage_density_km_per_km2',  #  Lt / Wts_area
            'MxFlwpth': 'maximum_flowpath_length_km',
            # Mean Vector Ruggedness Measure (Terrain Ruggedness) for a one- hectare grid
            'Avg_VRM': 'avg_vector_ruggedness', 
            'Avg_slpe': slope('%'),
            'Avg_elev': catchment_elevation_meters(),
            'Max_elev': max_catchment_elevation_meters(),
        },
            inplace=True)
        

        # convert area from hectars to km2
        static[catchment_area()] = static[catchment_area()] / 100
        static['land_area_km2'] = static['land_area_km2'] / 100
        static['waterbody_area_km2'] = static['waterbody_area_km2'] / 100
        static =  pd.concat([static.astype(self.fp), self.stn_coords()], axis=1)
        return static

    def fetch_static_features(
            self,
            stn_id: Union[str, list] = "all",
            static_features: Union[str, list] = "all"
    ) -> pd.DataFrame:
        """Fetches all or selected static features of one or more stations.

        Parameters
        ----------
            stn_id : str
                name/id of station of which to extract the data
            static_features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Returns
        -------
        pd.DataFrame
            a pandas dataframe

        Examples
        --------
        >>> from aqua_fetch import CAMELS_AUS
        >>> camels = CAMELS_AUS()
        >>> camels.fetch_static_features('224214A')
        >>> camels.static_features
        >>> camels.fetch_static_features('224214A',
        ... static_features=['area_km2', 'elev_catch_m', 'slope_%'])
        """

        stn_id = check_attributes(stn_id, self.stations(), 'stations')
        static_features = check_attributes(static_features, self.static_features, 'static_features')
        df =  self._get_static().loc[stn_id, static_features]


def _verify_timestep(timestep):
    assert timestep in ["Hourly", "5min"], f"""
    timestep must be either Hourly or 5min but it is {timestep}
    """
    return timestep