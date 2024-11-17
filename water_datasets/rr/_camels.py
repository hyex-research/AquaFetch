
import os
import json
import glob
import shutil
import warnings
import concurrent.futures as cf
from typing import Union, List, Dict

import numpy as np
import pandas as pd

from .camels import Camels
from ..utils import get_cpus
from ..utils import check_attributes, download, _unzip

from .._backend import netCDF4, xarray as xr

# directory separator
SEP = os.sep


class CAMELS_US(Camels):
    """
    This is a dataset of 671 US catchments with 59 static features
    and 8 dyanmic features for each catchment. The dyanmic features are
    timeseries from 1980-01-01 to 2014-12-31. This class
    downloads and processes CAMELS dataset of 671 catchments named as CAMELS
    from `ucar.edu <https://ral.ucar.edu/solutions/products/camels>`_
    following `Newman et al., 2015 <https://doi.org/10.5194/hess-19-209-2015>`_ ,
    `Newman et al., 2022 <https://gdex.ucar.edu/dataset/camels.html.>`_ and
    `Addor et al., 2017 <https://hess.copernicus.org/articles/21/5293/2017/>`_.

    Examples
    --------
    >>> from water_datasets import CAMELS_US
    >>> dataset = CAMELS_US()
    >>> df = dataset.fetch(stations=1, as_dataframe=True)
    >>> df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
    >>> df.shape
    (12784, 8)
    # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
    671
    # we can get data of 10% catchments as below
    >>> data = dataset.fetch(0.1, as_dataframe=True)
    >>> data.shape
    (460488, 51)
    # the data is multi-index with ``time`` and ``dynamic_features`` as indices
    >>> data.index.names == ['time', 'dynamic_features']
     True
    # get data by station id
    >>> df = dataset.fetch(stations='11478500', as_dataframe=True).unstack()
    >>> df.shape
    (12784, 8)
    # get names of available dynamic features
    >>> dataset.dynamic_features
    # get only selected dynamic features
    >>> df = dataset.fetch(1, as_dataframe=True,
    ... dynamic_features=['prcp(mm/day)', 'srad(W/m2)', 'tmax(C)', 'tmin(C)', 'Flow']).unstack()
    >>> df.shape
    (12784, 5)
    # get names of available static features
    >>> dataset.static_features
    # get data of 10 random stations
    >>> df = dataset.fetch(10, as_dataframe=True)
    >>> df.shape
    (102272, 10)  # remember this is multi-indexed DataFrame
    # when we get both static and dynamic data, the returned data is a dictionary
    # with ``static`` and ``dyanic`` keys.
    >>> data = dataset.fetch(stations='11478500', static_features="all", as_dataframe=True)
    >>> data['static'].shape, data['dynamic'].shape
    ((1, 59), (102272, 1))

    """
    DATASETS = ['CAMELS_US']
    url = "https://ral.ucar.edu/sites/default/files/public/product-tool/camels-catchment-attributes-and-meteorology-for-large-sample-studies-dataset-downloads/basin_timeseries_v1p2_metForcing_obsFlow.zip"
    catchment_attr_url = "https://ral.ucar.edu/sites/default/files/public/product-tool/camels-catchment-attributes-and-meteorology-for-large-sample-studies-dataset-downloads/camels_attributes_v2.0.zip"

    folders = {'basin_mean_daymet': f'basin_mean_forcing{SEP}daymet',
               'basin_mean_maurer': f'basin_mean_forcing{SEP}maurer',
               'basin_mean_nldas': f'basin_mean_forcing{SEP}nldas',
               'basin_mean_v1p15_daymet': f'basin_mean_forcing{SEP}v1p15{SEP}daymet',
               'basin_mean_v1p15_nldas': f'basin_mean_forcing{SEP}v1p15{SEP}nldas',
               'elev_bands': f'elev{SEP}daymet',
               'hru': f'hru_forcing{SEP}daymet'}

    dynamic_features_ = ['dayl(s)', 'prcp(mm/day)', 'srad(W/m2)',
                        'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)', 'Flow']

    def __init__(
            self,
            data_source:str='basin_mean_daymet',
            path=None,
            **kwargs
    ):

        """
        parameters
        ----------
        path : str
            If the data is alredy downloaded then provide the complete
            path to it. If None, then the data will be downloaded.
            The data is downloaded once and therefore susbsequent
            calls to this class will not download the data unless
            ``overwrite`` is set to True.
        data_source : str
            allowed values are
                - basin_mean_daymet
                - basin_mean_maurer
                - basin_mean_nldas
                - basin_mean_v1p15_daymet
                - basin_mean_v1p15_nldas
                - elev_bands
                - hru
        """
        assert data_source in self.folders, f'allwed data sources are {self.folders.keys()}'
        self.data_source = data_source

        super().__init__(path=path, name="CAMELS_US", **kwargs)

        self.path = path

        if os.path.exists(self.path):
            print(f"dataset is already downloaded at {self.path}")
        else:
            download(self.url, os.path.join(self.camels_dir, f'CAMELS_US{SEP}CAMELS_US.zip'))
            download(self.catchment_attr_url, os.path.join(self.camels_dir, f"CAMELS_US{SEP}catchment_attrs.zip"))
            _unzip(self.path)

        self.attr_dir = os.path.join(self.path, f'catchment_attrs{SEP}camels_attributes_v2.0')
        self.dataset_dir = os.path.join(self.path, f'CAMELS_US{SEP}basin_dataset_public_v1p2')

        self._maybe_to_netcdf('camels_us_dyn')

    @property
    def dyn_map(self):
        return {
        'Flow': 'obs_q_cms',
        'tmin(C)': 'min_temp_C',
        'tmax(C)': 'max_temp_C',
        'prcp(mm/day)': 'pcp_mm',
        'swe(mm)': 'swe_mm',
        'pet_mean': 'pet_mm',
        }
    
    @property
    def dyn_factors(self)->Dict[str, float]:
        return {
            'obs_q_cms': 0.0283168,
        }

    @property
    def start(self):
        return "19800101"

    @property
    def end(self):
        return "20141231"

    @property
    def static_features(self):
        static_fpath = os.path.join(self.path, 'static_features.csv')
        if not os.path.exists(static_fpath):
            files = glob.glob(f"{os.path.join(self.path, 'catchment_attrs', 'camels_attributes_v2.0')}/*.txt")
            cols = []
            for f in files:
                _df = pd.read_csv(f, sep=';', index_col='gauge_id', nrows=1)
                cols += list(_df.columns)
        else:
            df = pd.read_csv(static_fpath, index_col='gauge_id', nrows=1)
            cols = list(df.columns)

        return cols

    @property
    def _q_name(self)->str:
        return 'obs_q_cms'
    
    @property
    def dynamic_features(self)->List[str]:
        return [self.dyn_map.get(feat, feat) for feat in self.dynamic_features_]

    @property
    def _area_name(self)->str:
        return 'area_gages2'

    @property
    def _coords_name(self)->List[str]:
        return ['gauge_lat', 'gauge_lon']

    def stations(self) -> list:
        stns = []
        for _dir in os.listdir(os.path.join(self.dataset_dir, 'usgs_streamflow')):
            cat = os.path.join(self.dataset_dir, f'usgs_streamflow{SEP}{_dir}')
            stns += [fname.split('_')[0] for fname in os.listdir(cat)]

        # remove stations for which static values are not available
        for stn in ['06775500', '06846500', '09535100']:
            stns.remove(stn)

        return stns

    def _read_dynamic_from_csv(
            self,
            stations,
            dynamic_features: Union[str, list] = 'all',
            st=None,
            en=None,
            ):
        dyn = {}
        for station in stations:

            # attributes = check_attributes(dynamic_features, self.dynamic_features)

            assert isinstance(station, str)
            df = None
            df1 = None
            dir_name = self.folders[self.data_source]
            for cat in os.listdir(os.path.join(self.dataset_dir, dir_name)):
                cat_dirs = os.listdir(os.path.join(self.dataset_dir, f'{dir_name}{SEP}{cat}'))
                stn_file = f'{station}_lump_cida_forcing_leap.txt'
                if stn_file in cat_dirs:
                    df = pd.read_csv(os.path.join(self.dataset_dir,
                                                  f'{dir_name}{SEP}{cat}{SEP}{stn_file}'),
                                     sep="\s+|;|:",
                                     skiprows=4,
                                     engine='python',
                                     names=['Year', 'Mnth', 'Day', 'Hr', 'dayl(s)', 'prcp(mm/day)', 'srad(W/m2)',
                                            'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)'],
                                     )
                    df.index = pd.to_datetime(
                        df['Year'].map(str) + '-' + df['Mnth'].map(str) + '-' + df['Day'].map(str))

            flow_dir = os.path.join(self.dataset_dir, 'usgs_streamflow')
            for cat in os.listdir(flow_dir):
                cat_dirs = os.listdir(os.path.join(flow_dir, cat))
                stn_file = f'{station}_streamflow_qc.txt'
                if stn_file in cat_dirs:
                    fpath = os.path.join(flow_dir, f'{cat}{SEP}{stn_file}')
                    q_df = pd.read_csv(fpath, 
                                        sep=r"\s+",
                                      names=['station', 'Year', 'Month', 'Day', 'Flow', 'Flag'],
                                      engine='python')
                    q_df.index = pd.to_datetime(
                        q_df['Year'].map(str) + '-' + q_df['Month'].map(str) + '-' + q_df['Day'].map(str))

            stn_df = pd.concat([
                df[['dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)']],
                q_df['Flow']],
                axis=1)
            
            stn_df.columns.name = 'dynamic_features'
            stn_df.index.name = 'time'
            stn_df.rename(columns=self.dyn_map, inplace=True)

            for col, fact in self.dyn_factors.items():
                if col in stn_df.columns:
                    stn_df[col] *= fact

            dyn[station] = stn_df

        return dyn

    def fetch_static_features(
            self,
            stn_id: Union[str, List[str]] = "all",
            features:Union[str, List[str]] = "all"
    ):
        """
        gets one or more static features of one or more stations

        Parameters
        ----------
            stn_id : str
                name/id of station of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Examples
        --------
            >>> from water_datasets import CAMELS_US
            >>> camels = CAMELS_US()
            >>> st_data = camels.fetch_static_features('11532500')
            >>> st_data.shape
               (1, 59)
            get names of available static features
            >>> camels.static_features
            get specific features of one station
            >>> static_data = camels.fetch_static_features('11528700',
            >>> features=['area_gages2', 'geol_porostiy', 'soil_conductivity', 'elev_mean'])
            >>> static_data.shape
               (1, 4)
            get names of allstations
            >>> all_stns = camels.stations()
            >>> len(all_stns)
               671
            >>> all_static_data = camels.fetch_static_features(all_stns)
            >>> all_static_data.shape
               (671, 59)
        """
        features = check_attributes(features, self.static_features)
        stn_id = check_attributes(stn_id, self.stations(), 'stations')

        static_fpath = os.path.join(self.path, 'static_features.csv')
        if not os.path.exists(static_fpath):
            files = glob.glob(f"{os.path.join(self.path, 'catchment_attrs', 'camels_attributes_v2.0')}/*.txt")
            static_df = pd.DataFrame()
            for f in files:
                # index should be read as string
                idx = pd.read_csv(f, sep=';', usecols=['gauge_id'], dtype=str)
                _df = pd.read_csv(f, sep=';', index_col='gauge_id')
                _df.index = idx['gauge_id']
                static_df = pd.concat([static_df, _df], axis=1)
            static_df.to_csv(static_fpath, index_label='gauge_id')
        else:  # index should be read as string bcs it has 0s at the start
            idx = pd.read_csv(static_fpath, usecols=['gauge_id'], dtype=str)
            static_df = pd.read_csv(static_fpath, index_col='gauge_id')
            static_df.index = idx['gauge_id']

        static_df.index = static_df.index.astype(str)

        df = static_df.loc[stn_id][features]
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df).transpose()

        return df


class CAMELS_GB(Camels):
    """
    This is a dataset of 671 catchments with 145 static features
    and 10 dyanmic features for each catchment following the work of
    `Coxon et al., 2020 <https://doi.org/10.5194/essd-12-2459-2020>`__.
    The dyanmic features are timeseries from 1970-10-01 to 2015-09-30.
    The data is downloaded from `ceh website <https://data-package.ceh.ac.uk/data/8344e4f3-d2ea-44f5-8afa-86d2987543a9.zip>`_

    Examples
    --------
    >>> from water_datasets import CAMELS_GB
    >>> dataset = CAMELS_GB()
    >>> data = dataset.fetch(0.1, as_dataframe=True)
    >>> data.shape
     (164360, 67)
    >>> data.index.names == ['time', 'dynamic_features']
    True
    >>> df = dataset.fetch(stations=1, as_dataframe=True)
    >>> df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
    >>> df.shape
    (16436, 10)
    # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
    671
    # get data by station id
    >>> df = dataset.fetch(stations='97002', as_dataframe=True).unstack()
    >>> df.shape
    (16436, 10)
    # get names of available dynamic features
    >>> dataset.dynamic_features
    # get only selected dynamic features
    >>> df = dataset.fetch(1, as_dataframe=True,
    ... dynamic_features=['windspeed', 'temperature', 'pet', 'precipitation', 'discharge_vol']).unstack()
    >>> df.shape
    (16436, 5)
    # get names of available static features
    >>> dataset.static_features
    # get data of 10 random stations
    >>> df = dataset.fetch(10, as_dataframe=True)
    >>> df.shape
    (164360, 10)  # remember this is multi-indexed DataFrame
    # when we get both static and dynamic data, the returned data is a dictionary
    # with ``static`` and ``dyanic`` keys.
    >>> data = dataset.fetch(stations='97002', static_features="all", as_dataframe=True)
    >>> data['static'].shape, data['dynamic'].shape
    ((1, 290), (164360, 1))
    """
    dynamic_features_ = ["precipitation", "pet", "temperature", "discharge_spec",
                        "discharge_vol", "peti",
                        "humidity", "shortwave_rad", "longwave_rad", "windspeed"]

    def __init__(self, path=None, **kwargs):
        """
        parameters
        ------------
        path : str
            If the data is alredy downloaded then provide the complete
            path to it. If None, then the data will be downloaded.
            The data is downloaded once and therefore susbsequent
            calls to this class will not download the data unless
            ``overwrite`` is set to True.
        """
        super().__init__(name="CAMELS_GB", path=path, **kwargs)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if not os.path.exists(os.path.join(self.path, 'camels_gb')):
            download(
                outdir=self.path,
                url = "https://data-package.ceh.ac.uk/data/8344e4f3-d2ea-44f5-8afa-86d2987543a9.zip",
                fname="camels_gb.zip"
            )
            if self.verbosity>0:
                print("unzipping the downloaded file")
            _unzip(self.path, verbosity=self.verbosity)

            # rename the folder camels_gb/8344e4f3-d2ea-44f5-8afa-86d2987543a9 to camels_gb/caemls_gb
            shutil.move(
                os.path.join(self.path, 'camels_gb', '8344e4f3-d2ea-44f5-8afa-86d2987543a9'), 
                os.path.join(self.path, 'camels_gb', 'camels_gb')
                )
        else:
            if self.verbosity>0:
                print(f"dataset is already available at {self.path}")

        self._maybe_to_netcdf('camels_gb_dyn')

        self.boundary_file = os.path.join(
            self.data_path,
            "CAMELS_GB_catchment_boundaries",
            "CAMELS_GB_catchment_boundaries.shp"
    )
        
        if not os.path.exists(self.boundary_file):
            _unzip(self.data_path)
        
        self._create_boundary_id_map(self.boundary_file, 0)

    @property
    def dyn_map(self):
        return {
        'discharge_vol': 'obs_q_cms',
        'discharge_spec': 'obs_q_mmd',
        'temperature': 'mean_temp_C',
        'humidity': 'rh_%',
        'windspeed': 'windspeed_ms',
        'precipitation': 'pcp_mm',
        'pet': 'pm_pet_mm',
        'shorwave_rad': 'dwn_sw_rad_wm2',
        }

    @property
    def data_path(self):
        return os.path.join(self.path, 'camels_gb', 'camels_gb', 'data')

    @property
    def static_attribute_categories(self) -> list:
        features = []
        for f in os.listdir(self.data_path):
            if os.path.isfile(os.path.join(self.data_path, f)) and f.endswith('csv'):
                features.append(f.split('_')[2])

        return features

    @property
    def start(self):
        return pd.Timestamp("19701001")

    @property
    def end(self):
        return pd.Timestamp("20150930")

    @property
    def static_features(self):
        files = glob.glob(f"{self.data_path}/*.csv")
        cols = []
        for f in files:
            if 'static_features.csv' not in f:
                df = pd.read_csv(f, nrows=1, index_col='gauge_id')
                cols += (list(df.columns))
        return cols

    @property
    def dynamic_features(self)->List[str]:
        return [self.dyn_map.get(feat, feat) for feat in self.dynamic_features_]
    
    def stations(self, to_exclude=None):
        # CAMELS_GB_hydromet_timeseries_StationID_number
        path = os.path.join(self.data_path, 'timeseries')
        gauge_ids = []
        for f in os.listdir(path):
            gauge_ids.append(f.split('_')[4])

        return gauge_ids

    @property
    def _mmd_feature_name(self) ->str:
        return 'obs_q_mmd'

    @property
    def _area_name(self)->str:
        return 'area'
       
    @property
    def _coords_name(self)->List[str]:
        return ['gauge_lat', 'gauge_lon']

    def _read_dynamic_from_csv(
        self,
        stations,
        features: Union[str, list] = 'all',
        st=None,
        en=None,
    ):
        """Fetches dynamic attribute/features of one or more station."""
        dyn = {}
        for stn_id in stations:
            # making one separate dataframe for one station
            path = os.path.join(self.data_path, f"timeseries")
            fname = f"CAMELS_GB_hydromet_timeseries_{stn_id}_19701001-20150930.csv"

            df = pd.read_csv(os.path.join(path, fname), index_col='date')
            df.index = pd.to_datetime(df.index)
            df.index.freq = pd.infer_freq(df.index)

            df.rename(columns=self.dyn_map, inplace=True)

            df.columns.name = 'dynamic_features'
            df.index.name = 'time'

            dyn[stn_id] = df

        return dyn

    def fetch_static_features(
            self,
            stn_id: Union[str, List[str]] = "all",
            features:Union[str, List[str]] = "all"
    ) -> pd.DataFrame:
        """
        Fetches static features of one or more stations for one or
        more category as dataframe.

        Parameters
        ----------
            stn_id : str
                name/id of station of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Examples
        ---------
        >>> from water_datasets import CAMELS_GB
        >>> dataset = CAMELS_GB(path="path/to/CAMELS_GB")
        get the names of stations
        >>> stns = dataset.stations()
        >>> len(stns)
            671
        get all static data of all stations
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (671, 145)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('85004')
        >>> static_data.shape
           (1, 145)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['area', 'elev_mean'])
        >>> static_data.shape
           (671, 2)
        """

        features = check_attributes(features, self.static_features, 'static_features')
        static_fname = 'static_features.csv'
        static_fpath = os.path.join(self.data_path, static_fname)
        if os.path.exists(static_fpath):
            static_df = pd.read_csv(static_fpath, index_col='gauge_id')
        else:
            files = glob.glob(f"{self.data_path}/*.csv")
            static_dfs = []
            for f in files:
                _df = pd.read_csv(f, index_col='gauge_id')
                static_dfs.append(_df)
            static_df = pd.concat(static_dfs, axis=1)
            static_df.to_csv(static_fpath)

        station = check_attributes(stn_id, self.stations(), 'stations')

        static_df.index = static_df.index.astype(str)

        return static_df.loc[station][features]


class CAMELS_AUS(Camels):
    """
    This is a dataset of 561 Australian catchments with 187 static features and 
    26 dyanmic features for each catchment. The dyanmic features are timeseries 
    from 1950-01-01 to 2022-03-31. This class Reads CAMELS-AUS dataset of 
    `Fowler et al., 2024 <https://doi.org/10.5194/essd-2024-263>`_ .

    If ``version`` is 1 then this class reads data following `Fowler et al., 2021 <https://doi.org/10.5194/essd-13-3847-2021>`_
    which is a dataset of 222 Australian catchments with 161 static features
    and 26 dyanmic features for each catchment. The dyanmic features are
    timeseries from 1957-01-01 to 2018-12-31.

    Examples
    --------
    >>> from water_datasets import CAMELS_AUS
    >>> dataset = CAMELS_AUS()
    >>> df = dataset.fetch(stations=1, as_dataframe=True)
    >>> df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
    >>> df.shape
       (21184, 26)
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       222
    ... # get data of 10 % of stations as dataframe
    >>> df = dataset.fetch(0.1, as_dataframe=True)
    >>> df.shape
       (550784, 22)
    ... # The returned dataframe is a multi-indexed data
    >>> df.index.names == ['time', 'dynamic_features']
        True
    ... # get data by station id
    >>> df = dataset.fetch(stations='224214A', as_dataframe=True).unstack()
    >>> df.shape
        (21184, 26)
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> data = dataset.fetch(1, as_dataframe=True,
    ...  dynamic_features=['tmax_AWAP', 'precipitation_AWAP', 'et_morton_actual_SILO', 'streamflow_MLd']).unstack()
    >>> data.shape
       (21184, 4)
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> df = dataset.fetch(10, as_dataframe=True)
    >>> df.shape  # remember this is a multiindexed dataframe
       (21184, 260)
    # when we get both static and dynamic data, the returned data is a dictionary
    # with ``static`` and ``dyanic`` keys.
    >>> data = dataset.fetch(stations='224214A', static_features="all", as_dataframe=True)
    >>> data['static'].shape, data['dynamic'].shape
    >>> ((1, 166), (550784, 1))
    """

    url = 'https://doi.pangaea.de/10.1594/PANGAEA.921850'
    url_v2 = "https://zenodo.org/records/13350616"
    urls = {1: {
        "01_id_name_metadata.zip": "https://download.pangaea.de/dataset/921850/files/",
        "02_location_boundary_area.zip": "https://download.pangaea.de/dataset/921850/files/",
        "03_streamflow.zip": "https://download.pangaea.de/dataset/921850/files/",
        "04_attributes.zip": "https://download.pangaea.de/dataset/921850/files/",
        "05_hydrometeorology.zip": "https://download.pangaea.de/dataset/921850/files/",
        "CAMELS_AUS_Attributes&Indices_MasterTable.csv": "https://download.pangaea.de/dataset/921850/files/",
        #"Units_01_TimeseriesData.pdf": "https://download.pangaea.de/dataset/921850/files/",
        #"Units_02_AttributeMasterTable.pdf": "https://download.pangaea.de/dataset/921850/files/",
    },
    2: {
        "01_id_name_metadata.zip": "https://zenodo.org/records/13350616/files/",
        "02_location_boundary_area.zip": "https://zenodo.org/records/13350616/files/",
        "03_streamflow.zip": "https://zenodo.org/records/13350616/files/",
        "04_attributes.zip": "https://zenodo.org/records/13350616/files/",
        "05_hydrometeorology.zip": "https://zenodo.org/records/13350616/files/",
        "CAMELS_AUS_Attributes&Indices_MasterTable.csv": "https://zenodo.org/records/13350616/files/",
    }
    }

    folders = {1: {
        'streamflow_MLd': f'03_streamflow{SEP}03_streamflow',
        'streamflow_MLd_inclInfilled': f'03_streamflow{SEP}03_streamflow',
        'streamflow_mmd': f'03_streamflow{SEP}03_streamflow',

        'et_morton_actual_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
        'et_morton_point_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
        'et_morton_wet_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
        'et_short_crop_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
        'et_tall_crop_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
        'evap_morton_lake_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
        'evap_pan_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
        'evap_syn_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',

        'precipitation_AWAP': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}01_precipitation_timeseries',
        'precipitation_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}01_precipitation_timeseries',
        'precipitation_var_AWAP': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}01_precipitation_timeseries',

        'solarrad_AWAP': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AWAP',
        'tmax_AWAP': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AWAP',
        'tmin_AWAP': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AWAP',
        'vprp_AWAP': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AWAP',

        'mslp_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        'radiation_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        'rh_tmax_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        'rh_tmin_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        'tmax_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        'tmin_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        'vp_deficit_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        'vp_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
    },
    2: {
        'streamflow_MLd': f'03_streamflow{SEP}03_streamflow',
        'streamflow_MLd_inclInfilled': f'03_streamflow{SEP}03_streamflow',
        'streamflow_mmd': f'03_streamflow{SEP}03_streamflow',

        'et_morton_actual_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
        'et_morton_point_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
        'et_morton_wet_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
        'et_short_crop_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
        'et_tall_crop_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
        'evap_morton_lake_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
        'evap_pan_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
        'evap_syn_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',

        'precipitation_AGCD': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}01_precipitation_timeseries',
        'precipitation_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}01_precipitation_timeseries',
        'precipitation_var_AGCD': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}01_precipitation_timeseries',

        #'solarrad_AWAP': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AGCD{SEP}solarrad_AWAP',
        'tmax_AGCD': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AGCD',
        'tmin_AGCD': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AGCD',
        'vapourpres_h09_AGCD': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AGCD',
        'vapourpres_h15_AGCD': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AGCD',

        'mslp_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        'radiation_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        'rh_tmax_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        'rh_tmin_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        'tmax_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        'tmin_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        'vp_deficit_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        'vp_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',        
    }
    }

    def __init__(
            self,
            path: str = None,
            version:int = 2,
            to_netcdf:bool = True,
            overwrite:bool = False,
            verbosity:int = 1,
            **kwargs
    ):
        """
        Arguments:
            path: path where the CAMELS_AUS dataset has been downloaded. This path
                must contain five zip files and one xlsx file. If None, then the
                data will be downloaded.
            version: version of the dataset to download. Allowed values are 1 and 2.
            to_netcdf :
        """
        # if path is not None:
        #     assert isinstance(path, (str, os.PathLike)), f'path must be string like but it is "{path}" of type {path.__class__.__name__}'
        #     if not os.path.exists(path) or len(os.listdir(path)) < 2:
        #         raise FileNotFoundError(f"The path {path} does not exist")

        self.version = version

        super().__init__(path=path, verbosity=verbosity, **kwargs)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        for _file, url in self.urls[version].items():
            fpath = os.path.join(self.path, _file)
            
            if os.path.exists(fpath) and overwrite:
                os.remove(fpath)
                if verbosity > 0: print(f"Re-downloading {_file} from {url+ _file} at {fpath}")                
                download(url + _file, outdir=self.path, fname=_file)

            elif not os.path.exists(fpath):
                if verbosity > 0:
                    print(f"Downloading {_file} from {url+ _file} at {fpath}")
                download(url + _file, outdir=self.path, fname=_file)
            elif verbosity > 0:
                print(f"{_file} already exists at {self.path}")
            
        # maybe the .zip file has been downloaded previously but not unzipped
        _unzip(self.path, verbosity=verbosity, overwrite=overwrite)

        if netCDF4 is None:
            to_netcdf = False

        if to_netcdf:
            self._maybe_to_netcdf('camels_aus_dyn')

        self.boundary_file = os.path.join(
        self.path,
        "02_location_boundary_area",
        "02_location_boundary_area",
        "shp",
        "CAMELS_AUS_Boundaries_adopted.shp" if self.version==1 else "CAMELS_AUS_v2_Boundaries_adopted.shp"
    )
        self._create_boundary_id_map(self.boundary_file, 0)

    @property
    def dyn_map(self):
        return {
        'streamflow_MLd': 'obs_q_cms',
        'streamflow_mmd': 'obs_q_mmd',
        'tmin_SILO': 'min_temp_C',
        'tmax_SILO': 'max_temp_C',
        }

    @property
    def dyn_factors(self):
        return {
        'obs_q_cms': 0.01157,
        }

    @property
    def start(self):
        return "19500101"

    @property
    def end(self):
        return "20181231" if self.version==1 else "20220331"

    @property
    def location(self):
        return "Australia"

    def stations(self, as_list=True) -> list:
        fname = os.path.join(self.path, f"01_id_name_metadata{SEP}01_id_name_metadata{SEP}id_name_metadata.csv")
        df = pd.read_csv(fname)
        if as_list:
            return df['station_id'].to_list()
        else:
            return df

    @property
    def static_attribute_categories(self):
        features = []
        path = os.path.join(self.path, f'04_attributes{SEP}04_attributes')
        for f in os.listdir(path):
            if os.path.isfile(os.path.join(path, f)) and f.endswith('csv'):
                f = str(f.split('.csv')[0])
                features.append(''.join(f.split('_')[2:]))
        return features

    @property
    def static_features(self) -> list:
        static_fpath = os.path.join(self.path, 'CAMELS_AUS_Attributes&Indices_MasterTable.csv')

        df = pd.read_csv(static_fpath, index_col='station_id', nrows=1)
        cols = list(df.columns)

        return cols

    @property
    def dynamic_features(self) -> list:
        return [self.dyn_map.get(feat, feat) for feat in list(self.folders[self.version].keys())]

    def q_mmd(
            self,
            stations: Union[str, List[str]] = None
    )->pd.DataFrame:
        """
        returns streamflow in the units of milimeter per day. This is obtained
        by diving q_cms/area

        parameters
        ----------
        stations : str/list
            name/names of stations. Default is None, which will return
            area of all stations

        Returns
        --------
        pd.DataFrame
            a pandas DataFrame whose indices are time-steps and columns
            are catchment/station ids.

        """
        stations = check_attributes(stations, self.stations())
        q = self.fetch_stations_features(stations,
                                           dynamic_features='obs_q_cms',
                                           as_dataframe=True)
        q.index = q.index.get_level_values(0)
        q = q * 0.01157  # mega liter per day to cms
        area_m2 = self.area(stations) * 1e6  # area in m2
        q = (q / area_m2) * 86400  # to m/day
        return q * 1e3  # to mm/day

    @property
    def _area_name(self)->str:
        return 'catchment_area'
    
    @property
    def _coords_name(self)->List[str]:
        return ['lat_outlet', 'long_outlet']

    def _read_static(self, stations, features,
                     st=None, en=None):

        features = check_attributes(features, self.static_features)
        static_fname = 'CAMELS_AUS_Attributes&Indices_MasterTable.csv'
        static_fpath = os.path.join(self.path, static_fname)
        static_df = pd.read_csv(static_fpath, index_col='station_id')

        static_df.index = static_df.index.astype(str)
        df = static_df.loc[stations][features]
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df).transpose()

        return self.to_ts(df, st, en)

    def _read_dynamic_from_csv(self, stations, dynamic_features, **kwargs):

        dyn_attrs = {}
        dyn = {}
        for _attr in list(self.folders[self.version].keys()):
            _path = os.path.join(self.path, f'{self.folders[self.version][_attr]}{SEP}{_attr}.csv')
            attr_df = pd.read_csv(_path, na_values=['-99.99'])
            attr_df.index = pd.to_datetime(attr_df[['year', 'month', 'day']])
            [attr_df.pop(col) for col in ['year', 'month', 'day']]

            dyn_attrs[_attr] = attr_df

        # making one separate dataframe for one station
        for stn in stations:
            stn_df = pd.DataFrame()
            for attr, attr_df in dyn_attrs.items():
                #if attr in dynamic_features:
                stn_df[attr] = attr_df[stn]
            
            stn_df.rename(columns=self.dyn_map, inplace=True)

            for col, fact in self.dyn_factors.items():
                if col in stn_df.columns:
                    stn_df[col] = stn_df[col] * fact

            stn_df.index.name = 'time'
            stn_df.columns.name = 'dynamic_features'
            dyn[stn] = stn_df.loc[:, dynamic_features]

        return dyn

    def fetch_static_features(
            self,
            stn_id: Union[str, List[str]] = "all",
            features:Union[str, List[str]] = "all",
    ) -> pd.DataFrame:
        """Fetches static features of one or more stations as dataframe.

        Parameters
        ----------
            stn_id : str
                name/id of station of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Examples
        ---------
        >>> from water_datasets import CAMELS_AUS
        >>> dataset = CAMELS_AUS()
        get the names of stations
        >>> stns = dataset.stations()
        >>> len(stns)
            222
        get all static data of all stations
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (222, 161)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('305202')
        >>> static_data.shape
           (1, 161)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['catchment_di', 'elev_mean'])
        >>> static_data.shape
           (222, 2)
        """

        stn_id = check_attributes(stn_id, self.stations(), 'stations')

        return self._read_static(stn_id, features)


class CAMELS_CL(Camels):
    """
    This is a dataset of 516 Chilean catchments with
    104 static features and 12 dyanmic features for each catchment.
    The dyanmic features are timeseries from 1913-02-15 to 2018-03-09.
    This class downloads and processes CAMELS dataset of Chile following the work of
    `Alvarez-Garreton et al., 2018 <https://doi.org/10.5194/hess-22-5817-2018>`_ .

    Examples
    ---------
    >>> from water_datasets import CAMELS_CL
    >>> dataset = CAMELS_CL()
    >>> df = dataset.fetch(stations=1, as_dataframe=True)
    >>> df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
    >>> df.shape
        (38374, 12)
    # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
    516
    # we can get data of 10% catchments as below
    >>> data = dataset.fetch(0.1, as_dataframe=True)
    >>> data.shape
    (460488, 51)
    # the data is multi-index with ``time`` and ``dynamic_features`` as indices
    >>> df.index.names == ['time', 'dynamic_features']
     True
    # get data by station id
    >>> df = dataset.fetch(stations='8350001', as_dataframe=True).unstack()
    >>> df.shape
    (38374, 12)
    # get names of available dynamic features
    >>> dataset.dynamic_features
    # get only selected dynamic features
    >>> df = dataset.fetch(1, as_dataframe=True,
    ... dynamic_features=['pet_hargreaves', 'precip_tmpa', 'tmean_cr2met', 'streamflow_m3s']).unstack()
    >>> df.shape
    (38374, 4)
    # get names of available static features
    >>> dataset.static_features
    # get data of 10 random stations
    >>> df = dataset.fetch(10, as_dataframe=True)
    >>> df.shape
    (460488, 10)
    # when we get both static and dynamic data, the returned data is a dictionary
    # with ``static`` and ``dyanic`` keys.
    >>> data = dataset.fetch(stations='8350001', static_features="all", as_dataframe=True)
    >>> data['static'].shape, data['dynamic'].shape
    >>> ((1, 104), (460488, 1))
    """

    urls = {
        "1_CAMELScl_attributes.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "2_CAMELScl_streamflow_m3s.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "3_CAMELScl_streamflow_mm.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "4_CAMELScl_precip_cr2met.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "5_CAMELScl_precip_chirps.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "6_CAMELScl_precip_mswep.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "7_CAMELScl_precip_tmpa.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "8_CAMELScl_tmin_cr2met.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "9_CAMELScl_tmax_cr2met.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "10_CAMELScl_tmean_cr2met.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "11_CAMELScl_pet_8d_modis.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "12_CAMELScl_pet_hargreaves.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "13_CAMELScl_swe.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "14_CAMELScl_catch_hierarchy.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
        "CAMELScl_catchment_boundaries.zip": "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/",
    }

    dynamic_features_ = ['streamflow_m3s', 'streamflow_mm',
                        'precip_cr2met', 'precip_chirps', 'precip_mswep', 'precip_tmpa',
                        'tmin_cr2met', 'tmax_cr2met', 'tmean_cr2met',
                        'pet_8d_modis', 'pet_hargreaves',
                        'swe'
                        ]

    def __init__(self,
                 path: str = None,
                 **kwargs,
                 ):
        """
        Arguments:
            path: path where the CAMELS-CL dataset has been downloaded. This path must
                  contain five zip files and one xlsx file.
        """

        super().__init__(path=path, **kwargs)
        self.path = path

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        for _file, url in self.urls.items():
            fpath = os.path.join(self.path, _file)
            if not os.path.exists(fpath):
                download(url + _file, fpath)
                _unzip(self.path)

        self.dyn_fname = os.path.join(self.path, 'camels_cl_dyn.nc')
        self._maybe_to_netcdf('camels_cl_dyn')

        self.boundary_file = os.path.join(
        path,
        "CAMELS_CL",
        "CAMELScl_catchment_boundaries",
        "CAMELScl_catchment_boundaries",
        "catchments_camels_cl_v1_3.shp"
    )
        
        self._create_boundary_id_map(self.boundary_file, 0)

    @property
    def dyn_map(self):
        return {
        'streamflow_m3s': 'obs_q_cms',
        'streamflow_mm': 'obs_q_mmd',
        'tmin_cr2met': 'min_temp_C',
        'tmax_cr2met': 'max_temp_C',
        'tmean_cr2met': 'mean_temp_C',
        'precip_mswep': 'pcp_mm',
        }

    @property
    def _all_dirs(self):
        """All the folders in the dataset_directory"""
        return [f for f in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, f))]

    @property
    def start(self):
        return "19130215"

    @property
    def end(self):
        return "20180309"

    @property
    def location(self):
        return "Chile"

    @property
    def static_features(self) -> list:
        path = os.path.join(self.path, f"1_CAMELScl_attributes{SEP}1_CAMELScl_attributes.txt")
        df = pd.read_csv(path, sep='\t', index_col='gauge_id')
        return df.index.to_list()

    @property
    def dynamic_features(self) -> List[str]:
        return [self.dyn_map.get(feat, feat) for feat in self.dynamic_features_]

    @property
    def _mmd_feature_name(self) ->str:
        return 'obs_q_mmd'

    @property
    def _area_name(self)->str:
        return 'area'

    def stn_coords(
            self,
            stations:Union[str, List[str]] = "all"
    ) ->pd.DataFrame:
        """
        returns coordinates of stations as DataFrame
        with ``long`` and ``lat`` as columns.

        Parameters
        ----------
        stations :
            name/names of stations. If not given, coordinates
            of all stations will be returned.

        Returns
        -------
        coords :
            pandas DataFrame with ``long`` and ``lat`` columns.
            The length of dataframe will be equal to number of stations
            wholse coordinates are to be fetched.

        Examples
        --------
        >>> dataset = CAMELS_CL()
        >>> dataset.stn_coords() # returns coordinates of all stations
        >>> dataset.stn_coords('12872001')  # returns coordinates of station whose id is 912101A
        >>> dataset.stn_coords(['12872001', '12876004'])  # returns coordinates of two stations
        """
        fpath = os.path.join(self.path,
                             '1_CAMELScl_attributes',
                             '1_CAMELScl_attributes.txt')
        df = pd.read_csv(fpath, sep='\t', index_col='gauge_id')
        df = df.loc[['gauge_lat', 'gauge_lon'], :].transpose()
        df.columns = ['lat', 'long']
        stations = check_attributes(stations, self.stations(), 'stations')
        df.index  = [index.strip() for index in df.index]
        return df.loc[stations, :]

    def stations(self) -> list:
        """
        Tells all station ids for which a data of a specific attribute is available.
        """
        stn_fname = os.path.join(self.path, 'stations.json')
        if not os.path.exists(stn_fname):
            _stations = {}
            for dyn_attr in self.dynamic_features_:
                for _dir in self._all_dirs:
                    if dyn_attr in _dir:
                        fname = os.path.join(self.path, f"{_dir}{SEP}{_dir}.txt")
                        df = pd.read_csv(fname, sep='\t', nrows=2, index_col='gauge_id')
                        _stations[dyn_attr] = list(df.columns)

            stns = list(set.intersection(*map(set, list(_stations.values()))))
            with open(stn_fname, 'w') as fp:
                json.dump(stns, fp)
        else:
            with open(stn_fname, 'r') as fp:
                stns = json.load(fp)
        return stns

    def _read_dynamic_from_csv(self, stations, dynamic_features, st=None, en=None):

        dyn = {}
        st, en = self._check_length(st, en)

        assert all(stn in self.stations() for stn in stations)

        dynamic_features = check_attributes(dynamic_features, self.dynamic_features)

        # reading all dynnamic features
        dyn_attrs = {}
        for attr in self.dynamic_features_:
            fname = [f for f in self._all_dirs if '_' + attr in f][0]
            fname = os.path.join(self.path, f'{fname}{SEP}{fname}.txt')
            df = pd.read_csv(fname, sep='\t', index_col=['gauge_id'], na_values=" ")
            df.index = pd.to_datetime(df.index)

            dyn_attrs[attr] = df[st:en]

        # making one separate dataframe for one station
        for stn in stations:
            stn_df = pd.DataFrame()
            for attr, attr_df in dyn_attrs.items():
                #if attr in dynamic_features:
                stn_df[attr] = attr_df[stn]

            stn_df.rename(columns=self.dyn_map, inplace=True)
            stn_df.index.name = 'time'
            stn_df.columns.name = 'dynamic_features'
            dyn[stn] = stn_df[st:en]

        return dyn

    def _read_static(self, stations: list, features: list) -> pd.DataFrame:
        # overwritten for speed
        path = os.path.join(self.path, f"1_CAMELScl_attributes{SEP}1_CAMELScl_attributes.txt")
        _df = pd.read_csv(path, sep='\t', index_col='gauge_id')

        stns_df = []
        for stn in stations:
            df = pd.DataFrame()
            if stn in _df:
                df[stn] = _df[stn]
            elif ' ' + stn in _df:
                df[stn] = _df[' ' + stn]

            stns_df.append(df.transpose()[features])

        stns_df = pd.concat(stns_df)
        return stns_df

    def fetch_static_features(
            self,
            stn_id: Union[str, List[str]] = "all",
            features:Union[str, List[str]] = "all"
    ):
        """
        Returns static features of one or more stations.

        Parameters
        ----------
            stn_id : str
                name/id of station of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Examples
        ---------
        >>> from water_datasets import CAMELS_CL
        >>> dataset = CAMELS_CL()
        get the names of stations
        >>> stns = dataset.stations()
        >>> len(stns)
            516
        get all static data of all stations
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (516, 104)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('11315001')
        >>> static_data.shape
           (1, 104)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['slope_mean', 'area'])
        >>> static_data.shape
           (516, 2)
        >>> data = dataset.fetch_static_features('2110002', features=['slope_mean', 'area'])
        >>> data.shape
           (1, 2)
        """
        features = check_attributes(features, self.static_features)

        stn_id = check_attributes(stn_id, self.stations(), 'stations')

        return self._read_static(stn_id, features)


class CAMELS_CH(Camels):
    """
    Data of 331 Swiss catchments from 
    `Hoege et al., 2023 <https://doi.org/10.5194/essd-15-5755-2023>`_ .
    The dataset consists of 209 static catchment features and 9 dynamic features.
    The dynamic features span from 19810101 to 20201231 with daily timestep.
    For daily (``D``) ``timestep``, only streamflow is available for 170 swiss catchments.
    The hourly (``H``) streamflow data is obtained from `Kauzlaric et al., 2023 <https://zenodo.org/records/7691294>`_ .

    Examples
    ---------
    >>> from water_datasets import CAMELS_CH
    >>> dataset = CAMELS_CH()
    >>> data = dataset.fetch(0.1, as_dataframe=True)
    >>> data.shape
    (128560, 10)
    >>> data.index.names == ['time', 'dynamic_features']
    True
    >>> df = dataset.fetch(stations=1, as_dataframe=True)
    >>> df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
    >>> df.shape
    (8036, 9)
    # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
    331
    # get data by station id
    >>> df = dataset.fetch(stations='2004', as_dataframe=True).unstack()
    >>> df.shape
    (8036, 9)
    # get names of available dynamic features
    >>> dataset.dynamic_features
    # get only selected dynamic features
    >>> df = dataset.fetch(1, as_dataframe=True, dynamic_features=['precipitation(mm/d)', 'temperature_mean(°C)', 'discharge_vol(m3/s)']).unstack()
    >>> df.shape
    (8036, 3)
    # get names of available static features
    >>> dataset.static_features
    # get data of 10 random stations
    >>> df = dataset.fetch(10, as_dataframe=True)
    >>> df.shape
    (72324, 10)  # remember this is multi-indexed DataFrame
    # when we get both static and dynamic data, the returned data is a dictionary
    # with ``static`` and ``dyanic`` keys.
    >>> data = dataset.fetch(stations='2004', static_features="all", as_dataframe=True)
    >>> data['static'].shape, data['dynamic'].shape
    ((1, 209), (72324, 1))

    """
    url = {
        'camels_ch.zip': "https://zenodo.org/record/7957061",
        'DischargeDBHydroCH.zip': 'https://zenodo.org/records/7691294'
        }

    def __init__(
            self,
            path=None,
            overwrite:bool = False,
            to_netcdf: bool = True,
            timestep:str = 'D',
            **kwargs
    ):
        """

        Parameters
        ----------
        path : str
            If the data is alredy downloaded then provide the complete
            path to it. If None, then the data will be downloaded.
            The data is downloaded once and therefore susbsequent
            calls to this class will not download the data unless
            ``overwrite`` is set to True.
        overwrite : bool
            If the data is already down then you can set it to True,
            to make a fresh download.
        to_netcdf : bool
            whether to convert all the data into one netcdf file or not.
            This will fasten repeated calls to fetch etc. but will
            require netcdf5 package as well as xarry.
        """
        super().__init__(path=path, **kwargs)

        self.timestep = timestep

        if timestep == 'D' and 'DischargeDBHydroCH.zip' in self.url:
            self.url.pop('DischargeDBHydroCH.zip')

        self._download(overwrite=overwrite)

        self._dynamic_features = self._read_dynamic_for_stn(self.stations()[0]).columns.tolist()

        if to_netcdf:
            self._maybe_to_netcdf('camels_ch_dyn')
        
        self.boundary_file = os.path.join(
        path,
        'CAMELS_CH',
        'camels_ch',
        'camels_ch',
        'catchment_delineations',
        'CAMELS_CH_catchments.shp'
    )
        self._create_boundary_id_map(self.boundary_file, 0)

    @property
    def dyn_map(self):
        return {
        'discharge_vol(m3/s)': 'obs_q_cms',
        #'discharge_vol(m3/s)': 'sim_q_cms',
        'discharge_spec(mm/d)': 'obs_q_mmd',
        'temperature_min(°C)': 'min_temp_C',
        'temperature_max(°C)': 'max_temp_C',
        'temperature_mean(°C)': 'mean_temp_C',
        'precipitation(mm/d)': 'pcp_mm',
        'swe(mm)': 'swe_mm',
        }

    @property
    def camels_path(self)->Union[str, os.PathLike]:
        return os.path.join(self.path, 'camels_ch', 'camels_ch')

    @property
    def static_path(self)->Union[str, os.PathLike]:
        return os.path.join(self.camels_path, 'static_attributes')

    @property
    def dynamic_path(self)->Union[str, os.PathLike]:
        return os.path.join(self.camels_path, 'time_series', 'observation_based')

    @property
    def glacier_attr_path(self)->Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_glacier_attributes.csv')

    @property
    def clim_attr_path(self)->Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_climate_attributes_obs.csv')

    @property
    def geol_attr_path(self)->Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_geology_attributes.csv')

    @property
    def supp_geol_attr_path(self)->Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_geology_attributes_supplement.csv')

    @property
    def hum_inf_attr_path(self)->Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_humaninfluence_attributes.csv')

    @property
    def hydrogeol_attr_path(self)->Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_hydrogeology_attributes.csv')

    @property
    def hydrol_attr_path(self)->Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_hydrology_attributes_obs.csv')

    @property
    def lc_attr_path(self)->Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_landcover_attributes.csv')

    @property
    def soil_attr_path(self)->Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_soil_attributes.csv')

    @property
    def topo_attr_path(self)->Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_topographic_attributes.csv')

    @property
    def static_features(self):
        return self.fetch_static_features().columns.tolist()

    @property
    def dynamic_features(self) -> List[str]:
        return self._dynamic_features

    def all_hourly_stations(self)->List[str]:
        """Names of all stations which have hourly data"""
        return pd.read_excel(
            os.path.join(self.path, 'Inventory_discharge_hydroCH.xlsx'), dtype={'ID': str}
            )['ID'].values.tolist()

    def hourly_stations(self)->List[str]:
        """
        IDs of those stations which have hourly data and which are also part of 
        CAMELS-CH dataset
        """
        return [stn for stn in self.all_hourly_stations() if stn in self.stations()]

    @property
    def start(self):  # start of data
        return pd.Timestamp('1981-01-01')

    @property
    def end(self):  # end of data
        return pd.Timestamp('2020-12-31')

    def stations(self)->List[str]:
        """Returns station ids for catchments"""
        stns =  pd.read_csv(
            self.glacier_attr_path,
            sep=';',
            skiprows=1
        )['gauge_id'].values.tolist()
        return [str(stn) for stn in stns]

    @property
    def foen_path(self)->Union[str, os.PathLike]:
        return os.path.join(self.path, 'DischargeDBHydroCH', 'DischargeDBHydroCH', 'CH', 'FOEN')

    def foen_stations(self)->List[str]:
        """Returns all the stations in the FOEN folder"""
        return os.listdir(self.foen_path)

    def read_hourly_q_ch(self, stn:str)->pd.DataFrame:
        stn = f"Q_{stn}_hourly.asc"
        fname = [fname for fname in self.foen_stations() if stn in fname][0]
        fpath = os.path.join(self.foen_path, fname)

        q = pd.read_csv(fpath,
                    sep="\t",
                    parse_dates=[['YYYY', 'MM', 'DD', 'HH']],
                    index_col='YYYY_MM_DD_HH',
                    )
        q.index = pd.to_datetime(q.index)
        q.columns = ['q_cms']
        q.index.name = "time"
        return q

    def glacier_attrs(self)->pd.DataFrame:
        """
        returns a dataframe with four columns
            - 'glac_area'
            - 'glac_vol'
            - 'glac_mass'
            - 'glac_area_neighbours'
        """
        df = pd.read_csv(
            self.glacier_attr_path,
            sep=';',
            skiprows=1,
            index_col='gauge_id',
            dtype=np.float32
        )
        df.index = df.index.astype(int).astype(str)
        return df

    def climate_attrs(self)->pd.DataFrame:
        """returns 14 climate attributes of catchments.
        """
        df = pd.read_csv(
            self.clim_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id',
            dtype={
                'gauge_id': str,
                'p_mean': float,
                'aridity': float,
                'pet_mean': float,
                'p_seasonality': float,
                'frac_snow': float,
                'high_prec_freq': float,
                'high_prec_dur': float,
                'high_prec_timing': str,
                'low_prec_timing': str
                         }
)
        return df

    def geol_attrs(self)->pd.DataFrame:
        """15 geological features"""
        df = pd.read_csv(
            self.geol_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id',
            dtype=np.float32
        )
        df.index = df.index.astype(int).astype(str)
        return df

    def supp_geol_attrs(self)->pd.DataFrame:
        """supplimentary geological features"""
        df = pd.read_csv(
            self.supp_geol_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id',
            dtype=np.float32
        )

        df.index = df.index.astype(int).astype(str)
        return df

    def human_inf_attrs(self)->pd.DataFrame:
        """
        14 athropogenic factors
        """
        df = pd.read_csv(
    self.hum_inf_attr_path,
    skiprows=1,
    sep=';',
    index_col='gauge_id',
    dtype={
        'gauge_id': str,
        'n_inhabitants': int,
        'dens_inhabitants': float,
        'hp_count': int,
        'hp_qturb': float,
        'hp_inst_turb': float,
        'hp_max_power': float,
        'num_reservoir': int,
        'reservoir_cap': float,
        'reservoir_he': float,
        'reservoir_fs': float,
        'reservoir_irr': float,
        'reservoir_nousedata': float,
        #'reservoir_year_first': int,
        #'reservoir_year_last': int
    }
)
        return df

    def hydrogeol_attrs(self)->pd.DataFrame:
        """10 hydrogeological factors"""
        df = pd.read_csv(
            self.hydrogeol_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id',
            dtype=float
        )
        df.index = df.index.astype(int).astype(str)
        return df

    def hydrol_attrs(self)->pd.DataFrame:
        """14 hydrological parameters + 2 useful infos"""
        df = pd.read_csv(
    self.hydrol_attr_path,
    skiprows=1,
    sep=';',
    index_col='gauge_id',
    dtype={
        'gauge_id': str,
        'sign_number_of_years': int,
        'q_mean': float,
        'runoff_ratio': float, 'stream_elas': float, 'slope_fdc': float,
        'baseflow_index_landson': float,
        'hfd_mean': float,
        'Q5': float, 'Q95': float, 'high_q_freq': float, 'high_q_dur': float,
        'low_q_freq': float
    }
)
        return df

    def landcolover_attrs(self)->pd.DataFrame:
        """13 landcover parameters"""
        return pd.read_csv(
            self.lc_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id',
            dtype={
                'gauge_id': str,
                'crop_perc': float,
                'grass_perc': float,
                'scrub_perc': float,
                'dwood_perc': float,
                'mixed_wood_perc': float,
                'ewood_perc': float,
                'wetlands_perc': float,
                'inwater_perc': float,
                'ice_perc': float,
                'loose_rock_perc': float,
                'rock_perc': float,
                'urban_perc': float,
            'dom_land_cover': str
            }
        )

    def soil_attrs(self)->pd.DataFrame:
        """80 soil parameters"""
        df = pd.read_csv(
            self.soil_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id'
        )
        df.index = df.index.astype(int).astype(str)
        return df

    def topo_attrs(self)->pd.DataFrame:
        """topographic parameters"""
        df = pd.read_csv(
            self.topo_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id',
            encoding="unicode_escape"
        )

        df.index = df.index.astype(int).astype(str)
        return df

    def fetch_static_features(
            self,
            stn_id: Union[str, list] = "all",
            features: Union[str, list] = "all"
    )->pd.DataFrame:
        """
        Returns static features of one or more stations.

        Parameters
        ----------
            stn_id : str
                name/id of station/stations of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Returns
        -------
        pd.DataFrame
            a pandas dataframe of shape (stations, features)

        Examples
        ---------
        >>> from water_datasets import CAMELS_CH
        >>> dataset = CAMELS_CH()
        get the names of stations
        >>> stns = dataset.stations()
        >>> len(stns)
            331
        get all static data of all stations
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (331, 209)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('2004')
        >>> static_data.shape
           (1, 209)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['gauge_lon', 'gauge_lat', 'area'])
        >>> static_data.shape
           (331, 3)
        >>> data = dataset.fetch_static_features('2004', features=['gauge_lon', 'gauge_lat', 'area'])
        >>> data.shape
           (1, 3)
        """
        stations = check_attributes(stn_id, self.stations(), 'stations')

        df = pd.concat(
    [
        self.climate_attrs(),
        self.geol_attrs(),
        self.supp_geol_attrs(),
        self.glacier_attrs(),
        self.human_inf_attrs(),
        self.hydrogeol_attrs(),
        self.hydrol_attrs(),
        self.landcolover_attrs(),
        self.soil_attrs(),
        self.topo_attrs(),
     ],
    axis=1)
        df.index = df.index.astype(str)

        features = check_attributes(features, df.columns.tolist(),
                                    "static features")
        return df.loc[stations, features]

    def _read_dynamic_from_csv(
            self,
            stations,
            dynamic_features,
            st=None,
            en=None
    ) ->dict:
        """
        reads dynamic data of one or more catchments
        """

        attributes = check_attributes(dynamic_features, self.dynamic_features)
        stations = check_attributes(stations, self.stations())

        dyn = {
            stn: self._read_dynamic_for_stn(stn).loc["19810101": "20201231", attributes] for stn in stations
        }
        return dyn

    def _read_dynamic_for_stn(self, stn_id:str)->pd.DataFrame:
        """
        Reads daily dynamic (meteorological + streamflow) data for one catchment
        and returns as DataFrame
        """

        df = pd.read_csv(
            os.path.join(self.dynamic_path, f"CAMELS_CH_obs_based_{stn_id}.csv"),
            sep=';',
            index_col='date',
            parse_dates=True,
            dtype=np.float32
        )

        df.index.name = 'time'
        df.columns.name = 'dynamic_features'

        df.rename(columns=self.dyn_map, inplace=True)

        return df

    @property
    def _area_name(self) ->str:
        return 'area'

    @property
    def _coords_name(self)->List[str]:
        return ['gauge_lat', 'gauge_lon']

    @property
    def _mmd_feature_name(self)->str:
        return 'obs_q_mmd'


class CAMELS_DE(Camels):
    """
    This is the data from 1555 german catchments following the work of
    `Loritz et al., 2024 <https://doi.org/10.5194/essd-2024-318>`_ .
    The data is downloaded from `zenodo <https://zenodo.org/record/12733968>`_ .
    This data consists of 155 static and 21 dynamic features. The dynamic features
    span from 1951-01-01 to 2020-12-31 with daily timestep.

    Examples
    --------
    >>> from water_datasets import CAMELS_DE
    >>> dataset = CAMELS_DE()
    >>> df = dataset.fetch(stations=1, as_dataframe=True)
    >>> df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
    >>> df.shape
       (25568, 21)
    get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       1555
    get data of 10 % of stations as dataframe
    >>> df = dataset.fetch(0.1, as_dataframe=True)
    >>> df.shape
        (536928, 155)
    The returned dataframe is a multi-indexed data
    >>> df.index.names == ['time', 'dynamic_features']
        True
    get data by station id
    >>> df = dataset.fetch(stations='DE110260', as_dataframe=True).unstack()
    >>> df.shape
        (25568, 21)
    get names of available dynamic features
    >>> dataset.dynamic_features
    get only selected dynamic features
    >>> data = dataset.fetch(1, as_dataframe=True,
    ...  dynamic_features=['temperature_mean', 'humidity_mean', 'precipitation_mean', 'discharge_vol']).unstack()
    >>> data.shape
        (25568, 4)
    get names of available static features
    >>> dataset.static_features
    get data of 10 random stations
    >>> df = dataset.fetch(10, as_dataframe=True)
    >>> df.shape  # remember this is a multiindexed dataframe
        (536928, 10)
    when we get both static and dynamic data, the returned data is a dictionary
    with ``static`` and ``dyanic`` keys.
    >>> data = dataset.fetch(stations='DE110260', static_features="all", as_dataframe=True)
    >>> data['static'].shape, data['dynamic'].shape
        ((1, 111), (536928, 1))
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (1555, 2)
    >>> dataset.stn_coords('DE110250')  # returns coordinates of station whose id is DE110250
        47.925221	8.191595
    >>> dataset.stn_coords(['DE110250', 'DE110260'])  # returns coordinates of two stations
    """
    url = "https://zenodo.org/record/12733968"


    def __init__(
            self,
            path=None,
            overwrite:bool = False,
            to_netcdf: bool = True,
            verbosity: int = 1,
            **kwargs
    ):
        """

        Parameters
        ----------
        path : str
            If the data is alredy downloaded then provide the complete
            path to it. If None, then the data will be downloaded.
            The data is downloaded once and therefore susbsequent
            calls to this class will not download the data unless
            ``overwrite`` is set to True.
        overwrite : bool
            If the data is already down then you can set it to True,
            to make a fresh download.
        to_netcdf : bool
            whether to convert all the data into one netcdf file or not.
            This will fasten repeated calls to fetch etc. but will
            require netCDF5 package as well as xarray.
        """
        super().__init__(path=path, verbosity=verbosity,  **kwargs)

        self._download(overwrite=overwrite)

        if to_netcdf and netCDF4 is None:
            warnings.warn("netCDF4 is not installed. Therefore, the data will not be converted to netcdf format.")  
            to_netcdf = False

        if to_netcdf:
            self._maybe_to_netcdf('camels_de_dyn')

        self.boundary_file = os.path.join(path, "CAMELS_DE", "camels_de", 
                                          "CAMELS_DE_catchment_boundaries",
                                          "catchments", "CAMELS_DE_catchments.shp")
        self._create_boundary_id_map(self.boundary_file, 0)

    @property
    def dyn_map(self):
        return {
        'discharge_vol': 'obs_q_cms',
        'discharge_spec': 'obs_q_mmd',
        'temperature_min': 'min_temp_C',
        'temperature_max': 'max_temp_C',
        'temperature_mean': 'mean_temp_C',
        #'precipitation_mean': 'pcp_mm',
        'precipitation_mean': 'mean_pcp_mm',
        'precipitation_median': 'median_pcp_mm',
        'precipitation_stdev': 'std_pcp_mm',
        'precipitation_min': 'min_pcp_mm',
        'precipitation_max': 'max_pcp_mm',
        'humidity_mean': 'mean_rh_%',
        'humidity_median': 'median_rh_%',
        'humidity_stdev': 'std_rh_%',
        'humidity_min': 'min_rh_%',
        'humidity_max': 'max_rh_%',
        }

    @property
    def ts_dir(self)->str:
        return os.path.join(self.path, 'camels_de', 'timeseries')

    @property
    def clim_attr_path(self)->str:
        return os.path.join(self.path, 'camels_de', 'CAMELS_DE_climatic_attributes.csv')

    @property
    def hum_infl_path(self)->str:
        return os.path.join(self.path, 'camels_de', 'CAMELS_DE_humaninfluence_attributes.csv')

    @property
    def hydrogeol_attr_path(self)->str:
        return os.path.join(self.path, 'camels_de','CAMELS_DE_hydrogeology_attributes.csv')
    
    @property
    def hydrol_attr_path(self)->str:
        return os.path.join(self.path, 'camels_de','CAMELS_DE_hydrologic_attributes.csv')

    @property
    def lc_attr_path(self)->str:
        return os.path.join(self.path, 'camels_de', 'CAMELS_DE_landcover_attributes.csv')

    @property
    def sim_attr_path(self)->str:
        return os.path.join(self.path, 'camels_de',  'CAMELS_DE_simulation_benchmark.csv')

    @property
    def soil_attr_path(self)->str:
        return os.path.join(self.path, 'camels_de', 'CAMELS_DE_soil_attributes.csv')

    @property
    def topo_attr_path(self)->str:
        return os.path.join(self.path, 'camels_de',  'CAMELS_DE_topographic_attributes.csv')

    def stations(self)->List[str]:
        return [f.split('_')[4].split('.')[0] for f in os.listdir(self.ts_dir)]

    def clim_attrs(self)->pd.DataFrame:
        return pd.read_csv(self.clim_attr_path, index_col='gauge_id', 
                           #dtype=np.float32
                           )
    
    def hum_infl_attrs(self)->pd.DataFrame:
        return pd.read_csv(self.hum_infl_path, index_col='gauge_id', 
                           #dtype=np.float32
                           )
    
    def hydrogeol_attrs(self)->pd.DataFrame:
        return pd.read_csv(self.hydrogeol_attr_path, index_col='gauge_id', 
                           #dtype=np.float32
                           )
    
    def hydrol_attrs(self)->pd.DataFrame:
        return pd.read_csv(self.hydrol_attr_path, index_col='gauge_id', #dtype=np.float32
                           )
    
    def lc_attrs(self)->pd.DataFrame:
        return pd.read_csv(self.lc_attr_path, index_col='gauge_id', #dtype=np.float32
                           )
    
    def sim_attrs(self)->pd.DataFrame:
        return pd.read_csv(self.sim_attr_path, index_col='gauge_id', #dtype=np.float32
                           )
    
    def soil_attrs(self)->pd.DataFrame:
        return pd.read_csv(self.soil_attr_path, index_col='gauge_id', #dtype=np.float32
                           )
    
    def topo_attrs(self)->pd.DataFrame:
        return pd.read_csv(self.topo_attr_path, index_col='gauge_id', #dtype=np.float32
                           )
    
    def static_data(self)->pd.DataFrame:
        return pd.concat([
            self.clim_attrs(),
            self.hum_infl_attrs(),
            self.hydrogeol_attrs(),
            self.hydrol_attrs(),
            self.lc_attrs(),
            self.sim_attrs(),
            self.soil_attrs(),
            self.topo_attrs()
        ], axis=1)

    def fetch_static_features(
            self,
            stn_id: Union[str, list] = "all",
            features: Union[str, list] = "all"
    )->pd.DataFrame:
        """

        Returns static features of one or more stations.

        Parameters
        ----------
            stn_id : str
                name/id of station/stations of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Returns
        -------
        pd.DataFrame
            a pandas dataframe of shape (stations, features)

        Examples
        ---------
        >>> from water_datasets import CAMELS_CH
        >>> dataset = CAMELS_DE()
        get all static data of all stations
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (1555, 111)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('DE110010')
        >>> static_data.shape
           (1, 111)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['p_mean', 'p_seasonality', 'frac_snow'])
        >>> static_data.shape
           (1555, 3)
        >>> data = dataset.fetch_static_features('DE110000', features=['p_mean', 'p_seasonality', 'frac_snow'])
        >>> data.shape
           (1, 3)
        """
        stations = check_attributes(stn_id, self.stations(), 'stations')

        df = self.static_data()
        features = check_attributes(features, df.columns.tolist(),
                                    "static_features")
        return df.loc[stations, features]

    def _read_dynamic_from_csv(
            self,
            stations,
            dynamic_features,
            st="19510101",
            en="20201231"
    ) ->dict:
        """
        reads dynamic data of one or more catchments
        """

        features = check_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')
        stations = check_attributes(stations, self.stations(), 'stations')

        cpus = self.processes or min(get_cpus(), 32)

        if cpus == 1 or len(stations) < 32:

            dyn = {
                stn: self._read_dynamic_for_stn(stn).loc[st: en, features] for stn in stations
            }
        else:
            with cf.ProcessPoolExecutor(cpus) as executor:
                dyn = executor.map(self._read_dynamic_for_stn, stations)
            
            dyn = {stn: df.loc[st: en, features] for stn, df in zip(stations, dyn)}

        return dyn

    def _read_dynamic_for_stn(self, stn_id)->pd.DataFrame:
        """
        Reads daily dynamic (meteorological + streamflow) data for one catchment
        and returns as DataFrame
        """

        df = pd.read_csv(
            os.path.join(self.ts_dir, f"CAMELS_DE_hydromet_timeseries_{stn_id}.csv"),
            #sep=';',
            index_col='date',
            parse_dates=True,
            #dtype=np.float32
        )

        df.rename(columns=self.dyn_map, inplace=True)
        df.index.name = 'time'
        df.columns.name = 'dynamic_features'
        return df

    @property
    def start(self):
        return pd.Timestamp('1951-01-01')

    @property
    def end(self):
        return pd.Timestamp('2020-12-31')

    @property
    def dynamic_features(self)->List[str]:
        return self._read_dynamic_for_stn(self.stations()[0]).columns.tolist()
    
    @property
    def static_features(self)->List[str]:
        return self.static_data().columns.tolist()

    @property
    def _coords_name(self)->List[str]:
        return ['gauge_lat', 'gauge_lon']

    @property
    def _area_name(self) ->str:
        return 'area'

    @property
    def _mmd_feature_name(self) ->str:
        """Observed catchment-specific discharge (converted to millimetres per day
        using catchment areas"""
        return 'obs_q_mmd'


class GRDCCaravan(Camels):
    """
    This is a dataset of 5357 catchments from around the globe following the works of 
    `Faerber et al., 2023 <https://zenodo.org/records/10074416>`_ . The dataset consists of 39
    dynamic (timeseries) features and 211 static features. The dynamic (timeseries) data
    spands from 1950-01-02 to 2019-05-19. 

    if xarray + netCDF4 packages are installed then netcdf files will be downloaded
    otherwise csv files will be downloaded and used.

    Examples
    --------
    >>> from water_datasets import GRDCCaravan
    >>> dataset = GRDCCaravan()
    >>> df = dataset.fetch(stations=1, as_dataframe=True)
    >>> df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
    >>> df.shape
       (26801, 39)
    get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       5357
    get data of 10 % of stations as dataframe
    >>> df = dataset.fetch(0.1, as_dataframe=True)
    >>> df.shape
       (1045239, 535)
    The returned dataframe is a multi-indexed data
    >>> df.index.names == ['time', 'dynamic_features']
        True
    get data by station id
    >>> df = dataset.fetch(stations='GRDC_3664802', as_dataframe=True).unstack()
    >>> df.shape
         (26800, 39)
    get names of available dynamic features
    >>> dataset.dynamic_features
    get only selected dynamic features
    >>> data = dataset.fetch(1, as_dataframe=True,
    ...  dynamic_features=['total_precipitation_sum', 'potential_evaporation_sum', 'temperature_2m_mean', 'streamflow']).unstack()
    >>> data.shape
        (26800, 4)
    get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> df = dataset.fetch(10, as_dataframe=True)
    >>> df.shape  # remember this is a multiindexed dataframe
        (1045239, 10)
    when we get both static and dynamic data, the returned data is a dictionary
    with ``static`` and ``dyanic`` keys.
    >>> data = dataset.fetch(stations='GRDC_3664802', static_features="all", as_dataframe=True)
    >>> data['static'].shape, data['dynamic'].shape
        ((1, 211), (1045200, 1))
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (5357, 2)
    >>> dataset.stn_coords('GRDC_3664802')  # returns coordinates of station whose id is GRDC_3664802
        -26.2271	-51.0771
    >>> dataset.stn_coords(['GRDC_3664802', 'GRDC_1159337'])  # returns coordinates of two stations

    """

    url = {
        'caravan-grdc-extension-nc.tar.gz': 
            "https://zenodo.org/records/10074416/files/caravan-grdc-extension-nc.tar.gz?download=1",
        'caravan-grdc-extension-csv.tar.gz': 
            "https://zenodo.org/records/10074416/files/caravan-grdc-extension-csv.tar.gz?download=1"
    }
    def __init__(
            self,
            path=None,
            overwrite:bool = False,
            verbosity: int = 1,
            **kwargs
    ):
        
        if xr is None:
            self.ftype == 'csv'
            if "caravan-grdc-extension-nc.tar.gz" in self.url:
                self.url.pop("caravan-grdc-extension-nc.tar.gz")
        else:
            self.ftype = 'netcdf'
            if "caravan-grdc-extension-csv.tar.gz" in self.url:
                self.url.pop("caravan-grdc-extension-csv.tar.gz")
        
        super().__init__(path=path, verbosity=verbosity, **kwargs)

        for _file, url in self.url.items():
            fpath = os.path.join(self.path, _file)
            if not os.path.exists(fpath) and not overwrite:
                if self.verbosity > 0:
                    print(f"Downloading {_file} from {url+ _file}")
                download(url + _file, outdir=self.path, fname=_file,)
                _unzip(self.path)        
            elif self.verbosity > 0:
                print(f"{_file} at {self.path} already exists")                

        self.boundary_file = os.path.join(
            self.shapefiles_path, 
            'grdc_basin_shapes.shp'
            )
        self._create_boundary_id_map(self.boundary_file, 0)

        # so that we dont have to read the files again and again
        self._stations = self.other_attributes().index.to_list()
        self._static_attributes = self.static_data().columns.tolist()
        self._dynamic_attributes = self._read_dynamic_for_stn(self.stations()[0]).columns.tolist()

        self.dyn_fname = ''

    @property
    def dyn_map(self):
        return {
        'streamflow': 'obs_q_cms',
        'temperature_2m_mean': 'mean_temp_C',
        'temperature_2m_min': 'min_temp_C',
        'temperature_2m_min': 'max_temp_C',
        'total_precipitation_sum': 'pcp_mm',
        }
    
    @property
    def static_features(self):
        return self._static_attributes

    @property
    def dynamic_features(self):
        return self._dynamic_attributes

    @property
    def shapefiles_path(self):
        if self.ftype == 'csv':
            return os.path.join(self.path, 'GRDC-Caravan-extension-csv', 
                                'shapefiles', 'grdc')
        return os.path.join(self.path, 'GRDC-Caravan-extension-nc', 
                            'shapefiles', 'grdc')

    @property
    def attrs_path(self):
        if self.ftype == 'csv':
            return os.path.join(self.path, 'GRDC-Caravan-extension-csv', 
                                'attributes', 'grdc')
        return os.path.join(self.path, 'GRDC-Caravan-extension-nc', 
                            'attributes', 'grdc')
    
    @property
    def ts_path(self)->os.PathLike:
        if self.ftype == 'csv':
            return os.path.join(self.path, 'GRDC-Caravan-extension-csv', 
                                'timeseries', 'grdc')

        return os.path.join(self.path, 'GRDC-Caravan-extension-nc', 
                            'timeseries', self.ftype, 'grdc')

    def stations(self)->List[str]:
        return self._stations

    @property
    def _coords_name(self)->List[str]:
        return ['gauge_lat', 'gauge_lon']

    @property
    def _area_name(self) ->str:
        return 'area'    

    @property
    def start(self):
        return pd.Timestamp("19500102")

    @property
    def end(self):
        return pd.Timestamp("20230519")

    @property
    def _q_name(self) ->str:
        return 'obs_q_cms'
        
    def other_attributes(self)->pd.DataFrame:
        return pd.read_csv(os.path.join(self.attrs_path, 'attributes_other_grdc.csv'), index_col='gauge_id')
    
    def hydroatlas_attributes(self)->pd.DataFrame:
        return pd.read_csv(os.path.join(self.attrs_path, 'attributes_hydroatlas_grdc.csv'), index_col='gauge_id')
    
    def caravan_attributes(self)->pd.DataFrame:
        return pd.read_csv(os.path.join(self.attrs_path, 'attributes_caravan_grdc.csv'), index_col='gauge_id')
    
    def static_data(self)->pd.DataFrame:
        return pd.concat([
            self.other_attributes(),
            self.hydroatlas_attributes(),
            self.caravan_attributes(),
        ], axis=1)

    def fetch_station_features(
            self,
            station: str,
            dynamic_features: Union[str, list, None] = 'all',
            static_features: Union[str, list, None] = None,
            as_ts: bool = False,
            st: Union[str, None] = None,
            en: Union[str, None] = None,
            **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetches features for one station.

        Parameters
        -----------
            station :
                station id/gauge id for which the data is to be fetched.
            dynamic_features : str/list, optional
                names of dynamic features/attributes to fetch
            static_features :
                names of static features/attributes to be fetches
            as_ts : bool
                whether static features are to be converted into a time
                series or not. If yes then the returned time series will be of
                same length as that of dynamic attribtues.
            st : str,optional
                starting point from which the data to be fetched. By default,
                the data will be fetched from where it is available.
            en : str, optional
                end point of data to be fetched. By default the dat will be fetched

        Returns
        -------
        Dict
            dataframe if as_ts is True else it returns a dictionary of static and
            dynamic features for a station/gauge_id

        Examples
        --------
            >>> from water_datasets import GRDCCaravan
            >>> dataset = GRDCCaravan()
            >>> dataset.fetch_station_features('912101A')
        """
        dynamic_features = check_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')

        if self.ftype == "netcdf":
            fpath = os.path.join(self.ts_path, f'{station}.nc')
            df = xr.open_dataset(fpath).to_dataframe()
        else:
            fpath = os.path.join(self.ts_path, f'{station}.csv')
            df = pd.read_csv(fpath, index_col='date', parse_dates=True)

        df.rename(columns=self.dyn_map, inplace=True)

        if static_features is not None:
            static = self.fetch_static_features(station, static_features)
        
        return {'static': static, 'dynamic': df[dynamic_features]}

    def fetch_static_features(
            self,
            stn_id: Union[str, list] = "all",
            features: Union[str, list] = "all"
    )->pd.DataFrame:
        """

        Returns static features of one or more stations.

        Parameters
        ----------
            stn_id : str
                name/id of station/stations of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Returns
        -------
        pd.DataFrame
            a pandas dataframe of shape (stations, features)

        Examples
        ---------
        >>> from water_datasets import GRDCCaravan
        >>> dataset = GRDCCaravan()
        get all static data of all stations
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (1555, 111)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('DE110010')
        >>> static_data.shape
           (1, 111)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['p_mean', 'p_seasonality', 'frac_snow'])
        >>> static_data.shape
           (1555, 3)
        >>> data = dataset.fetch_static_features('DE110000', features=['p_mean', 'p_seasonality', 'frac_snow'])
        >>> data.shape
           (1, 3)
        """
        stations = check_attributes(stn_id, self.stations(), 'stations')

        df = self.static_data()
        features = check_attributes(features, df.columns.tolist(),
                                    "static features")
        return df.loc[stations, features]

    def _read_dynamic_from_csv(
            self, 
            stations, 
            dynamic_features, 
            st=None,
            en=None)->dict:

        dynamic_features = check_attributes(dynamic_features, self.dynamic_features)
        stations = check_attributes(stations, self.stations())

        if len(stations) > 10:
            cpus = self.processes or min(get_cpus(), 64)
            with  cf.ProcessPoolExecutor(max_workers=cpus) as executor:
                results = executor.map(
                    self._read_dynamic_for_stn,
                    stations,
                )
            dyn = {stn:data.loc[st:en, dynamic_features] for stn, data in zip(stations, results)}
        else:
            dyn = {
                stn: self._read_dynamic_for_stn(stn).loc[st: en, dynamic_features] for stn in stations
            }

        return dyn

    def _read_dynamic_for_stn(self, stn_id)->pd.DataFrame:
        if self.ftype == "netcdf":
            fpath = os.path.join(self.ts_path, f'{stn_id}.nc')
            df = xr.load_dataset(fpath).to_dataframe()
        else:
            fpath = os.path.join(self.ts_path, f'{stn_id}.csv')
            df = pd.read_csv(fpath, index_col='date', parse_dates=True)

        df.rename(columns=self.dyn_map, inplace=True)

        df.index.name = 'time'
        df.columns.name = 'dynamic_features'
        return df


class CAMELS_SE(Camels):
    """
    Dataset of 50 Swedish catchments following the works of 
    `Teutschbein et al., 2024 <https://doi.org/10.1002/gdj3.239>`_ . 
    The dataset consists of 76 static catchment features and 4 dynamic features.
    The dynamic features span from 19610101 to 20201231 with daily timestep.
    
    Examples
    --------
    >>> from water_datasets import CAMELS_SE
    >>> dataset = CAMELS_SE()
    >>> df = dataset.fetch(stations=1, as_dataframe=True)
    >>> df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
    >>> df.shape
       (21915, 4)
    get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       50
    get data of 10 % of stations as dataframe
    >>> df = dataset.fetch(0.1, as_dataframe=True)
    >>> df.shape
       (87660, 5)
    The returned dataframe is a multi-indexed data
    >>> df.index.names == ['time', 'dynamic_features']
        True
    get data by station id
    >>> df = dataset.fetch(stations='5', as_dataframe=True).unstack()
    >>> df.shape
         (21915, 4)
    get names of available dynamic features
    >>> dataset.dynamic_features
    get only selected dynamic features
    >>> data = dataset.fetch(1, as_dataframe=True,
    ...  dynamic_features=['Qobs_m3s', 'Qobs_mm', 'Pobs_mm', 'Tobs_C']).unstack()
    >>> data.shape
        (21915, 4)
    get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> df = dataset.fetch(10, as_dataframe=True)
    >>> df.shape  # remember this is a multiindexed dataframe
        (87660, 10)
    when we get both static and dynamic data, the returned data is a dictionary
    with ``static`` and ``dyanic`` keys.
    >>> data = dataset.fetch(stations='5', static_features="all", as_dataframe=True)
    >>> data['static'].shape, data['dynamic'].shape
        ((1, 76), (87660, 1))
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (50, 2)
    >>> dataset.stn_coords('5')  # returns coordinates of station whose id is GRDC_3664802
        68.0356	21.9758
    >>> dataset.stn_coords(['5', '200'])  # returns coordinates of two stations
    
    """

    url = {
        'catchment properties.zip': "https://snd.se/sv/catalogue/download/dataset/2023-173/1?principal=user.uu.se&filename=catchment+properties.zip",
        'catchment time series.zip': 'https://snd.se/sv/catalogue/download/dataset/2023-173/1?principal=user.uu.se&filename=catchment+time+series.zip',
        'catchment_GIS_shapefiles.zip': "https://snd.se/sv/catalogue/download/dataset/2023-173/2?principal=user.uu.se&filename=catchment_GIS_shapefiles.zip",
    }

    def __init__(
            self,
            path: str = None,
            to_netcdf:bool = True,
            overwrite:bool = False,
            verbosity:int = 1,
            **kwargs
    ):
        """
        Arguments:
            path: path where the CAMELS_SE dataset has been downloaded. This path
                must contain five zip files and one xlsx file. If None, then the
                data will be downloaded.
            to_netcdf :
        """
        super().__init__(path=path, verbosity=verbosity, **kwargs)

        for _file, url in self.url.items():
            fpath = os.path.join(self.path, _file)
            if not os.path.exists(fpath) and not overwrite:
                if verbosity > 0:
                    print(f"Downloading {_file} from {url+ _file}")
                download(url, outdir=self.path, fname=_file,)
                _unzip(self.path)
            else:
                if self.verbosity> 0: print(f"{_file} at {self.path} already exists")

        self.boundary_file = os.path.join(self.path, 
                                                  'catchment_GIS_shapefiles', 
                                                  'catchment_GIS_shapefiles', 
                                                  'Sweden_catchments_50_boundaries_WGS84.shp')

        self._create_boundary_id_map(self.boundary_file, 0)

        self._static_features = list(set(self.static_data().columns.tolist()))
        self._stations = self.physical_properties().index.to_list()
        self._dynamic_features = self._read_dynamic_for_stn(self.stations()[0], nrows=2).columns.tolist()

        if to_netcdf and netCDF4 is None:
            warnings.warn("netCDF4 is not installed. Therefore, the data will not be converted to netcdf format.")  
            to_netcdf = False

        if to_netcdf:
            self._maybe_to_netcdf('camels_se_dyn')

    @property
    def dyn_map(self):
        return {
        'Qobs_m3s': 'obs_q_cms',
        'Qobs_mm': 'obs_q_mmd',
        'Tobs_C': 'mean_temp_C',
        'Pobs_mm': 'pcp_mm',
        }

    @property
    def static_features(self):
        return self._static_features
    
    @property
    def dynamic_features(self)->List[str]:
        return self._dynamic_features

    @property
    def properties_path(self):
        return os.path.join(self.path, 'catchment properties', 'catchment properties')

    @property
    def ts_dir(self)->os.PathLike:
        return os.path.join(self.path, 'catchment time series', 'catchment time series')

    @property
    def _mmd_feature_name(self) ->str:
        return 'obs_q_cms'   

    @property
    def _coords_name(self)->List[str]:
        return ['Latitude_WGS84', 'Longitude_WGS84']

    @property
    def _area_name(self) ->str:
        return 'Area_km2'  

    @property
    def start(self):
        return pd.Timestamp("19610101")

    @property
    def end(self):
        return pd.Timestamp("20201231")

    def stations(self)->List[str]:
        return self._stations
    
    def landcover(self)->pd.DataFrame:
        return pd.read_csv(
            os.path.join(self.properties_path, 'catchments_landcover.csv'), 
            index_col='ID', dtype={'ID': str})
    
    def physical_properties(self)->pd.DataFrame:
        return pd.read_csv(
            os.path.join(self.properties_path, 'catchments_physical_properties.csv'), 
            index_col='ID', dtype={'ID': str})
    
    def soil_classes(self)->pd.DataFrame:
        df = pd.read_csv(
            os.path.join(self.properties_path, 'catchments_soil_classes.csv'), 
            index_col='ID', dtype={'ID': str})
        df.columns = [f"{c}_sc" for c in df.columns]
        return df
    
    def hydro_signatures_1961_2020(self)->pd.DataFrame:
        df = pd.read_csv(
            os.path.join(self.properties_path, 'catchments_hydrological_signatures_1961_2020.csv'), 
            index_col='ID', dtype={'ID': str})
        df.columns = [f"{c}_hs" for c in df.columns]
        return df

    def hydro_signatures_CNP_1961_1990(self)->pd.DataFrame:
        df = pd.read_csv(
            os.path.join(self.properties_path, 'catchments_hydrological_signatures_CNP1_1961_1990.csv'), 
            index_col='ID', dtype={'ID': str})    
        df.columns = [f"{c}_CNP_61_90" for c in df.columns]
        return df

    def hydro_signatures_CNP_1990_2020(self)->pd.DataFrame:
        df = pd.read_csv(
            os.path.join(self.properties_path, 'catchments_hydrological_signatures_CNP2_1991_2020.csv'), 
            index_col='ID', dtype={'ID': str})      
        df.columns = [f"{c}_CNP_91_20" for c in df.columns]
        return df

    def static_data(self)->pd.DataFrame:
        return pd.concat([
            self.landcover(),
            self.physical_properties(),
            self.soil_classes(),
            self.hydro_signatures_1961_2020(),
            self.hydro_signatures_CNP_1961_1990(),
            self.hydro_signatures_CNP_1990_2020()
        ], axis=1)    
 
    def _read_dynamic_from_csv(
            self,
            stations,
            dynamic_features,
            st="1961-01-01",
            en="2020-12-31"
    ) ->dict:
        """
        reads dynamic data of one or more catchments
        """

        attributes = check_attributes(dynamic_features, self.dynamic_features)
        stations = check_attributes(stations, self.stations())

        dyn = {
            stn: self._read_dynamic_for_stn(stn).loc[st: en, attributes] for stn in stations
        }

        return dyn

    def _read_dynamic_for_stn(self, stn_id, nrows=None)->pd.DataFrame:
        """
        Reads daily dynamic (meteorological + streamflow) data for one catchment
        and returns as DataFrame
        """
        # find file starting with 'catchment_id_stn_id_' in self.path
        stn_id = f'catchment_id_{stn_id}_'
        fname = [f for f in os.listdir(self.ts_dir) if f.startswith(stn_id)]
        assert len(fname) == 1
        fname = fname[0]

        df = pd.read_csv(
            os.path.join(self.ts_dir, fname),
            index_col='Year_Month_Day',
            parse_dates=[['Year', 'Month', 'Day']],
            dtype={'Qobs_m3s': np.float32, 'Qobs_mm': np.float32, 'Pobs_mm': np.float32, 'Tobs_C': np.float32},
            nrows=nrows,
        )   
        df.index.name = 'time'
        df.columns.name = 'dynamic_features'

        for old_name, new_name in self.dyn_map.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)

        return df

    def fetch_static_features(
            self,
            stn_id: Union[str, list] = "all",
            features: Union[str, list] = "all"
    )->pd.DataFrame:
        """

        Returns static features of one or more stations.

        Parameters
        ----------
            stn_id : str
                name/id of station/stations of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Returns
        -------
        pd.DataFrame
            a pandas dataframe of shape (stations, features)

        Examples
        ---------
        >>> from water_datasets import CAMELS_SE
        >>> dataset = CAMELS_SE()
        get all static data of all stations
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (50, 76)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('5')
        >>> static_data.shape
           (1, 76)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['Area_km2', 'Water_percentage', 'Elevation_mabsl'])
        >>> static_data.shape
           (50, 3)
        >>> data = dataset.fetch_static_features('5', features=['Area_km2', 'Water_percentage', 'Elevation_mabsl'])
        >>> data.shape
           (1, 3)
        """
        stations = check_attributes(stn_id, self.stations(), 'stations')

        df = self.static_data().copy()
        features = check_attributes(features, self.static_features,
                                    "static features")
        return df.loc[stations, features]        


class CAMELS_DK(Camels):
    """
    This is an updated version of :py class: `water_datasets.rr.CAMELS_DK0` 
    dataset . This dataset was presented
    by `Liu et al., 2024 <https://doi.org/10.5194/essd-2024-292>`_ and is 
    available at `dataverse <https://dataverse.geus.dk/dataset.xhtml?persistentId=doi:10.22008/FK2/AZXSYP>`_ .
    This dataset consists of 119 static and 13 dynamic features from 3330 danish catchments. 
    The dynamic (time series) features span from 1989-01-02 to 2023-12-31 with daily timestep.
    However, the streamflow observations are available for only 304 catchments.

    Examples
    ---------
    >>> from water_datasets import CAMELS_DK
    >>> dataset = CAMELS_DK()
    >>> data = dataset.fetch(0.1, as_dataframe=True)
    >>> data.shape
    (166166, 30)  # 30 represents number of stations
    Since data is a multi-index dataframe, we can get data of one station as below
    >>> data['54130033'].unstack().shape
    (12782, 13)
    If we don't set as_dataframe=True, then the returned data will be a xarray Dataset
    >>> data = dataset.fetch(0.1)
    >>> type(data)
        xarray.core.dataset.Dataset
    >>> data.dims
    FrozenMappingWarningOnValuesAccess({'time': 12782, 'dynamic_features': 13})
    >>> len(data.data_vars)
        30
    >>> df = dataset.fetch(stations=1, as_dataframe=True)  # get data of only one random station
    >>> df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
    >>> df.shape
    (12782, 13)
    # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
    304
    # get data by station id
    >>> df = dataset.fetch(stations='54130033', as_dataframe=True).unstack()
    >>> df.shape
    (12782, 13)
    # get names of available dynamic features
    >>> dataset.dynamic_features
    # get only selected dynamic features
    >>> df = dataset.fetch(1, as_dataframe=True,
    ... dynamic_features=['Abstraction', 'pet', 'temperature', 'precipitation', 'Qobs']).unstack()
    >>> df.shape
    (12782, 5)
    # get names of available static features
    >>> dataset.static_features
    # get data of 10 random stations
    >>> df = dataset.fetch(10, as_dataframe=True)
    >>> df.shape
    (166166, 10)  # remember this is multi-indexed DataFrame
    # when we get both static and dynamic data, the returned data is a dictionary
    # with ``static`` and ``dyanic`` keys.
    >>> data = dataset.fetch(stations='54130033', static_features="all", as_dataframe=True)
    >>> data['static'].shape, data['dynamic'].shape
    ((1, 119), (166166, 1))
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (304, 2)
    >>> dataset.stn_coords('54130033')  # returns coordinates of station whose id is GRDC_3664802
        6131379.493	559057.7232
    >>> dataset.stn_coords(['54130033', '13210113'])  # returns coordinates of two stations    
    """

    url = {
        'CAMELS_DK_304_gauging_catchment_boundaries.cpg': 'https://dataverse.geus.dk/api/access/datafile/83017',
        'CAMELS_DK_304_gauging_catchment_boundaries.prj': 'https://dataverse.geus.dk/api/access/datafile/83019',
        'CAMELS_DK_304_gauging_catchment_boundaries.shp': 'https://dataverse.geus.dk/api/access/datafile/83021',
        'CAMELS_DK_304_gauging_catchment_boundaries.dbf': 'https://dataverse.geus.dk/api/access/datafile/83020',
        'CAMELS_DK_304_gauging_catchment_boundaries.shx': 'https://dataverse.geus.dk/api/access/datafile/83018',
        'CAMELS_DK_304_gauging_stations.cpg': 'https://dataverse.geus.dk/api/access/datafile/83008',
        'CAMELS_DK_304_gauging_stations.dbf': 'https://dataverse.geus.dk/api/access/datafile/83010',
        'CAMELS_DK_304_gauging_stations.prj': 'https://dataverse.geus.dk/api/access/datafile/83009',
        'CAMELS_DK_304_gauging_stations.shp': 'https://dataverse.geus.dk/api/access/datafile/83011',
        'CAMELS_DK_304_gauging_stations.shx': 'https://dataverse.geus.dk/api/access/datafile/83007',
        'CAMELS_DK_climate.csv': 'https://dataverse.geus.dk/api/access/datafile/83123',
        'CAMELS_DK_geology.csv': 'https://dataverse.geus.dk/api/access/datafile/83124',
        'CAMELS_DK_georegion.dbf': 'https://dataverse.geus.dk/api/access/datafile/83030',
        'CAMELS_DK_georegion.prj': 'https://dataverse.geus.dk/api/access/datafile/83026',
        'CAMELS_DK_georegion.sbn': 'https://dataverse.geus.dk/api/access/datafile/83027',
        'CAMELS_DK_georegion.sbx': 'https://dataverse.geus.dk/api/access/datafile/83028',
        'CAMELS_DK_georegion.shp': 'https://dataverse.geus.dk/api/access/datafile/83029',
        'CAMELS_DK_georegion.shx': 'https://dataverse.geus.dk/api/access/datafile/83031',
        'CAMELS_DK_landuse.csv': 'https://dataverse.geus.dk/api/access/datafile/83125',
        'CAMELS_DK_script.py': 'https://dataverse.geus.dk/api/access/datafile/83135',
        'CAMELS_DK_signature_obs_based.csv': 'https://dataverse.geus.dk/api/access/datafile/83131',
        'CAMELS_DK_signature_sim_based.csv': 'https://dataverse.geus.dk/api/access/datafile/83132',
        'CAMELS_DK_soil.csv': 'https://dataverse.geus.dk/api/access/datafile/83126',
        'CAMELS_DK_topography.csv': 'https://dataverse.geus.dk/api/access/datafile/83127',
        'Data_description.pdf': 'https://dataverse.geus.dk/api/access/datafile/83138',
        'Gauged_catchments.zip': 'https://dataverse.geus.dk/api/access/datafile/83022',
        'Ungauged_catchments.zip': 'https://dataverse.geus.dk/api/access/datafile/83025',
    }

    def __init__(self,
                 path=None,
                 overwrite=False,
                 to_netcdf:bool = True,
                 **kwargs):
        """
        Parameters
        ----------
        path : str
            If the data is alredy downloaded then provide the complete
            path to it. If None, then the data will be downloaded.
            The data is downloaded once and therefore susbsequent
            calls to this class will not download the data unless
            ``overwrite`` is set to True.
        overwrite : bool
            If the data is already down then you can set it to True,
            to make a fresh download.
        to_netcdf : bool
            whether to convert all the data into one netcdf file or not.
            This will fasten repeated calls to fetch etc but will
            require netcdf5 package as well as xarry.
        """
        super(CAMELS_DK, self).__init__(path=path, **kwargs)
        self._download(overwrite=overwrite)

        #self.dyn_fname = os.path.join(self.path, 'camelsdk_dyn.nc')
        self._static_features = self.static_data().columns.to_list()
        self._dynamic_features = self._read_csv(self.stations()[0]).columns.to_list()

        if to_netcdf:
            self._maybe_to_netcdf('camels_dk_dyn')
        
        self.boundary_file = os.path.join(
        self.path,
        "CAMELS_DK_304_gauging_catchment_boundaries.shp"
    )   
        self._create_boundary_id_map(self.boundary_file, 0)

    @property
    def dyn_map(self):
        return {
            'Qobs': 'obs_q_cms',
            'temperature': 'mean_temp_C',
            'precipitation': 'pcp_mm',
            'pet': 'makkink_pet_mm',
            'Qsim': 'sim_q_cms',
        }

    @property
    def gaug_catch_path(self):
        return os.path.join(self.path, "Gauged_catchments", "Gauged_catchments")
    
    @property
    def climate_fpath(self):
        return os.path.join(self.path, "CAMELS_DK_climate.csv")
    
    @property
    def geology_fpath(self):
        return os.path.join(self.path, "CAMELS_DK_geology.csv")
    
    @property
    def landuse_fpath(self):
        return os.path.join(self.path, "CAMELS_DK_landuse.csv")
    
    @property
    def soil_fpath(self):
        return os.path.join(self.path, "CAMELS_DK_soil.csv")
    
    @property
    def topography_fpath(self):
        return os.path.join(self.path, "CAMELS_DK_topography.csv")
    
    @property
    def signature_obs_fpath(self):
        return os.path.join(self.path, "CAMELS_DK_signature_obs_based.csv")
    
    @property
    def signature_sim_fpath(self):
        return os.path.join(self.path, "CAMELS_DK_signature_sim_based.csv")
    
    def climate_data(self):
        df = pd.read_csv(self.climate_fpath, index_col=0)
        df.index = df.index.astype(str)
        return df
    
    def geology_data(self):
        df = pd.read_csv(self.geology_fpath, index_col=0)
        df.index = df.index.astype(str)
        return df
    
    def landuse_data(self):
        df = pd.read_csv(self.landuse_fpath, index_col=0)
        df.index = df.index.astype(str)
        return df
    
    def soil_data(self):
        df = pd.read_csv(self.soil_fpath, index_col=0)
        df.index = df.index.astype(str)
        return df
    
    def topography_data(self):
        df = pd.read_csv(self.topography_fpath, index_col=0)
        df.index = df.index.astype(str)
        return df
    
    def signature_obs_data(self):
        df = pd.read_csv(self.signature_obs_fpath, index_col=0)
        df.index = df.index.astype(str)
        return df
    
    def signature_sim_data(self):
        df = pd.read_csv(self.signature_sim_fpath, index_col=0)
        df.index = df.index.astype(str)
        return df
    
    def static_data(self)->pd.DataFrame:
        """combination of topographic + soil + landuse + geology + climate features

        Returns
        -------
        pd.DataFrame
            a pandas DataFrame of static features of all catchments of shape (3330, 119)
        """
        return pd.concat([self.climate_data(),
                          self.geology_data(),
                          self.landuse_data(),
                          self.soil_data(),
                          self.topography_data()
                          ], axis=1)

    def stations(self)->List[str]:
        return [fname.split(".csv")[0].split('_')[4] for fname in os.listdir(self.gaug_catch_path)]

    def _read_csv(self, stn:str)->pd.DataFrame:
        fpath = os.path.join(self.gaug_catch_path, f"CAMELS_DK_obs_based_{stn}.csv")
        df = pd.read_csv(os.path.join(fpath), parse_dates=True, index_col='time')
        df.columns.name = 'dynamic_features'
        df.pop('catch_id')
        df = df.astype(np.float32)

        for old_name, new_name in self.dyn_map.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        return df        

    @property
    def dynamic_features(self)->List[str]:
        """returns names of dynamic features"""
        return self._dynamic_features

    @property
    def static_features(self)->List[str]:
        """returns static features for Denmark catchments"""
        return self._static_features

    @property
    def _coords_name(self)->List[str]:
        return ['catch_outlet_lat', 'catch_outlet_lon']

    @property
    def _area_name(self) ->str:
        return 'catch_area' 

    @property
    def _q_name(self)->str:
        return 'Qobs'
    
    @property
    def start(self)->pd.Timestamp:  # start of data
        return pd.Timestamp('1989-01-02 00:00:00')

    @property
    def end(self)->pd.Timestamp:  # end of data
        return pd.Timestamp('2023-12-31 00:00:00')

    def _read_dynamic_from_csv(
            self,
            stations,
            dynamic_features,
            st=None,
            en=None)->dict:

        features = check_attributes(dynamic_features, self.dynamic_features)

        dyn = {stn: self._read_csv(stn)[features] for stn in stations}

        return dyn

    def fetch_static_features(
            self,
            stn_id: Union[str, List[str]] = "all",
            features:Union[str, List[str]] = "all"
    ) -> pd.DataFrame:
        """
        Returns static features of one or more stations.

        Parameters
        ----------
            stn_id : str
                name/id of station/stations of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Returns
        -------
        pd.DataFrame
            a pandas dataframe of shape (stations, features)

        Examples
        ---------
        >>> from water_datasets import CAMELS_DK
        >>> dataset = CAMELS_DK()
        get the names of stations
        >>> stns = dataset.stations()
        >>> len(stns)
            304
        get all static data of all stations
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (304, 119)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('42600042')
        >>> static_data.shape
           (1, 119)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['slope_mean', 'aridity'])
        >>> static_data.shape
           (304, 2)
        >>> data = dataset.fetch_static_features('42600042', features=['slope_mean', 'aridity'])
        >>> data.shape
           (1, 2)

        """
        stations = check_attributes(stn_id, self.stations(), 'stations')
        features = check_attributes(features, self.static_features)
        df = self.static_data()
        return df.loc[stations, features]
    
    def transform_coords(self, coords):
        """
        Transforms the coordinates to the required format.
        """
        # from EPSG:25832 - ETRS89 / UTM zone 32N to WGS84
        return coords


class CAMELS_IND(Camels):
    """
    Dataset of 472 catchments from Republic of India following the works of 
    `Mangukiya et al., 2024 <https://doi.org/10.5194/essd-2024-379>`_. 
    The dataset consists of 210 static catchment features and 20 dynamic features.
    The dynamic features span from 19800101 to 20201231 with daily timestep.
    
    Examples
    ---------
    >>> from water_datasets import CAMELS_IND
    >>> dataset = CAMELS_IND()
    >>> data = dataset.fetch(0.1, as_dataframe=True)
    >>> data.shape
    (299520, 47)  # 47 represents number of stations
    Since data is a multi-index dataframe, we can get data of one station as below
    >>> data['17015'].unstack().shape
    (14976, 20)
    If we don't set as_dataframe=True, then the returned data will be a xarray Dataset
    >>> data = dataset.fetch(0.1)
    >>> type(data)
        xarray.core.dataset.Dataset
    >>> data.dims
    FrozenMappingWarningOnValuesAccess({'time': 14976, 'dynamic_features': 20})
    >>> len(data.data_vars)
        47
    >>> df = dataset.fetch(stations=1, as_dataframe=True)  # get data of only one random station
    >>> df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
    >>> df.shape
    (14976, 20)
    # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
    472
    # get data by station id
    >>> df = dataset.fetch(stations='3001', as_dataframe=True).unstack()
    >>> df.shape
    (14976, 20)
    # get names of available dynamic features
    >>> dataset.dynamic_features
    # get only selected dynamic features
    >>> df = dataset.fetch(1, as_dataframe=True,
    ... dynamic_features=['prcp(mm/day)', 'rel_hum(%)', 'tavg(C)', 'pet(mm/day)', 'streamflow_cms']).unstack()
    >>> df.shape
    (14976, 5)
    # get names of available static features
    >>> dataset.static_features
    # get data of 10 random stations
    >>> df = dataset.fetch(10, as_dataframe=True)
    >>> df.shape
    (299520, 10)  # remember this is multi-indexed DataFrame
    # when we get both static and dynamic data, the returned data is a dictionary
    # with ``static`` and ``dyanic`` keys.
    >>> data = dataset.fetch(stations='3001', static_features="all", as_dataframe=True)
    >>> data['static'].shape, data['dynamic'].shape
    ((1, 220), (299520, 1))
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (472, 2)
    >>> dataset.stn_coords('3001')  # returns coordinates of station whose id is 3001
        18.3861	80.3917
    >>> dataset.stn_coords(['3001', '17021'])  # returns coordinates of two stations    
    """
    url = "https://zenodo.org/records/13221214"

    def __init__(self,
                 path=None,
                 overwrite=False,
                 to_netcdf:bool = True,
                 **kwargs):
        super(CAMELS_IND, self).__init__(path=path, **kwargs)
        self._download(overwrite=overwrite)    

        names = pd.read_csv(
            os.path.join(self.static_path, "camels_India_name.txt"),
            sep=";",
            index_col=0,
            dtype={0: str}
            )
        id_str = names.index.to_list()
        id_int = names.index.astype(int).to_list()
        self.id_map = {str(k):v for k,v in zip(id_int, id_str)}

        self._static_features = self.static_data().columns.to_list()
        self._dynamic_features = self._read_dyn_csv(self.stations()[0]).columns.to_list()

        if to_netcdf:
            self._maybe_to_netcdf('camels_ind_dyn')

        self.boundary_file = os.path.join(
        self.path,
        "shapefiles_catchment",
        "Merged",
        "all_catchments.shp"
    )

        self._create_boundary_id_map(self.boundary_file, 0)

    @property
    def dyn_map(self):
        return {
        #'streamflow_cms': 'obs_q_cms',
        'tmin(C)': 'min_temp_C',
        'tmax(C)': 'max_temp_C',
        'tavg(C)': 'mean_temp_C',
        'prcp(mm/day)': 'pcp_mm',
        'rel_hum(%)': 'rh_%',
        'wind(m/s)': 'windspeed_ms'
        }

    @property
    def static_path(self)->os.PathLike:
        return os.path.join(self.path, "attributes_txt")

    @property
    def q_path(self)->os.PathLike:
        return os.path.join(self.path, "streamflow_timeseries")
    
    @property
    def forcings_path(self)->os.PathLike:
        return os.path.join(self.path, "catchment_mean_forcings")

    @property
    def dynamic_features(self)->List[str]:
        """returns names of dynamic features"""
        return self._dynamic_features

    @property
    def static_features(self)->List[str]:
        """returns static features for Denmark catchments"""
        return self._static_features

    @property
    def _coords_name(self)->List[str]:
        return ['cwc_lat', 'cwc_lon']

    @property
    def _area_name(self) ->str:
        return 'cwc_area' 

    @property
    def _q_name(self)->str:
        return 'obs_q_cms'
    
    @property
    def start(self)->pd.Timestamp:  # start of data
        return pd.Timestamp('1980-01-01')

    @property
    def end(self)->pd.Timestamp:  # end of data
        return pd.Timestamp('2020-12-31')

    def _get_map(self, sf_reader, id_index=None, name:str='')->dict:


        fieldnames = [f[0] for f in sf_reader.fields[1:]]

        if len(fieldnames) > 1:
            if id_index is None:
                raise ValueError(f"""
                more than one fileds are present in {name} shapefile 
                i.e: {fieldnames}. 
                Please provide a value for id_idx_in_{name} that must be
                less than {len(fieldnames)}
                """)
        else:
            id_index = 0

        catch_ids_map = {
            str(int(rec[id_index])): idx for idx, rec in enumerate(sf_reader.iterRecords())
        }

        return catch_ids_map

    def stn_forcing_path(self, stn:str)->os.PathLike:
        return os.path.join(
            self.forcings_path, 
            self.id_map.get(stn)[0:2], 
            f"{self.id_map.get(stn)}.csv"
            )

    def stations(self)->List[str]:
        """
        returns names of stations a list

        **Node:** 0s are omitted from the start of the station names
        which means 03001 is returned as 3001
        """
        stns = pd.read_csv(
            os.path.join(self.static_path, "camels_India_name.txt"),
            sep=";",
            index_col=0,
            dtype={0: str}
            ).index.to_list()

        return [str(int(stn)) for stn in stns]

    def static_data(self)->pd.DataFrame:
        """
        combination of topographic + soil + landuse + geology + climate + hydro 
        + climate + anthropogenic features

        Returns
        -------
        pd.DataFrame
            a pandas DataFrame of static features of all catchments of shape (3330, 119)
        """
        files = glob.glob(f"{self.static_path}/*.txt")

        dfs = []
        for f in files:
            df = pd.read_csv(f, sep=";", index_col=0)
            df.index = df.index.astype(str)
            dfs.append(df)
        
        return pd.concat(dfs, axis=1)

    def _read_q(self, stn:str = None,)->pd.DataFrame:
        """reads observed streamflow data"""
        fpath = os.path.join(self.q_path, f"streamflow_observed.csv")

        cols = ['year', 'month', 'day']
        if stn is not None:
            cols.append(stn)
        else:
            cols = None

        df = pd.read_csv(os.path.join(fpath),
                        index_col='year_month_day',
                        parse_dates=[['year', 'month', 'day']],
                        usecols = cols,
                        )

        return df.astype(np.float32)    

    def _read_forcings(self, stn:str)->pd.DataFrame:
        """reads the foring data for a given station"""
        fpath = self.stn_forcing_path(stn)
        df = pd.read_csv(fpath, 
                        index_col='year_month_day',
                        parse_dates=[['year', 'month', 'day']],                         
                        )
        return df.astype(np.float32)    

    def _read_dyn_csv(self, stn:str)->pd.DataFrame:
        """reads dynamic data for a given station"""
        q = self._read_q(stn)[stn]
        q.name = "obs_q_cms"
        df = pd.concat([self._read_forcings(stn), pd.DataFrame(q)], axis=1)
        df.columns.name = 'dynamic_features'
        df.index.name = "time"

        for old_name, new_name in self.dyn_map.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)

        return df

    def _read_dynamic_from_csv(
            self,
            stations,
            dynamic_features,
            st=None,
            en=None)->dict:

        features = check_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')
        stations = check_attributes(stations, self.stations(), 'stations')

        dyn = {stn: self._read_dyn_csv(stn)[features] for stn in stations}

        return dyn


    def fetch_static_features(
            self,
            stn_id: Union[str, List[str]] = "all",
            features:Union[str, List[str]] = "all"
    ) -> pd.DataFrame:
        """
        Returns static features of one or more stations.

        Parameters
        ----------
            stn_id : str
                name/id of station/stations of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Returns
        -------
        pd.DataFrame
            a pandas dataframe of shape (stations, features)

        Examples
        ---------
        >>> from water_datasets import CAMELS_IND
        >>> dataset = CAMELS_IND()
        get the names of stations
        >>> stns = dataset.stations()
        >>> len(stns)
            472
        get all static data of all stations
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (472, 210)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('42600042')
        >>> static_data.shape
           (1, 210)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['slope_mean', 'aridity'])
        >>> static_data.shape
           (472, 2)
        >>> data = dataset.fetch_static_features('42600042', features=['slope_mean', 'aridity'])
        >>> data.shape
           (1, 2)

        """
        stations = check_attributes(stn_id, self.stations(), 'stations')
        features = check_attributes(features, self.static_features, 'static_features')
        df = self.static_data()
        return df.loc[stations, features]
