
__all__ = ['CAMELS_BR', 'CABra']

import os
import glob
import concurrent.futures as cf
from typing import Union, List, Dict

import numpy as np
import pandas as pd

from .utils import _RainfallRunoff
from ..utils import check_attributes, get_cpus, download, unzip
from ._map import (
    min_air_temp,
    max_air_temp,
    mean_air_temp,
    mean_air_temp_with_specifier,
    min_air_temp_with_specifier,
    max_air_temp_with_specifier,
    total_potential_evapotranspiration_with_specifier,
    actual_evapotranspiration_with_specifier,
    total_precipitation_with_specifier,
    observed_streamflow_cms,
    observed_streamflow_mm,
    mean_rel_hum_with_specifier,
    mean_windspeed_with_specifier,
    solar_radiation_with_specifier,
)

from ._map import (
    catchment_area,
    gauge_latitude,
    gauge_longitude,
    slope
    )

# directory separator
SEP = os.sep


class CAMELS_BR(_RainfallRunoff):
    """
    This is a dataset of 897 Brazilian catchments with 67 static features
    and 10 dyanmic features for each catchment. The dyanmic features are
    timeseries from 1920-01-01 to 2019-02-28. This class
    downloads and processes CAMELS dataset of Brazil as provided by
    `VP Changas et al., 2020 <https://doi.org/10.5194/essd-12-2075-2020>`_ .
    The simulated streamflow of 593 and raw streamflow of 3679 stations
    shipped with this data is not included in dynamic features. Both
    can be fetched through fetch_simulated_streamflow and fetch_raw_streamflow
    methods.

    Examples
    --------
    >>> from aqua_fetch import CAMELS_BR
    >>> dataset = CAMELS_BR()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='46035000', as_dataframe=True)
    >>> df = dynamic['46035000'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (14245, 10)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       593
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (59 out of 593)
       59
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(14245, 10), (14245, 10), (14245, 10),... (14245, 10), (14245, 10)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('46035000', as_dataframe=True,
    ...  dynamic_features=['pcp_mm_cpc', 'aet_mm_mgb', 'airtemp_C_mean', 'q_cms_obs'])
    >>> dynamic['46035000'].shape
       (14245, 4)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='46035000', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['46035000'].shape
    ((1, 67), 1, (14245, 10))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset

    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 14245, 'dynamic_features': 10})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (593, 2)
    >>> dataset.stn_coords('46035000')  # returns coordinates of station whose id is 46035000
        -12.8686	-43.3797
    >>> dataset.stn_coords(['46035000', '57170000'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('46035000')
    # get coordinates of two stations
    >>> dataset.area(['46035000', '57170000'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('46035000')

    """
    url = "https://zenodo.org/record/3964745#.YA6rUxZS-Uk"
    urls = {
        '01_CAMELS_BR_attributes.zip': 'https://zenodo.org/records/3964745/files/',
        '02_CAMELS_BR_streamflow_m3s.zip': 'https://zenodo.org/records/3964745/files/',
        '03_CAMELS_BR_streamflow_mm_selected_catchments.zip': 'https://zenodo.org/records/3964745/files/',
        '04_CAMELS_BR_streamflow_simulated.zip': 'https://zenodo.org/records/3964745/files/',
        '05_CAMELS_BR_precipitation_chirps.zip': 'https://zenodo.org/records/3964745/files/',
        '06_CAMELS_BR_precipitation_mswep.zip': 'https://zenodo.org/records/3964745/files/',
        '07_CAMELS_BR_precipitation_cpc.zip': 'https://zenodo.org/records/3964745/files/',
        '08_CAMELS_BR_evapotransp_gleam.zip': 'https://zenodo.org/records/3964745/files/',
        '09_CAMELS_BR_evapotransp_mgb.zip': 'https://zenodo.org/records/3964745/files/',
        '10_CAMELS_BR_potential_evapotransp_gleam.zip': 'https://zenodo.org/records/3964745/files/',
        '11_CAMELS_BR_temperature_min_cpc.zip': 'https://zenodo.org/records/3964745/files/',
        '12_CAMELS_BR_temperature_mean_cpc.zip': 'https://zenodo.org/records/3964745/files/',
        '13_CAMELS_BR_temperature_max_cpc.zip': 'https://zenodo.org/records/3964745/files/',
        '14_CAMELS_BR_catchment_boundaries.zip': 'https://zenodo.org/records/3964745/files/',
        '15_CAMELS_BR_gauges_location_shapefile.zip': 'https://zenodo.org/records/3964745/files/'
    }

    folders = {'streamflow_m3s_raw': '02_CAMELS_BR_streamflow_m3s',
               'streamflow_mm': '03_CAMELS_BR_streamflow_mm_selected_catchments',
               'simulated_streamflow_m3s': '04_CAMELS_BR_streamflow_simulated',
               'precipitation_cpc': '07_CAMELS_BR_precipitation_cpc',
               'precipitation_mswep': '06_CAMELS_BR_precipitation_mswep',
               'precipitation_chirps': '05_CAMELS_BR_precipitation_chirps',
               'evapotransp_gleam': '08_CAMELS_BR_evapotransp_gleam',
               'evapotransp_mgb': '09_CAMELS_BR_evapotransp_mgb',
               'potential_evapotransp_gleam': '10_CAMELS_BR_potential_evapotransp_gleam',
               'temperature_min': '11_CAMELS_BR_temperature_min_cpc',
               'temperature_mean': '12_CAMELS_BR_temperature_mean_cpc',
               'temperature_max': '13_CAMELS_BR_temperature_max_cpc'
               }

    def __init__(self, path=None, verbosity: int = 1, **kwargs):
        """
        parameters
        ----------
        path : str
            If the data is alredy downloaded then provide the complete
            path to it. If None, then the data will be downloaded.
            The data is downloaded once and therefore susbsequent
            calls to this class will not download the data unless
            ``overwrite`` is set to True.
        """
        super().__init__(path=path, name="CAMELS_BR", verbosity=verbosity, **kwargs)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        for fname, url in self.urls.items():
            fpath = os.path.join(self.path, fname)
            if not os.path.exists(fpath) or (os.path.exists(fpath) and self.overwrite):
                if self.verbosity:
                    print(f"Downloading {fname} from {url + fname} at {fpath}")
                download(url + fname, self.path, verbosity=self.verbosity)
                unzip(self.path, verbosity=self.verbosity)
            elif self.verbosity>1:
                print(f"{fpath} already exists")

        # todo : dynamic data must be stored for all stations and not only for stations which are common among all attributes
        self._maybe_to_netcdf()

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(
            self.path,
            "14_CAMELS_BR_catchment_boundaries",
            "14_CAMELS_BR_catchment_boundaries",
            "camels_br_catchments.shp"
        )

    @property
    def boundary_id_map(self) -> str:
        """
        Name of the attribute in the boundary (.shp/.gpkg) file that
        will be used to map the catchment/station id to the geometry of the
        catchment/station. This is used to create the boundary id map.
        """
        return "gauge_id"

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'area': catchment_area(),
                'slope_mean': slope('degrees'),
                'gauge_lat': gauge_latitude(),
                'gauge_lon': gauge_longitude(),
        }

    @property
    def dyn_map(self):
        # table 1 in paper
        return {
            'streamflow_mm': observed_streamflow_mm(),
            'temperature_min': min_air_temp(),
            'temperature_max': max_air_temp(),
            'temperature_mean': mean_air_temp(),
            'precipitation_mswep': total_precipitation_with_specifier('mswep'),
            'precipitation_chirps': total_precipitation_with_specifier('chirps'),
            'precipitation_cpc': total_precipitation_with_specifier('cpc'),
            'potential_evapotransp_gleam': total_potential_evapotranspiration_with_specifier('gleam'),
            'evapotransp_gleam': actual_evapotranspiration_with_specifier('gleam'),
            'evapotransp_mgb': actual_evapotranspiration_with_specifier('mgb'),
        }

    @property
    def dyn_generators(self):
        return {
            # new column to be created : function to be applied, inputs
            observed_streamflow_cms(): (self.mm_to_cms, observed_streamflow_mm()),
            #mean_air_temp(): (self.mean_temp, (min_air_temp(), max_air_temp())),
        }

    @property
    def _all_dirs(self):
        """All the folders in the dataset_directory"""
        return [f for f in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, f))]

    @property
    def static_dir(self):
        path = None
        for _dir in self._all_dirs:
            if "attributes" in _dir:
                # supposing that 'attributes' axist in only one file/folder in self.path
                path = os.path.join(self.path, f'{_dir}{SEP}{_dir}')
        return path

    @property
    def static_files(self):
        all_files = None
        if self.static_dir is not None:
            all_files = glob.glob(f"{self.static_dir}/*.txt")
        return all_files

    @property
    def dynamic_features(self) -> List[str]:
        features = list(CAMELS_BR.folders.keys())
        features.remove('simulated_streamflow_m3s')  # todo: why we need to remove this?
        features.remove('streamflow_m3s_raw')
        return [self.dyn_map.get(feature, feature) for feature in features] + list(self.dyn_generators.keys())

    @property
    def static_attribute_categories(self):
        static_attrs = []
        for f in self.static_files:
            ff = str(os.path.basename(f).split('.txt')[0])
            static_attrs.append('_'.join(ff.split('_')[2:]))
        return static_attrs

    @property
    def static_features(self):
        static_fpath = os.path.join(self.path, 'static_features.csv')
        if not os.path.exists(static_fpath):
            files = glob.glob(
                f"{os.path.join(self.path, '01_CAMELS_BR_attributes', '01_CAMELS_BR_attributes')}/*.txt")
            cols = []
            for f in files:
                _df = pd.read_csv(f, sep=' ', index_col='gauge_id', nrows=1)
                cols += list(_df.columns)
        else:
            df = pd.read_csv(static_fpath, index_col='gauge_id', nrows=1)
            cols = list(df.columns)

        return [self.static_map.get(feat, feat) for feat in cols]

    @property
    def start(self):
        return "19200601"

    @property
    def end(self):
        return "20190228"

    def q_mm(
            self,
            stations: Union[str, List[str]] = "all"
    ) -> pd.DataFrame:
        """
        returns streamflow in the units of milimeter per day. he name of
        original timeseries is ``streamflow_mm``.

        parameters
        ----------
        stations : str/list
            name/names of stations. Default is None, which will return
            area of all stations

        Returns
        --------
        pd.DataFrame
            a :obj:`pandas.DataFrame` whose indices are time-steps and columns
            are catchment/station ids.

        """
        # todo: better avoid remove this method since parent class has it

        stations = check_attributes(stations, self.stations())
        _, q = self.fetch_stations_features(stations,
                                         dynamic_features=observed_streamflow_mm(),
                                         as_dataframe=True)
        #q.index = q.index.get_level_values(0)
        q = pd.DataFrame.from_dict({stn:df[observed_streamflow_mm()] for stn,df in q.items()})
        # area_m2 = self.area(stations) * 1e6  # area in m2
        # q = (q / area_m2) * 86400  # cms to m/day
        return q  # * 1e3  # to mm/day

    def area(
            self,
            stations: Union[str, List[str]] = "all",
            source: str = "gsim",
    ) -> pd.Series:
        """
        Returns area (Km2) of all catchments as :obj:`pandas.Series`

        parameters
        ----------
        stations : str/list
            name/names of stations. Default is None, which will return
            area of all stations
        source : str
            source of area calculation. It should be either ``gsim`` or ``ana``

        Returns
        --------
        pd.Series
            a :obj:`pandas.Series` whose indices are catchment ids and values
            are areas of corresponding catchments.

        Examples
        ---------
        >>> from aqua_fetch import CAMELS_BR
        >>> dataset = CAMELS_BR()
        >>> dataset.area()  # returns area of all stations
        >>> dataset.stn_coords('65100000')  # returns area of station whose id is 912101A
        >>> dataset.stn_coords(['65100000', '64075000'])  # returns area of two stations
        """

        SRC_MAP = {
            'gsim': 'area_gsim',
            'ana': 'area_ana'
        }
        stations = check_attributes(stations, self.stations(), 'stations')

        fpath = os.path.join(self.path, '01_CAMELS_BR_attributes',
                             '01_CAMELS_BR_attributes',
                             'camels_br_location.txt')
        df = pd.read_csv(fpath, sep=' ')
        df.index = df['gauge_id'].astype(str)
        s = df[SRC_MAP[source]]

        s.name = 'area_km2'
        return s.loc[stations]

    def stn_coords(
            self,
            stations: Union[str, List[str]] = 'all'
    ) -> pd.DataFrame:
        """
        returns coordinates of stations as :obj:`pandas.DataFrame`
        with ``long`` and ``lat`` as columns.

        Parameters
        ----------
        stations :
            name/names of stations. If not given, coordinates
            of all stations will be returned.

        Returns
        -------
        pd.DataFrame
            :obj:`pandas.DataFrame` with ``long`` and ``lat`` columns.
            The length of dataframe will be equal to number of stations
            wholse coordinates are to be fetched.

        Examples
        --------
        >>> dataset = CAMELS_BR()
        >>> dataset.stn_coords() # returns coordinates of all stations
        >>> dataset.stn_coords('65100000')  # returns coordinates of station whose id is 912101A
        >>> dataset.stn_coords(['65100000', '64075000'])  # returns coordinates of two stations
        """
        fpath = os.path.join(self.path, '01_CAMELS_BR_attributes',
                             '01_CAMELS_BR_attributes',
                             'camels_br_location.txt')
        df = pd.read_csv(fpath, sep=' ')
        df.index = df['gauge_id'].astype(str)
        df = df[['gauge_lat', 'gauge_lon']]
        df.columns = ['lat', 'long']
        stations = check_attributes(stations, self.stations(), 'stations')

        return df.loc[stations, :]

    def all_stations(self, feature: str) -> List[str]:
        """Tells all station ids for which a data of a specific attribute is available."""
        p = self.folders[feature]
        return [f.split('_')[0] for f in os.listdir(os.path.join(self.path, p, p))]

    def stations(
            self,
    ) -> List[str]:
        """
        Returns a list of station ids.

        Example
        -------
        >>> dataset = CAMELS_BR()
        >>> stations = dataset.stations()
        """
        return self.all_stations('streamflow_mm')

    def fetch_raw_streamflow(
            self,
            stations: str = None
    ) -> pd.DataFrame:
        """
        returns raw streamflow data for one or more stations.

        Example
        -------
        >>> dataset = CAMELS_BR()
        >>> data = dataset.fetch_raw_streamflow('10500000')
        ... # fetch all time series data associated with a station.
        >>> x = dataset.fetch_raw_streamflow(dataset.all_stations())

        """

        if stations is None:
            stations = self.all_stations('streamflow_m3s_raw')

        if not isinstance(stations, list):
            stations = [stations]

        raw_q = []
        for stn in stations:
            self._read_dynamic_feature('streamflow_m3s_raw', stn)
        return pd.concat(raw_q, axis=1)

    def fetch_simulated_streamflow(
            self,
            stations: str = None
    ) -> pd.DataFrame:
        """
        returns raw streamflow data for one or more stations.

        Example
        -------
        >>> dataset = CAMELS_BR()
        >>> data = dataset.fetch_simulated_streamflow('10500000')
        ... # fetch all time series data associated with a station.
        >>> x = dataset.fetch_simulated_streamflow(dataset.all_stations())

        """

        if stations is None:
            stations = self.all_stations('simulated_streamflow_m3s')

        if not isinstance(stations, list):
            stations = [stations]

        raw_q = []
        for stn in stations:
            self._read_dynamic_feature('simulated_streamflow_m3s', stn)
        return pd.concat(raw_q, axis=1)

    def _read_dynamic(
            self,
            stations,
            attributes: Union[str, list] = 'all',
            st=None,
            en=None,
    ) -> Dict[str, pd.DataFrame]:
        """
        returns the dynamic/time series attribute/attributes for one station id.

        Example
        -------
        >>> dataset = CAMELS_BR()
        >>> pcp = dataset.fetch_dynamic_features('10500000', 'precipitation_cpc')
        ... # fetch all time series data associated with a station.
        >>> x = dataset.fetch_dynamic_features('51560000', dataset.dynamic_features)

        """

        features = check_attributes(attributes, self.dynamic_features, 'dynamic_features')

        st, en = self._check_length(st, en)

        cpus = self.processes or min(get_cpus(), 64)

        dyn = {}

        if cpus == 1:
            for idx, station in enumerate(stations):
                # making one separate dataframe for one station
                dyn[station] = self.get_dynamic_features(station, features).loc[st:en]

                if idx % 20 == 0:
                    print(f"completed {idx} stations")
        else:

            if self.verbosity > 0:
                print(f"getting data for {len(stations)} stations using {cpus} cpus")

            features = [features for _ in range(len(stations))]
            with cf.ProcessPoolExecutor(cpus) as executor:
                results = executor.map(self.get_dynamic_features, stations, features)

            for station, stn_df in zip(stations, results):
                dyn[station] = stn_df.loc[st:en]

            if self.verbosity > 1:
                print(f"completed fetching data for {len(stations)} stations")

        return dyn

    def get_dynamic_features(self, station, features, st=None, en=None):
        feature_dfs = []
        for feature, path in self.folders.items():
            if feature in ['simulated_streamflow_m3s', 'streamflow_m3s_raw']:
                continue
            feature_df = self._read_dynamic_feature(path, feature=feature, station=station, st=st, en=en)
            feature_dfs.append(feature_df)

        stn_df = pd.concat(feature_dfs, axis=1)

        for new_col, (func, old_col) in self.dyn_generators.items():
            if isinstance(old_col, str):
                if old_col in stn_df.columns:
                    # name of Series to func should be same as station id
                    stn_df[new_col] = func(pd.Series(stn_df[old_col], name=station))
            else:
                assert isinstance(old_col, tuple)
                if all([col in stn_df.columns for col in old_col]):
                    # feed all old_cols to the function
                    stn_df[new_col] = func(*[pd.Series(stn_df[col], name=station) for col in old_col])

        stn_df.columns.name = 'dynamic_features'
        stn_df.index.name = 'time'
        return stn_df

    def _read_dynamic_feature(self, folder, feature, station, st=None, en=None):
        path = os.path.join(self.path, f'{folder}{SEP}{folder}')
        # supposing that the filename starts with stn_id and has .txt extension.
        fname = [f for f in os.listdir(path) if f.startswith(str(station)) and f.endswith('.txt')]
        assert len(fname) == 1, f"{len(fname)} {station} in {folder} for {feature}"
        fname = fname[0]
        if os.path.exists(os.path.join(path, fname)):
            df = pd.read_csv(os.path.join(path, fname), sep=' ')
            df.index = pd.to_datetime(df[['year', 'month', 'day']])
            df = df.drop(['year', 'month', 'day'], axis=1)

            df = pd.DataFrame(df.loc[st:en, feature])

            df.rename(columns = self.dyn_map, inplace=True)

        else:
            raise FileNotFoundError(f"file {fname} not found at {path}")

        return df.astype(np.float32)

    def _static_data(self)->pd.DataFrame:

        static_fpath = os.path.join(self.path, 'static_features.csv')
        if not os.path.exists(static_fpath):
            files = glob.glob(
                f"{os.path.join(self.path, '01_CAMELS_BR_attributes', '01_CAMELS_BR_attributes')}/*.txt")
            static_df = pd.DataFrame()
            for f in files:
                _df = pd.read_csv(f, sep=' ', index_col='gauge_id')
                static_df = pd.concat([static_df, _df], axis=1)
            static_df.to_csv(static_fpath, index_label='gauge_id')
        else:
            static_df = pd.read_csv(static_fpath, index_col='gauge_id')

        static_df.index = static_df.index.astype(str)

        static_df.rename(columns = self.static_map, inplace=True)

        return static_df


class CABra(_RainfallRunoff):
    """
    Reads and fetches CABra dataset which is catchment attribute dataset
    following the work of `Almagro et al., 2021 <https://doi.org/10.5194/hess-25-3105-2021>`_
    This dataset consists of 87 static and 13 dynamic features of 735 Brazilian
    catchments. The temporal extent is from 1980 to 2020. The dyanmic features
    consist of daily hydro-meteorological time series

    Examples
    ---------
    >>> from aqua_fetch import CABra
    >>> dataset = CABra()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='92', as_dataframe=True)
    >>> df = dynamic['92'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (10956, 13)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       735
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (73 out of 735)
       73
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(10956, 13), (10956, 13), (10956, 13),... (10956, 13), (10956, 13)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('92', as_dataframe=True,
    ...  dynamic_features=['pcp_mm_ens', 'airtemp_C_ens_max', 'pet_mm_pm', 'rh_%_ens', 'q_cms_obs'])
    >>> dynamic['92'].shape
       (10956, 4)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='92', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['92'].shape
    ((1, 87), 1, (10956, 13))

    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 10956, 'dynamic_features': 13})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (735, 2)
    >>> dataset.stn_coords('92')  # returns coordinates of station whose id is 92
        -2.509	-47.764
    >>> dataset.stn_coords(['92', '5'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('92')
    # get coordinates of two stations
    >>> dataset.area(['92', '5'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('92')

    """

    url = 'https://zenodo.org/record/7612350'

    def __init__(self,
                 path=None,
                 overwrite=False,
                 to_netcdf: bool = True,
                 met_src: str = 'ens',
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
        met_src : str
            source of meteorological data, must be one of
            ``ens``, ``era5`` or ``ref``.
        """
        super(CABra, self).__init__(path=path,
                                    to_netcdf=to_netcdf,
                                    **kwargs)
        self.path = path
        self.met_src = met_src
        self._download(overwrite=overwrite)

        self._dynamic_features = self.__dynamic_features()
        self._static_features = self.__static_features()

        self._maybe_to_netcdf()

    @property
    def dyn_fname(self) -> Union[str, os.PathLike]:
        """
        name of the .nc file which contains dynamic features. This file is created during dataset initialization
        only if to_netcdf is True and xarray is installed and the file does not already exists. The creation of this
        file can take some time however it leads to faster I/O operations.
        """
        return self.name.lower() + f"_{self.timestep}_{self.met_src}.nc"

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(self.path, "CABra_boundaries", "CABra_boundaries.shp")

    @property
    def boundary_id_map(self) -> str:
        """
        Name of the attribute in the boundary (.shp/.gpkg) file that
        will be used to map the catchment/station id to the geometry of the
        catchment/station. This is used to create the boundary id map.
        """
        return "ID_CABra"
    
    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'catch_area': catchment_area(),
                'catch_slope': slope('perc'),
                'latitude': gauge_latitude(),
                'longitude': gauge_longitude(),
        }

    @property
    def dyn_map(self):
        # table 3 in the paper https://hess.copernicus.org/articles/25/3105/2021/#&gid=1&pid=1
        return {
            'Streamflow': observed_streamflow_cms(),
            'tmin_ens': min_air_temp_with_specifier('ens'),
            'tmax_ens': max_air_temp_with_specifier('ens'),
            'tmin_era5': min_air_temp_with_specifier('era5'),
            'tmax_era5': max_air_temp_with_specifier('era5'),
            'tmin_ref': min_air_temp_with_specifier('ref'),
            'tmax_ref': max_air_temp_with_specifier('ref'),
            'p_ens': total_precipitation_with_specifier('ens'),
            'p_ref': total_precipitation_with_specifier('ref'),
            'p_era5': total_precipitation_with_specifier('era5'),
            'rh_ens': mean_rel_hum_with_specifier('ens'),
            'rh_era5': mean_rel_hum_with_specifier('era5'),
            'rh_ref': mean_rel_hum_with_specifier('ref'),
            'wnd_ens': mean_windspeed_with_specifier('ens'),
            'wnd_era5': mean_windspeed_with_specifier('era5'),
            'wnd_ref': mean_windspeed_with_specifier('ref'),
            'et_ens': actual_evapotranspiration_with_specifier('ens'),
            'pet_pm': total_potential_evapotranspiration_with_specifier('pm'),
            'pet_pt': total_potential_evapotranspiration_with_specifier('pt'),
            'pet_hg': total_potential_evapotranspiration_with_specifier('hg'),
            'srad_ens': solar_radiation_with_specifier('ens'),  # todo: change units from MJ/m2/day to W/m2
            'srad_era5': solar_radiation_with_specifier('era5'),
            'srad_ref': solar_radiation_with_specifier('ref'),
        }

    @property
    def dyn_generators(self):
        return {
# new column to be created : function to be applied, inputs
mean_air_temp_with_specifier(self.met_src): (self.mean_temp, (min_air_temp_with_specifier(self.met_src), max_air_temp_with_specifier(self.met_src))),
#mean_air_temp_with_specifier('era5'): (self.mean_temp, (min_air_temp_with_specifier('era5'), max_air_temp_with_specifier('era5'))),
#mean_air_temp_with_specifier('ref'): (self.mean_temp, (min_air_temp_with_specifier('ref'), max_air_temp_with_specifier('ref'))),
        }

    @property
    def q_path(self):
        return os.path.join(self.path, "CABra_streamflow_daily_series",
                            "CABra_daily_streamflow")

    @property
    def attr_path(self):
        return os.path.join(self.path, 'CABra_attributes', 'CABra_attributes')

    @property
    def dynamic_features(self) -> List[str]:
        return self._dynamic_features

    def __dynamic_features(self) -> List[str]:
        stn = self.stations()[0]
        df = pd.concat([self._read_q_from_csv(stn), self._read_meteo_from_csv(stn, self.met_src)], axis=1)
        cols = df.columns.to_list()
        cols = [col for col in cols if col not in ['Year', 'Month', 'Day']]
        return cols

    @property
    def static_features(self) -> List[str]:
        """names of static features"""
        return self._static_features

    def __static_features(self) -> List[str]:
        df = pd.concat(
            [
                self.climate_attrs(),
                self.general_attrs(),
                self.geology_attrs(),
                self.gw_attrs(),
                self.hydro_distrub_attrs(),
                self.lc_attrs(),
                self.soil_attrs(),
                self.q_attrs(),
                self.topology_attrs()], axis=1)
        
        # drop duplicate columns from df which might have come due to concatenation
        df = df.loc[:, ~df.columns.duplicated()]

        df.rename(columns = self.static_map, inplace=True)
        return df.columns.to_list()

    def stations(self) -> List[str]:
        return self.add_attrs().index.astype(str).to_list()

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp("1980-10-01")

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp("2010-09-30")

    def add_attrs(self) -> pd.DataFrame:
        """
        Returns additional catchment attributes
        """
        fpath = os.path.join(self.attr_path, "CABra_additional_attributes.txt")

        dtypes = {"CABra_ID": int,  # todo shouldn't it be str?
                  "ANA_ID": int,
                  "longitude_centroid": np.float32,
                  "latitude_centroid": np.float32,
                  "dist_coast": np.float32}

        add_attributes = pd.read_csv(fpath, sep='\t',
                                     names=list(dtypes.keys()),
                                     dtype=dtypes,
                                     header=4)
        add_attributes.index = add_attributes.pop('CABra_ID')
        return add_attributes

    def climate_attrs(self) -> pd.DataFrame:
        """
        returns climate attributes for all catchments
        """
        fpath = os.path.join(self.attr_path,
                             "CABra_climate_attributes.txt")

        dtypes = {"CABra_ID": int,  # todo shouldn't it be str?
                  "ANA_ID": int,
                  "clim_p": np.float32,
                  "clim_tmin": np.float32,
                  "clim_tmax": np.float32,
                  "clim_rh": np.float32,
                  "clim_wind": np.float32,
                  "clim_srad": np.float32,
                  "clim_et": np.float32,
                  "clim_pet": np.float32,
                  "aridity_index": np.float32,
                  "p_seasonality": np.float32,
                  "clim_quality": int,
                  }

        clim_attrs = pd.read_csv(fpath, sep='\t',
                                 names=list(dtypes.keys()),
                                 dtype=dtypes,
                                 encoding_errors='ignore',
                                 header=6)
        clim_attrs.index = clim_attrs.pop('CABra_ID')
        return clim_attrs

    def general_attrs(self) -> pd.DataFrame:
        """
        returns general attributes for all catchments
        """
        fpath = os.path.join(self.attr_path,
                             "CABra_general_attributes.txt")

        dtypes = {"CABra_ID": int,  # todo shouldn't it be str?
                  "ANA_ID": int,
                  "longitude": np.float32,
                  "latitude": np.float32,
                  "gauge_hreg": str,
                  "gauge_biome": str,
                  "gauge_state": str,
                  "missing_data": np.float32,
                  "series_length": np.float32,
                  "quality_index": np.float32,
                  }

        gen_attrs = pd.read_csv(fpath,
                                sep='\t',
                                names=list(dtypes.keys()),
                                dtype=dtypes,
                                encoding_errors='ignore',
                                header=6)
        gen_attrs.index = gen_attrs.pop('CABra_ID')
        return gen_attrs

    def geology_attrs(self) -> pd.DataFrame:
        """
        returns geological attributes for all catchments
        """
        fpath = os.path.join(self.attr_path,
                             "CABra_geology_attributes.txt")

        dtypes = {"CABra_ID": int,
                  "ANA_ID": int,
                  "catch_lith": str,
                  "sub_porosity": np.float32,
                  "sub_permeability": np.float32,
                  "sub_hconduc": np.float32,
                  }

        gen_attrs = pd.read_csv(fpath,
                                sep='\t',
                                names=list(dtypes.keys()),
                                dtype=dtypes,
                                encoding_errors='ignore',
                                header=6)
        gen_attrs.index = gen_attrs.pop('CABra_ID')
        return gen_attrs

    def gw_attrs(self) -> pd.DataFrame:
        """
        returns groundwater attributes for all catchments


        """
        fpath = os.path.join(self.attr_path,
                             "CABra_groundwater_attributes.txt")

        dtypes = {"CABra_ID": int,
                  "ANA_ID": int,
                  "aquif_name": str,
                  "aquif_type": str,
                  "catch_wtd": np.float32,
                  "catch_hand": np.float32,
                  "hand_class": str,
                  "well_number": int,
                  "well_static": str,
                  "well_dynamic": str,
                  }

        gen_attrs = pd.read_csv(fpath,
                                sep='\t',
                                names=list(dtypes.keys()),
                                dtype=dtypes,
                                encoding_errors='ignore',
                                header=7)
        gen_attrs.index = gen_attrs.pop('CABra_ID')
        return gen_attrs

    def hydro_distrub_attrs(self) -> pd.DataFrame:
        """
        returns geological attributes for all catchments
        """
        fpath = os.path.join(self.attr_path,
                             "CABra_hydrologic_disturbance_attributes.txt")

        dtypes = {"CABra_ID": int,
                  "ANA_ID": int,
                  "dist_urban": int,
                  "cover_urban": np.float32,
                  "cover_crops": np.float32,
                  "res_number": int,
                  "res_area": np.float32,
                  "res_volume": np.float32,
                  "res_regulation": np.float32,
                  "water_demand": int,
                  "hdisturb_index": np.float32,
                  }

        gen_attrs = pd.read_csv(fpath,
                                sep='\t',
                                names=list(dtypes.keys()),
                                dtype=dtypes,
                                encoding_errors='ignore',
                                header=8)
        gen_attrs.index = gen_attrs.pop('CABra_ID')
        return gen_attrs

    def lc_attrs(self) -> pd.DataFrame:
        """
        returns land cover attributes for all catchments
        """
        fpath = os.path.join(self.attr_path,
                             "CABra_land-cover_attributes.txt")

        dtypes = {"CABra_ID": int,
                  "ANA_ID": int,
                  "cover_main": str,
                  "cover_bare": np.float32,
                  "cover_forest": np.float32,
                  "cover_crops": np.float32,
                  "cover_grass": np.float32,
                  "cover_moss": np.float32,
                  "cover_shrub": np.float32,
                  "cover_urban": np.float32,
                  "cover_snow": np.float32,
                  "cover_waterp": np.float32,
                  "cover_waters": np.float32,
                  "ndvi_djf": np.float32,
                  "ndvi_mam": np.float32,
                  "ndvi_jja": np.float32,
                  "ndvi_son": np.float32,
                  }

        lc_attrs = pd.read_csv(fpath,
                               sep='\t',
                               names=list(dtypes.keys()),
                               dtype=dtypes,
                               encoding_errors='ignore',
                               header=6)
        lc_attrs.index = lc_attrs.pop('CABra_ID')
        return lc_attrs

    def soil_attrs(self) -> pd.DataFrame:
        """
        returns soil attributes for all catchments
        """
        fpath = os.path.join(self.attr_path,
                             "CABra_soil_attributes.txt")

        dtypes = {"CABra_ID": int,
                  "ANA_ID": int,
                  "soil_type": str,
                  "soil_textclass": str,
                  "soil_sand": np.float32,
                  "soil_silt": np.float32,
                  "soil_clay": np.float32,
                  "soil_carbon": np.float32,
                  "soil_bulk": np.float32,
                  "soil_depth": np.float32,
                  }

        soil_attrs = pd.read_csv(fpath,
                                 sep='\t',
                                 names=list(dtypes.keys()),
                                 dtype=dtypes,
                                 encoding_errors='ignore',
                                 header=7)
        soil_attrs.index = soil_attrs.pop('CABra_ID')
        return soil_attrs

    def q_attrs(self) -> pd.DataFrame:
        """
        returns streamflow attributes for all catchments
        """
        fpath = os.path.join(self.attr_path,
                             "CABra_streamflow_attributes.txt")

        dtypes = {"CABra_ID": int,
                  "ANA_ID": int,
                  "q_mean": np.float32,
                  "q_1": np.float32,
                  "q_5": np.float32,
                  "q_95": np.float32,
                  "q_99": np.float32,
                  "q_lf": np.float32,
                  "q_ld": np.float32,
                  "q_hf": np.float32,
                  "q_hd": np.float32,
                  "q_hfd": np.float32,
                  "q_zero": int,
                  "q_cv": np.float32,
                  "q_lcv": np.float32,
                  "q_hcv": np.float32,
                  "q_elasticity": np.float32,
                  "fdc_slope": np.float32,
                  "baseflow_index": np.float32,
                  'runoff_coef': np.float32
                  }
        names = list(dtypes.keys())
        dtypes.pop('q_cv')
        dtypes.pop('q_mean')
        dtypes.pop('q_lcv')
        dtypes.pop('fdc_slope')
        q_attrs = pd.read_csv(fpath,
                              sep='\t',
                              names=names,
                              dtype=dtypes,
                              encoding_errors='ignore',
                              header=7)

        q_attrs.index = q_attrs.pop('CABra_ID')
        return q_attrs

    def topology_attrs(self) -> pd.DataFrame:
        """
        returns topology attributes for all catchments
        """
        fpath = os.path.join(self.attr_path,
                             "CABra_topography_attributes.txt")

        dtypes = {"CABra_ID": int,
                  "ANA_ID": int,
                  "catch_area": np.float32,
                  "elev_mean": np.float32,
                  "elev_min": np.float32,
                  "elev_max": np.float32,
                  "elev_gauge": np.float32,
                  "catch_slope": np.float32,
                  "catch_order": int,
                  }

        gen_attrs = pd.read_csv(fpath,
                                sep='\t',
                                names=list(dtypes.keys()),
                                dtype=dtypes,
                                encoding_errors='ignore',
                                header=7)
        gen_attrs.index = gen_attrs.pop('CABra_ID')
        return gen_attrs

    def _read_q_from_csv(self, station: str) -> pd.DataFrame:
        q_fpath = os.path.join(self.q_path, f"CABra_{station}_streamflow.txt")

        df = pd.read_csv(q_fpath, sep='\t',
                         header=8,
                         names=['Year', 'Month', 'Day', 'Streamflow', 'Quality'],
                         dtype={'Year': np.int16,
                                'Month': np.int16,
                                'Day': np.int16,
                                # 'Streamflow': np.float32,
                                'Quality': np.int16}
                         )
        df.rename(columns=self.dyn_map, inplace=True)
        df[observed_streamflow_cms()] = df[observed_streamflow_cms()].astype(np.float32)
        return df

    def _read_meteo_from_csv(
            self,
            station: str,
            source="ens") -> pd.DataFrame:

        meteo_path = os.path.join(self.path,
                                  'CABra_climate_daily_series',
                                  'climate_daily',
                                  source
                                  )
        meteo_fpath = os.path.join(meteo_path,
                                   f"CABra_{station}_climate_{source.upper()}.txt")

        dtypes = {"Year": int,
                  "Month": int,
                  "Day": int,
                  f"p_{source}": np.float32,
                  f"tmin_{source}": np.float32,
                  f"tmax_{source}": np.float32,
                  f"rh_{source}": np.float32,
                  f"wnd_{source}": np.float32,
                  f"srad_{source}": np.float32,
                  f"et_{source}": np.float32,
                  "pet_pm": np.float32,
                  "pet_pt": np.float32,
                  "pet_hg": np.float32}

        if source == "ref" and station in [
            '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '15', '17', '18', '19', '27', '28', '34', '526',
            '564', '567', '569'
        ]:
            df = pd.DataFrame(columns=list(dtypes.keys()))
        else:
            df = pd.read_csv(meteo_fpath,
                             sep="\t",
                             names=list(dtypes.keys()),
                             dtype=dtypes,
                             header=12)

        df.rename(columns=self.dyn_map, inplace=True)

        for new_col, (func, old_col) in self.dyn_generators.items():
            if isinstance(old_col, str):
                if old_col in df.columns:
                    # name of Series to func should be same as station id
                    df[new_col] = func(pd.Series(df[old_col], name=station))
            else:
                assert isinstance(old_col, tuple)
                if all([col in df.columns for col in old_col]):
                    # feed all old_cols to the function
                    df[new_col] = func(*[pd.Series(df[col], name=station) for col in old_col])
        return df

    def _static_data(self)->pd.DataFrame:
        df = pd.concat([self.climate_attrs(),
                        self.general_attrs(),
                        self.geology_attrs(),
                        self.gw_attrs(),
                        self.hydro_distrub_attrs(),
                        self.lc_attrs(),
                        self.soil_attrs(),
                        self.q_attrs(),
                        self.topology_attrs()], axis=1)

        df.index = df.index.astype(str)
        # drop duplicate columns
        df = df.loc[:, ~df.columns.duplicated()].copy()

        df.rename(columns=self.static_map, inplace=True)

        return df

    def _read_dynamic(
            self,
            stations,
            dynamic_features,
            st=None,
            en=None
    ) -> dict:

        st, en = self._check_length(st, en)
        features = check_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')

        if self.verbosity>1:
            print(f"getting data for {len(dynamic_features)} and for {len(stations)} stations")

        # qs and meteo data has different index

        if self.verbosity>2:
            print("getting streamflow data")

        qs = [self._read_q_from_csv(station=station) for station in stations]
        q_idx = pd.to_datetime(
            qs[0]['Year'].astype(str) + '-' + qs[0]['Month'].astype(str) + '-' + qs[0]['Day'].astype(str))

        if self.verbosity>2:
            print("getting meteo data")

        meteos = [
            self._read_meteo_from_csv(station=station, source=self.met_src) for station in stations]
        # todo : this will be correct only if we are getting data for all stations
        # but what if we want to get data for some random stations?
        # 10 because first 10 stations don't have data for "ref" source
        if len(meteos) < 10:
            met_idx = pd.to_datetime(
                meteos[0]['Year'].astype(str) + '-' + meteos[0]['Month'].astype(str) + '-' + meteos[0]['Day'].astype(
                    str))
        else:
            met_idx = pd.to_datetime(
                meteos[10]['Year'].astype(str) + '-' + meteos[10]['Month'].astype(str) + '-' + meteos[10]['Day'].astype(
                    str))

        met_cols = [col for col in meteos[0].columns if col not in ['Year', 'Month', 'Day']]

        if self.verbosity>2:
            print("putting data in dictionary")

        dyn = {}

        for stn, q, meteo in zip(self.stations(), qs, meteos):

            if len(meteo) == 0:
                meteo = pd.DataFrame(meteo, index=met_idx)
            else:
                meteo.index = met_idx
            q.index = q_idx

            stn_df = pd.concat(
                [meteo[met_cols].astype(np.float32), q[['Quality', observed_streamflow_cms()]]], axis=1)[features]

            stn_df.index.name = 'time'
            stn_df.columns.name = 'dynamic_features'

            dyn[stn] = stn_df.loc[st:en]

        return dyn
