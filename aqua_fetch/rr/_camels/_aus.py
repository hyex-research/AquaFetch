import os
from typing import List, Dict

import numpy as np
import pandas as pd

from ..utils import _RainfallRunoff
from ...utils import validate_attributes, download, unzip
from ..._backend import netCDF4
from .._map import (
    observed_streamflow_cms,
    observed_streamflow_mm,
    min_air_temp_with_specifier,
    max_air_temp_with_specifier,
    mean_air_temp_with_specifier,
    total_precipitation_with_specifier,
    actual_evapotranspiration_with_specifier,
    solar_radiation_with_specifier,
    mean_vapor_pressure_with_specifier,
    mean_rel_hum_with_specifier,
    MJ_M2_DAY_TO_WM2,
)
from .._map import catchment_area, gauge_latitude, gauge_longitude, slope

# directory separator
SEP = os.sep


class CAMELS_AUS(_RainfallRunoff):
    """
    This is a dataset of 561 Australian catchments with 187 static features and
    28 dyanmic features for each catchment. The dyanmic features are timeseries
    from 1950-01-01 to 2022-03-31. By default this class reads version 2 of CAMELS-AUS dataset
    following `Fowler et al., 2024 <https://doi.org/10.5194/essd-2024-263>`_ .

    If ``version`` is 1 then this class reads data following `Fowler et al., 2021 <https://doi.org/10.5194/essd-13-3847-2021>`_
    which is a dataset of 222 Australian catchments with 161 static features
    and 26 dyanmic features for each catchment. The dyanmic features are
    timeseries from 1957-01-01 to 2018-12-31.

    Examples
    --------
    >>> from aqua_fetch import CAMELS_AUS
    >>> dataset = CAMELS_AUS()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='912101A', as_dataframe=True)
    >>> df = dynamic['912101A'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (26388, 28)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       561
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (56 out of 561)
       56
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(26388, 28), (26388, 28), (26388, 28),... (26388, 28), (26388, 28)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('912101A', as_dataframe=True,
    ...  dynamic_features=['airtemp_C_awap_max', 'pcp_mm_awap', 'et_morton_actual_SILO', 'q_cms_obs'])
    >>> dynamic['912101A'].shape
       (26388, 4)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='912101A', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['912101A'].shape
    ((1, 187), 1, (26388, 28))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 26388, 'dynamic_features': 28})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (561, 2)
    >>> dataset.stn_coords('912101A')  # returns coordinates of station whose id is 912101A
        -38.214199	-71.8283
    >>> dataset.stn_coords(['912101A', '912105A'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('912101A')
    # get coordinates of two stations
    >>> dataset.area(['912101A', '912105A'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('912101A')
    ...
    # The version 1 can be of CAMELS_AUS can be accessed as below
    >>> dataset = CAMELS_AUS(version=1)
    >>> len(dataset.stations())
    222
    >>> _, dynamic = dataset.fetch(stations='912101A', as_dataframe=True)
    >>> dynamic['912101A'].shape
    (23376, 26)
    """

    # todo : v1 and v2 in the same folder with the same cache name, so switching 
    # version there can read the other version's files.

    url = 'https://doi.pangaea.de/10.1594/PANGAEA.921850'
    url_v2 = "https://zenodo.org/records/14289037"
    urls = {1: {
        "01_id_name_metadata.zip": "https://download.pangaea.de/dataset/921850/files/",
        "02_location_boundary_area.zip": "https://download.pangaea.de/dataset/921850/files/",
        "03_streamflow.zip": "https://download.pangaea.de/dataset/921850/files/",
        "04_attributes.zip": "https://download.pangaea.de/dataset/921850/files/",
        "05_hydrometeorology.zip": "https://download.pangaea.de/dataset/921850/files/",
        "CAMELS_AUS_Attributes&Indices_MasterTable.csv": "https://download.pangaea.de/dataset/921850/files/",
        # "Units_01_TimeseriesData.pdf": "https://download.pangaea.de/dataset/921850/files/",
        # "Units_02_AttributeMasterTable.pdf": "https://download.pangaea.de/dataset/921850/files/",
    },
        2: {
            "01_id_name_metadata.zip": "https://zenodo.org/records/14289037/files/",
            "02_location_boundary_area.zip": "https://zenodo.org/records/14289037/files/",
            "03_streamflow.zip": "https://zenodo.org/records/14289037/files/",
            "04_attributes.zip": "https://zenodo.org/records/14289037/files/",
            "05_hydrometeorology.zip": "https://zenodo.org/records/14289037/files/",
            "CAMELS_AUS_Attributes&Indices_MasterTable.csv": "https://zenodo.org/records/14289037/files/",
            "CAMELS_AUS_v2_Data_Description.pdf": "https://zenodo.org/records/14289037/files/",
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
            'streamflow_MLd': '03_streamflow',
            'streamflow_MLd_inclInfilled': '03_streamflow',
            'streamflow_mmd': '03_streamflow',

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

            # 'solarrad_AWAP': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AGCD{SEP}solarrad_AWAP',
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
            version: int = 2,
            to_netcdf: bool = True,
            overwrite: bool = False,
            verbosity: int = 1,
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
        self.version = version

        super().__init__(path=path, verbosity=verbosity, **kwargs)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        for _file, url in self.urls[version].items():
            fpath = os.path.join(self.path, _file)

            if os.path.exists(fpath) and overwrite:
                os.remove(fpath)
                if verbosity > 0: print(f"Re-downloading {_file} from {url + _file} at {fpath}")
                download(url + _file, outdir=self.path, fname=_file, verbosity=self.verbosity)

            elif not os.path.exists(fpath):
                if verbosity > 0:
                    print(f"Downloading {_file} from {url + _file} at {fpath}")
                download(url + _file, outdir=self.path, fname=_file, verbosity=self.verbosity)
            elif verbosity > 0:
                print(f"{_file} already exists at {self.path}")

        # maybe the .zip file has been downloaded previously but not unzipped
        unzip(self.path, verbosity=verbosity, overwrite=overwrite)

        if netCDF4 is None:
            to_netcdf = False

        # if to_netcdf:
        self._maybe_to_netcdf()

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(
            self.path,
            "02_location_boundary_area",
            "02_location_boundary_area",
            "shp",
            "CAMELS_AUS_Boundaries_adopted.shp" if self.version == 1 else "CAMELS_AUS_v2_Boundaries_adopted.shp"
        )

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'catchment_area': catchment_area(),
                'maen_slope_pct': slope('%'),
                'lat_outlet': gauge_latitude(),
                'long_outlet': gauge_longitude(),
        }

    @property
    def dyn_map(self):
        # table 2 in https://essd.copernicus.org/articles/13/3847/2021/#&gid=1&pid=1
        return {
            'streamflow_MLd': observed_streamflow_cms(),
            'streamflow_mmd': observed_streamflow_mm(),
            'tmin_SILO': min_air_temp_with_specifier('silo'),
            'tmax_SILO': max_air_temp_with_specifier('silo'),
            'tmin_AWAP': min_air_temp_with_specifier('awap'),
            'tmax_AWAP': max_air_temp_with_specifier('awap'),
            'et_morton_actual_SILO': actual_evapotranspiration_with_specifier('silo_morton'),
            'et_morton_point_SILO': actual_evapotranspiration_with_specifier('silo_morton_point'),
            'et_short_crop_SILO': actual_evapotranspiration_with_specifier('silo_short_crop'),
            'et_tall_crop_SILO': actual_evapotranspiration_with_specifier('silo_tall_crop'),
            'precipitation_AWAP': total_precipitation_with_specifier('awap'),
            'precipitation_SILO': total_precipitation_with_specifier('silo'),
            'solarrad_AWAP': solar_radiation_with_specifier('awap'),  # MJ/m2/day -> W/m2 in dyn_factors
            'radiation_SILO': solar_radiation_with_specifier('silo'),  # MJ/m2/day -> W/m2 in dyn_factors
            'vp_SILO': mean_vapor_pressure_with_specifier('silo'),
            'vprp_AWAP': mean_vapor_pressure_with_specifier('awap'),
            'rh_tmax_SILO': mean_rel_hum_with_specifier('silo_tmax'),
            'rh_tmin_SILO': mean_rel_hum_with_specifier('silo_tmin'),
            'vapourpres_h09_AGCD': mean_vapor_pressure_with_specifier('agcd_h09'),
            'vapourpres_h15_AGCD': mean_vapor_pressure_with_specifier('agcd_h15'),
            'tmin_AGCD': min_air_temp_with_specifier('agcd'),
            'tmax_AGCD': max_air_temp_with_specifier('agcd'),
            'precipitation_AGCD': total_precipitation_with_specifier('agcd'),
            #'mslp_SILO': mean_sea_level_pressure_with_specifier('silo'),
        }

    @property
    def dyn_factors(self):
        # Solar radiation is distributed as MJ m-2 day-1 ("Solar radiation
        # MJ m-2" in the v2 Data Description, Table `Other variables`) while the
        # canonical name promises W m-2. The AWAP series exists in v1 only; the
        # key is harmless for v2 because absent columns are skipped.
        return {
            observed_streamflow_cms(): 0.01157,
            solar_radiation_with_specifier('silo'): MJ_M2_DAY_TO_WM2,
            solar_radiation_with_specifier('awap'): MJ_M2_DAY_TO_WM2,
        }

    @property
    def dyn_generators(self):
        if self.version == 1:
            return {
    # new column to be created : function to be applied, inputs
    mean_air_temp_with_specifier('silo'): (self.mean_temp, (min_air_temp_with_specifier('silo'), max_air_temp_with_specifier('silo'))), 
    mean_air_temp_with_specifier('awap'): (self.mean_temp, (min_air_temp_with_specifier('awap'), max_air_temp_with_specifier('awap'))),
        }
        else:
            return {
    # new column to be created : function to be applied, inputs
    mean_air_temp_with_specifier('silo'): (self.mean_temp, (min_air_temp_with_specifier('silo'), max_air_temp_with_specifier('silo'))),
    mean_air_temp_with_specifier('agcd'): (self.mean_temp, (min_air_temp_with_specifier('agcd'), max_air_temp_with_specifier('agcd'))),
        }

    @property
    def start(self):
        return "19500101"

    @property
    def end(self):
        return "20181231" if self.version == 1 else "20220331"

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
    def static_features(self) -> List[str]:

        return self._static_data().columns.tolist()

    @property
    def dynamic_features(self) -> list:
        return [self.dyn_map.get(feat, feat) for feat in list(self.folders[self.version].keys())] + list(self.dyn_generators.keys())

    def _static_data(self, #stations, #features,
                     st=None, en=None):

        #features = validate_attributes(features, self.static_features, 'static_features')
        static_fname = 'CAMELS_AUS_Attributes&Indices_MasterTable.csv'
        static_fpath = os.path.join(self.path, static_fname)
        static_df = pd.read_csv(static_fpath, index_col='station_id')

        static_df.index = static_df.index.astype(str)
        #static_df = static_df.loc[stations]#[features]
        if isinstance(static_df, pd.Series):
            static_df = pd.DataFrame(static_df).transpose()
        
        static_df.rename(columns=self.static_map, inplace=True)

        return static_df

    def _read_dynamic(
            self, 
            stations, 
            dynamic_features, 
            st=None,
            en=None,
            ):

        st, en = self._check_length(st, en)
        dynamic_features = validate_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')

        dyn_attrs = {}
        dyn = {}
        dtype = {stn: np.float32 for stn in stations}
        dtype.update({'year': str, 'month': str, 'day': str})

        for _attr in list(self.folders[self.version].keys()):

            if self.dyn_map.get(_attr, _attr) not in dynamic_features:
                continue

            _path = os.path.join(self.path, f'{self.folders[self.version][_attr]}{SEP}{_attr}.csv')
            
            attr_df = pd.read_csv(_path, na_values=['-99.99'], usecols=['year', 'month', 'day'] + stations,
                                  dtype=dtype)
            attr_df.index = pd.to_datetime(attr_df[['year', 'month', 'day']])

            dyn_attrs[_attr] = attr_df[stations]

        # making one separate dataframe for one station
        for idx, stn in enumerate(stations):
            stn_df = pd.DataFrame()
            for attr, attr_df in dyn_attrs.items():
                # if attr in dynamic_features:
                stn_df[attr] = attr_df[stn]

            stn_df.rename(columns=self.dyn_map, inplace=True)

            self._apply_dyn_factors(stn_df)

            for new_col, (func, old_col) in self.dyn_generators.items():
                if isinstance(old_col, str):
                    if old_col in stn_df.columns:
                        # name of Series to func should be same as station id
                        stn_df[new_col] = func(pd.Series(stn_df[old_col], name=stn))
                else:
                    assert isinstance(old_col, tuple)
                    if all([col in stn_df.columns for col in old_col]):
                        # feed all old_cols to the function
                        stn_df[new_col] = func(*[pd.Series(stn_df[col], name=stn) for col in old_col])
            
            if self.verbosity>1 and idx % 100 == 0:
                print(f"processed {idx} stations")

            stn_df.index.name = 'time'
            stn_df.columns.name = 'dynamic_features'
            dyn[stn] = stn_df.loc[st:en, dynamic_features]

        return dyn
