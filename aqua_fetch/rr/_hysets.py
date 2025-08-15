
import os
from datetime import datetime
from typing import Union, List, Dict, Tuple

import numpy as np
import pandas as pd

try:
    from netCDF4 import Dataset
except (ModuleNotFoundError, ImportError):
    pass

from .utils import _RainfallRunoff
from .._backend import xarray as xr
from ..utils import check_attributes, download, unzip

from ._map import (
    observed_streamflow_cms,
    observed_streamflow_mmd,
    min_air_temp,
    max_air_temp,
    mean_air_temp,
    total_precipitation,
    snow_water_equivalent,
    mean_dewpoint_temperature_at_2m,
    max_air_temp_with_specifier,
    min_air_temp_with_specifier,
    u_component_of_wind_at_10m,
    v_component_of_wind_at_10m,
    mean_daily_evaporation_with_specifier,
    cloud_cover,
    downward_longwave_radiation,
    mean_thermal_radiation,
    snow_density,
    mean_daily_evaporation,
    snowfall,
    snowmelt,
    mean_air_pressure,
    solar_radiation,
    net_longwave_radiation,
    net_solar_radiation,
    )

from ._map import (
    catchment_area,
    gauge_latitude,
    gauge_longitude,
    slope
    )


class HYSETS(_RainfallRunoff):
    """
    database for hydrometeorological modeling of 14,425 North American watersheds
    from 1950-2023 following the work of `Arsenault et al., 2020 <https://doi.org/10.1038/s41597-020-00583-2>`_
    This data has 20 dynamic features and 30 static features. Most of the dynamic features
    have more than one source. The data is available in netcdf format therefore, 
    this package requires xarray and netCDF4 to be installed..

    Following data_source are available.

    +---------------+------------------------------+
    |sources        | dynamic_features             |
    +===============+==============================+
    |SNODAS_SWE     | dscharge, swe                |
    +---------------+------------------------------+
    |SCDNA          | discharge, pr, tasmin, tasmax|
    +---------------+------------------------------+
    |nonQC_stations | discharge, pr, tasmin, tasmax|
    +---------------+------------------------------+
    |Livneh         | discharge, pr, tasmin, tasmax|
    +---------------+------------------------------+
    |ERA5           | discharge, pr, tasmax, tasmin|
    +---------------+------------------------------+
    |ERAS5Land_SWE  | discharge, swe               |
    +---------------+------------------------------+
    |ERA5Land       | discharge, pr, tasmax, tasmin|
    +---------------+------------------------------+

    all sources contain one or more following dynamic_features
    with following shapes

    +----------------------------+------------------+
    |dynamic_features            |      shape       |
    +============================+==================+
    |time                        |   (25202,)       |
    +----------------------------+------------------+
    |watershedID                 |   (14425,)       |
    +----------------------------+------------------+
    |drainage_area               |   (14425,)       |
    +----------------------------+------------------+
    |drainage_area_GSIM          |   (14425,)       |
    +----------------------------+------------------+
    |flag_GSIM_boundaries        |   (14425,)       |
    +----------------------------+------------------+
    |flag_artificial_boundaries  |   (14425,)       |
    +----------------------------+------------------+
    |centroid_lat                |   (14425,)       |
    +----------------------------+------------------+
    |centroid_lon                |   (14425,)       |
    +----------------------------+------------------+
    |elevation                   |   (14425,)       |
    +----------------------------+------------------+
    |slope                       |   (14425,)       |
    +----------------------------+------------------+
    |discharge                   |   (14425, 25202) |
    +----------------------------+------------------+
    |pr                          |   (14425, 25202) |
    +----------------------------+------------------+
    |tasmax                      |   (14425, 25202) |
    +----------------------------+------------------+
    |tasmin                      |   (14425, 25202) |
    +----------------------------+------------------+

    Examples
    --------
    >>> from aqua_fetch import HYSETS
    >>> dataset = HYSETS()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='5', as_dataframe=True)
    >>> df = dynamic['5'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (27028, 20)

    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       14425
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (1442 out of 14425)
       1442

    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(27028, 20), (27028, 20), (27028, 20),... (27028, 20), (27028, 20)]

    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('5', as_dataframe=True,
    ...  dynamic_features=['evap_mm', 'pcp_mm', 'snowmelt_mm', 'swe_mm', 'q_cms_obs'])
    >>> dynamic['5'].shape
       (27028, 5)

    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10

    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='5', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['5'].shape
    ((1, 30), 1, (27028, 20))

    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset

    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 27028, 'dynamic_features': 20})

    >>> len(dynamic.data_vars)
    10

    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (14425, 2)
    >>> dataset.stn_coords('5')  # returns coordinates of station whose id is 5
        47.091389	-67.731392
    >>> dataset.stn_coords(['5', '12'])  # returns coordinates of two stations

    # get area of a single station
    >>> dataset.area('5')
    # get coordinates of two stations
    >>> dataset.area(['5', '12'])

    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('5')

    """
    doi = "https://doi.org/10.1038/s41597-020-00583-2"
    url = {
'HYSETS_watershed_boundaries.zip': 'https://osf.io/download/p8unw/',
'HYSETS_watershed_properties.txt': 'https://osf.io/download/us795/',
'HYSETS_2023_update_ERA5.nc': 'https://osf.io/download/fdnc8/',
'HYSETS_2023_update_ERA5Land.nc': 'https://osf.io/download/4vt2s/',
'HYSETS_2023_update_Livneh.nc': 'https://osf.io/download/4jgpt/',
'HYSETS_2023_update_monthly_meteorological_data.nc': 'https://osf.io/download/sc4ge/',
'HYSETS_2023_update_SNODAS.nc': 'https://osf.io/download/46wa7/',
'HYSETS_2023_update_SCDNA.nc': 'https://osf.io/download/q8za6/',
'HYSETS_2023_update_NRCAN.nc': 'https://osf.io/download/vfpre/',
'HYSETS_2023_update_nonQC_stations.nc': 'https://osf.io/download/eu8gr/',
'HYSETS_2023_update_QC_stations.nc': 'https://osf.io/download/sbfd2/',
'HYSETS_elevation_bands_100m.csv': 'https://osf.io/download/stzn7/',
'NOTES.txt': 'https://osf.io/download/cfm7q/',
    }

    sources = {
'10m_u_component_of_wind': ['ERA5', 'ERA5Land'],
'10m_v_component_of_wind': ['ERA5', 'ERA5Land'],
'2m_dewpoint': ['ERA5', 'ERA5Land'],
'2m_tasmax': ['ERA5', 'NRCAN', 'Livneh', 'QC_stations', 'nonQC_stations', 'ERA5Land', 'SCDNA'],
'2m_tasmin': ['ERA5', 'NRCAN', 'Livneh', 'QC_stations', 'nonQC_stations', 'ERA5Land', 'SCDNA'],
'discharge': ['ERA5', 'NRCAN', 'ERA5Land', 'Livneh', 'nonQC_stations', 'SCDNA', 'SNODAS', 'QC_stations'],
'evaporation': ['ERA5', 'ERA5Land'],
'snow_density': ['ERA5', 'ERA5Land'],
'snow_evaporation': ['ERA5', 'ERA5Land'],
'snow_water_equivalent': ['ERA5', 'ERA5Land'],
'snowfall': ['ERA5', 'ERA5Land'],
'snowmelt': ['ERA5', 'ERA5Land'],
'surface_downwards_solar_radiation': ['ERA5', 'ERA5Land'],
'surface_downwards_thermal_radiation': ['ERA5', 'ERA5Land'],
'surface_net_solar_radiation': ['ERA5', 'ERA5Land'],
'surface_net_thermal_radiation': ['ERA5', 'ERA5Land'],
'surface_pressure': ['ERA5', 'ERA5Land'],
'surface_runoff': ['ERA5', 'ERA5Land'],
'swe': ['SNODAS'],
'total_cloud_cover': ['ERA5'],
'total_precipitation': ['ERA5', 'NRCAN', 'Livneh', 'QC_stations', 'nonQC_stations', 'ERA5Land', 'SCDNA'],
'total_runoff': ['ERA5', 'ERA5Land'],
    }

    def_src = {
        '10m_u_component_of_wind': 'ERA5',
        '10m_v_component_of_wind': 'ERA5',
        '2m_dewpoint': 'ERA5',
        '2m_tasmax': 'ERA5',
        '2m_tasmin': 'ERA5',
        'discharge': 'ERA5',
        'evaporation': 'ERA5',
        'snow_density': 'ERA5',
        'snow_evaporation': 'ERA5',
        'snow_water_equivalent': 'ERA5',
        'snowfall': 'ERA5',
        'snowmelt': 'ERA5',
        'surface_downwards_solar_radiation': 'ERA5',
        'surface_downwards_thermal_radiation': 'ERA5',
        'surface_net_solar_radiation': 'ERA5',
        'surface_net_thermal_radiation': 'ERA5',
        'surface_pressure': 'ERA5',
        'surface_runoff': 'ERA5',
        #'swe': 'SNODAS',
        'total_cloud_cover': 'ERA5',
        'total_precipitation': 'ERA5',
        #'total_runoff': 'ERA5',
    }

    def __init__(self,
                 path: str,
                 sources:Dict[str, str] = None,
                 **kwargs
                 ):
        """
        parameters
        --------------
        path : str
            The path under which the data is to be saved or is saved already.
            If the data is alredy downloaded then provide the path under which
            HYSETS data is located. If None, then the data will be downloaded.
            The data is downloaded once and therefore susbsequent
            calls to this class will not download the data unless
            ``overwrite`` is set to True.
        sources : dict
            sources for each dynamic feature. The keys should be dynamic features
            and values should be sources. Available sources for the dynamic 
            features are as below
                
                - 10m_u_component_of_wind: ['ERA5', 'ERA5Land']
                - 10m_v_component_of_wind: ['ERA5', 'ERA5Land']
                - 2m_dewpoint: ['ERA5', 'ERA5Land']
                - 2m_tasmax: ['NRCAN', 'Livneh', 'QC_stations', 'ERA5', 'nonQC_stations', 'ERA5Land', 'SCDNA']
                - 2m_tasmin: ['NRCAN', 'Livneh', 'QC_stations', 'ERA5', 'nonQC_stations', 'ERA5Land', 'SCDNA']
                - discharge: ['NRCAN', 'ERA5', 'ERA5Land', 'Livneh', 'nonQC_stations', 'SCDNA', 'SNODAS', 'QC_stations']
                - evaporation: ['ERA5', 'ERA5Land']
                - snow_density: ['ERA5', 'ERA5Land']
                - snow_evaporation: ['ERA5', 'ERA5Land']
                - snow_water_equivalent: ['ERA5', 'ERA5Land', 'SNODAS']
                - snowfall: ['ERA5', 'ERA5Land']
                - snowmelt: ['ERA5', 'ERA5Land']
                - surface_downwards_solar_radiation: ['ERA5', 'ERA5Land']
                - surface_downwards_thermal_radiation: ['ERA5', 'ERA5Land']
                - surface_net_solar_radiation: ['ERA5', 'ERA5Land']
                - surface_net_thermal_radiation: ['ERA5', 'ERA5Land']
                - surface_pressure: ['ERA5', 'ERA5Land']
                - surface_runoff: ['ERA5', 'ERA5Land']
                - total_cloud_cover: ['ERA5']
                - total_precipitation: ['NRCAN', 'Livneh', 'QC_stations', 'ERA5', 'nonQC_stations', 'ERA5Land', 'SCDNA']

        kwargs :
            arguments for ``_RainfallRunoff`` base class

        """

        if sources is not None:
            assert isinstance(sources, dict), 'sources must be a dictionary'
            for key, val in sources.items():
                assert key in self.sources, f'{key} is not a valid source'
                assert val in self.sources[key], f'{val} is not a valid source for {key}. Available sources are {self.sources[key]}'
            self.sources = sources
        else:
            self.sources = self.def_src.copy()

        super().__init__(path=path, **kwargs)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        for fname, url in self.url.items():
            fpath = os.path.join(self.path, fname)
            if not os.path.exists(fpath):
                if self.verbosity: 
                    print(f'downloading {fname}')
                download(url, self.path, fname)

            unzip(self.path, verbosity=self.verbosity)

        self._maybe_to_netcdf()

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(self.path,  
                            "HYSETS_watershed_boundaries", 
                            "HYSETS_watershed_boundaries_20200730.shp")

    @property
    def boundary_id_map(self)->str:
        """
        Name of the attribute in the boundary (.shp/.gpkg) file that
        will be used to map the catchment/station id to the geometry of the
        catchment/station. This is used to create the boundary id map.        
        """
        return "OfficialID"

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'Drainage_Area_km2': catchment_area(), # todo: why give preference to, Drainage_Area_GSIM_km2
                'Centroid_Lat_deg_N': gauge_latitude(),
                'Slope_deg': slope('degrees'),
                'Centroid_Lon_deg_E': gauge_longitude(),
        }

    @property
    def dyn_map(self)->Dict[str, str]:
        return {
            '10m_u_component_of_wind': u_component_of_wind_at_10m(),
            '10m_v_component_of_wind': v_component_of_wind_at_10m(),
            '2m_dewpoint': mean_dewpoint_temperature_at_2m(),
            '2m_tasmax': max_air_temp_with_specifier('2m'),
            '2m_tasmin': min_air_temp_with_specifier('2m'),
            'discharge': observed_streamflow_cms(), 
            'evaporation': mean_daily_evaporation(),
            'snow_density': snow_density(),
            'snow_evaporation': mean_daily_evaporation_with_specifier('snow'),
            'snow_water_equivalent': snow_water_equivalent(),
            'snowfall': snowfall(),
            'snowmelt': snowmelt(),
            'surface_downwards_solar_radiation': solar_radiation(), # surface_downwards_solar_radiation_shortwave in J/m2
            'surface_downwards_thermal_radiation': downward_longwave_radiation(),  # surface_downwards_thermal_radiation_longwave in J/m2
            'surface_net_solar_radiation':   net_solar_radiation(), # surface_net_solar_radiation_shortwave in J/m2
            'surface_net_thermal_radiation': net_longwave_radiation(), # surface_net_thermal_radiation_longwave in J/m2
            'surface_pressure': mean_air_pressure(), # convert Pa to hPa
            'surface_runoff': observed_streamflow_mmd(),
            'total_cloud_cover': cloud_cover(),
            'total_precipitation': total_precipitation(),

            # 'total_runoff': observed_streamflow_mmd(), todo : it appears same as runoff?
        }

    @property
    def dyn_generators(self):
        return {
            # new column to be created : function to be applied, inputs
            mean_air_temp(): (self.mean_temp, (min_air_temp(), max_air_temp())),
        }
    
    @property
    def dynamic_features(self)->List[str]:
        return sorted(list(self.dyn_map.values()))

    def _maybe_to_netcdf(self):

        for src in list(set(list(self.sources.values()))):
            fname = f'HYSETS_2023_update_{src}.nc'
            outpath = os.path.join(self.path, f'HYSETS_2023_update_{src}1.nc')
            if not os.path.exists(outpath):
                self.transform(fname)
        return

    @property
    def static_features(self)->List[str]:
        df = self._static_data(nrows=2)
        return df.columns.to_list()

    def stations(self) -> List[str]:
        """
        retuns a list of station names. The ``Watershed_ID`` of the station is used
        as station name instead of ``Official_ID``. This is because in .nc files
        watershed_ID is used for stations instead of Official_ID. ``Official_ID``
        starts with 1, 2, 3 and so on while ``Watershed_ID`` is a code from
        meteo agency such as ``01AD002`` for station 1.

        Returns
        -------
        list
            a list of ids of stations

        Examples
        --------
        >>> from aqua_fetch import HYSETS
        >>> dataset = HYSETS()
        ... # get name of all stations as list
        >>> dataset.stations()

        """
        return super().stations()

    @property
    def WatershedID_OfficialID_map(self):
        """A dictionary mapping Watershed_ID to Official_ID.
        For example '01AD002': '1'
        """
        return self._static_data(
            usecols=['Watershed_ID', 'Official_ID']
            ).loc[:, 'Official_ID'].to_dict()

    @property
    def OfficialID_WatershedID_map(self):
        """A dictionary mapping Official_ID to Watershed_ID.
        For example '1': '01AD002'
        """
        s = self._static_data(usecols=['Watershed_ID', 'Official_ID'])
        return {v:k for k,v in s.loc[:, 'Official_ID'].to_dict().items()}

    @property
    def start(self)->pd.Timestamp:
        return pd.Timestamp("19500101")

    @property
    def end(self)->pd.Timestamp:
        return pd.Timestamp("20231231")

    def area(
            self,
            stations: Union[str, List[str]] = 'all',
            source:str = 'other'
    ) ->pd.Series:
        """
        Returns area_gov (Km2) of all catchments as :obj:`pandas.Series`

        parameters
        ----------
        stations : str/list
            name/names of stations. Default is None, which will return
            area of all stations
        source : str
            source of area calculation. It should be either ``gsim`` or ``other``

        Returns
        --------
        pd.Series
            a :obj:`pandas.Series` whose indices are catchment ids and values
            are areas of corresponding catchments.

        Examples
        ---------
        >>> from aqua_fetch import HYSETS
        >>> dataset = HYSETS()
        >>> dataset.area()  # returns area of all stations
        >>> dataset.area('92')  # returns area of station whose id is 912101A
        >>> dataset.area(['92', '142'])  # returns area of two stations
        """
        stations = check_attributes(stations, self.stations())

        SRC_MAP = {
            'gsim': 'Drainage_Area_GSIM_km2',
            'other': 'area_km2'
        }

        s = self.fetch_static_features(
            static_features=[SRC_MAP[source]],
        )

        s.columns = ['area_km2']
        return s.loc[stations, 'area_km2']

    def fetch_stations_features(
            self,
            stations: list,
            dynamic_features: Union[str, list, None] = 'all',
            static_features: Union[str, list, None] = None,
            st=None,
            en=None,
            as_dataframe: bool = False,
            **kwargs
              ) -> Tuple[pd.DataFrame, Union[pd.DataFrame, "Dataset"]]:
        """returns features of multiple stations
        Examples
        --------
        >>> from aqua_fetch import HYSETS
        >>> dataset = HYSETS()
        >>> stations = dataset.stations()[0:3]
        >>> features = dataset.fetch_stations_features(stations)
        """
        stations = check_attributes(stations, self.stations())
        stations_int = [int(stn) for stn in stations]

        static, dynamic = None, None

        if dynamic_features is not None:

            dynamic = self._fetch_dynamic_features(stations=stations_int,
                                               dynamic_features=dynamic_features,
                                               as_dataframe=as_dataframe,
                                               st=st,
                                               en=en,
                                               **kwargs
                                               )

            if static_features is not None:  # we want both static and dynamic
                static = self.fetch_static_features(stations,
                                                     static_features=static_features,
                                                     )

        elif static_features is not None:
            # we want only static
            static = self.fetch_static_features(
                stations,
                static_features=static_features,
            )
        else:
            raise ValueError

        return static, dynamic

    def fetch_dynamic_features(
            self,
            station,
            dynamic_features = 'all',
            st=None,
            en=None,
            as_dataframe=False
    ):
        """Fetches dynamic features of one station.

        Examples
        --------
        >>> from aqua_fetch import HYSETS
        >>> dataset = HYSETS()
        >>> dyn_features = dataset.fetch_dynamic_features('station_name')
        """
        station = [int(station)]
        return self._fetch_dynamic_features(
            stations=station,
            dynamic_features=dynamic_features,
            st=st,
            en=en,
            as_dataframe=as_dataframe
        )

    def _fetch_dynamic_features(
            self,
            stations: List[int],
            dynamic_features = 'all',
            st=None,
            en=None,
            as_dataframe=False
    ):
        """Fetches dynamic features of station."""
        # first put all dynamic features in a single Dataset of shape (time, watershed) with dynamic_features as data_vars.
        # Then converting it to (dynamic_features, time) with watershed as data_vars. This method is faster
        # for fewer dynamic features but slower for many dynamic features.

        stations_1 = np.subtract(stations, 1).astype(str).tolist()
        st, en = self._check_length(st, en)
        attrs = check_attributes(dynamic_features, self.dynamic_features)

        dyn_map_ = {v:k for k,v in self.dyn_map.items()}

        xds = None
        features = {}
        for idx, f in enumerate(attrs):
            f_ = dyn_map_[f]
            fpath = os.path.join(self.path, f'HYSETS_2023_update_{self.sources[f_]}1.nc')
            
            if xds is None or f_ not in xds.dynamic_features:
                xds = xr.open_dataset(fpath)

            features[f] = xds.sel(dynamic_features=[f_], time=slice(st, en))[stations_1]

            if self.verbosity>1:
                print(f"{idx+1}/{len(attrs)} fetched {f}")

        if self.verbosity>1: print('concatenating along features')
        xds = xr.concat(list(features.values()), dim='dynamic_features')

        if self.verbosity>1: print('transposing')

        old_vals = xds.coords["dynamic_features"].values
        new_vals = [self.dyn_map[val] for val in old_vals]

        xds = xds.assign_coords(dynamic_features=new_vals)

        # we need to add +1 to the names of data_vars
        old_vars = list(xds.data_vars)
        new_data_vars = [f"{int(var)+1}" for var in old_vars]
        data_vars_map = {old_var: new_var for old_var, new_var in zip(old_vars, new_data_vars)}
        xds = xds.rename(data_vars_map)

        if as_dataframe:
            stations = np.array(stations).astype(str).tolist()
            return {stn:xds[stn].to_pandas() for stn in stations}

        return xds

    def _static_data(self, usecols=None, nrows=None):
        """
        reads the HYSETS_watershed_properties.txt file while using `Watershed_ID`
        as index instead of ``Official_ID``. Watershed_ID starts with 1,2,3 and so on
        while ``Official_ID`` is code from meteo agency such as ``01AD002`` for station 1.
        """
        fname = os.path.join(self.path, 'HYSETS_watershed_properties.txt')
        static_df = pd.read_csv(fname, index_col='Watershed_ID', sep=',', usecols=usecols, nrows=nrows)
        static_df.index = static_df.index.astype(str)

        static_df.rename(columns=self.static_map, inplace=True)
        return static_df

    def transform(
            self,
            fname: str,
            ):

        fpath = os.path.join(self.path, fname)
        if self.verbosity: print(f'transforming {fname}')
        ds = xr.open_dataset(fpath)

        ds = ds[[var for var in ds.data_vars if len(ds[var].dims) == 2]]
        dyn_vars = list(ds.data_vars)  # e.g. ["var1", "var2", ...]

        # We'll manually combine them into a new DataArray
        arr_list = []
        for idx, var in enumerate(dyn_vars):

            array = ds[var].values
            arr_list.append(array)
            if self.verbosity>1: print(f"{idx+1}/{len(dyn_vars)} Fetched {var} {array.shape}")

        if self.verbosity: print('stacking arrays')
        data = np.stack(arr_list, axis=2)

        data_var_names = [str(i) for i in range(len(data))]

        if self.verbosity: print('creating xarray dataset')
        xds = xr.Dataset(
            {name: (["time", "dynamic_features"], data[i, :, :].astype(np.float32)) for i, name in enumerate(data_var_names)},
            coords={
                "time": ds.time,  # Replace with actual time coordinates if available
                "dynamic_features": dyn_vars  # Replace with actual feature names if available
            }
        )

        outpath = os.path.join(self.path, f"{fname.split('.')[0]}1.nc")

        if self.verbosity: print(f'saving as {outpath}')
        xds.to_netcdf(
            outpath,
            encoding={var: {"dtype": "float32", 
                            'zlib': True, 
                            'complevel': 3, 
                            'least_significant_digit': 4} for var in xds.data_vars}
            )

        return xds
