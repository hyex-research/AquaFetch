
import os
import warnings
from typing import Union, List, Dict, Tuple

import numpy as np
import pandas as pd

try:
    from netCDF4 import Dataset, num2date
except (ModuleNotFoundError, ImportError):
    pass

from .utils import _RainfallRunoff
from .._backend import xarray as xr
from ..utils import validate_attributes, download, unzip

from ._map import (
    observed_streamflow_cms,
    observed_streamflow_mm,
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
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       14425
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (1442 out of 14425)
       1442
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(27028, 20), (27028, 20), (27028, 20),... (27028, 20), (27028, 20)]
    ...
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
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='5', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['5'].shape
    ((1, 30), 1, (27028, 20))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 27028, 'dynamic_features': 20})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (14425, 2)
    >>> dataset.stn_coords('5')  # returns coordinates of station whose id is 5
        47.091389	-67.731392
    >>> dataset.stn_coords(['5', '12'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('5')
    # get coordinates of two stations
    >>> dataset.area(['5', '12'])
    ...
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

    # least-significant-digit the transformed cache is written with. The bulk
    # single-feature reader re-applies it to data read from the original
    # per-feature files so the returned values are byte-identical to the cache.
    LSD = 4

    # a single dynamic feature requested for at least this many stations is read
    # from the original per-feature file (one contiguous ``(watershed, time)``
    # read) instead of decompressing every station's full multi-feature chunk in
    # the transformed cache. Below this count the transformed per-station read is
    # faster, so the cache is used. Correctness is identical either way; this is
    # purely a speed heuristic around the layout cross-over.
    _orig_read_min_stns = 2000

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

        # lazily-populated caches. The heavy static table, the station list and
        # the (open) netCDF handles are loaded on first use and reused afterwards
        # instead of being re-read on every call.
        self._static_df = None
        self._stations_cache = None
        self._nc_handles = {}
        self._src_meta = {}
        # handles/time-index for the original per-feature files (bulk reader)
        self._orig_handles = {}
        self._orig_meta = {}

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        for fname, url in self.url.items():
            fpath = os.path.join(self.path, fname)
            if not os.path.exists(fpath):
                if self.verbosity:
                    print(f'downloading {fname}')
                download(url, self.path, fname)

        # unzip once after the download loop; ``unzip`` is idempotent (it skips
        # archives whose extracted directory already exists) so a single call
        # is equivalent to calling it inside the loop, only much cheaper.
        unzip(self.path, verbosity=self.verbosity)

        self._maybe_to_netcdf()

        self.bbox = {"llcrnrlat": 18, "urcrnrlat": 80.384358,
                     "llcrnrlon": -168.0,  "urcrnrlon": -55.0}
        self.parallels = np.arange(18, 80, 7)
        self.meridians = np.arange(-168, -55, 12)

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
            'surface_runoff': observed_streamflow_mm(),
            'total_cloud_cover': cloud_cover(),
            'total_precipitation': total_precipitation(),

            # 'total_runoff': observed_streamflow_mm(), todo : it appears same as runoff?
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
        if self._stations_cache is None:
            self._stations_cache = super().stations()
        # return a copy so a caller's in-place edit cannot corrupt the cache
        return list(self._stations_cache)

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

    def usgs_stations(self)->List[str]:
        """Returns the Watershed_IDs of (12004) stations which are taken from USGS as list"""
        return self._stations_from_source('USGS')

    def canada_stations(self)->List[str]:
        """Returns the Watershed_IDs of (2375) stations which are taken from Canada (HYDAT) as list"""
        return self._stations_from_source('HYDAT')

    def mexico_stations(self)->List[str]:
        """Returns the Watershed_IDs of (46) stations which are taken from Mexico as list"""
        return self._stations_from_source('Mexico')

    def _stations_from_source(self, source: str) -> List[str]:
        """Watershed_IDs (as strings) of the stations whose ``Source`` is ``source``.
        Uses the cached static table so the 3.4 MB properties file is read only once."""
        df = self._static_data()
        # the index of the cached static table is already ``Watershed_ID`` as str
        return df.index[df['Source'] == source].tolist()

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
        stations = validate_attributes(stations, self.stations())

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

        if xr is None:
            if not as_dataframe:
                if self.verbosity: warnings.warn("xarray module is not installed so as_dataframe will have no effect. "
                              "Dynamic features will be returned as pandas DataFrame")
                as_dataframe = True

        stations = validate_attributes(stations, self.stations())
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

    def _src_reader(self, src: str) -> Tuple["Dataset", pd.DatetimeIndex, Dict[str, int]]:
        """
        Opens (once, then caches) the transformed netCDF file of a ``source`` and
        returns the open :obj:`netCDF4.Dataset` handle together with the decoded
        ``time`` index and a ``{original_feature_name: column_position}`` map.

        The transformed file (``HYSETS_2023_update_{src}1.nc``) stores one
        variable *per station* of shape ``(time, dynamic_features)``, so reading a
        station is a single contiguous read of that variable. Opening the file
        (its header describes >14000 variables) is comparatively expensive, hence
        the handle is kept open and reused across ``fetch`` calls.
        """
        cached = self._src_meta.get(src)
        if cached is not None:
            return self._nc_handles[src], cached[0], cached[1]

        fpath = os.path.join(self.path, f'HYSETS_2023_update_{src}1.nc')
        nc = Dataset(fpath, "r")

        # decode the CF time axis with netCDF4 so its ``units`` and ``calendar``
        # are honoured exactly. For these files (standard calendar, "days since
        # 1950-01-01") this matches xarray's decoding byte-for-byte, while being
        # cheap and one-time because the handle and metadata are cached. A
        # non-standard calendar would yield cftime objects and raise here rather
        # than silently producing wrong dates.
        tvar = nc.variables['time']
        tindex = pd.DatetimeIndex(num2date(
            tvar[:],
            units=tvar.units,
            calendar=getattr(tvar, 'calendar', 'standard'),
            only_use_cftime_datetimes=False,
        ))

        feat_names = [str(f) for f in nc.variables['dynamic_features'][:]]
        feat_pos = {f: i for i, f in enumerate(feat_names)}

        self._nc_handles[src] = nc
        self._src_meta[src] = (tindex, feat_pos)
        return nc, tindex, feat_pos

    def close(self):
        """
        Closes any netCDF file handles cached by :meth:`_src_reader`. The handles
        are otherwise released when the instance is garbage-collected; call this
        for deterministic cleanup (e.g. when creating many instances). Subsequent
        ``fetch`` calls transparently re-open the files.
        """
        for handles in (self._nc_handles, self._orig_handles):
            for nc in handles.values():
                try:
                    nc.close()
                except Exception:
                    pass
            handles.clear()
        self._src_meta.clear()
        self._orig_meta.clear()

    def _fetch_dynamic_features(
            self,
            stations: List[int],
            dynamic_features = 'all',
            st=None,
            en=None,
            as_dataframe=False
    ):
        """Fetches dynamic features of one or more stations.

        The transformed netCDF stores each station as a single variable of shape
        ``(time, dynamic_features)``. We therefore read each requested station's
        variable directly (one contiguous read), slice the requested time range
        and select the requested feature columns. This avoids opening/indexing an
        xarray ``Dataset`` that carries a data variable for every one of the
        >14000 stations, which dominated the previous implementation's runtime.
        """
        st, en = self._check_length(st, en)
        attrs = validate_attributes(dynamic_features, self.dynamic_features)

        # standardized feature name -> original name used inside the netCDF file
        std_to_orig = {v: k for k, v in self.dyn_map.items()}
        srcs = {self.sources[std_to_orig[a]] for a in attrs}

        # output data-variable names are the 1-based station ids (as strings);
        # inside the file a station lives in variable str(id - 1).
        stn_names = [str(s) for s in stations]

        if len(srcs) > 1:
            # rare: the requested features come from more than one source file
            # (each with its own time axis). Fall back to the general, xarray
            # based combination so that time alignment stays correct.
            return self._fetch_multisource(
                stations, stn_names, attrs, std_to_orig, st, en, as_dataframe)

        src = next(iter(srcs))

        # Fast path: a single feature for many stations is much cheaper to read
        # from the original per-feature file (one contiguous (watershed, time)
        # variable) than to decompress every station's full multi-feature chunk
        # in the transformed cache. It only triggers above a station-count
        # threshold, and only if the original file is present (a user may have
        # deleted it to save space). Values are made byte-identical to the cache
        # by re-applying the cache's least-significant-digit quantization.
        if (len(attrs) == 1
                and len(stations) >= self._orig_read_min_stns
                and os.path.exists(
                    os.path.join(self.path, f'HYSETS_2023_update_{src}.nc'))):
            return self._fetch_feature_bulk(
                stations, stn_names, attrs[0], std_to_orig, src, st, en, as_dataframe)

        nc, tindex, feat_pos = self._src_reader(src)

        # positions of the time slice (inclusive of both ends, like label slicing)
        t_lo = int(tindex.searchsorted(st, 'left'))
        t_hi = int(tindex.searchsorted(en, 'right'))
        time_sel = tindex[t_lo:t_hi]

        # column positions of the requested features, in the requested order
        cols = np.array([feat_pos[std_to_orig[a]] for a in attrs])

        # Read the station variables in ascending file order. A random set of
        # stations is read ~2x faster sorted than in arbitrary order because the
        # accesses become monotonic in the file. Output order is kept identical
        # to the requested order by assembling below from ``stn_names``.
        data_vars = {}
        for k in sorted(range(len(stations)), key=lambda i: stations[i]):
            stn = stations[k]
            arr = np.asarray(nc.variables[str(stn - 1)][t_lo:t_hi, :])[:, cols]
            data_vars[stn_names[k]] = arr

        if self.verbosity > 1:
            print(f"fetched {len(attrs)} dynamic features for {len(stations)} stations")

        if as_dataframe:
            out = {}
            for name in stn_names:
                df = pd.DataFrame(data_vars[name], index=time_sel, columns=attrs)
                df.index.name = 'time'
                df.columns.name = 'dynamic_features'
                out[name] = df
            return out

        return xr.Dataset(
            {name: (["time", "dynamic_features"], data_vars[name]) for name in stn_names},
            coords={"time": time_sel, "dynamic_features": attrs},
        )

    def _fetch_multisource(
            self, stations, stn_names, attrs, std_to_orig, st, en, as_dataframe):
        """General path for when requested features span multiple source files.

        Builds one single-feature :obj:`xarray.Dataset` per feature (reading the
        stations directly from each source's transformed file) and concatenates
        them along ``dynamic_features``. xarray aligns the differing per-source
        time axes, mirroring the previous implementation's semantics.
        """
        feats = []
        for a in attrs:
            orig = std_to_orig[a]
            nc, tindex, feat_pos = self._src_reader(self.sources[orig])
            t_lo = int(tindex.searchsorted(st, 'left'))
            t_hi = int(tindex.searchsorted(en, 'right'))
            time_sel = tindex[t_lo:t_hi]
            col = feat_pos[orig]
            dvars = {}
            for name, stn in zip(stn_names, stations):
                arr = np.asarray(nc.variables[str(stn - 1)][t_lo:t_hi, col:col + 1])
                dvars[name] = (["time", "dynamic_features"], arr)
            feats.append(xr.Dataset(
                dvars, coords={"time": time_sel, "dynamic_features": [a]}))

        xds = xr.concat(feats, dim='dynamic_features')

        if as_dataframe:
            return {name: xds[name].to_pandas() for name in stn_names}
        return xds

    def _orig_reader(self, src: str) -> Tuple["Dataset", pd.DatetimeIndex]:
        """
        Opens (once, then caches) the *original* per-feature netCDF file of a
        source (``HYSETS_2023_update_{src}.nc``, layout ``(watershed, time)``)
        and returns the open handle together with its decoded time index. Used by
        the single-feature bulk reader.
        """
        cached = self._orig_meta.get(src)
        if cached is not None:
            return self._orig_handles[src], cached

        fpath = os.path.join(self.path, f'HYSETS_2023_update_{src}.nc')
        nc = Dataset(fpath, "r")
        tvar = nc.variables['time']
        tindex = pd.DatetimeIndex(num2date(
            tvar[:],
            units=tvar.units,
            calendar=getattr(tvar, 'calendar', 'standard'),
            only_use_cftime_datetimes=False,
        ))
        self._orig_handles[src] = nc
        self._orig_meta[src] = tindex
        return nc, tindex

    @staticmethod
    def _quantize(data: np.ndarray, lsd: int) -> np.ndarray:
        """
        Applies netCDF4's ``least_significant_digit`` quantization (the exact
        transformation the transformed cache was written with) so that values
        read from the original per-feature files are byte-identical to the cache.
        Returns ``float32`` (the cache's dtype). NaNs are preserved.
        """
        # a python-float scale keeps the arithmetic in float32 (NEP 50) — the
        # same result as a float64 scale but ~2x faster and half the memory.
        scale = float(2.0 ** np.ceil(np.log2(10.0 ** lsd)))
        return (np.around(data * scale) / scale).astype(np.float32, copy=False)

    def _fetch_feature_bulk(
            self, stations, stn_names, std_feat, std_to_orig, src, st, en, as_dataframe):
        """
        Reads a *single* dynamic feature for many stations from the original
        per-feature file. That file stores the feature as one contiguous
        ``(watershed, time)`` variable, so this is far cheaper than decompressing
        every station's full multi-feature chunk in the transformed cache. The
        cache's ``least_significant_digit`` quantization is re-applied so the
        returned values are identical to the transformed-cache path.
        """
        orig_name = std_to_orig[std_feat]
        nc, tindex = self._orig_reader(src)
        t_lo = int(tindex.searchsorted(st, 'left'))
        t_hi = int(tindex.searchsorted(en, 'right'))
        time_sel = tindex[t_lo:t_hi]

        # Read the whole (watershed, time) block for the time slice, then subset
        # the requested watersheds in memory. netCDF fancy row-indexing on this
        # chunked layout is pathologically slow, so it is never done on the file.
        block = np.asarray(nc.variables[orig_name][:, t_lo:t_hi])
        rows = [s - 1 for s in stations]
        block = self._quantize(block[rows, :], self.LSD)

        if self.verbosity > 1:
            print(f"fetched {std_feat} for {len(stations)} stations from {src} "
                  f"original file (bulk)")

        if as_dataframe:
            out = {}
            for i, name in enumerate(stn_names):
                df = pd.DataFrame(block[i][:, None], index=time_sel, columns=[std_feat])
                df.index.name = 'time'
                df.columns.name = 'dynamic_features'
                out[name] = df
            return out

        return xr.Dataset(
            {name: (["time", "dynamic_features"], block[i][:, None])
             for i, name in enumerate(stn_names)},
            coords={"time": time_sel, "dynamic_features": [std_feat]},
        )

    def _static_data(self, usecols=None, nrows=None):
        """
        reads the HYSETS_watershed_properties.txt file while using `Watershed_ID`
        as index instead of ``Official_ID``. Watershed_ID starts with 1,2,3 and so on
        while ``Official_ID`` is code from meteo agency such as ``01AD002`` for station 1.

        The full table is read from disk only once and cached; subsequent calls
        (including the ``usecols``/``nrows`` variants) are served from the cache.
        """
        if self._static_df is None:
            fname = os.path.join(self.path, 'HYSETS_watershed_properties.txt')
            static_df = pd.read_csv(fname, index_col='Watershed_ID', sep=',')
            static_df.index = static_df.index.astype(str)
            static_df.rename(columns=self.static_map, inplace=True)
            self._static_df = static_df

        df = self._static_df
        if usecols is not None:
            df = df.loc[:, [c for c in usecols if c in df.columns]]
        if nrows is not None:
            df = df.head(nrows)
        # return a copy so a caller's in-place edit cannot corrupt the cache
        return df.copy()

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
                            'least_significant_digit': self.LSD} for var in xds.data_vars}
            )

        return xds
