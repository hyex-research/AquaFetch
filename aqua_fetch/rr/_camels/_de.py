import os
import time
import shutil
import warnings
import functools
import concurrent.futures as cf
from typing import Union, List, Dict, Tuple

import pandas as pd

from ..utils import _RainfallRunoff
from ..._geom_utils import laea_to_wgs84, transform_geometry
from ...utils import get_cpus, download_and_unzip
from ...utils import validate_attributes, download, unzip
from ..._backend import netCDF4
from .._map import (
    observed_streamflow_cms,
    observed_streamflow_mm,
    observed_water_level_cm,
    cloud_cover,
    mean_air_temp,
    max_air_temp,
    min_air_temp,
    total_precipitation_with_specifier,
    mean_rel_hum,
    rel_hum_with_specifier,
    mean_windspeed,
    u_component_of_wind_at_10m,
    v_component_of_wind_at_10m,
    mean_air_pressure,
    solar_radiation,
    solar_radiation_with_spatial_stat,
    mean_specific_humidity,
    mean_dewpoint_temperature_at_2m,
)
from .._map import catchment_area, gauge_latitude, gauge_longitude, slope


class CAMELS_DE(_RainfallRunoff):
    """
    This is the data from 1582 German catchments following the work of
    `Loritz et al., 2024 <https://doi.org/10.5194/essd-16-5625-2024>`_ .
    The data is downloaded from `zenodo <https://zenodo.org/record/12733968>`_ .
    This data consists of 111 static and 21 dynamic features. The dynamic features
    span from 1951-01-01 to 2020-12-31 with daily timestep.

    Hourly data (CAMELS-DE-1h) is available by setting ``timestep='H'``. It has
    1611 catchments with 26 dynamic and 109 static features spanning
    2001-01-01 to 2024-12-31, following `Dolich et al., 2026
    <https://doi.org/10.5194/essd-2026-289>`_ and downloaded from
    `GFZ dataservices <https://doi.org/10.5880/fidgeo.2026.045>`_ .
    Only observed streamflow, meteorological forcing, static attributes and
    catchment boundaries are provided; the modelled benchmark simulations and
    the weather-forecast files are not downloaded. The netcdf consolidation is
    skipped by default for the hourly data (``to_netcdf`` defaults to False).

    Examples
    --------
    >>> from aqua_fetch import CAMELS_DE
    >>> dataset = CAMELS_DE()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='DE110260', as_dataframe=True)
    >>> df = dynamic['DE110260'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (25568, 21)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       1582
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (155 out of 1582)
       155
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(25568, 21), (25568, 21), (25568, 21),... (25568, 21), (25568, 21)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('DE110260', as_dataframe=True,
    ...  dynamic_features=['airtemp_C_mean', 'rh_%', 'pcp_mm_mean', 'q_cms_obs'])
    >>> dynamic['DE110260'].shape
       (25568, 4)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='DE110260', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['DE110260'].shape
    ((1, 111), 1, (25568, 21))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 25568, 'dynamic_features': 21})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (1582, 2)
    >>> dataset.stn_coords('DE110260')  # returns coordinates of station whose id is DE110260
        47.925221       8.191595
    >>> dataset.stn_coords(['DE110260', 'DE110250'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('DE110260')
    # get coordinates of two stations
    >>> dataset.area(['DE110260', 'DE110250'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('DE110260')
    ...
    # the hourly (CAMELS-DE-1h) data can be accessed by setting timestep to 'H'
    >>> dataset = CAMELS_DE(timestep='H')
    >>> len(dataset.stations())
    1611
    >>> _, dynamic = dataset.fetch(stations='DE110000', as_dataframe=True)
    >>> dynamic['DE110000'].shape
    (210383, 26)
    """
    url = "https://zenodo.org/record/16755906"

    # direct download of the hourly (CAMELS-DE-1h) archive from GFZ dataservices.
    # only the ~14 GB CAMELS-DE-1h.zip is listed here (it bundles the timeseries,
    # static attributes and shapefiles). The two weather-forecast .zarr.zip files
    # (deterministic ~15 GB and ensemble ~51 GB) which live at the same URL base
    # are deliberately NOT included so they are never downloaded.
    hourly_url = {
        "CAMELS-DE-1h.zip":
            "https://datapub.gfz.de/download/10.5880.FIDGEO.2026.045-Trfghbv/"
            "2026-045_Dolich-et-al_data/CAMELS-DE-1h.zip",
    }

    time_steps = ['D', 'H']

    def __init__(
            self,
            path=None,
            timestep: str = 'D',
            overwrite: bool = False,
            to_netcdf: bool = None,
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
        timestep : str
            possible values are ``D`` for daily (default) or ``H`` for the
            hourly (CAMELS-DE-1h) data.
        overwrite : bool
            If the data is already down then you can set it to True,
            to make a fresh download.
        to_netcdf : bool
            whether to convert all the data into one netcdf file or not.
            This will fasten repeated calls to fetch etc. but will
            require netCDF4 package as well as xarray. If not given, it
            defaults to True for the daily timestep and False for the hourly
            timestep. The hourly data is too large to consolidate into a single
            netcdf file and reading it directly from the csv files is already
            fast, so the conversion is skipped by default.
        """
        assert timestep in self.time_steps, \
            f"invalid timestep '{timestep}' given, choose from {self.time_steps}"

        if to_netcdf is None:
            # hourly data is too large for the single-file netcdf consolidation
            # and raw-csv fetching is already fast, so skip it by default.
            to_netcdf = (timestep == 'D')

        if to_netcdf and netCDF4 is None:
            warnings.warn("netCDF4 is not installed. Therefore, the data will not be converted to netcdf format.")
            to_netcdf = False

        # forward timestep and to_netcdf so the parent stores them before we use
        # them below (otherwise self.to_netcdf keeps the parent's default of True)
        super().__init__(path=path, timestep=timestep, to_netcdf=to_netcdf,
                         verbosity=verbosity, **kwargs)

        # Lazy caches for the heavy accessors. The base implementations recompute
        # these on every call, so a single bulk ``fetch`` ends up re-listing the
        # timeseries directory and re-reading a station csv hundreds of times
        # (``fetch``/``check_*`` reference ``dynamic_features`` and ``stations()``
        # once per requested station). Caching them here makes repeated fetches
        # fast without touching any values. They are populated on first use so
        # class initialization stays quick (unless it triggers the download).
        self._stations_cache = None
        self._dyn_features_cache = None
        self._static_data_cache = None

        if timestep == 'D':
            self._download_daily(overwrite=overwrite)
        else:
            # CAMELS-DE-1h corresponds to an ESSD preprint (essd-2026-289) that is
            # still under open review; the final accepted dataset may differ.
            warnings.warn(
                "CAMELS-DE-1h (timestep='H') corresponds to a preprint "
                "(https://doi.org/10.5194/essd-2026-289) which is under open "
                "review, not the final peer-reviewed dataset. Values may change "
                "in the accepted version; verify before using for critical work.",
                UserWarning,
            )
            self._download_hourly(overwrite=overwrite)

        self._maybe_to_netcdf()

        self.bbox = {'llcrnrlat': 47.0, 'urcrnrlat': 55.0, 'llcrnrlon': 5.0, 'urcrnrlon': 16.0}
        self.parallels = range(47, 55, 2)
        self.meridians = range(5, 16, 2)

    def _download_daily(self, overwrite: bool = False):
        """Downloads the daily CAMELS-DE data from zenodo.

        The guard is on the daily ``camels_de`` folder (not just a non-empty
        ``CAMELS_DE`` directory) so that the presence of the sibling hourly
        ``CAMELS-DE-1h`` folder does not fool the download check when both
        timesteps share the same ``path``. When the daily folder is missing we
        download directly (bypassing the parent-directory-non-empty check that
        the sibling hourly folder would otherwise satisfy).
        """
        if os.path.exists(self.ts_dir) and not overwrite:
            if self.verbosity:
                print(f"daily data already exists at {self._root}")
            return
        download_and_unzip(self.path, url=self.url, verbosity=self.verbosity)

    def _download_hourly(self, overwrite: bool = False):
        """Downloads and extracts only the CAMELS-DE-1h.zip archive from GFZ.

        The weather-forecast archives (the two large ``.zarr.zip`` files that
        sit at the same URL base) are never requested because ``hourly_url``
        only lists the main archive. The extracted (modelled) ``benchmark_models``
        folder is removed right after extraction since it is never read.

        If the extracted data is already present, nothing is downloaded or
        extracted (unless ``overwrite=True``).
        """
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        fname, url = next(iter(self.hourly_url.items()))
        zip_path = os.path.join(self.path, fname)

        # already have the extracted data -> nothing to do (unless overwrite)
        if os.path.exists(self.ts_dir) and not overwrite:
            return

        # download only if the archive is not already on disk (or overwrite).
        # This way, deleting the .zip after extraction does not re-trigger a
        # download as long as the extracted timeseries folder is present.
        if overwrite or not os.path.exists(zip_path):
            if self.verbosity:
                print(f"Downloading {fname} from {url}")
            download(url, outdir=self.path, fname=fname, verbosity=self.verbosity)

        # forward overwrite so that a fresh download also replaces a stale
        # extracted tree (unzip skips extraction when the target folder exists
        # unless overwrite=True is passed).
        unzip(self.path, overwrite=overwrite, verbosity=self.verbosity)

        # the extracted archive ships a modelled ``benchmark_models`` folder
        # (LSTM/HBV simulations, ~1.8 GB) which we do not expose; remove it so it
        # does not sit on disk. The ~14 GB CAMELS-DE-1h.zip is left in place (a
        # later `free_disk_space("archives")` call removes it) so that deleting it
        # does not force a re-download.
        benchmark_dir = os.path.join(self._root, 'benchmark_models')
        if os.path.exists(benchmark_dir):
            if self.verbosity:
                print(f"removing modelled benchmark data at {benchmark_dir}")
            shutil.rmtree(benchmark_dir)

    @property
    def _prefix(self) -> str:
        """filename prefix used by the dataset files for the current timestep"""
        return "CAMELS_DE_1h" if self.timestep == 'H' else "CAMELS_DE"

    @property
    def _root(self) -> os.PathLike:
        """folder into which the archive extracts for the current timestep.

        The hourly archive ``CAMELS-DE-1h.zip`` extracts into a doubly-nested
        ``CAMELS-DE-1h/CAMELS-DE-1h/`` folder which holds the timeseries, the
        attribute csvs and the catchment boundaries.
        """
        if self.timestep == 'H':
            return os.path.join(self.path, "CAMELS-DE-1h", "CAMELS-DE-1h")
        return os.path.join(self.path, "camels_de")

    @property
    def boundary_file(self) -> os.PathLike:
        if self.timestep == 'H':
            return os.path.join(self._root,
                                "CAMELS_DE_1h_catchment_boundaries",
                                "catchments",
                                "CAMELS_DE_1h_catchments.shp")
        return os.path.join(self._root,
                            "CAMELS_DE_catchment_boundaries",
                            "catchments",
                            "CAMELS_DE_catchments.shp")

    # ETRS89-LAEA (EPSG:3035), from the .prj of the daily and the hourly shapefiles;
    # laea_to_wgs84 reproduces pyproj to better than 1e-8 m on all vertices
    _ETRS89_LAEA = dict(lon_0=10.0, lat_0=52.0, false_easting=4321000.0, false_northing=3210000.0)

    def transform_boundary(self, boundary):
        """
        Transforms a catchment boundary from ETRS89-LAEA (EPSG:3035, metres) to
        WGS84 lon/lat, so that it matches the gauge coordinates.
        """
        return transform_geometry(boundary, lambda x, y: laea_to_wgs84(x, y, **self._ETRS89_LAEA))

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'area': catchment_area(),
                'slope_fdc': slope(''),
                'gauge_lat': gauge_latitude(),
                'gauge_lon': gauge_longitude(),
        }

    @property
    def dyn_map(self):
        if self.timestep == 'H':
            return self._dyn_map_hourly
        # table 1 in https://essd.copernicus.org/articles/16/5625/2024/#&gid=1&pid=1
        return {
            'discharge_vol_obs': observed_streamflow_cms(),
            'discharge_spec_obs': observed_streamflow_mm(),
            'temperature_min': min_air_temp(),
            'temperature_max': max_air_temp(),
            'temperature_mean': mean_air_temp(),
            # 'precipitation_mean': 'pcp_mm',
            'precipitation_mean': total_precipitation_with_specifier('mean'), # todo: is it mean or total?
            'precipitation_median': total_precipitation_with_specifier('median'),
            'precipitation_stdev': total_precipitation_with_specifier('std'),
            'precipitation_min': total_precipitation_with_specifier('min'),
            'precipitation_max': total_precipitation_with_specifier('max'),
            'humidity_mean': mean_rel_hum(),
            'humidity_median': rel_hum_with_specifier('med'),
            'humidity_stdev': rel_hum_with_specifier('std'),
            'humidity_min': rel_hum_with_specifier('min'),
            'humidity_max': rel_hum_with_specifier('max'),
            # 'water_level':  # observed daily water level,
            # The Data Description defines these as the "spatial mean, median,
            # minimum, maximum and standard deviation of the global radiation",
            # i.e. the spread of the gridded forcing ACROSS the catchment, not
            # a within-day range. They are therefore marked `spat`. The spatial
            # mean is what the bare canonical name already denotes, so
            # radiation_global_mean carries no token.
            'radiation_global_mean': solar_radiation(),
            'radiation_global_stdev': solar_radiation_with_spatial_stat('std'),
            'radiation_global_min': solar_radiation_with_spatial_stat('min'),
            'radiation_global_median': solar_radiation_with_spatial_stat('med'),
            'radiation_global_max': solar_radiation_with_spatial_stat('max'),
        }

    @property
    def _dyn_map_hourly(self) -> Dict[str, str]:
        # Table 1 of Dolich et al., 2026 (CAMELS-DE-1h). All observed and
        # meteorological-forcing columns are kept; those with a canonical name in
        # the aqua_fetch vocabulary are renamed here, the rest keep their original
        # dataset name (surface pressure, wind direction and the precip-coverage
        # diagnostic). The modelled/benchmark data lives in separate files/folders
        # and is never read.
        return {
            'discharge_vol_obs': observed_streamflow_cms(),
            'discharge_spec_obs': observed_streamflow_mm(),  # mm/hour
            'water_level_obs': observed_water_level_cm(),  # cm
            'precipitation_mean': total_precipitation_with_specifier('mean'),
            'precipitation_min': total_precipitation_with_specifier('min'),
            'precipitation_max': total_precipitation_with_specifier('max'),
            'precipitation_stdev': total_precipitation_with_specifier('std'),
            'precipitation_mean_gapfilled': total_precipitation_with_specifier('mean_gapfilled'),
            'precipitation_min_gapfilled': total_precipitation_with_specifier('min_gapfilled'),
            'precipitation_max_gapfilled': total_precipitation_with_specifier('max_gapfilled'),
            'precipitation_stdev_gapfilled': total_precipitation_with_specifier('std_gapfilled'),
            'air_temperature_mean': mean_air_temp(),
            'air_temperature_min': min_air_temp(),
            'air_temperature_max': max_air_temp(),
            'relative_humidity_mean': mean_rel_hum(),
            'water_vapor_mixing_ratio_mean': mean_specific_humidity(),  # g/kg
            'global_radiation_mean': solar_radiation(),
            'air_pressure_sea_level_mean': mean_air_pressure(),  # hPa
            'cloud_cover_mean': cloud_cover(),
            'dew_point_temperature_mean': mean_dewpoint_temperature_at_2m(),
            'wind_speed_eastward_mean': u_component_of_wind_at_10m(),
            'wind_speed_northward_mean': v_component_of_wind_at_10m(),
            'wind_speed_mean': mean_windspeed(),
            # kept with original names (no canonical mapping):
            #   air_pressure_surface_mean (hPa)
            #   wind_direction_mean (degrees)
            #   perc_nan_precipitation_original (% catchment w/o original radar precip)
        }

    @property
    def ts_dir(self) -> str:
        return os.path.join(self._root, 'timeseries')

    def _attr_path(self, name: str) -> str:
        """path to a static-attribute csv for the current timestep"""
        return os.path.join(self._root, f"{self._prefix}_{name}.csv")

    @property
    def clim_attr_path(self) -> str:
        return self._attr_path("climatic_attributes")

    @property
    def hum_infl_path(self) -> str:
        return self._attr_path("humaninfluence_attributes")

    @property
    def hydrogeol_attr_path(self) -> str:
        return self._attr_path("hydrogeology_attributes")

    @property
    def hydrol_attr_path(self) -> str:
        return self._attr_path("hydrologic_attributes")

    @property
    def lc_attr_path(self) -> str:
        return self._attr_path("landcover_attributes")

    @property
    def sim_attr_path(self) -> str:
        return self._attr_path("simulation_benchmark")

    @property
    def soil_attr_path(self) -> str:
        return self._attr_path("soil_attributes")

    @property
    def topo_attr_path(self) -> str:
        return self._attr_path("topographic_attributes")

    def stations(self) -> List[str]:
        # daily file: CAMELS_DE_hydromet_timeseries_<id>.csv  -> id at split index 4
        # hourly file: CAMELS_DE_1h_hydromet_timeseries_<id>.csv -> id at split index 5
        if self._stations_cache is None:
            self._stations_cache = [os.path.splitext(f)[0].split('_')[-1]
                                    for f in os.listdir(self.ts_dir)
                                    if f.endswith('.csv')]
        # return a copy so a caller's in-place edit cannot corrupt the cache
        return list(self._stations_cache)

    def clim_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self.clim_attr_path, index_col='gauge_id',
                           # dtype=np.float32
                           )

    def hum_infl_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self.hum_infl_path, index_col='gauge_id',
                           # dtype=np.float32
                           )

    def hydrogeol_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self.hydrogeol_attr_path, index_col='gauge_id',
                           # dtype=np.float32
                           )

    def hydrol_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self.hydrol_attr_path, index_col='gauge_id',  # dtype=np.float32
                           )

    def lc_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self.lc_attr_path, index_col='gauge_id',  # dtype=np.float32
                           )

    def sim_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self.sim_attr_path, index_col='gauge_id',  # dtype=np.float32
                           )

    def soil_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self.soil_attr_path, index_col='gauge_id',  # dtype=np.float32
                           )

    def topo_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self.topo_attr_path, index_col='gauge_id',  # dtype=np.float32
                           )

    def _static_data(self) -> pd.DataFrame:
        # the concatenated static table (8 csv reads) is cached because it is
        # requested repeatedly (static_features, area, stn_coords, fetch_static_*).
        # Callers only read from it (``.loc`` returns copies), so returning the
        # cached frame directly is safe and does not change any values.
        if self._static_data_cache is not None:
            return self._static_data_cache

        attrs = [
            self.clim_attrs(),
            self.hum_infl_attrs(),
            self.hydrogeol_attrs(),
            self.hydrol_attrs(),
            self.lc_attrs(),
            self.soil_attrs(),
            self.topo_attrs(),
        ]
        # the daily dataset ships a simulation_benchmark file with observed-data
        # signatures. For the hourly (CAMELS-DE-1h) dataset that file only holds
        # modelled benchmark scores (NSE_lstm, NSE_hbv, ...) which we do not make
        # available, so it is excluded from the static attributes.
        if self.timestep == 'D':
            attrs.append(self.sim_attrs())

        df = pd.concat(attrs, axis=1)

        df.rename(columns=self.static_map, inplace=True)

        self._static_data_cache = df

        return df

    def _read_stn_dyn(self, station) -> pd.DataFrame:
        """
        Reads dynamic (meteorological + streamflow) data for one catchment
        and returns as DataFrame
        """

        df = pd.read_csv(
            os.path.join(self.ts_dir, f"{self._prefix}_hydromet_timeseries_{station}.csv"),
            # sep=';',
            index_col='date',
            parse_dates=True,
            # dtype=np.float32
        )

        df.rename(columns=self.dyn_map, inplace=True)

        return df

    def _read_dynamic(
            self,
            stations,
            dynamic_features,
            st: Union[str, pd.Timestamp] = None,
            en: Union[str, pd.Timestamp] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Read dynamic data of many stations, in parallel where it pays off.

        Behaves exactly like the base implementation (same dict of
        ``{station: DataFrame}`` sliced to ``[st:en, dyn_feats]`` with the axis
        names set) but, for the parallel branch, dispatches the module-level
        :func:`_read_camels_de_dynamic` instead of the bound
        ``self._read_stn_dyn``. The station csv files here are large (the hourly
        files are ~31 MB each), so avoiding the per-task pickling of ``self`` —
        which the base incurs by handing a bound method to the pool — roughly
        halves the wall-clock of a big fetch. Values and dtypes are unchanged.
        """
        st, en = self._check_length(st, en)
        dyn_feats = validate_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')
        stations = validate_attributes(stations, self.stations(), 'stations')

        cpus = self.processes or min(get_cpus(), 16)
        start = time.time()
        if len(stations) < cpus:
            cpus = 1

        if cpus == 1:
            dyn = {}
            for idx, stn in enumerate(stations):
                stn_df = self._read_stn_dyn(stn).loc[st:en, dyn_feats]
                stn_df.columns.name = 'dynamic_features'
                stn_df.index.name = 'time'
                dyn[stn] = stn_df

                if self.verbosity and idx % 100 == 0:
                    print(f"Read {idx+1}/{len(stations)} stations.")
        else:
            # a plain function carrying only small, picklable arguments (paths,
            # the rename map and the requested window/features) — NOT the bound
            # method — so the pool does not ship a copy of ``self`` per task.
            reader = functools.partial(
                _read_camels_de_dynamic,
                self.ts_dir, self._prefix, tuple(self.dyn_map.items()),
                list(dyn_feats), st, en,
            )
            with cf.ProcessPoolExecutor(cpus) as executor:
                results = executor.map(reader, stations)

            dyn = {stn: stn_df for stn, stn_df in zip(stations, results)}

        if self.verbosity:
            total = time.time() - start
            print(f"Read {len(dyn)} stations for {len(dyn_feats)} dyn features "
                  f"in {total:.2f} seconds with {cpus} cpus.")

        return dyn

    @property
    def start(self):
        if self.timestep == 'H':
            return pd.Timestamp('2001-01-01 01:00:00')
        return pd.Timestamp('1951-01-01')

    @property
    def end(self):
        if self.timestep == 'H':
            return pd.Timestamp('2024-12-31 23:00:00')
        return pd.Timestamp('2020-12-31')

    @property
    def dynamic_features(self) -> List[str]:
        # cached: the base reads (and renames) a full station csv on every access,
        # and fetch()/check_* touch this once per requested station. The column
        # set is identical for every station, so read it once.
        if self._dyn_features_cache is None:
            self._dyn_features_cache = self._read_stn_dyn(self.stations()[0]).columns.tolist()
        # return a copy so a caller's in-place edit cannot corrupt the cache
        return list(self._dyn_features_cache)

    @property
    def static_features(self) -> List[str]:
        return self._static_data().columns.tolist()

    @property
    def _coords_name(self) -> List[str]:
        return ['gauge_lat', 'gauge_lon']

    @property
    def _area_name(self) -> str:
        return 'area'

    @property
    def _mm_feature_name(self) -> str:
        """Observed catchment-specific discharge (converted to millimetres per day
        using catchment areas"""
        return observed_streamflow_mm()


def _read_camels_de_dynamic(
        ts_dir: str,
        prefix: str,
        dyn_map_items: Tuple[Tuple[str, str], ...],
        dyn_feats: List[str],
        st: pd.Timestamp,
        en: pd.Timestamp,
        station: str,
) -> pd.DataFrame:
    """Read the dynamic timeseries of a single CAMELS-DE station.

    This is a module-level function (not a method) on purpose: it is dispatched
    to a :class:`concurrent.futures.ProcessPoolExecutor` by
    :meth:`CAMELS_DE._read_dynamic`. Handing a *bound* method to a process pool
    would pickle ``self`` — and hence every cached heavy attribute — to every
    worker on every task. Carrying only small, picklable arguments avoids that
    and roughly halves the wall-clock of a large hourly fetch. The window/feature
    slicing is done here (in the worker) so the parent does not repeat it
    serially over hundreds of stations.

    The result is byte-for-byte identical to
    ``CAMELS_DE._read_stn_dyn(station).loc[st:en, dyn_feats]`` with the axis
    names set — no values or dtypes are changed.
    """
    df = pd.read_csv(
        os.path.join(ts_dir, f"{prefix}_hydromet_timeseries_{station}.csv"),
        index_col='date',
        parse_dates=True,
    )
    df.rename(columns=dict(dyn_map_items), inplace=True)
    df = df.loc[st:en, list(dyn_feats)]
    df.columns.name = 'dynamic_features'
    df.index.name = 'time'
    return df