import os
import warnings
import functools
from typing import Union, List, Dict, Tuple

import numpy as np
import pandas as pd

from ..utils import _RainfallRunoff, cache_name
from ..._backend import netCDF4
from .._map import (
    observed_streamflow_cms,
    observed_streamflow_mm,
    mean_air_temp_with_specifier,
    total_precipitation_with_specifier,
    total_potential_evapotranspiration_with_specifier,
    mean_rel_hum,
    mean_windspeed,
    mean_specific_humidity,
    soil_moisture_layer1,
    soil_moisture_layer2,
    soil_moisture_layer3,
    soil_moisture_layer4,
)
from .._map import (
    catchment_area,
    gauge_latitude,
    gauge_longitude,
    slope,
    catchment_elevation_meters,
    urban_fraction,
    grass_fraction,
    crop_fraction,
    catchment_perimeter,
)
from ._common import _remove_stale, _extract_zip


class CAMELS_LUX(_RainfallRunoff):
    """
    Hydro-meteorological time series and catchment attributes of 56 partly
    nested stream gauges in and around Luxembourg following
    `Nijzink et al. <https://doi.org/10.5194/essd-2024-482>`_. Release 2.1 is
    downloaded as two zips (872 MB and 9 MB) from
    `Zenodo <https://zenodo.org/records/18776538>`_.

    The data is served at the timestep given to ``timestep``: ``D`` (6209 steps
    from 2004-11-01 to 2021-10-31), ``H`` (149016 steps) or ``15Min`` (596061
    steps, both from 2004-11-01 01:00 to 2021-11-01 00:00). Timestamps are in
    UTC+1 and mark the end of each accumulation interval.

    Each gauge has 26 dynamic features: observed streamflow in m3/s and in
    mm/timestep with its interpolation flag ``Qflag``, radar precipitation with
    its 5-minute minimum and maximum and its gap-filling flag ``RR_flag_rad``,
    station and ERA5 precipitation, ERA5 (``airtemp_C_mean_era5``) and
    station-interpolated (``airtemp_C_mean_station``) air temperature, Oudin and
    Penman-Monteith potential evapotranspiration, nine ERA5 thunderstorm
    parameters (``cape``, ``cin``, ``kx``, ``tcwv``, specific and relative
    humidity, wind speed and low- and deep-layer wind shear) and ERA5-Land soil
    moisture at four depths. Ten gauges (ID_12, 20, 31, 36, 37, 38, 46, 54, 55
    and 56) start one or two years late; their files begin there and
    :meth:`fetch` pads them with NaN up to the dataset's first date. Units are
    the published ones except specific humidity, published in kg/kg and served
    in g/kg as ``spechum_gkg`` promises.

    The 61 static features are the five attribute files of the release (meta,
    climatic, geologic, land use and topographic). The land use attributes are
    converted from the published percent to a fraction, the rest keep their
    published units. Catchment boundaries are shipped in WGS84, so
    :meth:`get_boundary` returns degrees without reprojection.

    Release 1.1, which this class read before, is superseded: release 2.1 revised
    the values (of one gauge's daily streamflow 10 % of the days differ, and every
    ERA5 series was recomputed), added station air temperature, renamed the time
    series files and dropped ``basin_id.csv``. A copy of release 1.1 in ``path``,
    and the netCDF caches built from it, are deleted and release 2.1 is
    downloaded once in their place.

    Timings on a 48-core machine: the first initialization downloads the two
    zips (881 MB), extracts them (6.0 GB) and builds the 72 MB daily netCDF
    cache in 453 s. Afterwards initialization takes 0.001 s and all 56 gauges
    with all features are fetched in 0.09 s from that cache and in 0.23 s from
    the csv files (0.62 s with ``processes=1``). The hourly and 15 minute data
    read in 2.8 s and 15.3 s from their csv files; their caches, built in 4 s
    and 31 s, are 1.74 GB and 6.95 GB and serve the same fetch in 0.36 s and
    0.99 s.

    Examples
    ---------
    >>> from aqua_fetch import CAMELS_LUX
    >>> dataset = CAMELS_LUX()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='ID_02', as_dataframe=True)
    >>> df = dynamic['ID_02'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (6209, 26)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       56
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (5)
       5
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(6209, 26), (6209, 26), (6209, 26), (6209, 26), (6209, 26)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('ID_02', as_dataframe=True,
    ...  dynamic_features=['pcp_mm_station', 'rh_%', 'airtemp_C_mean_era5', 'pet_mm_pm', 'q_cms_obs'])
    >>> dynamic['ID_02'].shape
       (6209, 5)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='ID_02', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['ID_02'].shape
    ((1, 61), 1, (6209, 26))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)
    xarray.core.dataset.Dataset
    ...
    >>> dict(dynamic.sizes)
    {'time': 6209, 'dynamic_features': 26}
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (56, 2)
    >>> dataset.stn_coords('ID_02')  # returns coordinates of station whose id is ID_02
                    lat     long
    gauge_id
    ID_02     49.586288  6.14908
    >>> dataset.stn_coords(['ID_02', 'ID_01'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('ID_02')
    gauge_id
    ID_02    317.779999
    Name: area_km2, dtype: float32
    # get area of two stations
    >>> dataset.area(['ID_02', 'ID_01'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> boundary = dataset.get_boundary('ID_02')
    >>> boundary.type
    'Polygon'
    >>> boundary.coordinates[0][0]   # (long, lat) in degrees
    (5.974566720413799, 49.378485550719546)
    ...
    # if we want to get hourly data we can do as below
    >>> dataset = CAMELS_LUX(timestep='H')
    >>> _, dynamic = dataset.fetch(stations='ID_02', as_dataframe=True)
    >>> dynamic['ID_02'].shape
    (149016, 26)
    ...
    # if we want to get 15Min data we can do as below
    >>> dataset = CAMELS_LUX(timestep='15Min')
    >>> _, dynamic = dataset.fetch(stations='ID_02', as_dataframe=True)
    >>> dynamic['ID_02'].shape
    (596061, 26)
    """

    url = "https://zenodo.org/records/18776538"

    # release read by this class. It is part of the netCDF cache name so that a
    # cache built from release 1.1 (camels_lux_D.nc, camels_lux_D_v2.nc), whose
    # feature names and values differ, is never served as this release.
    release = "2.1"

    time_steps = ['D', 'H', '15Min']

    # (archive in the zenodo record, folder it is extracted into)
    _archives = (("CAMELS-LUX.zip", "CAMELS-LUX"),
                 ("CAMELS-LUX_shapefiles.zip", "CAMELS-LUX_shapefiles"))

    _attr_files = (
        "CAMELS_LUX_meta_attributes.csv",
        "CAMELS_LUX_climatic_attributes.csv",
        "CAMELS_LUX_geologic_attributes.csv",
        "CAMELS_LUX_landuse_attributes.csv",
        "CAMELS_LUX_topographic_attributes.csv",
    )

    # shipped only by release 1.1, which listed the gauge ids in it, so its
    # presence tells the two releases apart without reading anything. A time
    # series file is not used as the marker because free_disk_space("redundant")
    # deletes those from a complete release 2.1 as well.
    _old_release_marker = "basin_id.csv"

    # folder and file name token of each timestep. The daily files carry a
    # double underscore (CAMELS_LUX_hydromet_timeseries__daily_ID_01.csv) and
    # the sub-hourly folder is "15min"; the dataset description states a single
    # underscore and "15Min", so the archive is followed, not the description.
    _ts_layout = {'D': ('daily', '_daily'),
                  'H': ('hourly', 'hourly'),
                  '15Min': ('15min', '15min')}

    # cached tables which are not worth shipping to a process pool worker
    _NOT_PICKLED = ('_static_df', 'bndry_id_map_')

    def __init__(self,
                 path=None,
                 timestep: str = 'D',
                 overwrite: bool = False,
                 to_netcdf: bool = True,
                 **kwargs):
        """
        Parameters
        ----------
        path : str
            directory under which the data is (or will be) saved in a
            ``CAMELS_LUX`` folder. All three timesteps share it. If None, the
            default data directory of aqua_fetch is used.
        timestep : str
            ``D`` (default), ``H`` or ``15Min``. Each timestep is cached
            separately.
        overwrite : bool
            if True, the archives, the extracted files and the netCDF cache of
            this timestep are deleted and downloaded again.
        to_netcdf : bool
            whether to save the dynamic data of this timestep in a netCDF cache
            for faster reading. Requires netCDF4 and xarray.
        **kwargs :
            any keyword argument of :py:class:`aqua_fetch.rr._RainfallRunoff`
            such as ``processes``, ``verbosity`` or ``remove_zip``.
        """
        if timestep not in self.time_steps:
            raise ValueError(f"timestep must be one of {self.time_steps}, not {timestep!r}")

        super(CAMELS_LUX, self).__init__(
            path=path,
            timestep=timestep,
            overwrite=overwrite,
            to_netcdf=to_netcdf,
            **kwargs)

        self._download_camels_lux(overwrite=overwrite)

        self._check_manifest()

        self._maybe_to_netcdf()

    # ------------------------------------------------------------------ paths

    @property
    def _root(self) -> Union[str, os.PathLike]:
        """folder that CAMELS-LUX.zip is extracted into"""
        return os.path.join(self.path, self._archives[0][1])

    @property
    def _boundary_dir(self) -> Union[str, os.PathLike]:
        """folder that CAMELS-LUX_shapefiles.zip is extracted into"""
        return os.path.join(self.path, self._archives[1][1])

    @property
    def ts_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self._root, "timeseries")

    def _ts_dir(self, timestep: str) -> Union[str, os.PathLike]:
        """folder with the time series files of ``timestep``"""
        return os.path.join(self.ts_path, self._ts_layout[timestep][0])

    def _ts_fname(self, stn: str, timestep: str = None) -> str:
        """name of the time series file of gauge ``stn`` at ``timestep``"""
        return (f"CAMELS_LUX_hydromet_timeseries_"
                f"{self._ts_layout[timestep or self.timestep][1]}_{stn}.csv")

    @property
    def daily_ts_path(self) -> Union[str, os.PathLike]:
        return self._ts_dir('D')

    @property
    def hourly_ts_path(self) -> Union[str, os.PathLike]:
        return self._ts_dir('H')

    @property
    def subhourly_ts_path(self) -> Union[str, os.PathLike]:
        return self._ts_dir('15Min')

    @property
    def topo_fpath(self) -> Union[str, os.PathLike]:
        return os.path.join(self._root, "CAMELS_LUX_topographic_attributes.csv")

    @property
    def boundary_file(self) -> Union[str, os.PathLike]:
        return os.path.join(self._boundary_dir, "catchments_CAMELS-LUX.shp")

    @property
    def boundary_id_map(self) -> str:
        """
        Release 2.1 added ``gauge_id`` (1 to 56) and ``station`` to the
        catchments shapefile and put ``Area`` first. Without this the base class
        would take the first attribute, ``Area``, and map every boundary to an
        id like ``ID_258344800``.
        """
        return "gauge_id"

    def _boundary_catch_id(self, value) -> str:
        """``1`` in the shapefile is gauge ``ID_01`` of this dataset"""
        return f"ID_{int(value):02d}"

    def _cache_fname(self, timestep: str) -> str:
        """name of the netCDF cache of ``timestep`` for this release"""
        precision = "" if np.dtype(self.fp) == np.float32 else f"_{np.dtype(self.fp).name}"
        return cache_name(f"{self.name.lower()}_{timestep}_{self.release}{precision}.nc")

    @property
    def dyn_fname(self) -> Union[str, os.PathLike]:
        """netCDF cache of this release and timestep, e.g. ``camels_lux_D_2.1_v2.nc``"""
        return self._cache_fname(self.timestep)

    @property
    def _old_release_caches(self) -> List[str]:
        """
        netCDF caches built from release 1.1. They carry no release in their
        name: ``camels_lux_D.nc`` before the cache version was introduced and
        ``camels_lux_D_v2.nc`` after it.
        """
        return [os.path.join(self.path, fname)
                for timestep in self.time_steps
                for fname in (f"{self.name.lower()}_{timestep}.nc",
                              cache_name(f"{self.name.lower()}_{timestep}.nc"))]

    # -------------------------------------------------------------- download

    def _download_camels_lux(self, overwrite: bool = False):
        """
        Downloads and extracts the two zips of release 2.1. Nothing is
        downloaded when the extracted folders are already there, even if the
        archives were deleted (``remove_zip=True``). A copy of release 1.1 is
        deleted first: it named its time series files differently and its values
        were revised, so the two releases must not be mixed in one folder.
        """
        # (archive, folder it is extracted into) of both zips of the release
        targets = [(os.path.join(self.path, fname), os.path.join(self.path, folder))
                   for fname, folder in self._archives]
        folders = [folder for _, folder in targets]
        archives = [archive for archive, _ in targets]

        if overwrite:
            _remove_stale([*archives, *folders, self.dyn_fpath,
                           *self._old_release_caches], self.verbosity)
        elif self._holds_old_release():
            stale = [path for path in (*folders, *archives, *self._old_release_caches)
                     if os.path.lexists(path)]
            # warned regardless of verbosity: data on disk is deleted
            warnings.warn(
                f"CAMELS_LUX: {self.path} holds release 1.1, whose time series files "
                f"are named differently and whose values were revised by the authors. "
                f"Replacing it with release {self.release} ({self.url}), which is "
                f"downloaded again ({len(stale)} paths removed: "
                f"{', '.join(os.path.basename(path) for path in stale)}).", UserWarning)
            _remove_stale(stale, self.verbosity, reason=f"replaced by release {self.release}")

        missing = [(archive, folder) for archive, folder in targets
                   if not os.path.isdir(folder)]

        if not missing:
            if self.verbosity:
                print(f"CAMELS_LUX release {self.release} already exists at {self.path}")
            self.maybe_remove_zip_files()
            return

        to_download = [os.path.basename(archive) for archive, _ in missing
                       if not os.path.exists(archive)]
        if to_download:
            os.makedirs(self.path, exist_ok=True)
            # imported here because that module installs a SIGINT handler on import
            from ...download_zenodo import download_from_zenodo
            # the record also holds dataset-description.pdf, which is not read
            download_from_zenodo(self.path, doi=self.url, include=to_download,
                                 verbosity=self.verbosity)

        for archive, folder in missing:
            _extract_zip(archive, folder, self.verbosity)

        self.maybe_remove_zip_files()
        return

    def _holds_old_release(self) -> bool:
        """
        True if :attr:`path` holds release 1.1. Its ``basin_id.csv`` is not part
        of release 2.1 and nothing else deletes it, so unlike a time series file
        it still marks the old release after ``free_disk_space("redundant")``.
        """
        return os.path.exists(os.path.join(self._root, self._old_release_marker))

    def _check_manifest(self):
        """
        Warns if an attribute file, a boundary file or the time series of a
        gauge is missing, e.g. after an interrupted extraction. The gauges are
        the ids of the metadata file, not whatever files happen to be on disk.
        """
        meta = os.path.join(self._root, self._attr_files[0])
        if not os.path.exists(meta):
            raise FileNotFoundError(
                f"{meta} not found. Re-initialize CAMELS_LUX with overwrite=True.")

        files = [os.path.join(self._root, fname) for fname in self._attr_files]
        files += [self.boundary_file[:-len(".shp")] + ext
                  for ext in (".shp", ".shx", ".dbf", ".prj")]

        ts_dir = self._ts_dir(self.timestep)
        if os.path.isdir(ts_dir):
            gauges = pd.read_csv(meta, usecols=[0], dtype=str).iloc[:, 0]
            files += [os.path.join(ts_dir, self._ts_fname(gauge)) for gauge in gauges]
        elif not self.dyn_fpath_exists:
            warnings.warn(
                f"CAMELS_LUX {self.release}: neither the {self.timestep} time series "
                f"({ts_dir}) nor their netCDF cache ({self.dyn_fpath}) exists. "
                f"Use overwrite=True to download them again.", UserWarning)

        missing = [fpath for fpath in files if not os.path.exists(fpath)]
        if missing:
            warnings.warn(
                f"CAMELS_LUX {self.release}: {len(missing)} of {len(files)} files are "
                f"missing: {missing[:5]}{' ...' if len(missing) > 5 else ''}. "
                f"Use overwrite=True to download them again.", UserWarning)
        return

    def _redundant_after_consolidation(self) -> List[Tuple[str, str]]:
        """
        The per-gauge csv files of a timestep become redundant once the netCDF
        cache of that timestep exists. All three timesteps are listed regardless
        of ``self.timestep``: the parent skips any folder whose cache is
        missing, so one ``free_disk_space("redundant")`` call cleans up every
        timestep that has been consolidated.
        """
        return [(self._ts_dir(timestep),
                 os.path.join(self.path, self._cache_fname(timestep)))
                for timestep in self.time_steps]

    # ------------------------------------------------------------------ names

    @property
    def static_map(self) -> Dict[str, str]:
        return {
            'Lat': gauge_latitude(),
            'Lon': gauge_longitude(),
            'area_km2': catchment_area(),
            'SLOPE_MEAN': slope('degree'),
            'Z_MEAN': catchment_elevation_meters(),
            'grassland': grass_fraction(),
            'agricultural_land': crop_fraction(),
            'urban': urban_fraction(),
            'perimeter_km': catchment_perimeter(),
        }

    @property
    def static_factors(self) -> Dict[str, float]:
        """the land use attributes are published in percent (Table 10)"""
        return {
            urban_fraction(): 0.01,
            grass_fraction(): 0.01,
            crop_fraction(): 0.01,
        }

    @property
    def dyn_map(self) -> Dict[str, str]:
        """
        dynamic features map for CAMELS-LUX catchments. Ten of the 26 columns
        keep their published name because they have no standardized one: the
        two quality flags ``Qflag`` and ``RR_flag_rad``, the six ERA5
        thunderstorm parameters ``cape``, ``cin``, ``kx``, ``tcwv``, ``lls`` and
        ``dls``, and ``RR_min_rad`` and ``RR_max_rad``, which are 5-minute
        extremes within a 1 km2 cell (mm 5min-1 km-2) and therefore not the same
        quantity as the precipitation totals.
        """
        return {
            'Q': observed_streamflow_cms(),
            'Qspec': observed_streamflow_mm(),
            'RR_rad': total_precipitation_with_specifier('radar'),
            'RR_stn': total_precipitation_with_specifier('station'),
            'tp': total_precipitation_with_specifier('era5'),
            # T_stn, added by release 2.1, is interpolated from 51 stations
            't2m': mean_air_temp_with_specifier('era5'),
            'T_stn': mean_air_temp_with_specifier('station'),
            'PET_Oudin': total_potential_evapotranspiration_with_specifier('oudin'),
            'PET_PM': total_potential_evapotranspiration_with_specifier('pm'),
            'q': mean_specific_humidity(),
            'rh': mean_rel_hum(),
            'ws10500': mean_windspeed(),
            'swvl1': soil_moisture_layer1(),
            'swvl2': soil_moisture_layer2(),
            'swvl3': soil_moisture_layer3(),
            'swvl4': soil_moisture_layer4(),
        }

    @property
    def dyn_factors(self) -> Dict[str, float]:
        return {
            # "Specific humidity q kg kg-1" (Table 3 of the dataset
            # description), while spechum_gkg promises g/kg. The largest value
            # over all gauges and days is 0.0165 kg/kg, i.e. 16.5 g/kg.
            mean_specific_humidity(): 1000.0,
        }

    # ------------------------------------------------------------------- data

    def stations(self) -> List[str]:
        """ids of the 56 gauges, e.g. ``ID_01``"""
        return self._static_df.index.to_list()

    @property
    def dynamic_features(self) -> List[str]:
        """the 26 dynamic features of this timestep"""
        return list(self._dyn_features)

    @functools.cached_property
    def _dyn_features(self) -> List[str]:
        """
        Names of the dynamic features of ``self.timestep``, read from the header
        of the first gauge's file, or from the netCDF cache when
        ``free_disk_space("redundant")`` has removed the csv files.
        """
        ts_dir = self._ts_dir(self.timestep)
        if os.path.isdir(ts_dir):
            fpath = os.path.join(ts_dir, self._ts_fname(self.stations()[0]))
            header = pd.read_csv(fpath, index_col=0, nrows=0)
            return [self.dyn_map.get(col, col) for col in header.columns]

        with netCDF4.Dataset(self.dyn_fpath, "r") as ds:
            return [str(name) for name in ds.variables["dynamic_features"][:]]

    @property
    def start(self) -> pd.Timestamp:
        return self._time_extent[0]

    @property
    def end(self) -> pd.Timestamp:
        return self._time_extent[1]

    @functools.cached_property
    def _time_extent(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        First and last timestamp over all gauges of ``self.timestep``, derived
        from the data instead of a hardcoded extent: ten gauges start one or two
        years late, and the three timesteps do not end on the same timestamp.
        Only the first and the last line of each file are read, so this costs
        two small reads per gauge even for the 88 MB 15-minute files.
        """
        ts_dir = self._ts_dir(self.timestep)
        if os.path.isdir(ts_dir):
            dates = [_first_last_csv_date(os.path.join(ts_dir, self._ts_fname(stn)))
                     for stn in self.stations()]
            return (pd.Timestamp(min(first for first, _ in dates)),
                    pd.Timestamp(max(last for _, last in dates)))

        with netCDF4.Dataset(self.dyn_fpath, "r") as ds:
            time = ds.variables["time"]
            bounds = netCDF4.num2date(time[[0, -1]], time.units,
                                      getattr(time, "calendar", "standard"),
                                      only_use_cftime_datetimes=False)
        return pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1])

    @functools.cached_property
    def _static_df(self) -> pd.DataFrame:
        """
        The five attribute files of the release as one (56, 61) table, read
        once. The files are named explicitly instead of globbed so that the
        order of the static features does not depend on the order the file
        system happens to return, and so that a missing file is noticed.
        """
        dfs = []
        for fname in self._attr_files:
            df = pd.read_csv(os.path.join(self._root, fname), index_col=0, dtype={0: str})
            df.index = df.index.astype(str)
            dfs.append(df)

        static = pd.concat(dfs, axis=1)
        static.index.name = 'gauge_id'
        static.rename(columns=self.static_map, inplace=True)

        # after renaming: static_factors is keyed on the standardized names
        for col, factor in self.static_factors.items():
            if col in static.columns:
                static[col] *= factor

        return static

    def _static_data(self) -> pd.DataFrame:
        """
        static attributes of catchments

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of static features of all catchments of
            shape (56, 61). A copy, so that a caller cannot corrupt the cache.
        """
        return self._static_df.copy()

    def _read_stn_dyn(self, stn: str, nrows: int = None) -> pd.DataFrame:
        """dynamic data of one gauge, with standardized names and units"""
        fpath = os.path.join(self._ts_dir(self.timestep), self._ts_fname(stn))

        stn_df = pd.read_csv(fpath, index_col=0, parse_dates=True, nrows=nrows)
        stn_df.index.name = 'time'

        if stn_df.index.has_duplicates:
            # warned regardless of verbosity: rows are dropped
            n_dup = int(stn_df.index.duplicated().sum())
            warnings.warn(f"CAMELS_LUX: {stn} ({self.timestep}) has {n_dup} duplicated "
                          f"timestamps; the first row of each is kept.", UserWarning)
            stn_df = stn_df[~stn_df.index.duplicated(keep='first')]

        stn_df.rename(columns=self.dyn_map, inplace=True)
        self._apply_dyn_factors(stn_df)

        return stn_df

    def __getstate__(self):
        """
        Drops the cached tables when the dataset is pickled, so that the static
        table and the 56 catchment boundaries (a 13 MB shapefile) do not travel
        to every worker of the process pool of :meth:`_read_dynamic`. Both are
        rebuilt lazily where they are needed.
        """
        return {name: value for name, value in self.__dict__.items()
                if name not in self._NOT_PICKLED}



def _first_last_csv_date(fpath: Union[str, os.PathLike]) -> Tuple[str, str]:
    """
    First and last value of the index column of a csv file with a header, read
    without loading the file: the first line after the header, and the last
    non-empty line of its tail. Lets a dataset derive its temporal extent from
    time series files that are too large to read only for their two end dates.
    """
    with open(fpath, 'rb') as f:
        f.readline()                                  # header
        first = f.readline()
        if not first.strip():
            raise ValueError(f"{fpath} has no data rows")

        f.seek(0, os.SEEK_END)
        size = f.tell()
        # a tail longer than the longest line of these files; a file shorter
        # than that is read from its start, where the header is skipped below
        f.seek(max(0, size - 4096))
        tail = [line for line in f.read().splitlines() if line.strip()]

    last = tail[-1]
    return first.split(b',')[0].decode(), last.split(b',')[0].decode()
