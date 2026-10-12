import os
import glob
import time
import zlib
import shutil
import zipfile
import warnings
import functools
import concurrent.futures as cf
from typing import Union, List, Dict, Tuple

import numpy as np
import pandas as pd

from ..utils import _RainfallRunoff, n_workers, cache_name, ymd_index
from ..._geom_utils import epsg2056_point_to_wgs84
from ...utils import validate_attributes
from ...download_zenodo import download_from_zenodo
from ..._backend import fiona
from .._map import (
    observed_streamflow_cms,
    observed_streamflow_mm,
    mean_air_temp,
    max_air_temp,
    min_air_temp,
    total_precipitation,
    snow_water_equivalent,
)
from .._map import catchment_area, gauge_latitude, gauge_longitude, slope
from ._common import _first_and_last_row, _remove_stale, _warn_duplicate_gauges


# CAMELS-CH version 0.9 (zenodo record 15025258) is the release that accompanies
# the published ESSD paper. It is read from a version named folder so that the
# 0.6 pre-release, which earlier versions of this class used and which is
# semicolon separated, is neither read nor deleted if it is still on disk.
_CH_VERSION = "0.9"
_CH_RECORD = "https://zenodo.org/records/15025258"
_CH_HOURLY_RECORD = "https://zenodo.org/records/7691294"
_CH_DIR = f"camels_ch_v{_CH_VERSION}"
_CH_INVENTORY = "Inventory_discharge_hydroCH.xlsx"


class CAMELS_CH(_RainfallRunoff):
    """
    Data of 331 catchments of hydrologic Switzerland (Switzerland and the
    neighbouring parts of Austria, France, Germany and Italy) following
    `Hoege et al., 2023 <https://doi.org/10.5194/essd-15-5755-2023>`_ , version
    0.9 from `zenodo <https://zenodo.org/records/15025258>`_ .

    With ``timestep='D'`` (default) the dataset has 9 dynamic features from
    1981-01-01 to 2020-12-31 and 209 static features. The dynamic features are
    observed discharge (``q_cms_obs`` m3/s and ``q_mm_obs`` mm/day), water level
    (``waterlevel(m)`` m a.s.l.), precipitation (``pcp_mm`` mm/day), minimum,
    mean and maximum air temperature (``airtemp_C_*`` degree Celsius), relative
    sunshine duration (``rel_sun_dur(%)`` %) and snow water equivalent
    (``swe_mm`` mm). The simulation based time series and attributes of the
    dataset are not provided, only the observation based ones.

    With ``timestep='H'`` the dataset has one dynamic feature, hourly observed
    discharge (``q_cms_obs`` m3/s) from
    `Kauzlaric et al., 2023 <https://zenodo.org/records/7691294>`_ , for 170 of
    the 331 catchments, from 1923-02-15 to 2021-02-08. The hourly data is not in
    UTC but in winter time (UTC+1) throughout the year. The hour ``HH`` of the
    source files labels the interval ``HH:00`` to ``HH+1:00``. The files mark a
    missing hour with ``-9999``,
    which is returned as NaN (289453 hours at 28 of the gauges); the 70626
    negative discharges that remain belong to gauges 2446 and 2447 and are
    real, both are channels with bidirectional flow between lakes and, like the
    33 lakes and gauge 2327, are better left out of a rainfall-runoff analysis.

    Catchment boundaries are provided and are converted from CH1903+ / LV95
    (EPSG:2056) to WGS84. 224 MB are downloaded for ``timestep='D'`` and 417 MB
    more for ``timestep='H'``; the Caravan extension of the record is not
    downloaded. On a 48-core machine the first initialization takes about 40 s
    for ``D``, of which 25 s is the download, and afterwards all 331 stations
    with all 9 features are fetched in 0.8 s from the csv files or 0.6 s from
    the netCDF cache. All 170 hourly stations are fetched in 8 s.

    Examples
    ---------
    >>> from aqua_fetch import CAMELS_CH
    >>> dataset = CAMELS_CH()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='2004', as_dataframe=True)
    >>> df = dynamic['2004'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (14610, 9)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       331
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (33 out of 331)
       33
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(14610, 9), (14610, 9), (14610, 9),... (14610, 9), (14610, 9)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('2004', as_dataframe=True,
    ...  dynamic_features=['pcp_mm', 'airtemp_C_mean', 'q_cms_obs'])
    >>> dynamic['2004'].shape
       (14610, 3)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='2004', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['2004'].shape
    ((1, 209), 1, (14610, 9))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 14610, 'dynamic_features': 9})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns WGS84 coordinates of all stations
    >>> coords.shape
        (331, 2)
    >>> dataset.stn_coords('2004')  # returns coordinates of station whose id is 2004
                    lat      long
        gauge_id
        2004      46.930752  7.116924
    >>> dataset.stn_coords(['2004', '2007'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('2004')
    # get coordinates of two stations
    >>> dataset.area(['2004', '2007'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('2004')
    ...
    # hourly discharge of the 170 catchments which have it
    >>> dataset = CAMELS_CH(timestep='H')
    >>> len(dataset.stations())
        170
    >>> _, dynamic = dataset.fetch(stations='2009', as_dataframe=True)
    >>> dynamic['2009'].shape
        (401017, 1)
    """

    # the zenodo record each archive comes from. Only the files listed in
    # ``_archives`` are downloaded from them, never the whole record.
    url = {
        'camels_ch.zip': _CH_RECORD,
        'DischargeDBHydroCH.zip': _CH_HOURLY_RECORD,
    }

    def __init__(
            self,
            path=None,
            timestep: str = 'D',
            overwrite: bool = False,
            to_netcdf: bool = None,
            **kwargs
    ):
        """
        Parameters
        ----------
        path : str
            directory under which the data is (or will be) saved in a
            ``CAMELS_CH`` folder. If None, the default data directory of
            aqua_fetch is used. The data is downloaded only once; a later
            initialization reads what is already there.
        timestep : str
            ``D`` (default) for the daily data of all 331 catchments or ``H``
            for the hourly discharge of the 170 catchments that have it.
            ``H`` additionally downloads 416 MB and needs ``openpyxl`` to read
            the inventory which maps a gauge to its file.
        overwrite : bool
            if True, the archives, extracted folders and netCDF caches this
            ``timestep`` uses are deleted and downloaded again.
        to_netcdf : bool
            whether to save the dynamic data in a netCDF cache for faster
            reading. Requires netCDF4 and xarray. Defaults to True for ``D``
            and to False for ``H``, whose stations do not share a time index,
            so that the cache would be a 584 MB array that is mostly NaN.
        **kwargs :
            any keyword argument of :py:class:`aqua_fetch.rr._RainfallRunoff`
            such as ``processes``, ``verbosity`` or ``remove_zip``.
        """
        if timestep not in ('D', 'H'):
            raise ValueError(f"timestep must be 'D' or 'H' but is {timestep!r}")

        if to_netcdf is None:
            to_netcdf = timestep == 'D'

        super().__init__(path=path, timestep=timestep, to_netcdf=to_netcdf,
                         overwrite=overwrite, **kwargs)

        self._download_camels_ch(overwrite)

        self._check_manifest()

        self._check_duplicates()

        self._maybe_to_netcdf()

    # ------------------------------------------------------------------
    # download
    # ------------------------------------------------------------------

    def _download_camels_ch(self, overwrite: bool = False):
        """
        Downloads the archives this ``timestep`` needs, unless their extracted
        folders are already there. ``overwrite=True`` first deletes the
        archives, extracted folders and netCDF caches of this ``timestep``, so
        that nothing stale survives.
        """
        if overwrite:
            stale = [self._version_dir,
                     *glob.glob(os.path.join(glob.escape(self.path),
                                             f"{self.name.lower()}_{self.timestep}*.nc"))]
            if self.timestep == 'H':
                stale += [os.path.join(self.path, 'DischargeDBHydroCH'),
                          os.path.join(self.path, 'DischargeDBHydroCH.zip'),
                          self._inventory_file]
            _remove_stale(stale, self.verbosity)

        # the daily archive is always needed: it holds the station list, the
        # static features and the catchment boundaries
        _download_zenodo_archive(
            doi=_CH_RECORD,
            archive='camels_ch.zip',
            outdir=self._version_dir,
            extracted=self.camels_path,
            remove_zip=self.remove_zip,
            verbosity=self.verbosity,
        )

        if self.timestep == 'H':
            _download_zenodo_archive(
                doi=_CH_HOURLY_RECORD,
                archive='DischargeDBHydroCH.zip',
                outdir=self.path,
                extracted=os.path.join(self.path, 'DischargeDBHydroCH'),
                extras=(_CH_INVENTORY,),
                remove_zip=self.remove_zip,
                verbosity=self.verbosity,
            )
        return

    def _check_manifest(self):
        """warns if a file this class reads is missing, e.g. because an
        extraction was interrupted or a file was deleted by hand"""
        files = [self._attr_path(fname, subdir) for fname, subdir in _CH_STATIC_FILES]
        files += [self.boundary_file[:-len(".shp")] + ext
                  for ext in (".shp", ".shx", ".dbf", ".prj")]
        missing = [fpath for fpath in files if not os.path.exists(fpath)]

        if not missing:
            # the attribute files are there, so the station list can be read and
            # the time series file of every station checked as well
            files += [self._stn_dyn_path(stn) for stn in self.stations()]
            missing = [fpath for fpath in files if not os.path.exists(fpath)]

        if missing:
            warnings.warn(
                f"CAMELS_CH {self.timestep}: {len(missing)} of {len(files)} files are "
                f"missing: {missing[:5]}. Use overwrite=True to download them again.",
                UserWarning)
        return

    def _check_duplicates(self):
        """warns if two gauges share a name and rounded coordinates"""
        coords = self.stn_coords()
        meta = pd.DataFrame({
            'gauge_id': coords.index,
            'gauge_name': self._static_data().loc[coords.index, 'gauge_name'].values,
            'gauge_lat': coords['lat'].values,
            'gauge_lon': coords['long'].values,
        })
        _warn_duplicate_gauges(self.name, meta)
        return

    # ------------------------------------------------------------------
    # paths
    # ------------------------------------------------------------------

    @property
    def _version_dir(self) -> Union[str, os.PathLike]:
        """folder holding the extracted archive of this version"""
        return os.path.join(self.path, _CH_DIR)

    @property
    def camels_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self._version_dir, 'camels_ch')

    @property
    def static_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.camels_path, 'static_attributes')

    @property
    def dynamic_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.camels_path, 'timeseries', 'observation_based')

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(self.camels_path, 'catchment_delineations',
                            'CAMELS_CH_catchments.shp')

    @property
    def foen_path(self) -> Union[str, os.PathLike]:
        """folder holding the hourly discharge files of the Swiss gauges"""
        return os.path.join(self.path, 'DischargeDBHydroCH', 'DischargeDBHydroCH', 'CH', 'FOEN')

    @property
    def _inventory_file(self) -> Union[str, os.PathLike]:
        return os.path.join(self.path, _CH_INVENTORY)

    def _attr_path(self, fname: str, subdir: str = "") -> Union[str, os.PathLike]:
        """path of one static attribute file"""
        return os.path.join(self.static_path, subdir, fname)

    @property
    def dyn_fname(self) -> Union[str, os.PathLike]:
        """name of the netCDF cache of the dynamic data, one per ``timestep``,
        dataset version and precision, e.g. camels_ch_D_0.9_v2.nc"""
        precision = "" if np.dtype(self.fp) == np.float32 else f"_{np.dtype(self.fp).name}"
        return cache_name(f"{self.name.lower()}_{self.timestep}_{_CH_VERSION}{precision}.nc")

    # ------------------------------------------------------------------
    # stations
    # ------------------------------------------------------------------

    def stations(self) -> List[str]:
        """
        ids of the catchments of this ``timestep``: all 331 for ``D`` and the
        170 with hourly discharge for ``H``.
        """
        if self.timestep == 'H':
            return list(self._hourly_stns)
        return list(self._all_stns)

    @functools.cached_property
    def _all_stns(self) -> List[str]:
        """ids of all 331 catchments, in the order of the attribute files"""
        stns = pd.read_csv(self._attr_path("CAMELS_CH_glacier_attributes.csv"),
                           sep=',', skiprows=1, usecols=['gauge_id'])['gauge_id']
        return [str(stn) for stn in stns]

    @functools.cached_property
    def _hourly_stns(self) -> List[str]:
        """ids of the catchments which have hourly discharge"""
        inventory = set(self._inventory.index)
        return [stn for stn in self._all_stns if stn in inventory]

    def hourly_stations(self) -> List[str]:
        """
        ids of those catchments which have hourly data and which are also part
        of the CAMELS-CH dataset
        """
        return list(self._hourly_stns)

    def all_hourly_stations(self) -> List[str]:
        """
        ids of every gauge of the hourly inventory, including the 121 outside
        CAMELS-CH
        """
        return list(self._inventory.index)

    def foen_stations(self) -> List[str]:
        """names of the hourly discharge files of the Swiss gauges"""
        return sorted(os.listdir(self.foen_path))

    @functools.cached_property
    def _inventory(self) -> pd.DataFrame:
        """
        the inventory of DischargeDBHydroCH, indexed by gauge id. It is the
        authoritative map from a gauge to its file: two files in the FOEN
        folder carry the id 2160 and only the one listed here (SarBro, the
        Sarine at Broc) belongs to that gauge.
        """
        if not os.path.exists(self._inventory_file):
            raise FileNotFoundError(
                f"{self._inventory_file} does not exist. It is downloaded with "
                f"the hourly data, so initialize CAMELS_CH with timestep='H'.")
        try:
            inventory = pd.read_excel(self._inventory_file, dtype={'ID': str})
        except ImportError as e:
            raise ImportError(
                f"reading {_CH_INVENTORY}, which maps a gauge to its hourly file, "
                f"needs the openpyxl package: pip install openpyxl") from e
        return inventory.set_index('ID')

    @functools.cached_property
    def _hourly_paths(self) -> Dict[str, str]:
        """id -> path of the hourly discharge file, for the catchments of CAMELS-CH"""
        fnames = self._inventory['Filename']
        return {stn: os.path.join(self.foen_path, fnames[stn]) for stn in self._hourly_stns}

    # ------------------------------------------------------------------
    # dynamic data
    # ------------------------------------------------------------------

    @property
    def dyn_map(self) -> Dict[str, str]:
        # table 1 in https://essd.copernicus.org/articles/15/5755/2023/
        return {
            'discharge_vol(m3/s)': observed_streamflow_cms(),
            'discharge_spec(mm/d)': observed_streamflow_mm(),
            'temperature_min(degC)': min_air_temp(),
            'temperature_max(degC)': max_air_temp(),
            'temperature_mean(degC)': mean_air_temp(),
            'precipitation(mm/d)': total_precipitation(),
            'swe(mm)': snow_water_equivalent(),
        }

    @property
    def dynamic_features(self) -> List[str]:
        return list(self._dyn_feats)

    @functools.cached_property
    def _dyn_feats(self) -> List[str]:
        if self.timestep == 'H':
            return [observed_streamflow_cms()]
        return self._read_stn_dyn(self._all_stns[0]).columns.tolist()

    @property
    def _dyn_reader(self):
        """the module level function which reads the file of one station, with
        everything but the file path already bound to it"""
        if self.timestep == 'H':
            return functools.partial(_read_camels_ch_hourly, self.fp)
        return functools.partial(_read_camels_ch_daily, tuple(self.dyn_map.items()), self.fp)

    def _stn_dyn_path(self, station: str) -> str:
        """path of the file holding the dynamic data of one station"""
        if self.timestep == 'H':
            return self._hourly_paths[station]
        return os.path.join(self.dynamic_path, f"CAMELS_CH_obs_based_{station}.csv")

    def _read_stn_dyn(self, station: str) -> pd.DataFrame:
        """
        Reads the dynamic data of one catchment: the daily meteorological and
        streamflow time series for ``timestep='D'`` and the hourly discharge
        for ``timestep='H'``.
        """
        return self._dyn_reader(self._stn_dyn_path(station))

    def _read_dynamic(
            self,
            stations,
            dynamic_features,
            st: Union[str, pd.Timestamp] = None,
            en: Union[str, pd.Timestamp] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Reads the dynamic data of many stations, in parallel where it pays off.

        Behaves exactly like the base implementation (the same dict of
        ``{station: DataFrame}`` sliced to ``[st:en, dyn_feats]`` with the axis
        names set) but dispatches a module-level reader instead of the bound
        ``self._read_stn_dyn``, which would pickle ``self`` (and with it the
        cached static table) to every worker on every task. Whether a pool is
        started is decided by the size of the files to read, not by their
        number: reading the 43 MB of daily csv files repays a pool only where
        starting one is cheap (the ``fork`` start method), the 1.4 GB of hourly
        files repay it everywhere.
        """
        st, en = self._check_length(st, en)
        dyn_feats = validate_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')
        stations = validate_attributes(stations, self.stations(), 'stations')

        paths = [self._stn_dyn_path(stn) for stn in stations]
        cpus = n_workers(sum(os.path.getsize(p) for p in paths), len(paths), self.processes)
        reader = self._dyn_reader

        start = time.time()
        if cpus == 1:
            results = [reader(fpath) for fpath in paths]
        else:
            with cf.ProcessPoolExecutor(cpus) as executor:
                results = list(executor.map(reader, paths))

        dyn = {}
        for stn, stn_df in zip(stations, results):
            stn_df = stn_df.loc[st:en, dyn_feats]
            stn_df.columns.name = 'dynamic_features'
            stn_df.index.name = 'time'
            dyn[stn] = stn_df

        if self.verbosity:
            print(f"Read {len(dyn)} stations for {len(dyn_feats)} dyn features "
                  f"in {time.time() - start:.2f} seconds with {cpus} cpus.")
        return dyn

    def read_hourly_q_ch(self, stn: str) -> pd.DataFrame:
        """
        Hourly discharge (m3/s) of one gauge, in winter time (UTC+1), with the
        ``-9999`` no-data marker of the source file returned as NaN.

        Examples
        --------
        >>> from aqua_fetch import CAMELS_CH
        >>> dataset = CAMELS_CH(timestep='H')
        >>> dataset.read_hourly_q_ch('2009').shape
        (401017, 1)
        """
        return _read_camels_ch_hourly(self.fp, self._hourly_paths[stn])

    @property
    def start(self) -> pd.Timestamp:  # start of data
        return self._extent[0]

    @property
    def end(self) -> pd.Timestamp:
        return self._extent[1]

    @functools.cached_property
    def _extent(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        First and last time step over all stations, read from the first and the
        last row of every file. The rows of every file are sorted in time, so
        this is exact and does not require reading 1.4 GB of hourly data.
        """
        stamps = []
        for station in self.stations():
            first, last = _first_and_last_row(self._stn_dyn_path(station))
            if self.timestep == 'H':
                stamps += [pd.Timestamp(*(int(v) for v in row.split(b'\t')[:4]))
                           for row in (first, last)]
            else:
                stamps += [pd.Timestamp(row.split(b',')[0].decode())
                           for row in (first, last)]
        return min(stamps), max(stamps)

    # ------------------------------------------------------------------
    # static data
    # ------------------------------------------------------------------

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'area': catchment_area(),
                'slope_mean': slope('degrees'),
                'gauge_lat': gauge_latitude(),
                'gauge_lon': gauge_longitude(),
        }

    @property
    def static_features(self) -> List[str]:
        return self._static_data().columns.tolist()

    def _read_attrs(self, fname: str, subdir: str = "", dtype=None) -> pd.DataFrame:
        """
        Reads one static attribute file. The first line of each file is a
        comment describing it, the values are comma separated (semicolon
        separated before version 0.7) and the gauge ids are returned as strings.
        The files are latin-1 encoded, only the topographic one has non-ascii
        characters (e.g. the Rhone).
        """
        df = pd.read_csv(self._attr_path(fname, subdir), sep=',', skiprows=1,
                         index_col='gauge_id', dtype=dtype, encoding='latin-1')
        if df.index.dtype.kind == 'f':  # read as a float because dtype applies to every column
            df.index = df.index.astype(int)
        df.index = df.index.astype(str)
        return df

    def glacier_attrs(self) -> pd.DataFrame:
        """
        returns a dataframe with four columns
            - 'glac_area'
            - 'glac_vol'
            - 'glac_mass'
            - 'glac_area_neighbours'
        """
        return self._read_attrs("CAMELS_CH_glacier_attributes.csv", dtype=np.float32)

    def climate_attrs(self) -> pd.DataFrame:
        """returns 14 climate attributes of catchments."""
        return self._read_attrs("CAMELS_CH_climate_attributes_obs.csv", dtype={
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
        })

    def geol_attrs(self) -> pd.DataFrame:
        """15 geological features"""
        return self._read_attrs("CAMELS_CH_geology_attributes.csv", dtype=np.float32)

    def supp_geol_attrs(self) -> pd.DataFrame:
        """supplimentary geological features"""
        return self._read_attrs("CAMELS_CH_geology_attributes_supplement.csv",
                                subdir="supplements", dtype=np.float32)

    def human_inf_attrs(self) -> pd.DataFrame:
        """14 athropogenic factors"""
        return self._read_attrs("CAMELS_CH_humaninfluence_attributes.csv", dtype={
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
        })

    def hydrogeol_attrs(self) -> pd.DataFrame:
        """10 hydrogeological factors"""
        return self._read_attrs("CAMELS_CH_hydrogeology_attributes.csv", dtype=float)

    def hydrol_attrs(self) -> pd.DataFrame:
        """14 hydrological parameters + 2 useful infos"""
        return self._read_attrs("CAMELS_CH_hydrology_attributes_obs.csv", dtype={
            'gauge_id': str,
            'sign_number_of_years': int,
            'q_mean': float,
            'runoff_ratio': float, 'stream_elas': float, 'slope_fdc': float,
            'baseflow_index_landson': float,
            'hfd_mean': float,
            'Q5': float, 'Q95': float, 'high_q_freq': float, 'high_q_dur': float,
            'low_q_freq': float
        })

    def landcolover_attrs(self) -> pd.DataFrame:
        """13 landcover parameters"""
        return self._read_attrs("CAMELS_CH_landcover_attributes.csv", dtype={
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
        })

    def soil_attrs(self) -> pd.DataFrame:
        """80 soil parameters"""
        return self._read_attrs("CAMELS_CH_soil_attributes.csv")

    def topo_attrs(self) -> pd.DataFrame:
        """topographic parameters"""
        return self._read_attrs("CAMELS_CH_topographic_attributes.csv")

    def _static_data(self) -> pd.DataFrame:
        return self._static

    @functools.cached_property
    def _static(self) -> pd.DataFrame:
        """all 209 static features of all 331 catchments, read once"""
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
        df.rename(columns=self.static_map, inplace=True)
        return df

    def stn_coords(
            self,
            stations: Union[str, List[str]] = 'all'
    ) -> pd.DataFrame:
        """
        Returns WGS84 (EPSG:4326) coordinates of stations as a DataFrame with
        ``lat`` and ``long`` columns.

        The coordinates are derived from the precise ``gauge_easting`` /
        ``gauge_northing`` columns (CH1903+ / LV95, EPSG:2056) and converted to
        WGS84 with the pyproj-free :func:`epsg2056_point_to_wgs84` helper. They
        are used in preference to the dataset's own ``gauge_lat`` / ``gauge_lon``
        columns, which are rounded to two decimals (~500 m). The conversion was
        verified against pyproj (EPSG:2056 -> EPSG:4326): the error is under
        3 m (mean 0.7 m) over all 331 stations.

        Parameters
        ----------
        stations :
            name/names of stations. If not given, coordinates of all stations
            will be returned.

        Returns
        -------
        pd.DataFrame
            with ``lat`` and ``long`` columns, indexed by station id.

        Examples
        --------
        >>> from aqua_fetch import CAMELS_CH
        >>> dataset = CAMELS_CH()
        >>> dataset.stn_coords('2004')
                        lat      long
            gauge_id
            2004      46.930752  7.116924
        """
        stations = validate_attributes(stations, self.stations(), 'stations')
        en = self._static_data().loc[stations, ['gauge_easting', 'gauge_northing']].astype(float)
        lat, long = epsg2056_point_to_wgs84(
            en['gauge_easting'].values, en['gauge_northing'].values)
        return pd.DataFrame(
            {'lat': lat, 'long': long}, index=en.index).astype(self.fp)

    def transform_boundary(self, boundary):
        """
        Transform a catchment boundary from CH1903+ / LV95 (EPSG:2056, the CRS
        of the CAMELS-CH shapefile) to WGS84 (EPSG:4326) lon/lat.

        Uses the pyproj-free :func:`epsg2056_point_to_wgs84` helper (swisstopo's
        approximate LV95 -> WGS84 formula). Verified against pyproj
        (EPSG:2056 -> EPSG:4326): the per-vertex error is below 3 m across all
        331 catchments (~3.2 M vertices) - finer than the dataset's own
        two-decimal (~500 m) gauge coordinates. Polygons with interior rings
        (holes) and MultiPolygons are handled; the geometry type and ring
        structure are preserved. The conversion is vectorised per ring (one
        array call rather than one call per vertex).
        """
        if fiona is None:
            return boundary

        def _ring_to_wgs84(ring):
            arr = np.asarray(ring, dtype=float)
            # fiona stores each vertex as (x=easting, y=northing[, z]); output
            # is (lon, lat) to keep the (x, y) ordering of the geometry.
            lat, long = epsg2056_point_to_wgs84(arr[:, 0], arr[:, 1])
            return list(zip(long.tolist(), lat.tolist()))

        if boundary.type == 'MultiPolygon':
            coords = [[_ring_to_wgs84(ring) for ring in polygon]
                      for polygon in boundary.coordinates]
        else:  # Polygon, possibly with interior rings (holes)
            coords = [_ring_to_wgs84(ring) for ring in boundary.coordinates]

        return fiona.Geometry(type=boundary.type, coordinates=coords)


# the .asc files of DischargeDBHydroCH mark a missing hour with this value. It
# is not a discharge: it occurs in long uninterrupted blocks (one of 12.8 years
# at gauge 2199) at 28 of the 170 gauges, while the genuinely negative values of
# the bidirectional channels 2446/2447 range from -422 to -0.001 m3/s.
_CH_HOURLY_NA = -9999.0
_CH_HOURLY_COLS = ("YYYY", "MM", "DD", "HH", "q")
_CH_HOURLY_DTYPE = {"YYYY": np.int16, "MM": np.int8, "DD": np.int8, "HH": np.int8}


# static attribute files, as (file name, sub folder of static_attributes)
_CH_STATIC_FILES = (
    ("CAMELS_CH_climate_attributes_obs.csv", ""),
    ("CAMELS_CH_geology_attributes.csv", ""),
    ("CAMELS_CH_geology_attributes_supplement.csv", "supplements"),
    ("CAMELS_CH_glacier_attributes.csv", ""),
    ("CAMELS_CH_humaninfluence_attributes.csv", ""),
    ("CAMELS_CH_hydrogeology_attributes.csv", ""),
    ("CAMELS_CH_hydrology_attributes_obs.csv", ""),
    ("CAMELS_CH_landcover_attributes.csv", ""),
    ("CAMELS_CH_soil_attributes.csv", ""),
    ("CAMELS_CH_topographic_attributes.csv", ""),
)


def _download_zenodo_archive(
        doi: str,
        archive: str,
        outdir: str,
        extracted: str,
        extras: Tuple[str, ...] = (),
        remove_zip: bool = False,
        verbosity: int = 1,
):
    """
    Makes sure that ``extracted`` and every file of ``extras`` are present in
    ``outdir``, downloading and unpacking ``archive`` of the zenodo record
    ``doi`` if they are not.

    Only the named files are taken from the record, never everything it holds.
    The decision to download is taken on the *extracted* folder, not on the
    archive, so ``remove_zip=True`` does not force a fresh download on the next
    initialization. ``archive`` must hold a single top level folder named after
    ``extracted``; it is unpacked into a temporary folder which is renamed only
    once the extraction has finished, so that an interrupted extraction is
    redone instead of being taken as complete. A corrupt archive is deleted.
    """
    os.makedirs(outdir, exist_ok=True)
    archive_path = os.path.join(outdir, archive)

    missing = [f for f in extras if not os.path.exists(os.path.join(outdir, f))]
    if not os.path.exists(extracted) and not os.path.exists(archive_path):
        missing.insert(0, archive)

    if missing:
        if verbosity:
            print(f"downloading {', '.join(missing)} of {doi} to {outdir}")
        download_from_zenodo(outdir, doi=doi, include=missing, verbosity=verbosity)

    if not os.path.exists(extracted):
        if verbosity:
            print(f"extracting {archive_path}")
        partial = f"{extracted}_extracting"
        shutil.rmtree(partial, ignore_errors=True)
        try:
            with zipfile.ZipFile(archive_path) as zf:
                zf.extractall(partial)
        except (zipfile.BadZipFile, zlib.error, EOFError):  # e.g. an error page saved as the archive
            shutil.rmtree(partial, ignore_errors=True)
            os.remove(archive_path)
            raise ValueError(f"{archive_path} is corrupt and was deleted. Initialize "
                             f"the dataset again to download it again.") from None
        os.replace(os.path.join(partial, os.path.basename(extracted)), extracted)
        shutil.rmtree(partial, ignore_errors=True)

    if remove_zip and os.path.exists(archive_path):
        if verbosity:
            print(f"remove_zip=True: removing {archive_path}")
        os.remove(archive_path)
    return


def _read_camels_ch_daily(
        dyn_map_items: Tuple[Tuple[str, str], ...],
        dtype,
        fpath: str,
) -> pd.DataFrame:
    """
    Reads the daily observation based time series of one CAMELS-CH catchment.

    A module-level function (not a method) on purpose: it is dispatched to a
    :class:`concurrent.futures.ProcessPoolExecutor` by
    :meth:`CAMELS_CH._read_dynamic`. Handing a *bound* method to a process pool
    would pickle ``self``, and with it the cached static table, to every worker
    on every task.
    """
    df = pd.read_csv(fpath, sep=',', index_col='date', parse_dates=True, dtype=dtype)
    df.rename(columns=dict(dyn_map_items), inplace=True)
    return df


def _read_camels_ch_hourly(dtype, fpath: str) -> pd.DataFrame:
    """
    Reads the hourly discharge (m3/s) of one gauge from a DischargeDBHydroCH
    ``.asc`` file. The file's ``-9999`` no-data marker is returned as NaN; every
    other value is passed through unchanged, including the negative discharges
    of the bidirectional channels. Module-level for the same reason as
    :func:`_read_camels_ch_daily`.
    """
    df = pd.read_csv(fpath, sep='\t', header=0, names=list(_CH_HOURLY_COLS),
                     dtype=_CH_HOURLY_DTYPE)
    q = df['q'].to_numpy(dtype=np.float64, copy=True)
    q[q == _CH_HOURLY_NA] = np.nan
    out = pd.DataFrame(
        {observed_streamflow_cms(): q.astype(dtype)},
        index=ymd_index(df['YYYY'].values, df['MM'].values, df['DD'].values, df['HH'].values),
    )
    out.columns.name = 'dynamic_features'
    out.index.name = 'time'
    return out
