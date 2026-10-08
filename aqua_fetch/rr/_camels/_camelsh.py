import os
import glob
import time
import warnings
from pathlib import Path
import concurrent.futures as cf
from typing import Union, List, Dict, Tuple

import numpy as np
import pandas as pd

from ..utils import _RainfallRunoff
from ...utils import get_cpus, download_and_unzip
from ...utils import validate_attributes, unzip
from ..._backend import netCDF4, xarray as xr

if netCDF4 is not None:
    from netCDF4 import date2num

from .._map import (
    observed_streamflow_cms,
    mean_air_temp,
    total_precipitation,
    total_potential_evapotranspiration,
    solar_radiation,
    downward_longwave_radiation,
)
from .._map import (
    catchment_area,
    gauge_latitude,
    gauge_longitude,
    slope,
    gauge_elevation_meters,
    catchment_elevation_meters,
    population_density,
)


class CAMELSH(_RainfallRunoff):
    """
    Hourly data of 5,767 catchments from United States of America with 13 dynamic
    features and 779 static features for each catchment. For more details on data see
    `Tran et al., (2025) <https://doi.org/10.1038/s41597-025-05612-6>`_ . The dynamic features
    span from 19800101 to 20241231 . The data is downloaded from
    `Zenodo <https://zenodo.org/records/16729675>`_.

    Please note that usage of this dataset requires xarray and netCDF4 libraries.

    Examples
    --------
    >>> from aqua_fetch import CAMELSH
    >>> dataset = CAMELSH()
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       5767
    ... # get data by station id/name
    >>> _, dynamic = dataset.fetch(stations='02342070', as_dataframe=True)
    >>> df = dynamic['02342070'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (394488, 13)
    ...
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (67 out of 5767)
       67
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(394488, 13), (394488, 8), (394488, 13),... (394488, 13), (394488, 13)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('02342070', as_dataframe=True,
    ...  dynamic_features=['swdownrad_wm2', 'pcp_mm', 'pet_mm', 'airtemp_C_mean', 'q_cms_obs'])
    >>> dynamic['02342070'].shape
       (394488, 5)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='02342070', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['02342070'].shape
    ((1, 779), 1, (394488, 13))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 394488, 'dynamic_features': 8})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (5767, 2)
    >>> dataset.stn_coords('02342070')  # returns coordinates of station whose id is 02342070
        32.37431	-84.957993
    >>> dataset.stn_coords(['02342070', '14316700'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('02342070')
    # get coordinates of two stations
    >>> dataset.area(['02342070', '14316700'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('02342070')

    """
    url = {
        "Hourly2.zip": "https://zenodo.org/records/16729675",  # contains observed q and water level
        "timeseries_nonobs.7z": "https://zenodo.org/records/15070091",  # contains NLDAS forcing data
        "timeseries.7z": "https://zenodo.org/records/15066778",
        "attributes.7z": "https://zenodo.org/records/15066778",
        "info.csv": "https://zenodo.org/records/15066778",
        "shapefiles.7z": "https://zenodo.org/records/15066778"
    }

    def __init__(self,
                 path=None,
                 overwrite=False,
                 timestep="H",
                 **kwargs,
    ):

        assert netCDF4 is not None, "netCDF4 library is required for CAMELSH dataset. Please install it using 'pip install netCDF4'"

        super(CAMELSH, self).__init__(
        path=path,
        timestep=timestep,
        **kwargs)

        assert self.timestep == "H", f"CAMELSH dataset only supports hourly timestep but got {self.timestep}."

        # For each remote resource, declare the on-disk paths that fully satisfy
        # it. If any of these exists, the archive is not needed and we skip
        # download + unzip. This prevents re-acquiring data that free_disk_space()
        # (or the user) has deleted because a consolidated NetCDF cache
        # (all_stations_q.nc / all_stn_forcings.nc) covers the same content.
        satisfied_by = {
            "Hourly2.zip":          [self.h2_path,         self.all_stations_q_path],
            "timeseries_nonobs.7z": [self.nonobs_path,     self.all_stn_forcings_path],
            "timeseries.7z":        [self.timeseries_path, self.all_stn_forcings_path],
            "attributes.7z":        [self.attr_path],
            "shapefiles.7z":        [self.sf_path],
            "info.csv":             [os.path.join(self.path, "info.csv")],
        }

        for fname, url in self.url.items():

            if not overwrite and any(os.path.exists(p) for p in satisfied_by.get(fname, [])):
                continue

            dirname = fname.split('.')[0]
            dirpath = os.path.join(self.path, dirname)

            fpath = os.path.join(self.path, fname)

            if not ((os.path.exists(fpath) or os.path.exists(dirpath)) or overwrite):
                download_and_unzip(self.path, url, include=[fname], verbosity=self.verbosity)

            uzipped_dir_path = os.path.join(self.path, fname.split('.')[0])
            if fname.endswith(('.zip', '.7z')) and not os.path.exists(uzipped_dir_path):
                unzip(self.path, keep_parent_dir=True, verbosity=self.verbosity)

        # lazily-populated caches for the consolidated-cache fast reader and the
        # static attribute table (both are heavy to build and read many times).
        # Set before the discovery below, which reads a station through them.
        self._axes_cache = None
        self._static_cache = None
        self._read_order_cache = None

        # Discover stations and dynamic features. Prefer the per-station files
        # when Hourly2/ is present; otherwise fall back to the consolidated nc
        # caches. The fallback uses netCDF4 directly (one open per file) instead
        # of xarray, which is much faster for metadata-only lookups when the
        # consolidated file has thousands of data variables.
        if os.path.exists(self.h2_path):
            self.__stations = [fname.split('_')[0] for fname in os.listdir(self.h2_path)]
            self.__dyn_features = self._read_stn_dyn(self.stations()[0]).dynamic_features.data.tolist()
        else:
            with netCDF4.Dataset(self.all_stations_q_path, "r") as _ncq:
                self.__stations = [v for v in _ncq.variables if v not in _ncq.dimensions]
                q_feats_raw = [str(s) for s in _ncq.variables["dynamic_features"][:]]
            with netCDF4.Dataset(self.all_stn_forcings_path, "r") as _ncf:
                f_feats_raw = [str(s) for s in _ncf.variables["dynamic_features"][:]]
            q_feats = [str(self.dyn_map.get(v, v)) for v in q_feats_raw]
            f_feats = [str(self.dyn_map.get(v, v)) for v in f_feats_raw]
            self.__dyn_features = q_feats + f_feats

        self.bbox = {"llcrnrlat": 22, "urcrnrlat": 75,
                     "llcrnrlon": -168.0,  "urcrnrlon": -65.0}
        self.parallels = np.arange(22, 75, 7)
        self.meridians = np.arange(-168, -65, 12)

    def stations(self) -> List[str]:
        return self.__stations

    @property
    def static_map(self) -> Dict[str, str]:
        return {
            'LAT_GAGE': gauge_latitude(),
            'LNG_GAGE': gauge_longitude(),
            #'LAT_CENT': centroid_latitude(),
            #'LONG_CENT': centroid_longitude(),
            'ELEV_MEAN_M_BASIN': catchment_elevation_meters(),
            'DRAIN_SQKM': catchment_area(),
            'ELEV_SITE_M': gauge_elevation_meters(),
            'SLOPE_PCT': slope('percent'),
            'PDEN_2000_BLOCK': population_density(2000),
            'PDEN_DAY_LANDSCAN_2007': population_density(2007),
            #'PDEN_NIGHT_LANDSCAN_2007': population_density(2007),
            # 'CLAYAVE': clay_content(),
            # 'SILTAVE': silt_content(),
            # 'SANDAVE': sand_content(),
        }

    @property
    def dyn_map(self) -> Dict[str, str]:
        return {
            #'water_level': water_level('m'),
            'Tair': mean_air_temp(),
            'PotEvap': total_potential_evapotranspiration(),
            'Rainf': total_precipitation(),
            # NLDAS-2 downward shortwave/longwave, already hourly W m-2
            # (verified: shortwave peaks near 1000 and is 0 at night)
            'SWdown': solar_radiation(),
            'LWdown': downward_longwave_radiation(),
            'streamflow': observed_streamflow_cms()
        }

    @property
    def boundary_id_map(self) -> str:
        return "GAGE_ID"

    @property
    def h2_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.path, "Hourly2", "Hourly2")
    
    @property
    def attr_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.path, "attributes")
    
    @property
    def sf_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.path, "shapefiles")
    
    @property
    def boundary_file(self) -> Union[str, os.PathLike]:
        return os.path.join(
            self.sf_path,
            "CAMELSH_shapefile.shp"
        )
    
    @property
    def nonobs_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.path, "timeseries_nonobs", "Data", "CAMELSH", "timeseries_nonobs")

    @property
    def timeseries_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.path, "timeseries", "Data", "CAMELSH", "timeseries")

    @property
    def all_stn_forcings_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.path, 'all_stn_forcings.nc')

    @property
    def all_stations_q_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.path, 'all_stations_q.nc')

    @property
    def dynamic_features(self) -> List[str]:
        """
        Returns a list of dynamic features that are available in the dataset.

        Returns
        -------
        List[str]
            a list of dynamic features that are available in the dataset.
            The names of the features are the same as the names used in the
            dataset. The names can be used to fetch the data using
            :meth:`fetch_dynamic_features`.
        """
        # overwriting because _read_stn_dyn in this class returns xarray Dataset
        return self.__dyn_features

    # ------------------------------------------------------------------
    # Fast reader for the consolidated NetCDF caches.
    #
    # When ``all_stations_q.nc`` + ``all_stn_forcings.nc`` are present (the
    # standard, disk-consolidated layout), dynamic data is read directly with
    # netCDF4 and assembled with numpy. This replaces xarray's per-variable
    # ``concat``/``sel`` machinery — which took minutes for a few hundred
    # stations because its cost scales with the ~5,767 data variables in each
    # file — with a direct read that is byte-for-byte identical.
    # ------------------------------------------------------------------

    def _consolidated_ready(self) -> bool:
        """True when both consolidated caches exist, so the fast netCDF4 reader
        can be used instead of the per-station files."""
        return (os.path.exists(self.all_stations_q_path)
                and os.path.exists(self.all_stn_forcings_path))

    @property
    def _q_handle(self) -> "netCDF4.Dataset":
        return _shared_camelsh_nc(self.all_stations_q_path)

    @property
    def _forcing_handle(self) -> "netCDF4.Dataset":
        return _shared_camelsh_nc(self.all_stn_forcings_path)

    def _consolidated_axes(self):
        """
        Returns ``(time_index, q_feats, f_feats, q_col, f_col)`` for the
        consolidated caches, decoded once and cached.

        - ``time_index`` : the shared hourly :obj:`pandas.DatetimeIndex`
          (identical in both files).
        - ``q_feats`` / ``f_feats`` : mapped dynamic-feature names held by
          ``all_stations_q.nc`` / ``all_stn_forcings.nc``.
        - ``q_col`` / ``f_col`` : maps each mapped feature name to its column
          index within that file's per-station ``(time, dynamic_features)``
          variable.
        """
        if self._axes_cache is None:
            q_exists = os.path.exists(self.all_stations_q_path)
            f_exists = os.path.exists(self.all_stn_forcings_path)
            # the decoded time axis is identical in both files; use whichever
            # exists to build it once.
            time_handle = self._q_handle if q_exists else self._forcing_handle
            tv = time_handle.variables['time']
            time_index = pd.DatetimeIndex(
                netCDF4.num2date(
                    tv[:], tv.units, getattr(tv, 'calendar', 'standard'),
                    only_use_cftime_datetimes=False,
                    only_use_python_datetimes=True),
                name='time')
            q_feats = ([str(self.dyn_map.get(str(s), str(s)))
                        for s in self._q_handle.variables['dynamic_features'][:]]
                       if q_exists else [])
            f_feats = ([str(self.dyn_map.get(str(s), str(s)))
                        for s in self._forcing_handle.variables['dynamic_features'][:]]
                       if f_exists else [])
            q_col = {name: i for i, name in enumerate(q_feats)}
            f_col = {name: i for i, name in enumerate(f_feats)}
            self._axes_cache = (time_index, q_feats, f_feats, q_col, f_col)
        return self._axes_cache

    def _time_positions(self, st, en):
        """Maps a ``(st, en)`` window onto integer positions in the shared time
        index. ``time_index[i0:i1]`` is inclusive of both endpoints, matching
        :meth:`xarray.Dataset.sel` with a ``slice``."""
        time_index = self._consolidated_axes()[0]
        st, en = self._check_length(st, en)
        i0 = int(time_index.searchsorted(pd.Timestamp(st), side='left'))
        i1 = int(time_index.searchsorted(pd.Timestamp(en), side='right'))
        return time_index, i0, i1

    def _read_order(self) -> Dict[str, int]:
        """
        Cached ``{station: position}`` in the variable-creation order of the
        (compressed, chunked, hence seek-sensitive) forcing file. Reading a
        random station subset in this order turns scattered back-and-forth
        seeks into a mostly-forward scan; because the data lives on a spinning
        disk this measurably speeds up cold reads and never changes the result
        (the caller still receives a station-keyed dict).
        """
        if self._read_order_cache is None:
            handle = (self._forcing_handle
                      if os.path.exists(self.all_stn_forcings_path)
                      else self._q_handle)
            self._read_order_cache = {
                v: i for i, v in enumerate(
                    v for v in handle.variables if v not in handle.dimensions)}
        return self._read_order_cache

    def _read_consolidated(self, stations: List[str], feats: List[str], i0: int, i1: int):
        """
        Reads ``feats`` for ``stations`` over time positions ``[i0:i1]`` directly
        from the consolidated netCDF4 caches.

        Returns ``{station: ndarray(shape=(i1-i0, len(feats)), dtype=float32)}``
        with columns ordered exactly as ``feats``. NaN handling matches xarray:
        ``all_stations_q.nc`` variables carry a ``_FillValue`` (returned as a
        masked array, filled with NaN) while ``all_stn_forcings.nc`` stores NaNs
        directly, and ``np.ma.filled`` is correct for both.

        Reads are issued in on-disk order (see :meth:`_read_order`) to keep the
        spinning-disk access pattern sequential; a process pool is deliberately
        *not* used because the cold read is disk-seek bound (parallel workers do
        not speed it up and the IPC of shipping the decoded arrays back is a net
        loss — both measured).
        """
        _, _, _, q_col, f_col = self._consolidated_axes()
        need_q = any(f in q_col for f in feats)
        need_f = any(f in f_col for f in feats)
        qh = self._q_handle if need_q else None
        fh = self._forcing_handle if need_f else None
        n = i1 - i0
        order = self._read_order()
        read_seq = sorted(stations, key=lambda s: order.get(s, 0))
        out: Dict[str, np.ndarray] = {}
        for stn in read_seq:
            qa = np.ma.filled(qh.variables[stn][i0:i1], np.nan) if need_q else None
            fa = np.ma.filled(fh.variables[stn][i0:i1], np.nan) if need_f else None
            arr = np.empty((n, len(feats)), dtype='float32')
            for j, feat in enumerate(feats):
                col = q_col.get(feat)
                if col is not None:
                    arr[:, j] = qa[:, col]
                else:
                    arr[:, j] = fa[:, f_col[feat]]
            out[stn] = arr
        return out

    def _build_dyn_dataset(self, data: Dict[str, np.ndarray], feats: List[str], time_index):
        """Wraps ``{station: ndarray(time, dynamic_features)}`` into an xarray
        Dataset (data_vars = stations, dims = time × dynamic_features) without
        copying the arrays."""
        return xr.Dataset(
            {stn: (('time', 'dynamic_features'), arr) for stn, arr in data.items()},
            coords={'time': time_index, 'dynamic_features': list(feats)},
        )

    def _fetch_dynamic(self, stations: List[str], feats: List[str], st, en, as_dataframe: bool):
        """Fast entry point for dynamic reads from the consolidated caches.
        Returns a dict of per-station DataFrames (``as_dataframe=True``) or an
        xarray Dataset."""
        time_index, i0, i1 = self._time_positions(st, en)
        data = self._read_consolidated(stations, feats, i0, i1)
        tsel = time_index[i0:i1]
        if as_dataframe:
            out: Dict[str, pd.DataFrame] = {}
            for stn in stations:
                df = pd.DataFrame(data[stn], index=tsel, columns=list(feats))
                df.columns.name = 'dynamic_features'
                df.index.name = 'time'
                out[stn] = df
            return out
        return self._build_dyn_dataset(data, feats, tsel)

    def close(self):
        """Closes the process-wide consolidated-cache handles for this dataset,
        if open, and drops the decoded-axes cache."""
        for path in (self.all_stations_q_path, self.all_stn_forcings_path):
            handle = _SHARED_CAMELSH_NC.pop(path, None)
            if handle is not None:
                handle.close()
        self._axes_cache = None
        self._read_order_cache = None

    def _read_stn_q(self, stn):
        fpath = os.path.join(self.h2_path, f"{stn}_hourly.nc")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"q data for station {stn} not found in {self.h2_path}")
        
        ds = xr.open_dataset(fpath, engine='netcdf4')
        return ds

    def _read_stn_q1(self, stn):
        fpath = os.path.join(self.h2_path, f"{stn}_hourly.nc")
        if os.path.exists(fpath):
            dyn_map = {k:v for k,v in self.dyn_map.items() if k in ['streamflow']}
            ds = xr.open_dataset(fpath, engine='netcdf4').rename(dyn_map)
            return ds.to_array("dynamic_features").astype('float32').to_dataset(name=stn).transpose()

        # Fall back to the consolidated cache if the per-station file is gone.
        # Read via the shared netCDF4 handle (never xarray) so we never hold two
        # handles on the same multi-GB file at once.
        if os.path.exists(self.all_stations_q_path):
            time_index, q_feats, _, _, _ = self._consolidated_axes()
            data = self._read_consolidated([stn], q_feats, 0, len(time_index))
            return self._build_dyn_dataset(data, q_feats, time_index)

        raise FileNotFoundError(
            f"q data for station {stn} not found in {self.h2_path} "
            f"nor in {self.all_stations_q_path}"
        )

    def fetch_q(
            self, 
            stations:List[str] = "all"
            ):
        """
        Since fetching q from other methods can be slower because of merging with 
        other dynamic (forcing) features, this method fetches only observed streamflow 
        data for given stations using multiprocessing.

        Returns
        --------
        xr.Dataset
            xarray Dataset whose data variables are station names and dimensions 
            are time and dynamic features
        """
        stations = validate_attributes(stations, self.stations(), 'stations')

        all_q_fname = self.all_stations_q_path
        if os.path.exists(all_q_fname) and not self.overwrite:
            if self.verbosity>1:
                print(f"Loading q data for {len(stations)} stations from {all_q_fname}")
            # read the requested stations directly via the shared netCDF4 handle
            # (numpy assembly) instead of xarray, which is orders of magnitude
            # faster for a file holding thousands of data variables.
            time_index, q_feats, _, _, _ = self._consolidated_axes()
            data = self._read_consolidated(stations, q_feats, 0, len(time_index))
            return self._build_dyn_dataset(data, q_feats, time_index)

        cpus = self.processes or min(32, get_cpus())
        if self.verbosity>2: print(f"Using {cpus} processes to read q data of {len(stations)} stations")

        st = time.time()
        with cf.ProcessPoolExecutor(cpus) as executor:
            results = executor.map(self._read_stn_q1, stations)
        et = time.time()
        total = round(et - st, 2)
        if self.verbosity: print(f"read q for {len(stations)} stations in {total} secs with {cpus} processes")

        results = xr.merge(results)

        if len(stations) == len(self.stations()):
            # saving it so that next time we don't have to read all stations separately again
            if self.verbosity>1:
                print(f"Saving all stations q data to {all_q_fname}")
            results.to_netcdf(all_q_fname, engine='netcdf4',
                              encoding={stn: {'dtype': 'float32'} for stn in stations})
        return results

    def _read_stn_forcing1(self, stn):
        """
        Returns
        -------
        xr.Dataset
            xarray Dataset with 'time' and 'dynamic_features' dimensions and 
            stn as data variable
        """
        fpath = os.path.join(self.nonobs_path, f"{stn}.nc")
        if not os.path.exists(fpath):
            fpath = os.path.join(self.timeseries_path, f"{stn}.nc")
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"forcing data for station {stn} not found in {self.nonobs_path}")
        
        ds = xr.open_dataset(fpath, engine='netcdf4')

        # todo : what is difference between Streamflow in forcing and streamflow in Hourly2 path?
        ds = ds.drop_vars("Streamflow", errors="ignore")

        ds = ds.rename({'DateTime': 'time'})
        return ds.to_array("dynamic_features").transpose().astype('float32').to_dataset(name=stn)

    def _read_stns_forcing(self, stations:List[str]):
        """retunrs forcings of multiple stations as xarray Dataset."""
        assert isinstance(stations, list), "stations should be a list of station names/ids"

        all_stn_forcings = self.all_stn_forcings_path
        if os.path.exists(all_stn_forcings) and not self.overwrite:
            if self.verbosity>1:
                print(f"Loading forcing data for {len(stations)} stations from {all_stn_forcings}")
            # direct netCDF4 + numpy assembly (see fetch_q); the feature names are
            # already mapped via dyn_map inside _consolidated_axes.
            time_index, _, f_feats, _, _ = self._consolidated_axes()
            data = self._read_consolidated(stations, f_feats, 0, len(time_index))
            return self._build_dyn_dataset(data, f_feats, time_index)

        cpus = self.processes or min(32, get_cpus())
        st = time.time()

        if self.verbosity>2: print(f"Using {cpus} processes to read forcing data of {len(stations)} stations")

        with cf.ProcessPoolExecutor(cpus) as executor:
            results = executor.map(self._read_stn_forcing1, stations)
        et = time.time()
        total = round(et - st, 2)
        if self.verbosity: print(f"read {len(stations)} stations forcing data in {total} secs with {cpus} processes")
        results = xr.merge(results)

        new_dyn = [str(self.dyn_map.get(v, v)) for v in results["dynamic_features"].data]
        results = results.assign_coords(dynamic_features=("dynamic_features", new_dyn))
        return results

    def _read_stn_forcing(self, stn):
        """prefers reading from all_stn_forcings_path if exists."""
        assert isinstance(stn, str), "station name/id should be a string"
    
        all_stn_forcings = self.all_stn_forcings_path
        if os.path.exists(all_stn_forcings) and not self.overwrite:
            if self.verbosity>1:
                print(f"Loading forcing data for {stn} from {all_stn_forcings}")
            time_index, _, f_feats, _, _ = self._consolidated_axes()
            data = self._read_consolidated([stn], f_feats, 0, len(time_index))
            return self._build_dyn_dataset(data, f_feats, time_index)

        fpath = os.path.join(self.nonobs_path, f"{stn}.nc")
        if not os.path.exists(fpath):
            fpath = os.path.join(self.timeseries_path, f"{stn}.nc")
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"forcing data for station {stn} not found in {self.nonobs_path}")
        
        ds = xr.open_dataset(fpath, engine='netcdf4')

        # todo : what is difference between Streamflow in forcing and streamflow in Hourly2 path?
        ds = ds.drop_vars("Streamflow", errors="ignore")

        ds = ds.rename({'DateTime': 'time'})
    
        new_dyn = {k:str(self.dyn_map.get(k, k)) for k in ds.data_vars}
        return ds.rename(new_dyn).to_array("dynamic_features").transpose().astype('float32').to_dataset(name=stn)

    def _read_stns_dyn(self, stns:List[str]):
        """
        Returns
        -------
        xr.Dataset
        """
        if self._consolidated_ready():
            time_index, q_feats, f_feats, _, _ = self._consolidated_axes()
            feats = q_feats + f_feats
            data = self._read_consolidated(stns, feats, 0, len(time_index))
            return self._build_dyn_dataset(data, feats, time_index)

        q = self.fetch_q(stns)
        forcing = self._read_stns_forcing(stns)

        ds = xr.concat([q, forcing], dim='dynamic_features')

        return ds

    def _read_stn_dyn(self, stn:str, nrows=None) -> pd.DataFrame:
        if self._consolidated_ready():
            time_index, q_feats, f_feats, _, _ = self._consolidated_axes()
            feats = q_feats + f_feats
            data = self._read_consolidated([stn], feats, 0, len(time_index))
            return self._build_dyn_dataset(data, feats, time_index)

        q = self._read_stn_q1(stn)
        forcing = self._read_stn_forcing(stn)

        ds = xr.concat([q, forcing], dim='dynamic_features')

        return ds

    def _static_data(self) -> pd.DataFrame:
        """
        reads static data for all stations. The assembled table is cached after
        the first read because it is consulted many times (static_features,
        stn_coords, area, every static fetch) and re-reading ~30 CSVs each time
        was a needless repeated cost.
        """
        if self._static_cache is not None:
            return self._static_cache

        csv_files = glob.glob(os.path.join(self.attr_path, '*.csv'))

        dfs = []
        for csv_file in csv_files:

            if 'attributes_hydroATLAS.csv' in csv_file:
                df = pd.read_csv(
                csv_file, 
                index_col=0, 
                sep='\t',
                dtype={0: str})
            else:
                df = pd.read_csv(
                csv_file, 
                index_col=0, 
                dtype={0: str})
            df.index = df.index.astype(str)
            dfs.append(df)

        df = pd.concat(dfs, axis=1)

        # drop duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]

        # rename columns using self.static_map
        df = df.rename(columns=self.static_map)
        self._static_cache = df
        return df

    def fetch_stations_features(
            self,
            stations: Union[str, List[str]],
            dynamic_features: Union[str, List[str]] = 'all',
            static_features: Union[str, List[str]] = None,
            st: Union[str, pd.Timestamp] = None,
            en: Union[str, pd.Timestamp] = None,
            as_dataframe: bool = False,
            **kwargs
              ) -> Tuple[pd.DataFrame, Union[Dict[str, pd.DataFrame], "Dataset"]]:
        """
        Reads features of more than one stations.

        parameters
        ----------
        stations :
            list of stations for which data is to be fetched.
        dynamic_features :
            list of dynamic features to be fetched.
            if ``all``, then all dynamic features will be fetched.
        static_features : list of static features to be fetched.
            If ``all``, then all static features will be fetched. If None,
            `then no static attribute will be fetched.
        st :
            start of data to be fetched.
        en :
            end of data to be fetched.
        as_dataframe :
            whether to return the dynamic data as pandas dataframe. default
            is :obj:`xarray.Dataset` object
        kwargs dict:
            additional keyword arguments

        Returns
        -------
        tuple
            A tuple of static and dynamic features. Static features are always
            returned as :obj:`pandas.DataFrame` with shape (stations, staticfeatures).
            The index of static features is the station/gauge ids while the columns 
            are the static features. Dynamic features are returned as either
            :obj:`xarray.Dataset` or a :obj:`dict` with keys as station names and values as
            :obj:`pandas.DataFrame` depending upon whether `as_dataframe`
            is True or False and whether the xarray module is installed or not.
            If dynamic features are xarray Dataset, then it consists of `data_vars`
            equal to the number of stations and `time` and `dynamic_features` as
            dimensions.

        Raises:
            ValueError, if both dynamic_features and static_features are None

        Examples
        --------
        >>> from aqua_fetch import CAMELSH
        >>> dataset = CAMELSH()
        ... # find out station ids
        >>> dataset.stations()
        ... # get data of selected stations as xarray Dataset
        >>> dataset.fetch_stations_features(['01141800', '02349900', '11062000'])
        ... # get data of selected stations as dictionary of pandas DataFrame
        >>> dataset.fetch_stations_features(['01141800', '02349900', '11062000'],
        ...  as_dataframe=True)
        ... # get both dynamic and static features of selected stations
        >>> dataset.fetch_stations_features(['01141800', '02349900', '11062000'],
        ... dynamic_features=['q_mm_obs', 'air_temp_C', 'pcp_mm'], static_features=['elev_catch_m'])
        """
        # overwriting because _read_stn_dyn in this class returns xarray Dataset

        if xr is None:
            if not as_dataframe:
                if self.verbosity: warnings.warn("xarray module is not installed so as_dataframe will have no effect. "
                              "Dynamic features will be returned as pandas DataFrame")
                as_dataframe = True

        st, en = self._check_length(st, en)
        static, dynamic = None, None

        stations = validate_attributes(stations, self.stations(), 'stations')

        if dynamic_features is not None:

            dynamic_features = validate_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')

            if self._consolidated_ready():
                # fast path: read/assemble directly from the consolidated caches,
                # building DataFrames without an intermediate xarray Dataset.
                dynamic = self._fetch_dynamic(stations, dynamic_features, st, en, as_dataframe)
            else:
                dynamic = self._read_dynamic(stations, dynamic_features, st=st, en=en)
                if as_dataframe:
                    # convert xarray Dataset to dictionary of pandas DataFrame
                    dynamic = {stn: dynamic[stn].to_pandas() for stn in stations}

            if static_features is not None:
                static = self.fetch_static_features(stations, static_features)

        elif static_features is not None:

            return self.fetch_static_features(stations, static_features), dynamic

        else:
            raise ValueError(f"static features are {static_features} and dynamic features are {dynamic_features}")

        return static, dynamic

    def _read_dynamic(
            self, 
            stations, 
            dynamic_features, 
            st:Union[str, pd.Timestamp] = None, 
            en:Union[str, pd.Timestamp] = None
            ):
        """
        Returns
        -------
        xr.Dataset
        """
        # overwriting because here we  always return xarray Dataset

        st, en = self._check_length(st, en)
        dyn_feats = validate_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')
        stations = validate_attributes(stations, self.stations(), 'stations')

        if self._consolidated_ready():
            # fast path: direct netCDF4 read + numpy assembly, already sliced to
            # the requested features and time window.
            return self._fetch_dynamic(stations, dyn_feats, st, en, as_dataframe=False)

        # fallback (per-station files; consolidated caches not built yet). There
        # can be 3 scenarios.
        if len(dyn_feats) == 1 and dyn_feats[0] == observed_streamflow_cms():
            # only q is asked
            results = self.fetch_q(stations)

        elif observed_streamflow_cms() not in dyn_feats and 'water_level' not in dyn_feats:
            # only forcing data is asked
            results = self._read_stns_forcing(stations)
        else:
            # both q and forcing data is asked
            if len(stations) > 1:
                results = self._read_stns_dyn(stations)
            else:
                results = self._read_stn_dyn(stations[0])

        # select required dynamic features and time range
        results = results.sel(dynamic_features=dyn_feats, time=slice(st, en))
        return results

    def collate_forcing_data(self):
        """
        Collate forcing data of all stations into a single NetCDF file using multiprocessing.

        """
        stations = self.stations()

        cpus = self.processes or min(32, get_cpus())

        out_path = self.all_stn_forcings_path
        out_path = Path(out_path)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

        if tmp_path.exists():
            tmp_path.unlink()
        if out_path.exists():
            out_path.unlink()  # start clean; or implement resume logic

        # Launch workers
        first_result = None
        results_queue = []
        st = time.time()

        with cf.ProcessPoolExecutor(max_workers=cpus) as ex:
            futures = [ex.submit(self._worker, i) for i in stations]

            # We’ll create the .nc after the first finished task (to learn dims/dtypes)
            for fut in cf.as_completed(futures):
                run_id, time_int, dyn, arr = fut.result()
                if first_result is None:
                    first_result = (run_id, time_int, dyn, arr)
                    break
                else:
                    results_queue.append((run_id, time_int, dyn, arr))

            # Initialize NetCDF using the first result
            fr_run_id, fr_time, fr_dyn, fr_arr = first_result

            # Create temp file to ensure crash-safety; rename at the end
            with netCDF4.Dataset(tmp_path, "w", format="NETCDF4") as nc:
                # Define dimensions
                T = int(fr_time.shape[0])
                D = int(fr_dyn.shape[0])
                nc.createDimension("time", T)
                nc.createDimension("dynamic_features", D)

                # Coordinate variables
                v_time = nc.createVariable("time", "f8", ("time",))
                v_time[:] = fr_time
                v_time.long_name = "time"
                v_time.units = "hours since 1970-01-01 00:00:00"
                v_time.calendar = "proleptic_gregorian"

                # NetCDF4 supports variable-length strings via dtype=str
                v_dyn = nc.createVariable("dynamic_features", str, ("dynamic_features",))
                v_dyn[:] = fr_dyn.data
                v_dyn.long_name = "dynamic feature names"

                # Create **all** data variables up-front (fast metadata, RAM-free)
                # Compression & chunking recommended
                chunks_t = min(T, 1024)
                var_handles = {}
                for stn in stations:
                    vname = f"{stn}"
                    var_handles[stn] = nc.createVariable(
                        vname,
                        fr_arr.dtype,
                        ("time", "dynamic_features"),
                        zlib=True,
                        complevel=4,
                        shuffle=True,
                        chunksizes=(chunks_t, D),
                    )
                    # Optional attrs (for CF/xarray friendliness)
                    var_handles[stn].coordinates = "dynamic_features time"

                # Write the first result
                var_handles[fr_run_id][:] = fr_arr
                nc.sync()

                # Consume already-completed results
                for run_id, time_f, dyn, arr in results_queue:
                    _validate_coords(run_id, time_f, dyn, T, D)
                    var_handles[run_id][:] = arr
                    nc.sync()

                # Consume the rest as they finish
                for fut in cf.as_completed(futures):
                    # Some futures were already consumed; skip them gracefully
                    try:
                        run_id, time_f, dyn, arr = fut.result()
                    except Exception as e:
                        warnings.warn(f"[worker] a forcing-data task failed and its "
                                      f"station was skipped: {e}")
                        continue
                    # Skip the one we already wrote
                    if run_id == fr_run_id:
                        continue
                    _validate_coords(run_id, time_f, dyn, T, D)
                    var_handles[run_id][:] = arr
                    nc.sync()

            # Atomic finalize
            os.replace(tmp_path, out_path)
        et = time.time()
        total = round(et - st, 2)
        if self.verbosity: print(f"collated {len(stations)} stations in {total} secs with {cpus} processes")
        return

    def _worker(self, stn_id: str):
        out = self._read_stn_forcing1(stn_id)
        # Return minimal payload; main proc writes to disk
        return stn_id, _encode_time_for_nc(out["time"].data), out["dynamic_features"], out[stn_id].values

    def q_mm(
            self,
            stations: Union[str, List[str]] = "all",
            as_dataframe: bool = True
    ) -> pd.DataFrame:
        """
        returns streamflow in the units of milimeter per timestep (mm/hour). This is obtained
        by diving ``q`` by area.

        parameters
        ----------
        stations : str/list
            name/names of stations. Default is ``all``, which will return
            q_mm of all stations
        as_dataframe : bool
            whether to return the data as pandas DataFrame. Default is True.
            Setting it to False will return xarray Dataset and can be faster.

        Returns
        --------
        pd.DataFrame or xr.Dataset
            a :obj:`pandas.DataFrame` whose indices are time-steps and columns
            are catchment/station ids.

        """
        # overwriting because we don't want to call fetch, which can be slow
        # instead we directly call .q method

        stations = validate_attributes(stations, self.stations(), 'stations')

        q = self.fetch_q(stations)
        q = q.sel(dynamic_features='q_cms_obs')
        if as_dataframe:
            q = q.to_pandas().drop(columns=['dynamic_features'], errors='ignore')

        area_m2 = self.area(stations) * 1e6  # area in m2

        time_conversion = 3600  # seconds per hour

        q = (q / area_m2) * time_conversion  # cms to m
        return q * 1e3  # to mm

    def _redundant_after_consolidation(self) -> List[Tuple[str, str]]:
        return [
            (os.path.join(self.path, "Hourly2"),           self.all_stations_q_path),
            (os.path.join(self.path, "timeseries"),        self.all_stn_forcings_path),
            (os.path.join(self.path, "timeseries_nonobs"), self.all_stn_forcings_path),
        ]


def _validate_coords(run_id, time_f, dyn, T, D):
    if time_f.shape[0] != T or dyn.shape[0] != D:
        raise ValueError(
            f"Run {run_id} dims differ: got {(time_f.shape[0], dyn.shape[0])}, expected {(T, D)}"
        )


def _encode_time_for_nc(time64: np.ndarray,
                       units="hours since 1970-01-01 00:00:00",
                       calendar="proleptic_gregorian") -> np.ndarray:
    # convert to Python datetimes (vectorized via pandas)
    py_dt = pd.to_datetime(time64).to_pydatetime()
    return date2num(py_dt, units=units, calendar=calendar).astype("float64")


# Process-wide, read-only netCDF4 handles for the two consolidated CAMELSH
# caches (``all_stations_q.nc`` ~18 GB and ``all_stn_forcings.nc`` ~67 GB).
# Re-opening these multi-GB HDF5 files on every fetch — or holding an xarray
# handle and a netCDF4 handle on the *same* file at once — can segfault HDF5
# (the same hazard fixed for EStreams' meteorology.nc). Keeping a single shared
# handle per path, opened once and never mixed with xarray, avoids both.
_SHARED_CAMELSH_NC: Dict[str, "netCDF4.Dataset"] = {}


def _shared_camelsh_nc(path: str) -> "netCDF4.Dataset":
    """Returns the process-wide read-only netCDF4 handle for ``path``, opening
    it once on first use."""
    handle = _SHARED_CAMELSH_NC.get(path)
    if handle is None:
        handle = netCDF4.Dataset(path, "r")
        _SHARED_CAMELSH_NC[path] = handle
    return handle
