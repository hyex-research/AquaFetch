import os
import glob
import time
import struct
import warnings
import functools
import concurrent.futures as cf
from typing import Union, List, Dict, Tuple

import numpy as np
import pandas as pd

from ..utils import _RainfallRunoff, n_workers, cache_name, apply_dyn_factors
from ...utils import validate_attributes
from ..._backend import netCDF4, xarray as xr
from .._map import (
    observed_streamflow_cms,
    observed_streamflow_mm,
    total_precipitation_with_specifier,
    mean_air_temp_with_specifier,
    min_air_temp_with_specifier,
    max_air_temp_with_specifier,
    total_potential_evapotranspiration_with_specifier,
    mean_potential_evaporation_with_specifier,
    snow_water_equivalent,
    min_snow_water_equivalent,
    max_snow_water_equivalent,
    net_solar_radiation,
    min_net_solar_radiation,
    max_net_solar_radiation,
    net_longwave_radiation,
    min_net_longwave_radiation,
    max_net_longwave_radiation,
    mean_dewpoint_temperature,
    min_dewpoint_temperature,
    max_dewpoint_temperature,
    u_component_of_wind_at_10m,
    v_component_of_wind_at_10m,
    u_component_of_wind_with_specifier,
    v_component_of_wind_with_specifier,
    soil_moisture_layer1,
    soil_moisture_layer2,
    soil_moisture_layer3,
    soil_moisture_layer4,
    soil_moisture_layer_with_specifier,
    mean_air_pressure,
)
from .._map import (
    catchment_area,
    gauge_latitude,
    gauge_longitude,
    catchment_elevation_meters,
    min_catchment_elevation_meters,
    max_catchment_elevation_meters,
    slope,
)
from ._common import (_extract_zip, _remove_stale, _warn_duplicate_gauges,
                      _first_and_last_row, _row_date)


class CAMELS_CZ(_RainfallRunoff):
    """
    Daily and hourly streamflow and meteorology of 249 catchments in Czechia
    following `Krejčí and Nearing, 2025 <https://doi.org/10.5281/zenodo.17769325>`_
    (CAMELS-CZ, in Caravan format).

    The streamflow is the discharge observed by the Czech Hydrometeorological
    Institute (CHMI), in mm per time step. The daily values are the
    `CHMI <https://opendata.chmi.cz/hydrology/historical/>`_ discharge times
    86.4 / ``area_km2``, rounded to 0.01 mm; the hourly ones its hourly
    discharge times 3.6 / ``BasinArea`` of the boundary shapefile, which is
    0.3 % larger than ``area_km2`` (checked at B4403000, B4440000 and
    P4226000). ``q_cms_obs`` is converted back with the same areas.

    With ``timestep='D'`` (default) there are 46 dynamic features from
    1950-01-02 to 2025-07-29 (27603 days). Streamflow and ERA5-Land come from
    ``Caravan-Extension-CZ.zip``; the five CHMI series from ``CAMELS-CZ.zip``,
    which starts on 1978-06-30 (47 gauges), 1980-01-01 (201) or 1994-01-01 (1)
    and ends on 2025-01-01. Its own streamflow and ERA5-Land columns are
    identical to those of the Caravan extension, which has 47 % more days of
    observed streamflow. With ``timestep='H'`` there are 18 dynamic features
    from 2016-06-16 22:00 to 2025-07-08 06:00 UTC (79401 hours); the 193 files
    that end on 2025-07-07 11:00 are padded with NaN.

    Values keep their source units, which the feature names show, except
    ``airpres_hpa`` (mean surface pressure, converted from kPa). Not shown by
    the names: ``q_mm_obs`` is in mm per time step, ``sml1`` to ``sml4``
    (soil water) in m3 m-3, the daily ``surface_pressure_min`` and
    ``surface_pressure_max`` keep their raw names and kPa, and the net
    radiations are positive downward. Apart from the streamflow and the
    features with ``chmi`` in their names, all are ERA5-Land. ``pet_mm_*`` is
    FAO Penman-Monteith; ``pevap_mm_era5land`` is ERA5-Land's potential
    evaporation.

    A daily ERA5-Land value is the sum or mean of the hourly ones stamped
    23:00 UTC of the day before to 22:00 UTC; a daily ``pcp_mm_chmi`` matches
    the sum of the hourly ones stamped 07:00 to 06:00 UTC of the next day
    (median r 0.999, lowest 0.96).

    The 215 static features are the 5 location attributes, 14 Caravan climate
    indices and 196 HydroATLAS attributes.

    .. note::
        Examples of source data issues, served unchanged:

        - daily ``q_mm_obs`` is 0.00 on 3847 days at 41 gauges, where CHMI
          reports flow below 0.005 mm (e.g. 0.02-0.04 m3 s-1 at B4440000 in 2018).
        - hourly ``q_mm_obs`` is negative in 112 hours (down to -0.33 mm h-1);
          O4350000 has no hourly streamflow.
        - C4128000 is on the Nová řeka, a canal fed from the Lužnice: its runoff
          is ~3.5 times its precipitation. 17 gauges are at dams (CHMI names
          "VD ...", e.g. B8479500), so their flow is regulated.
        - the daily streamflow of 24042409 is the discharge of CHMI's Žatec
          gauge (3944 km2), but its coordinates, boundary and ``area_km2``
          (3579 km2) are those of Stranná, 7.7 km upstream, so its
          ``q_mm_obs`` is ~10 % too high; its hourly streamflow differs from
          Žatec's (r = 0.96).
        - the CHMI temperatures do not fall with elevation: their 1990-2020
          mean is 10.3-12.7 °C at every catchment and 3.4 °C above ERA5-Land
          on average (5.8-9.5 °C, -4.3 °C per km). ``airtemp_C_chmi_max``
          exceeds the Czech record of 40.4 °C on 89 station-days, 58 of them on
          1983-07-27 (ERA5-Land: ~32 °C), and ``airtemp_C_mean_chmi`` lies
          outside [min, max] on 969 station-days. ``pet_mm_chmi`` (FAO
          Penman-Monteith from CHMI data) may inherit this.
        - ``pevap_mm_era5land`` reaches 39.75 mm day-1 (P4088000).
        - the zenodo record states CC-BY-4.0, the license file in the archives
          links to CC-BY-NC-4.0.

    :py:class:`aqua_fetch.rr.GRDCCaravan` has 26 of these gauges under GRDC
    ids (``common_stations(grdc_caravan, max_dist_km=0.5)``);
    :py:class:`aqua_fetch.rr.EStreams` lists 566 Czech CHMI gauges but has no
    streamflow.

    The first initialization took ~10 min for ``D`` (1.6 GB download, 2.7 GB
    extracted, 1.3 GB netCDF cache) and ~10 min for ``H`` (2.3 GB, 3.7 GB and
    1.4 GB). Afterwards initialization takes ~0.03 s, and fetching all features
    of all stations ~0.5 s from the cache (~10 s when it is first read from a
    network drive), ~3 s from the csv files with a process pool and ~25 s with
    ``processes=1``; one station takes ~0.05 s. With the ``spawn`` or
    ``forkserver`` start method (Windows, macOS, Linux from Python 3.14), run
    scripts under ``if __name__ == "__main__":`` because reading many stations
    uses a process pool.

    Examples
    --------
    >>> from aqua_fetch import CAMELS_CZ
    >>> dataset = CAMELS_CZ()
    >>> len(dataset.stations())
    249
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='B4403000', as_dataframe=True)
    >>> dynamic['B4403000'].shape
    (27603, 46)
    ... # get data of 10 % of (randomly selected) stations
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)
    24
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('B4403000', as_dataframe=True,
    ...  dynamic_features=['pcp_mm_chmi', 'airtemp_C_mean_chmi', 'pet_mm_chmi', 'q_cms_obs'])
    >>> dynamic['B4403000'].shape
    (27603, 4)
    ... # get data between two dates
    >>> _, dynamic = dataset.fetch('B4403000', st='2020-01-01', en='2020-12-31', as_dataframe=True)
    >>> dynamic['B4403000'].shape
    (366, 46)
    ... # get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='B4403000', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['B4403000'].shape
    ((1, 215), 1, (27603, 46))
    >>> dataset.fetch_static_features('B4403000', ['area_km2', 'slope_degrees', 'p_mean'])
                 area_km2  slope_degrees    p_mean
    gauge_id
    B4403000  6989.381996       5.761766  2.199431
    ... # without as_dataframe=True (and with xarray installed) an xarray Dataset is returned
    >>> _, dynamic = dataset.fetch(10)
    >>> dynamic.sizes
    Frozen({'time': 27603, 'dynamic_features': 46})
    >>> dataset.stn_coords().shape
    (249, 2)
    >>> dataset.area('B4403000')  # km2
    gauge_id
    B4403000    6989.381836
    Name: area_km2, dtype: float32
    >>> dataset.q_mm('B4403000').shape  # observed streamflow in mm day-1
    (27603, 1)
    ... # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('B4403000')['type']
    'Polygon'
    ... # hourly data
    >>> hourly = CAMELS_CZ(timestep='H')
    >>> _, dynamic = hourly.fetch('B4403000', as_dataframe=True)
    >>> dynamic['B4403000'].shape
    (79401, 18)
    """
    url = "https://zenodo.org/records/17769325"

    def __init__(
            self,
            path: str = None,
            timestep: str = 'D',
            overwrite: bool = False,
            to_netcdf: bool = True,
            verbosity: int = 1,
            **kwargs
    ):
        """
        Parameters
        ----------
        path : str
            folder in which the ``CAMELS_CZ`` folder is (or will be) created.
            If None, the default aqua_fetch data folder is used.
        timestep : str
            ``D`` (default) for daily or ``H`` for hourly data. ``D`` downloads
            ``Caravan-Extension-CZ.zip`` and ``CAMELS-CZ.zip`` (1.6 GB), ``H``
            downloads ``CAMELS-CZ-Hourly.zip`` (2.3 GB).
        overwrite : bool
            If True, the archives, the extracted data and the netCDF cache of
            this ``timestep`` are deleted, downloaded and built again.
        to_netcdf : bool
            whether to store all dynamic data of this ``timestep`` in a netCDF
            file (:attr:`dyn_fpath`) for faster fetching. Requires netCDF4 and
            xarray. Once the file exists, the data is read from it even with
            ``to_netcdf=False``.
        verbosity : int
            0: no message will be printed
        kwargs :
            passed to the parent class e.g. ``processes`` (``processes=1``
            disables multiprocessing) or ``remove_zip`` (delete the archives
            after extraction).
        """
        if timestep not in _CZ_PRODUCTS:
            raise ValueError(f"timestep must be 'D' or 'H' but is {timestep!r}")

        super().__init__(path=path, timestep=timestep, overwrite=overwrite,
                         to_netcdf=to_netcdf, verbosity=verbosity, **kwargs)

        # lazy caches
        self._attrs = {}
        self._static_df = None

        self._download_camels_cz(overwrite=overwrite)

        self._check_manifest()

        self._warn_unserved_gauges()

        # the source has no gauge names (gauge_name is the id), so duplicates
        # are searched by the rounded coordinates alone
        coords = self._static_table()[[gauge_latitude(), gauge_longitude()]]
        _warn_duplicate_gauges(self.name, pd.DataFrame({
            'gauge_id': coords.index, 'gauge_name': '',
            'gauge_lat': coords[gauge_latitude()].values,
            'gauge_lon': coords[gauge_longitude()].values}))

        self._maybe_to_netcdf()

        self.bbox = {'llcrnrlat': 48.5, 'urcrnrlat': 51.1,
                     'llcrnrlon': 12.0, 'urcrnrlon': 19.0}
        self.parallels = range(48, 52, 1)
        self.meridians = range(12, 20, 2)

    # ------------------------------------------------------------------
    # download
    # ------------------------------------------------------------------

    @property
    def _products(self) -> Tuple[str, ...]:
        """names of the archives (without .zip) this ``timestep`` reads"""
        return _CZ_PRODUCTS[self.timestep]

    def _download_camels_cz(self, overwrite: bool = False):
        """
        Downloads and extracts the archives of this ``timestep``. An archive
        whose extracted folder exists is neither downloaded nor extracted, even
        if the archive itself was deleted (``remove_zip=True``). The per-gauge
        netCDF copies of the csv files are not extracted.
        """
        if overwrite:
            stale = glob.glob(os.path.join(glob.escape(self.path),
                                           f"{self.name.lower()}_{self.timestep}_*.nc"))
            for product in self._products:
                stale += [os.path.join(self.path, f"{product}.zip"),
                          os.path.join(self.path, product)]
            _remove_stale(stale, self.verbosity)

        todo = [p for p in self._products if not os.path.isdir(os.path.join(self.path, p))]
        if not todo:
            if self.verbosity:
                print(f"CAMELS_CZ {self.timestep} data already exists at {self.path}")

        missing = [f"{p}.zip" for p in todo if not os.path.exists(os.path.join(self.path, f"{p}.zip"))]
        if missing:
            # imported here because this module installs a SIGINT handler on import
            from ...download_zenodo import download_from_zenodo
            os.makedirs(self.path, exist_ok=True)
            download_from_zenodo(self.path, doi=self.url, include=missing,
                                 verbosity=self.verbosity)

        for product in todo:
            _extract_zip(os.path.join(self.path, f"{product}.zip"),
                         os.path.join(self.path, product), self.verbosity,
                         skip=(f"{product}/timeseries/netcdf/",))

        if self.remove_zip:
            _remove_stale([os.path.join(self.path, f"{p}.zip") for p in self._products],
                          self.verbosity, reason="remove_zip=True")
        return

    def _check_manifest(self):
        """
        Warns if a file this ``timestep`` reads is missing, e.g. after an
        interrupted extraction. The expected time series files follow from the
        station list, not from what is on disk.
        """
        if not os.path.exists(self._attr_path('caravan')):
            raise FileNotFoundError(
                f"{self._attr_path('caravan')} not found. Re-initialize CAMELS_CZ "
                f"with overwrite=True.")

        boundary = os.path.splitext(self.boundary_file)[0]
        missing = [f for f in [self._attr_path(kind) for kind in _CZ_ATTR_KINDS]
                   + [f"{boundary}{ext}" for ext in ('.shp', '.dbf', '.shx', '.prj')]
                   if not os.path.exists(f)]

        for product in self._products:
            folder = self._ts_dir(product)
            present = set(os.listdir(folder)) if os.path.isdir(folder) else set()
            missing += [os.path.join(folder, _cz_fname(stn)) for stn in self.stations()
                        if _cz_fname(stn) not in present]

        if missing:
            warnings.warn(
                f"CAMELS_CZ {self.timestep}: {len(missing)} files are missing, e.g. "
                f"{missing[:3]}. The data is incomplete; re-initialize with overwrite=True.",
                UserWarning)
        return

    def _warn_unserved_gauges(self):
        """gauges with attributes but without time series are not served"""
        served = set(self.stations())
        unserved = [stn for stn in self._read_attrs('other').index if stn not in served]
        if unserved:
            warnings.warn(
                f"CAMELS_CZ: {unserved} have attributes but no time series and are "
                f"not served.", UserWarning)
        return

    # ------------------------------------------------------------------
    # paths
    # ------------------------------------------------------------------

    def _root(self, product: str) -> os.PathLike:
        """folder into which ``<product>.zip`` is extracted; the archive holds a
        single top-level folder of the same name"""
        return os.path.join(self.path, product, product)

    def _ts_dir(self, product: str) -> os.PathLike:
        return os.path.join(self._root(product), "timeseries", "csv", "camelscz")

    @property
    def _attr_product(self) -> str:
        """the archive whose attributes and boundaries are read"""
        return _CZ_ATTR_PRODUCT[self.timestep]

    def _attr_path(self, kind: str) -> os.PathLike:
        return os.path.join(self._root(self._attr_product), "attributes", "camelscz",
                            f"attributes_{kind}_camelscz.csv")

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(self._root(self._attr_product), "shapefiles", "camelscz",
                            "camelscz_basin_shapes.shp")

    @property
    def boundary_id_map(self) -> str:
        return "gauge_id"

    def _boundary_catch_id(self, value) -> str:
        return str(value).removeprefix(_CZ_PREFIX)

    @property
    def dyn_fname(self) -> str:
        """name of the netCDF cache, one per ``timestep`` and precision, e.g.
        camels_cz_D_v2.nc (float32) or camels_cz_D_float64_v2.nc"""
        precision = "" if np.dtype(self.fp) == np.float32 else f"_{np.dtype(self.fp).name}"
        return cache_name(f"{self.name.lower()}_{self.timestep}{precision}.nc")

    # ------------------------------------------------------------------
    # stations and static data
    # ------------------------------------------------------------------

    def _read_attrs(self, kind: str) -> pd.DataFrame:
        """one attribute file, indexed by the gauge id without its
        ``camelscz_`` prefix. Cached; never hand it out without copying."""
        if kind not in self._attrs:
            df = pd.read_csv(self._attr_path(kind), dtype={'gauge_id': str, 'gauge_name': str},
                             float_precision='round_trip')
            ids = df.pop('gauge_id')
            if not ids.str.startswith(_CZ_PREFIX).all():
                raise ValueError(f"unexpected gauge ids in {self._attr_path(kind)}: "
                                 f"{ids[~ids.str.startswith(_CZ_PREFIX)].tolist()[:5]}")
            df.index = pd.Index(ids.str.removeprefix(_CZ_PREFIX), name='gauge_id')
            self._attrs[kind] = df
        return self._attrs[kind]

    def stations(self) -> List[str]:
        """ids of the gauges with time series, from attributes_caravan_camelscz.csv"""
        return self._read_attrs('caravan').index.tolist()

    @property
    def static_map(self) -> Dict[str, str]:
        return {
            'area': catchment_area(),                         # km2
            'gauge_lat': gauge_latitude(),                    # deg N (WGS84)
            'gauge_lon': gauge_longitude(),                   # deg E (WGS84)
            'ele_mt_sav': catchment_elevation_meters(),       # m a.s.l.
            'ele_mt_smn': min_catchment_elevation_meters(),
            'ele_mt_smx': max_catchment_elevation_meters(),
            'slp_dg_sav': slope('degrees'),                   # degrees x 10 in HydroATLAS
        }

    @property
    def static_factors(self) -> Dict[str, float]:
        return {slope('degrees'): 0.1}

    def _static_table(self) -> pd.DataFrame:
        """the cached table of all static features of the served gauges"""
        if self._static_df is None:
            stns = self.stations()
            df = pd.concat([self._read_attrs(kind).loc[stns] for kind in _CZ_ATTR_KINDS], axis=1)
            df.rename(columns=self.static_map, inplace=True)
            for col, factor in self.static_factors.items():
                df[col] = df[col] * factor
            self._static_df = df
        return self._static_df

    def _static_data(self) -> pd.DataFrame:
        # a copy, so that a caller's in-place edit cannot corrupt the cache
        return self._static_table().copy()

    @property
    def static_features(self) -> List[str]:
        return self._static_table().columns.tolist()

    @property
    def _mm_feature_name(self) -> str:
        return observed_streamflow_mm()

    @property
    def _area_name(self) -> str:
        return catchment_area()

    @property
    def _coords_name(self) -> List[str]:
        return [gauge_latitude(), gauge_longitude()]

    # ------------------------------------------------------------------
    # dynamic data
    # ------------------------------------------------------------------

    @property
    def dyn_map(self) -> Dict[str, str]:
        return dict(_CZ_HOURLY_MAP if self.timestep == 'H' else _CZ_DAILY_MAP)

    @property
    def dyn_factors(self) -> Dict[str, float]:
        return {mean_air_pressure(): 10.0}  # kPa -> hPa

    @property
    def _time_col(self) -> str:
        return 'time' if self.timestep == 'H' else 'date'

    @functools.cached_property
    def _raw_columns(self) -> Dict[str, Tuple[str, ...]]:
        """
        product -> raw columns read from it, from the header of the first
        gauge's file (the tests check that every gauge has these columns).
        From CAMELS-CZ.zip only the five ``*_CHMI`` columns are read: its
        ERA5-Land columns and streamflow are identical to those of the Caravan
        extension, which starts 28 years earlier, and 24042409 has five more
        columns that copy the CHMI ones.
        """
        stn = self.stations()[0]
        cols = {}
        for product in self._products:
            header = pd.read_csv(os.path.join(self._ts_dir(product), _cz_fname(stn)), nrows=0).columns
            header = [c for c in header if c != self._time_col]
            if product == 'CAMELS-CZ':
                header = [c for c in header if c.endswith('_CHMI')]
            cols[product] = tuple(header)
        return cols

    @property
    def dynamic_features(self) -> List[str]:
        rename = self.dyn_map
        return [rename.get(col, col) for cols in self._raw_columns.values()
                for col in cols] + [observed_streamflow_cms()]

    @property
    def start(self) -> pd.Timestamp:
        return self._extent[0]

    @property
    def end(self) -> pd.Timestamp:
        return self._extent[1]

    @functools.cached_property
    def _extent(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """first and last time step over all files of this ``timestep``: from
        the time axis of the netCDF cache, which is built from them, or else
        from the first and last rows of every file (they are sorted in time)"""
        if netCDF4 is not None and xr is not None and self.dyn_fpath_exists:
            with xr.open_dataset(self.dyn_fpath) as ds:
                return pd.Timestamp(ds['time'].values[0]), pd.Timestamp(ds['time'].values[-1])
        stamps = []
        for product in self._products:
            for stn in self.stations():
                rows = _first_and_last_row(os.path.join(self._ts_dir(product), _cz_fname(stn)))
                stamps += [pd.Timestamp(_row_date(row, b',')) for row in rows]
        return min(stamps), max(stamps)

    def _reader_spec(self, features: List[str], st=None, en=None) -> Dict:
        """
        Small, picklable description of a read for :func:`_read_camels_cz_stn`.
        Only the raw columns needed for ``features`` are parsed. The first
        product (the Caravan extension for ``D``) is always read, at least its
        time column, so that every request has the same time index.
        """
        rename = self.dyn_map
        wanted = set(features)
        if observed_streamflow_cms() in wanted:
            wanted.add(observed_streamflow_mm())
        files = []
        for i, (product, cols) in enumerate(self._raw_columns.items()):
            use = [c for c in cols if rename.get(c, c) in wanted]
            if use or i == 0:
                files.append((self._ts_dir(product), [self._time_col] + use))
        # float64 values must be parsed exactly; at float32 the faster default
        # parser gives the same values
        parser = 'round_trip' if np.dtype(self.fp) == np.float64 else None
        return dict(files=files, time_col=self._time_col, rename=rename,
                    factors=self.dyn_factors, parser=parser,
                    freq='h' if self.timestep == 'H' else 'D', end=self.end,
                    features=list(features), st=st, en=en, fp=self.fp)

    @functools.cached_property
    def _q_areas(self) -> Dict[str, float]:
        """
        station -> catchment area (km2) with which the authors converted the
        CHMI discharge to mm per time step, and which therefore turns it back
        into m3 s-1: ``area`` of the attributes for ``D`` and ``BasinArea`` of
        the boundary shapefile for ``H`` (0.3 % larger). Both were verified
        against the discharge on the CHMI open data portal.
        """
        if self.timestep == 'H':
            dbf = _read_dbf(f"{os.path.splitext(self.boundary_file)[0]}.dbf", ('gauge_id', 'BasinArea'))
            return {stn.removeprefix(_CZ_PREFIX): int(area) / 1e6
                    for stn, area in zip(dbf['gauge_id'], dbf['BasinArea'])}
        return self._static_table()[catchment_area()].astype(float).to_dict()

    def mm_to_cms(self, q_mm: pd.Series) -> pd.Series:
        """converts streamflow of the station ``q_mm.name`` from mm per time
        step to m3 s-1 with the area the authors used (see :attr:`_q_areas`)"""
        return q_mm * self._q_areas[q_mm.name] * 1e3 / self._seconds

    def cms_to_mm(self, q_cms: pd.Series) -> pd.Series:
        """the inverse of :meth:`mm_to_cms`"""
        return q_cms * self._seconds / (self._q_areas[q_cms.name] * 1e3)

    @property
    def _seconds(self) -> int:
        """length of a time step in seconds"""
        return 3600 if self.timestep == 'H' else 86400

    def _read_stn_dyn(self, station: str) -> pd.DataFrame:
        """
        All dynamic features of one station with ``NaN`` for missing values.
        Values are cast to ``float_precision`` (float32 by default).
        """
        return _read_camels_cz_stn(self._reader_spec(self.dynamic_features),
                                   station, self._q_areas[station])

    def _read_dynamic(
            self,
            stations,
            dynamic_features,
            st: Union[str, pd.Timestamp] = None,
            en: Union[str, pd.Timestamp] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Reads dynamic features of several stations, with a process pool if the
        csv files to parse are large enough (see :func:`n_workers`).
        """
        st, en = self._check_length(st, en)
        features = validate_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')
        if not features:
            raise ValueError("no dynamic feature was requested")
        stations = validate_attributes(stations, self.stations(), 'stations')
        spec = self._reader_spec(features, st, en)
        areas = [self._q_areas[stn] for stn in stations]

        start = time.time()
        nbytes = len(stations) * sum(os.path.getsize(os.path.join(folder, _cz_fname(stations[0])))
                                     for folder, _ in spec['files']) if stations else 0
        cpus = n_workers(nbytes, len(stations), self.processes)

        if cpus == 1:
            dyn = {stn: _read_camels_cz_stn(spec, stn, area) for stn, area in zip(stations, areas)}
        else:
            with cf.ProcessPoolExecutor(cpus) as executor:
                results = executor.map(functools.partial(_read_camels_cz_stn, spec), stations, areas)
                dyn = dict(zip(stations, results))

        if self.verbosity > 1:
            print(f"Read {len(dyn)} stations for {len(features)} dyn features "
                  f"in {time.time() - start:.2f} seconds with {cpus} cpus.")
        return dyn


# the archives (without .zip) read for each timestep. For D, the Caravan
# extension comes first: its time index (1950 onwards) is the one served.
_CZ_PRODUCTS = {
    'D': ('Caravan-Extension-CZ', 'CAMELS-CZ'),
    'H': ('CAMELS-CZ-Hourly',),
}

# the archive whose attributes and boundaries are read for each timestep; the
# attribute files of the Caravan extension are identical to those of CAMELS-CZ
_CZ_ATTR_PRODUCT = {'D': 'CAMELS-CZ', 'H': 'CAMELS-CZ-Hourly'}

# attributes_<kind>_camelscz.csv
_CZ_ATTR_KINDS = ('other', 'caravan', 'hydroatlas')

_CZ_PREFIX = 'camelscz_'


def _cz_fname(station: str) -> str:
    return f"{_CZ_PREFIX}{station}.csv"


_SML = (soil_moisture_layer1, soil_moisture_layer2, soil_moisture_layer3, soil_moisture_layer4)

# Caravan units: mm day-1 (streamflow, precipitation, evaporation), deg C,
# W m-2, mm (snow water equivalent), m s-1, m3 m-3 (soil water) and kPa
# (surface pressure; the mean is converted to hPa, min and max keep their raw
# names as _map has no names for them). PET and potential evaporation
# are different quantities: FAO Penman-Monteith reference evapotranspiration
# (pet_mm_*) and ERA5-Land's potential evaporation (pevap_mm_era5land).
_CZ_DAILY_MAP = {
    'streamflow': observed_streamflow_mm(),
    'total_precipitation_sum': total_precipitation_with_specifier('era5land'),
    'temperature_2m_mean': mean_air_temp_with_specifier('era5land'),
    'temperature_2m_min': min_air_temp_with_specifier('era5land'),
    'temperature_2m_max': max_air_temp_with_specifier('era5land'),
    'potential_evaporation_sum_ERA5_LAND': mean_potential_evaporation_with_specifier('era5land'),
    'potential_evaporation_sum_FAO_PENMAN_MONTEITH': total_potential_evapotranspiration_with_specifier('era5land'),
    'total_precipitation_sum_CHMI': total_precipitation_with_specifier('chmi'),
    'temperature_2m_mean_CHMI': mean_air_temp_with_specifier('chmi'),
    'temperature_2m_min_CHMI': min_air_temp_with_specifier('chmi'),
    'temperature_2m_max_CHMI': max_air_temp_with_specifier('chmi'),
    'potential_evaporation_sum_FAO_PENMAN_MONTEITH_CHMI': total_potential_evapotranspiration_with_specifier('chmi'),
    'snow_depth_water_equivalent_mean': snow_water_equivalent(),
    'snow_depth_water_equivalent_min': min_snow_water_equivalent(),
    'snow_depth_water_equivalent_max': max_snow_water_equivalent(),
    'surface_net_solar_radiation_mean': net_solar_radiation(),
    'surface_net_solar_radiation_min': min_net_solar_radiation(),
    'surface_net_solar_radiation_max': max_net_solar_radiation(),
    'surface_net_thermal_radiation_mean': net_longwave_radiation(),
    'surface_net_thermal_radiation_min': min_net_longwave_radiation(),
    'surface_net_thermal_radiation_max': max_net_longwave_radiation(),
    'dewpoint_temperature_2m_mean': mean_dewpoint_temperature(),
    'dewpoint_temperature_2m_min': min_dewpoint_temperature(),
    'dewpoint_temperature_2m_max': max_dewpoint_temperature(),
    'u_component_of_wind_10m_mean': u_component_of_wind_at_10m(),
    'u_component_of_wind_10m_min': u_component_of_wind_with_specifier('min'),
    'u_component_of_wind_10m_max': u_component_of_wind_with_specifier('max'),
    'v_component_of_wind_10m_mean': v_component_of_wind_at_10m(),
    'v_component_of_wind_10m_min': v_component_of_wind_with_specifier('min'),
    'v_component_of_wind_10m_max': v_component_of_wind_with_specifier('max'),
    'surface_pressure_mean': mean_air_pressure(),
    **{f'volumetric_soil_water_layer_{i}_mean': sml() for i, sml in enumerate(_SML, 1)},
    **{f'volumetric_soil_water_layer_{i}_{stat}': soil_moisture_layer_with_specifier(i, stat)
       for i in range(1, 5) for stat in ('min', 'max')},
}

# the same quantities per hour: mm h-1, deg C, W m-2, mm, m s-1, m3 m-3, kPa
_CZ_HOURLY_MAP = {
    'streamflow': observed_streamflow_mm(),
    'total_precipitation_sum_CHMI': total_precipitation_with_specifier('chmi'),
    'temperature_2m_mean_CHMI': mean_air_temp_with_specifier('chmi'),
    'total_precipitation_ERA5_LAND': total_precipitation_with_specifier('era5land'),
    'temperature_2m_ERA5_LAND': mean_air_temp_with_specifier('era5land'),
    'potential_evaporation_ERA5_LAND': mean_potential_evaporation_with_specifier('era5land'),
    'snow_depth_water_equivalent_ERA5_LAND': snow_water_equivalent(),
    'surface_net_solar_radiation_ERA5_LAND': net_solar_radiation(),
    'surface_net_thermal_radiation_ERA5_LAND': net_longwave_radiation(),
    'dewpoint_temperature_2m_ERA5_LAND': mean_dewpoint_temperature(),
    'u_component_of_wind_10m_ERA5_LAND': u_component_of_wind_at_10m(),
    'v_component_of_wind_10m_ERA5_LAND': v_component_of_wind_at_10m(),
    'surface_pressure_ERA5_LAND': mean_air_pressure(),
    **{f'volumetric_soil_water_layer_{i}_ERA5_LAND': sml() for i, sml in enumerate(_SML, 1)},
}


def _read_camels_cz_stn(spec: Dict, station: str, area: float) -> pd.DataFrame:
    """
    Reads the dynamic data of one CAMELS-CZ station as described by
    :meth:`CAMELS_CZ._reader_spec`; ``area`` (km2) converts streamflow from mm
    per time step to m3 s-1. A file that ends before the dataset does is
    padded with NaN rows, so that all stations share one time axis. A
    module-level function so that a process pool pickles only the small
    ``spec``, not the dataset instance.
    """
    frames = [pd.read_csv(os.path.join(folder, _cz_fname(station)), usecols=usecols,
                          index_col=spec['time_col'], parse_dates=True,
                          float_precision=spec['parser'])
              for folder, usecols in spec['files']]
    df = frames[0]
    for other in frames[1:]:
        df = df.join(other, how='outer')
    if df.index[-1] < spec['end']:
        tail = pd.date_range(df.index[-1], spec['end'], freq=spec['freq'])[1:]
        df = df.reindex(df.index.append(tail))
    df.rename(columns=spec['rename'], inplace=True)
    apply_dyn_factors(df, spec['factors'])
    if observed_streamflow_cms() in spec['features']:
        seconds = pd.Timedelta(1, spec['freq']).total_seconds()
        df[observed_streamflow_cms()] = df[observed_streamflow_mm()] * area * 1e3 / seconds
    df = df.loc[spec['st']:spec['en'], spec['features']].astype(spec['fp'])
    df.index.name = 'time'
    df.columns.name = 'dynamic_features'
    return df


def _read_dbf(fpath: str, columns: Tuple[str, ...]) -> Dict[str, List[str]]:
    """
    The named columns of a dBASE table (the ``.dbf`` of a shapefile) as
    stripped strings, without fiona. Deleted records are skipped.
    """
    with open(fpath, 'rb') as f:
        data = f.read()
    n_records, header_len, record_len = struct.unpack('<IHH', data[4:12])
    fields, pos, offset = {}, 32, 1  # a record starts with its deletion flag
    while data[pos] != 0x0D:
        name = data[pos:pos + 11].split(b'\0', 1)[0].decode('ascii')
        fields[name] = (offset, data[pos + 16])
        offset += data[pos + 16]
        pos += 32
    out = {col: [] for col in columns}
    for i in range(n_records):
        rec = data[header_len + i * record_len: header_len + (i + 1) * record_len]
        if rec[:1] == b'*':
            continue
        for col in columns:
            beg, length = fields[col]
            out[col].append(rec[beg:beg + length].decode('utf-8').strip())
    return out
