import os
import time
import shutil
import zipfile
import warnings
import functools
import concurrent.futures as cf
from typing import Union, List, Dict

import pandas as pd

from ..utils import _RainfallRunoff, apply_dyn_factors, n_workers
from ...utils import validate_attributes
from .._map import (
    observed_streamflow_cms,
    observed_streamflow_mm,
    observed_water_level_m,
    mean_air_temp,
    max_air_temp,
    min_air_temp,
    total_precipitation,
    total_potential_evapotranspiration,
    total_potential_evapotranspiration_with_specifier,
    actual_evapotranspiration_with_specifier,
    mean_rel_hum,
    mean_windspeed,
    max_wind_gust,
    solar_radiation,
    MJ_M2_DAY_TO_WM2,
)
from .._map import (
    catchment_area,
    gauge_latitude,
    gauge_longitude,
    gauge_elevation_meters,
    population_density,
)
from ._common import _remove_stale, _warn_duplicate_gauges


class CAMELS_KR(_RainfallRunoff):
    """
    Daily hydrometeorological time series and static catchment attributes for
    282 catchments in South Korea following
    `Lee et al., 2026 <https://doi.org/10.5194/essd-2026-544>`_ (CAMELS-KR v1.1).
    The data is downloaded from its
    `zenodo repository <https://zenodo.org/records/21930882>`_.

    The dataset has 14 dynamic features from 1981-01-01 to 2025-12-31 (16436
    daily steps) and 75 static features. Streamflow and water level come from
    WAMIS and HRFCO (gauge-dependent coverage, missing days are ``NaN``). The
    meteorological series have no gaps: they are KMA station observations,
    gap-filled and interpolated to a 0.1° grid by the authors; ``pet_mm_gleam``
    and ``aet_mm_gleam`` are from GLEAM4. Catchment boundaries are shapefiles in
    WGS84.

    Dynamic features (raw name, unit):

    - ``q_cms_obs`` (discharge_vol, m3 s-1)
    - ``q_mm_obs`` (discharge_spec, mm day-1)
    - ``wl_m_obs`` (water_level, m above the station's zero datum)
    - ``pcp_mm`` (prec, mm day-1)
    - ``airtemp_C_min``, ``airtemp_C_max``, ``airtemp_C_mean`` (temp_min, temp_max, temp_avg, °C)
    - ``rh_%`` (rel_hum, %)
    - ``windspeed_mps`` (wind_speed at 10 m, m s-1)
    - ``windgust_mps_max`` (wind_speed_max, daily maximum gust, m s-1)
    - ``swdownrad_wm2`` (solar_radiation, MJ m-2 day-1 converted to W m-2)
    - ``pet_mm`` (pet, FAO Penman-Monteith, mm day-1)
    - ``pet_mm_gleam``, ``aet_mm_gleam`` (pet_gleam, aet_gleam, mm day-1)

    Static features keep their raw names except ``area_km2`` (basin_area),
    ``lat``/``long`` (gauge_lat/gauge_lon), ``elev_gauge_m`` (gauge_elev) and
    ``pop_density_<year>_km2`` (dens_<year>). ``lat``/``long`` are rounded to
    0.01° in the source (up to 0.55 km from the gauge); unrounded coordinates
    are in ``Catchment_boundaries/Stations/CAMELS_KR_stations.shp``.

    Not provided: the LSTM and HBV simulated streamflow and the HBV parameters
    are model output, so they are not extracted from the archive. Two served
    feature groups are modelled rather than measured: ``pet_mm_gleam`` and
    ``aet_mm_gleam`` come from GLEAM4 (which learns evaporative stress from
    flux-tower data) and the 18 soil attributes from SoilGrids 2.0 (machine
    learning on soil profiles), as in CAMELS_BR, CAMELS_IND and GSHA.

    .. note::
        Examples of source data issues, served unchanged:

        - ``elev_mean``, ``elev_min``, ``elev_5``, ``elev_95`` and ``elev_max``
          are mislabeled or misaligned (``elev_5`` > ``elev_max`` for every
          gauge), hence not renamed. ``gauge_elev`` (6-599 m) is plausible and
          is the only elevation column renamed.
        - ``flow_record`` counts days with water level, not discharge, and
          ``q_mean`` is lower than the mean of ``q_mm_obs`` at every gauge.
        - ``bulk_density_*`` is in g cm-3 (values 0.7-1.5), not kg m-3 as documented.
        - ``q_cms_obs`` is negative on 2350 days at 39 gauges and holds -999 and
          9999 at 3014610, -99.99 at 2005660 and 2013650, and 999.9 at 1019630.
          ``q_mm_obs`` exceeds 500 mm day-1 on 141 days at 32 gauges.
        - ``wl_m_obs`` holds -9999 or -999.9 at 2016650, 4009665 and 4005660;
          values ~100 times too large (probably cm) at 2018645 (1988-06-20 to
          1993-12-31), 1019630 (1997-1998) and 1018683 (1998); one-day spikes
          of 500-4168 m at 2016650, 4006680, 3101645 and 5005680; and 99.0-99.98 m
          (probably negative stages) at 2013615, 5003650, 2301630 and 2021675.
        - ``swdownrad_wm2`` exceeds the top-of-atmosphere irradiance on 0.85% of
          the days.

    :py:class:`aqua_fetch.rr.CAMELS_SK` covers 178 Korean gauges at hourly
    timestep for 2000-2019 (17 dynamic, 215 static features). Both use the
    official Korean gauge codes, so the 115 gauges they share are found by
    :meth:`common_stations`::

        >>> from aqua_fetch import CAMELS_KR, CAMELS_SK
        >>> shared = CAMELS_KR().common_stations(CAMELS_SK().stations())
        >>> len(shared)
        115

    :py:class:`aqua_fetch.rr.GSHA` has 4 Korean GRDC gauges under different ids;
    they are CAMELS_KR gauges 1007635, 2011650, 3012620 and 5004650, which
    ``common_stations(gsha, max_dist_km=2)`` finds.

    The first initialization (download of ~360 MB, extraction and a 260 MB
    netCDF cache) took ~3 minutes. Afterwards, fetching all 14 features of all
    282 stations takes ~0.3 s from the cache (~0.07 s for one station). Without
    a cache it takes ~1 s from the csv files with a process pool, ~8 s with
    ``processes=1`` and ~0.03 s for one station. With the ``spawn`` or
    ``forkserver`` start method (Windows, macOS, Linux from Python 3.14), run
    scripts under ``if __name__ == "__main__":`` because building the cache
    uses a process pool.

    Examples
    --------
    >>> from aqua_fetch import CAMELS_KR
    >>> dataset = CAMELS_KR()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='1001620', as_dataframe=True)
    >>> dynamic['1001620'].shape
    (16436, 14)
    >>> stns = dataset.stations()
    >>> len(stns)
    282
    ... # get data of 10 % of (randomly selected) stations
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)
    28
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('1001620', as_dataframe=True,
    ...  dynamic_features=['pcp_mm', 'airtemp_C_mean', 'pet_mm', 'q_cms_obs'])
    >>> dynamic['1001620'].shape
    (16436, 4)
    ... # get data between two dates
    >>> _, dynamic = dataset.fetch('1001620', st='2020-01-01', en='2020-12-31', as_dataframe=True)
    >>> dynamic['1001620'].shape
    (366, 14)
    ... # get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='1001620', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['1001620'].shape
    ((1, 75), 1, (16436, 14))
    >>> dataset.fetch_static_features('1001620', ['area_km2', 'aridity', 'forest_perc'])
              area_km2   aridity  forest_perc
    gauge_id
    1001620      160.9  0.701389    90.985306
    ... # without as_dataframe=True (and with xarray installed) an xarray Dataset is returned
    >>> _, dynamic = dataset.fetch(10)
    >>> dynamic.sizes
    Frozen({'time': 16436, 'dynamic_features': 14})
    >>> dataset.stn_coords().shape
    (282, 2)
    >>> dataset.area('1001620')  # km2
    gauge_id
    1001620    160.899994
    Name: area_km2, dtype: float32
    >>> dataset.q_mm('1001620').shape  # observed streamflow in mm day-1
    (16436, 1)
    ... # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('1001620')['type']
    'Polygon'
    """
    url = "https://zenodo.org/records/21930882"

    # the only file downloaded from the record (the pdf description is skipped)
    _archive_name = "CAMELS-KR.zip"

    # folders of model output inside the archive, not extracted (observations only)
    _model_output_dirs = ("Simulated hydrological time series", "HBV_model_parameters")

    # attribute files, named CAMELS_KR_<name>_attributes.csv
    _attr_files = ("location", "topography", "climate", "hydrology",
                   "land cover", "soil", "human influence")

    # the two time-series files of each station, CAMELS_KR_<kind>_timeseries_<id>.csv
    _ts_kinds = ("Hydrological", "Meteorological")

    def __init__(
            self,
            path: str = None,
            overwrite: bool = False,
            to_netcdf: bool = True,
            verbosity: int = 1,
            **kwargs
    ):
        """
        Parameters
        ----------
        path : str
            folder in which the ``CAMELS_KR`` folder is (or will be) created.
            If None, the default aqua_fetch data folder is used.
        overwrite : bool
            If True, the archive, the extracted data and the netCDF cache are
            deleted, downloaded and built again.
        to_netcdf : bool
            whether to store all dynamic data in a netCDF file (:attr:`dyn_fpath`)
            for faster fetching. Requires netCDF4 and xarray. Once the file
            exists, the data is read from it even with ``to_netcdf=False``.
        verbosity : int
            0: no message will be printed
        kwargs :
            passed to the parent class e.g. ``processes`` (``processes=1``
            disables multiprocessing) or ``remove_zip`` (delete the archive
            after extraction).
        """
        super().__init__(path=path, overwrite=overwrite, to_netcdf=to_netcdf,
                         verbosity=verbosity, **kwargs)

        # lazy caches
        self._location = None
        self._static_df = None
        self._raw_cols = None
        self._extent = None

        self._download_camels_kr(overwrite=overwrite)

        self._check_manifest()

        _warn_duplicate_gauges(self.name, self._location_attrs().reset_index())

        self._maybe_to_netcdf()

        self.bbox = {'llcrnrlat': 34.0, 'urcrnrlat': 39.5,
                     'llcrnrlon': 126.0, 'urcrnrlon': 130.0}
        self.parallels = range(34, 40, 2)
        self.meridians = range(126, 131, 2)

    def _download_camels_kr(self, overwrite: bool = False):
        """
        Downloads ``CAMELS-KR.zip`` and extracts it. Nothing happens if the
        extracted folder exists, even if the archive was deleted
        (``remove_zip=True``).
        """
        if os.path.isdir(self._root) and not overwrite:
            if self.verbosity:
                print(f"CAMELS_KR data already exists at {self._root}")
            self.maybe_remove_zip_files()
            return

        archive = os.path.join(self.path, self._archive_name)
        if overwrite:
            _remove_stale((archive, self._root, self.dyn_fpath), self.verbosity)

        if not os.path.exists(archive):
            # imported here because this module installs a SIGINT handler on import
            from ...download_zenodo import download_from_zenodo
            os.makedirs(self.path, exist_ok=True)
            download_from_zenodo(self.path, doi=self.url, include=[self._archive_name],
                                 verbosity=self.verbosity)

        self._extract(archive)

        self.maybe_remove_zip_files()
        return

    def _extract(self, archive: str):
        """
        Extracts ``archive``, except the model output, into a temporary folder
        which is renamed to :attr:`_root` once complete. An interrupted
        extraction is therefore redone on the next initialization instead of
        being taken as complete.
        """
        tmp = f"{self._root}_extracting"
        if os.path.exists(tmp):  # left over from an interrupted extraction
            shutil.rmtree(tmp)

        if self.verbosity:
            print(f"extracting {archive}")

        # the archive holds a single top-level folder with the same name as _root
        top = os.path.basename(self._root)
        skip = tuple(f"{top}/{folder}/" for folder in self._model_output_dirs)
        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.namelist() if not m.startswith(skip)]
            zf.extractall(tmp, members=members)

        os.replace(os.path.join(tmp, top), self._root)
        os.rmdir(tmp)

        warnings.warn(
            f"CAMELS_KR: {' and '.join(self._model_output_dirs)} are model output "
            f"and were not extracted from {archive}.", UserWarning)
        return

    def _check_manifest(self):
        """
        Warns if any attribute, boundary or time-series file is missing. The
        expected station files come from the location attributes, not from
        whatever is on disk.
        """
        location = self._attr_path("location")
        if not os.path.exists(location):
            raise FileNotFoundError(
                f"{location} not found. Re-initialize CAMELS_KR with overwrite=True.")

        # fiona needs the .dbf/.shx/.prj siblings of the boundary .shp as well
        boundary = os.path.splitext(self.boundary_file)[0]
        missing = [f for f in [self._attr_path(name) for name in self._attr_files]
                   + [f"{boundary}{ext}" for ext in ('.shp', '.dbf', '.shx', '.prj')]
                   if not os.path.exists(f)]

        for kind in self._ts_kinds:
            folder = self._ts_dir(kind)
            present = set(os.listdir(folder)) if os.path.isdir(folder) else set()
            missing += [self._ts_path(kind, stn) for stn in self.stations()
                        if _kr_ts_fname(kind, stn) not in present]

        if missing:
            warnings.warn(
                f"CAMELS_KR: {len(missing)} files are missing, e.g. {missing[:3]}. "
                f"The data is incomplete; re-initialize with overwrite=True.",
                UserWarning)
        return

    @property
    def _root(self) -> os.PathLike:
        """folder into which the archive is extracted"""
        return os.path.join(self.path, "CAMELS-KR")

    def _ts_dir(self, kind: str) -> os.PathLike:
        """folder of the ``Hydrological`` or ``Meteorological`` time-series files"""
        return os.path.join(self._root, f"{kind} time series")

    def _ts_path(self, kind: str, station: str) -> os.PathLike:
        return os.path.join(self._ts_dir(kind), _kr_ts_fname(kind, station))

    def _attr_path(self, name: str) -> os.PathLike:
        return os.path.join(self._root, f"CAMELS_KR_{name}_attributes.csv")

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(self._root, "Catchment_boundaries", "Catchments",
                            "CAMELS_KR_catchments.shp")

    @property
    def boundary_id_map(self) -> str:
        return "ID"

    @property
    def dyn_map(self) -> Dict[str, str]:
        # units (Table 1 of Lee et al., 2026) are verified against the data:
        # discharge_spec == discharge_vol * 86.4 / basin_area, and the daily
        # clearness index of solar_radiation (MJ m-2 day-1) has a median of 0.55
        return {
            'discharge_vol': observed_streamflow_cms(),     # m3 s-1
            'discharge_spec': observed_streamflow_mm(),     # mm day-1
            'water_level': observed_water_level_m(),        # m
            'prec': total_precipitation(),                  # mm day-1
            'temp_min': min_air_temp(),                     # deg C
            'temp_max': max_air_temp(),                     # deg C
            'temp_avg': mean_air_temp(),                    # deg C
            'rel_hum': mean_rel_hum(),                      # %
            'wind_speed': mean_windspeed(),                 # m s-1
            'wind_speed_max': max_wind_gust(),              # m s-1, daily maximum gust
            'solar_radiation': solar_radiation(),           # MJ m-2 day-1 -> W m-2
            'pet': total_potential_evapotranspiration(),    # mm day-1
            'pet_gleam': total_potential_evapotranspiration_with_specifier('gleam'),  # mm day-1
            'aet_gleam': actual_evapotranspiration_with_specifier('gleam'),  # mm day-1
        }

    @property
    def dyn_factors(self) -> Dict[str, float]:
        return {solar_radiation(): MJ_M2_DAY_TO_WM2}

    @property
    def static_map(self) -> Dict[str, str]:
        # the elev_* statistics are not renamed, see the class docstring
        return {
            'basin_area': catchment_area(),          # km2
            'gauge_lat': gauge_latitude(),           # deg N (WGS84)
            'gauge_lon': gauge_longitude(),          # deg E (WGS84)
            'gauge_elev': gauge_elevation_meters(),  # m a.s.l.
            'dens_2000': population_density(2000),   # persons km-2
            'dens_2010': population_density(2010),
            'dens_2020': population_density(2020),
            'dens_2024': population_density(2024),
        }

    @property
    def _mm_feature_name(self) -> str:
        return observed_streamflow_mm()

    @property
    def _area_name(self) -> str:
        return catchment_area()

    @property
    def _coords_name(self) -> List[str]:
        return [gauge_latitude(), gauge_longitude()]

    def _read_attr(self, name: str) -> pd.DataFrame:
        # most attribute files start with a byte order mark
        return pd.read_csv(self._attr_path(name), index_col='gauge_id',
                           dtype={'gauge_id': str}, encoding='utf-8-sig')

    def _location_attrs(self) -> pd.DataFrame:
        if self._location is None:
            self._location = self._read_attr("location")
        return self._location

    def stations(self) -> List[str]:
        """ids of the 282 gauges, from the location attributes file"""
        return self._location_attrs().index.tolist()

    def _static_table(self) -> pd.DataFrame:
        """the cached table of all static features"""
        if self._static_df is None:
            # the location file is already cached for the manifest/duplicate checks
            df = pd.concat([self._location_attrs() if name == "location" else self._read_attr(name)
                            for name in self._attr_files], axis=1)
            df.rename(columns=self.static_map, inplace=True)
            self._static_df = df
        return self._static_df

    def _static_data(self) -> pd.DataFrame:
        # a copy, so that a caller's in-place edit cannot corrupt the cache
        return self._static_table().copy()

    @property
    def static_features(self) -> List[str]:
        return self._static_table().columns.tolist()

    def _raw_columns(self) -> Dict[str, List[str]]:
        """raw column names of the two time-series files. They and the time
        extent are read once from the first gauge's files, which have the same
        columns and dates as every other gauge's (checked by the tests)."""
        if self._raw_cols is None:
            stn = self.stations()[0]
            frames = {kind: pd.read_csv(self._ts_path(kind, stn), index_col='date', parse_dates=True)
                      for kind in self._ts_kinds}
            self._extent = (min(df.index.min() for df in frames.values()),
                            max(df.index.max() for df in frames.values()))
            self._raw_cols = {kind: df.columns.tolist() for kind, df in frames.items()}
        return self._raw_cols

    @property
    def dynamic_features(self) -> List[str]:
        return [self.dyn_map.get(col, col)
                for cols in self._raw_columns().values() for col in cols]

    @property
    def start(self) -> pd.Timestamp:
        self._raw_columns()
        return self._extent[0]

    @property
    def end(self) -> pd.Timestamp:
        self._raw_columns()
        return self._extent[1]

    def _reader_spec(self, features: List[str], st=None, en=None) -> Dict:
        """
        Small, picklable description of a read for :func:`_read_camels_kr_stn`.
        Only the raw columns needed for ``features`` are parsed, and a file with
        none of them is not read at all.
        """
        rename = self.dyn_map
        wanted = set(features)
        files = {}
        for kind, cols in self._raw_columns().items():
            cols = [col for col in cols if rename.get(col, col) in wanted]
            if cols:
                files[kind] = (self._ts_dir(kind), ['date'] + cols)
        return dict(files=files, rename=rename, factors=self.dyn_factors,
                    features=list(features), st=st, en=en, fp=self.fp)

    def _read_stn_dyn(self, station: str) -> pd.DataFrame:
        """
        All dynamic features of one station with ``NaN`` for missing values.
        Values are cast to ``float_precision`` (float32 by default), which is
        safe here: the largest magnitude is ~1e5 (``q_cms_obs``) and the
        relative rounding error is below 1e-7. An existing netCDF cache keeps
        the precision it was built with.
        """
        return _read_camels_kr_stn(self._reader_spec(self.dynamic_features), station)

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

        start = time.time()
        nbytes = len(stations) * sum(os.path.getsize(self._ts_path(kind, stations[0]))
                                     for kind in spec['files']) if stations else 0
        cpus = n_workers(nbytes, len(stations), self.processes)

        if cpus == 1:
            dyn = {stn: _read_camels_kr_stn(spec, stn) for stn in stations}
        else:
            with cf.ProcessPoolExecutor(cpus) as executor:
                results = executor.map(functools.partial(_read_camels_kr_stn, spec), stations)
                dyn = dict(zip(stations, results))

        if self.verbosity > 1:
            print(f"Read {len(dyn)} stations for {len(features)} dyn features "
                  f"in {time.time() - start:.2f} seconds with {cpus} cpus.")
        return dyn


def _kr_ts_fname(kind: str, station: str) -> str:
    """name of a CAMELS-KR time-series file; ``kind`` is ``Hydrological`` or
    ``Meteorological``"""
    return f"CAMELS_KR_{kind}_timeseries_{station}.csv"


def _read_camels_kr_stn(spec: Dict, station: str) -> pd.DataFrame:
    """
    Reads the dynamic data of one CAMELS-KR station as described by
    :meth:`CAMELS_KR._reader_spec`. It is a module-level function so that a
    process pool pickles only the small ``spec``, not the dataset instance.
    """
    frames = [
        pd.read_csv(os.path.join(folder, _kr_ts_fname(kind, station)),
                    usecols=usecols, index_col='date', parse_dates=True)
        for kind, (folder, usecols) in spec['files'].items()
    ]
    df = pd.concat(frames, axis=1) if len(frames) > 1 else frames[0]
    df.rename(columns=spec['rename'], inplace=True)
    apply_dyn_factors(df, spec['factors'])
    df = df.loc[spec['st']:spec['en'], spec['features']].astype(spec['fp'])
    df.index.name = 'time'
    df.columns.name = 'dynamic_features'
    return df
