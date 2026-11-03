import os
import re
import zlib
import time
import shutil
import sqlite3
import struct
import zipfile
import pathlib
import warnings
import functools
import concurrent.futures as cf
from typing import Union, List, Dict, Tuple

import numpy as np
import pandas as pd

from .utils import _RainfallRunoff, n_workers
from ..utils import validate_attributes
from .._backend import netCDF4
from ._map import (
    observed_streamflow_cms,
    observed_streamflow_mm,
    observed_water_level_cm,
    total_precipitation_with_specifier,
    max_air_temp_with_specifier,
    min_air_temp_with_specifier,
)
from ._map import (
    catchment_area,
    gauge_latitude,
    gauge_longitude,
    gauge_elevation_meters,
    catchment_elevation_meters,
    slope,
)
from ._camels._common import _remove_stale, _warn_duplicate_gauges


class Russia(_RainfallRunoff):
    """
    Daily discharge and water level of 1,886 Roshydromet gauges in Russia with
    catchment-averaged meteorology, from 2008-01-01 to 2020-12-31 (4749 days),
    following Abramov, Kurochkina and Moreido (2026),
    `Russ. Meteorol. Hydrol. 51(3) <https://doi.org/10.3103/S1068373926030052>`_.
    The data (release 1.3) is downloaded from its
    `zenodo repository <https://zenodo.org/records/8432070>`_ (3.5 GB). Reading
    it requires the netCDF4 package.

    Discharge and water level are observations from AIS GMVO, Russia's state
    water-monitoring system; missing days are ``NaN``. 1,114 gauges have
    discharge on every day; 145 have it on fewer than half of the January to
    March days but on over 90 % of the June to September days, while their
    water level continues. Catchment boundaries are a GeoPackage in WGS84.

    Dynamic features (raw name, unit):

    - ``q_cms_obs`` (q_cms_s, m3 s-1)
    - ``q_mm_obs`` (q_mm_day, mm day-1, = q_cms_s * 86.4 / ws_area)
    - ``wl_cm_obs`` (lvl_sm, cm above the gauge datum)
    - ``lvl_mbs`` (lvl_mbs, m above sea level in the Baltic Height System)
    - ``airtemp_C_era5_max``, ``airtemp_C_era5_min``, ``airtemp_C_era5land_max``,
      ``airtemp_C_era5land_min`` (t_max_e5, t_min_e5, t_max_e5l, t_min_e5l, °C,
      catchment averages of ERA5 and ERA5-Land)
    - with ``suspect_pcp=True`` also ``pcp_mm_era5land``, ``pcp_mm_mswep``,
      ``pcp_mm_imerg``, ``pcp_mm_era5`` and ``pcp_mm_gpcp`` (prcp_e5l,
      prcp_mswep, prcp_imerg, prcp_e5, prcp_gpcp, mm day-1)

    The 25 static features are the gauge names (``name_ru``, ``name_en``) and
    coordinates, ``area_km2`` (ws_area, delineated by the authors),
    ``elev_gauge_m`` (height_bs, gauge datum in m of the Baltic Height System),
    ``ais_dif`` (% difference between the delineated and the official
    catchment area) and 18 HydroATLAS attributes under their HydroATLAS names
    and units, except ``elev_catch_m`` (ele_mt_sav), ``slope_degrees``
    (slp_dg_sav, in degrees, not degrees x 10 as in HydroATLAS) and
    ``lka_pc_use`` (in %, not % x 10 as in HydroATLAS).

    Not served:

    - by default, the precipitation. All five products are scaled down by an
      amount that grows as the catchment gets smaller and the product's grid
      coarser: below 30 km2 MSWEP, ERA5-Land and IMERG reach 0.1-0.3 of the
      value of nearby large catchments, at 300-1000 km2 about 0.7, and GPCP
      far less. Observed runoff exceeds MSWEP precipitation at 516 gauges.
    - the 10 GLEAM series (evaporation, its components, soil moisture and the
      stress factor): model output, scaled down in the same way (the stress
      factor peaks at 0.46 instead of ~1).
    - by default, the 278 catchments of GRDC gauges in the CIS (7-digit ids),
      of which only 6 have discharge and none has water level. With
      ``grdc_stns=True`` 272 are served: 6 are Roshydromet gauges of this
      dataset under another id (within 2 km, area within 10 %), without any
      observation, and are left out.
    - the folders ``nc_all_q``, ``nc_all_h`` and ``nc_all_q_h``: subsets of
      ``nc_concat`` with nothing it lacks (one gauge, 6390300, has -999 where
      ``nc_concat`` is empty); they are not extracted.

    .. note::
        Source data issues, served unchanged:

        - ``q_cms_obs`` is about 1000 times too large (probably L s-1) at the
          small catchments 1105, 2155, 2192, 2202 and 72191, and 34083 m3 s-1
          on 2008-11-22 at 70363 (median 12.7).
        - ``q_cms_obs`` is negative on 672 days at 49001, 70085 and 10524
          (likely reverse flow) and zero on 2015 days at 91 gauges.
        - water level is in whole cm, except tenths of cm at 10 gauges.
          ``elev_gauge_m`` differs from the datum of ``lvl_mbs`` at 69 gauges
          and is missing at 9 gauges that have ``lvl_mbs``.
        - empty series: ERA5 at 29 and ERA5-Land at 5 small catchments
          (< 171 km2); precipitation also: MSWEP at 3 (and on 2020-12-31
          everywhere), IMERG at 438 (436 reach north of 60°N, its limit).
        - ``cly_pc_sav``, ``slt_pc_sav``, ``snd_pc_sav`` and ``sgr_dk_sav`` are
          negative (down to -598) at 57 gauges.
        - 84243 and 84452 share coordinates and catchment polygon, so their
          meteorology is identical, while their discharge differs.
        - at the GRDC catchments ``q_mm_obs`` is computed with the GRDC
          catchment area, which differs from ``area_km2`` by 3 % (median).

    :py:class:`aqua_fetch.rr.Arcticnet` uses the same Roshydromet gauge codes
    and shares 78 of its 106 gauges (found by :meth:`common_stations`), but
    ends in 2003, before this dataset begins.

    The first initialization downloads 3.5 GB (16 min here) and extracts
    2.2 GB (8 s on a local disk). Afterwards initialization takes ~0.01 s.
    Fetching all dynamic features of all stations takes 1-2 s with a process
    pool (~2 s as xarray Dataset, ~9 s with ``processes=1``) and of one
    station ~0.02 s. No netCDF cache is built: reading one took 7 s for all
    stations and 0.5 s for one. With the ``spawn`` or
    ``forkserver`` start method (Windows, macOS, Linux from Python 3.14), run
    scripts under ``if __name__ == "__main__":`` because reading many stations
    uses a process pool.

    Examples
    --------
    >>> from aqua_fetch import Russia
    >>> dataset = Russia()
    >>> len(dataset.stations())
    1886
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='75299', as_dataframe=True)
    >>> dynamic['75299'].shape
    (4749, 8)
    ... # get data of 10 % of (randomly selected) stations
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)
    188
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('75299', as_dataframe=True,
    ...  dynamic_features=['airtemp_C_era5land_max', 'q_cms_obs'])
    >>> dynamic['75299'].shape
    (4749, 2)
    ... # get data between two dates
    >>> _, dynamic = dataset.fetch('75299', st='2012-01-01', en='2012-12-31', as_dataframe=True)
    >>> dynamic['75299'].shape
    (366, 8)
    ... # get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='75299', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['75299'].shape
    ((1, 25), 1, (4749, 8))
    >>> dataset.fetch_static_features('75299', ['name_en', 'area_km2', 'elev_gauge_m', 'for_pc_sse'])
                                 name_en  area_km2  elev_gauge_m  for_pc_sse
    gauge_id
    75299     r.Lojmina - d.Georgievskoe   151.652         80.34   43.426988
    ... # without as_dataframe=True (and with xarray installed) an xarray Dataset is returned
    >>> _, dynamic = dataset.fetch(10)
    >>> dynamic.sizes
    Frozen({'time': 4749, 'dynamic_features': 8})
    >>> dataset.stn_coords().shape
    (1886, 2)
    >>> dataset.area('75299')  # km2
    gauge_id
    75299    151.651993
    Name: area_km2, dtype: float32
    >>> dataset.q_mm('75299').shape  # observed streamflow in mm day-1
    (4749, 1)
    ... # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('75299')['type']
    'Polygon'
    ... # include the GRDC catchments and the precipitation
    >>> dataset = Russia(grdc_stns=True, suspect_pcp=True)
    >>> len(dataset.stations()), len(dataset.dynamic_features)
    (2158, 13)
    """
    url = "https://zenodo.org/records/8432070"

    # the only file of the record
    _archive_name = "Russia_HydroMeteo_Database_v04.zip"

    # folders extracted from the archive. nc_all_q, nc_all_h and nc_all_q_h
    # repeat subsets of nc_concat
    _extract_dirs = ("nc_concat", "attributes", "geometry")

    # raw names of the dynamic features, in the order they are served
    _obs_vars = ('q_cms_s', 'q_mm_day', 'lvl_sm', 'lvl_mbs')
    # served only with suspect_pcp=True
    _pcp_vars = ('prcp_e5l', 'prcp_mswep', 'prcp_imerg', 'prcp_e5', 'prcp_gpcp')
    _temp_vars = ('t_max_e5', 't_min_e5', 't_max_e5l', 't_min_e5l')
    # in every station file but never served (GLEAM model output)
    _gleam_vars = ('E', 'Ep', 'Eb', 'Es', 'Et', 'Ei', 'Ew', 'SMsurf', 'SMroot', 'S')

    def __init__(
            self,
            path: str = None,
            grdc_stns: bool = False,
            suspect_pcp: bool = False,
            overwrite: bool = False,
            to_netcdf: bool = True,
            verbosity: int = 1,
            **kwargs
    ):
        """
        Parameters
        ----------
        path : str
            folder in which the ``Russia`` folder is (or will be) created. If
            None, the default aqua_fetch data folder is used.
        grdc_stns : bool
            whether to include the catchments of GRDC gauges in the CIS: 272
            of 278, as 6 are Roshydromet gauges of this dataset under another
            id. Only 6 have discharge and none has water level.
        suspect_pcp : bool
            whether to include the five precipitation products. All are
            scaled down by an amount that grows as the catchment gets smaller
            (see above).
        overwrite : bool
            If True, the archive and the extracted data are deleted, downloaded
            and extracted again.
        to_netcdf : bool
            has no effect: the station files are netCDF already and reading
            them is faster than reading a combined netCDF file.
        verbosity : int
            0: no message will be printed
        kwargs :
            passed to the parent class e.g. ``processes`` (``processes=1``
            disables multiprocessing) or ``remove_zip`` (delete the archive
            after extraction).
        """
        if netCDF4 is None:
            raise ModuleNotFoundError(
                "Russia needs the netCDF4 package to read its station files: "
                "pip install netCDF4")

        # no netCDF cache: reading the station files is faster
        super().__init__(path=path, overwrite=overwrite, to_netcdf=False,
                         verbosity=verbosity, **kwargs)

        self._grdc_stns = grdc_stns
        self._suspect_pcp = suspect_pcp

        # lazy caches
        self._gauges_df = None
        self._stations = None
        self._grdc_dups = None
        self._static_df = None
        self._time = None

        self._download_russia(overwrite=overwrite)

        self._check_manifest()

        self._warn_not_served()

        meta = self._gauges().loc[self.stations()]
        _warn_duplicate_gauges(self.name, pd.DataFrame({
            'gauge_id': meta.index, 'gauge_name': meta['name_en'].to_numpy(),
            'gauge_lat': meta['lat'].to_numpy(), 'gauge_lon': meta['long'].to_numpy()}))
        if grdc_stns:
            pairs = [f"{grdc}={rosh}" for grdc, rosh in self._grdc_duplicates().items()]
            warnings.warn(f"Russia: {len(pairs)} GRDC gauges are left out because they are "
                          f"Roshydromet gauges of this dataset under another id: {pairs}.",
                          UserWarning)

        self.bbox = {'llcrnrlat': 37.0, 'urcrnrlat': 73.0,
                     'llcrnrlon': 19.0, 'urcrnrlon': 180.0}
        self.parallels = range(40, 75, 10)
        self.meridians = range(20, 181, 40)

    def _download_russia(self, overwrite: bool = False):
        """
        Downloads the archive and extracts it. Nothing happens if the extracted
        folder exists, even if the archive was deleted (``remove_zip=True``).
        """
        if os.path.isdir(self._root) and not overwrite:
            if self.verbosity > 1:
                print(f"Russia data already exists at {self._root}")
            self.maybe_remove_zip_files()
            return

        archive = os.path.join(self.path, self._archive_name)
        if overwrite:
            _remove_stale((archive, self._root), self.verbosity)

        if not os.path.exists(archive):
            # imported here because this module installs a SIGINT handler on import
            from ..download_zenodo import download_from_zenodo
            os.makedirs(self.path, exist_ok=True)
            download_from_zenodo(self.path, doi=self.url, include=[self._archive_name],
                                 verbosity=self.verbosity)

        self._extract(archive)

        self.maybe_remove_zip_files()
        return

    def _extract(self, archive: str):
        """
        Extracts the folders in :attr:`_extract_dirs` into a temporary folder
        which is renamed to :attr:`_root` once complete. An interrupted
        extraction is therefore redone on the next initialization instead of
        being taken as complete. A corrupt archive is deleted, so that the next
        initialization downloads it again.
        """
        tmp = f"{self._root}_extracting"
        if os.path.exists(tmp):  # left over from an interrupted extraction
            shutil.rmtree(tmp)

        if self.verbosity:
            print(f"extracting {archive}")

        top = os.path.basename(self._root)
        keep = tuple(f"{top}/{folder}/" for folder in self._extract_dirs)
        try:
            with zipfile.ZipFile(archive) as zf:
                members = [m for m in zf.namelist()
                           if m.startswith(keep) and not m.endswith('.DS_Store')]
                if not any(m.startswith(f"{top}/nc_concat/") for m in members):
                    raise ValueError(f"{archive} has no {top}/nc_concat folder; the "
                                     f"layout of the data has changed.")
                zf.extractall(tmp, members=members)
        except (zipfile.BadZipFile, zlib.error, EOFError):
            shutil.rmtree(tmp, ignore_errors=True)
            os.remove(archive)
            raise ValueError(f"{archive} is corrupt and was deleted. Initialize Russia "
                             f"again to download it again.") from None

        os.replace(os.path.join(tmp, top), self._root)
        os.rmdir(tmp)
        return

    def _check_manifest(self):
        """
        Warns if the attribute table, the boundary file or the file of any
        gauge is missing. The expected gauges come from the gauge GeoPackage,
        not from whatever is on disk.
        """
        if not os.path.exists(self._gauges_file):
            raise FileNotFoundError(
                f"{self._gauges_file} not found. Re-initialize Russia with overwrite=True.")

        missing = [f for f in (self._static_file, self.boundary_file) if not os.path.exists(f)]

        present = set(os.listdir(self._nc_dir)) if os.path.isdir(self._nc_dir) else set()
        missing += [self._stn_path(stn) for stn in self._gauges().index
                    if f"{stn}.nc" not in present]

        if missing:
            warnings.warn(
                f"Russia: {len(missing)} files are missing, e.g. {missing[:3]}. "
                f"The data is incomplete; re-initialize with overwrite=True.",
                UserWarning)
        return

    def _warn_not_served(self):
        """Warns (regardless of verbosity) about the data which is not served."""
        n_grdc = sum(_is_grdc(stn) for stn in self._gauges().index)
        scaled = "scaled down more the smaller the catchment"
        dropped = [f"the {len(self._gleam_vars)} GLEAM series (model output, {scaled})"]
        if self.suspect_pcp:
            notes = [f"All precipitation is {scaled}: below 30 km2 to 0.1-0.3 of nearby "
                     f"large catchments."]
        else:
            notes = []
            dropped.append(f"precipitation ({scaled}; suspect_pcp=True includes it)")
        if not self.grdc_stns:
            dropped.append(f"the {n_grdc} GRDC catchments (few have discharge; "
                           f"grdc_stns=True includes them)")
        warnings.warn(f"Russia: not served: {'; '.join(dropped)}. {' '.join(notes)}".strip(),
                      UserWarning)
        return

    def _grdc_duplicates(self, max_km: float = 2.0, max_area_diff: float = 0.1) -> Dict[str, str]:
        """
        GRDC gauges that lie within ``max_km`` of a Roshydromet gauge whose
        catchment area differs by less than ``max_area_diff``: the same gauge
        under another id and name, as ``{grdc_id: roshydromet_id}``.
        """
        if self._grdc_dups is not None:
            return self._grdc_dups
        static = self._static_table()
        grdc = static[[_is_grdc(stn) for stn in static.index]]
        rosh = static[[not _is_grdc(stn) for stn in static.index]]
        lat1, lon1 = np.radians(grdc['lat'].to_numpy())[:, None], np.radians(grdc['long'].to_numpy())[:, None]
        lat2, lon2 = np.radians(rosh['lat'].to_numpy())[None], np.radians(rosh['long'].to_numpy())[None]
        # haversine distance on a sphere of 6371 km
        h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
        dist = 2 * 6371.0 * np.arcsin(np.sqrt(h))
        area = np.abs(grdc['area_km2'].to_numpy()[:, None] / rosh['area_km2'].to_numpy()[None] - 1)
        i, j = np.nonzero((dist < max_km) & (area < max_area_diff))
        self._grdc_dups = {grdc.index[a]: rosh.index[b] for a, b in zip(i, j)}
        return self._grdc_dups

    @property
    def grdc_stns(self) -> bool:
        """whether the GRDC catchments are served (set at initialization)"""
        return self._grdc_stns

    @property
    def suspect_pcp(self) -> bool:
        """whether the precipitation is served (set at initialization)"""
        return self._suspect_pcp

    @property
    def _root(self) -> os.PathLike:
        """folder into which the archive is extracted"""
        return os.path.join(self.path, "Russia_HydroMeteo_Database_v04")

    @property
    def _nc_dir(self) -> os.PathLike:
        """folder of the station files, one ``<gauge_id>.nc`` per gauge"""
        return os.path.join(self._root, "nc_concat")

    def _stn_path(self, station: str) -> os.PathLike:
        return os.path.join(self._nc_dir, f"{station}.nc")

    @property
    def _static_file(self) -> os.PathLike:
        return os.path.join(self._root, "attributes", "russia_cis_static.csv")

    @property
    def _gauges_file(self) -> os.PathLike:
        return os.path.join(self._root, "geometry", "russia_cis_gauges.gpkg")

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(self._root, "geometry", "russia_cis_ws.gpkg")

    @property
    def boundary_id_map(self) -> str:
        return "gauge_id"

    @property
    def dyn_map(self) -> Dict[str, str]:
        # units are those of the zenodo description, verified against the data:
        # q_mm_day == q_cms_s * 86.4 / ws_area, lvl_sm is in whole cm and
        # lvl_mbs - lvl_sm / 100 is constant (the gauge datum) at every gauge
        return {
            'q_cms_s': observed_streamflow_cms(),                       # m3 s-1
            'q_mm_day': observed_streamflow_mm(),                       # mm day-1
            'lvl_sm': observed_water_level_cm(),                        # cm above gauge datum
            'lvl_mbs': 'lvl_mbs',                                       # m, Baltic Height System
            'prcp_e5l': total_precipitation_with_specifier('era5land'),  # mm day-1
            'prcp_mswep': total_precipitation_with_specifier('mswep'),
            'prcp_imerg': total_precipitation_with_specifier('imerg'),
            'prcp_e5': total_precipitation_with_specifier('era5'),
            'prcp_gpcp': total_precipitation_with_specifier('gpcp'),
            't_max_e5': max_air_temp_with_specifier('era5'),             # deg C
            't_min_e5': min_air_temp_with_specifier('era5'),
            't_max_e5l': max_air_temp_with_specifier('era5land'),
            't_min_e5l': min_air_temp_with_specifier('era5land'),
        }

    @property
    def static_map(self) -> Dict[str, str]:
        return {
            'ws_area': catchment_area(),                    # km2
            'height_bs': gauge_elevation_meters(),          # m, Baltic Height System
            'ele_mt_sav': catchment_elevation_meters(),     # m a.s.l.
            'slp_dg_sav': slope('degrees'),                 # degrees
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

    def _gauges(self) -> pd.DataFrame:
        """name_ru, name_en, lat and long of all 2164 gauges in the gauge
        GeoPackage, which lists every gauge of the dataset"""
        if self._gauges_df is None:
            self._gauges_df = _read_gpkg_points(self._gauges_file, "russia_cis_gauges",
                                                ["gauge_id", "name_ru", "name_en"]
                                                ).set_index("gauge_id")
        return self._gauges_df

    def stations(self) -> List[str]:
        """ids of the 1886 Roshydromet gauges (and of 272 GRDC gauges with
        ``grdc_stns=True``), in the order of the gauge GeoPackage"""
        if self._stations is None:
            if self.grdc_stns:
                duplicates = self._grdc_duplicates()
                self._stations = [stn for stn in self._gauges().index if stn not in duplicates]
            else:
                self._stations = [stn for stn in self._gauges().index if not _is_grdc(stn)]
        return list(self._stations)

    def _static_table(self) -> pd.DataFrame:
        """the cached table of all static features of all 2164 gauges"""
        if self._static_df is None:
            attrs = pd.read_csv(self._static_file, dtype={'gauge_id': str}).set_index('gauge_id')
            area_err = _read_gpkg_table(self.boundary_file, "russia_cis_ws",
                                        ["gauge_id", "ais_dif"]).set_index("gauge_id")
            df = pd.concat([self._gauges(), attrs, area_err], axis=1)
            df.rename(columns=self.static_map, inplace=True)
            df.index.name = 'gauge_id'
            self._static_df = df
        return self._static_df

    def _static_data(self) -> pd.DataFrame:
        # .loc with a list returns a new frame, so that a caller's in-place
        # edit cannot corrupt the cache
        return self._static_table().loc[self.stations()]

    @property
    def static_features(self) -> List[str]:
        return self._static_table().columns.tolist()

    def _raw_vars(self) -> List[str]:
        """raw names of the served dynamic features"""
        pcp = self._pcp_vars if self.suspect_pcp else ()
        return list(self._obs_vars + pcp + self._temp_vars)

    @property
    def dynamic_features(self) -> List[str]:
        return [self.dyn_map[var] for var in self._raw_vars()]

    def _time_index(self) -> pd.DatetimeIndex:
        """time axis of the first gauge's file, which every other gauge's file
        shares (checked by the tests)"""
        if self._time is None:
            with netCDF4.Dataset(self._stn_path(self.stations()[0])) as nc:
                self._time = _time_axis(nc)
        return self._time

    @property
    def start(self) -> pd.Timestamp:
        return self._time_index()[0]

    @property
    def end(self) -> pd.Timestamp:
        return self._time_index()[-1]

    def _reader_spec(self, features: List[str], st=None, en=None) -> Dict:
        """small, picklable description of a read for :func:`_read_russia_stn`"""
        wanted = set(features)
        return dict(folder=self._nc_dir,
                    rename={raw: std for raw, std in self.dyn_map.items() if std in wanted},
                    features=list(features), st=st, en=en, fp=self.fp)

    def _read_stn_dyn(self, station: str) -> pd.DataFrame:
        """
        All dynamic features of one station with ``NaN`` for missing values.
        Values are cast to ``float_precision`` (float32 by default), which
        keeps every published digit of discharge (3 significant digits except
        35 values) and water level (cm, at most one decimal) and changes no
        value by more than 1e-7 (relative); the largest magnitude is ~2e5.
        """
        return _read_russia_stn(self._reader_spec(self.dynamic_features), station)

    def _read_dynamic(
            self,
            stations,
            dynamic_features,
            st: Union[str, pd.Timestamp] = None,
            en: Union[str, pd.Timestamp] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Reads dynamic features of several stations, with a process pool if the
        files to read are large enough (see :func:`n_workers`).
        """
        st, en = self._check_length(st, en)
        features = validate_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')
        if not features:
            raise ValueError("no dynamic feature was requested")
        stations = validate_attributes(stations, self.stations(), 'stations')
        spec = self._reader_spec(features, st, en)

        start = time.time()
        nbytes = len(stations) * os.path.getsize(self._stn_path(stations[0])) if stations else 0
        cpus = n_workers(nbytes, len(stations), self.processes)

        if cpus == 1:
            dyn = {stn: _read_russia_stn(spec, stn) for stn in stations}
        else:
            with cf.ProcessPoolExecutor(cpus) as executor:
                results = executor.map(functools.partial(_read_russia_stn, spec), stations,
                                       chunksize=max(1, len(stations) // (4 * cpus)))
                dyn = dict(zip(stations, results))

        if self.verbosity > 1:
            print(f"Read {len(dyn)} stations for {len(features)} dyn features "
                  f"in {time.time() - start:.2f} seconds with {cpus} cpus.")
        return dyn


def _is_grdc(station: str) -> bool:
    """GRDC gauges have 7-digit ids, Roshydromet gauges 4 or 5 digits"""
    return len(station) == 7


def _time_axis(nc) -> pd.DatetimeIndex:
    """
    Decodes the time variable of a station file, which is named ``date`` in
    most files and ``index`` in some, with units "days since <date>".
    """
    var = nc.variables['date'] if 'date' in nc.variables else nc.variables['index']
    values = var[:]
    if np.ma.is_masked(values):
        raise ValueError(f"{nc.filepath()}: the time axis has missing values")
    calendar = getattr(var, 'calendar', 'standard')
    match = re.fullmatch(r"days since (\d{4}-\d{2}-\d{2})(?: 00:00(?::00)?)?", var.units.strip())
    # the Gregorian calendars only differ before 1582
    if match and calendar in ('standard', 'gregorian', 'proleptic_gregorian') \
            and int(match.group(1)[:4]) > 1582:
        days = np.asarray(values, dtype='int64').astype('timedelta64[D]')
        return pd.DatetimeIndex((np.datetime64(match.group(1), 'D') + days).astype('datetime64[ns]'))
    return pd.DatetimeIndex(netCDF4.num2date(values, var.units, calendar,
                                             only_use_cftime_datetimes=False,
                                             only_use_python_datetimes=True))


def _read_russia_stn(spec: Dict, station: str) -> pd.DataFrame:
    """
    Reads the dynamic data of one station as described by
    :meth:`Russia._reader_spec`. A variable missing from the file (``prcp_e5``
    at the GRDC gauges) is all ``NaN``. It is a module-level function so that
    a process pool pickles only the small ``spec``, not the dataset instance.
    """
    with netCDF4.Dataset(os.path.join(spec['folder'], f"{station}.nc")) as nc:
        index = _time_axis(nc)
        data = {}
        for raw, std in spec['rename'].items():
            if raw in nc.variables:
                # (time, gauge_id) or (time,) depending on the file
                values = np.ma.filled(nc.variables[raw][:].astype('float64'), np.nan).reshape(-1)
                if len(values) != len(index):
                    raise ValueError(f"{station}: {raw} has {len(values)} values for "
                                     f"{len(index)} days")
            else:
                values = np.full(len(index), np.nan)
            data[std] = values

    df = pd.DataFrame(data, index=index)[spec['features']].astype(spec['fp'])
    df = df.loc[spec['st']:spec['en']]
    df.index.name = 'time'
    df.columns.name = 'dynamic_features'
    return df


def _connect_read_only(gpkg: str) -> sqlite3.Connection:
    # immutable: the file is never written, so sqlite needs no locks (which
    # can fail on network file systems)
    return sqlite3.connect(f"{pathlib.Path(gpkg).resolve().as_uri()}?mode=ro&immutable=1",
                           uri=True)


def _read_gpkg_table(gpkg: str, table: str, columns: List[str]) -> pd.DataFrame:
    """the attribute ``columns`` of ``table`` of a GeoPackage, read with sqlite3"""
    con = _connect_read_only(gpkg)
    try:
        cols = ', '.join(f'"{col}"' for col in columns)
        rows = con.execute(f'SELECT {cols} FROM "{table}" ORDER BY fid').fetchall()
    finally:
        con.close()
    return pd.DataFrame(rows, columns=columns)


def _read_gpkg_points(gpkg: str, table: str, columns: List[str]) -> pd.DataFrame:
    """
    The attribute ``columns`` and the ``lat``/``long`` of the point geometry of
    ``table`` of a GeoPackage, read with sqlite3 so that fiona is not needed.
    Raises if the geometries are not 2D points in WGS84 (EPSG:4326).
    """
    con = _connect_read_only(gpkg)
    try:
        geom_col, geom_type, srs_id, z, m = con.execute(
            "SELECT column_name, geometry_type_name, srs_id, z, m FROM gpkg_geometry_columns "
            "WHERE table_name = ?", (table,)).fetchone()
        if geom_type != 'POINT' or srs_id != 4326 or z or m:
            raise ValueError(f"{gpkg}: expected 2D points in EPSG:4326, found "
                             f"{geom_type} in EPSG:{srs_id}")
        cols = ', '.join(f'"{col}"' for col in columns)
        rows = con.execute(f'SELECT {cols}, "{geom_col}" FROM "{table}" ORDER BY fid').fetchall()
    finally:
        con.close()
    long_lat = [_gpkg_point(row[-1]) for row in rows]
    df = pd.DataFrame([row[:-1] for row in rows], columns=columns)
    df[gauge_latitude()] = [lat for _, lat in long_lat]
    df[gauge_longitude()] = [long for long, _ in long_lat]
    return df


def _gpkg_point(blob: bytes) -> Tuple[float, float]:
    """(x, y) of a GeoPackage binary point geometry: a header of 8 bytes and
    an optional envelope, followed by the point as well-known binary"""
    if blob is None or blob[:2] != b'GP':
        raise ValueError("not a GeoPackage geometry")
    envelope = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(blob[3] >> 1) & 0b111]
    wkb = blob[8 + envelope:]
    order = '<' if wkb[0] == 1 else '>'
    (geom_type,) = struct.unpack(f"{order}I", wkb[1:5])
    if geom_type != 1:
        raise ValueError(f"not a 2D point geometry (type {geom_type})")
    return struct.unpack(f"{order}dd", wkb[5:21])
