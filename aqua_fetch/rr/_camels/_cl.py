import os
import glob
import zlib
import shutil
import zipfile
import warnings
import functools
import concurrent.futures as cf
from typing import Union, List, Dict, Tuple

import numpy as np
import pandas as pd

from ..utils import _RainfallRunoff, n_workers, cache_name
from ...utils import BROWSER_HEADERS
from ...utils import validate_attributes, download
from .._map import (
    observed_streamflow_cms,
    observed_streamflow_mm,
    mean_air_temp,
    max_air_temp,
    min_air_temp,
    total_precipitation_with_specifier,
    total_potential_evapotranspiration_with_specifier,
)
from .._map import (
    catchment_area,
    gauge_latitude,
    gauge_longitude,
    slope,
    catchment_elevation_meters,
    med_catchment_elevation_meters,
    min_catchment_elevation_meters,
    max_catchment_elevation_meters,
)
from ._common import (
    _row_date,
    _first_and_last_row,
    _remove_stale,
    _warn_duplicate_gauges,
)


_CL_PANGAEA_URL = "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/"


# the 2022 archive extracts into a folder holding another folder of the same name
_CL_2022_DIR = os.path.join("CAMELS_CL_v202201", "CAMELS_CL_v202201")


class CAMELS_CL(_RainfallRunoff):
    """
    Daily data of 516 Chilean catchments following
    `Alvarez-Garreton et al., 2018 <https://doi.org/10.5194/hess-22-5817-2018>`_ .
    Two releases are available through ``version``:

    - ``2022`` (default): the January 2022 release from
      `cr2.cl <https://www.cr2.cl/datos-informacion-integrada-por-cuencas/>`_ ,
      with 10 dynamic and 110 static features on a time index from 1900-01-01
      to 2021-06-22.
    - ``2018``: the August 2018 release from
      `PANGAEA <https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/>`_ ,
      with 12 dynamic and 104 static features from 1913-02-15 to 2018-03-09.

    The time index is the union of the dates in the files, and each feature is
    NaN outside its record. In 2022, observed streamflow runs from 1913-02-15 to
    2020-06-06, CR2MET precipitation, air temperature and Hargreaves PET from
    1979-01-01 to 2020-04-30, MSWEP to 2019-12-31, CHIRPS from 1981-01-01 to
    2019-12-31 and TMPA from 1998-01-01 to 2018-12-31.

    Compared with 2018, the 2022 release extends the records, uses newer CR2MET
    and MSWEP versions, redraws 27 catchment boundaries and corrects 570 daily
    flows of 1000 m3/s or more (15 gauges, 2016-2018) that 2018 stores as 1 to 4.
    Snow water equivalent (``swe``, mm) and MODIS PET (``pet_mm_modis``, 8-day
    totals in mm) exist only in 2018.

    Units: ``q_cms_obs`` m3/s; ``q_mm_obs``, ``pcp_mm_*`` and
    ``pet_mm_hargreaves`` mm/day; ``airtemp_C_*`` degree Celsius. Mean slope is
    ``slope_mkm-1`` (m/km) in 2018 and ``slope_%`` in 2022. Dynamic values are
    cast to ``float_precision`` (float32 by default, relative error below 1e-7).
    Not provided: the monthly aggregates, the yearly water-rights series and the
    water-rights register of the 2022 archive, and the 2018 catchment hierarchy.

    Timings for 2022 on a 48-core machine: the first initialization downloads
    288 MB, then extracts it and builds a 918 MB netCDF cache in about 6 s (2 GB
    of disk in total). Afterwards all 516 stations with all features are fetched
    in 0.7 s from the cache (2 s from the csv files) and one station in 0.1 s
    (0.6 s). The csv files are read in parallel processes for 100 or more
    stations or with ``float_precision=np.float64``, so on Windows and macOS run
    such scripts, including the first initialization, under
    ``if __name__ == "__main__":``.

    Examples
    ---------
    >>> from aqua_fetch import CAMELS_CL
    >>> dataset = CAMELS_CL()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='8350001', as_dataframe=True)
    >>> df = dynamic['8350001'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (44368, 10)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
    516
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (51 out of 516)
    51
    ...
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ['q_cms_obs', 'q_mm_obs', 'pcp_mm_cr2met', 'pcp_mm_chirps', 'pcp_mm_mswep', 'pcp_mm_tmpa', 'airtemp_C_min', 'airtemp_C_max', 'airtemp_C_mean', 'pet_mm_hargreaves']
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('8350001', as_dataframe=True,
    ...  dynamic_features=['pet_mm_hargreaves', 'pcp_mm_mswep', 'airtemp_C_mean', 'q_cms_obs'])
    >>> dynamic['8350001'].shape
    (44368, 4)
    ...
    ... # get data of a selected period
    >>> _, dynamic = dataset.fetch('8350001', st='2000-01-01', en='2000-12-31', as_dataframe=True)
    >>> dynamic['8350001'].shape
    (366, 10)
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='8350001', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['8350001'].shape
    ((1, 110), 1, (44368, 10))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    >>> type(dynamic)
    <class 'xarray.core.dataset.Dataset'>
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
    (516, 2)
    >>> dataset.stn_coords('8350001')  # returns coordinates of station whose id is 8350001
                    lat     long
    gauge_id
    8350001  -38.214199 -71.8283
    ...
    # get area (km2) of two stations
    >>> dataset.area(['8350001', '3820003'])
    gauge_id
    8350001      46.104347
    3820003    7383.890625
    Name: area_km2, dtype: float32
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('8350001')
    ...
    # the 2018 release
    >>> dataset = CAMELS_CL(version=2018)
    >>> _, dynamic = dataset.fetch(stations='8350001', as_dataframe=True)
    >>> dynamic['8350001'].shape
    (38374, 12)
    """

    # todo : Duplicated code: CAMELS_GB, CAMELS_KR and CAMELS_CL each have their own "extract to a temporary folder"

    # archives of each release: file name -> url
    urls = {
        2018: {f"{stem}.zip": f"{_CL_PANGAEA_URL}{stem}.zip" for stem in (
            "1_CAMELScl_attributes", "2_CAMELScl_streamflow_m3s", "3_CAMELScl_streamflow_mm",
            "4_CAMELScl_precip_cr2met", "5_CAMELScl_precip_chirps", "6_CAMELScl_precip_mswep",
            "7_CAMELScl_precip_tmpa", "8_CAMELScl_tmin_cr2met", "9_CAMELScl_tmax_cr2met",
            "10_CAMELScl_tmean_cr2met", "11_CAMELScl_pet_8d_modis", "12_CAMELScl_pet_hargreaves",
            "13_CAMELScl_swe", "14_CAMELScl_catch_hierarchy", "CAMELScl_catchment_boundaries",
        )},
        2022: {"CAMELS_CL_v202201.zip": "https://www.cr2.cl/download/camels-cl-v202201/?wpdmdl=35317"},
    }

    # daily time series of each release: raw feature name -> file, relative to ``path``
    _dyn_files = {
        2018: {feature: os.path.join(stem, f"{stem}.txt") for feature, stem in (
            ('streamflow_m3s', '2_CAMELScl_streamflow_m3s'),
            ('streamflow_mm', '3_CAMELScl_streamflow_mm'),
            ('precip_cr2met', '4_CAMELScl_precip_cr2met'),
            ('precip_chirps', '5_CAMELScl_precip_chirps'),
            ('precip_mswep', '6_CAMELScl_precip_mswep'),
            ('precip_tmpa', '7_CAMELScl_precip_tmpa'),
            ('tmin_cr2met', '8_CAMELScl_tmin_cr2met'),
            ('tmax_cr2met', '9_CAMELScl_tmax_cr2met'),
            ('tmean_cr2met', '10_CAMELScl_tmean_cr2met'),
            ('pet_8d_modis', '11_CAMELScl_pet_8d_modis'),
            ('pet_hargreaves', '12_CAMELScl_pet_hargreaves'),
            ('swe', '13_CAMELScl_swe'),
        )},
        2022: {feature: os.path.join(_CL_2022_DIR, fname) for feature, fname in (
            ('streamflow_m3s', 'q_m3s_day.csv'),
            ('streamflow_mm', 'q_mm_day.csv'),
            ('precip_cr2met', 'precip_cr2met_mm_day.csv'),
            ('precip_chirps', 'precip_chirps_mm_day.csv'),
            ('precip_mswep', 'precip_mswep_mm_day.csv'),
            ('precip_tmpa', 'precip_tmpa_mm_day.csv'),
            ('tmin_cr2met', 'tmin_cr2met_C_day.csv'),
            ('tmax_cr2met', 'tmax_cr2met_C_day.csv'),
            ('tmean_cr2met', 'tmean_cr2met_C_day.csv'),
            ('pet_hargreaves', 'pet_hargreaves_mm_day.csv'),
        )},
    }

    def __init__(
            self,
            path: str = None,
            version: int = 2022,
            overwrite: bool = False,
            to_netcdf: bool = True,
            verbosity: int = 1,
            **kwargs
    ):
        """
        Parameters
        ----------
        path : str
            directory under which the data is (or will be) saved in a
            ``CAMELS_CL`` folder. Both releases can share it. If None, the
            default data directory of aqua_fetch is used.
        version : int
            ``2022`` (default) or ``2018``.
        overwrite : bool
            if True, the archives, extracted files and netCDF caches (of all
            precisions) of this ``version`` are deleted and downloaded again.
        to_netcdf : bool
            whether to save the dynamic data of this ``version`` in a netCDF
            cache for faster reading. Requires netCDF4 and xarray.
        verbosity : int
            0 prints nothing.
        **kwargs :
            any keyword argument of :py:class:`aqua_fetch.rr._RainfallRunoff`
            such as ``processes``.
        """
        # isinstance, because 2022.0 == 2022 would name a second cache camels_cl_D_2022.0_v2.nc
        if not isinstance(version, (int, np.integer)) or version not in self.urls:
            raise ValueError(f"version must be one of {list(self.urls)} but is {version!r}")
        self.version = int(version)
        self._all_dates = None  # see _dates_of_all_files

        super().__init__(path=path, overwrite=overwrite, to_netcdf=to_netcdf,
                         verbosity=verbosity, **kwargs)

        self._download_camels_cl(overwrite)

        self._check_manifest()

        self._check_duplicates()

        self._maybe_to_netcdf()

    def _download_camels_cl(self, overwrite: bool = False):
        """
        Downloads and extracts the archives of ``self.version`` whose extracted
        folders do not exist, so an archive deleted after extraction is not
        downloaded again. ``overwrite=True`` first deletes this version's
        archives, extracted folders and netCDF caches.
        """
        archives = {os.path.join(self.path, fname): url for fname, url in self.urls[self.version].items()}
        folders = [archive[:-len(".zip")] for archive in archives]

        if overwrite:
            caches = glob.glob(os.path.join(glob.escape(self.path),
                                            f"{self.name.lower()}_{self.timestep}_{self.version}*.nc"))
            _remove_stale([*archives, *folders, *caches], self.verbosity)

        os.makedirs(self.path, exist_ok=True)
        for (archive, url), folder in zip(archives.items(), folders):
            if os.path.exists(folder):
                continue
            if not os.path.exists(archive):
                if self.verbosity:
                    print(f"downloading {url} to {archive}")
                # cr2.cl answers the default user agent of urllib with 403 Forbidden
                download(url, outdir=self.path, fname=os.path.basename(archive),
                         verbosity=self.verbosity,
                         headers=BROWSER_HEADERS if self.version == 2022 else None)

            # into a temporary folder that is renamed once complete, so that an
            # interrupted extraction is redone instead of being taken as complete
            if self.verbosity:
                print(f"extracting {archive}")
            partial = f"{folder}_extracting"
            shutil.rmtree(partial, ignore_errors=True)
            try:
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(partial)
            except (zipfile.BadZipFile, zlib.error, EOFError):  # e.g. an error page saved as the archive
                shutil.rmtree(partial, ignore_errors=True)
                os.remove(archive)
                raise ValueError(f"{archive} is corrupt and was deleted. "
                                 f"Initialize CAMELS_CL again to download it again.") from None
            os.replace(partial, folder)

        if self.remove_zip:
            for archive in archives:
                if os.path.exists(archive):
                    if self.verbosity:
                        print(f"remove_zip=True: removing {archive}")
                    os.remove(archive)
        return

    def _check_manifest(self):
        """warns if files of ``self.version`` are missing, e.g. deleted by hand"""
        shapefile = [self.boundary_file[:-len(".shp")] + ext for ext in (".shp", ".shx", ".dbf", ".prj")]
        files = [self._static_file, *shapefile, *self._dyn_paths.values()]
        missing = [fpath for fpath in files if not os.path.exists(fpath)]
        if missing:
            warnings.warn(
                f"CAMELS_CL {self.version}: {len(missing)} of {len(files)} files are missing: "
                f"{missing}. Use overwrite=True to download them again.", UserWarning)
        return

    def _check_duplicates(self):
        """warns if two gauges share a name and rounded coordinates"""
        meta = self._attributes[['gauge_name', gauge_latitude(), gauge_longitude()]].reset_index()
        meta.columns = ['gauge_id', 'gauge_name', 'gauge_lat', 'gauge_lon']
        _warn_duplicate_gauges(self.name, meta)
        return

    @property
    def _static_file(self) -> os.PathLike:
        if self.version == 2018:
            return os.path.join(self.path, "1_CAMELScl_attributes", "1_CAMELScl_attributes.txt")
        return os.path.join(self.path, _CL_2022_DIR, "catchment_attributes.csv")

    @property
    def boundary_file(self) -> os.PathLike:
        if self.version == 2018:
            return os.path.join(self.path, "CAMELScl_catchment_boundaries",
                                "CAMELScl_catchment_boundaries", "catchments_camels_cl_v1.3.shp")
        return os.path.join(self.path, _CL_2022_DIR, "camels_cl_boundaries", "camels_cl_boundaries.shp")

    @property
    def boundary_id_map(self) -> str:
        return "gauge_id"

    @property
    def _dyn_paths(self) -> Dict[str, os.PathLike]:
        """standardized name of each dynamic feature -> path of its file"""
        return {self.dyn_map.get(raw, raw): os.path.join(self.path, fname)
                for raw, fname in self._dyn_files[self.version].items()}

    @property
    def _ts_format(self) -> Dict:
        """how the time series files of ``self.version`` are written"""
        if self.version == 2018:
            return dict(sep='\t', date_col='gauge_id', na_values=[' '])
        return dict(sep=',', date_col='date', na_values=None)

    @property
    def dyn_fname(self) -> Union[str, os.PathLike]:
        """name of the netCDF cache of the dynamic data, one per ``version`` and
        precision, e.g. camels_cl_D_2022_v2.nc or camels_cl_D_2022_float64_v2.nc"""
        precision = "" if np.dtype(self.fp) == np.float32 else f"_{np.dtype(self.fp).name}"
        return cache_name(f"{self.name.lower()}_{self.timestep}_{self.version}{precision}.nc")

    @property
    def static_map(self) -> Dict[str, str]:
        if self.version == 2018:
            return {
                'area': catchment_area(),                         # km2
                'slope_mean': slope('mkm-1'),                     # m/km
                'gauge_lat': gauge_latitude(),
                'gauge_lon': gauge_longitude(),
            }
        # area_km2 already has the standardized name
        return {
            'mean_slope_perc': slope('%'),                        # % (the 2018 m/km value / 10)
            'mean_elev': catchment_elevation_meters(),            # m a.s.l.
            'med_elev': med_catchment_elevation_meters(),
            'min_elev': min_catchment_elevation_meters(),
            'max_elev': max_catchment_elevation_meters(),
            'gauge_lat': gauge_latitude(),
            'gauge_lon': gauge_longitude(),
        }

    @property
    def dyn_map(self) -> Dict[str, str]:
        # swe (mm) keeps its raw name; pet_8d_modis is mm per 8 days
        return {
            'streamflow_m3s': observed_streamflow_cms(),
            'streamflow_mm': observed_streamflow_mm(),
            'tmin_cr2met': min_air_temp(),
            'tmax_cr2met': max_air_temp(),
            'tmean_cr2met': mean_air_temp(),
            'precip_mswep': total_precipitation_with_specifier('mswep'),
            'precip_tmpa': total_precipitation_with_specifier('tmpa'),
            'precip_cr2met': total_precipitation_with_specifier('cr2met'),
            'precip_chirps': total_precipitation_with_specifier('chirps'),
            'pet_hargreaves': total_potential_evapotranspiration_with_specifier('hargreaves'),
            'pet_8d_modis': total_potential_evapotranspiration_with_specifier('modis'),
        }

    @functools.cached_property
    def _time_extent(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """first and last date over the time series files of ``self.version``.
        A missing file raises, rather than silently narrowing the dates."""
        sep = self._ts_format['sep'].encode()
        dates = [pd.Timestamp(_row_date(row, sep)) for fpath in self._dyn_paths.values()
                 for row in _first_and_last_row(fpath)]
        return min(dates), max(dates)

    def _dates_of_all_files(self, known: Dict[str, pd.DatetimeIndex]) -> pd.DatetimeIndex:
        """union of the dates in the time series files of ``self.version``, found
        once; the dates of the files in ``known`` are not read again"""
        if self._all_dates is None:
            sep = self._ts_format['sep'].encode()
            dates = [known[fpath] if fpath in known else _read_camels_cl_dates(fpath, sep)
                     for fpath in self._dyn_paths.values()]
            self._all_dates = functools.reduce(pd.DatetimeIndex.union, dates).rename('time')
        return self._all_dates

    @property
    def start(self) -> pd.Timestamp:
        return self._time_extent[0]

    @property
    def end(self) -> pd.Timestamp:
        return self._time_extent[1]

    @property
    def location(self):
        return "Chile"

    @functools.cached_property
    def _attributes(self) -> pd.DataFrame:
        """catchment attributes of all gauges with gauge_id as index, read once"""
        if self.version == 2018:
            # gauges are columns, and ids and numbers are padded with spaces
            df = pd.read_csv(self._static_file, sep='\t', index_col='gauge_id', dtype=str).T
            df.index = df.index.str.strip()
            df.columns.name = None
            for col in df.columns:
                text = df[col].str.strip()
                numbers = pd.to_numeric(text, errors='coerce')
                if numbers.notna().sum() == text.notna().sum():  # every value is a number
                    # float() reads e.g. 0.00000000000000000003 exactly, to_numeric does not
                    df[col] = numbers if numbers.dtype.kind == 'i' else text.astype(float)
        else:
            df = pd.read_csv(self._static_file, index_col='gauge_id', dtype={'gauge_id': str},
                             float_precision='round_trip')
        df.index.name = 'gauge_id'
        return df.rename(columns=self.static_map)

    def _static_data(self) -> pd.DataFrame:
        return self._attributes.copy()

    def stations(self) -> List[str]:
        """ids of the 516 gauges, as listed in the catchment attributes table"""
        return self._attributes.index.tolist()

    @property
    def static_features(self) -> List[str]:
        return self._attributes.columns.tolist()

    @property
    def dynamic_features(self) -> List[str]:
        return list(self._dyn_paths)

    @property
    def _mm_feature_name(self) -> str:
        return observed_streamflow_mm()

    def _read_stn_dyn(self, stn: str) -> pd.DataFrame:
        return self._read_dynamic([stn], 'all')[stn]

    def _read_dynamic(
            self,
            stations,
            dynamic_features,
            st: Union[str, pd.Timestamp] = None,
            en: Union[str, pd.Timestamp] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Reads each file of ``dynamic_features`` once for all ``stations``, in
        parallel when the files are large. Returns one DataFrame per station on
        the dates of all the files of ``self.version`` between ``st`` and ``en``.
        """
        st, en = self._check_length(st, en)
        stations = validate_attributes(stations, self.stations(), 'stations')
        features = validate_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')

        # pandas' default parser reads e.g. 0.0000000000000000016 as 0. 'round_trip'
        # is exact but holds the GIL. 'legacy' releases it and, cast to float32,
        # gives the same values as 'round_trip' for every value of both releases.
        exact = np.dtype(self.fp).itemsize > 4
        paths = [self._dyn_paths[feature] for feature in features]
        read = functools.partial(_read_camels_cl_ts, stations=stations, dtype=self.fp,
                                 float_precision='round_trip' if exact else 'legacy', **self._ts_format)

        workers = n_workers(sum(map(os.path.getsize, paths)), len(paths), self.processes)
        if workers == 1:
            results = [read(fpath) for fpath in paths]
        else:
            # Threads need no ``if __name__ == "__main__"`` guard on Windows or
            # macOS and are as fast as processes for a few stations; processes
            # are faster for many stations and for the GIL-holding 'round_trip'.
            pool = cf.ProcessPoolExecutor if len(stations) >= 100 or exact else cf.ThreadPoolExecutor
            with pool(workers) as executor:
                results = list(executor.map(read, paths))

        # the same dates whichever features are read, as in the netCDF cache
        time = self._dates_of_all_files(dict(zip(paths, (dates for dates, _ in results))))
        time = time[(time >= st) & (time <= en)]

        # (stations, time, features), so that each station is a contiguous block. The
        # frames share this array, which is 3 times faster to fill than one per station.
        data = np.full((len(stations), len(time), len(features)), np.nan, dtype=self.fp)
        for k, (dates, values) in enumerate(results):
            rows = time.get_indexer(dates)
            inside = rows >= 0
            data[:, rows[inside], k] = values[inside].T

        columns = pd.Index(features, name='dynamic_features')
        return {stn: pd.DataFrame(data[j], index=time, columns=columns)
                for j, stn in enumerate(stations)}


def _read_camels_cl_ts(
        fpath: str,
        stations: List[str],
        dtype,
        float_precision: str,
        sep: str,
        date_col: str,
        na_values,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads the columns of ``stations`` from one CAMELS-CL time series file, whose
    rows are days and columns are gauges. Returns the dates and a
    (days, stations) array. A module-level function, so that a process pool
    does not have to pickle the dataset.
    """
    df = pd.read_csv(fpath, sep=sep, usecols=[date_col] + stations, na_values=na_values,
                     dtype={stn: dtype for stn in stations}, float_precision=float_precision)
    return _checked_dates(df[date_col], fpath), df[stations].to_numpy()


def _checked_dates(dates, fpath: str) -> pd.DatetimeIndex:
    """the dates of a CAMELS-CL time series file, which must be sorted, because
    CAMELS_CL.start/end are read from the first and last rows only, and unique,
    because the values of a repeated date would overwrite each other"""
    dates = pd.DatetimeIndex(pd.to_datetime(dates, format="%Y-%m-%d"))
    if not (dates.is_monotonic_increasing and dates.is_unique):
        raise ValueError(f"dates in {fpath} are not sorted or not unique")
    return dates


def _read_camels_cl_dates(fpath: str, sep: bytes) -> pd.DatetimeIndex:
    """dates in the first column of a CAMELS-CL time series file, without parsing its values"""
    with open(fpath, 'rb') as f:
        f.readline()  # header
        return _checked_dates([_row_date(row, sep) for row in f if row.strip()], fpath)
