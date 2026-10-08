import os
import warnings
from typing import Union, List, Dict, Tuple

import pandas as pd

from ..utils import _RainfallRunoff, cache_name
from ..._geom_utils import nzmg_to_wgs84, transform_geometry
from ...utils import download
from .._map import (
    observed_streamflow_cms,
    mean_air_temp,
    total_precipitation,
    total_potential_evapotranspiration,
    mean_rel_hum,
)
from .._map import (
    catchment_area,
    gauge_latitude,
    gauge_longitude,
    slope,
    gauge_elevation_meters,
)
from ._common import (
    _first_and_last_row,
    _remove_stale,
    _extract_zip,
    _warn_duplicate_gauges,
)


class CAMELS_NZ(_RainfallRunoff):
    """
    Daily and hourly data of 369 catchments in New Zealand following
    `Bushra et al., 2025 <https://doi.org/10.5194/essd-2025-244>`_. Release 5 is
    downloaded from
    `figshare <https://doi.org/10.26021/canterburynz.28827644.v5>`_ as one
    archive per variable and timestep, of which only the ones of the ``timestep``
    asked for are fetched: 296 MB for ``D`` and 4.9 GB for ``H``, plus 19 MB of
    attributes and boundaries.

    The five dynamic features are catchment averaged potential
    evapotranspiration (``pet_mm``), precipitation (``pcp_mm``), relative
    humidity (``rh_%``) and air temperature (``airtemp_C_mean``), and observed
    streamflow (``q_cms_obs``, m3/s). Precipitation and potential
    evapotranspiration are mm per timestep and humidity is percent; the
    temperature is published in Kelvin and served in degree Celsius. Streamflow
    covers 1972-01-01 to 2024-04-01 and the meteorology 1972-01-03 to 2024-08-02
    (09:00 at the hourly timestep), each NaN outside its own record: 19208 daily
    or 460978 hourly steps in total. The streamflow of the 14 gauges of
    :attr:`_nodata_stns` needs the owner's permission and is served as NaN.
    The hourly timestamps are New Zealand standard time (UTC+12), not UTC.

    Release 5 corrects two errors of release 2, which this class read before:
    its daily potential evapotranspiration was a mean hourly rate, 24 times too
    small (36 instead of 870 mm a year), and its hourly meteorology was stamped
    in New Zealand local time, which left 50 hours missing and 50 repeated in
    every station file at the daylight saving switches. A copy of release 2 in
    ``path`` is not read any more; it is reported, not deleted, so that you can
    remove it yourself. Note that the hourly meteorology of this dataset is
    disaggregated from the daily one rather than measured hourly, as the Readme
    of the release states.

    Timings on a 48-core machine: the first initialization of the daily data
    downloads 315 MB, extracts it and builds the 143 MB netCDF cache, which
    takes 5 s plus the download and 1.1 GB of disk without the archives.
    Afterwards initialization takes 0.01 s, all 369 stations are fetched in
    0.2 s from that cache or in 1.9 s from the csv files (17 s with
    ``processes=1``), and one station in 0.07 s from either. The hourly data
    is 4.9 GB of downloads, 28 GB extracted and a 3.4 GB cache which takes
    ~1 minute to build; all 369 stations are then fetched from it in 28 s and
    3.6 GB of memory, and one station in 0.7 s.

    Examples
    ---------
    >>> from aqua_fetch import CAMELS_NZ
    >>> dataset = CAMELS_NZ()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='74321', as_dataframe=True)
    >>> df = dynamic['74321'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (19208, 5)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
    369
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (36 out of 369)
    36
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
    [(19208, 5), (19208, 5), ..., (19208, 5)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
    1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ['pet_mm', 'pcp_mm', 'rh_%', 'airtemp_C_mean', 'q_cms_obs']
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('74321', as_dataframe=True,
    ...  dynamic_features=['pcp_mm', 'rh_%', 'airtemp_C_mean', 'pet_mm'])
    >>> dynamic['74321'].shape
    (19208, 4)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
    10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='74321', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['74321'].shape
    ((1, 37), 1, (19208, 5))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    >>> type(dynamic)
    xarray.core.dataset.Dataset
    ...
    >>> dict(dynamic.sizes)
    {'time': 19208, 'dynamic_features': 5}
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
    (369, 2)
    >>> dataset.stn_coords('74321')  # returns coordinates of station whose id is 74321
                     lat        long
    Station_ID
    74321     -45.945599  170.101486
    >>> dataset.stn_coords(['74321', '802'])  # returns coordinates of two stations
    ...
    # get area (km2) of a single station
    >>> dataset.area('74321')
    Station_ID
    74321    400.0
    Name: area_km2, dtype: float32
    # get area of two stations
    >>> dataset.area(['74321', '802'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('74321').type
    'Polygon'
    # The hourly data can be accessed by specifying the timestep to 'H'
    >>> dataset = CAMELS_NZ(timestep='H')
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='74321', as_dataframe=True)
    >>> df = dynamic['74321'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (460978, 5)
    """

    url = "https://doi.org/10.26021/canterburynz.28827644.v5"

    time_steps = ['D', 'H']

    # release of the dataset that this class reads. It is part of the name of
    # the folder and of the netCDF cache, so that release 2, which this class
    # read before and whose daily PET and hourly timestamps are wrong, is never
    # served as this release.
    release = 5

    # archive name (without .zip) -> figshare file id in release 5, from
    # https://api.figshare.com/v2/articles/28827644/versions/5
    _file_ids = {
        "CAMELS_NZ_Catchment_Atrributes": 56902355,
        "CAMELS_NZ_Shapefiles": 56902358,
        "CAMELS_NZ_daily_PET": 65695842,
        "CAMELS_NZ_daily_Precipitation": 56902367,
        "CAMELS_NZ_daily_Relative_Humidity": 56902370,
        "CAMELS_NZ_daily_Streamflow": 56902373,
        "CAMELS_NZ_daily_Temperature": 56902382,
        "CAMELS_NZ_hourly_PET": 61527847,
        "CAMELS_NZ_hourly_Precipitation": 61527853,
        "CAMELS_NZ_hourly_Relative_Humidity": 61527856,
        "CAMELS_NZ_hourly_Streamflow": 61527859,
        "CAMELS_NZ_hourly_Temperature": 61527862,
    }

    # column name in the csv files -> (archive it comes from, name in the file
    # names). The order is the order of :attr:`dynamic_features`.
    _variables = {
        'PET': ('PET', 'PET'),
        'precipitation': ('Precipitation', 'precipitation'),
        'Relative_humidity': ('Relative_Humidity', 'RH'),
        'temperature': ('Temperature', 'temperature'),
        'flow': ('Streamflow', 'flow'),
    }

    # the attribute files of the release, in the order they are joined in
    _attr_files = (
        "1.CAMELS_NZ_Catchment_information.csv",
        "2.CAMELS_NZ_Climatic_attribute.csv",
        "3.CAMELS_NZ_Landcover_attribute.csv",
        "4.CAMELS_NZ_Geology.csv",
        "5.CAMELS_NZ_Anthropogenic_attribute.csv",
    )

    def __init__(self,
                 path: Union[str, os.PathLike] = None,
                 timestep: str = 'D',
                 overwrite: bool = False,
                 to_netcdf: bool = True,
                 verbosity: int = 1,
                 **kwargs):
        """
        Parameters
        ----------
        path : str
            directory under which the data is (or will be) saved in a
            ``CAMELS_NZ`` folder. If None, the default data directory of
            aqua_fetch is used.
        timestep : str
            ``D`` (default) for the daily data or ``H`` for the hourly one. Only
            the archives of this timestep are downloaded and each timestep has
            its own netCDF cache.
        overwrite : bool
            if True, the archives, the extracted folders and the netCDF cache of
            this timestep are deleted and downloaded again.
        to_netcdf : bool
            whether to save the dynamic data of this timestep in a netCDF cache
            for faster reading. Requires netCDF4 and xarray.
        verbosity : int
            0 prints nothing.
        **kwargs :
            any keyword argument of :py:class:`aqua_fetch.rr._RainfallRunoff`
            such as ``processes`` or ``remove_zip``.
        """
        if timestep not in self.time_steps:
            raise ValueError(f"timestep must be one of {self.time_steps}, not {timestep!r}")

        super().__init__(name="CAMELS_NZ", path=path, timestep=timestep,
                         overwrite=overwrite, to_netcdf=to_netcdf,
                         verbosity=verbosity, **kwargs)

        # filled on first use
        self._stns = None
        self._static_feats = None
        self._var_dirs_ = None
        self._period_ = None

        self._warn_old_release()

        self._download_camels_nz(overwrite)

        self._check_manifest()

        self._warn_duplicate_gauges()

        self._maybe_to_netcdf()

    @property
    def timestep_(self) -> str:
        """``daily`` or ``hourly``, the way the file and folder names spell it"""
        return 'daily' if self.timestep == 'D' else 'hourly'

    @property
    def _root(self) -> os.PathLike:
        """folder with the archives and the extracted files of this release"""
        return os.path.join(self.path, f"{self.name.lower()}_v{self.release}")

    @property
    def _archives(self) -> List[str]:
        """archives of release 5 that this timestep needs"""
        return ["CAMELS_NZ_Catchment_Atrributes", "CAMELS_NZ_Shapefiles"] + [
            f"CAMELS_NZ_{self.timestep_}_{archive}"
            for archive, _ in self._variables.values()]

    def _cache_fname(self, timestep: str) -> str:
        """name of the netCDF cache of this release at ``timestep``"""
        return cache_name(f"{self.name.lower()}_v{self.release}_{timestep}.nc")

    @property
    def dyn_fname(self) -> Union[str, os.PathLike]:
        """netCDF cache of this release and timestep, e.g. ``camels_nz_v5_D_v2.nc``"""
        return self._cache_fname(self.timestep)

    def _warn_old_release(self):
        """
        Warns, regardless of verbosity, about a release 2 left in ``path`` by an
        earlier version of this class. It is not read any more but it is not
        deleted either, because it may be the only copy the user has.
        """
        stem = self.name.lower()
        # the folder and the archive of release 2 and the caches built from it,
        # from before and after the cache version was added to their names
        names = [stem, f"{stem}.zip"]
        names += [f"{stem}_{ts}.nc" for ts in self.time_steps]
        names += [cache_name(f"{stem}_{ts}.nc") for ts in self.time_steps]

        stale = [name for name in names
                 if os.path.lexists(os.path.join(self.path, name))]
        if stale:
            warnings.warn(
                f"CAMELS_NZ: {self.path} holds release 2 of the dataset, whose daily "
                f"potential evapotranspiration is 24 times too small and whose hourly "
                f"timestamps skip and repeat the daylight saving hours. It is not read "
                f"any more; release {self.release} is downloaded into {self._root}. "
                f"Delete these yourself to free the disk space: "
                f"{', '.join(stale)}.", UserWarning)
        return

    def _download_camels_nz(self, overwrite: bool = False):
        """
        Downloads and extracts the archives of release 5 that this timestep
        needs. An archive whose folder is already extracted is neither
        downloaded nor extracted again, so deleting the archives
        (``remove_zip=True``) does not force a new download. With
        ``overwrite=True`` this timestep's archives, extracted folders and
        netCDF cache are deleted first; the files of the other timestep are left
        alone.
        """
        os.makedirs(self._root, exist_ok=True)

        folders = {name: os.path.join(self._root, name) for name in self._archives}
        archives = {name: f"{folder}.zip" for name, folder in folders.items()}

        if overwrite:
            _remove_stale([*archives.values(), *folders.values(), self.dyn_fpath],
                          self.verbosity)

        for name, folder in folders.items():
            if os.path.isdir(folder):
                if self.verbosity > 1:
                    print(f"{name} of CAMELS_NZ release {self.release} already exists")
                continue

            archive = archives[name]
            if not os.path.exists(archive):
                if self.verbosity:
                    print(f"downloading {name}.zip of CAMELS_NZ release {self.release}")
                download(url=f"https://ndownloader.figshare.com/files/{self._file_ids[name]}",
                         outdir=self._root, fname=f"{name}.zip", verbosity=self.verbosity)

            _extract_zip(archive, folder, self.verbosity)

        if self.remove_zip:
            # the archives of this timestep only; the ones of the other timestep
            # belong to the instance which downloaded them
            _remove_stale([path for path in archives.values() if os.path.exists(path)],
                          self.verbosity, reason="remove_zip=True")
        return

    def _check_manifest(self):
        """
        Warns if an attribute file, a boundary file or a station time series of
        this timestep is missing, e.g. after an interrupted extraction. The
        expected station files come from the gauge ids of the attribute file,
        not from what happens to be on disk.
        """
        info = os.path.join(self.static_path, self._attr_files[0])
        if not os.path.exists(info):
            raise FileNotFoundError(
                f"{info} not found. Initialize CAMELS_NZ with overwrite=True.")

        expected, missing = 0, []
        folders = {self.static_path: self._attr_files,
                   self.shapefile_path: tuple(
                       f"{os.path.basename(self.boundary_file)[:-len('.shp')]}{ext}"
                       for ext in ('.shp', '.shx', '.dbf', '.prj'))}
        folders.update({self._var_dirs[variable]: tuple(
            os.path.basename(self._stn_file(stn, variable)) for stn in self.stations())
            for variable in self._variables})

        for folder, names in folders.items():
            # one listing per folder instead of one os.path.exists per file
            present = set(os.listdir(folder)) if os.path.isdir(folder) else set()
            expected += len(names)
            missing += [os.path.join(folder, name) for name in names if name not in present]

        if missing:
            warnings.warn(
                f"CAMELS_NZ release {self.release}: {len(missing)} of {expected} files "
                f"of the {self.timestep_} data are missing: {missing[:5]}"
                f"{' ...' if len(missing) > 5 else ''}. "
                f"Use overwrite=True to download them again.", UserWarning)
        return

    def _warn_duplicate_gauges(self):
        """warns if two gauges have the same name and coordinates; both are kept"""
        meta = pd.read_csv(
            os.path.join(self.static_path, self._attr_files[0]),
            usecols=['Station_ID', 'Station Name', 'Latitude (WGS 84)', 'Longitude(WGS 84)'],
            dtype={'Station_ID': str})
        meta.columns = ['gauge_id', 'gauge_name', 'gauge_lat', 'gauge_lon']
        _warn_duplicate_gauges(self.name, meta)
        return

    @property
    def _var_dirs(self) -> Dict[str, os.PathLike]:
        """
        variable -> folder with its csv files at this timestep. The daily PET
        archive of release 5 holds one more folder of the same name, so its
        files are one level deeper than those of the other archives. Resolved
        once, because the folders do not change while the class lives.
        """
        if self._var_dirs_ is None:
            dirs = {}
            for variable, (archive, _) in self._variables.items():
                folder = os.path.join(self._root, f"CAMELS_NZ_{self.timestep_}_{archive}")
                nested = os.path.join(folder, os.path.basename(folder))
                dirs[variable] = nested if os.path.isdir(nested) else folder
            self._var_dirs_ = dirs
        return self._var_dirs_

    @property
    def static_path(self) -> os.PathLike:
        """folder with the five attribute files"""
        return os.path.join(self._root, "CAMELS_NZ_Catchment_Atrributes")

    @property
    def shapefile_path(self) -> os.PathLike:
        """folder with the boundary and gauge shapefiles"""
        return os.path.join(self._root, "CAMELS_NZ_Shapefiles")

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(self.shapefile_path, "All_Nested_Catchments.shp")

    def transform_boundary(self, boundary):
        """
        Transforms a catchment boundary from New Zealand Map Grid (EPSG:27200,
        the CRS of the shapefile) to WGS84 (EPSG:4326) lon/lat, so that it
        matches the gauge coordinates.

        Uses the pyproj-free :func:`nzmg_to_wgs84` helper, which reproduces
        pyproj (EPSG:27200 -> EPSG:4326) on all vertices to below 0.1 mm; that
        transformation itself is accurate to 4 m.
        """
        return transform_geometry(boundary, nzmg_to_wgs84)

    @property
    def pet_path(self) -> os.PathLike:
        """folder with the potential evapotranspiration files of this timestep"""
        return self._var_dirs['PET']

    @property
    def precip_path(self) -> os.PathLike:
        """folder with the precipitation files of this timestep"""
        return self._var_dirs['precipitation']

    @property
    def rh_path(self) -> os.PathLike:
        """folder with the relative humidity files of this timestep"""
        return self._var_dirs['Relative_humidity']

    @property
    def temp_path(self) -> os.PathLike:
        """folder with the air temperature files of this timestep"""
        return self._var_dirs['temperature']

    @property
    def q_path(self) -> os.PathLike:
        """folder with the streamflow files of this timestep"""
        return self._var_dirs['flow']

    @property
    def dyn_map(self) -> Dict[str, str]:
        return {
            'PET': total_potential_evapotranspiration(),
            'precipitation': total_precipitation(),
            'Relative_humidity': mean_rel_hum(),
            'temperature': mean_air_temp(),
            'flow': observed_streamflow_cms(),
        }

    @property
    def static_map(self) -> Dict[str, str]:
        return {
            'Latitude (WGS 84)': gauge_latitude(),
            'Longitude(WGS 84)': gauge_longitude(),
            'uparea': catchment_area(),
            'elevation': gauge_elevation_meters(),
            'usAveSlope': slope('degrees')
        }

    @property
    def dynamic_features(self) -> List[str]:
        """returns names of dynamic features"""
        return [self.dyn_map[variable] for variable in self._variables]

    @property
    def static_features(self) -> List[str]:
        """returns the 37 static features of the New Zealand catchments"""
        if self._static_feats is None:
            self._static_feats = self._static_data().columns.to_list()
        return list(self._static_feats)

    def stations(self) -> List[str]:
        """ids of the 369 gauges, read from the catchment information file"""
        if self._stns is None:
            fpath = os.path.join(self.static_path, self._attr_files[0])
            self._stns = pd.read_csv(fpath, usecols=[0], dtype=str).iloc[:, 0].tolist()
        return list(self._stns)

    @property
    def start(self) -> pd.Timestamp:
        return self._period[0]

    @property
    def end(self) -> pd.Timestamp:
        return self._period[1]

    @property
    def _period(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        First and last timestamp of this timestep, the union over the five
        variables. Only the first and last row of one file per variable is read,
        because every file of a variable has the same time axis (asserted in
        tests/rr/test_camels_nz.py). The timestamps in between come from the
        files themselves, so a gap inside a record is never filled in.
        """
        if self._period_ is None:
            stn = next(stn for stn in self.stations() if stn not in self._nodata_stns)
            starts, ends = [], []
            for variable in self._variables:
                fpath = self._stn_file(stn, variable)
                if not os.path.exists(fpath):
                    continue
                first, last = _first_and_last_row(fpath)
                starts.append(self._timestamp(first))
                ends.append(self._timestamp(last))
            if not starts:
                raise FileNotFoundError(
                    f"no {self.timestep_} time series in {self._root}. "
                    f"Initialize CAMELS_NZ with overwrite=True.")
            self._period_ = (min(starts), max(ends))
        return self._period_

    def _stn_file(self, stn: str, variable: str) -> os.PathLike:
        """path of the csv file of one station and variable at this timestep"""
        archive, stem = self._variables[variable]
        prefix = 'daily_' if self.timestep == 'D' else ''
        return os.path.join(self._var_dirs[variable],
                            f"{prefix}{stem}_station_id_{stn}.csv")

    def _date_format(self, date: str) -> str:
        """
        Format of a date of a time series file. The files are ISO
        (``1972-01-03``, ``1972-01-03 09:00:00``) except the streamflow of a few
        gauges, which was exported with day-first slashes (``1/01/1972``,
        ``1/01/1972 0:00``).
        """
        if '/' in date:
            return '%d/%m/%Y %H:%M' if self.timestep == 'H' else '%d/%m/%Y'
        return '%Y-%m-%d %H:%M:%S' if self.timestep == 'H' else '%Y-%m-%d'

    def _timestamp(self, row: bytes) -> pd.Timestamp:
        """the timestamp at the start of one row of a time series file"""
        date = row.decode().split(',')[0].strip().strip('"')
        return pd.to_datetime(date, format=self._date_format(date))

    @property
    def _nodata_stns(self) -> Tuple[str, ...]:
        """
        The 14 gauges whose streamflow needs the permission of its owner, as
        listed in ``1.Readme.txt`` of the streamflow archive. Their files hold
        blank rows instead of values, so their streamflow is served as NaN.
        """
        return ('75253', '75261', '75265', '75276', '75294', '15408', '15410',
                '15453', '33356', '52916', '74318', '74321', '74368', '1114629')

    def _read_stn_dyn_para(self, stn: str, variable: str) -> pd.Series:
        """
        One variable of one station, with the dates and the values of the file
        unchanged. An empty Series is returned for a gauge whose streamflow
        needs permission, for an empty or malformed file and for a file which is
        not there.
        """
        # an empty DatetimeIndex, so that concatenating a variable which is not
        # there with the others keeps the index a DatetimeIndex
        empty = pd.Series(dtype=self.fp, name=variable, index=pd.DatetimeIndex([]))

        if variable == 'flow' and stn in self._nodata_stns:
            return empty

        fpath = self._stn_file(stn, variable)
        if not os.path.exists(fpath):
            if self.verbosity > 1:
                print(f"{fpath} does not exist. Skipping {variable} of station {stn}.")
            return empty

        try:
            df = pd.read_csv(fpath, index_col=0, na_values=['NA  '])
        except pd.errors.EmptyDataError:
            warnings.warn(f"CAMELS_NZ: {fpath} is empty, {variable} of station "
                          f"{stn} is served as NaN.", UserWarning)
            return empty

        if variable not in df.columns or df.empty:
            warnings.warn(f"CAMELS_NZ: {fpath} has no {variable} values, they are "
                          f"served as NaN.", UserWarning)
            return empty

        # the format is read from the first date instead of being inferred per
        # file, which is both faster and loud when a release changes it
        df.index = pd.to_datetime(df.index, format=self._date_format(str(df.index[0])))

        values = df[variable]
        if variable == 'temperature':
            # published in Kelvin, the canonical name promises degree Celsius.
            # Converted before the cast, so that the result is the published
            # value rounded once instead of a difference of two rounded ones.
            values = values - 273.15

        stn_q = values.astype(self.fp).rename(variable)

        duplicated = stn_q.index.duplicated(keep='first')
        if duplicated.any():
            # release 5 has none; a release which reintroduces local time would
            warnings.warn(f"CAMELS_NZ: dropping {int(duplicated.sum())} rows of {fpath} "
                          f"whose timestamps are repeated.", UserWarning)
            stn_q = stn_q[~duplicated]

        return stn_q

    def _read_stn_dyn(self, stn: str) -> pd.DataFrame:
        """
        dynamic data of one station, one column per variable. The index is the
        union of the time axes of the variables.
        """
        stn_df = pd.concat(
            [self._read_stn_dyn_para(stn, variable) for variable in self._variables],
            axis=1)

        stn_df.rename(columns=self.dyn_map, inplace=True)

        return stn_df

    def _static_data(self) -> pd.DataFrame:
        """
        static attributes of catchments

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of static features of all catchments of
            shape (369, 37). The gauge id, name and coordinates, which all five
            attribute files carry, are kept from the catchment information file
            only.
        """
        dfs = []
        for idx, fname in enumerate(self._attr_files):
            df = pd.read_csv(os.path.join(self.static_path, fname), index_col=0)
            df.index = df.index.astype(str)

            if idx:
                df = df.drop(columns=['RID', 'StationName', 'latitude', 'longitude'],
                             errors='ignore')

            dfs.append(df)

        static_data = pd.concat(dfs, axis=1)

        return static_data.rename(columns=self.static_map)

    def _redundant_after_consolidation(self) -> List[Tuple[str, str]]:
        # The five variable folders of a timestep become redundant once that
        # timestep's netCDF cache is built. Both timesteps are listed regardless
        # of self.timestep so that cleanup works whichever instance the user
        # invokes free_disk_space on.
        pairs: List[Tuple[str, str]] = []
        for ts_code, ts_word in (("D", "daily"), ("H", "hourly")):
            cache = os.path.join(self.path, self._cache_fname(ts_code))
            for archive, _ in self._variables.values():
                pairs.append(
                    (os.path.join(self._root, f"CAMELS_NZ_{ts_word}_{archive}"), cache)
                )
        return pairs
