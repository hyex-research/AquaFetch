import io
import os
import glob
import warnings
import functools
from typing import Union, List, Dict, Tuple

import numpy as np
import pandas as pd

from ..utils import _RainfallRunoff, cache_name, _path_size
from ..._geom_utils import world_mercator_to_wgs84
from ..._backend import fiona
from .._map import (
    observed_streamflow_cms,
    max_air_temp,
    min_air_temp,
    total_precipitation,
    total_potential_evapotranspiration,
)
from .._map import (
    catchment_area,
    gauge_latitude,
    gauge_longitude,
    gauge_elevation_meters,
    catchment_elevation_meters,
    catchment_perimeter,
    min_catchment_elevation_meters,
    max_catchment_elevation_meters,
)
from ._common import _row_date, _first_and_last_row, _remove_stale, _extract_zip


class CAMELS_COL(_RainfallRunoff):
    """
    Daily hydrometeorological time series and static catchment attributes for
    346 catchments in Colombia following
    `Jimenez et al., 2025 <https://doi.org/10.5194/essd-2025-200>`_
    (CAMELS-COL), downloaded from its
    `zenodo record <https://zenodo.org/records/18794895>`_.

    The dataset has 5 dynamic features from 1981-01-01 to 2022-12-31 (15340
    daily steps) and 79 static features. Precipitation is CHIRPS v2.0, the
    temperatures and the potential evapotranspiration are MSWX and the
    streamflow is observed by IDEAM. The gauge positions and the catchment
    boundary shapefiles are both in EPSG:3395: the record's ``gauge_lat`` and
    ``gauge_lon`` are a northing and an easting **in metres**, so the ``lat``
    and ``long`` served here, like :meth:`get_boundary`, are converted to
    WGS84 and do not match those two columns of the source table.

    Every gauge has gaps: its file holds only the days on which the streamflow
    was observed, between 4151 and 15308 of the 15340 days, and the served
    series are padded with ``NaN`` to the common daily index. **The gridded
    forcings are shipped only for those same days**, so ``pcp_mm``, ``pet_mm``
    and the two temperatures carry exactly the gaps of ``q_cms_obs`` and this
    dataset cannot give a complete 1981-2022 CHIRPS/MSWX series. Gauge
    11017010, for instance, starts on 1981-05-19 although both products begin
    on 1981-01-01. The values are served as ``float32``, the precision of this
    library: the source writes at most 3 decimals and float32 reproduces every
    one of them, bar 0.001 m3/s on 6 discharges of gauge 31097010.

    ``gauge_department``, ``gauge_star`` and ``gauge_end`` are text among
    otherwise numeric attributes; the last two are the first and last day of
    each record, written as in the source (``1/01/1981``), and with ``gauge_n``
    they give a gauge's record window without reading its file.
    ``01_CAMELS_COL_Attributes.zip``, which is downloaded and extracted,
    describes every attribute. Gauges 21247040 and 21247050 share a position;
    both are kept and a warning names them. The gauge point shapefile
    ``CAMELS_COL_GAUGING.shp`` is downloaded but not read by this class.

    Of the 3.5 GB record, the 199 MB which are observations are downloaded. The
    drainage network, the figures and the 3.2 GB topographic wetness index,
    which is derived from the DEM, are not.

    This release (February 2026) supersedes the May 2025 one, whose zenodo
    record is `restricted <https://zenodo.org/records/15554735>`_ and serves no
    file. This release has one catchment less (13077030 was dropped), revised
    catchment boundaries (162 of the 346 changed, and with them areas by up to
    17 %, the precipitation of 26 gauges by up to 16 mm and the temperatures
    and potential evapotranspiration of a few by up to 1.1 degC and 0.9 mm; the
    streamflow is unchanged apart from the third decimal the new files carry),
    recomputed signatures and indices, and no
    ``t_mean`` column. Its 255 static features included 188 lithology codes and
    19 land cover classes which this release aggregates into 7 rock types and 6
    land cover shares. Files of that release which are already in ``path`` are
    neither read nor deleted; this release is kept in its own sub-folder.

    The first initialization took ~2 minutes: downloading and extracting the
    199 MB and building the netCDF cache. Afterwards initialization takes
    ~0.01 s and reading all 346 gauges ~0.3 s.

    Examples
    ---------
    >>> from aqua_fetch import CAMELS_COL
    >>> dataset = CAMELS_COL()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='35067040', as_dataframe=True)
    >>> df = dynamic['35067040'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (15340, 5)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       346
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (34 out of 346)
       34
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(15340, 5), (15340, 5), (15340, 5),... (15340, 5), (15340, 5)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ['pcp_mm', 'pet_mm', 'airtemp_C_min', 'airtemp_C_max', 'q_cms_obs']
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('35067040', as_dataframe=True,
    ...  dynamic_features=['pcp_mm', 'airtemp_C_max', 'pet_mm', 'q_cms_obs'])
    >>> dynamic['35067040'].shape
       (15340, 4)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='35067040', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['35067040'].shape
    ((1, 79), 1, (15340, 5))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 15340, 'dynamic_features': 5})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (346, 2)
    >>> dataset.stn_coords('35067040')  # returns coordinates of station whose id is 35067040
                    lat       long
    gauge_id
    35067040   4.778274 -73.587807
    >>> dataset.stn_coords(['35067040', '21187030'])  # returns coordinates of two stations
    ...
    # get area (km2) of a single station
    >>> dataset.area('35067040')
    # get areas of two stations
    >>> dataset.area(['35067040', '21187030'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('35067040')
    """

    url = "https://zenodo.org/records/18794895"

    # this release is kept in this sub-folder of ``path`` so that the files of
    # the superseded 2025 release, if they are already there, are neither read
    # nor overwritten
    _RELEASE_DIR = 'camels_col'

    # the release, used in the name of the netCDF cache so that a cache built
    # for another release, or for another precision, is never read as this one
    _RELEASE = '2026'

    # archives of the record which are downloaded, each extracted into a folder
    # of its own name
    _ARCHIVES = (
        '01_CAMELS_COL_Attributes.zip',                 # readme and attribute descriptions
        '03_CAMELS_COL_Basin_boundary.zip',             # catchment and gauge shapefiles
        '04_CAMELS_COL_Hydrometeorological_data.zip',   # one time series file per gauge
    )

    # attribute tables of the record, in the order in which they are served.
    # The record also has 00 (a .docx description), 12 (the drainage network),
    # 13 (the topographic wetness index, a 3.2 GB raster derived from the DEM)
    # and 14 (figures). None of them is read by the class, and 13 is a derived
    # index rather than an observation, so none of them is downloaded.
    _TABLES = (
        '02_CAMELS_COL_Catchment_information.csv',
        '05_CAMELS_COL_Geologic_characteristics.csv',
        '06_CAMELS_COL_Land_cover_characteristics.csv',
        '07_CAMELS_COL_Soil_characteristics.csv',
        '08_CAMELS_COL_Climatic_indices.csv',
        '09_CAMELS_COL_Hydrological_signatures.csv',
        '10_CAMELS_COL_Physiograpic_characteristics.csv',
        '11_CAMELS_COL_Land_use_capability.csv',
    )

    # the land cover (06) and the soil (07) table both name their water body
    # share water_bodies_perc, the first after Mapbiomas and the second after
    # the IGAC soil map, so the soil one is renamed to keep both readable
    _RENAMED = {'07_CAMELS_COL_Soil_characteristics.csv':
                {'water_bodies_perc': 'water_bodies_soil_perc'}}

    # what the superseded 2025 release left directly in ``path``: its attribute
    # workbooks, its archives, its netCDF caches and its description
    # '*.nc' is unambiguous: this release keeps its cache in _RELEASE_DIR
    _SUPERSEDED = ('*_CAMELS_COL_*.xlsx', '*_CAMELS_COL_*.zip', '*.nc',
                   '00_CAMELS-COL*.docx',
                   # and the folders its archives were extracted into
                   '0?_CAMELS_COL_*/')

    # cached tables which are not worth shipping to a process pool worker
    _NOT_PICKLED = ('_static_table', '_daily_index', 'bndry_id_map_')

    def __init__(self,
                 path=None,
                 overwrite: bool = False,
                 to_netcdf: bool = True,
                 verbosity: int = 1,
                 **kwargs):
        """
        Parameters
        ----------
        path : str
            directory under which the data is (or will be) saved in a
            ``CAMELS_COL`` folder. If None, the default data directory of
            aqua_fetch is used.
        overwrite : bool
            if True, the archives, extracted folders, attribute tables and
            netCDF cache of this release are deleted and downloaded again.
        to_netcdf : bool
            whether to save the dynamic data in a netCDF cache for faster
            reading. Requires netCDF4 and xarray.
        verbosity : int
            0 prints nothing.
        **kwargs :
            any keyword argument of :py:class:`aqua_fetch.rr._RainfallRunoff`
            such as ``processes`` or ``remove_zip``, which deletes the archives
            of this release once they are extracted.
        """
        super(CAMELS_COL, self).__init__(path=path, overwrite=overwrite,
                                         to_netcdf=to_netcdf,
                                         verbosity=verbosity, **kwargs)

        self._download_camels_col(overwrite=overwrite)

        self._check_manifest()

        self._maybe_to_netcdf()

        self.bbox = {'llcrnrlat': -5.0, 'urcrnrlat': 15.0, 'llcrnrlon': -80.0, 'urcrnrlon': -65.0}
        self.parallels = range(-5, 15, 5)
        self.meridians = range(-80, -65, 5)

    @property
    def _release_dir(self) -> os.PathLike:
        """folder with the files of this release and with its netCDF cache"""
        return os.path.join(self.path, self._RELEASE_DIR)

    def _release_file(self, name: str) -> os.PathLike:
        """path of one of the :attr:`_ARCHIVES` or :attr:`_TABLES`"""
        return os.path.join(self._release_dir, name)

    def _folder_path(self, name: str) -> os.PathLike:
        """folder into which one of the :attr:`_ARCHIVES` is extracted"""
        return self._release_file(name)[:-len('.zip')]

    def _download_camels_col(self, overwrite: bool = False):
        """
        Downloads the tables which are not on disk and extracts the archives
        whose folders are not on disk, so an archive deleted after extraction
        (``remove_zip=True``) is not downloaded again.

        What is on disk is judged only by the files of *this* release, inside
        :attr:`_release_dir`. The superseded 2025 release wrote its files
        directly into ``path`` under some of the same names, so a ``path``
        which already holds it is never mistaken for a complete download.
        """
        self._warn_superseded_files()

        if overwrite:
            _remove_stale([*(self._folder_path(name) for name in self._ARCHIVES),
                           *(self._release_file(name) for name in self._ARCHIVES),
                           *(self._release_file(name) for name in self._TABLES),
                           *glob.glob(os.path.join(glob.escape(self._release_dir),
                                                   f"{self.name.lower()}_{self.timestep}*.nc"))],
                          self.verbosity)

        missing_folders = [name for name in self._ARCHIVES
                           if not os.path.isdir(self._folder_path(name))]
        missing_tables = [name for name in self._TABLES
                          if not os.path.exists(self._release_file(name))]

        if not missing_folders and not missing_tables:
            if self.verbosity:
                print(f"CAMELS_COL is already available at {self._release_dir}")
            self.maybe_remove_zip_files()
            return

        os.makedirs(self._release_dir, exist_ok=True)

        to_download = missing_tables + [name for name in missing_folders
                                        if not os.path.exists(self._release_file(name))]
        if to_download:
            # imported here, although the module is also imported at the top of
            # this file, so that the name is looked up on the module at call
            # time: this is what lets a test replace it
            from ...download_zenodo import download_from_zenodo
            download_from_zenodo(self._release_dir, doi=self.url, include=to_download,
                                 verbosity=self.verbosity)

        for name in missing_folders:
            _extract_zip(self._release_file(name), self._folder_path(name),
                         self.verbosity)

        self.maybe_remove_zip_files()
        return

    def _warn_superseded_files(self):
        """
        Warns, regardless of ``verbosity``, if ``path`` holds attribute files
        of the superseded 2025 release. They are neither read nor deleted: this
        release is downloaded into its own sub-folder.
        """
        old = sorted({fpath.rstrip(os.sep) for pattern in self._SUPERSEDED
                      for fpath in glob.glob(os.path.join(glob.escape(self.path), pattern))})
        if not old:
            return

        sizes = {fpath: _path_size(fpath) for fpath in old}
        biggest = max(sizes, key=sizes.get)
        warnings.warn(
            f"CAMELS_COL: {self.path} holds {len(old)} files and folders "
            f"({sum(sizes.values()) / 1e6:.0f} MB) of the superseded 2025 release "
            f"(zenodo 15554735), the largest being "
            f"{os.path.basename(biggest)} ({sizes[biggest] / 1e6:.0f} MB). They "
            f"are not read and not deleted: this release is kept in "
            f"{self._release_dir}. The 2025 zenodo record is restricted and no "
            f"longer serves these files, so deleting them cannot be undone.",
            UserWarning)
        return

    def _archive_files(self) -> List[str]:
        """
        The archives of this release whose folder is on disk. The base class
        walks the whole of ``path``, which would let :meth:`free_disk_space`
        delete the archives of the superseded release lying next to the
        release folder.
        """
        return [self._release_file(name) for name in self._ARCHIVES
                if os.path.exists(self._release_file(name))
                and os.path.isdir(self._folder_path(name))]

    def remove_zip_files(self):
        """
        Deletes the archives of this release whose folder is extracted, so that
        a call on a half extracted release cannot delete an archive which would
        then have to be downloaded again. Files of the superseded 2025 release,
        which may lie in ``path`` next to the release folder, are left alone.
        """
        for archive in self._archive_files():
            if self.verbosity:
                print(f"remove_zip=True: removing {archive}")
            os.remove(archive)
        return

    def _check_manifest(self):
        """
        Warns if files of this release are missing, if fewer time series files
        were extracted than the record lists, or if the first of those files
        holds columns this class does not know, instead of silently serving a
        truncated
        dataset. Each check is independent, so one absent file cannot hide
        another problem.
        """
        shapefile = [self.boundary_file[:-len('.shp')] + ext
                     for ext in ('.shp', '.shx', '.dbf', '.prj')]
        files = [*(self._release_file(name) for name in self._TABLES), *shapefile, self.ts_path]

        missing = [fpath for fpath in files if not os.path.exists(fpath)]
        if missing:
            warnings.warn(
                f"CAMELS_COL: {len(missing)} of {len(files)} files and folders of "
                f"the release are missing: {missing}. Use overwrite=True to "
                f"download them again.", UserWarning)

        # the attribute tables are the record's own list of gauges, so the time
        # series files are counted against them instead of being trusted: a
        # directory exists whether it holds 346 files or none
        manifest = self._release_file(self._TABLES[0])
        try:
            listed = set(_read_col_csv(manifest).index)
        except (ValueError, OSError) as err:
            # e.g. an interrupted copy of the data folder between machines
            warnings.warn(
                f"CAMELS_COL: {err} Use overwrite=True to download the release "
                f"again.", UserWarning)
        else:
            on_disk = set(self._stn_ids)
            if listed != on_disk:
                warnings.warn(
                    f"CAMELS_COL: the record lists {len(listed)} gauges but "
                    f"{self.ts_path} holds {len(on_disk)} time series files. Missing: "
                    f"{sorted(listed - on_disk)}. Unexpected: {sorted(on_disk - listed)}. "
                    f"Use overwrite=True to download the release again.", UserWarning)

        if not self._stn_ids:
            return

        first = self._ts_file(self.stations()[0])
        try:
            header = pd.read_csv(first, sep='\t', nrows=0)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as err:
            warnings.warn(
                f"CAMELS_COL: {first} cannot be read ({err}). Use overwrite=True "
                f"to download the release again.", UserWarning)
            return

        # [1:] because the first column is the date, 'Fecha', which becomes the index
        unknown = [col for col in header.columns[1:] if col not in self.dyn_map]
        if unknown:
            warnings.warn(
                f"CAMELS_COL: the time series files hold the columns {unknown}, "
                f"which this class does not know and does not serve. The record "
                f"may have changed.", UserWarning)
        return

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(self._release_dir,
                            '03_CAMELS_COL_Basin_boundary',
                            '03_CAMELS_COL_Basin_boundary',
                            'CAMELS_COL_catchments_boundaries.shp')

    @property
    def boundary_id_map(self) -> str:
        """the only property of the catchment shapefile of this release"""
        return 'IDEAM_CODE'

    @property
    def ts_path(self) -> os.PathLike:
        """folder with the time series file of each gauge"""
        return os.path.join(self._release_dir,
                            '04_CAMELS_COL_Hydrometeorological_data',
                            '3_Hydrometeorological_data')

    def _ts_file(self, stn: str) -> os.PathLike:
        """path of the time series file of one gauge"""
        return os.path.join(self.ts_path, f"Hydromet_data_{stn}.txt")

    @property
    def dyn_fname(self) -> Union[str, os.PathLike]:
        """name of the netCDF cache of the dynamic data, one per release and
        precision, e.g. camels_col_D_2026_v2.nc or
        camels_col_D_2026_float64_v2.nc"""
        precision = "" if np.dtype(self.fp) == np.float32 else f"_{np.dtype(self.fp).name}"
        return cache_name(f"{self.name.lower()}_{self.timestep}_{self._RELEASE}{precision}.nc")

    @property
    def dyn_fpath(self) -> os.PathLike:
        """netCDF cache, kept in the folder of this release so that a cache of
        the superseded release in ``path`` is never read as this one"""
        return os.path.join(self._release_dir, self.dyn_fname)

    @property
    def dyn_map(self) -> Dict[str, str]:
        """
        the columns of the time series files, in the order they are written.
        Precipitation (CHIRPS v2.0) and potential evapotranspiration (MSWX) are
        in mm/day, the temperatures (MSWX) in degC and the observed streamflow
        (IDEAM) in m3/s.
        """
        return {
            'Precipitacion': total_precipitation(),
            'ETP_': total_potential_evapotranspiration(),
            'Temperatura_minima': min_air_temp(),
            'Temperatura_maxima': max_air_temp(),
            'Caudal': observed_streamflow_cms(),
        }

    @property
    def static_map(self) -> Dict[str, str]:
        return {
            'gauge_lat': gauge_latitude(),
            'gauge_lon': gauge_longitude(),
            'gauge_elev': gauge_elevation_meters(),
            'area': catchment_area(),
            'perimeter': catchment_perimeter(),
            'maximum_ele': max_catchment_elevation_meters(),
            'mean_ele': catchment_elevation_meters(),
            'minimum_ele': min_catchment_elevation_meters(),
        }

    def transform_boundary(self, boundary):
        """
        Transforms a catchment boundary from WGS 84 / World Mercator
        (EPSG:3395, the CRS of the shapefile) to WGS84 (EPSG:4326) lon/lat, so
        that it matches the gauge coordinates.

        Uses the pyproj-free :func:`world_mercator_to_wgs84` helper. Verified
        against pyproj (EPSG:3395 -> EPSG:4326) on all catchments: the
        per-vertex error is 1.1e-10 degrees, i.e. ~0.01 mm. MultiPolygons and
        Polygons with interior rings (holes) are handled, and the geometry type
        and the ring structure are kept. The conversion is vectorised per ring.
        """
        if fiona is None:
            return boundary

        def _ring_to_wgs84(ring):
            arr = np.asarray(ring, dtype=float)
            # fiona stores each vertex as (x=easting, y=northing[, z]); output
            # is (lon, lat) to keep the (x, y) ordering of the geometry.
            lat, long = world_mercator_to_wgs84(arr[:, 0], arr[:, 1])
            return list(zip(long.tolist(), lat.tolist()))

        if boundary.type == 'MultiPolygon':
            coords = [[_ring_to_wgs84(ring) for ring in polygon]
                      for polygon in boundary.coordinates]
        else:  # Polygon, possibly with interior rings (holes)
            coords = [_ring_to_wgs84(ring) for ring in boundary.coordinates]

        return fiona.Geometry(type=boundary.type, coordinates=coords)

    @functools.cached_property
    def _stn_ids(self) -> List[str]:
        """one id per time series file, sorted so that the order does not depend
        on the order the file system happens to list them in"""
        pattern = os.path.join(glob.escape(self.ts_path), 'Hydromet_data_*.txt')
        return sorted(os.path.basename(fpath).split('Hydromet_data_')[1].split('.')[0]
                      for fpath in glob.glob(pattern))

    def stations(self) -> List[str]:
        """ids of the gauges of the release, one per time series file"""
        return list(self._stn_ids)

    @property
    def dynamic_features(self) -> List[str]:
        return list(self.dyn_map.values())

    @property
    def static_features(self) -> List[str]:
        return self._static_table.columns.tolist()

    @functools.cached_property
    def _time_extent(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        first and last date over the time series files, read from their first
        and last row so that the extent follows the data instead of a literal
        which drifts with the next release. Only those two rows are read, so a
        file whose last date precedes its first is rejected here; any other day
        falling outside the extent is reported by :meth:`_read_stn_dyn`.
        """
        if not self._stn_ids:
            raise FileNotFoundError(
                f"{self.ts_path} holds no Hydromet_data_*.txt file, so the period "
                f"covered by CAMELS_COL cannot be read. Use overwrite=True to "
                f"download the release again.")

        first, last = [], []
        for stn in self._stn_ids:
            head, tail = self._first_last_rows(stn)
            begins = pd.to_datetime(_row_date(head, b'\t'), format=_COL_DATE_FMT)
            ends = pd.to_datetime(_row_date(tail, b'\t'), format=_COL_DATE_FMT)
            if begins > ends:
                raise ValueError(
                    f"the dates in {self._ts_file(stn)} are not sorted: the file "
                    f"begins on {begins.date()} and ends on {ends.date()}.")
            first.append(begins)
            last.append(ends)
        return min(first), max(last)

    def _first_last_rows(self, stn: str) -> Tuple[bytes, bytes]:
        """
        first and last data row of the time series file of one gauge.

        ``_first_and_last_row`` reads the row right after the header, which is
        blank if the file has an empty line there, and raises ``IndexError`` on
        an empty file. Both are read as "no data" here and, when only the first
        row is blank, the first non-empty one is looked up instead, so a file
        with data is never reported as having none.
        """
        fpath = self._ts_file(stn)
        try:
            head, tail = _first_and_last_row(fpath)
        except IndexError:          # nothing but (at most) a header
            head = tail = b''

        if not head.strip() and tail.strip():
            with open(fpath, 'rb') as f:
                f.readline()        # header
                head = next((row for row in f if row.strip()), b'')

        if not head.strip() or not tail.strip():
            raise ValueError(
                f"{fpath} holds no data row. Use overwrite=True to download the "
                f"release again.")
        return head, tail

    @functools.cached_property
    def _daily_index(self) -> pd.DatetimeIndex:
        """the daily index every gauge is served on. A file holds only the days
        which were observed, so it is padded to this index."""
        return pd.date_range(self.start, self.end, freq='D', name='time')

    @property
    def start(self) -> pd.Timestamp:
        return self._time_extent[0]

    @property
    def end(self) -> pd.Timestamp:
        return self._time_extent[1]

    @property
    def location(self) -> str:
        return "Colombia"

    def _read_stn_dyn(self, stn: str) -> pd.DataFrame:
        """
        reads the time series of one gauge. The record writes its dates as
        dd/mm/yyyy, so the format is given: pandas would otherwise read the
        first date of 37 of the 346 gauges month first and shift the whole
        series by months without saying so.
        """
        df = pd.read_csv(self._ts_file(stn), sep='\t', index_col=0)
        df.index = pd.to_datetime(df.index, format=_COL_DATE_FMT)

        # These two are raised rather than warned about: this method runs in a
        # process pool worker, whose warnings never reach the caller, while an
        # exception does. Serving one row of a duplicated date, or dropping a
        # day, would otherwise be a silent guess. CAMELS_CL rejects the same two
        # problems in :func:`_checked_dates`.
        if df.index.has_duplicates:
            repeated = df.index[df.index.duplicated()].unique()
            raise ValueError(
                f"{self._ts_file(stn)} has {len(repeated)} duplicated dates, e.g. "
                f"{repeated[0].date()}. Serving one of the two rows would be a "
                f"guess, and a warning would be lost because this file is read in "
                f"a worker process, so the read is stopped. Use overwrite=True to "
                f"download the release again; if a fresh copy is the same, the "
                f"release itself is broken.")

        outside = df.index.difference(self._daily_index)
        if len(outside):
            raise ValueError(
                f"{len(outside)} days of {self._ts_file(stn)}, e.g. "
                f"{outside[0].date()}, lie outside {self.start.date()} .. "
                f"{self.end.date()}, which is read from the first and last row of "
                f"every file: this file is not sorted by date. Serving it would "
                f"drop those days, so the read is stopped. Use overwrite=True to "
                f"download the release again.")

        # the file holds only the days which were observed; padding here rather
        # than leaving it to the netCDF cache means every gauge is served on the
        # same index whether or not the cache exists
        df = df.rename(columns=self.dyn_map).reindex(self._daily_index)

        # float32, the precision of this library, is safe for this record: its
        # largest value is a discharge of 17027.22 m3/s and it writes at most 3
        # decimals, which float32 reproduces for all but 6 of the 20364340
        # values, all of them discharges of gauge 31097010 above 16000 m3/s
        # which move by 0.001 m3/s, their 8th significant digit.
        return df.astype(self.fp)

    @functools.cached_property
    def _static_table(self) -> pd.DataFrame:
        """
        the 79 attributes of all 346 gauges, read and cached on first use.
        ``gauge_star`` and ``gauge_end``, the first and last day of the
        streamflow record, are kept as the dd/mm/yyyy strings of the record.
        """
        expected = set(self._stn_ids)
        tables = []
        for name in self._TABLES:
            df = _read_col_csv(self._release_file(name))
            # every table of the record lists every gauge; one that does not
            # would otherwise leave its columns NaN without saying so
            short = sorted(expected.difference(df.index))
            if short:
                warnings.warn(
                    f"CAMELS_COL: {len(short)} gauges are missing from {name}, so "
                    f"its {len(df.columns)} attributes are served as NaN for them: "
                    f"{short}.", UserWarning)
            tables.append(df.rename(columns=self._RENAMED[name]) if name in self._RENAMED else df)

        static = pd.concat(tables, axis=1)

        no_attributes = sorted(set(self._stn_ids).difference(static.index))
        no_series = sorted(set(static.index).difference(self._stn_ids))
        if no_attributes or no_series:
            warnings.warn(
                f"CAMELS_COL: {len(no_attributes)} gauges have a time series file "
                f"but no attributes, which are served as NaN ({no_attributes}), and "
                f"{len(no_series)} gauges have attributes but no time series file, "
                f"which are not served at all ({no_series}).", UserWarning)
        static = static.reindex(self._stn_ids)

        static = static.rename(columns=self.static_map)

        duplicated = sorted(set(static.columns[static.columns.duplicated()]))
        if duplicated:
            warnings.warn(
                f"CAMELS_COL: the attribute names {duplicated} come from more than "
                f"one table of the record, so only the first of each is reachable "
                f"by name.", UserWarning)

        for col, fac in self.static_factors.items():
            if col in static.columns:
                static[col] *= fac

        # the record gives the gauge position in EPSG:3395 meters: the column
        # named gauge_lat holds the northing and gauge_lon the easting
        lat, lon = world_mercator_to_wgs84(
            static[gauge_longitude()].values.astype(float),
            static[gauge_latitude()].values.astype(float))
        static[gauge_latitude()] = lat
        static[gauge_longitude()] = lon

        # the record gives no gauge name, so unlike the shared
        # _warn_duplicate_gauges this compares positions only, and says so
        rounded = static[[gauge_latitude(), gauge_longitude()]].round(3)
        duplicated = rounded.duplicated(keep=False)
        if duplicated.any():
            warnings.warn(
                f"CAMELS_COL: {int(duplicated.sum())} gauges share a position to 3 "
                f"decimals: {sorted(static.index[duplicated])}. The record gives no "
                f"gauge name, so only the coordinates are compared. They are kept "
                f"in the dataset.", UserWarning)
        return static

    def _static_data(self) -> pd.DataFrame:
        return self._static_table.copy()

    def __getstate__(self):
        """
        Drops the cached static table when the dataset is pickled, so that it
        does not travel to every worker of the process pool the base
        :meth:`_read_dynamic` starts. It is rebuilt lazily where it is needed.
        """
        return {name: value for name, value in self.__dict__.items()
                if name not in self._NOT_PICKLED}


_COL_DATE_FMT = '%d/%m/%Y'


def _read_col_csv(fpath: Union[str, os.PathLike]) -> pd.DataFrame:
    """
    Reads one attribute table of CAMELS-COL with the gauge id as index.

    The tables of the record are not written consistently: most are separated
    by ``;`` but the hydrological signatures by ``,``, the catchment
    information is latin-1 while the others are utf-8, two column names of the
    physiographic table end in a space, and the land use capability table
    writes its ids as ``11017010.00`` and is padded with 16037 empty rows. The
    separator and the encoding are therefore taken from the file itself, the
    ids are normalized and the empty rows are dropped. No value is changed.
    """
    with open(fpath, 'rb') as f:
        raw = f.read()  # a few hundred kB at most
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        text = raw.decode('latin-1')

    header = text.partition('\n')[0]
    sep = ';' if header.count(';') > header.count(',') else ','

    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, index_col=0, dtype={0: str})
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as err:
        raise ValueError(
            f"{fpath} cannot be read ({err}). It may be empty, or the release "
            f"may have been copied only in part.") from None
    df = df[df.index.notna()]
    df = df.dropna(how='all')
    if df.empty or df.shape[1] == 0:
        # a wrong separator parses the whole line into the index and leaves no
        # column, which dropna would then turn into an empty table
        raise ValueError(
            f"{fpath} was read as {df.shape} with the separator {sep!r}. The "
            f"record may have changed the way this table is written.")
    # '11017010.00' -> '11017010', and '11017010' is left as it is
    df.index = df.index.str.split('.').str[0]
    if df.index.has_duplicates:
        repeated = sorted(set(df.index[df.index.duplicated()]))
        raise ValueError(
            f"{fpath} lists {len(repeated)} gauges more than once, e.g. "
            f"{repeated[0]}, so their attributes contradict each other. Use "
            f"overwrite=True to download the release again.")
    df.index.name = 'gauge_id'
    df.columns = df.columns.str.strip()  # e.g. 'gravelius_index '
    return df
