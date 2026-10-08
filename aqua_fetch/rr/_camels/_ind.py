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

from ..utils import _RainfallRunoff, n_workers, ymd_index
from ...utils import validate_attributes
from .._map import (
    observed_streamflow_cms,
    mean_air_temp,
    max_air_temp,
    min_air_temp,
    total_precipitation,
    total_potential_evapotranspiration,
    total_potential_evapotranspiration_with_specifier,
    actual_evapotranspiration_with_specifier,
    mean_rel_hum,
    mean_windspeed,
    u_component_of_wind,
    v_component_of_wind,
    solar_radiation,
    downward_longwave_radiation,
)
from .._map import catchment_area, gauge_latitude, gauge_longitude, slope
from ._common import _remove_stale, _warn_duplicate_gauges


class CAMELS_IND(_RainfallRunoff):
    """
    Daily hydrometeorological time series and static catchment attributes for
    472 catchments in Peninsular India following
    `Mangukiya et al., 2025 <https://doi.org/10.5194/essd-17-461-2025>`_
    (CAMELS-IND). The data is downloaded from its
    `zenodo repository <https://zenodo.org/records/14999580>`_.

    The dataset has 20 dynamic features from 1980-01-01 to 2020-12-31 (14976
    daily steps) and 210 static features. The meteorological series are gap
    free, except ``pet_mm`` which the source files leave empty for all of 1980
    (2.4 % of that feature). Observed streamflow is available at 313 of the 472
    gauges and covers more than 30 % of the period at 242 of them; all other
    days are ``NaN``. Catchment boundaries are shapefiles in WGS84.

    Two releases are available through ``version``:

    - ``version='2.2'`` (default): the March 2025 release, which downloads
      350 MB into ``CAMELS_IND/CAMELS_IND_All_Catchments/`` (881 MB extracted,
      plus a 568 MB netCDF cache).
    - ``version='2'``: the August 2024 release
      (`zenodo <https://zenodo.org/records/13221214>`_) that earlier releases of
      this class read, extracted directly into ``CAMELS_IND/``. Both releases
      can share one ``path`` and each keeps its own netCDF cache. The authors
      have since restricted the Zenodo records of every release before 2.2, so
      release 2 can only be read where its files already are; asking for it
      anywhere else raises. ``overwrite=True`` rebuilds it from the archives on
      disk and never deletes one it cannot fetch again.

    Both releases have the same 472 gauges, features and period. Release 2.2
    corrects release 2 in three ways, so prefer it unless you are reproducing
    older work:

    - 55 gauges of basins 12 and 15 (ids 12001-12042 and 15001-15013) carry the
      name, river, coordinates, areas and gauge elevation of the *previous*
      gauge of their basin in release 2, so :meth:`area`, :meth:`stn_coords`
      and :meth:`q_mm` do not
      describe the catchment whose boundary, forcings and streamflow are served
      under the same id;
    - release 2 labels the forcings ``evap_canopy`` and ``evap_surface``
      kg m-2 s-1 although the values, which are the same in both releases, are
      mm day-1;
    - the streamflow observations were revised: 181 gauges have a different
      record (+1.9 % observations in total) and the hydrological signatures were
      recomputed from them.

    The static feature ``dspbar`` of release 2 is named ``dpsbar`` in release
    2.2.

    Not provided: the LSTM simulated streamflow which both releases ship is
    model output, so it is not extracted from the archive. Of the features which
    are served, only streamflow (India-WRIS) and precipitation, ``airtemp_C_max``
    and ``airtemp_C_min`` (IMD gridded station observations) are measurements
    (``airtemp_C_mean`` is the average of the two); radiation,
    wind, humidity, evaporation and soil moisture are IMDAA reanalysis, the
    evapotranspirations hPET and GLEAM, and the soil attributes HiHydroSoil,
    HWSD and Pelletier et al. See Tables 1 and 3 of the data description for
    each feature's source.

    The first initialization took 80 seconds: downloading and extracting the
    350 MB archive, reading all 472 gauges from the source files and writing the
    568 MB netCDF cache. Afterwards initialization takes 0.008 s and the first
    fetch in a process reads the whole dataset from that cache in 0.8 s as
    DataFrames or 0.3 s as an :obj:`xarray.Dataset` (0.3 s for one gauge; later
    fetches in the same process are faster). Reading all 472 gauges from the
    source files instead takes 1.6 s on 32 worker processes and 8.7 s with
    ``processes=1``, measured on a 48 cpu machine.

    Examples
    ---------
    >>> from aqua_fetch import CAMELS_IND
    >>> dataset = CAMELS_IND()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='3001', as_dataframe=True)
    >>> df = dynamic['3001'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (14976, 20)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       472
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (47 out of 472)
       47
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(14976, 20), (14976, 20), (14976, 20),... (14976, 20), (14976, 20)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('3001', as_dataframe=True,
    ...  dynamic_features=['pcp_mm', 'rh_%', 'airtemp_C_mean', 'pet_mm', 'q_cms_obs'])
    >>> dynamic['3001'].shape
       (14976, 5)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10

    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='3001', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['3001'].shape
    ((1, 210), 1, (14976, 20))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 14976, 'dynamic_features': 20})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (472, 2)
    >>> dataset.stn_coords('3001')  # returns coordinates of station whose id is 3001
                    lat       long
    gauge_id
    3001      18.386101  80.391701
    >>> dataset.stn_coords(['3001', '17021'])  # returns coordinates of two stations
    ...
    # get area (km2) of a single station
    >>> dataset.area('3001')
    gauge_id
    3001    1537.0
    Name: area_km2, dtype: float32
    # get areas of two stations
    >>> dataset.area(['3001', '17021'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('3001')
    ...
    # the August 2024 release
    >>> dataset = CAMELS_IND(version='2')
    """

    # Zenodo record of each release, oldest first
    urls = {
        '2': "https://zenodo.org/records/13221214",
        '2.2': "https://zenodo.org/records/14999580",
    }

    # the single archive of release 2.2 is extracted into a folder of its name
    _V22_DIR = 'CAMELS_IND_All_Catchments'

    # archives of each release, named after the folder they are extracted into.
    # The 2.2 record also has CAMELS_IND_Catchments_Streamflow_Sufficient.zip,
    # a 178 MB copy of the gauges with more than 30 % streamflow (242 of them,
    # counted in its own file list; the data description still says 228, the
    # count of the 2.1 subset), so it is not downloaded.
    _ARCHIVES = {
        '2': ('attributes_csv', 'attributes_txt', 'catchment_mean_forcings',
              'shapefiles_catchment', 'streamflow_timeseries'),
        '2.2': (_V22_DIR,),
    }

    # data description of release 2, downloaded along with its archives. The
    # archive of release 2.2 already holds its own copy, byte for byte the same
    # file as the record's, so that one is not downloaded twice.
    _DOC_FILE = {'2': "00_camels_India_data_description.pdf"}

    # the attribute files of both releases, in the order of the data description
    _ATTR_FILES = ('name', 'topo', 'clim', 'hydro', 'land', 'soil', 'geol', 'anth')

    # simulated streamflow, named this way in release 2 and 2.2 respectively
    _MODEL_OUTPUT = ('LSTM_pred_streamflow.csv', 'lstm_pred_streamflow.csv')

    # cached attributes which are not worth shipping to a process pool worker
    _NOT_PICKLED = ('_static_df', 'bndry_id_map_')

    def __init__(self,
                 path=None,
                 version: str = '2.2',
                 overwrite: bool = False,
                 to_netcdf: bool = True,
                 verbosity: int = 1,
                 **kwargs):
        """
        Parameters
        ----------
        path : str
            directory under which the data is (or will be) saved in a
            ``CAMELS_IND`` folder. Both releases can share it. If None, the
            default data directory of aqua_fetch is used.
        version : str
            ``'2.2'`` (default) or ``'2'``, see the class docstring.
        overwrite : bool
            if True, the archives, extracted files and netCDF cache of this
            ``version`` are deleted and downloaded again.
        to_netcdf : bool
            whether to save the dynamic data of this ``version`` in a netCDF
            cache for faster reading. Requires netCDF4 and xarray.
        verbosity : int
            0 prints nothing.
        **kwargs :
            any keyword argument of :py:class:`aqua_fetch.rr._RainfallRunoff`
            such as ``processes`` or ``remove_zip``, which deletes the archives
            of this ``version`` once they are extracted.
        """
        version = str(version)
        if version not in self.urls:
            raise ValueError(
                f"version must be one of {list(self.urls)} but is {version!r}")
        self.version = version

        super(CAMELS_IND, self).__init__(path=path, overwrite=overwrite,
                                         to_netcdf=to_netcdf,
                                         verbosity=verbosity, **kwargs)

        # lazy caches, see the properties of the same name
        self._static_features = None
        self._dynamic_features = None

        self._download_camels_ind(overwrite=overwrite)

        self._check_manifest()

        self._warn_known_errors()

        _warn_duplicate_gauges(f"{self.name} {self.version}", self._gauge_meta())

        self._maybe_to_netcdf()

    @property
    def url(self) -> str:
        """zenodo record of the selected release"""
        return self.urls[self.version]

    def __getstate__(self):
        """
        Drops the large cached tables when the dataset is pickled, so that the
        static table and the 472 catchment boundaries do not travel with it.
        :meth:`_read_dynamic` hands a module level function to its process pool
        and no longer pickles the dataset at all, so this only guards the paths
        that still could, e.g. the base class pool or a user's own. Both tables
        are rebuilt lazily where they are needed.
        """
        return {name: value for name, value in self.__dict__.items()
                if name not in self._NOT_PICKLED}

    @property
    def _latest_version(self) -> str:
        """the newest release this class knows, by release number rather than
        by the order :attr:`urls` happens to be written in"""
        return max(self.urls, key=lambda v: tuple(int(part) for part in v.split('.')))

    @property
    def _is_latest(self) -> bool:
        """
        Whether this is the newest release the class knows. The authors have
        restricted the Zenodo record of every release they superseded so far
        (1, 2 and 2.1 list no file at all), so the archives of an older release
        may be the only copy there is. Asking the class's own release list
        rather than a second list of restricted records keeps this answer
        offline and true after the next release is added.
        """
        return self.version == self._latest_version

    @property
    def _version_dir(self) -> os.PathLike:
        """
        folder with the files of the selected release and with its netCDF
        cache. Release 2 ships one archive per folder, which is extracted
        directly into :attr:`path`, while release 2.2 ships a single archive
        which is extracted into a folder of its own name.
        """
        if self.version == '2':
            return self.path
        return os.path.join(self.path, self._V22_DIR)

    @property
    def _attr_prefix(self) -> str:
        """the attribute files are named camels_India_* in release 2 and
        camels_ind_* in release 2.2"""
        return 'camels_India_' if self.version == '2' else 'camels_ind_'

    def _attr_file(self, name: str) -> os.PathLike:
        """path of one of the :attr:`_ATTR_FILES` attribute files"""
        return os.path.join(self.static_path, f"{self._attr_prefix}{name}.txt")

    @property
    def static_path(self) -> os.PathLike:
        """folder with the attribute files"""
        return os.path.join(self._version_dir, "attributes_txt")

    @property
    def q_path(self) -> os.PathLike:
        """folder with the streamflow file"""
        return os.path.join(self._version_dir, "streamflow_timeseries")

    @property
    def _q_file(self) -> os.PathLike:
        """csv with the observed streamflow (m3 s-1) of all gauges"""
        return os.path.join(self.q_path, "streamflow_observed.csv")

    @property
    def forcings_path(self) -> os.PathLike:
        """folder with the catchment mean forcing files"""
        return os.path.join(self._version_dir, "catchment_mean_forcings")

    @property
    def boundary_file(self) -> os.PathLike:
        # the folder is Merged in release 2 and merged in release 2.2
        merged = "Merged" if self.version == '2' else "merged"
        return os.path.join(self._version_dir, "shapefiles_catchment",
                            merged, "all_catchments.shp")

    @property
    def boundary_id_map(self) -> str:
        """the only property the boundary shapefile of both releases has"""
        return "gauge_id"

    def _boundary_catch_id(self, value) -> str:
        """the shapefile spells the id as in the file names, ``'03001'``, while
        the class drops the leading zeros, ``'3001'``"""
        return str(int(value))

    @property
    def dyn_fpath(self) -> os.PathLike:
        """netCDF cache, kept in the folder of the selected release"""
        return os.path.join(self._version_dir, self.dyn_fname)

    def stn_forcing_path(self, stn: str) -> os.PathLike:
        """path of the forcing file of one station. Release 2 groups the files
        into one folder per basin code, release 2.2 keeps them in one folder."""
        gauge_id = self.id_map[stn]
        if self.version == '2':
            return os.path.join(self.forcings_path, gauge_id[0:2], f"{gauge_id}.csv")
        return os.path.join(self.forcings_path, f"{gauge_id}.csv")

    def _download_camels_ind(self, overwrite: bool = False):
        """
        Downloads and extracts the archives of ``self.version`` whose extracted
        folders are not on disk, so an archive deleted after extraction
        (``remove_zip=True``) is not downloaded again. ``overwrite=True`` deletes
        this release's archives, extracted folders and netCDF cache first, but
        only once Zenodo has confirmed that it can serve them again. An archive
        which is on disk and which Zenodo no longer serves is kept and its
        folder extracted from it again, so that a release which cannot be
        downloaded can still be repaired, as long as every one of its archives
        is on disk.
        """
        folders = {stem: os.path.join(self.path, stem)
                   for stem in self._ARCHIVES[self.version]}
        archives = {stem: self._archive_path(stem) for stem in folders}
        doc = self._DOC_FILE.get(self.version)

        if overwrite:
            _remove_stale(self._stale_for_overwrite(archives), self.verbosity)

        # with overwrite every folder is extracted again, otherwise only the
        # ones which are not on disk
        missing = [stem for stem, folder in folders.items()
                   if overwrite or not os.path.isdir(folder)]

        if not missing:
            if self.verbosity:
                print(f"CAMELS_IND {self.version} is already available "
                      f"at {self._version_dir}")
            self.maybe_remove_zip_files()
            return

        os.makedirs(self.path, exist_ok=True)

        to_download = [f"{stem}.zip" for stem in missing
                       if not os.path.exists(archives[stem])]

        if to_download:
            self._check_downloadable(to_download)
            # the data description is fetched along with the data, never on its
            # own and never blocking: the class does not read it, so a missing
            # one must not stop an extraction that needs no download at all
            if (doc is not None and doc in self._record_files
                    and not os.path.exists(os.path.join(self.path, doc))):
                to_download.append(doc)
            # imported here because that module installs a SIGINT handler on import
            from ...download_zenodo import download_from_zenodo
            download_from_zenodo(self.path, doi=self.url, include=to_download,
                                 verbosity=self.verbosity)

        for stem in missing:
            self._extract(archives[stem], folders[stem])

        self.maybe_remove_zip_files()
        return

    def _stale_for_overwrite(self, archives: Dict[str, str]) -> List[str]:
        """
        What ``overwrite=True`` deletes before downloading this release again:
        its netCDF caches, including one written by an older ``CACHE_VERSION``,
        and the archives which Zenodo can serve again, with any corrupt copy of
        them. It raises before returning if the record cannot serve one that is
        not on disk.

        An archive which is on disk and which the record does not offer is the
        only copy there is, so it is kept and its folder extracted from it
        again. The extracted folders are not listed here either: each is
        replaced by :meth:`_extract` once its own archive has been read, so one
        unreadable archive cannot take the whole release with it.
        """
        kept = [stem for stem in archives
                if os.path.exists(archives[stem])
                and os.path.basename(archives[stem]) not in self._record_files]

        self._check_downloadable([os.path.basename(archives[stem])
                                  for stem in archives if stem not in kept])

        if kept:
            warnings.warn(
                f"CAMELS_IND: the archives {sorted(kept)} of release "
                f"{self.version} are kept although overwrite is True: the "
                f"Zenodo record of this release no longer serves them. Their "
                f"folders are extracted from them again.", UserWarning)

        stale = glob.glob(os.path.join(glob.escape(self._version_dir),
                                       f"{self.name.lower()}_{self.timestep}*.nc"))
        for stem in archives:
            if stem not in kept:
                stale += [archives[stem], f"{archives[stem]}.corrupt"]
        return stale

    def _archive_path(self, stem: str) -> os.PathLike:
        """path of the archive which is extracted into the folder ``stem``"""
        return os.path.join(self.path, f"{stem}.zip")

    @functools.cached_property
    def _record_files(self) -> List[str]:
        """names of the files the Zenodo record of this release offers, asked
        of Zenodo once per instance"""
        import requests   # a minimal requirement of this library

        record = self.url.rstrip('/').rsplit('/', 1)[-1]
        response = requests.get(f"https://zenodo.org/api/records/{record}", timeout=30)
        response.raise_for_status()
        return [f['key'] for f in response.json().get('files', [])]

    def _check_downloadable(self, files: List[str]):
        """
        Raises, before anything is deleted or downloaded, if the Zenodo record
        of this release does not offer ``files``. The authors restricted the
        records of the releases before 2.2, which now list no file at all, so
        release 2 can only be read where its files already are: deleting them
        first and asking Zenodo afterwards would destroy the only copy.
        """
        missing = [fname for fname in files if fname not in self._record_files]
        if missing:
            raise FileNotFoundError(
                f"CAMELS_IND release {self.version} cannot be downloaded: its "
                f"Zenodo record ({self.url}) does not offer {missing}. The "
                f"authors have restricted the records of the releases they "
                f"superseded, so release {self.version} can only be used where "
                f"its files already are ({self.path}). Nothing was deleted. Use "
                f"version='{self._latest_version}' to download the dataset.")
        return

    def remove_zip_files(self):
        """deletes the archives of this release once they are extracted. The
        other release's archives, which lie in the same folder, are left alone,
        and so are those of a release which is not the newest one."""
        archives = [self._archive_path(stem) for stem in self._ARCHIVES[self.version]
                    if os.path.exists(self._archive_path(stem))]

        if not self._is_latest:
            if archives:
                warnings.warn(
                    f"CAMELS_IND: the {len(archives)} archives of release "
                    f"{self.version} are kept although remove_zip is True: it is "
                    f"not the newest release ({self._latest_version}), and the "
                    f"authors have so far restricted the record of every release "
                    f"they superseded, so these archives may not be downloadable "
                    f"again.", UserWarning)
            return

        for archive in archives:
            if self.verbosity:
                print(f"remove_zip=True: removing {archive}")
            os.remove(archive)
        return

    def _extract(self, archive: str, folder: str):
        """
        Extracts ``archive``, except the simulated streamflow, into a temporary
        folder which is renamed to ``folder`` once complete. An interrupted
        extraction is therefore redone on the next initialization instead of
        being taken as complete.
        """
        tmp = f"{folder}_extracting"
        shutil.rmtree(tmp, ignore_errors=True)  # left by an interrupted extraction
        # extractall does not create the folder when every member is filtered
        # out, and os.replace then has nothing to rename
        os.makedirs(tmp)

        if self.verbosity:
            print(f"extracting {archive} to {folder}")

        try:
            with zipfile.ZipFile(archive) as zf:
                names = zf.namelist()
                members = [name for name in names
                           if os.path.basename(name) not in self._MODEL_OUTPUT]
                zf.extractall(tmp, members=members)
        except (zipfile.BadZipFile, zlib.error, EOFError):  # e.g. an error page saved as the archive
            shutil.rmtree(tmp, ignore_errors=True)
            if not self._is_latest:
                # left where it is: there is no copy to put in its place, and
                # moving it would make the release look as if an archive were
                # missing, which is what stops it from being read at all
                raise ValueError(
                    f"{archive} is not a readable zip file, and Zenodo may no "
                    f"longer serve release {self.version}. Replace it with a "
                    f"good copy and initialize CAMELS_IND again.") from None
            # moved aside rather than deleted, so that the next initialization
            # downloads it again and the bytes are still there to look at
            broken = f"{archive}.corrupt"
            os.replace(archive, broken)
            raise ValueError(f"{archive} is not a readable zip file and was "
                             f"moved to {broken}. Initialize CAMELS_IND again "
                             f"to fetch it again.") from None

        # the old folder is swapped out by two renames rather than deleted in
        # place: a delete which fails half way through would leave the release
        # without its data and the replacement waiting beside it unused
        previous = f"{folder}_previous"
        shutil.rmtree(previous, ignore_errors=True)
        if os.path.isdir(folder):
            os.replace(folder, previous)
        os.replace(tmp, folder)
        shutil.rmtree(previous, ignore_errors=True)  # best effort, it is a copy now

        # the archive read fine, so a copy set aside by an earlier attempt is
        # of no use to anybody
        broken = f"{archive}.corrupt"
        if os.path.exists(broken):
            if self.verbosity:
                print(f"removing {broken}, replaced by a readable archive")
            os.remove(broken)

        if len(members) < len(names):
            warnings.warn(
                f"CAMELS_IND {self.version}: the LSTM simulated streamflow is "
                f"model output and was not extracted from {archive}.", UserWarning)
        return

    def _check_manifest(self):
        """
        Warns if an attribute, streamflow, forcing or boundary file of this
        release is missing, e.g. because an extraction was interrupted. The
        expected forcing files come from the gauge ids of the release, not from
        whatever is on disk.
        """
        # without these two the class cannot even name its gauges
        required = [self._attr_file(name) for name in ('name', 'topo')]
        gone = [fpath for fpath in required if not os.path.exists(fpath)]
        if gone:
            raise FileNotFoundError(
                f"{gone} not found. Re-initialize CAMELS_IND with overwrite=True.")

        # fiona needs the .dbf/.shx/.prj siblings of the boundary .shp as well
        boundary = os.path.splitext(self.boundary_file)[0]
        expected = [self._attr_file(name) for name in self._ATTR_FILES]
        expected += [self._q_file]
        expected += [f"{boundary}{ext}" for ext in ('.shp', '.dbf', '.shx', '.prj')]
        missing = [fpath for fpath in expected
                   if fpath not in required and not os.path.exists(fpath)]

        # one listing per folder instead of one stat per station
        listed = {}
        for stn in self.stations():
            fpath = self.stn_forcing_path(stn)
            folder = os.path.dirname(fpath)
            if folder not in listed:
                listed[folder] = set(os.listdir(folder)) if os.path.isdir(folder) else set()
            if os.path.basename(fpath) not in listed[folder]:
                missing.append(fpath)

        if missing:
            warnings.warn(
                f"CAMELS_IND {self.version}: {len(missing)} files are missing, "
                f"e.g. {missing[:3]}. The data is incomplete; re-initialize "
                f"with overwrite=True.", UserWarning)

        # an extraction that was interrupted, or whose last cleanup failed,
        # leaves a folder which nothing else looks at and which can be as large
        # as the release itself
        leftovers = [fpath for suffix in ('_extracting', '_previous')
                     for fpath in glob.glob(os.path.join(glob.escape(self.path),
                                                         f"*{suffix}"))]
        if leftovers:
            warnings.warn(
                f"CAMELS_IND: {leftovers} are left over from an interrupted "
                f"extraction. The data does not need them and they can be "
                f"deleted.", UserWarning)
        return

    def _warn_known_errors(self):
        """warns (regardless of ``verbosity``) about the errors of release 2
        which release 2.2 corrects"""
        if self.version != '2':
            return
        warnings.warn(
            f"CAMELS_IND release 2 gives 55 gauges of basins 12 and 15 (ids "
            f"12001-12042 and 15001-15013) the name, coordinates, areas and "
            f"gauge elevation of the previous gauge of the basin, so area(), "
            f"stn_coords() and q_mm() do not describe the catchment whose "
            f"boundary, forcings and streamflow are served under the same id. It "
            f"also labels evap_canopy and evap_surface kg m-2 s-1 although the "
            f"values are mm day-1. Both are corrected in release "
            f"{self._latest_version}, the default.", UserWarning)
        return

    def _gauge_meta(self) -> pd.DataFrame:
        """gauge id, name and coordinates of every gauge, for the duplicate check"""
        names = pd.read_csv(self._attr_file('name'), sep=";",
                            usecols=['gauge_id', 'cwc_site_name'],
                            dtype={'gauge_id': str})
        topo = pd.read_csv(self._attr_file('topo'), sep=";",
                           usecols=['gauge_id', 'cwc_lat', 'cwc_lon'],
                           dtype={'gauge_id': str})
        meta = names.merge(topo, on='gauge_id')
        return meta.rename(columns={'cwc_site_name': 'gauge_name',
                                    'cwc_lat': 'gauge_lat',
                                    'cwc_lon': 'gauge_lon'})

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'cwc_area': catchment_area(),      # km2
                # % (Table 3 of the data description); slope_max, which the
                # same table calls a slope too, is above 90 at 308 of the 472
                # gauges and reaches 385, so these columns cannot be degrees
                'slope_mean': slope('%'),
                'cwc_lat': gauge_latitude(),       # deg N (WGS84)
                'cwc_lon': gauge_longitude(),      # deg E (WGS84)
        }

    @property
    def dyn_map(self) -> Dict[str, str]:
        # Table A1 of the data description. evap_canopy and evap_surface are
        # not renamed: they are labelled kg m-2 s-1 in release 2 and mm day-1
        # in release 2.2 although the values are the same, so each release
        # serves them under the name its own files use.
        return {
            # 'streamflow_cms': 'obs_q_cms',
            'tmin(C)': min_air_temp(),
            'tmax(C)': max_air_temp(),
            'tavg(C)': mean_air_temp(),
            'prcp(mm/day)': total_precipitation(),
            'rel_hum(%)': mean_rel_hum(),
            'wind(m/s)': mean_windspeed(),
            'wind_u(m/s)': u_component_of_wind(),
            'wind_v(m/s)': v_component_of_wind(),
            # surface downward short-wave radiation flux
            'srad_sw(w/m2)': solar_radiation(),
            # surface downward long-wave radiation flux
            'srad_lw(w/m2)': downward_longwave_radiation(),
            #'sm_lvl2(kg/m2)',   # soil moisture of layer 1 (0-0.1 m below ground)
            #'sm_lvl2(kg/m2)',
            #'sm_lvl3(kg/m2)',
            #'sm_lvl4(kg/m2)': ,
            'pet_gleam(mm/day)': total_potential_evapotranspiration_with_specifier('gleam'),
            'pet(mm/day)': total_potential_evapotranspiration(),
            'aet_gleam(mm/day)': actual_evapotranspiration_with_specifier('gleam'),
            #'evap_canopy(kg/m2/s)': evaporation
        }

    @property
    def dynamic_features(self) -> List[str]:
        """returns names of dynamic features"""
        if self._dynamic_features is None:
            self._dynamic_features = self._read_stn_dyn(self.stations()[0]).columns.to_list()
        return list(self._dynamic_features)

    @property
    def static_features(self) -> List[str]:
        """returns names of static features"""
        if self._static_features is None:
            self._static_features = self._static_data().columns.to_list()
        return list(self._static_features)

    @functools.cached_property
    def _extent(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """first and last day of the streamflow file, which holds every gauge
        of the release on one time axis. Taken from the file rather than
        written down, so that a release with a longer record is not truncated.
        The forcing files share this axis, which ``test_start_end_follow_the
        _files`` checks."""
        dates = pd.read_csv(self._q_file, usecols=['year', 'month', 'day'])
        index = ymd_index(dates['year'], dates['month'], dates['day'])
        return index.min(), index.max()

    @property
    def start(self) -> pd.Timestamp:  # start of data
        return self._extent[0]

    @property
    def end(self) -> pd.Timestamp:  # end of data
        return self._extent[1]

    @functools.cached_property
    def id_map(self) -> Dict[str, str]:
        """maps the station id (``'3001'``) to the gauge id the files of the
        dataset use (``'03001'``)"""
        gauge_ids = pd.read_csv(self._attr_file('name'), sep=";",
                                usecols=['gauge_id'], dtype={'gauge_id': str})
        return {str(int(gauge_id)): gauge_id for gauge_id in gauge_ids['gauge_id']}

    def stations(self) -> List[str]:
        """
        returns names of stations as a list

        **Note:** 0s are omitted from the start of the station names
        which means 03001 is returned as 3001
        """
        return list(self.id_map)

    def _static_data(self) -> pd.DataFrame:
        """
        combination of topographic + soil + landuse + geology + climate + hydro
        + climate + anthropogenic features

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of static features of all catchments of
            shape (472, 210)
        """
        return self._static_df.copy()

    @functools.cached_property
    def _static_df(self) -> pd.DataFrame:
        """the attribute files of this release, concatenated and renamed, read
        once and then kept in memory"""
        dfs = []
        for name in self._ATTR_FILES:
            df = pd.read_csv(self._attr_file(name), sep=";", index_col=0)
            df.index = df.index.astype(str)
            dfs.append(df)

        return pd.concat(dfs, axis=1).rename(columns=self.static_map)

    def _read_q(self, stations: Union[str, List[str]] = None) -> pd.DataFrame:
        """reads observed streamflow (m3 s-1) of one, several or, when
        ``stations`` is None, of all gauges. They are all in one file."""
        if stations is None:
            usecols = None
        else:
            if isinstance(stations, str):
                stations = [stations]
            usecols = ['year', 'month', 'day'] + list(stations)

        df = pd.read_csv(self._q_file, usecols=usecols)
        df.index = ymd_index(df.pop('year'), df.pop('month'), df.pop('day'))

        return df.astype(np.float32)

    def _read_forcings(self, stn: str) -> pd.DataFrame:
        """reads the forcing data for a given station"""
        return _read_camels_ind_forcings(self.stn_forcing_path(stn))

    def _read_stn_dyn(self, stn: str) -> pd.DataFrame:
        """reads dynamic data for a given station"""
        return self._assemble(self._read_forcings(stn), self._read_q(stn), stn)

    def _assemble(self, forcings: pd.DataFrame, q: pd.DataFrame,
                  stn: str) -> pd.DataFrame:
        """puts the forcings and the streamflow column of one gauge together
        and gives them their standardized names"""
        q = q[stn].rename(observed_streamflow_cms())
        return pd.concat([forcings, q], axis=1).rename(columns=self.dyn_map)

    def _read_dynamic(
            self,
            stations,
            dynamic_features,
            st: Union[str, pd.Timestamp] = None,
            en: Union[str, pd.Timestamp] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Reads the dynamic data of ``stations`` from the source files.

        The base class reads one station at a time, which reads the 19 MB
        streamflow file of all 472 gauges once per station (0.085 s each, 40 s
        for all of them). Here it is read once for every gauge asked for
        (0.23 s) and only the forcing files, one per gauge, are shared out.
        """
        st, en = self._check_length(st, en)
        dyn_feats = validate_attributes(dynamic_features, self.dynamic_features,
                                        'dynamic_features')
        stations = validate_attributes(stations, self.stations(), 'stations')

        start = time.time()
        q = self._read_q(stations)

        paths = [self.stn_forcing_path(stn) for stn in stations]
        cpus = n_workers(sum(os.path.getsize(fpath) for fpath in paths),
                         len(paths), self.processes)

        def assembled(forcings):
            """the frames of ``stations``, sliced and named, as they arrive:
            keeping all 472 forcing frames until the end costs 0.6 GB"""
            for stn, stn_forcings in zip(stations, forcings):
                stn_df = self._assemble(stn_forcings, q, stn)
                stn_df.index.name = 'time'
                stn_df.columns.name = 'dynamic_features'
                yield stn, stn_df.loc[st:en, dyn_feats]

        if cpus == 1:
            dyn = dict(assembled(map(_read_camels_ind_forcings, paths)))
        else:
            # a module level function, so that a worker does not have to pickle
            # the dataset for every station
            with cf.ProcessPoolExecutor(cpus) as executor:
                dyn = dict(assembled(executor.map(_read_camels_ind_forcings, paths)))

        if self.verbosity:
            print(f"Read {len(dyn)} stations for {len(dyn_feats)} dyn features "
                  f"in {time.time() - start:.2f} seconds with {cpus} cpus.")

        return dyn


def _read_camels_ind_forcings(fpath: str) -> pd.DataFrame:
    """
    Reads one catchment mean forcing file of CAMELS_IND. A module level
    function, so that the process pool of :meth:`CAMELS_IND._read_dynamic`
    pickles a path instead of the dataset.
    """
    df = pd.read_csv(fpath)
    df.index = ymd_index(df.pop('year'), df.pop('month'), df.pop('day'))
    return df.astype(np.float32)
