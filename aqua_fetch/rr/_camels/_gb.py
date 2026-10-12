import io
import os
import csv
import json
import time
import shutil
import zipfile
import warnings
import functools
import concurrent.futures as cf
from typing import Union, List, Dict, Tuple

import numpy as np
import pandas as pd

from ..utils import _RainfallRunoff, n_workers
from ..._geom_utils import osgb36_to_wgs84
from ...utils import get_cpus
from ...utils import validate_attributes, download
from ..._backend import fiona
from .._map import (
    observed_streamflow_cms,
    observed_streamflow_mm,
    observed_water_level_m,
    mean_air_temp,
    mean_air_temp_with_specifier,
    total_precipitation,
    total_precipitation_with_specifier,
    total_potential_evapotranspiration,
    total_potential_evapotranspiration_with_specifier,
    mean_windspeed,
    solar_radiation,
    downward_longwave_radiation,
    mean_specific_humidity,
)
from .._map import catchment_area, gauge_latitude, gauge_longitude
from ._common import _remove_stale, _warn_duplicate_gauges


class CAMELS_GB(_RainfallRunoff):
    """
    Daily hydro-meteorological time series and catchment attributes of 671
    catchments in Great Britain, in two versions:

    - ``version=2`` (default): CAMELS-GB v2 of
      `Coxon et al., 2026 <https://doi.org/10.5194/essd-18-4345-2026>`_,
      1970-10-01 to 2022-09-30, 10 daily dynamic and 219 static features. 686
      files (~0.8 GB) are downloaded from the
      `EIDC <https://doi.org/10.5285/9a46d428-958f-4ac1-86eb-94eee70c0955>`_.
      It also has hourly data, see ``timestep`` below.
    - ``version=1``: CAMELS-GB of
      `Coxon et al., 2020 <https://doi.org/10.5194/essd-12-2459-2020>`_,
      1970-10-01 to 2015-09-30, 10 dynamic and 145 static features, downloaded
      as one ~0.26 GB zip from the
      `EIDC <https://doi.org/10.5285/8344e4f3-d2ea-44f5-8afa-86d2987543a9>`_.

    Both versions give streamflow as depth and as discharge, plus precipitation,
    potential evapotranspiration (with and without interception) and air
    temperature. Version 2 has two sources for precipitation, evapotranspiration
    and temperature instead of one, and drops the specific humidity, radiation
    and wind speed of version 1. It also re-processed the common period, so its
    values, including some catchment areas, differ slightly from version 1. Use
    :attr:`dynamic_features` and :attr:`static_features` for the current names
    and the paper for what each one means; values keep their published units.

    Each source covers its own period, and the missing steps are NaN: the
    CEH-GEAR and CHESS series end on 2019-12-31, which leaves 5% of the daily
    steps empty, while the HadUK-Grid and Hydro-PE series run to the end.

    With ``timestep='H'`` version 2 also gives hourly precipitation, streamflow
    and river level with their quality flags, 1990-10-01 09:00 to 2022-10-01
    08:00, for the same 671 catchments. The timestamps are UTC, as the dataset's
    own supporting documentation states, and label the hour that **ends** at
    them: the streamflow of 09:00 is the mean of the 08:15, 08:30, 08:45 and
    09:00 readings of the gauge. That is another ~10.6 GB of
    downloads. Its precipitation comes from other sources than the daily one:
    CEH-GEAR1hr, whose last value is on 2016-12-31 (82% of the steps; the
    dataset's own table says 2019, but every file ends in 2016), and GRaD-GB,
    which starts on 2006-01-01 (50%). Streamflow covers 95% of the steps but is
    missing altogether for 7 catchments (their ``hourly_flow_perc_complete``
    attribute is 0), and river level covers 81% and is missing altogether for
    101 catchments. The two flags are the UK-Flow15 three-digit quality codes,
    served as numbers: pad them back to three digits (``f'{flag:03.0f}'``)
    before decoding. A flag is NaN where there is no flagged observation, and a
    few stations have no flags at all. The groundwater level data of version 2
    is not downloaded.

    That hourly streamflow is the 15-minute record of :class:`UKFlow15`, hourly
    averaged. 664 of these 671 catchments are among UK-Flow15's 1369 gauges
    (the 7 that are not are exactly those without hourly flow here) with the
    same gauge name and grid reference, and over 2.28 million hourly values of
    11 shared gauges the two agree to the 3 decimals they are published with;
    the hourly flag is the highest of the hour's four 15-minute codes. UK-Flow15
    keeps the 15-minute values, covers twice as many gauges and runs to
    2023-12-31, but has no meteorology, catchment attributes or boundaries.

    The first initialization of the daily version 2 took ~3 minutes: downloading
    plus building the 1 GB netCDF cache (1.8 GB of disk in total). Fetching all
    671 stations then took 0.9 s from that cache, or 1.6 s from the csv files
    (11 s with ``processes=1``); a single station took 0.13 s from the cache. The
    hourly data is served from the csv files (no cache by default): downloading
    it took ~6 minutes and fetching all 671 stations at once takes ~70 s and
    ~12 GB of memory as DataFrames (~23 GB as an xarray Dataset), so fetch fewer
    stations at a time if memory is tight.

    Examples
    --------
    >>> from aqua_fetch import CAMELS_GB
    >>> dataset = CAMELS_GB()  # version 2
    >>> len(dataset.stations())
    671
    >>> len(dataset.dynamic_features), len(dataset.static_features)
    (10, 219)
    ... # dynamic features of one station as a dictionary of DataFrames
    >>> _, dynamic = dataset.fetch(stations='38017', as_dataframe=True)
    >>> dynamic['38017'].shape
    (18993, 10)
    ... # selected features for one year
    >>> _, dynamic = dataset.fetch('38017', dynamic_features=['pcp_mm_haduk', 'q_mm_obs'],
    ...                            st='2020-01-01', en='2020-12-31', as_dataframe=True)
    >>> dynamic['38017'].shape
    (366, 2)
    ... # 10% of the stations, chosen randomly
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)
    67
    ... # static and dynamic features together
    >>> static, dynamic = dataset.fetch(stations='38017', static_features="all", as_dataframe=True)
    >>> static.shape
    (1, 219)
    ... # the hourly data of version 2 (downloads another ~10.6 GB)
    >>> hourly = CAMELS_GB(timestep='H')
    >>> _, dynamic = hourly.fetch(stations='38017', as_dataframe=True)
    >>> dynamic['38017'].shape
    (280512, 7)
    ... # without as_dataframe=True, an xarray Dataset is returned
    >>> _, dynamic = dataset.fetch(stations=['38017', '42001'])
    >>> dict(dynamic.sizes)
    {'time': 18993, 'dynamic_features': 10}
    >>> dataset.stn_coords('38017')
                    lat  long
    gauge_id
    38017     51.880001 -0.28
    >>> dataset.area('38017')
    gauge_id
    38017    38.400002
    Name: area_km2, dtype: float32
    >>> dataset.get_boundary('38017').type  # needs fiona
    'Polygon'
    ... # version 1
    >>> dataset = CAMELS_GB(version=1)
    >>> _, dynamic = dataset.fetch(stations='38017', as_dataframe=True)
    >>> dynamic['38017'].shape
    (16436, 10)
    """
    time_steps = ['D', 'H']

    # EIDC record ids of version 1 and version 2
    _v1_id = "8344e4f3-d2ea-44f5-8afa-86d2987543a9"
    _v2_id = "9a46d428-958f-4ac1-86eb-94eee70c0955"

    def __init__(
            self,
            path: str = None,
            version: int = 2,
            timestep: str = 'D',
            overwrite: bool = False,
            to_netcdf: bool = None,
            verbosity: int = 1,
            **kwargs
    ):
        """
        Parameters
        ----------
        path : str
            folder in which the ``CAMELS_GB`` folder is (or will be) created. If
            None, the default data folder of aqua_fetch is used.
        version : int
            2 (default) for CAMELS-GB v2 or 1 for the original CAMELS-GB. Each
            version is kept in its own sub-folder, so both can share ``path``.
        timestep : str
            ``D`` (default) for the daily data or ``H`` for the hourly data of
            version 2. Each timestep is downloaded and cached separately.
        overwrite : bool
            if True, the files of this version and timestep, including the
            netCDF cache, are deleted and downloaded again.
        to_netcdf : bool
            whether to save all dynamic data in one netCDF file for faster
            reading. Needs the xarray and netCDF4 packages. Defaults to True for
            the daily and False for the hourly data, which is too large for one
            file and fast enough to read from the csv files.
        verbosity : int
            0 prints nothing.
        **kwargs :
            passed to :py:class:`aqua_fetch.rr._RainfallRunoff`, e.g. ``processes``
        """
        if isinstance(version, bool) or version not in (1, 2):
            raise ValueError(f"version must be 1 or 2, not {version!r}")
        if timestep not in self.time_steps:
            raise ValueError(f"timestep must be one of {self.time_steps}, not {timestep!r}")
        if timestep == 'H' and version == 1:
            raise ValueError("hourly data is only in version 2 of CAMELS-GB")
        self.version = version

        if to_netcdf is None:
            # the hourly data (10.5 GB) is too large for one netCDF file, and
            # reading it from the csv files is fast enough
            to_netcdf = timestep == 'D'
        elif to_netcdf and timestep == 'H':
            warnings.warn("caching the hourly data of CAMELS_GB in one netCDF file needs "
                          "~23 GB of memory and ~10.5 GB of disk; pass to_netcdf=False to "
                          "read the csv files instead")

        super().__init__(path=path, timestep=timestep, overwrite=overwrite,
                         to_netcdf=to_netcdf, verbosity=verbosity, **kwargs)

        # filled on first use
        self._static_df = None
        self._ts_fnames = None
        self._dyn_feats = None
        self._period_ = None

        self._download(overwrite)

        self._warn_duplicate_gauges()

        self._maybe_to_netcdf()

    @property
    def _version_dir(self) -> str:
        """folder with all files of the selected version, and its netCDF cache"""
        return os.path.join(self.path, 'camels_gb' if self.version == 1 else 'camels_gb_v2')

    def _download(self, overwrite: bool = False):
        """downloads the selected version and timestep. With ``overwrite``, the
        files that are downloaded again, and the netCDF cache of this timestep,
        are deleted first. Files of the other timestep are left alone."""
        if overwrite:
            if self.version == 1:
                stale = [self._version_dir, os.path.join(self.path, 'camels_gb.zip')]
            else:
                # the folders that are downloaded again, so that the files of
                # the other timestep survive
                stale = [os.path.join(self._version_dir, *folder.rstrip('/').split('/'))
                         for folder in self._v2_folders]
                stale += [self._v2_docs_dir, f"{self._v2_docs_dir}.zip", self.dyn_fpath]
            _remove_stale(stale, self.verbosity)

        if self.version == 1:
            self._download_v1()
        else:
            self._download_v2()
        return

    @staticmethod
    def _download_zip(url: str, zip_path: str, verbosity: int = 0):
        """downloads a zip unless a readable copy is already at ``zip_path``"""
        if os.path.exists(zip_path) and not zipfile.is_zipfile(zip_path):
            os.remove(zip_path)
        if not os.path.exists(zip_path):
            os.makedirs(os.path.dirname(zip_path), exist_ok=True)
            download(url=url, outdir=os.path.dirname(zip_path),
                     fname=os.path.basename(zip_path), verbosity=verbosity)
        return

    def _download_v1(self):
        """
        Downloads and extracts the version 1 zip unless it is already extracted;
        a readable zip on disk is extracted instead of downloaded again. The
        extracted folders get their final names only when extraction has
        finished, so an interrupted extraction is repeated at the next
        initialization.
        """
        if os.path.exists(self.data_path):
            if self.verbosity:
                print(f"CAMELS_GB version 1 is already available at {self._version_dir}")
        else:
            if os.path.exists(self._version_dir):
                if self.verbosity:
                    print(f"removing {self._version_dir}, left by an interrupted extraction")
                shutil.rmtree(self._version_dir)

            zip_path = os.path.join(self.path, "camels_gb.zip")
            self._download_zip(f"https://data-package.ceh.ac.uk/data/{self._v1_id}.zip",
                               zip_path, self.verbosity)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(self._version_dir)
            # the zip holds one folder named after the EIDC record id
            os.rename(os.path.join(self._version_dir, self._v1_id),
                      os.path.join(self._version_dir, 'camels_gb'))
            if self.remove_zip:
                os.remove(zip_path)

        # the boundaries are in a zip inside the data folder
        boundary_dir = os.path.dirname(self.boundary_file)
        if not os.path.exists(boundary_dir):
            part = f"{boundary_dir}_part"
            if os.path.exists(part):
                shutil.rmtree(part)
            with zipfile.ZipFile(f"{boundary_dir}.zip") as zf:
                zf.extractall(part)
            os.rename(part, boundary_dir)
        return

    def _download_v2(self):
        """
        Downloads those files of version 2 which are not on disk or whose size
        differs from the record's own manifest. So a download that was
        interrupted is completed on the next initialization.
        """
        manifest = self._v2_manifest()
        root = self._version_dir

        sizes = {}
        for folder in {os.path.dirname(rel) for rel in manifest}:
            if os.path.isdir(os.path.join(root, folder)):
                with os.scandir(os.path.join(root, folder)) as entries:
                    sizes.update({f"{folder}/{e.name}": e.stat().st_size for e in entries})
        missing = [rel for rel, nbytes in manifest.items() if sizes.get(rel) != nbytes]

        if not missing:
            if self.verbosity:
                print(f"CAMELS_GB version 2 is already available at {root}")
            return

        incomplete = [rel for rel in missing if rel in sizes]
        if incomplete:
            warnings.warn(f"{len(incomplete)} files of CAMELS_GB version 2 are incomplete and are "
                          f"downloaded again, e.g. {incomplete[0]}")
            for rel in incomplete:
                # otherwise download() would save the new file under another name
                os.remove(os.path.join(root, rel))

        if self.verbosity:
            size = sum(manifest[rel] for rel in missing) / 1e6
            print(f"downloading {len(missing)} files ({size:.0f} MB) of CAMELS_GB version 2 to {root}")

        for folder in {os.path.dirname(rel) for rel in missing}:
            os.makedirs(os.path.join(root, folder), exist_ok=True)

        def _download_file(rel: str, attempts: int = 3):
            outdir, fname = os.path.join(root, os.path.dirname(rel)), os.path.basename(rel)
            for attempt in range(1, attempts + 1):
                try:
                    return download(url=f"https://catalogue.ceh.ac.uk/datastore/eidchub/{self._v2_id}/{rel}",
                                    outdir=outdir, fname=fname, verbosity=0)
                except Exception:
                    # the server can fail a request now and then when many are sent
                    if os.path.exists(os.path.join(outdir, fname)):
                        os.remove(os.path.join(outdir, fname))
                    if attempt == attempts:
                        raise
                    time.sleep(1.5 * attempt)

        # the server, not the CPU, limits the speed, so threads are used; more
        # than 8 connections would only burden the server
        with cf.ThreadPoolExecutor(min(self.processes or 8, 8)) as executor:
            for i, _ in enumerate(executor.map(_download_file, missing), start=1):
                if self.verbosity and i % 100 == 0:
                    print(f"downloaded {i} of {len(missing)} files")

        wrong = [rel for rel in missing if os.path.getsize(os.path.join(root, rel)) != manifest[rel]]
        if wrong:
            raise RuntimeError(
                f"{len(wrong)} downloaded files of CAMELS_GB version 2, e.g. {wrong[0]}, do not "
                f"have the size given in the manifest. Initialize the class again to download "
                f"them once more, or use overwrite=True if the record itself was updated.")
        return

    def _v2_manifest(self) -> Dict[str, int]:
        """
        ``{path relative to the version folder: size in bytes}`` of the version 2
        files used by this class. It is read from ``ro-crate-metadata.json``,
        the file list which EIDC ships with the supporting documents of the
        record (a 0.2 MB zip).
        """
        docs_dir = self._v2_docs_dir
        fpath = os.path.join(docs_dir, 'ro-crate-metadata.json')

        def _read():
            with open(fpath, encoding='utf-8') as fp:
                return json.load(fp)['@graph']

        try:
            graph = _read()
        except (OSError, ValueError):  # not extracted yet, or cut short
            zip_path = f"{self._v2_docs_dir}.zip"
            self._download_zip(f"https://data-package.ceh.ac.uk/sd/{self._v2_id}.zip", zip_path)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(docs_dir)
            if self.remove_zip:
                os.remove(zip_path)
            graph = _read()

        manifest = {}
        for item in graph:
            # data files are listed as data/<folder>/<file name>
            if item.get('@type') != 'File' or not item['@id'].startswith('data/'):
                continue
            rel = item['@id'][len('data/'):]
            if rel.startswith(self._v2_folders) and 'groundwaterwell' not in rel:
                manifest[rel] = item['bytes']
        return manifest

    @property
    def _v2_folders(self) -> tuple:
        """folders of the version 2 record that are downloaded. Of the two
        hydro-meteorological folders only the one of the selected timestep is
        taken; the groundwater time series and well attributes are left out."""
        return ('Catchment_Attributes/', 'Catchment_Boundaries/',
                f"{self._v2_ts_folder}/")

    @property
    def _v2_ts_folder(self) -> str:
        return ('Catchment_Timeseries/hydro-meteorological/'
                + ('daily' if self.timestep == 'D' else 'hourly'))

    @property
    def _v2_docs_dir(self) -> str:
        """folder with the record's own file list and documentation"""
        return os.path.join(self._version_dir, 'supporting_documents')

    def _warn_duplicate_gauges(self):
        """warns if two gauges have the same name and coordinates; both are kept"""
        meta = pd.read_csv(
            self._attr_fpath('topographic'),
            usecols=['gauge_id', 'gauge_name', 'gauge_lat', 'gauge_lon'],
            dtype={'gauge_id': str})
        _warn_duplicate_gauges(self.name, meta)
        return

    @property
    def data_path(self) -> str:
        """folder containing the attribute, time series and boundary files"""
        if self.version == 1:
            return os.path.join(self._version_dir, 'camels_gb', 'data')
        return self._version_dir

    @property
    def ts_dir(self) -> str:
        """folder with one time series csv file per station"""
        if self.version == 1:
            return os.path.join(self.data_path, 'timeseries')
        return os.path.join(self.data_path, *self._v2_ts_folder.split('/'))

    def _attr_fpath(self, category: str) -> str:
        """path of the attribute file of a category e.g. ``topographic``"""
        if self.version == 1:
            return os.path.join(self.data_path, f"CAMELS_GB_{category}_attributes.csv")
        return os.path.join(self.data_path, 'Catchment_Attributes',
                            f"camels_gb_v2_{category}_attributes.csv")

    @property
    def boundary_file(self) -> os.PathLike:
        if self.version == 1:
            return os.path.join(self.data_path, "CAMELS_GB_catchment_boundaries",
                                "CAMELS_GB_catchment_boundaries.shp")
        return os.path.join(self.data_path, "Catchment_Boundaries",
                            "camels_gb_v2_catchment_boundaries.shp")

    @property
    def boundary_id_map(self) -> str:
        return 'ID_STRING'

    def transform_boundary(self, boundary):
        """
        Transforms a catchment boundary from the British National Grid
        (OSGB36 / EPSG:27700, the CRS of both shapefiles) to WGS84 (EPSG:4326)
        lon/lat, so that it matches the gauge coordinates.

        Uses the pyproj-free :func:`osgb36_to_wgs84` helper. Verified against
        pyproj (EPSG:27700 -> EPSG:4326) on both versions: the per-vertex error
        is below 4 mm. Polygons with interior rings (holes) and MultiPolygons
        are handled, and the geometry type and ring structure are kept. The
        conversion is vectorised per ring.
        """
        if fiona is None:
            return boundary

        def _ring_to_wgs84(ring):
            arr = np.asarray(ring, dtype=float)
            # fiona stores each vertex as (x=easting, y=northing[, z]); output
            # is (lon, lat) to keep the (x, y) ordering of the geometry.
            lat, long = osgb36_to_wgs84(arr[:, 0], arr[:, 1])
            return list(zip(long.tolist(), lat.tolist()))

        if boundary.type == 'MultiPolygon':
            coords = [[_ring_to_wgs84(ring) for ring in polygon]
                      for polygon in boundary.coordinates]
        else:  # Polygon, possibly with interior rings (holes)
            coords = [_ring_to_wgs84(ring) for ring in boundary.coordinates]

        return fiona.Geometry(type=boundary.type, coordinates=coords)

    @property
    def dyn_fpath(self) -> os.PathLike:
        """netCDF cache, kept in the folder of the selected version"""
        return os.path.join(self._version_dir, self.dyn_fname)

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'area': catchment_area(),
                # slope_fdc (the slope of the flow duration curve between its
                # log-transformed 33rd and 66th streamflow percentiles) is
                # deliberately not mapped onto slope: it is a streamflow
                # signature, not a terrain slope. The terrain slope of this
                # dataset is dpsbar, catchment mean drainage path slope in m/km.
                'gauge_lat': gauge_latitude(),
                'gauge_lon': gauge_longitude(),
        }

    @property
    def dyn_map(self) -> Dict[str, str]:
        """maps the column names of the time series files to standard names.
        No unit is converted."""
        if self.version == 1:
            # Table 1 of the supporting documentation of version 1
            return {
                'precipitation': total_precipitation(),  # mm day-1
                'pet': total_potential_evapotranspiration(),  # mm day-1
                'temperature': mean_air_temp(),  # degC
                'discharge_spec': observed_streamflow_mm(),  # mm day-1
                'discharge_vol': observed_streamflow_cms(),  # m3 s-1
                'peti': total_potential_evapotranspiration_with_specifier('intercep'),  # mm day-1
                'humidity': mean_specific_humidity(),  # g kg-1
                'shortwave_rad': solar_radiation(),  # W m-2
                'longwave_rad': downward_longwave_radiation(),  # W m-2
                'windspeed': mean_windspeed(),  # m s-1
            }
        if self.timestep == 'H':
            # Table 3 of the supporting information of version 2. The two
            # quality flags keep their names, they are codes, not measurements.
            # UKFlow15 reads the same three-digit codes as text to keep their
            # leading zeros; here they stay numbers, because a dynamic feature
            # ends up in a numeric (time, feature) array.
            return {
                'precipitation_cehgear': total_precipitation_with_specifier('cehgear'),  # mm hour-1
                'precipitation_gradgb': total_precipitation_with_specifier('gradgb'),  # mm hour-1
                'discharge_spec': observed_streamflow_mm(),  # mm hour-1
                'discharge_vol': observed_streamflow_cms(),  # m3 s-1
                'level': observed_water_level_m(),  # m above the river bed
            }
        # Table 2 of the supporting information of version 2
        return {
            'precipitation_cehgear': total_precipitation_with_specifier('cehgear'),  # mm day-1
            'precipitation_haduk': total_precipitation_with_specifier('haduk'),  # mm day-1
            'pet_chess': total_potential_evapotranspiration_with_specifier('chess'),  # mm day-1
            'peti_chess': total_potential_evapotranspiration_with_specifier('intercep_chess'),  # mm day-1
            'pet_hydrope': total_potential_evapotranspiration_with_specifier('hydrope'),  # mm day-1
            'peti_hydrope': total_potential_evapotranspiration_with_specifier('intercep_hydrope'),  # mm day-1
            'temperature_chess': mean_air_temp_with_specifier('chess'),  # degC
            'temperature_haduk': mean_air_temp_with_specifier('haduk'),  # degC
            'discharge_spec': observed_streamflow_mm(),  # mm day-1
            'discharge_vol': observed_streamflow_cms(),  # m3 s-1
        }

    @property
    def static_attribute_categories(self) -> List[str]:
        return ['climatic', 'humaninfluence', 'hydrogeology', 'hydrologic',
                'hydrometry', 'landcover', 'soil', 'topographic']

    def _ts_files(self) -> Dict[str, str]:
        """``{station: file name}`` of the time series files. A file name ends
        with ``_<station>_<first day>-<last day>.csv``."""
        if self._ts_fnames is None:
            self._ts_fnames = {f.split('_')[-2]: f for f in os.listdir(self.ts_dir)
                               if f.endswith('.csv')}
        return self._ts_fnames

    def _period(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        First and last timestamp of the time series files, read once from their
        first and last line. The file names give only the dates, while the
        hourly files start at 09:00 and end at 08:00 (UTC).
        """
        if self._period_ is None:
            firsts, lasts = [], []
            for fname in self._ts_files().values():
                with open(os.path.join(self.ts_dir, fname), 'rb') as fp:
                    fp.readline()  # the header
                    first = fp.readline().split(b',')[0]
                    fp.seek(max(0, os.fstat(fp.fileno()).st_size - 4096))
                    # the first line of the tail is the header or a part of a
                    # line, so it is dropped, as are trailing blank lines
                    tail = [line for line in fp.read().splitlines()[1:] if line]
                    if not tail:  # a row longer than the tail: read it all
                        fp.seek(0)
                        tail = [line for line in fp.read().splitlines()[1:] if line]
                    if not first or not tail:
                        raise ValueError(f"{fname} of CAMELS_GB has no data rows")
                    firsts.append(first)
                    lasts.append(tail[-1].split(b',')[0])
            self._period_ = (pd.Timestamp(min(firsts).decode()),
                             pd.Timestamp(max(lasts).decode()))
        return self._period_

    @property
    def start(self) -> pd.Timestamp:
        return self._period()[0]

    @property
    def end(self) -> pd.Timestamp:
        return self._period()[1]

    def stations(self) -> List[str]:
        """ids of the gauges, read from the names of the time series files"""
        return sorted(self._ts_files())

    @property
    def dynamic_features(self) -> List[str]:
        if self._dyn_feats is None:
            fname = next(iter(self._ts_files().values()))
            columns = pd.read_csv(os.path.join(self.ts_dir, fname), index_col='date', nrows=0).columns
            self._dyn_feats = [self.dyn_map.get(col, col) for col in columns]
        return list(self._dyn_feats)

    @property
    def _mm_feature_name(self) -> str:
        return observed_streamflow_mm()

    @property
    def _area_name(self) -> str:
        return catchment_area()

    @property
    def _coords_name(self) -> List[str]:
        return [gauge_latitude(), gauge_longitude()]

    def _read_stn_dyn(self, stn: str) -> pd.DataFrame:
        return _read_camels_gb_stn(os.path.join(self.ts_dir, self._ts_files()[stn]), self.dyn_map)

    def _read_dynamic(
            self,
            stations,
            dynamic_features,
            st: Union[str, pd.Timestamp] = None,
            en: Union[str, pd.Timestamp] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Reads the time series files of several stations, with a process pool
        when the files are large enough to repay starting it (see
        :func:`n_workers`).
        """
        st, en = self._check_length(st, en)
        features = validate_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')
        stations = validate_attributes(stations, self.stations(), 'stations')

        files = self._ts_files()
        fpaths = [os.path.join(self.ts_dir, files[stn]) for stn in stations]
        reader = functools.partial(_read_camels_gb_stn, rename=self.dyn_map,
                                   features=features, st=st, en=en)

        start = time.time()
        # all files have about the same size; more than 16 processes was slower
        nbytes = len(fpaths) * os.path.getsize(fpaths[0]) if fpaths else 0
        cpus = n_workers(nbytes, len(fpaths), self.processes or min(get_cpus(), 16))
        if cpus > 1:
            with cf.ProcessPoolExecutor(cpus) as executor:
                frames = list(executor.map(reader, fpaths, chunksize=4))
        else:
            frames = [reader(fpath) for fpath in fpaths]

        if self.verbosity:
            print(f"Read {len(frames)} stations for {len(features)} dyn features "
                  f"in {time.time() - start:.2f} seconds with {cpus} cpus.")
        return dict(zip(stations, frames))

    def _static_data(self) -> pd.DataFrame:
        """all static features with gauge ids as index. Read once; a copy is returned."""
        if self._static_df is None:
            dfs = [_read_camels_gb_attributes(self._attr_fpath(category))
                   for category in self.static_attribute_categories]
            self._static_df = pd.concat(dfs, axis=1).rename(columns=self.static_map)
        return self._static_df.copy()


def _read_camels_gb_attributes(fpath: str) -> pd.DataFrame:
    """
    Reads one CAMELS-GB attribute csv with ``gauge_id`` (as str) as index.

    In the version 2 hydrometry file, the rows of gauges 27038 and 42010 have an
    unquoted comma inside the free-text last column
    ``station_quality_hourlyflow_comment``, so they have one field too many. For
    these two rows the extra field is joined back into that column, which
    restores the published text. Any other row with too many fields raises an
    error.
    """
    with open(fpath, newline='', encoding='utf-8') as fp:
        rows = list(csv.reader(fp))
    header = rows[0]
    n = len(header)

    for i, row in enumerate(rows):
        if len(row) <= n:
            continue
        if not (header[-1] == 'station_quality_hourlyflow_comment'
                and row[0] in ('27038', '42010') and len(row) == n + 1):
            raise ValueError(f"line {i + 1} of {fpath} has {len(row)} fields instead of {n}")
        rows[i] = row[:n - 1] + [','.join(row[n - 1:])]

    # parsed by pandas from text, so that values and dtypes are the same as for
    # a file without such rows
    text = io.StringIO()
    csv.writer(text).writerows(rows)
    text.seek(0)
    return pd.read_csv(text, index_col='gauge_id', dtype={'gauge_id': str})


def _read_camels_gb_stn(
        fpath: str,
        rename: Dict[str, str],
        features: List[str] = None,
        st: pd.Timestamp = None,
        en: pd.Timestamp = None,
) -> pd.DataFrame:
    """
    Reads the daily or hourly time series csv of one CAMELS-GB station and
    renames its columns. If ``features`` is given, only these features from
    ``st`` to ``en`` are returned. It is a module-level function so that a
    process pool pickles only these small arguments and not the dataset.
    """
    df = pd.read_csv(fpath, index_col='date', parse_dates=True)
    df.rename(columns=rename, inplace=True)
    if features is not None:
        df = df.loc[st:en, features]
    df.index.name = 'time'
    df.columns.name = 'dynamic_features'
    return df
