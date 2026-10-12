import os
import glob
import zlib
import shutil
import zipfile
import warnings
from typing import Union, List, Dict

import numpy as np
import pandas as pd

from ..utils import _RainfallRunoff, cache_name
from ..._geom_utils import tmerc_to_wgs84
from ..._backend import fiona
from .._map import (
    observed_streamflow_cms,
    observed_streamflow_mm,
    mean_air_temp,
    max_air_temp,
    min_air_temp,
    total_precipitation,
    total_potential_evapotranspiration,
    mean_rel_hum,
    solar_radiation,
    KJ_M2_DAY_TO_WM2,
    snow_water_equivalent_with_specifier,
    snow_depth,
)
from .._map import (
    catchment_area,
    gauge_latitude,
    gauge_longitude,
    slope,
    gauge_elevation_meters,
    urban_fraction_with_specifier,
    grass_fraction_with_specifier,
    crop_fraction_with_specifier,
    aridity_index,
    med_catchment_elevation_meters,
    soil_depth,
    population_density,
)
from ._common import _remove_stale


class CAMELS_FI(_RainfallRunoff):
    """
    Daily data of 320 Finnish catchments from 1961-01-01 to 2023-12-31 with 16
    dynamic and 112 static features. Release 1.2.0 is downloaded as one 382 MB
    zip from `Zenodo <https://zenodo.org/records/20225368>`_.

    The dynamic features are catchment averages: observed streamflow in m3/s and
    in mm/day, precipitation, four potential evapotranspiration series, snow
    evaporation, two snow water equivalents, snow depth, four air temperatures,
    relative humidity and global radiation. Each one is NaN outside its own
    record. Values keep their published units except snow depth, published in cm
    and served in m, and global radiation, published as a kJ m-2 day-1 sum and
    served in W m-2. ``pet_mm`` is the dataset's own blend: ERA5 snow evaporation
    when snow depth exceeds 1 cm, otherwise ``pet_fmi`` from April to September
    and ``pet_singer`` from October to March.

    The 112 static features are the eight attribute files of the release (meta,
    topographic, climatic, hydrologic, land cover, soil, geology and human
    influence); 10 of them are text. The ``*_frac_*`` land-cover features are
    converted from the published percent to a fraction, the remaining land-cover
    attributes keep their published percent. Gauge coordinates are published
    in WGS84 and served unchanged, while the catchment boundaries are a
    shapefile in EPSG:3067 (ETRS-TM35FIN, metres) which :meth:`get_boundary`
    reprojects to WGS84 degrees; pass ``to_wgs84=False`` for the published
    metres. Not read by this class: the annual land-cover series, the
    by-attribute copies of the time series, the daily discharge quality flags
    and remarks, and the artificial cross-catchment bifurcations.

    Release 1.0.1, which this class read before, is no longer offered: its gauge
    coordinates disagree with its own map projection by up to 0.4 degrees
    (~45 km) for every one of the 320 gauges, and its potential
    evapotranspiration was withdrawn by the authors as too high. A copy of it in
    ``path``, and the netCDF caches built from it, are deleted and release 1.2.0
    is downloaded once in their place.

    Timings on a 48-core machine: the first initialization downloads 382 MB,
    extracts it and builds the 0.9 GB netCDF cache in 82 s (2.4 GB of disk in
    total, 2.8 GB with ``remove_zip=False``). Afterwards all 320 stations with
    all features are fetched in 0.6 s from the cache and in 2 s from the csv
    files (8.8 s with ``processes=1``), and one station in 0.1 s from either.

    Examples
    ---------
    >>> from aqua_fetch import CAMELS_FI
    >>> dataset = CAMELS_FI()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='1156', as_dataframe=True)
    >>> df = dynamic['1156'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (23010, 16)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       320
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (32)
       32
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(23010, 16), (23010, 16), (23010, 16),... (23010, 16), (23010, 16)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('1156', as_dataframe=True,
    ...  dynamic_features=['pcp_mm', 'snowdepth_m', 'airtemp_C_mean', 'pet_mm', 'q_cms_obs'])
    >>> dynamic['1156'].shape
       (23010, 5)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='1156', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['1156'].shape
    ((1, 112), 1, (23010, 16))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)
    xarray.core.dataset.Dataset
    ...
    >>> dict(dynamic.sizes)
    {'time': 23010, 'dynamic_features': 16}
    ...
    >>> len(dynamic.data_vars)   # -> 10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (320, 2)
    >>> dataset.stn_coords('1156')  # returns coordinates of station whose id is 1156
                    lat       long
    gauge_id
    1156      62.425537  24.741474
    >>> dataset.stn_coords(['1156', '1116'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('1156')
    gauge_id
    1156    147.949997
    Name: area_km2, dtype: float32
    # get area of two stations
    >>> dataset.area(['1156', '1116'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> boundary = dataset.get_boundary('1156')
    >>> boundary.type
    'MultiPolygon'
    >>> boundary.coordinates[0][0][0]   # (long, lat) in degrees
    (24.740851184206278, 62.42737010267462)
    """

    url = "https://zenodo.org/records/20225368"

    # release of the dataset that this class reads. It is part of the netCDF
    # cache name so that a cache built from release 1.0.1 (camels_fi_D.nc,
    # camels_fi_D_v2.nc), whose feature names and values differ, is never served
    # as this release.
    release = "1.2.0"

    _archive_name = "CAMELS-FI.zip"

    # only release 1.2.0 has the geology attributes, so their file tells the two
    # releases apart without reading anything
    _marker = "CAMELS_FI_geology_attributes.csv"

    _attr_files = (
        "CAMELS_FI_meta_attributes.csv",
        "CAMELS_FI_topographic_attributes.csv",
        "CAMELS_FI_climatic_attributes.csv",
        "CAMELS_FI_hydrologic_attributes.csv",
        "CAMELS_FI_landcover_attributes.csv",
        "CAMELS_FI_soil_attributes.csv",
        "CAMELS_FI_geology_attributes.csv",
        "CAMELS_FI_humaninfluence_attributes.csv",
    )

    def __init__(self,
                 path=None,
                 overwrite=False,
                 to_netcdf: bool = True,
                 **kwargs):
        """
        Parameters
        ----------
        path : str
            directory under which the data is (or will be) saved in a
            ``CAMELS_FI`` folder. If None, the default data directory of
            aqua_fetch is used.
        overwrite : bool
            if True, the archive, the extracted files and the netCDF cache are
            deleted and downloaded again.
        to_netcdf : bool
            whether to save the dynamic data in a netCDF cache for faster
            reading. Requires netCDF4 and xarray.
        **kwargs :
            any keyword argument of :py:class:`aqua_fetch.rr._RainfallRunoff`
            such as ``processes``, ``verbosity`` or ``remove_zip``.
        """

        super(CAMELS_FI, self).__init__(
            path=path,
            overwrite=overwrite,
            to_netcdf=to_netcdf,
            **kwargs)

        self._download_camels_fi(overwrite=overwrite)

        self._unzip_boundaries()

        self._check_manifest()

        self._maybe_to_netcdf()

    @property
    def _root(self) -> Union[str, os.PathLike]:
        """folder that :attr:`_archive_name` is extracted into"""
        return os.path.join(self.path, "CAMELS-FI")

    @property
    def data_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self._root, "data")

    @property
    def dyn_fname(self) -> Union[str, os.PathLike]:
        """netCDF cache of this release, e.g. ``camels_fi_D_1.2.0_v2.nc``"""
        return cache_name(f"{self.name.lower()}_{self.timestep}_{self.release}.nc")

    @property
    def boundary_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.data_path, "CAMELS_FI_catchment_boundaries")

    @property
    def boundary_file(self) -> Union[str, os.PathLike]:
        return os.path.join(
            self.boundary_path,
            "CAMELS_FI_catchment_boundaries.shp"
        )

    def transform_boundary(self, boundary):
        """
        Transforms a catchment boundary from ETRS89 / TM35FIN (EPSG:3067,
        meters, as stated in the shapefile's .prj) to WGS84 (lat/lon). Both
        ``Polygon`` and ``MultiPolygon`` geometries occur in this dataset and
        both are supported. The transformation uses the dependency-free
        :func:`aqua_fetch._geom_utils.tmerc_to_wgs84` helper, which reproduces
        ``pyproj`` on all 11 million vertices of this dataset to within 4 cm,
        far below the 10 m vertex spacing of the shapefile itself.
        """
        # EPSG:3067 parameters (from CAMELS_FI_catchment_boundaries.prj)
        lon_0, k0 = 27.0, 0.9996
        false_easting, false_northing = 500000.0, 0.0

        def _convert(coords):
            # a ring is a sequence of coordinate pairs; convert it in one go
            if len(coords) and isinstance(coords[0][0], (int, float)):
                ring = np.asarray(coords, dtype='float64')
                lats, longs = tmerc_to_wgs84(ring[:, 0], ring[:, 1], lon_0, k0,
                                             false_easting, false_northing)
                # fiona stores coordinates as (long, lat)
                return list(zip(longs.tolist(), lats.tolist()))
            return [_convert(c) for c in coords]

        new_coords = _convert(boundary['coordinates'])

        if fiona is not None:
            boundary = fiona.Geometry(type=boundary['type'], coordinates=new_coords)

        return boundary

    @property
    def ts_path(self) -> Union[str, os.PathLike]:
        return os.path.join(
            self.data_path,
            "timeseries",
        )

    def stations(self) -> List[str]:
        return [
            fname.split('.')[0].split('_')[4] for fname in os.listdir(self.ts_path) if fname.endswith('.csv')
            ]

    @property
    def dyn_map(self) -> Dict[str, str]:
        """
        dynamic features map for CAMELS-FI catchments
        """
        return {
            'discharge_vol': observed_streamflow_cms(),
            'discharge_spec': observed_streamflow_mm(),
            'precipitation': total_precipitation(),
            'pet': total_potential_evapotranspiration(),
            'temperature_min': min_air_temp(),
            'temperature_mean': mean_air_temp(),
            'temperature_max': max_air_temp(),
            'humidity_rel': mean_rel_hum(),
            'snow_depth': snow_depth(),  # cm, converted to m in dyn_factors
            'swe': snow_water_equivalent_with_specifier('era5'),
            'swe_cci3-1': snow_water_equivalent_with_specifier('cci3-1'),
            # "catchment daily averaged global radiation sum, kJ m-2" (support
            # document). Global radiation is downward shortwave; converted to
            # W m-2 in dyn_factors. Sanity check: 8383 kJ m-2 day-1 -> 97 W m-2,
            # clearness index 0.395-0.400 at lat 62-63, i.e. exactly Finland's.
            'radiation_global': solar_radiation(),
        }

    @property
    def dyn_factors(self) -> Dict[str, float]:
        return {
            # "catchment daily averaged snow depth at 6:00 UTC, cm" (support
            # document); the canonical name promises m. Max over all catchments
            # and days is 184.4 cm, i.e. 1.844 m.
            snow_depth(): 0.01,
            solar_radiation(): KJ_M2_DAY_TO_WM2,
        }

    @property
    def static_map(self) -> Dict[str, str]:
        return {
            'gauge_lat': gauge_latitude(),
            'gauge_lon': gauge_longitude(),
            'area': catchment_area(),   
            'slope': slope('percent'),
            #'slope_fdc': catchment_elevation_meters(),
            'aridity': aridity_index(),
            'grass_perc_2000': grass_fraction_with_specifier('2000'),
            'urban_perc_2000': urban_fraction_with_specifier('2000'),
            'crop_perc_2000': crop_fraction_with_specifier('2000'),
            'grass_perc_2006': grass_fraction_with_specifier('2006'),
            'urban_perc_2006': urban_fraction_with_specifier('2006'),
            'crop_perc_2006': crop_fraction_with_specifier('2006'),
            'grass_perc_2012': grass_fraction_with_specifier('2012'),
            'urban_perc_2012': urban_fraction_with_specifier('2012'),
            'crop_perc_2012': crop_fraction_with_specifier('2012'),
            'grass_perc_2018': grass_fraction_with_specifier('2018'),
            'urban_perc_2018': urban_fraction_with_specifier('2018'),
            'crop_perc_2018': crop_fraction_with_specifier('2018'), 
            'elev_gauge': gauge_elevation_meters(),
            'elev_50': med_catchment_elevation_meters(),
            'soil_depth': soil_depth(),
            'dens_inhabitants': population_density(),
        }

    @property
    def static_factors(self) -> Dict[str, float]:
        """
        static factors for CAMELS-FI catchments
        """
        return {
            grass_fraction_with_specifier('2000'): 0.01,
            urban_fraction_with_specifier('2000'): 0.01,
            crop_fraction_with_specifier('2000'): 0.01,
            grass_fraction_with_specifier('2006'): 0.01,
            urban_fraction_with_specifier('2006'): 0.01,
            crop_fraction_with_specifier('2006'): 0.01,
            grass_fraction_with_specifier('2012'): 0.01,
            urban_fraction_with_specifier('2012'): 0.01,
            crop_fraction_with_specifier('2012'): 0.01,
            grass_fraction_with_specifier('2018'): 0.01,
            urban_fraction_with_specifier('2018'): 0.01,
            crop_fraction_with_specifier('2018'): 0.01,
        }

    @property
    def start(self) -> pd.Timestamp:
        """
        start of data
        """
        return pd.Timestamp('1961-01-01')
    
    @property
    def end(self) -> pd.Timestamp:
        """
        end of data
        """
        return pd.Timestamp('2023-12-31')

    def _static_data(self) -> pd.DataFrame:
        """
        static attributes of catchments

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of static features of all catchments of shape (320, 112)
        """

        csv_files = glob.glob(os.path.join(glob.escape(self.data_path), '*.csv'))

        dfs = []
        for csv_file in csv_files:

            df = pd.read_csv(
                csv_file, 
                index_col=0, 
                dtype={0: str}
            )

            df.index = df.index.astype(str)

            dfs.append(df)
        
        static_data = pd.concat(dfs, axis=1)

        static_data.rename(columns=self.static_map, inplace=True)

        # static_factors should be called after renaming the columns
        for col, fac in  self.static_factors.items():
            if col in static_data.columns:
                static_data[col] *= fac
        
        return static_data

    def _read_stn_dyn(self, stn:str, nrows=None)->pd.DataFrame:
        """
        reads dynamic data for a given station
        """

        fpath = os.path.join(
            self.ts_path, 
            f"CAMELS_FI_hydromet_timeseries_{stn}_19610101-20231231.csv")
        
        df = pd.read_csv(fpath, index_col=0, parse_dates=True, nrows=nrows)

        df.index = pd.to_datetime(df.index)
        if df.index.has_duplicates:
            warnings.warn(f"{stn} has duplicated index. Removing duplicates.")
          
        df.rename(columns=self.dyn_map, inplace=True)

        self._apply_dyn_factors(df)

        return df

    def _download_camels_fi(self, overwrite: bool = False):
        """
        Downloads and extracts ``CAMELS-FI.zip`` of release 1.2.0. Nothing is
        downloaded if the extracted data is already there, even when the archive
        was deleted (``remove_zip=True``). Data of release 1.0.1, which an
        earlier version of this class extracted into ``CAMELS-FI/CAMELS-FI/``,
        and the netCDF caches built from it are removed first.
        """
        archive = os.path.join(self.path, self._archive_name)
        # caches of release 1.0.1; the cache of this release carries the release
        # in its name and is therefore never one of these
        stem = f"{self.name.lower()}_{self.timestep}"
        old_caches = [os.path.join(self.path, f"{stem}.nc"),
                      os.path.join(self.path, cache_name(f"{stem}.nc"))]

        if overwrite:
            _remove_stale([archive, self._root, self.dyn_fpath, *old_caches],
                          self.verbosity)
        elif self._holds_old_release():
            stale = [path for path in (self._root, archive, *old_caches)
                     if os.path.lexists(path)]
            # warned regardless of verbosity: the data on disk is deleted
            warnings.warn(
                f"CAMELS_FI: {self.path} holds release 1.0.1, whose gauge coordinates "
                f"are wrong and whose potential evapotranspiration was withdrawn by "
                f"the authors. Replacing it with release {self.release} ({self.url}), "
                f"which is downloaded again ({len(stale)} paths removed: "
                f"{', '.join(os.path.basename(path) for path in stale)}).", UserWarning)
            _remove_stale(stale, self.verbosity, reason=f"replaced by release {self.release}")

        if os.path.isdir(self.data_path):
            if self.verbosity:
                print(f"CAMELS_FI release {self.release} already exists at {self._root}")
            self.maybe_remove_zip_files()
            return

        if not os.path.exists(archive):
            # imported here because this module installs a SIGINT handler on import
            from ...download_zenodo import download_from_zenodo
            os.makedirs(self.path, exist_ok=True)
            # the record also holds support_document.pdf, which is inside the archive
            download_from_zenodo(self.path, doi=self.url, include=[self._archive_name],
                                 verbosity=self.verbosity)

        self._extract(archive)

        self.maybe_remove_zip_files()
        return

    def _holds_old_release(self) -> bool:
        """
        True if :attr:`path` holds data that is not release 1.2.0, i.e. release
        1.0.1 or an extraction that was interrupted. An empty :attr:`path` and a
        complete release 1.2.0 both give False.
        """
        if not os.path.isdir(self._root):
            return False
        return not os.path.exists(os.path.join(self.data_path, self._marker))

    def _extract(self, archive: Union[str, os.PathLike]):
        """
        Extracts ``archive`` into a temporary folder which is renamed to
        :attr:`_root` once complete, so that an interrupted extraction is redone
        on the next initialization instead of being taken as complete.
        """
        tmp = f"{self._root}_extracting"
        shutil.rmtree(tmp, ignore_errors=True)  # left over from an interrupted extraction

        if self.verbosity:
            print(f"extracting {archive}")

        try:
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(tmp)
        except (zipfile.BadZipFile, zlib.error, EOFError):  # e.g. an error page saved as the archive
            shutil.rmtree(tmp, ignore_errors=True)
            os.remove(archive)
            raise ValueError(f"{archive} is corrupt and was deleted. "
                             f"Initialize CAMELS_FI again to download it again.") from None

        # the archive holds a single top-level folder named CAMELS-FI
        shutil.rmtree(self._root, ignore_errors=True)  # os.replace needs a free name
        os.replace(os.path.join(tmp, "CAMELS-FI"), self._root)
        shutil.rmtree(tmp, ignore_errors=True)
        return

    def _check_manifest(self):
        """
        Warns if any attribute file, boundary file or station time series of the
        release is missing, e.g. after an interrupted extraction. The expected
        stations are the gauge ids of the metadata file, not whatever csv files
        happen to be in :attr:`ts_path`.
        """
        meta = os.path.join(self.data_path, self._attr_files[0])
        if not os.path.exists(meta):
            raise FileNotFoundError(
                f"{meta} not found. Re-initialize CAMELS_FI with overwrite=True.")

        gauges = pd.read_csv(meta, usecols=[0], dtype=str).iloc[:, 0]

        files = [os.path.join(self.data_path, fname) for fname in self._attr_files]
        files += [self.boundary_file[:-len(".shp")] + ext
                  for ext in (".shp", ".shx", ".dbf", ".prj")]
        files += [os.path.join(self.ts_path,
                               f"CAMELS_FI_hydromet_timeseries_{gauge}_19610101-20231231.csv")
                  for gauge in gauges]

        missing = [fpath for fpath in files if not os.path.exists(fpath)]
        if missing:
            warnings.warn(
                f"CAMELS_FI {self.release}: {len(missing)} of {len(files)} files are "
                f"missing: {missing[:5]}{' ...' if len(missing) > 5 else ''}. "
                f"Use overwrite=True to download them again.", UserWarning)
        return

    def _unzip_boundaries(self):
        if not os.path.exists(self.boundary_path):
            zip_file = os.path.join(self.data_path, "CAMELS_FI_catchment_boundaries.zip")
            if os.path.exists(zip_file):
                if self.verbosity:
                    print(f"Unzipping boundary file {os.path.basename(zip_file)} to {self.data_path}")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall(self.boundary_path)
            else:
                raise FileNotFoundError(f"Boundary file {zip_file} not found.")
        return
