import os
from typing import List, Dict

import numpy as np
import pandas as pd

from ..utils import _RainfallRunoff
from ..._geom_utils import tmerc_to_wgs84
from ...utils import download_and_unzip
from ..._backend import fiona
from .._map import (
    observed_streamflow_cms,
    observed_streamflow_mm,
    observed_water_level_cm,
    mean_air_temp,
    max_air_temp,
    min_air_temp,
    total_precipitation_with_specifier,
    mean_rel_hum,
    mean_windspeed,
    solar_radiation,
)
from .._map import (
    catchment_area,
    gauge_latitude,
    gauge_longitude,
    gauge_elevation_meters,
    catchment_elevation_meters,
)


class CAMELS_PL(_RainfallRunoff):
    """
    Hydro-meteorological time series and static catchment attributes for 354
    streamflow gauges across Poland following the work of Brzezinska et al.
    (CAMELS-PL v1.0.0). The data is downloaded from its
    `zenodo repository <https://zenodo.org/records/20133183>`_ .

    This dataset consists of 74 static and 13 dynamic features for each
    catchment. The dynamic (time series) features span from 1951-01-01 to
    2024-12-31 with a daily timestep (27029 steps). The observed discharge and
    water-level records are provided by the Institute of Meteorology and Water
    Management - National Research Institute (IMGW-PIB), the meteorological
    forcing is derived from E-OBS v31.0e and the static attributes describe
    topography, climate, hydrology, soils and land cover. The machine-learning
    generated LSTM/HBV benchmark scores are **not** part of the static features
    (per the library's observational-data-only policy); they can be accessed
    separately via :meth:`benchmark_attrs`.

    .. note::
        This is a different dataset from :py:class:`aqua_fetch.Poland`. The
        ``Poland`` class provides 1287 catchments whose observed streamflow comes
        from `IMGW <https://danepubliczne.imgw.pl>`_ while the meteorological
        forcing, static attributes and boundaries are taken from
        :py:class:`aqua_fetch.EStreams` (214 static and 10 dynamic features).
        ``CAMELS_PL`` on the other hand is a self-contained, CAMELS-style dataset
        of 354 gauges published on Zenodo with its own harmonised attributes,
        benchmark model simulations and catchment boundaries.

    Examples
    --------
    >>> from aqua_fetch import CAMELS_PL
    >>> dataset = CAMELS_PL()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='149180020', as_dataframe=True)
    >>> df = dynamic['149180020'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (27029, 13)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       354
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (35 out of 354)
       35
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(27029, 13), (27029, 13), (27029, 13),... (27029, 13), (27029, 13)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('149180020', as_dataframe=True,
    ...  dynamic_features=['airtemp_C_mean', 'rh_%', 'pcp_mm_mean', 'q_cms_obs'])
    >>> dynamic['149180020'].shape
       (27029, 4)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='149180020', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['149180020'].shape
    ((1, 74), 1, (27029, 13))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 27029, 'dynamic_features': 13})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (354, 2)
    >>> dataset.stn_coords('149180020')  # returns coordinates of station whose id is 149180020
        49.921268       18.327517
    >>> dataset.stn_coords(['149180020', '149180040'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('149180020')
    # get area of two stations
    >>> dataset.area(['149180020', '149180040'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('149180020')
    """
    url = "https://zenodo.org/records/20133183"

    # name of the single ~780 MB archive on the Zenodo record. Only this file is
    # downloaded (the accompanying data-description pdf is skipped).
    _archive_name = "CAMELS-PL.zip"

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
            If the data is already downloaded then provide the complete
            path to it. If None, then the data will be downloaded.
            The data is downloaded once and therefore subsequent
            calls to this class will not download the data unless
            ``overwrite`` is set to True.
        overwrite : bool
            If the data is already downloaded then you can set it to True,
            to make a fresh download.
        to_netcdf : bool
            whether to convert all the dynamic data into one netcdf file or not.
            This will fasten repeated calls to fetch etc. but will require the
            netCDF4 package as well as xarray. When enabled, a consolidated
            ``camels_pl_D_v2.nc`` cache (~500 MB) is written once next to the data.
            It is silently disabled if netCDF4 is not installed (handled by the
            base class).
        verbosity : int
            0: no message will be printed
        """
        super().__init__(path=path, overwrite=overwrite, to_netcdf=to_netcdf,
                         verbosity=verbosity, **kwargs)

        self._download_camels_pl(overwrite=overwrite)

        self._maybe_to_netcdf()

        # bounding box of Poland (used for plotting the station map)
        self.bbox = {'llcrnrlat': 49.0, 'urcrnrlat': 55.0, 'llcrnrlon': 14.0, 'urcrnrlon': 24.5}
        self.parallels = range(49, 55, 2)
        self.meridians = range(14, 25, 2)

    def _download_camels_pl(self, overwrite: bool = False):
        """
        Downloads and extracts the CAMELS-PL archive from Zenodo.

        The guard is on the extracted ``timeseries`` folder so that once the
        data is on disk (even if the ~780 MB ``CAMELS-PL.zip`` archive has been
        deleted afterwards, e.g. via :meth:`free_disk_space`) no re-download is
        triggered. Only ``CAMELS-PL.zip`` is requested (``include=...``); the
        supplementary pdf on the record is not downloaded.
        """
        if os.path.exists(self.ts_dir) and not overwrite:
            if self.verbosity:
                print(f"CAMELS_PL data already exists at {self._root}")
            return
        download_and_unzip(self.path, url=self.url, include=[self._archive_name],
                           verbosity=self.verbosity)

    @property
    def _root(self) -> os.PathLike:
        """folder into which the archive extracts.

        ``unzip`` extracts ``CAMELS-PL.zip`` into a folder named ``CAMELS-PL``
        and the archive itself has a top-level ``CAMELS-PL`` folder, hence the
        doubly-nested path.
        """
        return os.path.join(self.path, "CAMELS-PL", "CAMELS-PL")

    @property
    def ts_dir(self) -> os.PathLike:
        """folder containing the 354 daily hydro-meteorological csv files"""
        return os.path.join(self._root, "timeseries")

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(self._root,
                            "CAMELS_PL_catchment_boundaries",
                            "catchments",
                            "CAMELS_PL_catchments.shp")

    @property
    def boundary_id_map(self) -> str:
        """attribute in the boundary shapefile used to map to the gauge id"""
        return "gauge_id"

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp("1951-01-01")

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp("2024-12-31")

    @property
    def dyn_map(self) -> Dict[str, str]:
        """maps the raw column names of the hydro-meteorological time series
        files (Table 2 of the data description) to the standardised
        aqua_fetch names. The units of the raw and standardised names are
        identical so no unit conversion is required."""
        return {
            'discharge_vol_obs': observed_streamflow_cms(),                    # m3 s-1
            'discharge_spec_obs': observed_streamflow_mm(),                    # mm day-1
            'water_level_obs': observed_water_level_cm(),                      # cm
            'precipitation_min': total_precipitation_with_specifier('min'),    # mm day-1
            'precipitation_mean': total_precipitation_with_specifier('mean'),  # mm day-1
            'precipitation_max': total_precipitation_with_specifier('max'),    # mm day-1
            'precipitation_stdev': total_precipitation_with_specifier('std'),  # mm day-1
            'temperature_mean': mean_air_temp(),                               # deg C
            'maximum_temperature_mean': max_air_temp(),                        # deg C
            'minimum_temperature_mean': min_air_temp(),                        # deg C
            'wind_speed_mean': mean_windspeed(),                               # m s-1
            'relative_humidity_mean': mean_rel_hum(),                          # %
            'radiation_global_mean': solar_radiation(),                        # W m-2
        }

    @property
    def static_map(self) -> Dict[str, str]:
        """maps the raw static attribute column names to the standardised
        aqua_fetch names.

        .. warning::
            The ``gauge_lon`` and ``gauge_lat`` columns are **swapped** in the
            source ``CAMELS_PL_topographic_attributes.csv`` file: the column
            labelled ``gauge_lon`` actually holds latitude values (~49-55) while
            ``gauge_lat`` holds longitude values (~14-24). This was verified
            against the EPSG:2180 gauging-station geometry. They are therefore
            mapped to the *correct* canonical names below so that
            :meth:`stn_coords` returns valid (lat, long) pairs.
        """
        return {
            'area_metadata': catchment_area(),           # km2
            'gauge_lon': gauge_latitude(),               # -> lat  (see warning: source columns are swapped)
            'gauge_lat': gauge_longitude(),              # -> long (see warning: source columns are swapped)
            'gauge_elev': gauge_elevation_meters(),      # m a.s.l
            'elev_mean': catchment_elevation_meters(),   # m a.s.l
        }

    @property
    def _mm_feature_name(self) -> str:
        """observed catchment-specific discharge (mm day-1) is provided directly
        in the dataset, so :meth:`q_mm` uses it instead of converting from cms"""
        return observed_streamflow_mm()

    @property
    def _area_name(self) -> str:
        return 'area_metadata'

    @property
    def _coords_name(self) -> List[str]:
        # note: the source columns are swapped (gauge_lon holds latitude), see static_map
        return ['gauge_lon', 'gauge_lat']

    def _attr_path(self, name: str) -> os.PathLike:
        """path to a static-attribute csv file"""
        return os.path.join(self._root, f"CAMELS_PL_{name}.csv")

    def stations(self) -> List[str]:
        """names/ids of the 354 gauges (parsed from the time-series filenames)"""
        prefix = "CAMELS_PL_hydromet_timeseries_"
        return [f[len(prefix):-len(".csv")]
                for f in os.listdir(self.ts_dir)
                if f.startswith(prefix) and f.endswith(".csv")]

    def topographic_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self._attr_path("topographic_attributes"),
                           index_col='gauge_id', dtype={'gauge_id': str})

    def climatic_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self._attr_path("climatic_attributes"),
                           index_col='gauge_id', dtype={'gauge_id': str})

    def hydrologic_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self._attr_path("hydrologic_attributes"),
                           index_col='gauge_id', dtype={'gauge_id': str})

    def soil_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self._attr_path("soil_attributes"),
                           index_col='gauge_id', dtype={'gauge_id': str})

    def landcover_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self._attr_path("landcover_attributes"),
                           index_col='gauge_id', dtype={'gauge_id': str})

    def bdot10k_attrs(self) -> pd.DataFrame:
        """BDOT10k land-cover attributes. Note this file only covers the 241
        catchments that are fully located inside Polish territory; the remaining
        catchments will have NaN for these columns."""
        return pd.read_csv(self._attr_path("BDOT10K_land_cover_catchments"),
                           index_col='gauge_id', dtype={'gauge_id': str})

    def benchmark_attrs(self) -> pd.DataFrame:
        """
        Machine-learning / model benchmark results from
        ``CAMELS_PL_simulation_benchmark.csv``.

        This returns the **model-generated** LSTM/HBV benchmark scores
        (``NSE_lstm_ensemble_median``/``std``, ``NSE_hbv_ensemble_median``/``std``
        and the per-seed ``NSE_lstm_seed1..5`` / ``NSE_hbv_seed1..5``) together
        with the three observed-data completeness fractions
        (``training``/``validation``/``testing_perc_complete``).

        Because these NSE values are machine-learning generated (not
        observations), they are deliberately **excluded** from
        :meth:`static_features`; use this method if you specifically need them.
        Only the three observational ``*_perc_complete`` columns are folded into
        the static attributes.

        Returns
        -------
        pd.DataFrame
            index is the gauge id, columns are the 14 NSE scores plus the 3
            completeness fractions.

        Examples
        --------
        >>> from aqua_fetch import CAMELS_PL
        >>> dataset = CAMELS_PL()
        >>> scores = dataset.benchmark_attrs()
        >>> scores.loc['149180020', 'NSE_lstm_ensemble_median']
        """
        return pd.read_csv(self._attr_path("simulation_benchmark"),
                           index_col='gauge_id', dtype={'gauge_id': str})

    def _static_data(self) -> pd.DataFrame:
        # only the observed-data completeness fractions from the benchmark file
        # are treated as static attributes. The NSE_* columns are machine-learning
        # generated model scores and are intentionally left out (they remain
        # accessible via :meth:`benchmark_attrs`), following the library policy of
        # exposing observational data only.
        benchmark = self.benchmark_attrs()
        completeness = benchmark[[c for c in benchmark.columns
                                  if c.endswith('_perc_complete')]]

        df = pd.concat([
            self.topographic_attrs(),
            self.climatic_attrs(),
            self.hydrologic_attrs(),
            self.soil_attrs(),
            self.landcover_attrs(),
            self.bdot10k_attrs(),
            completeness,
        ], axis=1)

        df.rename(columns=self.static_map, inplace=True)

        return df

    def _read_stn_dyn(self, station: str) -> pd.DataFrame:
        """
        Reads daily dynamic (meteorological + streamflow) data for one catchment
        and returns it as a DataFrame with time as index and the standardised
        dynamic-feature names as columns.
        """
        fpath = os.path.join(self.ts_dir,
                             f"CAMELS_PL_hydromet_timeseries_{station}.csv")

        df = pd.read_csv(fpath, index_col='date', parse_dates=True)

        df = df.astype(np.float32)

        df.rename(columns=self.dyn_map, inplace=True)

        return df

    def observed_q_cms(self) -> pd.DataFrame:
        """
        Returns the supplementary wide-format observed volumetric discharge
        (m3 s-1) from ``CAMELS_PL_Q_354_data.csv``. Unlike the per-catchment
        time-series files (which run over calendar years 1951-01-01 to
        2024-12-31), this file runs from 1950-11-01 to 2024-10-31 so that
        complete Polish hydrological years (1 November - 31 October) can be
        analysed. Missing values are encoded as NA.

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` whose index is time and whose columns are
            the 354 gauge ids. The dataset's ``hyy`` (hydrological year) helper
            column is dropped.

        Examples
        --------
        >>> from aqua_fetch import CAMELS_PL
        >>> dataset = CAMELS_PL()
        >>> q = dataset.observed_q_cms()
        >>> q.shape
        (27029, 354)
        """
        fpath = os.path.join(self._root, "CAMELS_PL_Q_354_data.csv")
        df = pd.read_csv(fpath, index_col='date', parse_dates=True)
        df = df.drop(columns=['hyy'], errors='ignore')
        df.index.name = 'time'
        return df

    def transform_boundary(self, boundary):
        """
        Transforms a catchment boundary from ETRS89 / Poland CS92 (EPSG:2180,
        meters) to WGS84 (lat/lon). Both ``Polygon`` and ``MultiPolygon``
        geometries are supported. The transformation uses the dependency-free
        :func:`aqua_fetch._geom_utils.tmerc_to_wgs84` helper which matches
        ``pyproj`` to within ~5 cm.
        """
        # EPSG:2180 parameters (from the shapefile .prj)
        lon_0, k0 = 19.0, 0.9993
        false_easting, false_northing = 500000.0, -5300000.0

        def _convert(coords):
            # a coordinate pair is a sequence whose first element is a number
            if len(coords) >= 2 and isinstance(coords[0], (int, float)):
                lat_, long_ = tmerc_to_wgs84(coords[0], coords[1], lon_0, k0,
                                             false_easting, false_northing)
                return (long_, lat_)   # fiona stores coordinates as (long, lat)
            return [_convert(c) for c in coords]

        new_coords = _convert(boundary['coordinates'])

        if fiona is not None:
            boundary = fiona.Geometry(type=boundary['type'], coordinates=new_coords)
        return boundary
