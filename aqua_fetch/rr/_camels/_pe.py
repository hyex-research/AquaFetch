import os
from typing import List, Dict

import pandas as pd

from ..utils import _RainfallRunoff
from ...utils import download_and_unzip
from .._map import (
    observed_streamflow_mm,
    mean_air_temp,
    max_air_temp,
    min_air_temp,
    total_precipitation,
    total_potential_evapotranspiration,
    mean_vapor_pressure,
    solar_radiation,
    MJ_M2_DAY_TO_WM2,
)
from .._map import (
    catchment_area,
    gauge_latitude,
    gauge_longitude,
    slope,
    gauge_elevation_meters,
    catchment_elevation_meters,
    catchment_perimeter,
    med_catchment_elevation_meters,
    min_catchment_elevation_meters,
    max_catchment_elevation_meters,
)
from ._common import _remove_stale, _warn_duplicate_gauges


class CAMELS_PE(_RainfallRunoff):
    """
    Daily hydrometeorological time series and static catchment attributes for
    136 catchments in Peru following Llauca et al. (CAMELS-PE v1.0.1). The data
    is downloaded from its
    `zenodo repository <https://zenodo.org/records/21195425>`_ .

    The dataset provides 9 daily dynamic features and 78 static features for
    each catchment. The dynamic (time series) features span from 1981-01-01 to
    2025-12-31 with a daily timestep (16436 steps), although individual
    variables retain the temporal coverage of their source product (observed
    streamflow is gauge-dependent and mostly incomplete, PET ends in 2016, air
    temperature ends in 2020) with the remaining dates filled with ``NaN``.
    Observed streamflow (``q_mm_obs``, catchment-averaged runoff depth in mm/day)
    is provided by SENAMHI while the meteorological forcing is derived from the
    PISCO (PISCOp v2.1, PISCOt v1.2, PISCOeo_pm v1.0) and ERA5-Land gridded
    products.

    .. note::
        The source dataset also ships a model-simulated streamflow series
        (``flow_sim``, from PISCO-ARNOVIC v1.1). Following the library's
        observational-data-only policy, this simulated series is **not** presented
        as a dynamic feature, so ``CAMELS_PE`` exposes 9 dynamic features rather
        than the 10 documented in the paper.

    The 78 static features comprise 64 thematic catchment attributes (7
    topography, 10 climatic indices, 13 hydrological signatures, 8 land cover, 7
    geology, 10 soil, 9 human intervention) plus 14 gauge-metadata / relational
    columns from ``stations.csv`` (name, region, record start/end, percentage of
    valid observations, official catchment name, and the nested-catchment
    up/down-stream relations). The 13 hydrological signatures are computed from
    the streamflow records; consult the shipped ``data_dictionary.csv`` for the
    exact source of each attribute (per the dictionary most cite the observed
    SENAMHI series, while the paper describes computing them from the simulated
    series for temporal completeness). Catchment boundaries and gauge outlets are
    provided as GeoPackage files in WGS84 (EPSG:4326).

    On a typical machine the one-time download (~121 MB), extraction and netCDF
    cache build take ~45 s; thereafter fetching all 136 stations (all 9 dynamic
    features) from the cache takes ~0.2 s.

    .. note::
        ``srad`` (MJ m-2 day-1) is served as ``swdownrad_wm2``, converted to
        W m-2. ``prec_var`` (spatial precipitation variance, mm2 day-2) keeps
        its original name because no aqua_fetch canonical name carries those
        units.

    Examples
    --------
    >>> from aqua_fetch import CAMELS_PE
    >>> dataset = CAMELS_PE()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='PE_204617', as_dataframe=True)
    >>> df = dynamic['PE_204617'] # dynamic is a dictionary with keys as station names and values as DataFrames
    >>> df.shape
    (16436, 9)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       136
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (13 out of 136)
       13
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(16436, 9), (16436, 9), ... (16436, 9)]
    ...
    ... # get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('PE_204617', as_dataframe=True,
    ...  dynamic_features=['airtemp_C_mean', 'pcp_mm', 'pet_mm', 'q_mm_obs'])
    >>> dynamic['PE_204617'].shape
       (16436, 4)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='PE_204617', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['PE_204617'].shape
    ((1, 78), 1, (16436, 9))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)
    xarray.core.dataset.Dataset
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (136, 2)
    >>> dataset.stn_coords('PE_204617')  # returns coordinates of station whose id is PE_204617
    ...
    # get area (km2) of a single station
    >>> dataset.area('PE_204617')
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('PE_204617')
    """
    url = "https://zenodo.org/records/21195425"

    # name of the single ~121 MB archive on the Zenodo record. Only this file is
    # downloaded (the accompanying technical-description pdf is skipped).
    _archive_name = "CAMELS-PE_v1.0.1.zip"

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
            ``camels_pe_D_v2.nc`` cache is written once next to the data. It is
            silently disabled if netCDF4 is not installed (handled by the base
            class).
        verbosity : int
            0: no message will be printed
        """
        super().__init__(path=path, overwrite=overwrite, to_netcdf=to_netcdf,
                         verbosity=verbosity, **kwargs)

        # lazy caches (heavy attributes are loaded on first use)
        self._static_df = None
        self._static_feats = None
        self._stns = None
        self._dyn_feats = None

        self._download_camels_pe(overwrite=overwrite)

        self._warn_duplicate_gauges()

        self._maybe_to_netcdf()

        # bounding box of Peru (used for plotting the station map)
        self.bbox = {'llcrnrlat': -18.5, 'urcrnrlat': 0.5,
                     'llcrnrlon': -81.5, 'urcrnrlon': -68.5}
        self.parallels = range(-18, 2, 4)
        self.meridians = range(-81, -68, 4)

    def _download_camels_pe(self, overwrite: bool = False):
        """
        Downloads and extracts the CAMELS-PE archive from Zenodo.

        The guard is on the extracted ``03_timeseries`` folder so that once the
        data is on disk (even if the ~121 MB ``CAMELS-PE_v1.0.1.zip`` archive has
        been deleted afterwards, e.g. via :meth:`free_disk_space`) no
        re-download is triggered. Only ``CAMELS-PE_v1.0.1.zip`` is requested
        (``include=...``); the supplementary technical pdf on the record is not
        downloaded.
        """
        if os.path.exists(self.ts_dir) and not overwrite:
            if self.verbosity:
                print(f"CAMELS_PE data already exists at {self._root}")
            return

        if overwrite:
            archive = os.path.join(self.path, self._archive_name)
            extracted = os.path.dirname(self._root)  # <path>/CAMELS-PE_v1.0.1
            _remove_stale((archive, extracted, self.dyn_fpath), self.verbosity)

        download_and_unzip(self.path, url=self.url, include=[self._archive_name],
                           verbosity=self.verbosity)

    @property
    def _root(self) -> os.PathLike:
        """folder holding the four CAMELS-PE component directories.

        ``unzip`` extracts ``CAMELS-PE_v1.0.1.zip`` into a folder named
        ``CAMELS-PE_v1.0.1`` and the archive itself has a top-level
        ``CAMELS-PE`` folder, hence the doubly-nested path.
        """
        return os.path.join(self.path, "CAMELS-PE_v1.0.1", "CAMELS-PE")

    @property
    def _meta_dir(self) -> os.PathLike:
        return os.path.join(self._root, "01_metadata")

    @property
    def _attr_dir(self) -> os.PathLike:
        return os.path.join(self._root, "02_attributes")

    @property
    def ts_dir(self) -> os.PathLike:
        """folder containing the daily hydro-meteorological time-series files"""
        return os.path.join(self._root, "03_timeseries")

    @property
    def _geo_dir(self) -> os.PathLike:
        return os.path.join(self._root, "04_geospatial")

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(self._geo_dir, "camels_pe_catchments.gpkg")

    @property
    def boundary_id_map(self) -> str:
        """attribute in the boundary GeoPackage used to map to the gauge id"""
        return "gauge_id"

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp("1981-01-01")

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp("2025-12-31")

    @property
    def dyn_map(self) -> Dict[str, str]:
        """maps the raw time-series column names (Table 11 of the technical
        documentation) to the standardised aqua_fetch names. The units of the
        raw and standardised names are identical so no unit conversion is
        required.

        The raw ``flow_sim`` column (model-simulated streamflow from
        PISCO-ARNOVIC v1.1) is intentionally omitted here and dropped in
        :meth:`_read_stn_dyn`, following the library's observational-data-only
        policy (simulated data is not presented). ``prec_var`` (mm2 day-2) is
        left unmapped because no aqua_fetch canonical name carries those units,
        so renaming would misrepresent it.
        """
        return {
            'prec': total_precipitation(),                   # mm day-1
            'flow_obs': observed_streamflow_mm(),            # mm day-1 (observed, SENAMHI)
            'pet': total_potential_evapotranspiration(),     # mm day-1
            'tmin': min_air_temp(),                          # deg C
            'tmean': mean_air_temp(),                        # deg C
            'tmax': max_air_temp(),                          # deg C
            'vprp': mean_vapor_pressure(),                   # hPa
            # README: "Radiation: MJ m-2 d-1", source ERA5-Land. Converted to
            # W m-2 in dyn_factors. Read as DOWNWARD shortwave: it is listed
            # among the forcing variables (prec/pet/temp/srad/vprp), and a
            # forcing set uses ERA5-Land's ssrd rather than the model-output
            # ssr. Clearness index after conversion is 0.39-0.52 across lat
            # -0.9 to -12.6, consistent with the humid tropics and the Andes.
            # Note this is inference, not proof: unlike GSHA there is no second
            # dataset here publishing both ERA5-Land fields for the same
            # catchments to check against.
            'srad': solar_radiation(),                       # MJ m-2 day-1
        }

    @property
    def dyn_factors(self) -> Dict[str, float]:
        return {
            solar_radiation(): MJ_M2_DAY_TO_WM2,
        }

    @property
    def static_map(self) -> Dict[str, str]:
        """maps the raw static-attribute column names to the standardised
        aqua_fetch names. Only the attributes that have a canonical aqua_fetch
        name (and are needed by :meth:`area`, :meth:`stn_coords` etc.) are
        renamed; the remaining CAMELS-PE attributes keep their original,
        already CAMELS-standard names.
        """
        return {
            'area': catchment_area(),                        # km2
            'perimeter': catchment_perimeter(),              # km
            'gauge_lat': gauge_latitude(),                   # deg N (WGS84)
            'gauge_lon': gauge_longitude(),                  # deg E (WGS84)
            'gauge_elev': gauge_elevation_meters(),          # m a.s.l
            'elev_mean': catchment_elevation_meters(),       # m a.s.l
            'elev_min': min_catchment_elevation_meters(),    # m a.s.l
            'elev_max': max_catchment_elevation_meters(),    # m a.s.l
            'elev_median': med_catchment_elevation_meters(), # m a.s.l
            'slope_mean': slope('mkm-1'),                    # m km-1
        }

    @property
    def _mm_feature_name(self) -> str:
        """observed catchment-specific discharge (mm day-1) is provided directly
        in the dataset, so :meth:`q_mm` uses it instead of converting from cms"""
        return observed_streamflow_mm()

    @property
    def _area_name(self) -> str:
        # post-rename (standardised) name, matching the columns of _static_data()
        return catchment_area()

    @property
    def _coords_name(self) -> List[str]:
        # post-rename (standardised) names, matching the columns of _static_data()
        return [gauge_latitude(), gauge_longitude()]

    def _warn_duplicate_gauges(self):
        """
        Warns (unconditionally, i.e. regardless of ``verbosity``) if any two
        gauges share the same name **and** rounded coordinates. This is a cheap
        one-time check on the small ``stations.csv`` file and does not slow the
        fetching process. Duplicates are only warned about, never excluded.
        """
        meta = pd.read_csv(
            os.path.join(self._meta_dir, "stations.csv"),
            usecols=['gauge_id', 'gauge_name', 'gauge_lat', 'gauge_lon'],
            dtype={'gauge_id': str})
        _warn_duplicate_gauges(self.name, meta)
        return

    def stations(self) -> List[str]:
        """ids of the 136 gauges (read from the ``gauge_id`` column of
        ``stations.csv``)"""
        if self._stns is None:
            meta = pd.read_csv(
                os.path.join(self._meta_dir, "stations.csv"),
                usecols=['gauge_id'], dtype={'gauge_id': str})
            self._stns = meta['gauge_id'].tolist()
        return list(self._stns)

    @property
    def static_features(self) -> List[str]:
        # cache the column list so we don't copy the whole frame just to read it
        if self._static_feats is None:
            self._static_feats = self._static_data().columns.tolist()
        return list(self._static_feats)

    @property
    def dynamic_features(self) -> List[str]:
        if self._dyn_feats is None:
            self._dyn_feats = self._read_stn_dyn(self.stations()[0]).columns.tolist()
        return list(self._dyn_feats)

    def _static_data(self) -> pd.DataFrame:
        """all 78 static attributes (index is ``gauge_id``). Built once from the
        metadata + the seven thematic attribute tables and then cached. A copy
        of the cache is returned so a caller's in-place edit cannot corrupt the
        cached frame."""
        if self._static_df is not None:
            return self._static_df.copy()

        dfs = [pd.read_csv(os.path.join(self._meta_dir, "stations.csv"),
                           index_col='gauge_id', dtype={'gauge_id': str})]
        for name in ("topographic_attributes", "climatic_indices",
                     "hydrological_signatures", "landcover_attributes",
                     "geologic_attributes", "soil_attributes",
                     "human_intervention_attributes"):
            dfs.append(pd.read_csv(os.path.join(self._attr_dir, f"{name}.csv"),
                                   index_col='gauge_id', dtype={'gauge_id': str}))

        df = pd.concat(dfs, axis=1)
        df.rename(columns=self.static_map, inplace=True)
        df.index.name = 'gauge_id'

        self._static_df = df
        return df.copy()

    def _read_stn_dyn(self, station: str) -> pd.DataFrame:
        """
        Reads the daily dynamic (meteorological + streamflow) data for one
        catchment and returns it as a DataFrame with time as index and the
        standardised dynamic-feature names as columns. Missing observations are
        retained as ``NaN``.

        Values are cast to ``self.fp`` (the ``float_precision`` of the base
        class, ``np.float32`` by default). This is safe for CAMELS-PE: the
        largest value across the whole dataset is ~2.55e4 (``prec_var``), far
        below the ``float32`` overflow limit, and the worst-case rounding error
        is ~5e-4 on ``prec_var`` only (relative error ~6e-8 everywhere), i.e.
        well below the 3-decimal resolution of every measurement. Pass
        ``float_precision=np.float64`` to keep full precision.
        """
        fpath = os.path.join(self.ts_dir, "by_catchment", f"{station}.csv")

        df = pd.read_csv(fpath, index_col='date', parse_dates=True)
        df.index.name = 'time'

        # simulated streamflow (raw ``flow_sim``, PISCO-ARNOVIC v1.1) is model
        # output, not an observation, so it is not presented as a dynamic feature
        # (library observational-data-only policy). Drop it before renaming.
        df = df.drop(columns=['flow_sim'], errors='ignore')

        df.rename(columns=self.dyn_map, inplace=True)

        self._apply_dyn_factors(df)

        return df.astype(self.fp)
