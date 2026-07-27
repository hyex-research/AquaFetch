
import os
import time
import warnings
import concurrent.futures as cf
from functools import partial
from typing import Union, List, Dict

import numpy as np
import pandas as pd

from .utils import _RainfallRunoff
from ..utils import download, get_cpus, validate_attributes
from .._geom_utils import osgb36_to_wgs84
from ._map import (
    observed_streamflow_cms,
    catchment_area,
    gauge_latitude,
    gauge_longitude,
)

# CEH EIDC datastore for UK-Flow15 (https://doi.org/10.5285/211710ac-f01b-4b52-807f-373babb1c368)
DATASTORE = ("https://catalogue.ceh.ac.uk/datastore/eidchub/"
             "211710ac-f01b-4b52-807f-373babb1c368")

# metadata file -> its sub-directory on the datastore
_META_FILES = {
    '00_station_id_meta.csv': '2_1_station_identification_metadata',
    '01_common_sense_anomalies_meta.csv': '2_2_quality_control_metadata',
    '02_uk_products_meta.csv': '2_2_quality_control_metadata',
    '03_traditional_qc_meta.csv': '2_2_quality_control_metadata',
    '04_high_flows_qc_meta.csv': '2_2_quality_control_metadata',
    '05_resolution_meta.csv': '2_3_traceability_of_procedures_metadata',
    '06_duplicates_meta.csv': '2_3_traceability_of_procedures_metadata',
}

# short name -> metadata filename for the qc_metadata() accessor
_QC_TABLES = {
    'common_sense_anomalies': '01_common_sense_anomalies_meta.csv',
    'uk_products': '02_uk_products_meta.csv',
    'traditional_qc': '03_traditional_qc_meta.csv',
    'high_flows_qc': '04_high_flows_qc_meta.csv',
    'resolution': '05_resolution_meta.csv',
    'duplicates': '06_duplicates_meta.csv',
}


def _download_station_file(stem: str, flow_dir: str, verbosity: int = 0,
                           retries: int = 4) -> str:
    """
    Module-level (picklable) worker that downloads a single station flow file.

    Kept at module level rather than as a bound method so that handing it to a
    :obj:`concurrent.futures.ProcessPoolExecutor` does not pickle ``self`` (and
    with it the cached static-metadata table) to every worker.

    Retries a few times with exponential backoff, because the CEH datastore can
    return transient HTTP errors when many station files are requested
    concurrently (e.g. when fetching a large number of stations in parallel).
    Any partial file from a failed attempt is removed before retrying so the
    download helper does not fall back to a ``name.csv1`` sidecar.
    """
    url = f"{DATASTORE}/1_RiverFlowStations/{stem}.csv"
    fpath = os.path.join(flow_dir, f"{stem}.csv")
    for attempt in range(retries):
        try:
            download(url=url, outdir=flow_dir, fname=f"{stem}.csv",
                     verbosity=verbosity)
            return fpath
        except Exception:
            if os.path.exists(fpath):
                os.remove(fpath)
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))
    return fpath


class UKFlow15(_RainfallRunoff):
    """
    UK-Flow15: quality-controlled 15-minute river flow observations from 1369
    gauging stations across the United Kingdom, spanning 1948-08-09 to
    2023-12-31, following the work of
    `Coxon et al., 2026 <https://doi.org/10.5194/essd-2026-152>`__ .
    The data is downloaded from the
    `CEH EIDC datastore <https://doi.org/10.5285/211710ac-f01b-4b52-807f-373babb1c368>`__ .

    Unlike the CAMELS-style datasets, UK-Flow15 provides **only observed river
    discharge** (no meteorological forcing and no catchment boundaries). Each
    per-station flow file therefore yields three dynamic columns:

        - ``q_cms_obs``  : river discharge in m3/s (source column ``value``).
          Returned at ``float_precision`` (``numpy.float32`` by default, which
          halves memory). float32 preserves the source's 3-decimal values
          exactly for any discharge below ~16000 m3/s -- far above any UK 15-min
          river flow (even the flashiest gauges stay in the low thousands of
          m3/s), so the downcast is lossless in practice (verified
          float32 == float64 to 3 decimals over the full record of the
          largest-catchment station). Pass ``float_precision=numpy.float64`` for
          double precision.
        - ``resolution`` : native temporal resolution of the observation in
          minutes (15 for genuine 15-min data, larger where a coarser native
          record was placed on the 15-min grid, e.g. 120 for a 2-hour interval;
          special source codes are 0 for irregular intervals and 1 for a
          one-second gap; ``NaN`` on gaps and on the first observation of each
          record)
        - ``flag``       : the paper's 3-digit ``QC_code`` (``XYZ``), kept as a
          zero-padded string because the leading zeros are meaningful. ``000``
          means no issue and ``NaN`` marks a missing observation. The digits are
          independent quality-control categories: ``X`` = consistency against
          the NRFA daily / AMAX / POT products, ``Y`` = traditional QC
          (spikes, drops, truncations, ...), ``Z`` = high-flow QC. The full code
          tables are in the paper; per-station counts are in :meth:`qc_metadata`.

    No screening is applied: every observation is returned as-is together with
    its ``flag`` and ``resolution`` so the user can curate to the level their
    application needs (see :meth:`qc_metadata` for the per-station QC summaries).

    The per-station static metadata provides catchment area (km2), station
    coordinates (converted from British National Grid to WGS84), the record
    period, the percentage of missing values and the NRFA quality status. The
    per-station quality-control / traceability tables are exposed unchanged
    through :meth:`qc_metadata`.

    .. note::
        The full dataset is ~64 GB (individual station files are 10-90 MB).
        To keep initialisation quick, only the small (~0.4 MB) metadata is
        downloaded when the class is instantiated; each station's flow file is
        downloaded on first access and cached on disk. Use
        :meth:`download_all_stations` to pre-fetch the whole archive.

    Examples
    --------
    >>> from aqua_fetch import UKFlow15
    >>> dataset = UKFlow15()
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
    1369
    ... # names of available dynamic features
    >>> dataset.dynamic_features
    ['q_cms_obs', 'resolution', 'flag']
    ... # get all dynamic data of one station (downloaded on first access)
    >>> _, dynamic = dataset.fetch(stations='24007', as_dataframe=True)
    >>> dynamic['24007'].shape
    (356, 3)
    ... # get only discharge of one station
    >>> _, dynamic = dataset.fetch('24007', dynamic_features='q_cms_obs', as_dataframe=True)
    >>> dynamic['24007'].shape
    (356, 1)
    ... # get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='24007', static_features="all", as_dataframe=True)
    >>> static.shape, dynamic['24007'].shape
    ((1, 11), (356, 3))
    ... # coordinates (WGS84) of a station
    >>> dataset.stn_coords('24007')
    ... # catchment area (km2) of a station
    >>> dataset.area('24007')
    ... # per-station quality-control metadata
    >>> dataset.qc_metadata('traditional_qc').shape
    (1369, 9)
    """

    def __init__(
            self,
            path: Union[str, os.PathLike] = None,
            to_netcdf: bool = False,
            overwrite: bool = False,
            verbosity: int = 1,
            **kwargs
    ):
        """
        parameters
        ------------
        path : str
            If the data is already downloaded then provide the complete
            path to it. If None, then the data will be downloaded in the
            default directory. Only the metadata is downloaded at
            initialisation; the (large) per-station flow files are downloaded
            on first access.
        to_netcdf : bool
            kept for API compatibility only. Consolidating 1369 stations of
            15-min data (1948-2023) into a single netCDF is not feasible, so
            this is forced to ``False`` and a warning is issued if set to True.
        overwrite : bool
            if True, the metadata is re-downloaded even if already present.
        verbosity : int
            controls the verbosity of logging messages.
        """
        # UK-Flow15 currently corresponds to an ESSD preprint (essd-2026-152)
        # that is still under open review; the final accepted dataset may differ.
        warnings.warn(
            "UK-Flow15 corresponds to a preprint "
            "(https://doi.org/10.5194/essd-2026-152) which is under open review, "
            "not the final peer-reviewed dataset. The data may change in the "
            "accepted version; verify before using for critical work.",
            UserWarning)

        if to_netcdf:
            warnings.warn(
                "UK-Flow15 is a 15-min dataset of 1369 stations spanning "
                "1948-2023 (~64 GB); consolidating it into a single netCDF is "
                "not supported. Ignoring to_netcdf=True.", UserWarning)

        super().__init__(path=path, timestep="15min", to_netcdf=False,
                         overwrite=overwrite, verbosity=verbosity, **kwargs)

        # heavy attributes are loaded lazily and cached on first use
        self._static_df = None
        self._qc_cache = {}
        self._start = None
        self._end = None
        # stations whose flow file has already been refreshed this session
        # (so overwrite=True refreshes each file once, not on every read), and a
        # one-shot flag for the "downloading on demand" heads-up
        self._refreshed_stations = set()
        self._warned_lazy_download = False

        os.makedirs(self.path, exist_ok=True)
        os.makedirs(self.flow_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)

        self._download_metadata()

        # bounding box (UK) used by plot_stations
        self.bbox = {"llcrnrlat": 49.5, "urcrnrlat": 61.5,
                     "llcrnrlon": -8.5, "urcrnrlon": 2.0}
        self.parallels = np.arange(50, 61, 2)
        self.meridians = np.arange(-8, 2, 2)

    @property
    def flow_dir(self) -> os.PathLike:
        """directory holding the per-station 15-min flow csv files"""
        return os.path.join(self.path, '1_RiverFlowStations')

    @property
    def meta_dir(self) -> os.PathLike:
        """directory holding the (small) metadata csv files"""
        return os.path.join(self.path, '2_metadata')

    def _download_metadata(self):
        """downloads the small metadata csv files (only the missing ones)."""
        for fname, subdir in _META_FILES.items():
            fpath = os.path.join(self.meta_dir, fname)
            if self.overwrite and os.path.exists(fpath):
                os.remove(fpath)
            if not os.path.exists(fpath):
                download(url=f"{DATASTORE}/2_metadata/{subdir}/{fname}",
                         outdir=self.meta_dir, fname=fname,
                         verbosity=self.verbosity)

    @property
    def boundary_file(self) -> os.PathLike:
        raise NotImplementedError(
            "UK-Flow15 does not distribute catchment boundaries.")

    @property
    def static_map(self) -> Dict[str, str]:
        return {
            'Catchment_Area': catchment_area(),
            'Missing_values_%': 'missing_pct',
        }

    @property
    def dyn_map(self) -> Dict[str, str]:
        return {'value': observed_streamflow_cms()}

    @property
    def start(self) -> pd.Timestamp:
        if self._start is None:
            s = pd.to_datetime(self._static_data()['Start_date'],
                               format='%d/%m/%Y %H:%M')
            self._start = s.min()
        return self._start

    @property
    def end(self) -> pd.Timestamp:
        if self._end is None:
            e = pd.to_datetime(self._static_data()['End_date'],
                               format='%d/%m/%Y %H:%M')
            self._end = e.max()
        return self._end

    @property
    def dynamic_features(self) -> List[str]:
        # declared explicitly so that reading the column names does not trigger
        # the download of a (large) station file
        return [observed_streamflow_cms(), 'resolution', 'flag']

    @property
    def static_features(self) -> List[str]:
        return self._static_data().columns.tolist()

    def stations(self) -> List[str]:
        # .tolist() returns a fresh list, so the cached index cannot be mutated
        return self._static_data().index.tolist()

    @property
    def _area_name(self) -> str:
        return catchment_area()

    @property
    def _coords_name(self) -> List[str]:
        return [gauge_latitude(), gauge_longitude()]

    @property
    def _q_name(self) -> str:
        return observed_streamflow_cms()

    def _read_static(self) -> pd.DataFrame:
        """reads the station-identification metadata (00_station_id_meta.csv)
        and returns it as a per-station static table with standardised names
        for area and coordinates. The British National Grid easting/northing
        are retained and additionally converted to WGS84 lat/long."""
        fpath = os.path.join(self.meta_dir, '00_station_id_meta.csv')
        df = pd.read_csv(fpath)

        # NRFA station numbers as (non zero-padded) strings, matching CAMELS_GB
        df['station_id'] = df['station_id'].astype(int).astype(str)
        df = df.set_index('station_id')

        # OSGB36 (EPSG:27700) easting/northing -> WGS84 (EPSG:4326) lat/long.
        # verified against pyproj with a max error of ~4 mm.
        lat, lon = osgb36_to_wgs84(df['Easting'].values.astype(float),
                                   df['Northing'].values.astype(float))
        df[gauge_latitude()] = lat
        df[gauge_longitude()] = lon

        df = df.rename(columns=self.static_map)
        return df

    def _static_data(self) -> pd.DataFrame:
        if self._static_df is None:
            df = self._read_static()
            self._check_duplicates(df)
            self._static_df = df
        return self._static_df

    def _check_duplicates(self, df: pd.DataFrame):
        """warns (but does not exclude) about stations that share a gauge name
        (River + Location) and/or rounded coordinates. Runs once, vectorised
        over the 1369-row metadata, so it does not slow fetching."""
        names = (df['River'].astype(str).str.strip() + '|' +
                 df['Location'].astype(str).str.strip())
        dup_names = set(names[names.duplicated(keep=False)].index)

        coords = pd.Series(
            list(zip(df[gauge_latitude()].round(3),
                     df[gauge_longitude()].round(3))),
            index=df.index)
        dup_coords = set(coords[coords.duplicated(keep=False)].index)

        n = len(dup_names | dup_coords)
        if n and self.verbosity:
            warnings.warn(
                f"{n} stations appear to be potential duplicates (matching "
                f"gauge name and/or rounded coordinates). They are retained, "
                f"not excluded. See qc_metadata('duplicates') for the dataset's "
                f"own duplicate flags.", UserWarning)

    def _read_stn_dyn(self, stn: str) -> pd.DataFrame:
        """reads the 15-min time series of a single station. The station's flow
        file is downloaded on first access and then read from disk. The source
        columns are returned unchanged except that ``value`` is renamed to the
        standardised ``q_cms_obs``; ``resolution`` and ``flag`` (the QC code,
        kept as a zero-padded string) are provenance columns."""
        stem = str(stn).zfill(6)
        fpath = os.path.join(self.flow_dir, f"{stem}.csv")

        # honor overwrite: refresh a cached station file once per session. The
        # stale file is removed first so download() writes a genuinely fresh
        # copy rather than appending a '1' to the name.
        if self.overwrite and stem not in self._refreshed_stations \
                and os.path.exists(fpath):
            os.remove(fpath)

        if not os.path.exists(fpath):
            if not self._warned_lazy_download:
                warnings.warn(
                    "UK-Flow15 station flow files are downloaded on demand "
                    "(10-90 MB each; the full archive is ~64 GB) and cached in "
                    f"{self.flow_dir}. Use download_all_stations() to pre-fetch.",
                    UserWarning)
                self._warned_lazy_download = True
            _download_station_file(stem, self.flow_dir,
                                   verbosity=max(self.verbosity - 1, 0))
        self._refreshed_stations.add(stem)

        # flag read as str to preserve meaningful leading zeros; datetime read
        # as str then parsed with an explicit format (fast and version-safe);
        # value/resolution read at self.fp (float32 by default) to halve memory.
        # Downcasting is safe: float32 represents 3-decimal values exactly up to
        # ~16000 m3/s (mantissa 2**24), far above any UK 15-min river flow (peak
        # flow is not driven by catchment size, but even the flashiest gauges
        # stay in the low thousands of m3/s). Verified float32 == float64 to 3
        # decimals over the full 1.4M-value record of the largest-catchment
        # station; every observed resolution value is likewise well within
        # 2**24. Pass float_precision=numpy.float64 to keep double precision.
        df = pd.read_csv(
            fpath,
            usecols=['datetime', 'value', 'resolution', 'flag'],
            dtype={'datetime': str, 'flag': str, 'value': self.fp,
                   'resolution': self.fp},
        )
        df.index = pd.to_datetime(df.pop('datetime'), format='%Y-%m-%d %H:%M:%S')
        df.index.name = 'time'
        df.rename(columns=self.dyn_map, inplace=True)
        df.columns.name = 'dynamic_features'
        return df

    def q_mm(self, stations: Union[str, List[str]] = "all") -> pd.DataFrame:
        # 9 UK-Flow15 stations have no catchment area in the source metadata;
        # for those the cms->mm conversion is undefined and yields NaN. Warn
        # unconditionally so this is never a silent substitution.
        stns = validate_attributes(stations, self.stations(), 'stations')
        area = self.area(stns)
        missing = [s for s in stns if bool(pd.isna(area.get(s)))]
        if missing:
            warnings.warn(
                f"{len(missing)} of the requested stations have no catchment "
                f"area in the source metadata (e.g. {sorted(missing)[:5]}); "
                f"their q_mm is undefined and returned as NaN.", UserWarning)
        return super().q_mm(stations)

    def qc_metadata(self, name: str = 'traditional_qc') -> pd.DataFrame:
        """
        Returns one of the quality-control / traceability metadata tables
        unchanged (a faithful raw accessor).

        Parameters
        ----------
        name : str
            which table to return. One of:

                - ``'common_sense_anomalies'`` : manually reviewed anomalies
                  (one row per anomaly, 327 rows)
                - ``'uk_products'`` : comparison against NRFA daily/AMAX/POT
                  products (one row per station)
                - ``'traditional_qc'`` : counts of automatically detected
                  anomalies per station (spikes, drops, truncations, ...)
                - ``'high_flows_qc'`` : high-flow quality-control results per
                  station
                - ``'resolution'`` : per-station counts of steps by native
                  resolution and gap statistics
                - ``'duplicates'`` : stations flagged by the authors as
                  duplicates (66 rows)

        Returns
        -------
        pd.DataFrame

        Examples
        --------
        >>> from aqua_fetch import UKFlow15
        >>> dataset = UKFlow15()
        >>> dataset.qc_metadata('traditional_qc').shape
        (1369, 9)
        >>> dataset.qc_metadata('duplicates').shape
        (66, 3)
        """
        if name not in _QC_TABLES:
            raise ValueError(
                f"name must be one of {list(_QC_TABLES)}, got {name!r}")
        if name not in self._qc_cache:
            fpath = os.path.join(self.meta_dir, _QC_TABLES[name])
            self._qc_cache[name] = pd.read_csv(fpath)
        return self._qc_cache[name].copy()

    def download_all_stations(
            self,
            stations: Union[str, List[str]] = 'all',
            processes: int = None,
            overwrite: bool = False,
    ):
        """
        Pre-downloads the per-station flow files (which are otherwise fetched
        lazily on first access). The full archive is ~64 GB.

        Parameters
        ----------
        stations : str/list
            station ids to download. Default ``'all'`` downloads every station.
        processes : int
            number of parallel download workers. Defaults to ``self.processes``
            or a sensible number of cpus.
        overwrite : bool
            if True, re-download files that are already present.
        """
        stations = validate_attributes(stations, self.stations(), 'stations')
        todo = []
        for stn in stations:
            stem = str(stn).zfill(6)
            fp = os.path.join(self.flow_dir, f"{stem}.csv")
            if overwrite and os.path.exists(fp):
                os.remove(fp)
            if not os.path.exists(fp):
                todo.append(stem)

        if not todo:
            if self.verbosity:
                print("all requested station files are already present")
            return

        warnings.warn(
            f"downloading {len(todo)} station flow files from {DATASTORE}; "
            f"the complete archive is ~64 GB.", UserWarning)

        cpus = processes or self.processes or min(get_cpus(), 16)
        if cpus == 1 or len(todo) < 2:
            for stem in todo:
                _download_station_file(stem, self.flow_dir,
                                       verbosity=self.verbosity)
        else:
            fn = partial(_download_station_file, flow_dir=self.flow_dir,
                         verbosity=0)
            with cf.ProcessPoolExecutor(cpus) as executor:
                list(executor.map(fn, todo))

        if self.verbosity:
            print(f"downloaded {len(todo)} station files to {self.flow_dir}")
