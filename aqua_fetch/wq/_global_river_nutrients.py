
__all__ = ["GlobalRiverNutrients"]

import os
import warnings
import concurrent.futures as cf
from functools import partial
from typing import Union, List, Dict

import numpy as np
import pandas as pd

from .._datasets import Datasets
from ..utils import download, get_cpus
from ._map import tot_N, tot_P
from ..rr._map import observed_streamflow_cms


# figshare article that always resolves to the latest published version of the
# dataset of Peters et al., 2026 (https://doi.org/10.25400/lincolnuninz.30546371)
ARTICLE_ID = 30546371
API_URL = f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"

# user facing parameter -> (file prefix on figshare, site-id prefix,
#                           standardised value-column name)
_PARAMS = {
    'TN':         ('TN',         'WQ', tot_N()),                   # mg/l
    'TP':         ('TP',         'WQ', tot_P()),                   # mg/l
    'streamflow': ('Streamflow', 'S',  observed_streamflow_cms()),  # m3/s
}


def _read_site_csv(fpath: str, value_name: str, source: bool) -> pd.DataFrame:
    """
    Reads a single per-site csv (``date,value`` for the harmonised data, or
    ``WQ.source,WQ.ID,date,value`` for the raw source data) and returns a
    :obj:`pandas.DataFrame` indexed by date.

    ``value`` is read at full ``float64`` precision -- the measurement column is
    never downcast (a ``float64 -> float32`` cast can silently lose precision or
    overflow, so it is avoided for every observation).
    """
    if source:
        # ids/source read as str so a numeric-looking id can never lose leading
        # zeros; the measurement column stays at full float64 precision
        df = pd.read_csv(fpath, dtype={'value': 'float64',
                                       'WQ.source': str, 'S.source': str,
                                       'WQ.ID': str, 'S.ID': str})
        df.index = pd.to_datetime(df.pop('date'), format='%Y-%m-%d')
    else:
        df = pd.read_csv(fpath, usecols=['date', 'value'],
                         dtype={'value': 'float64'})
        df.index = pd.to_datetime(df.pop('date'), format='%Y-%m-%d')
        df.columns = [value_name]
    df.index.name = 'time'
    return df


def _read_site_worker(fname: str, data_dir: str, value_name: str, source: bool):
    """
    Module-level (picklable) worker used by the process pool. Kept at module
    level, and given only small string arguments, so that handing it to a
    :obj:`concurrent.futures.ProcessPoolExecutor` does not pickle ``self`` (and
    with it the cached heavy metadata tables) to every worker.
    """
    stn = fname[:-4]  # strip '.csv'
    return stn, _read_site_csv(os.path.join(data_dir, fname), value_name, source)


class GlobalRiverNutrients(Datasets):
    """
    A comprehensive global dataset of **observed** river total nitrogen (TN) and
    total phosphorus (TP) concentrations and streamflow, harmonised from
    seventeen national and international databases (GEMStat, GRQA, USGS, ECCC,
    Waterbase, Naïades, LAWA, HYDAT, NRFA, CAMELS-BR, ...) following
    `Peters et al., 2026 <https://doi.org/10.1038/s41597-026-07625-1>`_ .
    The data is downloaded from the associated
    `figshare repository <https://doi.org/10.25400/lincolnuninz.30546371>`_ .

    Only the **observational** measurements are exposed by this class:

        - ``TN``         : total nitrogen concentration in mg/l (``tot_N_mg/l``)
        - ``TP``         : total phosphorus concentration in mg/l (``tot_P_mg/l``)
        - ``streamflow`` : river discharge in m3/s (``q_cms_obs``)

    The three variables are each measured at their own set of monitoring sites
    (TN and TP sites use ``WQ`` ids, streamflow sites use ``S`` ids; the id
    spaces are independent). The paper additionally derives daily
    concentrations with a WRTDS model and globally extrapolated loads/yields
    with random forests -- these **modelled/derived** products are intentionally
    neither downloaded nor exposed here.

    For each variable two levels are available:

        - the **harmonised** product (default): the deduplicated list of sites
          (co-located sites within 200 m whose overlapping values agreed to
          within 0.1 % were combined by the authors). This is the analysis-ready
          observational product.
        - the **source** product (``source=True``): the faithful raw per-site
          files exactly as extracted from each source database, retaining the
          ``source``/``ID`` provenance columns and the (pre-deduplication)
          duplicate sites.

    Sites that could not be redistributed for licensing reasons (the UK NRFA and
    some GEMStat/GRDC sites) are absent from the data files but are listed, with
    their original database ids, by :meth:`unavailable_stations` so they can be
    fetched from the original repository.

    Approximate counts of available (redistributable) harmonised sites, the
    observations available per site and the covered period (all derived from the
    data, not hardcoded):

        ========== =============== =============== ========= ===========
        variable   available sites median obs/site total obs period
        ========== =============== =============== ========= ===========
        TN         19,100          35              1.23 M    1900 - 2024
        TP         40,279          37              3.22 M    1900 - 2025
        streamflow 27,497          8,294           278 M     1806 - 2024
        ========== =============== =============== ========= ===========

    Observations per site are strongly right-skewed (a few very long records pull
    the mean to roughly twice the median for TN/TP), so the **median** number of
    observations per site is shown rather than the mean; the exact per-site count
    is the ``n.days`` column of :meth:`metadata` (equal to the number of rows in
    that site's file). Water-quality sites are typically sampled monthly, so their
    counts are in the tens-to-hundreds; streamflow is daily, hence thousands.

    Units follow the source dataset: concentrations in mg/l, streamflow in m3/s.
    No catchment boundaries are distributed.

    .. note::
        To keep initialisation quick, only the small (~7 MB) per-site metadata
        is downloaded when the class is instantiated. The per-site time-series
        of a variable are bundled in a single zip that is downloaded and
        extracted on first access to that variable. The TN (~11 MB) and TP
        (~25 MB) archives are small, but the **streamflow archive is ~1.5 GB**
        (27,497 daily-discharge files), so only fetch streamflow when needed.

        Fetching is fast once the archive is extracted: reading every available
        site of a variable takes about 4 s for TN (19,100 sites, 1.23 M
        observations) and 13 s for TP (40,279 sites, 3.22 M observations) on 8
        processes; a handful of sites is essentially instant.

    Examples
    --------
    >>> from aqua_fetch import GlobalRiverNutrients
    >>> ds = GlobalRiverNutrients()
    ... # available variables
    >>> ds.parameters()
    ['TN', 'TP', 'streamflow']
    ... # number of available total-nitrogen sites
    >>> len(ds.stations('TN'))
    19100
    ... # number of available total-phosphorus sites
    >>> len(ds.stations('TP'))
    40279
    ... # fetch the total-nitrogen series of two sites (downloads TN archive once)
    >>> data = ds.fetch('TN', ['WQ000002', 'WQ000004'])
    >>> data['WQ000002'].shape
    (248, 1)
    >>> data['WQ000002'].columns.tolist()
    ['tot_N_mg/l']
    ... # a single site as a DataFrame
    >>> ds.fetch_stn('TN', 'WQ000004').shape
    (159, 1)
    ... # WGS84 coordinates of the total-phosphorus sites
    >>> ds.stn_coords('TP').shape
    (40279, 2)
    ... # per-site metadata (coordinates, period, n. observations, availability)
    >>> ds.metadata('TN').shape
    (19424, 9)
    """

    url = "https://doi.org/10.25400/lincolnuninz.30546371"

    def __init__(
            self,
            path: Union[str, os.PathLike] = None,
            overwrite: bool = False,
            verbosity: int = 1,
            **kwargs
    ):
        """
        Parameters
        ----------
        path : str
            If the data is already downloaded then provide the complete path to
            it. If None, the data is downloaded into the default directory. Only
            the metadata is downloaded at initialisation; the per-site time
            series are downloaded on first access.
        overwrite : bool
            if True, re-downloads (and re-extracts) even if the data is already
            present.
        verbosity : int
            controls the amount of printed information.
        """
        super().__init__(path=path, overwrite=overwrite, verbosity=verbosity,
                         **kwargs)

        os.makedirs(self.path, exist_ok=True)

        # heavy attributes are loaded lazily and cached on first use
        self._files_map = None      # figshare {name: download_url}
        self._meta_cache = {}       # (parameter, source) -> DataFrame
        self._start = None
        self._end = None
        self._dup_checked = set()   # parameters whose duplicate check has run
        self._warned_streamflow = False
        self._refreshed = set()     # files/dirs already refreshed this session
        self._verified = set()      # (parameter, source) dirs checked for completeness

        # metadata is small; download it eagerly so that stations(), stn_coords(),
        # start/end and metadata() work without pulling any per-site archive. This
        # also verifies at construction time that the download machinery works.
        for p in _PARAMS:
            self._download_file(self._meta_name(p, source=False))

    # ------------------------------------------------------------------ #
    # figshare download helpers
    # ------------------------------------------------------------------ #
    def _figshare_files(self) -> Dict[str, str]:
        """{filename: download_url} for the latest version, fetched once."""
        if self._files_map is None:
            import requests
            resp = requests.get(API_URL, timeout=30)
            resp.raise_for_status()
            self._files_map = {f['name']: f['download_url']
                               for f in resp.json()['files']}
        return self._files_map

    def _download_file(self, fname: str) -> str:
        """
        Downloads a single figshare file into ``self.path`` if it is not already
        present (or if ``overwrite``). Honours ``overwrite`` by removing the
        stale file first so the download helper writes a genuinely fresh copy
        rather than a ``name1`` sidecar. Returns the local path.
        """
        fpath = os.path.join(self.path, fname)
        # honour overwrite, but refresh each file at most once per session (so a
        # repeated aux-table call does not re-download). The stale file is removed
        # first so download() writes a fresh copy rather than a 'name1' sidecar.
        if self.overwrite and fname not in self._refreshed and os.path.exists(fpath):
            os.remove(fpath)
        self._refreshed.add(fname)
        if not os.path.exists(fpath):
            url = self._figshare_files()[fname]
            download(url=url, outdir=self.path, fname=fname,
                     verbosity=self.verbosity)
        return fpath

    def _data_dir(self, parameter: str, source: bool) -> str:
        """
        Ensures the per-site archive for ``parameter`` is downloaded and
        extracted, and returns the directory that holds the per-site csv files.

        The "already present" decision is gated on the presence of the
        **extracted** directory (not the archive) so that ``remove_zip=True``
        does not force a re-download on the next run.
        """
        prefix = _PARAMS[parameter][0]
        folder = (f"{prefix}.1_-_Source_Data" if source
                  else f"{prefix}.2_-_Harmonised")
        zip_name = folder + ".zip"
        out_dir = os.path.join(self.path, folder)

        if self.overwrite and folder not in self._refreshed and os.path.isdir(out_dir):
            import shutil
            shutil.rmtree(out_dir)
        self._refreshed.add(folder)

        if not os.path.isdir(out_dir):
            if parameter == 'streamflow' and not self._warned_streamflow:
                warnings.warn(
                    "the streamflow per-site archive is large (~1.5 GB "
                    "harmonised / ~1.7 GB source, tens of thousands of "
                    "daily-discharge files) and is being downloaded now; this "
                    "can take a while.", UserWarning)
                self._warned_streamflow = True
            zpath = self._download_file(zip_name)
            self._extract(zpath)
            self.maybe_remove_zip_files()

        self._check_complete(parameter, source, out_dir)
        return out_dir

    def _check_complete(self, parameter: str, source: bool, out_dir: str):
        """
        Warns loudly (once) if the extracted directory holds fewer per-site
        files than the dataset's manifest expects (e.g. an interrupted
        extraction). Completeness is judged against the metadata's count of
        available sites, not against whatever happens to be on disk.
        """
        key = (parameter, source)
        if key in self._verified:
            return
        self._verified.add(key)
        with os.scandir(out_dir) as it:
            n_files = sum(1 for f in it if f.name.endswith('.csv'))
        if source:
            avail = self._meta(parameter, source=True)
            expected = int((avail[self._avail_col(parameter)] == 1).sum())
        else:
            expected = len(self.stations(parameter))
        if n_files < expected:
            warnings.warn(
                f"only {n_files} of the expected {expected} {parameter} "
                f"{'source' if source else 'harmonised'} per-site files are "
                f"present in {out_dir}; the extraction may be incomplete. "
                f"Re-run with overwrite=True to re-download.", UserWarning)

    def _extract(self, zip_path: str):
        """extracts a per-site archive (whose entries live under a top-level
        folder matching the archive name) directly into ``self.path``."""
        import zipfile
        if self.verbosity:
            print(f"extracting {os.path.basename(zip_path)}")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(self.path)

    # ------------------------------------------------------------------ #
    # metadata
    # ------------------------------------------------------------------ #
    @staticmethod
    def _meta_name(parameter: str, source: bool) -> str:
        prefix = _PARAMS[parameter][0]
        return f"{prefix}.{'1' if source else '2'}_meta.csv"

    @staticmethod
    def _id_col(parameter: str) -> str:
        return f"{_PARAMS[parameter][1]}.ID"

    def _meta(self, parameter: str, source: bool = False) -> pd.DataFrame:
        """cached loader of a per-site metadata table (indexed by site id)."""
        assert parameter in _PARAMS, (
            f"unknown parameter {parameter!r}; use one of {list(_PARAMS)}")
        key = (parameter, source)
        if key not in self._meta_cache:
            fpath = self._download_file(self._meta_name(parameter, source))
            df = pd.read_csv(fpath, parse_dates=['start', 'end'])
            df = df.set_index(self._id_col(parameter))
            self._meta_cache[key] = df
            if not source:
                self._check_duplicates(parameter, df)
        return self._meta_cache[key]

    def _avail_col(self, parameter: str) -> str:
        return f"{_PARAMS[parameter][1]}.available"

    @staticmethod
    def _validate_stns(stations, valid) -> List[str]:
        """
        Resolves a ``stations`` argument to a fresh list of ids. The ``"all"``
        sentinel is short-circuited (O(n)); any explicit selection is
        materialised (so a one-shot generator is not silently drained) and
        validated against a ``set`` (O(k)). On an unknown id a concise
        ``ValueError`` naming only the offending ids is raised.

        This deliberately does not delegate to ``validate_attributes``: over a
        list that helper is O(n*m) and, on a bad id, prints the *entire* allowed
        set to stdout before raising -- here that is tens of thousands of ids
        (hundreds of KB) for a single typo.
        """
        if isinstance(stations, str):
            if stations == 'all':
                return list(valid)
            stations = [stations]
        elif not hasattr(stations, '__iter__'):
            raise TypeError(
                "stations must be a site id (str), a list of ids or 'all', "
                f"not {type(stations).__name__}")
        stations = list(stations)  # materialise before iterating more than once
        valid_set = valid if isinstance(valid, set) else set(valid)
        missing = [s for s in stations if s not in valid_set]
        if missing:
            extra = '' if len(missing) <= 10 else f" (+{len(missing) - 10} more)"
            raise ValueError(
                f"{len(missing)} unknown station id(s): {missing[:10]}{extra}")
        return stations

    def _check_duplicates(self, parameter: str, meta: pd.DataFrame):
        """
        Warns (but does not exclude) about available sites that share **exactly**
        the same coordinates (rounded to ~0.1 m). The harmonised product is
        already deduplicated by the authors (co-located sites within 200 m with
        agreeing values were merged), but co-located sites whose values
        disagreed were kept separate and are surfaced here. Runs once per
        parameter, vectorised, so it does not slow fetching. The dataset's own
        duplicate groupings are available from :meth:`duplicates`.
        """
        if parameter in self._dup_checked:
            return
        self._dup_checked.add(parameter)
        prefix = _PARAMS[parameter][1]
        avail = meta[meta[self._avail_col(parameter)] == 1]
        coords = pd.Series(list(zip(avail[f'{prefix}.lat'].round(6),
                                    avail[f'{prefix}.lon'].round(6))),
                           index=avail.index)
        n = int(coords.duplicated(keep=False).sum())
        if n and self.verbosity:
            warnings.warn(
                f"{n} available {parameter} sites share their exact coordinates "
                f"with another site (co-located sites whose values disagreed and "
                f"so were not merged by the authors). They are retained, not "
                f"excluded; see duplicates('{parameter}') for the dataset's own "
                f"duplicate list.", UserWarning)

    def metadata(self, parameter: str = 'TN', source: bool = False) -> pd.DataFrame:
        """
        Returns the per-site metadata table for ``parameter`` (a copy).

        For the harmonised product the columns are the site id (index),
        ``{p}.lat``, ``{p}.lon`` (WGS84), ``region`` (continent), ``start``,
        ``end``, ``duration`` (years), ``sum`` (years of high-frequency
        measurement), ``n.days`` (days with a measurement) and ``{p}.available``
        (1 if the site's data is redistributable). The ``source`` table
        additionally carries the ``{p}.source`` database name. Note that the
        metadata lists all sites, including the ``{p}.available == 0`` sites for
        which no data file is distributed.

        Parameters
        ----------
        parameter : str
            one of ``'TN'``, ``'TP'`` or ``'streamflow'``.
        source : bool
            if True, returns the raw source metadata (pre-deduplication, with
            provenance) instead of the harmonised metadata. Its index is the
            original database id (``{p}.ID``); the corresponding per-site file
            and ``fetch(..., source=True)`` id is the stem ``{source}_{ID}``
            (combine the ``{p}.source`` column with the index).

        Examples
        --------
        >>> from aqua_fetch import GlobalRiverNutrients
        >>> ds = GlobalRiverNutrients()
        >>> ds.metadata('TN').shape
        (19424, 9)
        """
        return self._meta(parameter, source).copy()

    def num_obs(self, parameter: str = 'TN') -> pd.Series:
        """
        Returns the number of observations available at each (redistributable)
        site of ``parameter`` -- the ``n.days`` metadata column, which equals the
        number of rows in that site's per-site file. Use it to summarise how much
        data is available per site, e.g. ``.median()``, ``.sum()``,
        ``.describe()``.

        Examples
        --------
        >>> from aqua_fetch import GlobalRiverNutrients
        >>> ds = GlobalRiverNutrients()
        >>> n = ds.num_obs('TN')
        >>> len(n)
        19100
        >>> int(n.median())
        35
        >>> int(n.sum())
        1232518
        """
        meta = self._meta(parameter)
        return meta.loc[meta[self._avail_col(parameter)] == 1, 'n.days'].copy()

    def parameters(self) -> List[str]:
        """Returns the list of observed variables in the dataset."""
        return list(_PARAMS)

    def stations(self, parameter: str = 'TN', available_only: bool = True) -> List[str]:
        """
        Returns the list of site ids for ``parameter``.

        Parameters
        ----------
        parameter : str
            one of ``'TN'``, ``'TP'`` or ``'streamflow'``.
        available_only : bool
            if True (default) only the sites whose data is redistributable (and
            therefore fetchable) are returned; if False, every site in the
            metadata is returned (including licensing-excluded sites).

        Examples
        --------
        >>> from aqua_fetch import GlobalRiverNutrients
        >>> ds = GlobalRiverNutrients()
        >>> len(ds.stations('TN'))
        19100
        """
        meta = self._meta(parameter)
        if available_only:
            meta = meta[meta[self._avail_col(parameter)] == 1]
        return meta.index.tolist()

    def stn_coords(
            self,
            parameter: str = 'TN',
            stations: Union[str, List[str]] = "all",
    ) -> pd.DataFrame:
        """
        Returns WGS84 coordinates of the ``parameter`` sites as a DataFrame with
        columns ``lat`` and ``long`` (indexed by site id).

        Examples
        --------
        >>> from aqua_fetch import GlobalRiverNutrients
        >>> ds = GlobalRiverNutrients()
        >>> ds.stn_coords('TN').shape
        (19100, 2)
        >>> ds.stn_coords('TN', 'WQ000002')
                      lat      long
        WQ.ID
        WQ000002  55.9254 -130.0355
        """
        prefix = _PARAMS[parameter][1]
        meta = self._meta(parameter)
        meta = meta[meta[self._avail_col(parameter)] == 1]
        coords = meta[[f'{prefix}.lat', f'{prefix}.lon']].copy()
        coords.columns = ['lat', 'long']
        stns = self._validate_stns(stations, coords.index)
        return coords.loc[stns]

    @property
    def start(self) -> pd.Timestamp:
        """earliest observation date across all three variables (from the data)."""
        if self._start is None:
            self._start = min(self._meta(p)['start'].min() for p in _PARAMS)
        return self._start

    @property
    def end(self) -> pd.Timestamp:
        """latest observation date across all three variables (from the data)."""
        if self._end is None:
            self._end = max(self._meta(p)['end'].max() for p in _PARAMS)
        return self._end

    # ------------------------------------------------------------------ #
    # time series
    # ------------------------------------------------------------------ #
    def fetch(
            self,
            parameter: str = 'TN',
            stations: Union[str, List[str]] = "all",
            source: bool = False,
            processes: int = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetches the observed time series of one or more sites.

        Parameters
        ----------
        parameter : str
            one of ``'TN'``, ``'TP'`` or ``'streamflow'``.
        stations : str or list of str
            the site id(s) to fetch. ``'all'`` fetches every available site
            (this reads tens of thousands of small files and is slow; for
            streamflow it also downloads the ~1.5 GB archive).
        source : bool
            if True, returns the raw source files (with ``source``/``ID``
            provenance columns and the original database dates) instead of the
            harmonised ``date,value`` files. Source site ids are the file stems
            ``{source}_{ID}`` (see ``stations(..., )`` on the source metadata).
        processes : int
            number of parallel reader processes. Defaults to ``self.processes``.
            Set to 1 to disable multiprocessing.

        Returns
        -------
        dict
            ``{site_id: DataFrame}`` where each DataFrame is indexed by date and
            has a single column named after the standardised variable (for the
            harmonised product) in the source units (mg/l or m3/s).

        Examples
        --------
        >>> from aqua_fetch import GlobalRiverNutrients
        >>> ds = GlobalRiverNutrients()
        >>> data = ds.fetch('TN', ['WQ000002', 'WQ000004'])
        >>> data['WQ000002'].shape
        (248, 1)
        """
        value_name = _PARAMS[parameter][2]
        data_dir = self._data_dir(parameter, source)

        if source:
            valid = [f[:-4] for f in os.listdir(data_dir) if f.endswith('.csv')]
        else:
            valid = self.stations(parameter)
        stns = self._validate_stns(stations, valid)

        files = [f"{s}.csv" for s in stns]
        cpus = processes if processes is not None else self.processes
        cpus = cpus or min(get_cpus(), 16)

        if cpus == 1 or len(files) < 32:
            results = {}
            for f in files:
                stn, df = _read_site_worker(f, data_dir, value_name, source)
                results[stn] = df
        else:
            fn = partial(_read_site_worker, data_dir=data_dir,
                         value_name=value_name, source=source)
            results = {}
            with cf.ProcessPoolExecutor(cpus) as ex:
                for stn, df in ex.map(fn, files):
                    results[stn] = df
        return results

    def fetch_stn(
            self,
            parameter: str = 'TN',
            station: str = None,
            source: bool = False,
    ) -> pd.DataFrame:
        """
        Fetches the observed time series of a single site as a DataFrame.

        Examples
        --------
        >>> from aqua_fetch import GlobalRiverNutrients
        >>> ds = GlobalRiverNutrients()
        >>> ds.fetch_stn('TN', 'WQ000004').shape
        (159, 1)
        """
        assert isinstance(station, str), "station must be a single site id (str)"
        value_name = _PARAMS[parameter][2]
        data_dir = self._data_dir(parameter, source)
        if source:
            valid = [f[:-4] for f in os.listdir(data_dir) if f.endswith('.csv')]
        else:
            valid = self.stations(parameter)
        assert station in valid, (
            f"{station} is not an available {parameter} site")
        return _read_site_csv(os.path.join(data_dir, f"{station}.csv"),
                              value_name, source)

    # ------------------------------------------------------------------ #
    # provenance / auxiliary tables
    # ------------------------------------------------------------------ #
    def unavailable_stations(self, parameter: str = 'TN') -> pd.DataFrame:
        """
        Returns the sites that are **not** redistributed for licensing reasons,
        with their original source-database ids and the harmonised id they were
        assigned, so they can be obtained from the original repository. Columns
        are ``{p}.source``, ``{p}.ID`` (original id) and ``{p}.ID2`` (harmonised
        id).

        Examples
        --------
        >>> from aqua_fetch import GlobalRiverNutrients
        >>> ds = GlobalRiverNutrients()
        >>> ds.unavailable_stations('TN').shape
        (326, 3)
        """
        prefix = _PARAMS[parameter][0]
        fpath = self._download_file(f"{prefix}.2_unavailable_ID.csv")
        return pd.read_csv(fpath)

    def duplicates(self, parameter: str = 'TN') -> pd.DataFrame:
        """
        Returns the dataset's own list of duplicate source sites (the pairings
        the authors used to deduplicate the harmonised product). Each row pairs
        two co-located source sites (columns suffixed ``_1`` for the second
        site) with the distance between them in metres. This is a faithful raw
        accessor and is downloaded on demand.

        Examples
        --------
        >>> from aqua_fetch import GlobalRiverNutrients
        >>> ds = GlobalRiverNutrients()
        >>> ds.duplicates('TN').shape
        (63227, 13)
        """
        prefix = _PARAMS[parameter][0]
        fpath = self._download_file(f"{prefix}.1_duplicates.csv")
        return pd.read_csv(fpath)

    def measurement_years(self, parameter: str = 'TN') -> pd.DataFrame:
        """
        Returns, for each harmonised site, its consecutive stretches of years
        with high-frequency measurement (one row per stretch). Downloaded on
        demand.

        Examples
        --------
        >>> from aqua_fetch import GlobalRiverNutrients
        >>> ds = GlobalRiverNutrients()
        >>> ds.measurement_years('TN').shape
        (15409, 7)
        """
        prefix = _PARAMS[parameter][0]
        fpath = self._download_file(f"{prefix}.2_years.csv")
        return pd.read_csv(fpath)

    def download_data(self, parameter: str = 'TN', source: bool = False):
        """
        Pre-downloads and extracts the per-site archive of ``parameter`` (which
        is otherwise fetched lazily on first access). The streamflow archive is
        ~1.5 GB.
        """
        self._data_dir(parameter, source)
        if self.verbosity:
            print(f"{parameter} ({'source' if source else 'harmonised'}) "
                  f"data is available in {self.path}")
