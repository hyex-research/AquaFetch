
import os
import shutil
import zipfile
import warnings
import functools
import concurrent.futures as cf
from collections.abc import Iterable
from typing import Union, List, Dict

import pandas as pd

from .._datasets import Datasets
from ..utils import validate_attributes, download, get_cpus

from .._backend import xarray as xr


# Zenodo record for Caravan-Qual (lite). We pin to the latest published
# version (v1.1, record 21337532, 2026-07-13) as required by the project
# conventions. The concept DOI is 10.5281/zenodo.17787065.
__all__ = ["CaravanQual"]

_RECORD = "21337532"
_BASE = f"https://zenodo.org/api/records/{_RECORD}/files"

# Reading constituent files in a process pool only pays off once there are
# enough bytes to amortise process start-up; below this the serial read wins.
_PARALLEL_READ_BYTES = 100 * 1024 * 1024

# Units that a constituent may legitimately carry in addition to the canonical
# unit of the data dictionary. These are spelling/case variants of the *same*
# quantity, so observations carrying them are kept as they are. Any other unit
# denotes a different quantity (or a different measurement method) and is
# dropped by :meth:`CaravanQual.fetch`.
#
# Deliberately NOT included: ``no/100ml`` for ``cfu/100ml``. A plain count
# (typically MPN) and colony-forming units are different enumeration methods,
# not a spelling variant, so the 125,620 ``no/100ml`` observations of EColi/FC/
# TotColi are treated as a foreign unit (dropped from the wide table, kept in
# :meth:`data`) rather than silently merged.
_UNIT_SYNONYMS = {
    "deg C": {"deg c", "cel"},
    "uS/cm": {"us/cm"},
    "pH": {"ph", "phunits"},
    # 'ug' is a truncated spelling of 'ug/l'. The only constituent that uses it
    # is Carbamazepine, whose 48 'ug' values (0.0011 - 5.27) lie inside the
    # range of its 56,733 'ug/l' values (3.5e-07 - 170), i.e. they are
    # concentrations, not masses.
    "ug/l": {"ug"},
}


def _constituent_path(csv_dir: str, parameter: str) -> str:
    return os.path.join(csv_dir, f"{parameter}.csv")


def _read_constituent_file(csv_dir: str, parameter: str) -> pd.DataFrame:
    """
    Reads one constituent csv, unchanged. Module-level (not a method) so that a
    process pool pickles only ``csv_dir`` (77 bytes) rather than a bound method,
    which would drag the cached 15 MB ``site_info`` along with ``self`` for every
    task (~1.5 GB of IPC across 100 constituents).
    """
    return pd.read_csv(
        _constituent_path(csv_dir, parameter),
        dtype={
            "wqms_id": str,
            "obs": "float64",
            "unit": str,
            "flag": str,
            "detection_limit": "float64",
            "imputation_method": str,
            "variable": str,
            "streamflow": "float64",
        },
        parse_dates=["dates"],
    )


def _date_bounds_file(csv_dir: str, parameter: str) -> tuple:
    """(min, max) ``dates`` of one constituent, as ISO strings. Module-level for
    the same pickling reason as :func:`_read_constituent_file`."""
    dates = pd.read_csv(_constituent_path(csv_dir, parameter), usecols=["dates"])["dates"]
    return dates.min(), dates.max()


def _reject_all_parameters(parameters):
    """
    Rejects the inputs that would materialise the whole corpus.

    ``None`` and ``'all'`` are both refused: ``validate_attributes`` expands
    ``'all'`` to the full list of 100 constituents, so guarding only on ``None``
    would let the single most likely input walk straight through.
    """
    if parameters is None or (isinstance(parameters, str) and parameters == "all"):
        raise ValueError(
            "`parameters` must name the constituent(s) to fetch; neither None "
            "nor 'all' is accepted. All 100 constituents are ~96 million rows "
            "(~25 GB) in memory. Use data() to stream one constituent at a time."
        )
    return


def _validate_ids(selection, allowed, name: str) -> List[str]:
    """
    Validates station/gauge ids against ``allowed``.

    This exists because :func:`aqua_fetch.utils.validate_attributes` checks
    membership against a *list*, which for the 151,859 stations of this dataset
    makes ``selection='all'`` an O(n^2) operation (~65 s). Here the ``all`` case
    short-circuits and the rest is checked against a set.
    """
    if isinstance(selection, str) and selection == "all":
        return list(allowed)

    if isinstance(selection, str):
        selection = [selection]
    elif not isinstance(selection, Iterable):
        raise TypeError(
            f"unknown type {type(selection)} for {name}; expected 'all', a "
            f"string or a list of strings")
    else:
        # materialise one-shot iterables (e.g. generators) so the membership
        # check below does not consume them, leaving nothing to return
        selection = list(selection)

    allowed_set = set(allowed)
    invalid = [s for s in selection if s not in allowed_set]
    if invalid:
        raise ValueError(
            f"invalid {name}: {invalid[:5]}{' ...' if len(invalid) > 5 else ''}")

    return list(selection)


class CaravanQual(Datasets):
    """
    Caravan-Qual (lite): a global integration of stream water quality into the
    Caravan large-sample-hydrology framework, following `Jones et al., 2026
    <https://doi.org/10.1038/s41597-026-07352-7>`_. ~96 million observations of
    **100 constituents** at **151,859 monitoring stations** (``wqms_id``)
    worldwide, spanning 1894-08-30 to 2025-12-15 (:attr:`start` / :attr:`end`).
    ("lite" is the Zenodo release used here: daily water quality, co-located
    streamflow and catchment attributes, but only *monthly* weather. A "full"
    release adding daily weather is hosted separately and is not downloaded by
    this class.)

    What accompanies each of the 151,859 stations:

    - streamflow: 59,724 (39%) are linked to a Caravan gauge and carry its
      co-located daily discharge (m3/s) per observation, NaN on dates the gauge
      has no record; see :meth:`linked_stations`.
    - catchment attributes: 151,229 (99.6%) map to a river segment (``LINKNO``)
      with HydroATLAS/GEOGLOWS attributes in the optional ``.zarr``; see
      :meth:`stn_attributes`.
    - catchment boundaries: none — the lite release ships no polygon geometry,
      hence no ``get_boundary`` here (unlike the ``rr`` Caravan classes).

    Downloaded from `zenodo <https://zenodo.org/records/21337532>`_. The water
    quality csv data needs only ``pandas``; the catchment/stream attributes are a
    separate ``.zarr`` store, downloaded only when ``attributes=True`` (needs
    ``xarray``+``zarr``). Attributes are *segment-level* — keyed to a reach of the
    GEOGLOWS/TDX-Hydro river network (``LINKNO``), which a station inherits — 205
    in all (197 catchment + 8 stream). The ``.zarr`` covers only 1980-01-01 to
    2025-09-30 (:attr:`zarr_start` / :attr:`zarr_end`), ~10% narrower than the csv.

    The linked gauges come from ``rr``-submodule collections; the ``gauge_id``
    prefix says which (mostly ``hysets`` 30,054 and ``grdc`` 13,521, plus
    ``camels*`` and ``lamah``), so the streamflow overlaps those classes. 1,258
    gauges share a name with another (the same station appearing in two
    collections): :meth:`duplicate_gauges` reports them (a warning fires at
    construction) but none are excluded.

    Performance (16 cpus, warm cache; scales with the constituent's file size,
    <1 MB to 554 MB for ``TEMP``): ``data('TEMP')`` ~8 s;
    ``fetch(parameters=p)`` over all stations ~1.6 s (Atrazine, 27 MB) to ~25 s
    (``TEMP``, 554 MB); all 100 constituents for one station ~24 s. :meth:`fetch`
    builds a per-station dict; for one constituent across all stations
    :meth:`data` returns a single DataFrame ~3x faster, and for repeated
    whole-station access the random-access ``.zarr`` beats these csv files.
    One-time download ~5 min (csv) / ~16 min (with ``attributes=True``).

    Examples
    --------
    >>> from aqua_fetch import CaravanQual
    >>> ds = CaravanQual(path='/path/to/data')
    ... # names of the 100 water quality constituents
    >>> len(ds.parameters)
    100
    >>> ds.parameters[:5]
    ['Amoxicillin', 'As-Dis', 'As-Tot', 'Atenolol', 'Atrazine']
    ... # unit of each constituent (faithful to the original data)
    >>> ds.parameter_units['TEMP'], ds.parameter_units['EC'], ds.parameter_units['NO3N']
    ('deg C', 'uS/cm', 'mg/l')
    ... # number of water quality monitoring stations
    >>> len(ds.stations())
    151859
    ... # number of Caravan streamflow gauges
    >>> len(ds.gauges())
    27073
    ... # faithful long-format table for a single constituent
    >>> temp = ds.data('TEMP')
    >>> temp.columns.tolist()
    ['wqms_id', 'dates', 'obs', 'unit', 'flag', 'detection_limit', 'imputation_method', 'streamflow']
    ... # station-centric wide table (date x constituents) for selected stations
    >>> data = ds.fetch(stations=['wqms_01200003'], parameters=['Atrazine', 'NO3N'])
    >>> data['wqms_01200003'].columns.tolist()
    ['Atrazine', 'NO3N']
    ... # coordinates of stations (wgs84)
    >>> coords = ds.stn_coords()
    >>> coords.shape
    (151859, 2)
    ... # all water quality stations linked to a Caravan gauge
    >>> len(ds.gauge_stations('grdc_4146360'))
    206
    """

    url = {
        "wqms-csv.zip": f"{_BASE}/wqms-csv.zip/content",
        "wqms_site_info.csv": f"{_BASE}/wqms_site_info.csv/content",
        "caravan_site_info.csv": f"{_BASE}/caravan_site_info.csv/content",
        "Caravan-Qual_zarr_variables.csv": f"{_BASE}/Caravan-Qual_zarr_variables.csv/content",
    }

    _zarr_url = {
        "Caravan-Qual_lite.zarr.zip": f"{_BASE}/Caravan-Qual_lite.zarr.zip/content",
    }

    def __init__(
            self,
            path=None,
            overwrite: bool = False,
            attributes: bool = False,
            verbosity: int = 1,
            **kwargs
    ):
        """
        Parameters
        ----------
        path : str, optional
            If the data is already downloaded then provide the complete
            path to it. If None, the data will be downloaded into the
            default directory. The data is downloaded only once; subsequent
            calls do not re-download unless ``overwrite`` is True.
        overwrite : bool, optional (default=False)
            If the data is already downloaded, set this to True to make a
            fresh download. The previously extracted directories are removed
            and re-extracted as well, so that the fresh download is actually
            reflected on disk.
        attributes : bool, optional (default=False)
            Whether to also download (~1.5 GB) and read the catchment
            attributes stored in the ``.zarr`` store. This requires the
            optional ``xarray`` and ``zarr`` packages. The core water quality
            functionality does not need this.
        verbosity : int, optional (default=1)
            Controls the amount of information printed.
        remove_zip : bool, optional (default=False)
            If True the downloaded ``.zip`` archives are deleted once they have
            been extracted, saving ~2.1 GB of disk.
        """
        super().__init__(path=path, verbosity=verbosity, overwrite=overwrite,
                         **kwargs)

        self._attributes = attributes
        self._site_info = None
        self._caravan_site_info = None
        self._temporal_extent = None

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self._download_files(self.url, overwrite=self.overwrite,
                             extracted={"wqms-csv.zip": self.csv_dir})

        # extract the water quality csv files (wqms-csv.zip -> wqms-csv/*.csv)
        self._maybe_extract("wqms-csv.zip", self.csv_dir, overwrite=self.overwrite)

        if attributes:
            self._download_files(
                self._zarr_url, overwrite=self.overwrite,
                extracted={"Caravan-Qual_lite.zarr.zip": self.zarr_dir})
            self._maybe_extract("Caravan-Qual_lite.zarr.zip", self.zarr_dir,
                                overwrite=self.overwrite)

        self.maybe_remove_zip_files()

        # cache the (expensive to recompute) look-ups
        variables = self.variables()
        wq = variables[variables['category'] == 'water_quality']
        expected = wq['variable'].tolist()
        self._parameter_units = dict(zip(wq['variable'], wq['units']))
        self._parameter_desc = dict(zip(wq['variable'], wq['description']))

        # The data dictionary is the authority on which constituents exist.
        # A mismatch against the extracted files means the extraction was
        # incomplete (e.g. interrupted), so warn instead of silently reporting
        # a truncated dataset.
        available = {f[:-4] for f in os.listdir(self.csv_dir) if f.endswith('.csv')}
        missing = [p for p in expected if p not in available]
        if missing:
            warnings.warn(
                f"{len(missing)} of {len(expected)} constituent files are missing "
                f"from {self.csv_dir} (e.g. {missing[:3]}). The extraction is "
                f"incomplete; re-initialise with overwrite=True to repair it.",
                stacklevel=2)
        self._parameters = [p for p in expected if p in available]

        self._stations = self.site_info().index.tolist()
        self._gauges = self.caravan_site_info().index.tolist()

        if self.verbosity > 0:
            self._warn_duplicate_gauges()

    # ------------------------------------------------------------------ #
    # downloading / extraction
    # ------------------------------------------------------------------ #
    def _download_files(
            self,
            urls: Dict[str, str],
            overwrite: bool = False,
            extracted: Dict[str, str] = None,
    ):
        """
        Downloads the given ``{filename: url}`` pairs into ``self.path``.

        Parameters
        ----------
        urls :
            mapping of file name to the url it is downloaded from.
        overwrite :
            if True the local copy is deleted first. This is required because
            :func:`aqua_fetch.utils.download` appends a ``1`` to the file name
            when the target already exists, which would leave the stale file in
            place and write the fresh download to a name nothing reads.
        extracted :
            optional mapping of file name to the directory that is extracted
            from it. If that directory is present the file is not downloaded,
            so that deleting the archives (``remove_zip=True``) does not cause
            a re-download on the next instantiation.
        """
        extracted = extracted or {}

        for fname, url in urls.items():
            fpath = os.path.join(self.path, fname)
            target = extracted.get(fname)

            if overwrite:
                # download() would otherwise save this as f"{fname}1"
                if os.path.exists(fpath):
                    if self.verbosity > 1:
                        print(f"Removing stale {fname} before re-downloading")
                    os.remove(fpath)
            else:
                if os.path.exists(fpath):
                    if self.verbosity > 1:
                        print(f"{fname} already exists at {self.path}")
                    continue
                if target is not None and os.path.isdir(target):
                    if self.verbosity > 1:
                        print(f"{fname} is already extracted into {target}")
                    continue

            if self.verbosity > 0:
                print(f"Downloading {fname} from {url}")
            download(url, outdir=self.path, fname=fname, verbosity=self.verbosity)
        return

    def _maybe_extract(self, zipname: str, target_dir: str, overwrite: bool = False):
        """
        extracts ``zipname`` into ``self.path`` unless ``target_dir`` exists.
        When ``overwrite`` is True any pre-existing ``target_dir`` is removed
        first, so that a fresh download is actually reflected on disk.
        """
        if os.path.isdir(target_dir):
            if not overwrite:
                if self.verbosity > 1:
                    print(f"{target_dir} already exists, skipping extraction")
                return
            if self.verbosity > 0:
                print(f"Removing pre-existing {target_dir} before re-extraction")
            shutil.rmtree(target_dir)

        zpath = os.path.join(self.path, zipname)
        if self.verbosity > 0:
            print(f"Extracting {zipname}")
        with zipfile.ZipFile(zpath, 'r') as zf:
            zf.extractall(self.path)
        return

    @property
    def csv_dir(self) -> str:
        """directory holding the per-constituent water quality csv files."""
        return os.path.join(self.path, "wqms-csv")

    @property
    def zarr_dir(self) -> str:
        """directory holding the (optional) catchment attributes zarr store."""
        return os.path.join(self.path, "Caravan-Qual_lite.zarr")

    # ------------------------------------------------------------------ #
    # metadata / look-ups
    # ------------------------------------------------------------------ #
    def variables(self) -> pd.DataFrame:
        """
        Reads ``Caravan-Qual_zarr_variables.csv``, the data dictionary that
        describes every variable (constituent, catchment attribute, weather
        forcing, ...) together with its ``units`` and ``description``.

        Returns
        -------
        pd.DataFrame
            with columns ``['variable', 'category', 'units', 'description']``
        """
        return pd.read_csv(os.path.join(self.path, "Caravan-Qual_zarr_variables.csv"))

    @property
    def parameters(self) -> List[str]:
        """names of the 100 water quality constituents in the dataset."""
        return list(self._parameters)

    @property
    def parameter_units(self) -> Dict[str, str]:
        """mapping of each constituent to its (original, unchanged) unit."""
        return {p: self._parameter_units[p] for p in self._parameters}

    @property
    def parameter_description(self) -> Dict[str, str]:
        """mapping of each constituent to its human readable description."""
        return {p: self._parameter_desc[p] for p in self._parameters}

    # The .zarr store is published on a fixed daily grid, which is narrower
    # than the observations in the csv files. These bound the zarr only.
    zarr_start = pd.Timestamp("1980-01-01")
    zarr_end = pd.Timestamp("2025-09-30")

    @property
    def start(self) -> pd.Timestamp:
        """
        date of the earliest observation in the dataset (1894-08-30).

        This is derived from the data rather than asserted, because ~10% of the
        observations predate the 1980-01-01 start of the ``.zarr`` grid (see
        :attr:`zarr_start`). The first access scans the ``dates`` column of all
        100 files (~15-30 s) and the result is cached for the session.
        """
        return self._extent()[0]

    @property
    def end(self) -> pd.Timestamp:
        """
        date of the latest observation in the dataset (2025-12-15). See
        :attr:`start` for why this is derived rather than hardcoded.
        """
        return self._extent()[1]

    def _extent(self) -> tuple:
        """min/max observation date over all constituents, computed once."""
        if getattr(self, "_temporal_extent", None) is None:
            if self.verbosity > 0:
                print(f"Scanning {len(self.parameters)} files for the temporal "
                      f"extent (once per session) ...")

            cpus = self.processes or min(get_cpus() - 2, 16)
            worker = functools.partial(_date_bounds_file, self.csv_dir)
            if cpus > 1:
                with cf.ProcessPoolExecutor(max_workers=cpus) as executor:
                    bounds = list(executor.map(worker, self.parameters))
            else:
                bounds = [worker(p) for p in self.parameters]

            # dates are ISO formatted, so lexicographic order is chronological
            lo = min(b[0] for b in bounds)
            hi = max(b[1] for b in bounds)
            self._temporal_extent = (pd.Timestamp(lo), pd.Timestamp(hi))
        return self._temporal_extent

    def site_info(self) -> pd.DataFrame:
        """
        Reads ``wqms_site_info.csv`` describing the water quality monitoring
        stations, including the linkage to the nearest Caravan streamflow gauge.

        Returns
        -------
        pd.DataFrame
            indexed by ``wqms_id`` with columns ``wqms_lat``, ``wqms_lon``,
            ``country_name``, ``hydrobasin_level12``, ``LINKNO``,
            ``merged_LINKNO``, ``gauge_id`` and ``gauge_distance_km``.
        """
        # cached because this 15 MB file is read by stn_coords, countries,
        # linked_stations, gauge_stations, fetch_by_gauge and stn_attributes
        if self._site_info is None:
            self._site_info = pd.read_csv(
                os.path.join(self.path, "wqms_site_info.csv"),
                index_col="wqms_id",
                dtype={"wqms_id": str, "gauge_id": str, "country_name": str},
            )
        return self._site_info.copy()

    def caravan_site_info(self) -> pd.DataFrame:
        """
        Reads ``caravan_site_info.csv`` describing the Caravan streamflow gauges.

        Returns
        -------
        pd.DataFrame
            indexed by ``gauge_id``.
        """
        if self._caravan_site_info is None:
            self._caravan_site_info = pd.read_csv(
                os.path.join(self.path, "caravan_site_info.csv"),
                index_col="gauge_id",
                dtype={"gauge_id": str, "country": str, "country_name": str},
            )
        return self._caravan_site_info.copy()

    def stations(self) -> List[str]:
        """returns the ids (``wqms_id``) of the 151,859 water quality stations."""
        return list(self._stations)

    def gauges(self) -> List[str]:
        """returns the ids (``gauge_id``) of the Caravan streamflow gauges."""
        return list(self._gauges)

    def countries(self) -> List[str]:
        """returns the names of countries with water quality stations."""
        return self.site_info()["country_name"].dropna().unique().tolist()

    def stn_coords(
            self,
            stations: Union[str, List[str]] = "all"
    ) -> pd.DataFrame:
        """
        Returns coordinates of water quality stations in wgs84 projection.

        Parameters
        ----------
        stations : str or list, optional
            id/ids of ``wqms_id`` station(s). Defaults to all stations.

        Returns
        -------
        pd.DataFrame
            with columns ``lat`` and ``long`` indexed by ``wqms_id``.

        Examples
        --------
        >>> ds = CaravanQual(path='/path/to/data')
        >>> ds.stn_coords().shape
        (151859, 2)
        >>> ds.stn_coords('wqms_00100001')
        """
        stations = _validate_ids(stations, self.stations(), "stations")
        coords = self.site_info().loc[stations, ["wqms_lat", "wqms_lon"]]
        coords.columns = ["lat", "long"]
        return coords

    def gauge_coords(
            self,
            gauges: Union[str, List[str]] = "all"
    ) -> pd.DataFrame:
        """
        Returns coordinates of the Caravan streamflow gauges in wgs84 projection.

        Parameters
        ----------
        gauges : str or list, optional
            id/ids of ``gauge_id`` gauge(s). Defaults to all gauges.

        Returns
        -------
        pd.DataFrame
            with columns ``lat`` and ``long`` indexed by ``gauge_id``.
        """
        gauges = _validate_ids(gauges, self.gauges(), "gauges")
        coords = self.caravan_site_info().loc[gauges, ["gauge_lat", "gauge_lon"]]
        coords.columns = ["lat", "long"]
        return coords

    # ------------------------------------------------------------------ #
    # water quality / gauge linkage helpers
    # ------------------------------------------------------------------ #
    def duplicate_gauges(self) -> pd.DataFrame:
        """
        Returns Caravan streamflow gauges that share a ``gauge_name`` with
        another gauge, i.e. the same physical station appearing under more than
        one ``gauge_id`` (as CLAUDE.md's duplicate check requires: compare gauge
        names, warn, do not exclude).

        1,258 gauges in 628 name-groups are present, dominated by the same USGS
        station appearing in two source collections with an identical trailing
        id, e.g. ``camels_14306500`` and ``hysets_14306500`` (both "ALSEA RIVER
        NEAR TIDEWATER, OR", same river segment, coordinates agreeing to <0.5 m).
        Detecting these on names is essential: their coordinates differ at the
        ~1e-6 degree level, so an exact match on ``gauge_lat``/``gauge_lon``
        catches only 16 of the 1,258.

        They are reported but never excluded: they carry the same name yet are
        distinct ``gauge_id`` values that may hold different discharge records,
        and 182 of the groups have water quality stations linked to both members
        (so :meth:`fetch_by_gauge` will address them under both ids).

        Returns
        -------
        pd.DataFrame
            the rows of :meth:`caravan_site_info` whose ``gauge_name`` is shared
            with another gauge, with an extra ``duplicate_group`` column, sorted
            by that group.

        Examples
        --------
        >>> ds = CaravanQual(path='/path/to/data')
        >>> dups = ds.duplicate_gauges()
        >>> dups['duplicate_group'].nunique()
        628
        """
        cs = self.caravan_site_info()
        name = cs["gauge_name"]
        mask = name.notna() & name.duplicated(keep=False)

        dups = cs[mask].copy()
        # group by name; sort so members of a group are adjacent
        dups["duplicate_group"] = pd.factorize(name[mask])[0]
        return dups.sort_values(["duplicate_group", "gauge_name"])

    def _warn_duplicate_gauges(self):
        """warns about duplicated gauges without excluding any of them."""
        dups = self.duplicate_gauges()
        if len(dups) == 0:
            return
        n_groups = dups["duplicate_group"].nunique()
        example = dups.index[:2].tolist()
        warnings.warn(
            f"{len(dups)} Caravan gauges share a gauge_name with another gauge "
            f"({n_groups} name-groups, e.g. {example}); this is mostly the same "
            f"station appearing in two source collections. They are kept, not "
            f"excluded; see duplicate_gauges().",
            stacklevel=3)
        return

    def linked_stations(self) -> List[str]:
        """
        returns the ``wqms_id`` of water quality stations that are linked to a
        Caravan streamflow gauge (i.e. that carry co-located streamflow).
        """
        info = self.site_info()
        return info.index[info["gauge_id"].notna()].tolist()

    def gauge_stations(self, gauge: str) -> List[str]:
        """
        returns the ``wqms_id`` of the water quality stations linked to the
        given Caravan ``gauge``.

        Examples
        --------
        >>> ds = CaravanQual(path='/path/to/data')
        >>> len(ds.gauge_stations('grdc_4146360'))
        206
        """
        assert isinstance(gauge, str), "gauge must be a string"
        assert gauge in self.gauges(), f"invalid gauge {gauge}"
        info = self.site_info()
        return info.index[info["gauge_id"] == gauge].tolist()

    # ------------------------------------------------------------------ #
    # reading water quality data
    # ------------------------------------------------------------------ #
    def _read_constituent(self, parameter: str) -> pd.DataFrame:
        """
        reads the raw csv file of a single constituent, unchanged.

        The numeric columns are read as float64 to preserve the exact source
        values: some constituents (e.g. DOSAT) carry outlier values as large as
        4e59 that would overflow float32 to inf. Those are marked in the
        ``flag`` column but kept unchanged here for fidelity. The reader itself
        is the module-level :func:`_read_constituent_file` so a process pool can
        use it without pickling ``self``.
        """
        return _read_constituent_file(self.csv_dir, parameter)

    def data(
            self,
            parameter: str,
            stations: Union[str, List[str]] = "all",
            st: Union[str, pd.Timestamp] = None,
            en: Union[str, pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Returns the faithful long-format table of a single water quality
        constituent, exactly as stored in the dataset (units unchanged). This
        is the most efficient access pattern because the data is stored one
        file per constituent.

        Parameters
        ----------
        parameter : str
            name of the constituent, see :attr:`parameters`.
        stations : str or list, optional
            ``wqms_id`` station(s) to keep. Defaults to all stations that have
            an observation for this constituent.
        st : str, optional
            start date to trim the observations from.
        en : str, optional
            end date to trim the observations to.

        Returns
        -------
        pd.DataFrame
            with columns ``wqms_id``, ``dates``, ``obs``, ``unit``, ``flag``,
            ``detection_limit``, ``imputation_method`` and ``streamflow``.
            ``obs`` is the measured value in the constituent's unit.
            ``streamflow`` (m3/s) is the co-located gauge discharge on the
            sampling date; it is NaN when the station has no linked gauge and
            also when the linked gauge simply has no discharge record for that
            date (of the 2,464 linked Amoxicillin observations, 832 are NaN on
            this second count). A NaN therefore does not imply the station is
            unlinked; use :meth:`site_info` for the linkage itself.

            The ``flag`` column marks below-detection values with ``<`` (these
            always carry an ``imputation_method`` of ``LOD/2`` or ``ROS``) and
            other quality-flagged values with ``*`` (never imputed); see
            `Jones et al., 2026`_ for the exact flag semantics. Values are
            returned unchanged so callers can filter as their analysis requires.

        Examples
        --------
        >>> ds = CaravanQual(path='/path/to/data')
        >>> temp = ds.data('TEMP')
        >>> temp['unit'].unique().tolist()   # note the spelling variant
        ['deg C', 'cel']
        >>> ds.data('NO3N', stations='wqms_01200003').shape

        Notes
        -----
        A few constituents carry more than one unit string, see
        :meth:`unit_counts`. Because this method is deliberately faithful it
        returns them unchanged; :meth:`fetch` drops observations whose unit is
        not the constituent's quantity.
        """
        assert isinstance(parameter, str), "parameter must be a string"
        assert parameter in self.parameters, f"invalid parameter {parameter}"

        df = self._read_constituent(parameter)
        # the ``variable`` column is constant (== parameter); drop the redundancy
        df = df.drop(columns="variable")

        if not (isinstance(stations, str) and stations == "all"):
            stations = _validate_ids(stations, self.stations(), "stations")
            df = df[df["wqms_id"].isin(stations)]

        if st is not None:
            df = df[df["dates"] >= pd.Timestamp(st)]
        if en is not None:
            df = df[df["dates"] <= pd.Timestamp(en)]

        return df.reset_index(drop=True)

    def num_obs(self, parameter: str) -> pd.Series:
        """
        returns the number of observations of ``parameter`` available at each
        station as a :obj:`pandas.Series` indexed by ``wqms_id``.

        Examples
        --------
        >>> ds = CaravanQual(path='/path/to/data')
        >>> ds.num_obs('TEMP').sum()
        """
        ids = self._read_column(parameter, "wqms_id")
        return ids.groupby(ids).size().rename(parameter)

    def stations_with_parameter(self, parameter: str) -> List[str]:
        """
        returns the ``wqms_id`` of stations that have at least one observation
        of the given ``parameter``.

        Examples
        --------
        >>> ds = CaravanQual(path='/path/to/data')
        >>> len(ds.stations_with_parameter('pH'))
        """
        return self._read_column(parameter, "wqms_id").unique().tolist()

    def unit_counts(self, parameter: str) -> pd.Series:
        """
        returns how many observations of ``parameter`` carry each unit string.

        Nine constituents (``Carbamazepine``, ``EC``, ``EColi``, ``FC``, ``pH``,
        ``POC``, ``TEMP``, ``TIC``, ``TotColi``) carry more than one unit. Most
        are spelling variants of the same quantity (e.g. ``cel`` for ``deg C``),
        but a few rows report a genuinely different quantity (e.g. ``%`` instead
        of ``mg/l``); see :meth:`fetch` for how those are handled.

        Examples
        --------
        >>> ds = CaravanQual(path='/path/to/data')
        >>> ds.unit_counts('TEMP')
        """
        units = self._read_column(parameter, "unit")
        return units.groupby(units).size().rename(parameter)

    def _drop_foreign_units(self, df: pd.DataFrame, parameter: str) -> pd.DataFrame:
        """
        removes observations whose ``unit`` denotes a quantity (or enumeration
        method) other than the constituent's own, e.g. the ``no/100ml`` counts
        of EColi/FC/TotColi against their canonical ``cfu/100ml``. Spelling/case
        variants of the canonical unit (see ``_UNIT_SYNONYMS``) are kept
        unchanged.

        Rows with a missing ``unit`` are also kept, since a blank unit is not
        evidence of a different quantity. No observation in the current release
        (v1.1) has a missing unit, so this branch is purely defensive.
        """
        if df.empty:
            return df

        canonical = self._parameter_units.get(parameter)
        accepted = {str(canonical).strip().lower()}
        accepted |= _UNIT_SYNONYMS.get(canonical, set())

        unit = df["unit"].astype(str).str.strip().str.lower()
        keep = unit.isin(accepted) | df["unit"].isna()

        n_dropped = int((~keep).sum())
        if n_dropped:
            # discarding observations is always worth flagging, regardless of
            # verbosity - silent data loss is exactly what a reviewer cannot see
            foreign = sorted(df.loc[~keep, "unit"].dropna().unique().tolist())
            warnings.warn(
                f"{parameter}: dropped {n_dropped} observation(s) reported in "
                f"{foreign} rather than {canonical!r}. Use data() to access them.",
                stacklevel=3)

        return df[keep]

    def _read_column(self, parameter: str, column: str) -> pd.Series:
        """reads a single column of a constituent file (much cheaper than all)."""
        assert isinstance(parameter, str), "parameter must be a string"
        assert parameter in self.parameters, f"invalid parameter {parameter}"
        fpath = os.path.join(self.csv_dir, f"{parameter}.csv")
        return pd.read_csv(fpath, usecols=[column], dtype={column: str})[column]

    def _read_parameters(
            self,
            parameters: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        reads the raw csv of every requested constituent, in parallel when that
        actually pays off.

        The decision is made on total bytes rather than on the number of files:
        spawning processes for a handful of small constituents is slower than
        reading them serially (measured 0.97 s pooled vs 0.62 s for 4 small
        files), while the largest ones benefit substantially (26.3 s -> 14.6 s).
        """
        cpus = self.processes or min(get_cpus() - 2, 16)

        nbytes = sum(
            os.path.getsize(os.path.join(self.csv_dir, f"{p}.csv"))
            for p in parameters)

        if len(parameters) > 1 and cpus > 1 and nbytes > _PARALLEL_READ_BYTES:
            if self.verbosity > 1:
                print(f"Using {cpus} cpus to read {len(parameters)} constituents "
                      f"({nbytes / 1e6:.0f} MB)")
            worker = functools.partial(_read_constituent_file, self.csv_dir)
            with cf.ProcessPoolExecutor(max_workers=cpus) as executor:
                results = list(executor.map(worker, parameters))
            return dict(zip(parameters, results))

        return {p: self._read_constituent(p) for p in parameters}

    def fetch(
            self,
            stations: Union[str, List[str]] = "all",
            parameters: Union[str, List[str]] = None,
            drop_imputed: bool = False,
            drop_flagged: bool = False,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetches water quality observations for the given station(s) as a
        station-centric wide table (dates as index, constituents as columns).

        ``parameters`` is required: the data is stored one file per constituent
        and the full corpus is 96 million rows (~25 GB in memory), so there is
        no safe default. Read one constituent at a time with :meth:`data` for
        large jobs.

        .. warning::
            The wide table contains only the ``obs`` values, so the ``flag``,
            ``detection_limit`` and ``imputation_method`` context is not
            carried. The values are of three kinds (counts for ``TEMP``):
            11,575,727 plain measurements; 132,553 quality-flagged (``flag`` =
            ``*``); and 14,690 below-detection values substituted by ``LOD/2``
            or ``ROS`` (``flag`` = ``<``). ``drop_imputed=True`` removes the
            third kind only; ``drop_flagged=True`` removes the second and third
            (anything with a non-empty ``flag``). Neither is on by default. For
            full control use :meth:`data`, which returns all the context columns.

        Parameters
        ----------
        stations : str or list, optional
            ``wqms_id`` station(s) to fetch. Defaults to all stations.
        parameters : str or list
            constituent(s) to fetch. Required.
        drop_imputed : bool, optional (default=False)
            if True, observations that were substituted for a below-detection
            value (``imputation_method`` of ``LOD/2`` or ``ROS``) are removed,
            leaving only measured values. This does not remove the ``*``-flagged
            values, which are not imputed; use ``drop_flagged`` for those.
        drop_flagged : bool, optional (default=False)
            if True, every observation carrying a non-empty ``flag`` (``<``,
            ``*`` or ``>``) is removed, keeping only unflagged measurements.
            This is a superset of ``drop_imputed``.

        Returns
        -------
        Dict[str, pd.DataFrame]
            a dictionary keyed by ``wqms_id``. Each value is a DataFrame indexed
            by observation date with the requested constituents as columns
            (values in each constituent's own unit). Only stations with at least
            one non-missing observation of the requested constituents appear, and
            each station carries only the constituents it actually measured.
            Where a station has more than one sample on the same date, the mean
            of the replicates is used.

            Rows whose ``obs`` is NaN are dropped before this table is built: a
            NaN is the absence of a value, so it contributes nothing to a wide
            cell or to the per-date mean. No actual measurement is lost; the raw
            NaN rows remain visible in :meth:`data`.

            Observations whose ``unit`` denotes a quantity other than the
            constituent's own are dropped (with a warning), since the wide table
            has no column to carry a differing unit. This is the two ``%`` rows
            of ``POC``/``TIC`` (vs ``mg/l``) and, more substantially, the
            125,620 ``no/100ml`` counts of ``EColi``/``FC``/``TotColi`` (a
            different enumeration method from their canonical ``cfu/100ml``).
            All remain available through :meth:`data`. Spelling variants of the
            same unit (``cel`` for ``deg C``, ``ug`` for ``ug/l``) are kept.
            See :meth:`unit_counts`.

        Examples
        --------
        >>> ds = CaravanQual(path='/path/to/data')
        >>> data = ds.fetch(stations='wqms_01200003', parameters=['Atrazine', 'NO3N'])
        >>> data['wqms_01200003'].columns.tolist()
        ['Atrazine', 'NO3N']
        >>> data = ds.fetch(stations=['wqms_01200003', 'wqms_01200006'], parameters='Atrazine')
        """
        _reject_all_parameters(parameters)
        parameters = validate_attributes(parameters, self.parameters, "parameters")

        all_stns = isinstance(stations, str) and stations == "all"
        if not all_stns:
            stations = _validate_ids(stations, self.stations(), "stations")
            stn_set = set(stations)

        if len(parameters) > 10 and self.verbosity > 0:
            warnings.warn(
                f"Reading {len(parameters)} constituent files; this may be slow "
                f"and memory intensive. Consider using data() per constituent.",
                stacklevel=2)

        raw = self._read_parameters(parameters)

        # collect per-station pivots for each constituent
        frames = []
        for parameter, df in raw.items():
            if not all_stns:
                df = df[df["wqms_id"].isin(stn_set)]
            df = self._drop_foreign_units(df, parameter)
            if drop_flagged:
                df = df[df["flag"].isna()]
            elif drop_imputed:
                df = df[df["imputation_method"].isna()]
            # drop NaN observations up front: they contribute nothing to the
            # per-date mean, and removing them here means a station's column is
            # all-NaN in the wide table iff it genuinely lacks that constituent
            sub = df.loc[df["obs"].notna(), ["wqms_id", "dates", "obs"]]
            if sub.empty:
                continue
            frames.append(sub.assign(variable=parameter))

        if not frames:
            return {}

        big = pd.concat(frames, ignore_index=True)

        # One vectorised groupby+unstack instead of a per-station pivot_table
        # loop: the loop is O(n_stations) in Python and dominates (23 s vs 1.3 s
        # for a 14,627-station fetch). The mean over (wqms_id, dates, variable)
        # reproduces the aggfunc='mean' of the replicate samples exactly.
        wide = (big.groupby(["wqms_id", "dates", "variable"])["obs"]
                   .mean()
                   .unstack("variable"))
        wide.columns.name = None
        wide = wide.reindex(columns=[p for p in parameters if p in wide.columns])

        # With NaN observations already removed, a single-constituent request
        # can never produce an all-NaN column, so the per-station dropna (which
        # otherwise costs ~20 s of a 32 s TEMP fetch) is only needed when more
        # than one constituent was requested.
        single = wide.shape[1] == 1
        out = {}
        for stn, sub in wide.groupby(level="wqms_id"):
            sub = sub.droplevel("wqms_id")
            out[stn] = sub if single else sub.dropna(axis=1, how="all")

        return out

    def fetch_by_gauge(
            self,
            gauges: Union[str, List[str]] = "all",
            parameters: Union[str, List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetches water quality observations grouped by the Caravan streamflow
        gauge they are linked to. This is the gauge-centric counterpart of
        :meth:`fetch` and is useful for concentration-discharge style analyses.

        The observations of every ``wqms_id`` linked to a gauge are returned as
        a single faithful long-format table (no aggregation across the distinct
        stations), so the ``wqms_id`` column is retained to disambiguate them.

        Parameters
        ----------
        gauges : str or list, optional
            Caravan ``gauge_id``(s) to fetch. Defaults to all gauges that have
            at least one linked water quality station.
        parameters : str or list
            constituent(s) to fetch. Required, for the same memory reason as
            in :meth:`fetch`.

        Returns
        -------
        Dict[str, pd.DataFrame]
            a dictionary keyed by ``gauge_id``. Each value is a long-format
            DataFrame with columns ``wqms_id``, ``dates``, ``variable``,
            ``obs``, ``unit``, ``flag``, ``detection_limit``,
            ``imputation_method`` and ``streamflow``. Because this is the
            faithful long form, the ``unit`` and flag columns are retained and
            nothing is dropped.

        Examples
        --------
        >>> ds = CaravanQual(path='/path/to/data')
        >>> data = ds.fetch_by_gauge('grdc_4146360', parameters='TEMP')
        >>> data['grdc_4146360'].shape
        (8635, 9)
        """
        _reject_all_parameters(parameters)
        parameters = validate_attributes(parameters, self.parameters, "parameters")

        info = self.site_info()
        linked = info[info["gauge_id"].notna()]

        if isinstance(gauges, str) and gauges == "all":
            gauges = linked["gauge_id"].unique().tolist()
        else:
            gauges = _validate_ids(gauges, self.gauges(), "gauges")

        # wqms_id -> gauge_id for the stations of interest
        gauge_set = set(gauges)
        stn_to_gauge = linked["gauge_id"][linked["gauge_id"].isin(gauge_set)]
        wanted_stns = set(stn_to_gauge.index)

        raw = self._read_parameters(parameters)

        frames = []
        for parameter, df in raw.items():
            df = df[df["wqms_id"].isin(wanted_stns)]
            if df.empty:
                continue
            frames.append(df)

        out = {g: [] for g in gauges}
        if frames:
            big = pd.concat(frames, ignore_index=True)
            big["gauge_id"] = big["wqms_id"].map(stn_to_gauge)
            for gauge, g_df in big.groupby("gauge_id"):
                out[gauge] = g_df.drop(columns="gauge_id").reset_index(drop=True)

        # gauges without data get an empty frame
        out = {g: (v if isinstance(v, pd.DataFrame) else pd.DataFrame()) for g, v in out.items()}
        return out

    # ------------------------------------------------------------------ #
    # optional : catchment attributes (require xarray + zarr)
    # ------------------------------------------------------------------ #
    def _check_attributes(self):
        # gate on the data being present rather than on how this instance was
        # constructed: if the zarr store is already extracted there is no
        # reason to send the user off to re-download 1.5 GB they have.
        if not self._attributes and not os.path.isdir(self.zarr_dir):
            raise ValueError(
                "Catchment attributes are not available at "
                f"{self.zarr_dir}. Re-initialise the class with "
                "attributes=True (requires xarray and zarr) to download them.")
        if xr is None:
            raise ModuleNotFoundError(
                "xarray (with zarr) is required to read the catchment attributes. "
                "Install with `pip install xarray zarr`.")

    def _zarr(self):
        """opens the zarr store lazily and caches the handle."""
        if getattr(self, "_zarr_ds", None) is None:
            self._zarr_ds = xr.open_zarr(self.zarr_dir, consolidated=True)
        return self._zarr_ds

    @property
    def static_features(self) -> List[str]:
        """
        names of the segment-level static attributes available in the optional
        zarr store. These are the 197 catchment attributes (HydroATLAS/GEOGLOWS)
        plus 8 stream-segment attributes, all indexed by river segment
        (``LINKNO``). Requires ``attributes=True``.
        """
        self._check_attributes()
        ds = self._zarr()
        return [v for v in ds.data_vars if ds[v].dims == ("LINKNO",)]

    def attributes(
            self,
            features: Union[str, List[str]] = "all",
    ) -> pd.DataFrame:
        """
        Returns the catchment attributes indexed by river segment (``LINKNO``),
        exactly as stored (units unchanged, see :meth:`variables`). Requires
        ``attributes=True``.

        Parameters
        ----------
        features : str or list, optional
            attribute name(s), see :attr:`static_features`.

        Returns
        -------
        pd.DataFrame
            indexed by ``LINKNO`` with the requested attributes as columns.
        """
        self._check_attributes()
        features = validate_attributes(features, self.static_features, "features")
        ds = self._zarr()
        df = ds[features].to_dataframe()
        return df

    def stn_attributes(
            self,
            stations: Union[str, List[str]] = "all",
            features: Union[str, List[str]] = "all",
    ) -> pd.DataFrame:
        """
        Returns the catchment attributes joined onto the water quality stations
        (``wqms_id``) via their river segment (``LINKNO``). Requires
        ``attributes=True``.

        Parameters
        ----------
        stations : str or list, optional
            ``wqms_id`` station(s). Defaults to all stations.
        features : str or list, optional
            attribute name(s), see :attr:`static_features`.

        Returns
        -------
        pd.DataFrame
            indexed by ``wqms_id`` with the requested attributes as columns.
            Stations whose ``LINKNO`` is unknown get NaN rows.
        """
        self._check_attributes()
        features = validate_attributes(features, self.static_features, "features")
        stations = _validate_ids(stations, self.stations(), "stations")

        attrs = self.attributes(features)  # indexed by LINKNO
        link = self.site_info().loc[stations, "LINKNO"]

        out = attrs.reindex(link.values)
        out.index = link.index  # back to wqms_id
        return out
