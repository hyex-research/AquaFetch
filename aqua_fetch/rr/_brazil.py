
__all__ = ['CAMELS_BR', 'CABra']

import os
import re
import glob
import json
import time
import shutil
import hashlib
import zipfile
import calendar
import warnings
import functools
import urllib.error
import urllib.request
import concurrent.futures as cf
from typing import Union, List, Dict, Tuple

import numpy as np
import pandas as pd

from .utils import _RainfallRunoff, cache_name, ymd_index
from ._camels import _remove_stale, _first_and_last_row, _warn_duplicate_gauges
from ..utils import validate_attributes, download
from ._map import (
    min_air_temp,
    max_air_temp,
    mean_air_temp,
    mean_air_temp_with_specifier,
    min_air_temp_with_specifier,
    max_air_temp_with_specifier,
    total_potential_evapotranspiration_with_specifier,
    actual_evapotranspiration_with_specifier,
    total_precipitation_with_specifier,
    observed_streamflow_cms,
    observed_streamflow_mm,
    simulated_streamflow_cms,
    reference_evapotranspiration_with_specifier,
    surface_soil_moisture_with_specifier,
    rootzone_soil_moisture_with_specifier,
    soil_moisture_layer_with_specifier,
    mean_rel_hum_with_specifier,
    mean_windspeed_with_specifier,
    solar_radiation_with_specifier,
    MJ_M2_DAY_TO_WM2,
)

from ._map import (
    catchment_area,
    gauge_latitude,
    gauge_longitude,
    slope
    )

from .._backend import xarray as xr

# directory separator
SEP = os.sep


# %% ANA HidroWeb
# The Brazilian National Water and Sanitation Agency (ANA) keeps serving the
# gauge records from which CAMELS-BR itself is built. The legacy endpoint of its
# HidroWeb portal (https://www.snirh.gov.br/hidroweb/) answers without
# authentication with one xml block per gauge and month, holding 31 discharge
# fields (Vazao01..Vazao31) and 31 matching status fields.

ANA_SERVICE_URL = "http://telemetriaws1.ana.gov.br/ServiceANA.asmx/HidroSerieHistorica"

# tipoDados of the service: 1 = stage, 2 = rainfall, 3 = discharge
_ANA_DISCHARGE = 3

# columns written for every downloaded gauge. ``status`` is ANA's per day code
# and uses the same values as the ``qual_flag`` of CAMELS-BR: 0 no description,
# 1 from stage and rating curve, 2 estimated by ANA, 3 doubtful, 4 stage below
# the range, 5 changed cross section, 6 dry river bed, 7 stage above the range.
# ``nivel_consistencia`` is the consistency level of the month and is encoded
# differently from the ``qual_control_by_ana`` of CAMELS-BR: 2 (reviewed by ANA,
# consistido) is what the release publishes as 1, and 1 (raw, bruto) as 0. The
# relation holds one way only, because ANA reviews a month after a release.
ANA_COLUMNS = ['streamflow_m3s', 'status', 'nivel_consistencia']

_ANA_TAG = re.compile(r"<(?P<tag>[A-Za-z0-9_]+)>(?P<val>[^<]*)</(?P=tag)>")
_ANA_DAY_FIELD = re.compile(r"^Vazao(\d{2})(Status)?$")


def _ana_request(
        station: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        retries: int = 5,
        timeout: float = 180.0,
) -> Union[str, None]:
    """
    The xml answer of the ANA web service for one gauge, or None when the
    service did not answer. It replies 502 and 404 when hit concurrently, so a
    failure is retried with a growing pause instead of being raised.
    """
    url = (f"{ANA_SERVICE_URL}?codEstacao={station}"
           f"&dataInicio={start.strftime('%d/%m/%Y')}"
           f"&dataFim={end.strftime('%d/%m/%Y')}"
           f"&tipoDados={_ANA_DISCHARGE}&nivelConsistencia=")

    wait = 5.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                if response.status == 200:
                    return response.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, OSError):
            pass
        if attempt < retries - 1:
            time.sleep(wait)
            wait *= 2
    return None


def _ana_parse(
        xml_text: str,
        start: pd.Timestamp,
        end: pd.Timestamp
) -> pd.DataFrame:
    """
    Expands the gauge-month blocks of an ANA answer into one row per day, with
    the discharge in m3/s (the unit the service serves), ANA's status code of
    that day and the consistency level of that month. A month served at both
    levels is kept at the reviewed one, which is what CAMELS-BR publishes.
    """
    values = {}

    for block in xml_text.split("<SerieHistorica")[1:]:
        month, level, flow, status = None, None, {}, {}

        for match in _ANA_TAG.finditer(block):
            tag, val = match.group("tag"), match.group("val").strip()
            if tag == "DataHora":
                month = val.split(" ")[0] or None
            elif tag == "NivelConsistencia":
                level = val
            else:
                day_field = _ANA_DAY_FIELD.match(tag)
                if day_field:
                    day = int(day_field.group(1))
                    if day_field.group(2):
                        status[day] = val
                    else:
                        flow[day] = val

        if not month:
            continue
        year, mon = int(month[:4]), int(month[5:7])
        level = int(level) if level else 0

        for day in range(1, calendar.monthrange(year, mon)[1] + 1):
            date = pd.Timestamp(year=year, month=mon, day=day)
            if not start <= date <= end:
                continue
            if date in values and values[date][2] > level:
                continue
            raw = flow.get(day, "")
            values[date] = (float(raw) if raw else float("nan"),
                            status.get(day, ""),
                            level)

    df = pd.DataFrame(values.values(), columns=ANA_COLUMNS,
                      index=pd.DatetimeIndex(values.keys(), name='date'))
    return df.sort_index()


def _ana_fetch_station(
        station: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        out_dir: str,
        retries: int = 5,
        timeout: float = 180.0,
        merge: bool = True,
) -> Dict:
    """
    Downloads one gauge and writes ``out_dir/<station>.csv``. Returns what
    happened: the outcome (``ok``, ``no data`` or ``failed``), the number of days
    with a discharge value, the first and the last of those days, a digest of the
    values, the number of days whose value differs from an earlier download, the
    number of days ANA has withdrawn since, the number of days lost because the
    file was replaced, and whether the file was written at all.

    With ``merge`` the days the file already holds are kept, so that a narrow
    request never throws away a wider download; where ANA now serves another
    value for such a day the freshly served one wins. Without it the file is
    replaced, which is what ``update_streamflow(overwrite=True)`` asks for.

    A failed request writes no file, so that the gauge is requested again on the
    next call instead of being taken for a gauge without data.

    A module level function, so that a thread or process pool never has to
    pickle the dataset.
    """
    xml_text = _ana_request(station, start, end, retries=retries, timeout=timeout)

    if xml_text is None:
        return {'status': 'failed', 'days': 0, 'first': None, 'last': None,
                'digest': None, 'changed': 0, 'withdrawn': 0, 'dropped': 0,
                'wrote': False}

    if "SerieHistorica" in xml_text:
        df = _ana_parse(xml_text, start, end)
    else:  # answered, but the gauge has no discharge at ANA
        df = pd.DataFrame(columns=ANA_COLUMNS, index=pd.DatetimeIndex([], name='date'))

    fpath = os.path.join(out_dir, f"{station}.csv")
    changed, withdrawn, dropped = 0, 0, 0
    if os.path.exists(fpath):
        old = _read_ana_csv(fpath)
        had = old.index[old['streamflow_m3s'].notna()]
        now_valued = df.index[df['streamflow_m3s'].notna()]

        common = old.index.intersection(df.index)
        if len(common):
            before = old.loc[common, 'streamflow_m3s'].to_numpy(dtype=float)
            now = df.loc[common, 'streamflow_m3s'].to_numpy(dtype=float)
            changed = int(((before != now) & ~np.isnan(before) & ~np.isnan(now)).sum())

        # a day this answer covers without a value has been withdrawn by ANA; a
        # day it does not cover at all is only lost if the file is replaced
        withdrawn = int(len(had.intersection(df.index).difference(now_valued)))
        dropped = 0 if merge else int(len(had.difference(df.index)))

        if merge:
            # only the days this answer does not cover are taken from the file,
            # whole rows, so that every row keeps its own provenance
            untouched = old.loc[old.index.difference(df.index)]
            if df.empty:
                df = old           # nothing was served, the file stands
            elif not untouched.empty:
                df = pd.concat([df, untouched])

    df = df.sort_index()
    df.to_csv(fpath)

    with_value = df[df['streamflow_m3s'].notna()]
    values = with_value['streamflow_m3s']
    return {'status': 'ok' if len(with_value) else 'no data',
            'days': len(with_value),
            'first': str(with_value.index[0].date()) if len(with_value) else None,
            'last': str(with_value.index[-1].date()) if len(with_value) else None,
            # of the values themselves, so that a revision renames the cache
            'digest': hashlib.sha1(values.to_csv().encode()).hexdigest()[:12],
            'changed': changed, 'withdrawn': withdrawn, 'dropped': dropped,
            'wrote': True}


def _as_station_list(stations) -> Union[str, List[str]]:
    """
    ``stations`` as a list of ids, or the ``all`` sentinel. Any iterable of ids
    is accepted and materialized here, so that a generator is not emptied by a
    first pass over it; a number or a mapping is refused with a clear error
    instead of failing deep inside a read.
    """
    if isinstance(stations, str):
        return stations if stations == 'all' else [stations]
    if isinstance(stations, dict) or not hasattr(stations, '__iter__'):
        raise TypeError(f"stations must be a gauge id or an iterable of gauge "
                        f"ids, not {type(stations).__name__}")
    return [str(station) for station in stations]


def _read_ana_csv(fpath: str, only_with_value: bool = False) -> pd.DataFrame:
    """one gauge as downloaded from ANA, index is the date. With
    ``only_with_value`` the days ANA served without a discharge are dropped, so
    that they do not extend the time index of the dataset with empty rows."""
    df = pd.read_csv(fpath, index_col='date', parse_dates=['date'],
                     dtype={'streamflow_m3s': np.float64, 'status': str})
    # a day ANA serves without a status is written as an empty field and read
    # back as NaN; it stays the empty string it was served as
    df['status'] = df['status'].fillna('')
    if only_with_value:
        df = df[df['streamflow_m3s'].notna()]
    return df


def _ymd(row: bytes) -> pd.Timestamp:
    """date from the ``year month day`` fields of a CAMELS-BR time series row"""
    year, month, day = row.split()[:3]
    return pd.Timestamp(int(year), int(month), int(day))


def _last_valid_date(fpath: str, column: str) -> Union[pd.Timestamp, None]:
    """last day with a value in ``column`` of a CAMELS-BR time series file"""
    df = pd.read_csv(fpath, sep=' ', usecols=['year', 'month', 'day', column])
    valid = df[df[column].notna()]
    if valid.empty:
        return None
    return pd.Timestamp(*valid.iloc[-1][['year', 'month', 'day']].astype(int))


class CAMELS_BR(_RainfallRunoff):
    """
    Daily data of 897 Brazilian catchments following
    `Chagas et al., 2020 <https://doi.org/10.5194/essd-12-2075-2020>`_ .
    Two releases are available through ``version``:

    - ``'1.2'`` (default): the March 2025 release
      (`Zenodo 15025488 <https://zenodo.org/records/15025488>`_) with 26 dynamic
      and 66 static features. Every time series starts on 1980-01-01 and each
      one is NaN after its own last value, which differs per product: observed
      streamflow (taken from ANA on 2025-02-27) ends between 2009-07-27 and
      2024-12-31 depending on the gauge, ERA5-Land 2024-10-18, CPC precipitation
      2024-10-21 and CPC temperature 2024-10-22, CHIRPS and ERA5-Land air
      temperature 2024-09-30, BR-DWGD 2024-03-20, GLEAM and MSWEP earlier still
      (2023-12-31 and 2020-12-30) and MGB-SA actual evapotranspiration
      2014-12-31. ERA5-Land soil moisture runs to 2024-12-31, which is where the
      time index ends.
    - ``'1.1'``: the July 2020 release
      (`Zenodo 3964745 <https://zenodo.org/records/3964745>`_) with 11 dynamic
      and 67 static features from 1980-01-01 to 2018-12-31.

    Release 1.2 adds ERA5-Land and BR-DWGD products, reference evapotranspiration
    and soil moisture, drops CPC mean temperature and the ``gauge_region``
    attribute, and revises the published streamflow following ANA's quality
    control: 34 % of the 1980-2018 daily values differ from 1.1 by more than 1 %.
    The two releases hold the same 897 gauges.

    BR-DWGD precipitation, temperature and reference evapotranspiration exist
    for 864 of the 897 catchments and GLEAM soil moisture for 896, the rest
    are NaN. Catchment boundaries are provided for every gauge, 22 of which lie
    outside their own boundary, the farthest by about 10 km. The quality columns
    of the streamflow files are not dynamic features; they are returned by
    :meth:`fetch_raw_streamflow`. :meth:`update_streamflow` extends the observed
    streamflow beyond the release with data downloaded from ANA.

    Not provided: the rainfall of the 11853 ANA rain gauges of release 1.2
    (individual gauges, not catchment averages) and its MGB-SA simulated
    streamflow, which is model output; release 1.1 ships the latter and it is
    returned by :meth:`fetch_simulated_streamflow`.

    The sources themselves contain a few implausible values, which are served
    unchanged: isolated days of -42 degC in the CPC minimum temperature (in both
    releases) and 212 negative potential evapotranspiration values: 33 in
    ERA5-Land, down to -7.4 mm/day, and 179 in GLEAM, down to -0.1 mm/day.

    Timings for 1.2 on a 48-core machine: the first initialization downloads
    1.0 GB in about 6 minutes, extracts it (5.5 GB on disk) and builds a 1.5 GB
    netCDF cache in 8 s. Afterwards initialization takes 0.01 s and all 897
    stations with all 26 features are fetched in about 1 s from the cache (5 to
    6 s from the text files). :meth:`update_streamflow` adds a second cache of that size,
    the one which carries the extended record.

    Examples
    --------
    >>> from aqua_fetch import CAMELS_BR
    >>> dataset = CAMELS_BR()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='46035000', as_dataframe=True)
    >>> df = dynamic['46035000'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (16437, 26)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
    897
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (89 out of 897)
    89
    ...
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('46035000', as_dataframe=True,
    ...  dynamic_features=['pcp_mm_cpc', 'aet_mm_mgb', 'airtemp_C_mean_era5land', 'q_cms_obs'])
    >>> dynamic['46035000'].shape
    (16437, 4)
    ...
    ... # get data of a selected period
    >>> _, dynamic = dataset.fetch('46035000', st='2000-01-01', en='2000-12-31', as_dataframe=True)
    >>> dynamic['46035000'].shape
    (366, 26)
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='46035000', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['46035000'].shape
    ((1, 66), 1, (16437, 26))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    >>> type(dynamic)
    <class 'xarray.core.dataset.Dataset'>
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
    (897, 2)
    >>> dataset.stn_coords('46035000')  # returns coordinates of station whose id is 46035000
                  lat     long
    gauge_id
    46035000 -12.8686 -43.3797
    ...
    # get area (km2) of a single station
    >>> dataset.area('46035000')
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('46035000')
    ...
    # the released record of the previous release
    >>> old = CAMELS_BR(version='1.1')
    """

    url = "https://zenodo.org/records/15025488"

    # {release: {archive: base url}}. The MGB-SA simulated streamflow (model
    # output) and the rainfall of the 11853 individual ANA rain gauges of
    # release 1.2 are not downloaded.
    urls = {
        '1.2': {fname: 'https://zenodo.org/records/15025488/files/' for fname in (
            '01_CAMELS_BR_attributes.zip',
            '02_CAMELS_BR_streamflow_all_catchments.zip',
            '03_CAMELS_BR_streamflow_selected_catchments.zip',
            '05_CAMELS_BR_precipitation.zip',
            '06_CAMELS_BR_actual_evapotransp.zip',
            '07_CAMELS_BR_potential_evapotransp.zip',
            '08_CAMELS_BR_reference_evapotransp.zip',
            '09_CAMELS_BR_temperature.zip',
            '10_CAMELS_BR_soil_moisture.zip',
            '12_CAMELS_BR_catchment_boundaries.zip',
            '13_CAMELS_BR_gauge_location.zip',
        )},
        '1.1': {fname: 'https://zenodo.org/records/3964745/files/' for fname in (
            '01_CAMELS_BR_attributes.zip',
            '02_CAMELS_BR_streamflow_m3s.zip',
            '03_CAMELS_BR_streamflow_mm_selected_catchments.zip',
            '04_CAMELS_BR_streamflow_simulated.zip',
            '05_CAMELS_BR_precipitation_chirps.zip',
            '06_CAMELS_BR_precipitation_mswep.zip',
            '07_CAMELS_BR_precipitation_cpc.zip',
            '08_CAMELS_BR_evapotransp_gleam.zip',
            '09_CAMELS_BR_evapotransp_mgb.zip',
            '10_CAMELS_BR_potential_evapotransp_gleam.zip',
            '11_CAMELS_BR_temperature_min_cpc.zip',
            '12_CAMELS_BR_temperature_mean_cpc.zip',
            '13_CAMELS_BR_temperature_max_cpc.zip',
            '14_CAMELS_BR_catchment_boundaries.zip',
            '15_CAMELS_BR_gauges_location_shapefile.zip',
        )},
    }

    # time series of each release: group -> (folder, file name suffix, value
    # columns). The columns are the release's own manifest, so a change in the
    # source makes the read fail loudly instead of returning fewer features. The
    # quality columns of the streamflow files are deliberately not listed here,
    # see fetch_raw_streamflow. The order of the groups is the order of the
    # returned dynamic features.
    _DYN_GROUPS = {
        '1.2': {
            'streamflow': ('03_CAMELS_BR_streamflow_selected_catchments', '_streamflow.txt',
                           ('streamflow_m3s', 'streamflow_mm')),
            'precipitation': ('05_CAMELS_BR_precipitation', '_precipitation.txt',
                              ('p_cpc', 'p_mswep', 'p_chirps', 'p_era5land', 'p_brdwgd')),
            'actual_evapotransp': ('06_CAMELS_BR_actual_evapotransp', '_actual_evapotransp.txt',
                                   ('aet_gleam', 'aet_mgb', 'aet_era5land')),
            'potential_evapotransp': ('07_CAMELS_BR_potential_evapotransp', '_potential_evapotransp.txt',
                                      ('pet_gleam', 'pet_era5land')),
            'reference_evapotransp': ('08_CAMELS_BR_reference_evapotransp', '_reference_evapotransp.txt',
                                      ('eto_brdwgd',)),
            'temperature': ('09_CAMELS_BR_temperature', '_temperature.txt',
                            ('tmin_cpc', 'tmax_cpc', 'tmin_era5land', 'tmean_era5land',
                             'tmax_era5land', 'tmin_brdwgd', 'tmax_brdwgd')),
            'soil_moisture': ('10_CAMELS_BR_soil_moisture', '_soil_moisture.txt',
                              ('sm_surface_gleam', 'sm_rootzone_gleam', 'sm_layer1_era5land',
                               'sm_layer2_era5land', 'sm_layer3_era5land', 'sm_layer4_era5land')),
        },
        '1.1': {  # one file per feature
            'streamflow': ('03_CAMELS_BR_streamflow_mm_selected_catchments', '_streamflow_mm.txt',
                           ('streamflow_mm',)),
            'precipitation_cpc': ('07_CAMELS_BR_precipitation_cpc', '_precipitation_cpc.txt',
                                  ('precipitation_cpc',)),
            'precipitation_mswep': ('06_CAMELS_BR_precipitation_mswep', '_precipitation_mswep.txt',
                                    ('precipitation_mswep',)),
            'precipitation_chirps': ('05_CAMELS_BR_precipitation_chirps', '_precipitation_chirps.txt',
                                     ('precipitation_chirps',)),
            'evapotransp_gleam': ('08_CAMELS_BR_evapotransp_gleam', '_evapotransp_gleam.txt',
                                  ('evapotransp_gleam',)),
            'evapotransp_mgb': ('09_CAMELS_BR_evapotransp_mgb', '_evapotransp_mgb.txt',
                                ('evapotransp_mgb',)),
            'potential_evapotransp_gleam': ('10_CAMELS_BR_potential_evapotransp_gleam',
                                            '_potential_evapotransp_gleam.txt',
                                            ('potential_evapotransp_gleam',)),
            'temperature_min': ('11_CAMELS_BR_temperature_min_cpc', '_temperature_min.txt',
                                ('temperature_min',)),
            'temperature_mean': ('12_CAMELS_BR_temperature_mean_cpc', '_temperature_mean.txt',
                                 ('temperature_mean',)),
            'temperature_max': ('13_CAMELS_BR_temperature_max_cpc', '_temperature_max.txt',
                                ('temperature_max',)),
        },
    }

    # groups which the release itself provides for a subset of the catchments,
    # so that a missing file is not reported as an incomplete download
    _PARTIAL_GROUPS = {'1.2': ('reference_evapotransp',), '1.1': ()}

    # streamflow of every gauge of the release (4025 in 1.2, 3679 in 1.1),
    # quality columns included: group -> (folder, file name suffix, columns)
    _EXTRA_GROUPS = {
        '1.2': {
            'streamflow_m3s_raw': ('02_CAMELS_BR_streamflow_all_catchments', '_streamflow.txt',
                                   ('streamflow_m3s', 'qual_control_by_ana', 'qual_flag')),
        },
        '1.1': {
            'streamflow_m3s_raw': ('02_CAMELS_BR_streamflow_m3s', '_streamflow_m3s.txt',
                                   ('streamflow_m3s', 'qual_control_by_ana', 'qual_flag')),
            'simulated_streamflow_m3s': ('04_CAMELS_BR_streamflow_simulated', '_simulated_streamflow.txt',
                                         ('simulated_streamflow_m3s',)),
        },
    }

    # raw column -> standardized name for the groups of _EXTRA_GROUPS
    _EXTRA_MAP = {
        'streamflow_m3s': observed_streamflow_cms(),
        'simulated_streamflow_m3s': simulated_streamflow_cms(),
    }

    # the attribute files of both releases, the manifest of the static data
    _STATIC_FILES = ('camels_br_climate.txt', 'camels_br_geology.txt',
                     'camels_br_human_intervention.txt', 'camels_br_hydrology.txt',
                     'camels_br_land_cover.txt', 'camels_br_location.txt',
                     'camels_br_quality_check.txt', 'camels_br_soil.txt',
                     'camels_br_topography.txt')

    # cached tables which are rebuilt instead of being shipped to a pool worker
    _NOT_PICKLED = ('_location_table', '_static_table', 'bndry_id_map_',
                    '_ana_manifest', '_released_q_ends')

    def __init__(
            self,
            path: str = None,
            version: str = '1.2',
            overwrite: bool = False,
            to_netcdf: bool = True,
            use_ana_update: bool = True,
            verbosity: int = 1,
            **kwargs
    ):
        """
        parameters
        ----------
        path : str
            directory under which the data is (or will be) saved in a
            ``CAMELS_BR`` folder. Both releases can share it. If None, the
            default data directory of aqua_fetch is used.
        version : str
            ``'1.2'`` (default) or ``'1.1'``.
        overwrite : bool
            if True, the archives, extracted folders and netCDF caches of this
            ``version`` are deleted and downloaded again.
        to_netcdf : bool
            whether to cache the dynamic data of this version in a netCDF file
            for faster reading. Requires netCDF4 and xarray.
        use_ana_update : bool
            whether the streamflow downloaded by :meth:`update_streamflow` is
            served by :meth:`fetch`. It has no effect until that method has been
            called for this ``path``; with False the released record is served.
        verbosity : int
            0 prints nothing.
        **kwargs :
            any keyword argument of :py:class:`aqua_fetch.rr._RainfallRunoff`
            such as ``processes`` or ``float_precision``.
        """
        version = str(version)
        if version not in self.urls:
            raise ValueError(
                f"version must be one of {list(self.urls)} but is {version!r}")
        self.version = version
        self.use_ana_update = use_ana_update
        # gauge -> last released streamflow day, filled by _released_q_end
        self._released_q_ends = {}

        super().__init__(path=path, name="CAMELS_BR", overwrite=overwrite,
                         to_netcdf=to_netcdf, verbosity=verbosity, **kwargs)

        self._download_camels_br(overwrite=self.overwrite)

        self._incomplete = self._check_manifest()

        if not self._incomplete:
            self._check_duplicates()

        if self._serves_ana:
            # the served record is not the published one any more
            first, last = self._ana_extent
            warnings.warn(
                f"CAMELS_BR {self.version}: the released streamflow is extended "
                f"with data downloaded from ANA for {len(self._ana_stations)} "
                f"gauges, covering {first.date()} to {last.date()}. Released "
                f"values are kept; ANA fills only the days without one. Use "
                f"use_ana_update=False for the released record.", UserWarning)

        if not self._incomplete and (self.version == '1.1' or self._serves_ana):
            # needed by the read path, so it is built before the pool workers
            # are given a copy of this dataset
            _ = self._gsim_areas

        if self._incomplete:
            # a cache built now would carry the missing files into every later
            # run, under the name of a complete one
            warnings.warn("CAMELS_BR: the netCDF cache is not built because "
                          "files are missing.", UserWarning)
        else:
            self._maybe_to_netcdf()

    def __getstate__(self):
        """
        Drops the large cached tables when the dataset is pickled. The process
        pool of :meth:`_read_dynamic` pickles it once per station, so shipping
        the static table with it would cost more than reading a file.
        """
        state = self.__dict__.copy()
        for cached in self._NOT_PICKLED:
            state.pop(cached, None)
        # _released_q_ends is rebuilt lazily and only used by update_streamflow
        state['_released_q_ends'] = {}
        return state

    # ------------------------------------------------------------------ download

    def _download_camels_br(self, overwrite: bool = False):
        """
        Downloads and extracts the archives of ``self.version`` whose extracted
        folder does not exist. The guard is on the extracted folder and not on
        the archive, so that an archive deleted after extraction (``remove_zip``
        or :meth:`free_disk_space`) does not trigger a fresh download.

        ``overwrite=True`` first deletes the archives, the extracted folders and
        the files derived from them (netCDF caches and the static table) of this
        version: ``download`` would otherwise save the new archive as
        ``<name>.zip1``, which nothing reads, and the stale extracted folder
        would be taken for the new one.
        """
        release_dir = self._release_dir   # resolved before anything is deleted
        os.makedirs(release_dir, exist_ok=True)

        archives = {os.path.join(release_dir, fname): url + fname
                    for fname, url in self.urls[self.version].items()}
        folders = [archive[:-len('.zip')] for archive in archives]

        if overwrite:
            caches = glob.glob(os.path.join(
                glob.escape(self.path),
                f"{self.name.lower()}_{self.timestep}_{self.version}*.nc"))
            _remove_stale([*archives, *folders, *caches, self._static_fpath],
                          self.verbosity)

        for (archive, url), folder in zip(archives.items(), folders):

            if os.path.exists(folder):
                if self.verbosity > 1:
                    print(f"{folder} already exists. Skipping download.")
                continue

            if not os.path.exists(archive):
                if self.verbosity:
                    print(f"Downloading {url} at {archive}")
                download(url, outdir=release_dir,
                         fname=os.path.basename(archive), verbosity=self.verbosity)

            # extracted into a temporary folder which is renamed only once the
            # extraction is complete, so that an interrupted extraction is
            # redone on the next initialization instead of being taken for a
            # complete one
            if self.verbosity:
                print(f"extracting {os.path.basename(archive)}")
            partial = f"{folder}_extracting"
            shutil.rmtree(partial, ignore_errors=True)
            try:
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(partial)
            except BaseException:
                shutil.rmtree(partial, ignore_errors=True)
                raise
            os.rename(partial, folder)

        self.maybe_remove_zip_files()

    def remove_zip_files(self):
        """deletes the archives of this release, which are in its own folder"""
        for fname in self.urls[self.version]:
            archive = os.path.join(self._release_dir, fname)
            if os.path.exists(archive):
                os.remove(archive)
        return

    def _check_manifest(self):
        """
        Warns if this release is incomplete on disk and returns what is wrong,
        e.g. after an interrupted extraction or a file deleted by hand: every
        archive must be extracted,
        the attribute and boundary files must exist and every gauge must have a
        file in every time series folder. The groups of :attr:`_PARTIAL_GROUPS`
        are the documented exception (BR-DWGD covers 864 of the 897 catchments).
        """
        problems = []

        not_extracted = [fname for fname in self.urls[self.version]
                         if not os.path.isdir(self._group_dir(fname[:-len('.zip')]))]
        if not_extracted:
            problems.append(f"{len(not_extracted)} archives are not extracted: "
                            f"{not_extracted}")

        files = [os.path.join(self.static_dir, fname) for fname in self._STATIC_FILES]
        files.append(self.boundary_file)
        missing = [fpath for fpath in files if not os.path.exists(fpath)]
        if missing:
            problems.append(f"{len(missing)} files are missing: {missing}")

        if not not_extracted:
            stations = set(self.stations())
            groups = self._DYN_GROUPS[self.version]
            for group, (folder, suffix, _) in groups.items():
                if group in self._PARTIAL_GROUPS[self.version]:
                    continue
                have = {fname[:-len(suffix)] for fname
                        in os.listdir(self._group_dir(folder))
                        if fname.endswith(suffix)}
                absent = stations - have
                if absent:
                    problems.append(f"{folder}: {len(absent)} of {len(stations)} "
                                    f"gauges have no file")

        if problems:
            warnings.warn(
                f"CAMELS_BR {self.version} is incomplete at {self.path}: "
                f"{'; '.join(problems)}. Use overwrite=True to download it "
                f"again.", UserWarning)
        return problems

    def _check_duplicates(self):
        """warns if two gauges share a name and rounded coordinates"""
        meta = self._location_table.loc[self.stations(),
                                        ['gauge_name', 'gauge_lat', 'gauge_lon']]
        meta = meta.reset_index()
        meta.columns = ['gauge_id', 'gauge_name', 'gauge_lat', 'gauge_lon']
        _warn_duplicate_gauges(f"{self.name} {self.version}", meta)
        return

    # ------------------------------------------------------------------ layout

    @functools.cached_property
    def _release_dir(self) -> os.PathLike:
        """
        Folder of this release inside :attr:`path`. Both releases ship an
        ``01_CAMELS_BR_attributes`` folder whose content differs, so each release
        is kept in a folder of its own name. Release 1.1 was extracted directly
        into ``path`` by earlier versions of this class; such a copy is used
        where it exists, so that it does not have to be downloaded again.

        Worked out once and then cached: ``overwrite=True`` deletes the very
        folder this looks for, so a property would answer differently half way
        through :meth:`_download_camels_br` and the archives would be downloaded
        into a folder the extraction never looks at.
        """
        if self.version == '1.1' and os.path.isdir(os.path.join(
                self.path, '03_CAMELS_BR_streamflow_mm_selected_catchments')):
            return self.path
        return os.path.join(self.path, self.version)

    def _group_dir(self, folder: str) -> os.PathLike:
        """folder of one group of files: the archives hold a folder of their own
        name, which is extracted into a folder of that name as well"""
        return os.path.join(self._release_dir, folder, folder)

    def _group_file(self, group: Tuple[str, str, Tuple[str, ...]],
                    station: str) -> os.PathLike:
        """file of one station within one group of :attr:`_DYN_GROUPS`"""
        folder, suffix, _ = group
        return os.path.join(self._group_dir(folder), f"{station}{suffix}")

    @property
    def _q_group(self) -> Tuple[str, str, Tuple[str, ...]]:
        """the observed streamflow group of the 897 selected catchments"""
        return self._DYN_GROUPS[self.version]['streamflow']

    @property
    def _q_column(self) -> str:
        """name of the observed streamflow column in the files of this release"""
        return self._q_group[2][0]

    @property
    def static_dir(self) -> os.PathLike:
        """folder with the attribute files"""
        return self._group_dir('01_CAMELS_BR_attributes')

    @property
    def _static_fpath(self) -> os.PathLike:
        """csv into which the attribute files of this release are consolidated"""
        return os.path.join(self.path, f"static_features_{self.version}.csv")

    @property
    def boundary_file(self) -> os.PathLike:
        if self.version == '1.1':
            return os.path.join(self._group_dir('14_CAMELS_BR_catchment_boundaries'),
                                'camels_br_catchments.shp')
        return os.path.join(self._group_dir('12_CAMELS_BR_catchment_boundaries'),
                            'camels_br_catchments.gpkg')

    @property
    def boundary_id_map(self) -> str:
        """attribute of the boundary file which holds the gauge id"""
        return "gauge_id"

    def _boundary_catch_id(self, value) -> str:
        """the GeoPackage of release 1.2 stores the gauge id as a float"""
        return str(int(value))

    @property
    def _groups(self) -> Dict[str, Tuple[str, str, Tuple[str, ...]]]:
        """every group of files of this release: the time series which make up
        the dynamic features and the streamflow of all gauges"""
        return {**self._DYN_GROUPS[self.version], **self._EXTRA_GROUPS[self.version]}

    @property
    def group_folders(self) -> Dict[str, str]:
        """group of files of this release -> the folder which holds them"""
        return {group: folder for group, (folder, _, _) in self._groups.items()}

    @property
    def dyn_fname(self) -> Union[str, os.PathLike]:
        """
        name of the netCDF cache of the dynamic data, one per release, precision
        and ANA download, e.g. camels_br_D_1.2_v2.nc or
        camels_br_D_1.2_ana3f2a91c4_v2.nc
        """
        ana = f"_ana{self._ana_stamp}" if self._serves_ana else ""
        return cache_name(f"{self._cache_stem}{ana}.nc")

    @property
    def _cache_stem(self) -> str:
        """name of the netCDF cache of this release and precision, without the
        part which describes the ANA download"""
        precision = "" if np.dtype(self.fp) == np.float32 else f"_{np.dtype(self.fp).name}"
        return f"{self.name.lower()}_{self.timestep}_{self.version}{precision}"

    # ------------------------------------------------------------------ features

    @property
    def static_map(self) -> Dict[str, str]:
        return {
            'area': catchment_area(),                 # km2
            'slope_mean': slope('degrees'),
            'gauge_lat': gauge_latitude(),
            'gauge_lon': gauge_longitude(),
        }

    @property
    def dyn_map(self) -> Dict[str, str]:
        """raw column name -> standardized name. The units of the raw and the
        standardized names are the same, so no conversion is required."""
        if self.version == '1.1':
            return {
                'streamflow_mm': observed_streamflow_mm(),                        # mm/day
                'temperature_min': min_air_temp(),                                # degC
                'temperature_max': max_air_temp(),
                'temperature_mean': mean_air_temp(),
                'precipitation_mswep': total_precipitation_with_specifier('mswep'),   # mm/day
                'precipitation_chirps': total_precipitation_with_specifier('chirps'),
                'precipitation_cpc': total_precipitation_with_specifier('cpc'),
                'potential_evapotransp_gleam': total_potential_evapotranspiration_with_specifier('gleam'),
                'evapotransp_gleam': actual_evapotranspiration_with_specifier('gleam'),
                'evapotransp_mgb': actual_evapotranspiration_with_specifier('mgb'),
            }
        return {
            'streamflow_m3s': observed_streamflow_cms(),                          # m3/s
            'streamflow_mm': observed_streamflow_mm(),                            # mm/day
            'p_brdwgd': total_precipitation_with_specifier('brdwgd'),             # mm/day
            'p_chirps': total_precipitation_with_specifier('chirps'),
            'p_cpc': total_precipitation_with_specifier('cpc'),
            'p_era5land': total_precipitation_with_specifier('era5land'),
            'p_mswep': total_precipitation_with_specifier('mswep'),
            'aet_gleam': actual_evapotranspiration_with_specifier('gleam'),       # mm/day
            'aet_era5land': actual_evapotranspiration_with_specifier('era5land'),
            'aet_mgb': actual_evapotranspiration_with_specifier('mgb'),
            'pet_gleam': total_potential_evapotranspiration_with_specifier('gleam'),   # mm/day
            'pet_era5land': total_potential_evapotranspiration_with_specifier('era5land'),
            'eto_brdwgd': reference_evapotranspiration_with_specifier('brdwgd'),  # mm/day
            'tmin_cpc': min_air_temp_with_specifier('cpc'),                       # degC
            'tmax_cpc': max_air_temp_with_specifier('cpc'),
            'tmin_era5land': min_air_temp_with_specifier('era5land'),
            'tmean_era5land': mean_air_temp_with_specifier('era5land'),
            'tmax_era5land': max_air_temp_with_specifier('era5land'),
            'tmin_brdwgd': min_air_temp_with_specifier('brdwgd'),
            'tmax_brdwgd': max_air_temp_with_specifier('brdwgd'),
            'sm_surface_gleam': surface_soil_moisture_with_specifier('gleam'),    # m3/m3
            'sm_rootzone_gleam': rootzone_soil_moisture_with_specifier('gleam'),
            'sm_layer1_era5land': soil_moisture_layer_with_specifier(1, 'era5land'),
            'sm_layer2_era5land': soil_moisture_layer_with_specifier(2, 'era5land'),
            'sm_layer3_era5land': soil_moisture_layer_with_specifier(3, 'era5land'),
            'sm_layer4_era5land': soil_moisture_layer_with_specifier(4, 'era5land'),
        }

    @property
    def dyn_generators(self) -> Dict:
        """release 1.1 publishes the streamflow of the selected catchments only
        in mm/day, so m3/s is computed from it with the GSIM area, which is the
        area the release itself used for the conversion"""
        if self.version == '1.1':
            return {observed_streamflow_cms(): (self.mm_to_cms, observed_streamflow_mm())}
        return {}

    @property
    def _raw_columns(self) -> List[str]:
        """value columns of this release, in the order they are read"""
        return [column for _, _, columns in self._DYN_GROUPS[self.version].values()
                for column in columns]

    @property
    def dynamic_features(self) -> List[str]:
        mapping = self.dyn_map
        return ([mapping.get(column, column) for column in self._raw_columns]
                + list(self.dyn_generators))

    @property
    def static_features(self) -> List[str]:
        return self._static_table.columns.tolist()

    @property
    def _area_name(self) -> str:
        return catchment_area()

    @property
    def _mm_feature_name(self) -> str:
        """streamflow in mm/day is published, q_mm does not have to compute it"""
        return observed_streamflow_mm()

    @property
    def _q_name(self) -> str:
        return observed_streamflow_cms()

    @property
    def _coords_name(self) -> List[str]:
        return [gauge_latitude(), gauge_longitude()]

    @functools.cached_property
    def _stations(self) -> List[str]:
        """gauge ids, from the names of the observed streamflow files"""
        folder, suffix, _ = self._q_group
        return sorted(fname[:-len(suffix)] for fname
                      in os.listdir(self._group_dir(folder))
                      if fname.endswith(suffix))

    def stations(self) -> List[str]:
        """
        Returns a list of station ids.

        Example
        -------
        >>> dataset = CAMELS_BR()
        >>> stations = dataset.stations()
        """
        return list(self._stations)

    def all_stations(self, group: str) -> List[str]:
        """
        ids of the gauges for which a group of files exists, e.g. ``streamflow``
        for the 897 selected catchments or ``streamflow_m3s_raw`` for every
        gauge of the release. The available groups are the keys of
        :attr:`group_folders`.
        """
        groups = self._groups
        if group not in groups:
            raise ValueError(f"group must be one of {list(groups)} but is {group!r}")
        folder, suffix, _ = groups[group]
        return sorted(fname[:-len(suffix)] for fname
                      in os.listdir(self._group_dir(folder))
                      if fname.endswith(suffix))

    # ------------------------------------------------------------------ reading

    def _read_stn_dyn(self, stn: str) -> pd.DataFrame:
        """dynamic features of one station, the index is the union of the dates
        of its files"""
        frames = []
        for group in self._DYN_GROUPS[self.version].values():
            folder, suffix, columns = group
            fpath = self._group_file(group, stn)
            if not os.path.exists(fpath):
                # a group which the release provides for a subset of the
                # catchments, the columns are added as NaN below
                continue
            df = pd.read_csv(fpath, sep=' ', usecols=['year', 'month', 'day', *columns])
            df.index = ymd_index(df['year'], df['month'], df['day'])
            frames.append(df[list(columns)])

        stn_df = pd.concat(frames, axis=1).reindex(columns=self._raw_columns)
        stn_df = stn_df.astype(self.fp)
        stn_df.rename(columns=self.dyn_map, inplace=True)

        for new_col, (func, old_col) in self.dyn_generators.items():
            stn_df[new_col] = func(stn_df[old_col].rename(stn))

        if self._serves_ana and stn in self._ana_stations:
            stn_df = self._extend_with_ana(stn, stn_df)

        stn_df.columns.name = 'dynamic_features'
        stn_df.index.name = 'time'
        return stn_df

    @functools.cached_property
    def _static_table(self) -> pd.DataFrame:
        """the attribute files of this release as one table, cached on disk"""
        if os.path.exists(self._static_fpath):
            static_df = pd.read_csv(self._static_fpath, index_col='gauge_id',
                                    dtype={'gauge_id': str})
        else:
            frames = [pd.read_csv(os.path.join(self.static_dir, fname), sep=' ',
                                  index_col='gauge_id', dtype={'gauge_id': str})
                      for fname in self._STATIC_FILES]
            # the location file covers every gauge of the release, the other
            # files only the 897 selected catchments
            static_df = pd.concat(frames, axis=1).loc[self.stations()]
            static_df.to_csv(self._static_fpath, index_label='gauge_id')

        static_df.index = static_df.index.astype(str)
        static_df.rename(columns=self.static_map, inplace=True)
        return static_df

    def _static_data(self) -> pd.DataFrame:
        return self._static_table.copy()

    @functools.cached_property
    def _location_table(self) -> pd.DataFrame:
        """name, coordinates and areas of every gauge of the release"""
        df = pd.read_csv(os.path.join(self.static_dir, 'camels_br_location.txt'),
                         sep=' ', dtype={'gauge_id': str})
        return df.set_index('gauge_id')

    @functools.cached_property
    def _gsim_areas(self) -> pd.Series:
        """GSIM area (km2) of the selected catchments. Small, so that a pool
        worker gets it with the dataset instead of reading the file again."""
        return self._location_table['area_gsim'].loc[self._stations].astype(np.float64)

    def area(
            self,
            stations: Union[str, List[str]] = "all",
            source: str = "gsim",
    ) -> pd.Series:
        """
        Returns area (Km2) of all catchments as :obj:`pandas.Series`

        parameters
        ----------
        stations : str/list
            name/names of stations. Default is ``all``, which will return
            the area of all stations
        source : str
            source of the area, either ``gsim`` (the areas with which the
            release converts m3/s into mm/day) or ``ana``

        Returns
        --------
        pd.Series
            a :obj:`pandas.Series` whose indices are catchment ids and values
            are areas of corresponding catchments.

        Examples
        ---------
        >>> from aqua_fetch import CAMELS_BR
        >>> dataset = CAMELS_BR()
        >>> dataset.area()  # returns area of all stations
        >>> dataset.area('65100000')  # returns area of station whose id is 65100000
        >>> dataset.area(['65100000', '64075000'])  # returns area of two stations
        """
        if source not in ('gsim', 'ana'):
            raise ValueError(f"source must be 'gsim' or 'ana' but is {source!r}")

        stations = validate_attributes(stations, self.stations(), 'stations')

        if source == 'gsim':
            s = self._gsim_areas
        else:
            s = self._location_table['area_ana']

        s = s.loc[stations].astype(self.fp)
        s.name = 'area_km2'
        s.index.name = 'gauge_id'
        return s

    def stn_coords(
            self,
            stations: Union[str, List[str]] = 'all'
    ) -> pd.DataFrame:
        """
        returns coordinates of stations as :obj:`pandas.DataFrame`
        with ``long`` and ``lat`` as columns.

        Parameters
        ----------
        stations :
            name/names of stations. If not given, coordinates
            of all stations will be returned.

        Returns
        -------
        pd.DataFrame
            :obj:`pandas.DataFrame` with ``lat`` and ``long`` columns.

        Examples
        --------
        >>> dataset = CAMELS_BR()
        >>> dataset.stn_coords() # returns coordinates of all stations
        >>> dataset.stn_coords('65100000')  # returns coordinates of station whose id is 65100000
        >>> dataset.stn_coords(['65100000', '64075000'])  # returns coordinates of two stations
        """
        stations = validate_attributes(stations, self.stations(), 'stations')

        df = self._location_table.loc[stations, ['gauge_lat', 'gauge_lon']]
        df.columns = ['lat', 'long']
        return df

    @functools.cached_property
    def _time_extent(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """
        First and last date of the dynamic data, read from the data itself and
        cached. When the netCDF cache exists its time coordinate is used, since
        that is the index which is served. Otherwise the first and the last row
        of every time series file of this release is read, so that files ending
        on different dates are accounted for.
        """
        if self.dyn_fpath_exists and xr is not None:
            with xr.open_dataset(self.dyn_fpath) as ds:
                time = ds['time'].values
            return pd.Timestamp(time.min()), pd.Timestamp(time.max())

        dates = []
        for folder, suffix, _ in self._DYN_GROUPS[self.version].values():
            fdir = self._group_dir(folder)
            for fname in os.listdir(fdir):
                if fname.endswith(suffix):
                    dates += [_ymd(row) for row
                              in _first_and_last_row(os.path.join(fdir, fname))]

        if not dates:
            raise FileNotFoundError(f"no time series files found in {self.path}")

        if self._serves_ana:
            # both ends: a download which reaches before the release starts must
            # not be clipped away by fetch
            dates.extend(self._ana_extent)

        return min(dates), max(dates)

    @property
    def start(self) -> pd.Timestamp:
        return self._time_extent[0]

    @property
    def end(self) -> pd.Timestamp:
        return self._time_extent[1]

    def fetch_raw_streamflow(
            self,
            stations: Union[str, List[str]]
    ) -> pd.DataFrame:
        """
        Observed streamflow of the requested gauges as published, with the
        quality columns and including the gauges which are not among the 897
        selected catchments (4025 gauges in release 1.2, 3679 in 1.1).

        Parameters
        ----------
        stations : str/list
            gauge id or ids, see :meth:`all_stations`. ``all`` is rejected
            because the whole folder does not fit in memory.

        Returns
        -------
        pd.DataFrame
            one row per gauge and day, with a (gauge_id, time) MultiIndex and
            the columns ``q_cms_obs``, ``qual_control_by_ana`` (1 when ANA had
            reviewed the value at the time of the release) and ``qual_flag``
            (ANA's per day code, see :data:`ANA_COLUMNS`).

        Examples
        --------
        >>> from aqua_fetch import CAMELS_BR
        >>> dataset = CAMELS_BR()
        >>> q = dataset.fetch_raw_streamflow('10500000')
        """
        return self._read_extra_group('streamflow_m3s_raw', stations)

    def fetch_simulated_streamflow(
            self,
            stations: Union[str, List[str]]
    ) -> pd.DataFrame:
        """
        Streamflow of release 1.1 simulated with the MGB-SA model for 593 of the
        catchments. It is model output, not an observation, and release 1.2 is
        not distributed with it by this class.

        Parameters
        ----------
        stations : str/list
            gauge id or ids, see :meth:`all_stations`. ``all`` is rejected
            because the whole folder does not fit in memory.

        Returns
        -------
        pd.DataFrame
            one row per gauge and day with a (gauge_id, time) MultiIndex.

        Examples
        --------
        >>> from aqua_fetch import CAMELS_BR
        >>> dataset = CAMELS_BR(version='1.1')
        >>> q = dataset.fetch_simulated_streamflow('10500000')
        """
        if 'simulated_streamflow_m3s' not in self._EXTRA_GROUPS[self.version]:
            raise ValueError(
                f"the simulated streamflow is not available for release "
                f"{self.version}, it is only distributed with release 1.1")
        return self._read_extra_group('simulated_streamflow_m3s', stations)

    def _read_extra_group(
            self,
            group: str,
            stations: Union[str, List[str]]
    ) -> pd.DataFrame:
        """reads a group of files which is not part of the dynamic features"""
        if stations is None or (isinstance(stations, str) and stations == 'all'):
            raise ValueError(
                f"the gauges must be given explicitly: reading the whole "
                f"{group} folder does not fit in memory. See all_stations().")
        stations = _as_station_list(stations)

        folder, suffix, columns = self._EXTRA_GROUPS[self.version][group]
        available = set(self.all_stations(group))
        missing = [stn for stn in stations if stn not in available]
        if missing:
            raise ValueError(f"{len(missing)} gauges are not in {folder}: {missing}")

        frames = {}
        for stn in stations:
            fpath = os.path.join(self._group_dir(folder), f"{stn}{suffix}")
            df = pd.read_csv(fpath, sep=' ', usecols=['year', 'month', 'day', *columns])
            df.index = ymd_index(df['year'], df['month'], df['day'])
            df.index.name = 'time'
            frames[stn] = df[list(columns)].rename(columns=self._EXTRA_MAP)

        out = pd.concat(frames, names=['gauge_id'])
        return out

    # ------------------------------------------------------------ ANA update

    @property
    def _ana_dir(self) -> os.PathLike:
        """
        Folder with the streamflow downloaded by :meth:`update_streamflow`. It
        sits inside the folder of the release, because which days are missing,
        and therefore downloaded and served, follows the record of that release.
        """
        return os.path.join(self._release_dir, 'ana_streamflow')

    @property
    def _ana_manifest_fpath(self) -> os.PathLike:
        return os.path.join(self._ana_dir, 'manifest.json')

    @functools.cached_property
    def _ana_manifest(self) -> Dict:
        """what :meth:`update_streamflow` has downloaded, per gauge"""
        if not os.path.exists(self._ana_manifest_fpath):
            return {}
        with open(self._ana_manifest_fpath, 'r') as f:
            return json.load(f)

    def _ana_file(self, station: str) -> os.PathLike:
        """file which holds the ANA download of one gauge"""
        return os.path.join(self._ana_dir, f"{station}.csv")

    @functools.cached_property
    def _ana_stations(self) -> frozenset:
        """
        Gauges which ANA served with data. A frozenset of ids rather than a dict
        of paths, because the process pool of :meth:`_read_dynamic` ships this
        with the dataset once per station.
        """
        served = frozenset(
            stn for stn, info in self._ana_manifest.get('stations', {}).items()
            if info.get('status') == 'ok' and info.get('first')
            and os.path.exists(self._ana_file(stn)))
        incomplete = [stn for stn, info in self._ana_manifest.get('stations', {}).items()
                      if info.get('status') == 'ok' and not info.get('first')]
        if incomplete:
            warnings.warn(
                f"CAMELS_BR: {len(incomplete)} gauges of the ANA download were "
                f"written by an older version of this class and do not say which "
                f"days they hold, so they are not served. Call "
                f"update_streamflow() for them again.", UserWarning)
        return served

    @property
    def use_ana_update(self) -> bool:
        """whether :meth:`fetch` extends the released streamflow with the data
        downloaded by :meth:`update_streamflow`"""
        return self._use_ana_update

    @use_ana_update.setter
    def use_ana_update(self, value: bool):
        # start, end and the name of the netCDF cache describe the served
        # record, so they must be worked out again when this changes
        self._use_ana_update = bool(value)
        self.__dict__.pop('_time_extent', None)

    @property
    def _serves_ana(self) -> bool:
        """whether the served streamflow is extended with the ANA download"""
        return self._use_ana_update and bool(self._ana_stations)

    @property
    def _ana_stamp(self) -> str:
        """
        Short fingerprint of the ANA data on disk, from the window, the number of
        days and a digest of the values of every gauge. It names the netCDF
        cache: a date would not do, because two updates on the same day would
        share it, and a window alone would not, because ANA revises values
        without changing the days they cover.
        """
        served = self._ana_manifest.get('stations', {})
        fingerprint = repr(sorted(
            (stn, served[stn].get('first'), served[stn].get('last'),
             served[stn].get('days'), served[stn].get('digest'))
            for stn in self._ana_stations))
        return hashlib.sha1(fingerprint.encode()).hexdigest()[:8]

    @functools.cached_property
    def _ana_extent(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """first and last day for which the ANA download holds a discharge"""
        served = [info for stn, info in self._ana_manifest.get('stations', {}).items()
                  if stn in self._ana_stations and info.get('first')]
        if not served:
            raise ValueError("the ANA download holds no discharge")
        return (pd.Timestamp(min(info['first'] for info in served)),
                pd.Timestamp(max(info['last'] for info in served)))

    def _extend_with_ana(self, stn: str, stn_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills the days for which the release has no observed streamflow with the
        streamflow downloaded from ANA. A released value is never replaced.
        """
        ana = _read_ana_csv(self._ana_file(stn), only_with_value=True)
        if ana.empty:
            return stn_df

        q_cms, q_mm = observed_streamflow_cms(), observed_streamflow_mm()

        stn_df = stn_df.reindex(stn_df.index.union(ana.index))
        ana_cms = ana['streamflow_m3s'].reindex(stn_df.index).astype(self.fp)
        ana_cms.name = stn

        stn_df[q_cms] = stn_df[q_cms].combine_first(ana_cms)
        stn_df[q_mm] = stn_df[q_mm].combine_first(self.cms_to_mm(ana_cms).astype(self.fp))
        return stn_df

    def ana_streamflow(
            self,
            stations: Union[str, List[str]] = 'all'
    ) -> pd.DataFrame:
        """
        The streamflow downloaded by :meth:`update_streamflow`, exactly as ANA
        served it and before it is merged into the released record.

        Parameters
        ----------
        stations : str/list
            gauge id or ids, ``all`` (default) for every downloaded gauge.

        Returns
        -------
        pd.DataFrame
            one row per gauge and day with a (gauge_id, time) MultiIndex and the
            columns ``streamflow_m3s`` (m3/s), ``status`` (ANA's per day code as
            it is served, a string) and ``nivel_consistencia`` (1 raw,
            2 reviewed by ANA).

        Examples
        --------
        >>> from aqua_fetch import CAMELS_BR
        >>> dataset = CAMELS_BR()
        >>> dataset.update_streamflow(stations=['10500000'])
        >>> dataset.ana_streamflow('10500000')
        """
        if not self._ana_stations:
            raise ValueError("no streamflow has been downloaded from ANA yet, "
                             "call update_streamflow() first")

        stations = _as_station_list(stations)
        stations = validate_attributes(stations, sorted(self._ana_stations), 'stations')

        if len(stations) > 100:
            warnings.warn(
                f"reading the ANA download of {len(stations)} gauges, which is "
                f"one row per gauge and day; name the gauges to read fewer.",
                UserWarning)

        frames = {stn: _read_ana_csv(self._ana_file(stn)) for stn in stations}
        return pd.concat(frames, names=['gauge_id']).rename_axis(['gauge_id', 'time'])

    def _released_q_end(self, station: str) -> Union[pd.Timestamp, None]:
        """last day with an observed streamflow value in the released file of
        one gauge, None when the gauge has none. Read once per gauge."""
        if station not in self._released_q_ends:
            self._released_q_ends[station] = _last_valid_date(
                self._group_file(self._q_group, station), self._q_column)
        return self._released_q_ends[station]

    def update_streamflow(
            self,
            stations: Union[str, List[str]] = 'all',
            st: Union[str, pd.Timestamp] = None,
            en: Union[str, pd.Timestamp] = None,
            overwrite: bool = False,
            max_workers: int = 2,
            retries: int = 5,
            timeout: float = 180.0,
    ) -> pd.DataFrame:
        """
        Extends the observed streamflow of the release with data downloaded from
        ANA's HidroWeb web service, the source CAMELS-BR itself is built from.

        Only the days for which the release has no value are taken from ANA; a
        released value is never replaced, even where ANA has since revised it.
        The download is written to ``ana_streamflow`` inside the folder of the
        release and is served by :meth:`fetch` from here on (unless the dataset
        was created with ``use_ana_update=False``), which also moves
        :attr:`end`. The netCDF cache is rebuilt under a name which carries a
        fingerprint of the download, so an older cache is never served.

        A gauge which was downloaded before keeps what it has: the new days are
        merged into its file, and where ANA now serves a different value for a
        day that was downloaded earlier, the new value wins and the number of
        such days is reported.

        Parameters
        ----------
        stations : str/list
            gauge or gauges to update, ``all`` (default) for the 897 catchments.
        st : str/pd.Timestamp
            first day to request. By default the day after the last released
            value of each gauge, which is what has to be added. Pass an earlier
            day (e.g. ``dataset.start``) to also fill the gaps inside the
            released record.
        en : str/pd.Timestamp
            last day to request, today by default.
        overwrite : bool
            by default a gauge whose file already covers the requested period is
            not downloaded again, which makes an interrupted update resumable.
            True downloads every requested gauge again and replaces its file,
            which is how a record that ANA has revised is refreshed.
        max_workers : int
            number of concurrent requests. ANA answers 502 above two. Ignored
            when the dataset was created with ``processes=1``.
        retries : int
            attempts per gauge before it is reported as failed.
        timeout : float
            seconds to wait for one answer.

        Returns
        -------
        pd.DataFrame
            observed streamflow (m3/s) of the requested gauges, index is the
            date and columns are the gauge ids: the released values extended
            with ANA's.

        Notes
        -----
        ANA answers one gauge in 0.6 to 2 s, so updating all 897 gauges with the
        default two workers takes roughly 10 to 20 minutes. An interrupted run
        is resumed by calling the method again.

        Examples
        --------
        >>> from aqua_fetch import CAMELS_BR
        >>> dataset = CAMELS_BR()
        >>> q = dataset.update_streamflow(stations=['10500000'])
        """
        stations = validate_attributes(_as_station_list(stations), self.stations(),
                                       'stations')
        # a gauge named twice would be downloaded by two threads at once, which
        # write the same file
        stations = list(dict.fromkeys(stations))

        en = pd.Timestamp.today().normalize() if en is None else pd.Timestamp(en)
        known = dict(self._ana_manifest.get('stations', {}))

        windows, todo = {}, []
        for stn in stations:
            if st is not None:
                start = pd.Timestamp(st)
            else:
                last = self._released_q_end(stn)
                start = self.start if last is None else last + pd.Timedelta(days=1)
            if start > en:
                continue

            info = known.get(stn)
            has_file = info is not None and os.path.exists(self._ana_file(stn))
            if has_file and not overwrite:
                if pd.Timestamp(info['start']) <= start and pd.Timestamp(info['end']) >= en:
                    continue   # already downloaded, nothing to add
                # only the days which are not on disk yet are requested and the
                # answer is merged into the file, so an earlier, wider download
                # is never thrown away
                if pd.Timestamp(info['start']) <= start:
                    start = pd.Timestamp(info['end']) + pd.Timedelta(days=1)
                    if start > en:
                        continue
            windows[stn] = (start, en)
            todo.append(stn)

        skipped = len(stations) - len(todo)
        if self.verbosity:
            print(f"downloading {len(todo)} of {len(stations)} gauges from ANA "
                  f"up to {en.date()} ({skipped} are already on disk)")

        os.makedirs(self._ana_dir, exist_ok=True)
        workers = 1 if self.processes == 1 else max(1, int(max_workers))
        fetch = functools.partial(_ana_fetch_station, out_dir=self._ana_dir,
                                  retries=retries, timeout=timeout,
                                  merge=not overwrite)

        results = {}
        try:
            if workers == 1:
                for idx, stn in enumerate(todo):
                    results[stn] = fetch(stn, *windows[stn])
                    if self.verbosity and idx % 50 == 0:
                        print(f"downloaded {idx + 1}/{len(todo)} gauges")
            else:
                with cf.ThreadPoolExecutor(workers) as executor:
                    futures = {executor.submit(fetch, stn, *windows[stn]): stn
                               for stn in todo}
                    for idx, future in enumerate(cf.as_completed(futures)):
                        stn = futures[future]
                        results[stn] = future.result()
                        if self.verbosity and idx % 50 == 0:
                            print(f"downloaded {idx + 1}/{len(todo)} gauges")
        finally:
            # written even when the run is interrupted, so that a partial
            # download is not requested again from the start
            self._write_ana_manifest(known, results, windows, bool(results),
                                     merged=not overwrite)

        failed = [stn for stn, info in results.items() if info['status'] == 'failed']
        no_data = [stn for stn, info in results.items() if info['status'] == 'no data']
        revised = sum(info['changed'] for info in results.values())
        withdrawn = sum(info['withdrawn'] for info in results.values())
        dropped = sum(info['dropped'] for info in results.values())
        if withdrawn:
            warnings.warn(
                f"CAMELS_BR: {withdrawn} days which an earlier download held are "
                f"no longer served by ANA and were removed from the record.",
                UserWarning)
        if dropped:
            warnings.warn(
                f"CAMELS_BR: overwrite=True replaced the file of every requested "
                f"gauge, which dropped {dropped} days outside the requested "
                f"period that an earlier, wider download held. Call "
                f"update_streamflow() without overwrite, or with the same st, to "
                f"get them back.", UserWarning)
        if results:
            warnings.warn(
                f"CAMELS_BR: ANA served data for "
                f"{len(results) - len(failed) - len(no_data)} of {len(results)} "
                f"requested gauges, has no discharge for {len(no_data)} and did "
                f"not answer for {len(failed)} "
                f"({failed[:5]}{' ...' if len(failed) > 5 else ''}), which are "
                f"requested again on the next call. {revised} days which were "
                f"downloaded earlier now have a different value at ANA and were "
                f"replaced with it. Released values are kept, ANA fills only the "
                f"days without one.", UserWarning)
        elif self.verbosity:
            print(f"every requested gauge is already downloaded up to {en.date()}, "
                  f"use overwrite=True to download them again")

        return self._extended_q(stations)

    def _write_ana_manifest(self, known: Dict, results: Dict, windows: Dict,
                            downloaded: bool, merged: bool = True):
        """
        Writes the manifest of the ANA download and drops the cached properties
        which describe it. ``known`` is what the manifest held before the run.
        With ``merged`` the file of a gauge holds both windows, otherwise it was
        replaced and only the new window describes it.
        """
        for stn, info in results.items():
            start, en = windows[stn]
            previous = dict(known.get(stn, {}))

            if not info['wrote'] and previous.get('status') == 'ok':
                # the request failed, so the file of the earlier download is
                # untouched and still describes what is served
                previous['last_attempt'] = info['status']
                known[stn] = previous
                continue

            if merged and previous and info['status'] == 'ok' and previous.get('status') == 'ok':
                # the file now holds the union of both windows
                start = min(start, pd.Timestamp(previous['start']))
                en = max(en, pd.Timestamp(previous['end']))
            known[stn] = {'start': str(start.date()), 'end': str(en.date()),
                          'status': info['status'], 'days': info['days'],
                          'first': info['first'], 'last': info['last'],
                          'digest': info['digest'],
                          'downloaded_on': str(pd.Timestamp.today().date())}

        manifest = dict(self._ana_manifest)
        manifest['stations'] = known
        manifest['source'] = ANA_SERVICE_URL
        if downloaded:
            manifest['downloaded_on'] = str(pd.Timestamp.today().date())
        manifest.setdefault('downloaded_on', str(pd.Timestamp.today().date()))

        os.makedirs(self._ana_dir, exist_ok=True)
        with open(self._ana_manifest_fpath, 'w') as f:
            json.dump(manifest, f, indent=1)

        # the served record, its extent and the name of its cache all changed
        for cached in ('_ana_manifest', '_ana_stations', '_ana_extent', '_time_extent'):
            self.__dict__.pop(cached, None)
        return

    def _extended_q(self, stations: List[str]) -> pd.DataFrame:
        """
        Observed streamflow (m3/s) of ``stations``, released values extended with
        the ANA download, whether or not this dataset serves it. The netCDF cache
        of the new record is built on the way, unless the dataset does not serve
        the download or was created with ``to_netcdf=False``.
        """
        served = self.use_ana_update
        self.use_ana_update = True
        try:
            if not served:
                warnings.warn(
                    "CAMELS_BR: the download is on disk but this dataset was "
                    "created with use_ana_update=False, so fetch() keeps serving "
                    "the released record. The extended streamflow is only "
                    "returned by this call.", UserWarning)
            elif self._incomplete:
                warnings.warn(
                    "CAMELS_BR: the extended streamflow is not cached because "
                    "files of the release are missing.", UserWarning)
            else:
                self._drop_stale_ana_caches()
                self._maybe_to_netcdf()
            _, q = self.fetch_stations_features(
                stations, dynamic_features=observed_streamflow_cms(),
                as_dataframe=True)
        finally:
            self.use_ana_update = served
        return pd.DataFrame({stn: df[observed_streamflow_cms()] for stn, df in q.items()})

    def _drop_stale_ana_caches(self):
        """deletes the netCDF caches of earlier ANA downloads of this release and
        precision, which nothing reads any more"""
        pattern = os.path.join(glob.escape(self.path), f"{self._cache_stem}_ana*.nc")
        for fpath in glob.glob(pattern):
            if os.path.basename(fpath) != self.dyn_fname:
                if self.verbosity:
                    print(f"removing the cache of an earlier ANA download: {fpath}")
                os.remove(fpath)
        return


class CABra(_RainfallRunoff):
    """
    Reads and fetches CABra dataset which is catchment attribute dataset
    following the work of `Almagro et al., 2021 <https://doi.org/10.5194/hess-25-3105-2021>`_
    This dataset consists of 87 static and 13 dynamic features of 735 Brazilian
    catchments. The temporal extent is from 1980 to 2020. The dyanmic features
    consist of daily hydro-meteorological time series

    Examples
    ---------
    >>> from aqua_fetch import CABra
    >>> dataset = CABra()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='92', as_dataframe=True)
    >>> df = dynamic['92'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (10956, 13)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       735
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (73 out of 735)
       73
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(10956, 13), (10956, 13), (10956, 13),... (10956, 13), (10956, 13)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('92', as_dataframe=True,
    ...  dynamic_features=['pcp_mm_ens', 'airtemp_C_ens_max', 'pet_mm_pm', 'rh_%_ens', 'q_cms_obs'])
    >>> dynamic['92'].shape
       (10956, 4)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='92', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['92'].shape
    ((1, 87), 1, (10956, 13))

    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 10956, 'dynamic_features': 13})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (735, 2)
    >>> dataset.stn_coords('92')  # returns coordinates of station whose id is 92
        -2.509	-47.764
    >>> dataset.stn_coords(['92', '5'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('92')
    # get coordinates of two stations
    >>> dataset.area(['92', '5'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('92')

    """

    url = 'https://zenodo.org/record/7612350'

    def __init__(self,
                 path=None,
                 overwrite=False,
                 to_netcdf: bool = True,
                 met_src: str = 'ens',
                 **kwargs):
        """
        Parameters
        ----------
        path : str
            If the data is alredy downloaded then provide the complete
            path to it. If None, then the data will be downloaded.
            The data is downloaded once and therefore susbsequent
            calls to this class will not download the data unless
            ``overwrite`` is set to True.
        overwrite : bool
            If the data is already down then you can set it to True,
            to make a fresh download.
        to_netcdf : bool
            whether to convert all the data into one netcdf file or not.
            This will fasten repeated calls to fetch etc but will
            require netCDF4 package as well as xarry.
        met_src : str
            source of meteorological data, must be one of
            ``ens``, ``era5`` or ``ref``.
        """
        super(CABra, self).__init__(path=path,
                                    to_netcdf=to_netcdf,
                                    **kwargs)
        self.path = path
        self.met_src = met_src
        self._download(overwrite=overwrite)

        self._dynamic_features = self.__dynamic_features()
        self._static_features = self.__static_features()

        self._maybe_to_netcdf()

    @property
    def dyn_fname(self) -> Union[str, os.PathLike]:
        """
        name of the .nc file which contains dynamic features. This file is created during dataset initialization
        only if to_netcdf is True and xarray is installed and the file does not already exists. The creation of this
        file can take some time however it leads to faster I/O operations.
        """
        return cache_name(self.name.lower() + f"_{self.timestep}_{self.met_src}.nc")

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(self.path, "CABra_boundaries", "CABra_boundaries.shp")

    @property
    def boundary_id_map(self) -> str:
        """
        Name of the attribute in the boundary (.shp/.gpkg) file that
        will be used to map the catchment/station id to the geometry of the
        catchment/station. This is used to create the boundary id map.
        """
        return "ID_CABra"
    
    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'catch_area': catchment_area(),
                'catch_slope': slope('perc'),
                'latitude': gauge_latitude(),
                'longitude': gauge_longitude(),
        }

    @property
    def dyn_map(self):
        # table 3 in the paper https://hess.copernicus.org/articles/25/3105/2021/#&gid=1&pid=1
        return {
            'Streamflow': observed_streamflow_cms(),
            'tmin_ens': min_air_temp_with_specifier('ens'),
            'tmax_ens': max_air_temp_with_specifier('ens'),
            'tmin_era5': min_air_temp_with_specifier('era5'),
            'tmax_era5': max_air_temp_with_specifier('era5'),
            'tmin_ref': min_air_temp_with_specifier('ref'),
            'tmax_ref': max_air_temp_with_specifier('ref'),
            'p_ens': total_precipitation_with_specifier('ens'),
            'p_ref': total_precipitation_with_specifier('ref'),
            'p_era5': total_precipitation_with_specifier('era5'),
            'rh_ens': mean_rel_hum_with_specifier('ens'),
            'rh_era5': mean_rel_hum_with_specifier('era5'),
            'rh_ref': mean_rel_hum_with_specifier('ref'),
            'wnd_ens': mean_windspeed_with_specifier('ens'),
            'wnd_era5': mean_windspeed_with_specifier('era5'),
            'wnd_ref': mean_windspeed_with_specifier('ref'),
            'et_ens': actual_evapotranspiration_with_specifier('ens'),
            'pet_pm': total_potential_evapotranspiration_with_specifier('pm'),
            'pet_pt': total_potential_evapotranspiration_with_specifier('pt'),
            'pet_hg': total_potential_evapotranspiration_with_specifier('hg'),
            'srad_ens': solar_radiation_with_specifier('ens'),  # MJ/m2/day -> W/m2 in dyn_factors
            'srad_era5': solar_radiation_with_specifier('era5'),
            'srad_ref': solar_radiation_with_specifier('ref'),
        }

    @property
    def dyn_factors(self) -> Dict[str, float]:
        # The units row inside every ``CABra_*_climate_*.txt`` gives srad as
        # "MJ m-2" accumulated over the day, while the canonical name promises
        # W m-2. Only the column of the selected ``met_src`` is present in a
        # given frame; the other keys are skipped.
        return {
            solar_radiation_with_specifier('ens'): MJ_M2_DAY_TO_WM2,
            solar_radiation_with_specifier('era5'): MJ_M2_DAY_TO_WM2,
            solar_radiation_with_specifier('ref'): MJ_M2_DAY_TO_WM2,
        }

    @property
    def dyn_generators(self):
        return {
# new column to be created : function to be applied, inputs
mean_air_temp_with_specifier(self.met_src): (self.mean_temp, (min_air_temp_with_specifier(self.met_src), max_air_temp_with_specifier(self.met_src))),
#mean_air_temp_with_specifier('era5'): (self.mean_temp, (min_air_temp_with_specifier('era5'), max_air_temp_with_specifier('era5'))),
#mean_air_temp_with_specifier('ref'): (self.mean_temp, (min_air_temp_with_specifier('ref'), max_air_temp_with_specifier('ref'))),
        }

    @property
    def q_path(self):
        return os.path.join(self.path, "CABra_streamflow_daily_series",
                            "CABra_daily_streamflow")

    @property
    def attr_path(self):
        return os.path.join(self.path, 'CABra_attributes', 'CABra_attributes')

    @property
    def dynamic_features(self) -> List[str]:
        return self._dynamic_features

    def __dynamic_features(self) -> List[str]:
        stn = self.stations()[0]
        df = pd.concat([self._read_q_from_csv(stn), self._read_meteo_from_csv(stn, self.met_src)], axis=1)
        cols = df.columns.to_list()
        cols = [col for col in cols if col not in ['Year', 'Month', 'Day']]
        return cols

    @property
    def static_features(self) -> List[str]:
        """names of static features"""
        return self._static_features

    def __static_features(self) -> List[str]:
        df = pd.concat(
            [
                self.climate_attrs(),
                self.general_attrs(),
                self.geology_attrs(),
                self.gw_attrs(),
                self.hydro_distrub_attrs(),
                self.lc_attrs(),
                self.soil_attrs(),
                self.q_attrs(),
                self.topology_attrs()], axis=1)
        
        # drop duplicate columns from df which might have come due to concatenation
        df = df.loc[:, ~df.columns.duplicated()]

        df.rename(columns = self.static_map, inplace=True)
        return df.columns.to_list()

    def stations(self) -> List[str]:
        return self.add_attrs().index.astype(str).to_list()

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp("1980-10-01")

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp("2010-09-30")

    def add_attrs(self) -> pd.DataFrame:
        """
        Returns additional catchment attributes
        """
        fpath = os.path.join(self.attr_path, "CABra_additional_attributes.txt")

        dtypes = {"CABra_ID": int,  # todo shouldn't it be str?
                  "ANA_ID": int,
                  "longitude_centroid": np.float32,
                  "latitude_centroid": np.float32,
                  "dist_coast": np.float32}

        add_attributes = pd.read_csv(fpath, sep='\t',
                                     names=list(dtypes.keys()),
                                     dtype=dtypes,
                                     header=4)
        add_attributes.index = add_attributes.pop('CABra_ID')
        return add_attributes

    def climate_attrs(self) -> pd.DataFrame:
        """
        returns climate attributes for all catchments
        """
        fpath = os.path.join(self.attr_path,
                             "CABra_climate_attributes.txt")

        dtypes = {"CABra_ID": int,  # todo shouldn't it be str?
                  "ANA_ID": int,
                  "clim_p": np.float32,
                  "clim_tmin": np.float32,
                  "clim_tmax": np.float32,
                  "clim_rh": np.float32,
                  "clim_wind": np.float32,
                  "clim_srad": np.float32,
                  "clim_et": np.float32,
                  "clim_pet": np.float32,
                  "aridity_index": np.float32,
                  "p_seasonality": np.float32,
                  "clim_quality": int,
                  }

        clim_attrs = pd.read_csv(fpath, sep='\t',
                                 names=list(dtypes.keys()),
                                 dtype=dtypes,
                                 encoding_errors='ignore',
                                 header=6)
        clim_attrs.index = clim_attrs.pop('CABra_ID')
        return clim_attrs

    def general_attrs(self) -> pd.DataFrame:
        """
        returns general attributes for all catchments
        """
        fpath = os.path.join(self.attr_path,
                             "CABra_general_attributes.txt")

        dtypes = {"CABra_ID": int,  # todo shouldn't it be str?
                  "ANA_ID": int,
                  "longitude": np.float32,
                  "latitude": np.float32,
                  "gauge_hreg": str,
                  "gauge_biome": str,
                  "gauge_state": str,
                  "missing_data": np.float32,
                  "series_length": np.float32,
                  "quality_index": np.float32,
                  }

        gen_attrs = pd.read_csv(fpath,
                                sep='\t',
                                names=list(dtypes.keys()),
                                dtype=dtypes,
                                encoding_errors='ignore',
                                header=6)
        gen_attrs.index = gen_attrs.pop('CABra_ID')
        return gen_attrs

    def geology_attrs(self) -> pd.DataFrame:
        """
        returns geological attributes for all catchments
        """
        fpath = os.path.join(self.attr_path,
                             "CABra_geology_attributes.txt")

        dtypes = {"CABra_ID": int,
                  "ANA_ID": int,
                  "catch_lith": str,
                  "sub_porosity": np.float32,
                  "sub_permeability": np.float32,
                  "sub_hconduc": np.float32,
                  }

        gen_attrs = pd.read_csv(fpath,
                                sep='\t',
                                names=list(dtypes.keys()),
                                dtype=dtypes,
                                encoding_errors='ignore',
                                header=6)
        gen_attrs.index = gen_attrs.pop('CABra_ID')
        return gen_attrs

    def gw_attrs(self) -> pd.DataFrame:
        """
        returns groundwater attributes for all catchments


        """
        fpath = os.path.join(self.attr_path,
                             "CABra_groundwater_attributes.txt")

        dtypes = {"CABra_ID": int,
                  "ANA_ID": int,
                  "aquif_name": str,
                  "aquif_type": str,
                  "catch_wtd": np.float32,
                  "catch_hand": np.float32,
                  "hand_class": str,
                  "well_number": int,
                  "well_static": str,
                  "well_dynamic": str,
                  }

        gen_attrs = pd.read_csv(fpath,
                                sep='\t',
                                names=list(dtypes.keys()),
                                dtype=dtypes,
                                encoding_errors='ignore',
                                header=7)
        gen_attrs.index = gen_attrs.pop('CABra_ID')
        return gen_attrs

    def hydro_distrub_attrs(self) -> pd.DataFrame:
        """
        returns geological attributes for all catchments
        """
        fpath = os.path.join(self.attr_path,
                             "CABra_hydrologic_disturbance_attributes.txt")

        dtypes = {"CABra_ID": int,
                  "ANA_ID": int,
                  "dist_urban": int,
                  "cover_urban": np.float32,
                  "cover_crops": np.float32,
                  "res_number": int,
                  "res_area": np.float32,
                  "res_volume": np.float32,
                  "res_regulation": np.float32,
                  "water_demand": int,
                  "hdisturb_index": np.float32,
                  }

        gen_attrs = pd.read_csv(fpath,
                                sep='\t',
                                names=list(dtypes.keys()),
                                dtype=dtypes,
                                encoding_errors='ignore',
                                header=8)
        gen_attrs.index = gen_attrs.pop('CABra_ID')
        return gen_attrs

    def lc_attrs(self) -> pd.DataFrame:
        """
        returns land cover attributes for all catchments
        """
        fpath = os.path.join(self.attr_path,
                             "CABra_land-cover_attributes.txt")

        dtypes = {"CABra_ID": int,
                  "ANA_ID": int,
                  "cover_main": str,
                  "cover_bare": np.float32,
                  "cover_forest": np.float32,
                  "cover_crops": np.float32,
                  "cover_grass": np.float32,
                  "cover_moss": np.float32,
                  "cover_shrub": np.float32,
                  "cover_urban": np.float32,
                  "cover_snow": np.float32,
                  "cover_waterp": np.float32,
                  "cover_waters": np.float32,
                  "ndvi_djf": np.float32,
                  "ndvi_mam": np.float32,
                  "ndvi_jja": np.float32,
                  "ndvi_son": np.float32,
                  }

        lc_attrs = pd.read_csv(fpath,
                               sep='\t',
                               names=list(dtypes.keys()),
                               dtype=dtypes,
                               encoding_errors='ignore',
                               header=6)
        lc_attrs.index = lc_attrs.pop('CABra_ID')
        return lc_attrs

    def soil_attrs(self) -> pd.DataFrame:
        """
        returns soil attributes for all catchments
        """
        fpath = os.path.join(self.attr_path,
                             "CABra_soil_attributes.txt")

        dtypes = {"CABra_ID": int,
                  "ANA_ID": int,
                  "soil_type": str,
                  "soil_textclass": str,
                  "soil_sand": np.float32,
                  "soil_silt": np.float32,
                  "soil_clay": np.float32,
                  "soil_carbon": np.float32,
                  "soil_bulk": np.float32,
                  "soil_depth": np.float32,
                  }

        soil_attrs = pd.read_csv(fpath,
                                 sep='\t',
                                 names=list(dtypes.keys()),
                                 dtype=dtypes,
                                 encoding_errors='ignore',
                                 header=7)
        soil_attrs.index = soil_attrs.pop('CABra_ID')
        return soil_attrs

    def q_attrs(self) -> pd.DataFrame:
        """
        returns streamflow attributes for all catchments
        """
        fpath = os.path.join(self.attr_path,
                             "CABra_streamflow_attributes.txt")

        dtypes = {"CABra_ID": int,
                  "ANA_ID": int,
                  "q_mean": np.float32,
                  "q_1": np.float32,
                  "q_5": np.float32,
                  "q_95": np.float32,
                  "q_99": np.float32,
                  "q_lf": np.float32,
                  "q_ld": np.float32,
                  "q_hf": np.float32,
                  "q_hd": np.float32,
                  "q_hfd": np.float32,
                  "q_zero": int,
                  "q_cv": np.float32,
                  "q_lcv": np.float32,
                  "q_hcv": np.float32,
                  "q_elasticity": np.float32,
                  "fdc_slope": np.float32,
                  "baseflow_index": np.float32,
                  'runoff_coef': np.float32
                  }
        names = list(dtypes.keys())
        dtypes.pop('q_cv')
        dtypes.pop('q_mean')
        dtypes.pop('q_lcv')
        dtypes.pop('fdc_slope')
        q_attrs = pd.read_csv(fpath,
                              sep='\t',
                              names=names,
                              dtype=dtypes,
                              encoding_errors='ignore',
                              header=7)

        q_attrs.index = q_attrs.pop('CABra_ID')
        return q_attrs

    def topology_attrs(self) -> pd.DataFrame:
        """
        returns topology attributes for all catchments
        """
        fpath = os.path.join(self.attr_path,
                             "CABra_topography_attributes.txt")

        dtypes = {"CABra_ID": int,
                  "ANA_ID": int,
                  "catch_area": np.float32,
                  "elev_mean": np.float32,
                  "elev_min": np.float32,
                  "elev_max": np.float32,
                  "elev_gauge": np.float32,
                  "catch_slope": np.float32,
                  "catch_order": int,
                  }

        gen_attrs = pd.read_csv(fpath,
                                sep='\t',
                                names=list(dtypes.keys()),
                                dtype=dtypes,
                                encoding_errors='ignore',
                                header=7)
        gen_attrs.index = gen_attrs.pop('CABra_ID')
        return gen_attrs

    def _read_q_from_csv(self, station: str) -> pd.DataFrame:
        q_fpath = os.path.join(self.q_path, f"CABra_{station}_streamflow.txt")

        df = pd.read_csv(q_fpath, sep='\t',
                         header=8,
                         names=['Year', 'Month', 'Day', 'Streamflow', 'Quality'],
                         dtype={'Year': np.int16,
                                'Month': np.int16,
                                'Day': np.int16,
                                # 'Streamflow': np.float32,
                                'Quality': np.int16}
                         )
        df.rename(columns=self.dyn_map, inplace=True)
        df[observed_streamflow_cms()] = df[observed_streamflow_cms()].astype(np.float32)
        return df

    def _read_meteo_from_csv(
            self,
            station: str,
            source="ens") -> pd.DataFrame:

        meteo_path = os.path.join(self.path,
                                  'CABra_climate_daily_series',
                                  'climate_daily',
                                  source
                                  )
        meteo_fpath = os.path.join(meteo_path,
                                   f"CABra_{station}_climate_{source.upper()}.txt")

        dtypes = {"Year": int,
                  "Month": int,
                  "Day": int,
                  f"p_{source}": np.float32,
                  f"tmin_{source}": np.float32,
                  f"tmax_{source}": np.float32,
                  f"rh_{source}": np.float32,
                  f"wnd_{source}": np.float32,
                  f"srad_{source}": np.float32,
                  f"et_{source}": np.float32,
                  "pet_pm": np.float32,
                  "pet_pt": np.float32,
                  "pet_hg": np.float32}

        if source == "ref" and station in [
            '1', '2', '3', '4', '5', '6', '7', '8', '9',
            '15', '17', '18', '19', '27', '28', '34', '526',
            '564', '567', '569'
        ]:
            df = pd.DataFrame(columns=list(dtypes.keys()))
        else:
            df = pd.read_csv(meteo_fpath,
                             sep="\t",
                             names=list(dtypes.keys()),
                             dtype=dtypes,
                             header=12)

        df.rename(columns=self.dyn_map, inplace=True)

        self._apply_dyn_factors(df)

        for new_col, (func, old_col) in self.dyn_generators.items():
            if isinstance(old_col, str):
                if old_col in df.columns:
                    # name of Series to func should be same as station id
                    df[new_col] = func(pd.Series(df[old_col], name=station))
            else:
                assert isinstance(old_col, tuple)
                if all([col in df.columns for col in old_col]):
                    # feed all old_cols to the function
                    df[new_col] = func(*[pd.Series(df[col], name=station) for col in old_col])
        return df

    def _static_data(self)->pd.DataFrame:
        df = pd.concat([self.climate_attrs(),
                        self.general_attrs(),
                        self.geology_attrs(),
                        self.gw_attrs(),
                        self.hydro_distrub_attrs(),
                        self.lc_attrs(),
                        self.soil_attrs(),
                        self.q_attrs(),
                        self.topology_attrs()], axis=1)

        df.index = df.index.astype(str)
        # drop duplicate columns
        df = df.loc[:, ~df.columns.duplicated()].copy()

        df.rename(columns=self.static_map, inplace=True)

        return df

    def _read_dynamic(
            self,
            stations,
            dynamic_features,
            st=None,
            en=None
    ) -> dict:

        st, en = self._check_length(st, en)
        features = validate_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')

        if self.verbosity>1:
            print(f"getting data for {len(dynamic_features)} and for {len(stations)} stations")

        # qs and meteo data has different index

        if self.verbosity>2:
            print("getting streamflow data")

        qs = [self._read_q_from_csv(station=station) for station in stations]
        q_idx = pd.to_datetime(
            qs[0]['Year'].astype(str) + '-' + qs[0]['Month'].astype(str) + '-' + qs[0]['Day'].astype(str))

        if self.verbosity>2:
            print("getting meteo data")

        meteos = [
            self._read_meteo_from_csv(station=station, source=self.met_src) for station in stations]
        # todo : this will be correct only if we are getting data for all stations
        # but what if we want to get data for some random stations?
        # 10 because first 10 stations don't have data for "ref" source
        if len(meteos) < 10:
            met_idx = pd.to_datetime(
                meteos[0]['Year'].astype(str) + '-' + meteos[0]['Month'].astype(str) + '-' + meteos[0]['Day'].astype(
                    str))
        else:
            met_idx = pd.to_datetime(
                meteos[10]['Year'].astype(str) + '-' + meteos[10]['Month'].astype(str) + '-' + meteos[10]['Day'].astype(
                    str))

        met_cols = [col for col in meteos[0].columns if col not in ['Year', 'Month', 'Day']]

        if self.verbosity>2:
            print("putting data in dictionary")

        dyn = {}

        for stn, q, meteo in zip(self.stations(), qs, meteos):

            if len(meteo) == 0:
                meteo = pd.DataFrame(meteo, index=met_idx)
            else:
                meteo.index = met_idx
            q.index = q_idx

            stn_df = pd.concat(
                [meteo[met_cols].astype(np.float32), q[['Quality', observed_streamflow_cms()]]], axis=1)[features]

            stn_df.index.name = 'time'
            stn_df.columns.name = 'dynamic_features'

            dyn[stn] = stn_df.loc[st:en]

        return dyn
