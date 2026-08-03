
import gc
import os
import shutil
import warnings
from functools import partial
import concurrent.futures as cf
from typing import Union, List, Dict, Tuple

import numpy as np
import pandas as pd

from .._backend import xarray as xr
from .._backend import netCDF4
from .._backend import fiona

from ..utils import get_cpus
from ..utils import validate_attributes, download, unzip
from .utils import _RainfallRunoff, _handle_dynamic
from .._geom_utils import laea_to_wgs84, lcc_to_wgs84

from ._map import (
    observed_streamflow_cms,
    min_air_temp,
    min_air_temp_with_specifier,
    max_air_temp_with_specifier,
    mean_air_temp_with_specifier,
    max_air_temp,
    mean_air_temp,
    total_precipitation,
    snow_water_equivalent,
    solar_radiation,
    max_solar_radiation,
    max_thermal_radiation,
    mean_thermal_radiation,
    u_component_of_wind_at_10m,
    v_component_of_wind_at_10m,
    max_dewpoint_temperature_at_2m,
    min_dewpoint_temperature_at_2m,
    mean_dewpoint_temperature_at_2m,
    mean_air_pressure,
    total_potential_evapotranspiration,
)

from ._map import (
    catchment_area,
    catchment_area_with_specifier,
    gauge_latitude,
    gauge_longitude,
    slope
    )

SEP = os.sep

# todo : currently when saving .nc files for each variable, we first fetch data for all meteo
# variables and then save a single variable and extract data for all meteos again. This
# is extremely inefficient. We should not make multiple calls to .fetch we saving .ncs
# todo : try to use mfopen_dataset instead of opening multiple .nc files separately
# without xarray, LamaHCE seems to be much faster


# columns that only encode the timestamp and are therefore never exposed as
# dynamic features
_CE_DATE_COLS = ('YYYY', 'MM', 'DD', 'hh', 'mm', 'DOY', 'HOD')

# quality/provenance flags shipped alongside ``qobs`` in D_gauges. They are not
# hydro-meteorological observations, so they are not dynamic features. They stay
# reachable in their raw form through :meth:`LamaHCE.fetch_stn_q_raw`.
_CE_Q_FLAG_COLS = ('ckhs', 'qceq', 'qcol')

# value used by LamaH-CE to mark a gap in the runoff time series which could not
# be interpolated (Info_english/2_Timeseries.txt, point 4)
_CE_Q_NODATA = -999.0

# dtypes of every column that can appear in a LamaH-CE csv (meteorological or
# runoff, daily or hourly). Declaring them avoids pandas' type inference and
# halves the memory of the parsed frame. float32 is safe here: the widest
# quantity is ``surf_press`` (~1e5 Pa, 5 significant digits) while float32
# carries ~7.
_CE_DTYPES = {
    'YYYY': np.int32, 'MM': np.int32, 'DD': np.int32,
    'hh': np.int32, 'mm': np.int32, 'DOY': np.int32, 'HOD': np.int32,
    # daily meteorological columns
    '2m_temp_max': np.float32, '2m_temp_mean': np.float32, '2m_temp_min': np.float32,
    '2m_dp_temp_max': np.float32, '2m_dp_temp_mean': np.float32, '2m_dp_temp_min': np.float32,
    'surf_net_solar_rad_max': np.float32, 'surf_net_solar_rad_mean': np.float32,
    'surf_net_therm_rad_max': np.float32, 'surf_net_therm_rad_mean': np.float32,
    # hourly meteorological columns
    '2m_temp': np.float32, '2m_dp_temp': np.float32,
    'surf_net_solar_rad': np.float32, 'surf_net_therm_rad': np.float32,
    # common to both timesteps
    '10m_wind_u': np.float32, '10m_wind_v': np.float32,
    'fcst_alb': np.float32, 'lai_high_veg': np.float32, 'lai_low_veg': np.float32,
    'swe': np.float32, 'surf_press': np.float32, 'total_et': np.float32,
    'prec': np.float32, 'volsw_123': np.float32, 'volsw_4': np.float32,
    # runoff. The quality flags are deliberately absent: they are only ever read
    # by fetch_stn_q_raw(), which reproduces the file with pandas' own dtypes.
    'qobs': np.float32,
}


def _ymd_index(
        year: np.ndarray,
        month: np.ndarray,
        day: np.ndarray,
        hour: np.ndarray = None,
        minute: np.ndarray = None
) -> pd.DatetimeIndex:
    """
    Builds a :obj:`pandas.DatetimeIndex` from integer year/month/day(/hour/minute)
    columns using numpy's datetime64 arithmetic.

    This is ~8x faster than ``pd.PeriodIndex(...).to_timestamp()`` on the 341856
    row hourly files and yields a bit-identical index (asserted in
    ``tests/rr/test_lamah.py::test_lamahce_index_construction``).
    """
    year = np.asarray(year, dtype='int64')
    month = np.asarray(month, dtype='int64')
    day = np.asarray(day, dtype='int64')

    months = (year - 1970).astype('datetime64[Y]').astype('datetime64[M]') + (month - 1)
    idx = months.astype('datetime64[D]') + (day - 1)

    if hour is not None:
        idx = idx.astype('datetime64[m]') + np.asarray(hour, dtype='int64') * 60
        if minute is not None:
            idx = idx + np.asarray(minute, dtype='int64')

    return pd.DatetimeIndex(idx.astype('datetime64[ns]'))


def _read_ce_csv(
        fpath: str,
        timestep: str,
        usecols: List[str]
) -> pd.DataFrame:
    """
    Reads one LamaH-CE csv (meteorological or runoff) and returns it indexed by
    time with the date columns removed.

    ``usecols`` must contain the date columns needed to build the index; only
    the listed columns are parsed off disk.
    """
    df = pd.read_csv(fpath, sep=';', usecols=usecols,
                     dtype={k: v for k, v in _CE_DTYPES.items() if k in usecols})

    if timestep == 'H':
        df.index = _ymd_index(df['YYYY'], df['MM'], df['DD'], df['hh'], df['mm'])
    else:
        df.index = _ymd_index(df['YYYY'], df['MM'], df['DD'])

    return df.drop(columns=[c for c in _CE_DATE_COLS if c in df.columns])


_CE_FREQ = {'D': 'D', 'H': 'h'}


def _first_last_stamps(fpath: str, timestep: str) -> tuple:
    """
    First and last timestamp of a LamaH-CE csv, read with two seeks instead of
    parsing the whole file.

    This only reports the two timestamps that are written in the file; it makes
    no assumption about what lies between them. Do not use it to reconstruct an
    index - a file with a gap and a compensating duplicate would be mis-stamped
    without any cheap way of noticing.
    """
    with open(fpath, 'rb') as fh:
        fh.readline()                                   # header
        first = fh.readline().decode().split(';')
        size = fh.seek(0, os.SEEK_END)
        fh.seek(max(0, size - 4096))
        last = fh.read().decode().strip().splitlines()[-1].split(';')

    def _stamp(fields):
        year, month, day = int(fields[0]), int(fields[1]), int(fields[2])
        if timestep == 'H':
            return pd.Timestamp(year=year, month=month, day=day,
                                hour=int(fields[3]), minute=int(fields[4]))
        return pd.Timestamp(year=year, month=month, day=day)

    return _stamp(first), _stamp(last)


def _read_ce_stn(spec: Dict, station: str) -> pd.DataFrame:
    """
    Reads the dynamic data of a single LamaH-CE station.

    This is a module level function (and ``spec`` is a small dict of paths and
    column names) so that handing it to a :obj:`concurrent.futures.ProcessPoolExecutor`
    does not pickle the dataset instance - and with it every cached table - once
    per station.
    """
    frames = []
    n_nodata = 0

    if spec['met_usecols'] is not None:
        frames.append(_read_ce_csv(
            os.path.join(spec['met_dir'], f"ID_{station}.csv"),
            spec['timestep'], spec['met_usecols']))

    if spec['q_usecols'] is not None:
        q_fpath = os.path.join(spec['q_dir'], f"ID_{station}.csv")
        if os.path.exists(q_fpath):
            q = _read_ce_csv(q_fpath, spec['timestep'], spec['q_usecols'])
            # LamaH-CE marks non-interpolated gaps in the runoff series with
            # -999; leaving them in would turn a gap into a large negative
            # discharge (33267 daily values across 179 gauges).
            # copy=True: the array must be writable also under pandas'
            # copy-on-write mode, where to_numpy() hands out a read-only view
            vals = q['qobs'].to_numpy(copy=True)
            missing = vals == _CE_Q_NODATA
            n_nodata = int(missing.sum())
            if n_nodata:
                vals[missing] = np.nan
                q['qobs'] = vals
            frames.append(q)
        else:
            # never happens for the published archive (every basin has a gauge
            # file) but a truncated extraction must not pass silently
            warnings.warn(f"no runoff file found at {q_fpath}. Returning NaNs "
                          f"for the streamflow of station {station}")
            index = frames[0].index if frames else pd.DatetimeIndex([], name='time')
            frames.append(pd.DataFrame(
                {'qobs': np.full(len(index), np.nan, dtype=np.float32)}, index=index))

    df = frames[0] if len(frames) == 1 else pd.concat(frames, axis=1)

    if spec['met_usecols'] is None:
        # only the runoff file was read. Its index starts later (and ends
        # earlier) than the meteorological grid, so put it back on that grid:
        # a fetch of q alone must be indexed exactly like a fetch of everything.
        grid = pd.date_range(spec['grid_start'], spec['grid_end'],
                             freq=spec['grid_freq'])
        n_before = int(df.notna().to_numpy().sum())
        df = df.reindex(grid)
        n_after = int(df.notna().to_numpy().sum())
        if n_after != n_before:
            warnings.warn(
                f"{n_before - n_after} runoff observations of station {station} "
                f"lie outside the {spec['grid_start']} - {spec['grid_end']} "
                f"period of the meteorological forcings and were dropped.")

    df.rename(columns=spec['rename'], inplace=True)

    for col, factor in spec['factors'].items():
        if col in df.columns:
            df[col] = df[col] * factor

    if spec['st'] is not None or spec['en'] is not None:
        df = df.loc[spec['st']:spec['en']]

    if spec['features'] is not None:
        df = df.loc[:, spec['features']]

    df.columns.name = "dynamic_features"
    df.index.name = "time"
    # reported by the caller, which warns once for the whole fetch instead of
    # once per worker process
    df.attrs['n_q_nodata'] = n_nodata
    return df


def _read_area_calc(fpath: str) -> pd.Series:
    """
    ``area_calc`` column of a LamaH ``Catchment_attributes.csv``, indexed by the
    (string) catchment id.

    The separator and the spelling of the id column vary across the LamaH
    products (``ID`` in LamaH-CE, ``id`` in LamaH-Ice, and LamaH-Ice's
    ``intermediate_lowimp`` file is comma separated), so both are detected
    rather than assumed.
    """
    for sep in (';', ','):
        df = pd.read_csv(fpath, sep=sep, index_col=0)
        if 'area_calc' in df.columns:
            df.index = df.index.astype(str)
            return df['area_calc']

    raise ValueError(f"no 'area_calc' column in {fpath}")


def _warn_nodata(n_values: int, n_stations: int):
    """single, unconditional warning about the -999 -> NaN substitution"""
    if n_values:
        warnings.warn(
            f"{n_values} runoff value(s) of {n_stations} station(s) carry the "
            f"LamaH-CE no-data marker -999 (see Info_english/2_Timeseries.txt) "
            f"and are returned as NaN. Use fetch_stn_q_raw() for the unmodified "
            f"source values together with the ckhs/qceq/qcol quality flags.",
            UserWarning)
    return


_WORKER_SPEC = {}


def _init_ce_worker(spec: Dict):
    """stores the (small) reader spec once per worker process"""
    _WORKER_SPEC['spec'] = spec


def _read_ce_stn_in_worker(station: str) -> pd.DataFrame:
    return _read_ce_stn(_WORKER_SPEC['spec'], station)


class LamaHCE(_RainfallRunoff):
    """
    Large-Sample Data for Hydrology and Environmental Sciences for Central Europe
    (mainly Austria). The dataset is downloaded from
    `zenodo <https://zenodo.org/record/5153305>`_
    following the work of
    `Klingler et al., 2021 <https://doi.org/10.5194/essd-13-4529-2021>`_ .

    The meteorological forcings cover 1981-01-01 to 2019-12-31 while the observed
    runoff ends on 2017-12-31. Depending on ``data_type`` the dataset consists of

        - ``total_upstrm``      : 859 stations, 84 static features
        - ``intermediate_all``  : 859 stations, 86 static features
        - ``intermediate_lowimp``: 454 stations, 86 static features

    There are 22 dynamic features at daily and 16 at hourly timestep.

    All time series are in UTC (no summer time), labelled with the start of the
    interval; the returned index is timezone-naive.

    LamaH-CE marks gaps in the runoff series that could not be interpolated with
    the value ``-999``. These are returned as ``NaN``; the untouched source rows
    (including the ``ckhs``/``qceq``/``qcol`` quality flags) are available from
    :meth:`fetch_stn_q_raw`. Catchment boundaries are provided (EPSG:3035) and
    are reprojected to WGS84 by :meth:`get_boundary`.

    The runoff files in ``D_gauges`` are the same for all three ``data_type``\\ s
    and always hold the *total* discharge at the gauge, while the catchment
    attributes of ``intermediate_all``/``intermediate_lowimp`` describe only the
    incremental sub-catchment between this gauge and the ones above it. So that
    ``area_km2`` means the same thing here as in every other dataset, :meth:`area`
    (and hence :meth:`q_mm`) always reports the area upstream of the gauge; under
    the two intermediate delineations the incremental area is kept alongside it
    as ``area_km2_intermediate``, which is why they carry one static feature more.
    """

    url = {
        '1_LamaH-CE_daily_hourly.tar.gz': 'https://zenodo.org/records/5153305/files/1_LamaH-CE_daily_hourly.tar.gz',
        # this archive carries the daily files only
        '2_LamaH-CE_daily.tar.gz': 'https://zenodo.org/records/5153305/files/2_LamaH-CE_daily.tar.gz'
    }

    # Both archives extract into the *same* top level folders
    # (``A_basins_total_upstrm``, ``D_gauges``, ...) and differ only in the
    # ``2_timeseries`` sub-folder they fill: archive 2 carries ``daily`` alone
    # while archive 1 carries ``daily`` and ``hourly``. The "is it already
    # extracted?" test must therefore name the timestep specific sub-folder -
    # checking the top level folder would let a daily-only installation pass
    # archive 1 off as present, so the hourly time series would never be
    # downloaded. ``self.url`` is filtered by ``timestep`` in ``__init__``, so
    # each archive below is only ever consulted for the timestep it carries.
    dirs_to_check = {
        # consulted only when timestep == 'H'
        '1_LamaH-CE_daily_hourly.tar.gz': {
            'total_upstrm': [
                os.path.join('A_basins_total_upstrm', '2_timeseries', 'hourly'),
                os.path.join('D_gauges', '2_timeseries', 'hourly')],
            'intermediate_all': [
                os.path.join('B_basins_intermediate_all', '2_timeseries', 'hourly'),
                os.path.join('D_gauges', '2_timeseries', 'hourly')],
            'intermediate_lowimp': [
                os.path.join('C_basins_intermediate_lowimp', '2_timeseries', 'hourly'),
                os.path.join('D_gauges', '2_timeseries', 'hourly')],
            },
        # consulted only when timestep == 'D'
        '2_LamaH-CE_daily.tar.gz': {
            'total_upstrm': [
                os.path.join('A_basins_total_upstrm', '2_timeseries', 'daily'),
                os.path.join('D_gauges', '2_timeseries', 'daily')],
            'intermediate_all': [
                os.path.join('B_basins_intermediate_all', '2_timeseries', 'daily'),
                os.path.join('D_gauges', '2_timeseries', 'daily')],
            'intermediate_lowimp': [
                os.path.join('C_basins_intermediate_lowimp', '2_timeseries', 'daily'),
                os.path.join('D_gauges', '2_timeseries', 'daily')],
            },
    }

    _data_types = ['total_upstrm', 'intermediate_all', 'intermediate_lowimp']
    time_steps = ['D', 'H']

    def __init__(
            self,
            path=None,
            *,
            timestep: str = 'D',
            data_type: str = 'total_upstrm',
            to_netcdf: bool = False,  # todo : current IO for .ncs are slow
            overwrite=False,
            **kwargs
            ):

        """
    Parameters
    ----------
        path : str
            If the data is alredy downloaded then provide the complete
            path to it. If None, then the data will be downloaded.
            The data is downloaded once and therefore susbsequent
            calls to this class will not download the data unless
            ``overwrite`` is set to True.
        timestep :
                possible values are ``D`` for daily or ``H`` for hourly timestep
        data_type :
                possible values are ``total_upstrm``, ``intermediate_all``
                or ``intermediate_lowimp``

    Examples
    --------
    >>> from aqua_fetch import LamaHCE
    # by default the timestep is daily and data_type is 'total_upstrm'
    >>> dataset = LamaHCE()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='826', as_dataframe=True)
    >>> df = dynamic['826'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (14244, 22)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       859
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (85 out of 859)
       85
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(14244, 22), (14244, 22), (14244, 22),... (14244, 22), (14244, 22)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('826', as_dataframe=True,
    ...  dynamic_features=['airtemp_C_mean', 'total_et', 'pcp_mm', 'q_cms_obs'])
    >>> dynamic['826'].shape
       (14244, 4)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='826', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['826'].shape
    ((1, 84), 1, (14244, 22))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 14244, 'dynamic_features': 22})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (859, 2)
    >>> dataset.stn_coords('826')  # returns coordinates of station whose id is 826
              lat       long
    ID
    826  49.86693  16.838505
    >>> dataset.stn_coords(['826', '819'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('826')
    # get area of two stations
    >>> dataset.area(['826', '819'])
    ...
    # if fiona library is installed we can get the boundary (in WGS84) as fiona Geometry
    >>> dataset.get_boundary('826')
    ...
    # the raw runoff file of a gauge, including the quality flags and the
    # -999 no-data markers exactly as published
    >>> dataset.fetch_stn_q_raw('826').columns.tolist()
    ['qobs', 'ckhs', 'qceq', 'qcol']
    ...
    # the data_type can also be 'intermediate_all'
    >>> dataset = LamaHCE(data_type='intermediate_all')
    ...
    # or 'intermediate_lowimp'
    >>> dataset = LamaHCE(data_type='intermediate_lowimp')
    >>> len(dataset.stations())
    454
    ...
    # the timestep can also be hourly i.e. 'H'
    >>> dataset = LamaHCE(timestep='H')
    >>> _, dynamic = dataset.fetch(stations='79', as_dataframe=True)
    >>> dynamic['79'].shape
    (341856, 16)  # there are 16 dynamic features for hourly data
    """

        assert timestep in self.time_steps, f"invalid timestep '{timestep}' given, choose from {self.time_steps}"
        assert data_type in self._data_types, f"invalid data_type '{data_type}' given, choose from {self._data_types}"

        self.data_type = data_type

        # caches for the small-but-repeatedly-read tables. They are filled on
        # first use so that instantiating the class stays cheap.
        self._static_data_cache = None
        self._total_area_cache = None
        self._stations_cache = None
        self._data_type_dir_cache = None
        self._extent_cache = None
        self._met_cols_cache = None
        self._q_cols_cache = None

        # forward timestep and to_netcdf so the parent doesn't reset them
        super().__init__(path=path, timestep=timestep, to_netcdf=to_netcdf,
                         overwrite=overwrite, **kwargs)

        # copy so we don't mutate the class-level dict shared across instances
        self.url = dict(self.url)
        if timestep == "D":
            self.url.pop("1_LamaH-CE_daily_hourly.tar.gz", None)
        if timestep == 'H':
            self.url.pop('2_LamaH-CE_daily.tar.gz', None)

        self._download_and_extract()

        self._static_features = self.static_data().columns.to_list()

        self._dynamic_features = self._infer_dynamic_features()

        if self.to_netcdf and not self.all_ncs_exist:
            self._maybe_to_netcdf(fdir=f"{data_type}_{timestep}")

        self.bbox = {"llcrnrlat": 46, "urcrnrlat": 50.5,
                        "llcrnrlon": 7.5, "urcrnrlon": 19}
        self.parallels = range(46, 51, 1)
        self.meridians = range(7, 19, 2)

    def _download_and_extract(self):
        """
        Downloads and extracts whatever the requested ``data_type`` needs.

        The "is it already there?" decision is taken on the *extracted*
        directories, never on the archive, so that ``remove_zip=True`` does not
        force a re-download on the next instantiation. Those directories are the
        timestep specific ones declared in :attr:`dirs_to_check`, because the
        two LamaH-CE archives share their top level folder names and only the
        daily+hourly archive fills the ``hourly`` sub-folders.

        With ``overwrite=True`` both the stale archive and the previously
        extracted directories are removed first, otherwise
        :func:`aqua_fetch.utils.download` would write to ``<name>1``. What is
        removed there is the *top level* folder of each declared directory, so
        that the attributes and shapefiles are refreshed too and not only the
        time series.
        """
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        for fname, url in self.url.items():
            fpath = os.path.join(self.path, fname)
            declared = self.dirs_to_check[fname][self.data_type]
            folders = [os.path.join(self.path, folder) for folder in declared]

            if not folders:
                # the archive does not declare anything that this data_type
                # needs, so it is neither downloaded nor extracted
                continue

            if self.overwrite:
                # dict, not set, so that the removal order stays deterministic
                top_level = {os.path.join(self.path, folder.split(os.sep)[0]): None
                             for folder in declared}
                for target in [fpath] + list(top_level):
                    if os.path.isdir(target):
                        if self.verbosity:
                            print(f"removing {target} because overwrite=True")
                        shutil.rmtree(target)
                    elif os.path.exists(target):
                        os.remove(target)

            # if the extracted folders already exist, nothing to do
            elif all(os.path.exists(folder) for folder in folders):
                continue

            # archive missing -> download
            if not os.path.exists(fpath):
                if self.verbosity:
                    print(f'downloading {fname}')
                download(url, self.path, fname, verbosity=self.verbosity)

            # archive present (just downloaded or already on disk) but not unzipped
            unzip(self.path, verbosity=self.verbosity)

            missing = [f for f in folders if not os.path.exists(f)]
            if missing:
                warnings.warn(
                    f"the following directories are missing after extracting "
                    f"{fname}: {missing}. The extraction may have been "
                    f"interrupted; the data of {self.name} is incomplete.")

        self.maybe_remove_zip_files()
        return

    @property
    def dyn_fname(self) -> Union[str, os.PathLike]:
        """
        name of the .nc file which contains dynamic features. This file is created during dataset initialization
        only if to_netcdf is True and xarray is installed and the file does not already exists. The creation of this
        file can take some time however it leads to faster I/O operations.
        """
        return self.name.lower() + f"_{self.timestep}_{self.data_type}.nc"

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'area_calc': catchment_area(), # todo : difference between area_calc and area_gov?
                'lat': gauge_latitude(),
                'slope_mean': slope('mkm-1'),
                'lon': gauge_longitude(),
        }

    @property
    def dyn_map(self):
        return {
            'D': {
                'qobs': observed_streamflow_cms(),
                '2m_temp_min': min_air_temp(),  # todo : what about height?
                '2m_temp_max': max_air_temp(),
                '2m_temp_mean': mean_air_temp(),
                'prec': total_precipitation(),
                'swe': snow_water_equivalent(),
                'surf_net_solar_rad_max': max_solar_radiation(),
                'surf_net_solar_rad_mean': solar_radiation(),
                'surf_net_therm_rad_max': max_thermal_radiation(),
                'surf_net_therm_rad_mean': mean_thermal_radiation(),
                '10m_wind_u': u_component_of_wind_at_10m(),
                '10m_wind_v': v_component_of_wind_at_10m(),
                '2m_dp_temp_max': max_dewpoint_temperature_at_2m(),
                '2m_dp_temp_mean': mean_dewpoint_temperature_at_2m(),
                '2m_dp_temp_min': min_dewpoint_temperature_at_2m(),
                'surf_press': mean_air_pressure(),  # todo : is this air pressure?
            },
            'H': {
                'qobs': observed_streamflow_cms(),
                '2m_temp': mean_air_temp(),
                'prec': total_precipitation(),
                'swe': snow_water_equivalent(),
                '10m_wind_u': u_component_of_wind_at_10m(),
                '10m_wind_v': v_component_of_wind_at_10m(),
                '2m_dp_temp': mean_dewpoint_temperature_at_2m(),
                'surf_net_solar_rad': solar_radiation(),
                'surf_net_therm_rad': mean_thermal_radiation(),
                'surf_press': mean_air_pressure(),  # todo : is this air pressure?
            }
        }

    @property
    def dyn_factors(self) -> Dict[str, float]:
        return {
            mean_air_pressure(): 0.01,
        }

    @property
    def boundary_file(self) -> os.PathLike:
        # A_basins_total_upstrm -> Basins_A.shp,
        # B_basins_intermediate_all -> Basins_B.shp,
        # C_basins_intermediate_lowimp -> Basins_C.shp
        letters = {'total_upstrm': 'A',
                   'intermediate_all': 'B',
                   'intermediate_lowimp': 'C'}
        return os.path.join(self.data_type_dir,
                            "3_shapefiles",
                            f"Basins_{letters[self.data_type]}.shp")

    def transform_boundary(self, boundary):
        """
        The LamaH-CE shapefiles are in ETRS89-LAEA (EPSG:3035, metres, see
        ``Info_english/1_Folder_structure.txt``). This reprojects them to WGS84
        so that the returned geometry is in the same coordinate system as
        :meth:`stn_coords`. The conversion is the ellipsoidal (GRS80) inverse
        LAEA and was verified against pyproj to better than 1e-8 m.
        """
        if fiona is None:
            return boundary

        # parameters of EPSG:3035, taken from the accompanying .prj file
        lon_0, lat_0 = 10.0, 52.0
        false_easting, false_northing = 4321000.0, 3210000.0

        def _to_wgs84(ring):
            xy = np.asarray(ring, dtype='float64')
            lat, lon = laea_to_wgs84(xy[:, 0], xy[:, 1], lon_0, lat_0,
                                     false_easting, false_northing)
            return list(zip(lon.tolist(), lat.tolist()))

        if boundary.type == 'MultiPolygon':
            coords = [[_to_wgs84(ring) for ring in polygon]
                      for polygon in boundary.coordinates]
        else:
            coords = [_to_wgs84(ring) for ring in boundary.coordinates]

        return fiona.Geometry(type=boundary.type, coordinates=coords)

    def _maybe_to_netcdf(self, fdir: str):
        # since data is very large, saving all the data in one file
        # consumes a lot of memory, which is impractical for most of the personal
        # computers! Therefore, saving each feature separately. Each of these
        # passes now only parses the columns of the feature being written
        # (see ``_read_dynamic``), so the cost is roughly one file scan per
        # feature rather than one full parse per feature.

        fdir = os.path.join(self.path, fdir)
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        if not self.all_ncs_exist:
            if self.verbosity:
                print(f'converting data to netcdf format for faster io operations')

            for feature in self.dynamic_features:

                # we must specify class level dyn_fname feature
                dyn_fname = os.path.join(fdir, f"{feature}.nc")

                if not os.path.exists(dyn_fname):
                    if self.verbosity:
                        print(f'Saving {feature} as {dyn_fname}')
                    _, data = self.fetch(static_features=None, dynamic_features=feature)

                    data.to_netcdf(dyn_fname)

                    gc.collect()
        return

    def _archive_files(self) -> List[str]:
        """LamaHCE archives don't unzip into a same-named wrapper folder
        (they extract ``A_basins_total_upstrm``, ``D_gauges``, ... directly
        under ``self.path``), so the parent class's stem-matching heuristic
        skips them. Use the per-archive ``dirs_to_check`` declaration as the
        proof that an archive's content is on disk.
        """
        if not os.path.isdir(self.path):
            return []
        out: List[str] = []
        for fname, dt_map in self.dirs_to_check.items():
            fpath = os.path.join(self.path, fname)
            if not os.path.exists(fpath):
                continue
            expected = dt_map.get(self.data_type)
            if not expected:
                continue
            if all(os.path.exists(os.path.join(self.path, folder))
                   for folder in expected):
                out.append(fpath)
        return out

    @property
    def dynamic_fnames(self):
        return [f"{feature}.nc" for feature in self.dynamic_features]

    @property
    def all_ncs_exist(self):
        fdir = os.path.join(self.path, f"{self.data_type}_{self.timestep}")
        return all(os.path.exists(os.path.join(fdir, fname_)) for fname_ in self.dynamic_fnames)

    @property
    def dynamic_features(self) -> List[str]:
        return list(self._dynamic_features)

    def _infer_dynamic_features(self) -> List[str]:
        """
        Names of the dynamic features, derived from the headers of the two csv
        files of the first station. Only the headers are parsed, so this does
        not slow down the instantiation of the class (reading a whole hourly
        meteorological file to learn its column names used to cost ~0.5 s).
        """
        station = self.stations()[0]
        cols = self._met_columns(station) + self._q_columns(station)
        skip = set(_CE_DATE_COLS) | set(_CE_Q_FLAG_COLS)
        rename = self.dyn_map[self.timestep]
        return [rename.get(col, col) for col in cols if col not in skip]

    def _met_columns(self, station: str) -> List[str]:
        """column names of the meteorological file of ``station`` (cached)"""
        if self._met_cols_cache is None:
            self._met_cols_cache = pd.read_csv(
                self.met_fname(station), sep=';', nrows=0).columns.to_list()
        return list(self._met_cols_cache)

    def _q_columns(self, station: str) -> List[str]:
        """column names of the runoff file of ``station`` (cached)"""
        if self._q_cols_cache is None:
            self._q_cols_cache = pd.read_csv(
                self.q_fname(station), sep=';', nrows=0).columns.to_list()
        return list(self._q_cols_cache)

    @property
    def static_features(self) -> List[str]:
        return list(self._static_features)

    @property
    def data_type_dir(self):
        # cached: this is resolved for every single station file name
        if self._data_type_dir_cache is None:
            f = [f for f in os.listdir(self.path) if f.endswith(self.data_type)][0]
            self._data_type_dir_cache = os.path.join(self.path, f)
        return self._data_type_dir_cache

    @property
    def q_dir(self):
        return os.path.join(self.path, 'D_gauges', '2_timeseries')

    @property
    def ts_dir_name(self) -> str:
        """name of the sub-folder (``daily``/``hourly``) holding the time series"""
        return {'H': 'hourly', 'D': 'daily'}[self.timestep]

    @property
    def met_dir(self) -> str:
        """directory holding the meteorological time series of all stations"""
        return os.path.join(self.data_type_dir, '2_timeseries', self.ts_dir_name)

    @property
    def q_ts_dir(self) -> str:
        """directory holding the runoff time series of all gauges"""
        return os.path.join(self.q_dir, self.ts_dir_name)

    def q_fname(self, station: str) -> str:
        """path of the runoff file of ``station``"""
        return os.path.join(self.q_ts_dir, f'ID_{station}.csv')

    def stations(self) -> List[str]:
        # assuming file_names of the format ID_{stn_id}.csv
        if self._stations_cache is None:
            self._stations_cache = [f.split('_')[1].split('.csv')[0]
                                    for f in os.listdir(self.met_dir)]
        # a copy, so that a caller sorting/appending to the returned list
        # cannot corrupt the cache
        return list(self._stations_cache)

    def transform_stn_coords(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        transforms the gauge coordinates from EPSG:3035 (ETRS89-LAEA) to
        WGS84 (EPSG:4326)
        """

        # following 2 lines are from .prj file
        false_easting, false_northing = 4321000.0, 3210000.0
        lat_0, lon_0 = 52, 10

        # the caller hands over float32 eastings/northings; do the trigonometry
        # in double precision, otherwise the result is off by ~0.2 m
        lat, lon = laea_to_wgs84(df.loc[:, 'long'].astype('float64'),
                                 df.loc[:, 'lat'].astype('float64'),
                                 lon_0, lat_0, false_easting, false_northing)
        return pd.DataFrame({'lat': lat, 'long': lon}, index=df.index)

    def fetch_stations_features(
            self,
            stations: Union[str, List[str]],
            dynamic_features: Union[str, List[str]] = 'all',
            static_features: Union[str, List[str]] = None,
            st: Union[str, pd.Timestamp] = None,
            en: Union[str, pd.Timestamp] = None,
            as_dataframe: bool = False,
            **kwargs
    ) -> Tuple[pd.DataFrame, Union[Dict[str, pd.DataFrame], "Dataset"]]:
        """Reads attributes of more than one stations.

        This function checks of .nc files exist, then they are not prepared
        and saved otherwise first nc files are prepared and then the data is
        read again from nc files. Upon subsequent calls, the nc files are used
        for reading the data.

        Arguments:
            stations : list of stations for which data is to be fetched.
            dynamic_features : list of dynamic attributes to be fetched.
                if 'all', then all dynamic attributes will be fetched.
            static_features : list of static attributes to be fetched.
                If `all`, then all static attributes will be fetched. If None,
                then no static attribute will be fetched.
            st : start of data to be fetched.
            en : end of data to be fetched.
            as_dataframe : whether to return the data as pandas dataframe. default
                is :obj:`xarray.Dataset` object
            kwargs dict: additional keyword arguments

        Returns
        -------
        tuple
            A tuple of static and dynamic features. Static features are always
            returned as :obj:`pandas.DataFrame` with shape (stations, static features).
            The index of static features' DataFrame is the station/gauge ids while the columns 
            are names of the static features. Dynamic features are returned either as
            :obj:`xarray.Dataset` or a dictionary with keys as station names and values
            as :obj:`pandas.DataFrame` depending upon whether `as_dataframe`
            is True or False and whether the :obj:`xarray` library is installed or not.
            If dynamic features are :obj:`xarray.Dataset`, then this dataset consists of `data_vars`
            equal to the number of stations and station names as :obj:`xarray.Dataset.variables`  
            and `time` and `dynamic_features` as dimensions and coordinates.

        Raises:
            ValueError, if both dynamic_features and static_features are None

        Examples
        --------
        >>> from aqua_fetch import CAMELS_AUS
        >>> dataset = CAMELS_AUS()
        ... # find out station ids
        >>> dataset.stations()
        ... # get data of selected stations
        >>> dataset.fetch_stations_features(['912101A', '912105A', '915011A'],
        ...  as_dataframe=True)
        """

        if xr is None:
            if not as_dataframe:
                if self.verbosity: warnings.warn("xarray module is not installed so as_dataframe will have no effect. "
                              "Dynamic features will be returned as pandas DataFrame")
                as_dataframe = True

        st, en = self._check_length(st, en)
        static, dynamic = None, None

        stations = validate_attributes(stations, self.stations(), 'stations')

        if dynamic_features is not None:

            dynamic_features = validate_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')

            if self.verbosity > 2:
                print(f'fetching data for {len(dynamic_features)} dynamic '
                      f'features for {len(stations)} stations')

            if netCDF4 is None or not self.all_ncs_exist:
                # read from csv files
                # following code will run only once when fetch is called inside init method
                dynamic = self._read_dynamic(stations, dynamic_features, st=st, en=en)
            else:
                dynamic = self._make_ds_from_ncs(dynamic_features, stations, st, en)

                if as_dataframe:
                    dynamic = {stn: dynamic[stn].to_pandas() for stn in dynamic}

            if static_features is not None:
                static = self.fetch_static_features(stations, static_features)

            dynamic = _handle_dynamic(dynamic, as_dataframe)

        elif static_features is not None:

            static = self.fetch_static_features(stations, static_features)

        else:
            raise ValueError(f"static features are {static_features} and "
                             f"dynamic features are {dynamic_features}")

        return static, dynamic

    @property
    def _q_name(self) -> str:
        return observed_streamflow_cms()

    def gauge_attributes(self) -> pd.DataFrame:
        fname = os.path.join(self.path,
                             'D_gauges',
                             '1_attributes',
                             'Gauge_attributes.csv')
        df = pd.read_csv(fname, sep=';', index_col='ID')

        df.index = df.index.astype(str)
        return df

    def catchment_attributes(self) -> pd.DataFrame:
        fname = os.path.join(self.data_type_dir,
                             f'1_attributes{SEP}Catchment_attributes.csv')

        df = pd.read_csv(fname, sep=';', index_col='ID')
        df.index = df.index.astype(str)
        return df

    @property
    def total_upstrm_dir(self) -> str:
        """
        directory of the ``A`` (total upstream) basin delineation, whatever
        ``data_type`` is. The three delineations are siblings of each other, so
        this is resolved from :attr:`data_type_dir`.
        """
        return os.path.join(os.path.dirname(self.data_type_dir),
                            'A_basins_total_upstrm')

    def total_upstrm_area(
            self,
            stations: Union[str, List[str]] = "all"
    ) -> pd.Series:
        """
        Area (km2) upstream of the gauge, i.e. ``area_calc`` of the
        ``A_basins_total_upstrm`` delineation, regardless of ``data_type``.

        This is the area the observed runoff drains and is therefore what
        :meth:`q_mm` divides by. It differs from :meth:`area` when ``data_type``
        is ``intermediate_all`` or ``intermediate_lowimp``, where the catchment
        attributes describe the incremental sub-catchment between two gauges
        while ``D_gauges`` still reports the total discharge.

        It is what :meth:`area` reports for every ``data_type``; the incremental
        area of an intermediate delineation stays available under
        ``area_km2_intermediate``.

        Parameters
        ----------
        stations : str/list
            name/names of stations. Default is ``all``.

        Returns
        -------
        pd.Series
            a :obj:`pandas.Series` whose index are station ids and values are
            the total upstream areas in km2

        Examples
        --------
        >>> from aqua_fetch import LamaHCE
        >>> dataset = LamaHCE(data_type='intermediate_all')
        >>> dataset.total_upstrm_area('1')
        ID
        1    4668.378906
        Name: area_km2, dtype: float32
        >>> dataset.area('1')   # the same thing
        ID
        1    4668.378906
        Name: area_km2, dtype: float32
        >>> # the incremental catchment between this gauge and the ones above it
        >>> dataset.fetch_static_features('1', 'area_km2_intermediate')
            area_km2_intermediate
        ID
        1                 442.343
        """
        stations = validate_attributes(stations, self.stations(), 'stations')
        return self._total_upstrm_areas().loc[stations].astype(self.fp)

    def _total_upstrm_areas(self) -> pd.Series:
        """``area_calc`` of the ``A`` delineation for every catchment (cached)"""
        if self._total_area_cache is None:
            fname = os.path.join(self.total_upstrm_dir,
                                 '1_attributes', 'Catchment_attributes.csv')
            if not os.path.exists(fname):
                raise FileNotFoundError(
                    f"{fname} holds the area upstream of each gauge, which "
                    f"{self.name} needs whatever the data_type is, but it is "
                    f"not on disk. Re-instantiate the class with "
                    f"overwrite=True to extract it again.")
            self._total_area_cache = _read_area_calc(fname).rename(catchment_area())
        return self._total_area_cache

    def area(self, stations: List[str]) -> pd.Series:
        # D_gauges reports the total discharge at the gauge for every
        # data_type, so the runoff height must always use the total upstream
        # area and never the (possibly much smaller) intermediate one.
        return self.total_upstrm_area(stations)

    def static_data(self) -> pd.DataFrame:
        """returns all static attributes of LamaHCE dataset"""
        if self._static_data_cache is None:
            df = pd.concat([self.catchment_attributes(), self.gauge_attributes()], axis=1)
            duplicated = df.columns.duplicated(keep='last')
            if duplicated.any():
                # does not happen for the published archives (the catchment and
                # the gauge tables share no field name) but must not silently
                # produce a table with repeated columns if that ever changes
                warnings.warn(
                    f"Catchment_attributes.csv and Gauge_attributes.csv of "
                    f"{self.name} share the field(s) "
                    f"{df.columns[duplicated].to_list()}. Keeping the gauge copy.")
                df = df.loc[:, ~duplicated]
            df.rename(columns=self.static_map, inplace=True)
            df = self._standardize_area(df)
            self._static_data_cache = df
            self._warn_about_duplicate_gauges(df)
        # a copy, so that an in-place edit by the caller cannot corrupt the cache
        return self._static_data_cache.copy()

    def _standardize_area(self, static: pd.DataFrame) -> pd.DataFrame:
        """
        Makes ``area_km2`` mean the same thing as in every other dataset of the
        library: the area upstream of the gauge.

        For ``total_upstrm`` that is already what the catchment attributes hold
        and the table is returned unchanged. For the two intermediate
        delineations ``area_calc`` is the incremental catchment between this
        gauge and the ones above it - a real area, but not the one the gauge's
        discharge drains - so it is kept under ``area_km2_intermediate`` and
        ``area_km2`` is taken from the ``A`` delineation instead. Nothing is
        dropped; the table gains one column.
        """
        if self.data_type == 'total_upstrm':
            return static

        area, intermediate = catchment_area(), catchment_area_with_specifier('intermediate')
        if area not in static:
            return static

        static = static.rename(columns={area: intermediate})
        # same position as before, so area_km2 stays the leading column
        static.insert(static.columns.get_loc(intermediate), area,
                      self._total_upstrm_areas().reindex(static.index))
        return static

    def _warn_about_duplicate_gauges(self, static: pd.DataFrame):
        """
        Warns (but does not exclude) when two stations describe the same gauge.

        Two gauges are considered duplicates when their coordinates agree to
        within 100 m; the gauge name alone is not sufficient because LamaH-CE
        legitimately contains e.g. two gauges called *Anger* on two different
        rivers 270 km apart. This runs once per instance on a ~880 row table, so
        it does not slow down fetching.
        """
        lat, lon = gauge_latitude(), gauge_longitude()
        if lat not in static or lon not in static:
            return

        # gauges without coordinates cannot be compared and must not all end up
        # in one "NaN" group
        df = static.reindex(self.stations()).dropna(subset=[lat, lon])

        # coordinates are in metres (EPSG:3035), so a 100 m tolerance is a
        # plain rounding of the easting/northing
        key = list(zip((df[lon] // 100).to_list(), (df[lat] // 100).to_list()))
        keys = pd.Series(key, index=df.index)
        dupes = keys[keys.duplicated(keep=False)]

        if len(dupes) == 0:
            return

        groups = {}
        for stn, k in dupes.items():
            groups.setdefault(k, []).append(stn)
        msg = "; ".join(
            f"{grp} (name(s): {df.loc[grp, 'name'].to_list() if 'name' in df else '?'})"
            for grp in groups.values())
        warnings.warn(
            f"{self.name} contains gauges whose coordinates agree to within "
            f"100 m and which may therefore be duplicates: {msg}. They are kept "
            f"in the dataset.", UserWarning)
        return

    def _reader_spec(
            self,
            dynamic_features: List[str] = None,
            st: pd.Timestamp = None,
            en: pd.Timestamp = None
    ) -> Dict:
        """
        Builds the (small, picklable) description of what a station read has to
        do. Translating the requested feature names back into the on-disk column
        names lets :func:`_read_ce_csv` parse only the needed columns and lets
        the whole meteorological or runoff file be skipped when none of its
        columns were asked for. Fetching only ``q_cms_obs`` from the hourly
        product therefore no longer parses a 29 MB meteorological file per
        station.
        """
        rename = dict(self.dyn_map[self.timestep])
        src_of = {ren: src for src, ren in rename.items()}

        date_cols = ['YYYY', 'MM', 'DD'] + (['hh', 'mm'] if self.timestep == 'H' else [])

        met_cols = [c for c in self._met_columns(self.stations()[0])
                    if c not in _CE_DATE_COLS]
        q_cols = [c for c in self._q_columns(self.stations()[0])
                  if c not in _CE_DATE_COLS and c not in _CE_Q_FLAG_COLS]

        if dynamic_features is None:
            wanted_met, wanted_q = met_cols, q_cols
        else:
            src = [src_of.get(f, f) for f in dynamic_features]
            wanted_met = [c for c in met_cols if c in src]
            wanted_q = [c for c in q_cols if c in src]

        grid_start, grid_end = self._dyn_extent()

        return dict(
            met_dir=self.met_dir,
            q_dir=self.q_ts_dir,
            timestep=self.timestep,
            met_usecols=date_cols + wanted_met if wanted_met else None,
            q_usecols=date_cols + wanted_q if wanted_q else None,
            rename=rename,
            factors=dict(self.dyn_factors),
            features=list(dynamic_features) if dynamic_features is not None else None,
            st=st,
            en=en,
            grid_start=grid_start,
            grid_end=grid_end,
            grid_freq=_CE_FREQ[self.timestep],
        )

    def _n_workers(self, spec: Dict, n_stations: int) -> int:
        """
        Number of worker processes to read ``n_stations`` with.

        The decision is taken on the *size* of the workload rather than on the
        number of stations: three hourly stations are ~100 MB of csv and are
        worth a pool, while a hundred daily stations of a single feature are not.
        """
        cpus = self.processes or min(get_cpus(), 32)

        if cpus == 1 or n_stations < 2:
            return 1

        station = self.stations()[0]
        nbytes = 0
        for key, directory in (('met_usecols', spec['met_dir']),
                               ('q_usecols', spec['q_dir'])):
            if spec[key] is None:
                continue
            fpath = os.path.join(directory, f"ID_{station}.csv")
            if os.path.exists(fpath):
                # only the requested columns are parsed
                ncols = len(spec[key])
                total_cols = len(self._met_columns(station) if key == 'met_usecols'
                                 else self._q_columns(station))
                nbytes += os.path.getsize(fpath) * ncols / total_cols

        # ~30 MB is where the cost of starting the pool and shipping the frames
        # back starts to be repaid by the parallel parsing
        if nbytes * n_stations < 30e6:
            return 1
        return min(cpus, n_stations)

    def _read_dynamic(
            self,
            stations,
            dynamic_features: Union[str, list] = 'all',
            st=None,
            en=None,
    ) -> Dict[str, pd.DataFrame]:
        """Reads dynamic features of one or more stations"""

        stations = list(stations)
        dyn_feats = validate_attributes(dynamic_features, self.dynamic_features,
                                        'dynamic_features')
        st, en = self._check_length(st, en)

        spec = self._reader_spec(dyn_feats, st=st, en=en)
        cpus = self._n_workers(spec, len(stations))

        if self.verbosity > 1:
            print(f"reading {len(dyn_feats)} dynamic features of "
                  f"{len(stations)} stations with {cpus} cpus")

        if cpus == 1:
            results = {}
            for idx, stn in enumerate(stations):
                results[stn] = _read_ce_stn(spec, stn)

                if self.verbosity > 2 and idx % 10 == 0:
                    print(f'{idx} stations read')
        else:
            # the reader is a module level function and the (small) spec is
            # shipped to each worker exactly once by the initializer, so no
            # cached table of this instance is pickled per station
            with cf.ProcessPoolExecutor(max_workers=cpus,
                                        initializer=_init_ce_worker,
                                        initargs=(spec,)) as executor:
                data = executor.map(_read_ce_stn_in_worker, stations)
                results = {stn: df for stn, df in zip(stations, data)}

        nodata = {stn: df.attrs.get('n_q_nodata', 0) for stn, df in results.items()}
        _warn_nodata(sum(nodata.values()), sum(1 for n in nodata.values() if n))

        return results

    def _make_ds_from_ncs(self, dynamic_features, stations, st, en):
        """makes xarray Dataset by reading multiple .nc files"""

        if self.verbosity>1:
            print(f'fetching data for {len(dynamic_features)} dynamic features for {len(stations)} stations')

        dyns = []
        for idx, f in enumerate(dynamic_features):
            dyn_fpath = os.path.join(self.path, f"{self.data_type}_{self.timestep}", f'{f}.nc')
            with xr.open_dataset(dyn_fpath) as dyn:
                dyns.append(dyn[stations].sel(time=slice(st, en)).load())

            if self.verbosity>3:
                print(f'{idx}: {f} read')

        xds = xr.concat(dyns, dim='dynamic_features')  # dataset todo: taking too much time!

        if self.verbosity>3:
            print(f'concatenated')
        return xds

    def fetch_static_features(
            self,
            stations: Union[str, List[str]] = "all",
            static_features: Union[str, List[str]] = "all"
    ) -> pd.DataFrame:
        """
        static features of LamaHCE

        Parameters
        ----------
            stations : str
                name/id of station of which to extract the data
            static_features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Examples
        --------
            >>> from aqua_fetch import LamaHCE
            >>> dataset = LamaHCE(timestep='D', data_type='total_upstrm')
            >>> dataset.fetch_static_features('99').shape
            (1, 84)
            ...  # get list of all static features
            >>> dataset.static_features
            >>> dataset.fetch_static_features('99',
            ... static_features=['area_km2', 'elev_mean', 'agr_fra', 'sand_fra']).shape
            (1, 4)
        """

        static_features = validate_attributes(static_features, self.static_features, 'static features')
        stations = validate_attributes(stations, self.stations(), 'stations')

        df = self.static_data()
        df.index = df.index.astype(str)

        return df.loc[stations, static_features]

    @property
    def chk_col(self) -> str:
        """
        name of the column carrying the "checked by the hydrographic service"
        flag in the runoff files. It is named ``ckhs`` at both timesteps.
        """
        return 'ckhs'

    def _read_stn_dyn(
            self,
            station: str,
            dynamic_features: Union[str, List[str]] = None,
            st: Union[str, pd.Timestamp] = None,
            en: Union[str, pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        reads the dynamic (meteorological + runoff) data of a single station.

        ``dynamic_features`` restricts both the columns that are parsed off disk
        and the columns of the returned DataFrame. ``None`` (default) returns
        all dynamic features.
        """
        if isinstance(dynamic_features, str):
            dynamic_features = [dynamic_features]

        spec = self._reader_spec(dynamic_features,
                                 st=None if st is None else pd.Timestamp(st),
                                 en=None if en is None else pd.Timestamp(en))
        df = _read_ce_stn(spec, station)
        _warn_nodata(df.attrs.get('n_q_nodata', 0), 1)
        return df

    def met_fname(self, station: str) -> str:
        """path of the meteorological file of ``station``"""
        return os.path.join(self.met_dir, f'ID_{station}.csv')

    def fetch_stn_q_raw(self, station: str) -> pd.DataFrame:
        """
        The runoff file of one gauge exactly as published, i.e. with the
        ``-999`` no-data markers still in place and with the quality flags

            - ``ckhs`` : 1 where the value was checked by the hydrographic service
            - ``qceq`` : 1 where at least 10 consecutive values are equal
            - ``qcol`` : 1 where the value is an outlier w.r.t. the calendar
              day/hour statistics

        Use this when the flags matter; :meth:`fetch` returns the analysis ready
        view where ``-999`` has been replaced by ``NaN`` and the flags are
        dropped.

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` indexed by time whose columns are
            ``qobs``, ``ckhs``, ``qceq`` and ``qcol``.

        Examples
        --------
        >>> from aqua_fetch import LamaHCE
        >>> dataset = LamaHCE()
        >>> raw = dataset.fetch_stn_q_raw('124')
        >>> (raw['qobs'] == -999).sum()
        3
        """
        fpath = self.q_fname(station)
        # no dtype/usecols: reproduce the file with pandas' own type inference
        df = pd.read_csv(fpath, sep=';')

        if self.timestep == 'H':
            df.index = _ymd_index(df['YYYY'], df['MM'], df['DD'], df['hh'], df['mm'])
        else:
            df.index = _ymd_index(df['YYYY'], df['MM'], df['DD'])

        df.index.name = 'time'
        return df.drop(columns=[c for c in _CE_DATE_COLS if c in df.columns])

    def _dyn_extent(self) -> tuple:
        """
        First and last timestamp of the meteorological forcings, read from the
        first and last line of one file rather than hardcoded. All 859 (454 for
        ``intermediate_lowimp``) files of a given product share the same period.
        """
        if self._extent_cache is None:
            self._extent_cache = _first_last_stamps(
                self.met_fname(self.stations()[0]), self.timestep)
        return self._extent_cache

    @property
    def start(self) -> pd.Timestamp:
        """first timestamp of the (meteorological) time series"""
        return self._dyn_extent()[0]

    @property
    def end(self) -> pd.Timestamp:
        """
        last timestamp of the (meteorological) time series. Note that the
        observed runoff of every gauge ends earlier (2017-12-31).
        """
        return self._dyn_extent()[1]


class LamaHIce(LamaHCE):
    """
    Daily and hourly hydro-meteorological time series data of river basins
    of Iceland following `Helgason et al., 2024 <https://doi.org/10.5194/essd-16-2741-2024>`_.
    The total period of dataset is from 1950 to 2021 from 111 catchments for daily
    and from 1976-2023 for hourly timestep. The average
    length of daily data is 33 years while for that of hourly it is 11 years.
    The dataset is available on `hydroshare <https://www.hydroshare.org/resource/86117a5f36cc4b7c90a5d54e18161c91/>`_

    Examples
    --------
    >>> from aqua_fetch import LamaHIce
    # by default the timestep is daily and data_type is 'total_upstrm'
    >>> dataset = LamaHIce()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='92', as_dataframe=True)
    >>> df = dynamic['92'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (26298, 36)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       111
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (11 out of 111)
       11
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(26298, 36), (26298, 36), (26298, 36),... (26298, 36), (26298, 36)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('92', as_dataframe=True,
    ...  dynamic_features=['swe', 'pet_mm', 'pcp_mm', 'q_cms_obs'])
    >>> dynamic['92'].shape
       (26298, 4)
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
    ((1, 154), 1, (26298, 36))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 26298, 'dynamic_features': 36})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (111, 2)
    >>> dataset.stn_coords('92')  # returns coordinates of station whose id is 92
        571777.0	309737.0
    >>> dataset.stn_coords(['92', '5'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('92')
    # get coordinates of two stations
    >>> dataset.area(['92', '5'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('92')
    ...
    # the data_type can also be 'intermediate_all'
    >>> dataset = LamaHIce(data_type='intermediate_all')
    ...
    # or 'intermediate_lowimp'
    >>> dataset = LamaHIce(data_type='intermediate_lowimp')
    >>> len(dataset.stations())
    86
    ...
    # the timestep can also be 'H'
    >>> dataset = LamaHIce(timestep='H')
    >>> _, dynamic = dataset.fetch(stations='79', as_dataframe=True)
    >>> dynamic['79'].shape
    (412848, 28)  # there are 28 dynamic features for hourly data
    
    """

    dirs_to_check = {
        'lamah_ice.zip': {
            'total_upstrm': 
                [os.path.join('lamah_ice', 'lamah_ice', 'A_basins_total_upstrm'), os.path.join('lamah_ice', 'lamah_ice', 'D_gauges')],
            'intermediate_all': 
                [os.path.join('lamah_ice', 'lamah_ice', 'B_basins_intermediate_all'), os.path.join('lamah_ice', 'lamah_ice', 'D_gauges')],
            'intermediate_lowimp': 
                [os.path.join('lamah_ice', 'lamah_ice', 'C_basins_intermediate_lowimp'), os.path.join('lamah_ice', 'lamah_ice', 'D_gauges')],
                          },
        'lamah_ice_hourly.zip': {
            'total_upstrm':
                [os.path.join('lamah_ice_hourly', 'lamah_ice_hourly', 'A_basins_total_upstrm'),
                 os.path.join('lamah_ice_hourly', 'lamah_ice_hourly', 'D_gauges')],
            'intermediate_all':
                [os.path.join('lamah_ice_hourly', 'lamah_ice_hourly', 'B_basins_intermediate_all'),
                 os.path.join('lamah_ice_hourly', 'lamah_ice_hourly', 'D_gauges')],
            'intermediate_lowimp':
                [os.path.join('lamah_ice_hourly', 'lamah_ice_hourly', 'C_basins_intermediate_lowimp'),
                 os.path.join('lamah_ice_hourly', 'lamah_ice_hourly', 'D_gauges')],
                                    },
        'Caravan_extension_lamahice.zip': {
            'total_upstrm': [],
            'intermediate_all': [],
            'intermediate_lowimp': []},
    }

    url = {
        'Caravan_extension_lamahice.zip':
            'https://www.hydroshare.org/resource/86117a5f36cc4b7c90a5d54e18161c91/data/contents/Caravan_extension_lamahice.zip',
        'lamah_ice.zip':
            'https://www.hydroshare.org/resource/86117a5f36cc4b7c90a5d54e18161c91/data/contents/lamah_ice.zip',
        'lamah_ice_hourly.zip':
            'https://www.hydroshare.org/resource/86117a5f36cc4b7c90a5d54e18161c91/data/contents/lamah_ice_hourly.zip'
    }
    _data_types = ['total_upstrm', 'intermediate_all', 'intermediate_lowimp']
    time_steps = ['D', 'H']
    DTYPES = {
        'total_upstrm': 'A_basins_total_upstrm',
        'intermediate_all': 'B_basins_intermediate_all',
        'intermediate_lowimp': 'C_basins_intermediate_lowimp'
    }

    def __init__(
            self,
            path=None,
            overwrite=False,
            *,
            timestep: str = "D",
            data_type: str = "total_upstrm",
            to_netcdf: bool = False,
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
            timestep :
                    possible values are ``D`` for daily or ``H`` for hourly timestep
            data_type :
                    possible values are ``total_upstrm``, ``intermediate_all``
                    or ``intermediate_lowimp``
        """

        # copy so we don't mutate the class-level dict shared across instances
        self.url = dict(self.url)
        if timestep == "D":
            self.url.pop("lamah_ice_hourly.zip", None)
        if timestep == 'H':
            # hourly mode reads everything from lamah_ice_hourly.zip; the
            # daily archive and the caravan extension are not used.
            self.url.pop('lamah_ice.zip', None)
            self.url.pop('Caravan_extension_lamahice.zip', None)

        super().__init__(path=path,
                         timestep=timestep,
                         data_type=data_type,
                         overwrite=overwrite,
                         to_netcdf=to_netcdf,
                         **kwargs)

        self.bbox = {'llcrnrlat': 63.0, 'urcrnrlat': 67.0, 'llcrnrlon': -25.0, 'urcrnrlon': -13.0}
        self.parallels = range(63, 67, 1)
        self.meridians = range(-25, -12, 2)

    def _infer_dynamic_features(self) -> List[str]:
        """
        LamaH-Ice ships different column sets than LamaH-CE (e.g. a ``qc_flag``
        column in the runoff files), so the names are taken from a real read of
        the first station rather than from the csv headers.
        """
        station = self.stations()[0]
        cols = self._read_stn_dyn(station).columns.to_list()
        skip = set(_CE_DATE_COLS) | set(_CE_Q_FLAG_COLS) | {'checked'}
        return [col for col in cols if col not in skip]

    def transform_boundary(self, boundary):
        """
        The LamaH-Ice shapefiles are in EPSG:3057 (Lambert conformal conic), not
        in the EPSG:3035 that :meth:`LamaHCE.transform_boundary` assumes, so the
        geometry is returned untouched (i.e. in the projected coordinates of the
        source shapefile).
        """
        return boundary

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'area_calc_basin': catchment_area(),
                'lat_gauge': gauge_latitude(),
                'slope_mean_basin': slope('mkm-1'),
                'lon_gauge': gauge_longitude(),
        }

    @property
    def dyn_map(self):
        return {
            'D': {
                'qobs': observed_streamflow_cms(),
                '2m_temp_min': min_air_temp_with_specifier('2m'),
                '2m_temp_max': max_air_temp_with_specifier('2m'),
                '2m_temp_mean': mean_air_temp_with_specifier('2m'),
                'prec': total_precipitation(),
                'pet': total_potential_evapotranspiration(),
                'ref_et_rav': 'ref_et_mm',
            },
            'H': {
                'qobs': observed_streamflow_cms(),
                '2m_temp': mean_air_temp_with_specifier('2m'),
                'prec': total_precipitation(),
                'pet': total_potential_evapotranspiration(),
                'ref_et_rav': 'ref_et_mm',
            }
        }

    @property
    def dyn_factors(self) -> Dict[str, float]:
        return {}

    @property
    def q_dir(self):
        """returns the path where q files are located"""
        if self.timestep == 'H':
            return os.path.join(
                self.path, 
                    "lamah_ice_hourly", 
                    "lamah_ice_hourly",
                    'D_gauges', '2_timeseries')

        return os.path.join(self.path, "lamah_ice", 
                            "lamah_ice",
                            'D_gauges', '2_timeseries')

    @property
    def boundary_file(self) -> os.PathLike:
        # A_basins_total_upstrm -> Basins_A.shp,
        # B_basins_intermediate_all -> Basins_B.shp,
        # C_basins_intermediate_lowimp -> Basins_C.shp
        letters = {'total_upstrm': 'A',
                   'intermediate_all': 'B',
                   'intermediate_lowimp': 'C'}
        return os.path.join(self.data_type_dir,
                            "3_shapefiles",
                            f"Basins_{letters[self.data_type]}.shp")

    @property
    def start(self):
        if self.timestep == "H":
            return pd.Timestamp("19760826 00:00")
        return pd.Timestamp("19500101")

    @property
    def end(self):
        if self.timestep == "H":
            return pd.Timestamp("20230930 23:00")
        return pd.Timestamp("20211231")

    @property
    def gauges_path(self):
        """returns the path where gauge data files are located"""
        if self.timestep == "H":
            return os.path.join(self.path, "lamah_ice_hourly", "lamah_ice_hourly", "D_gauges")
        return os.path.join(self.path, "lamah_ice", "lamah_ice", "D_gauges")

    @property
    def q_path(self):
        """path where all q files are located"""
        if self.timestep == "H":
            return os.path.join(self.gauges_path, "2_timeseries", "hourly")
        return os.path.join(self.gauges_path, "2_timeseries", "daily")

    def transform_stn_coords(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        transforms coordinates from EPSG:3057 (Lambert 1993) to EPSG:4326 (WGS84)

        """

        # following values are from .prj file 
        # Parameters for EPSG:3057
        lon_0 = -19.0        # Central Meridian
        lat_0 = 65.0         # Latitude of Origin
        lat_1 = 64.25        # First standard parallel
        lat_2 = 65.75        # Second standard parallel
        false_easting = 500000.0
        false_northing = 500000.0

        lat, lon = lcc_to_wgs84(
        df['long'].values, df['lat'].values, 
        lon_0, lat_0, lat_1, lat_2, false_easting, 
        false_northing)

        coords_m = pd.DataFrame({'lat': lat, 'long': lon}, index=df.index)
        return coords_m

    def met_fname(self, station):
        ts_folder = {'D': 'daily', 'H': 'hourly'}[self.timestep]
        return os.path.join(
                self.data_type_dir,
                f'2_timeseries{SEP}{ts_folder}{SEP}meteorological_data{SEP}ID_{station}.csv')

    def stations(self) -> List[str]:
        """
        returns names of stations as a list
        """
        return [fname.split('.')[0].split('_')[1] for fname in os.listdir(self._clim_ts_path())]

    def static_data(self) -> pd.DataFrame:
        """
        returns static data of all stations
        """

        if self.verbosity>2:
            print('reading static data')
        df = pd.concat([self.basin_attributes(), self.gauge_attributes()], axis=1)

        df.rename(columns=self.static_map, inplace=True)

        return self._standardize_area(df)

    def gauge_attributes(self) -> pd.DataFrame:
        """
        returns gauge attributes from following two files

            - Gauge_attributes.csv
            - hydro_indices_1981_2018.csv

        Returns
        -------
        pd.DataFrame
            a dataframe of shape (111, 28)
        """
        g_attr_fpath = os.path.join(self.gauges_path, "1_attributes", "Gauge_attributes.csv")

        df_gattr = pd.read_csv(g_attr_fpath, sep=';', index_col='id')
        df_gattr.index = df_gattr.index.astype(str)

        hydro_idx_fpath = os.path.join(self.gauges_path, "1_attributes", "hydro_indices_1981_2018.csv")

        df_hidx = pd.read_csv(hydro_idx_fpath, sep=';', index_col='id')
        df_hidx.index = df_hidx.index.astype(str)

        df = pd.concat([df_gattr, df_hidx], axis=1)

        df.columns = [col + "_gauge" for col in df.columns]

        return df

    def _catch_attr_path(self) -> os.PathLike:
        return os.path.join(self.data_type_dir, "1_attributes")

    def _clim_ts_path(self) -> str:
        p0 = "lamah_ice"
        p1 = "2_timeseries"
        p2 = "daily"

        if self.timestep == "H":
            p0 = "lamah_ice_hourly"
            p1 = "2_timeseries"
            p2 = "hourly"

        path = os.path.join(self.path, p0, p0,
                            self.DTYPES[self.data_type],
                            p1, p2, "meteorological_data")
        return path

    def catchment_attributes(self) -> pd.DataFrame:
        """returns catchment attributes as DataFrame with 90 columns
        """

        fpath = os.path.join(self._catch_attr_path(), "Catchment_attributes.csv")

        if self.data_type == 'intermediate_lowimp':
            df = pd.read_csv(fpath, index_col='id')
        else:
            df = pd.read_csv(fpath, sep=';', index_col='id')
        df.index = df.index.astype(str)
        return df

    def wat_bal_attrs(self) -> pd.DataFrame:
        """water balance attributes"""
        fpath = os.path.join(self._catch_attr_path(),
                             "water_balance.csv")

        df = pd.read_csv(fpath, sep=';', index_col='id')
        df.index = df.index.astype(str)
        df.columns = [col + "_all" for col in df.columns]
        return df

    def wat_bal_unfiltered(self) -> pd.DataFrame:
        """water balance attributes from unfiltered q"""
        fpath = os.path.join(self._catch_attr_path(),
                             "water_balance_unfiltered.csv")

        df = pd.read_csv(fpath, sep=';', index_col='id')
        df.index = df.index.astype(str)
        df.columns = [col + "_unfiltered" for col in df.columns]
        return df

    def basin_attributes(self) -> pd.DataFrame:
        """returns basin attributes which are catchment attributes, water
        balance all attributes and water balance filtered attributes

        Returns
        -------
        pd.DataFrame
            a dataframe of shape (111, 104) where 104 are the static
            catchment/basin attributes
        """
        cat = self.catchment_attributes()

        if self.timestep == 'D' and self.data_type == 'total_upstrm':
            wat_bal_all = self.wat_bal_attrs()
            wat_bal_filt = self.wat_bal_unfiltered()
            df = pd.concat([cat, wat_bal_all, wat_bal_filt], axis=1)
        else:
            df = cat
        df.columns = [col + '_basin' for col in df.columns]
        return df

    def fetch_static_features(
            self,
            stations: Union[str, list] = 'all',
            static_features: Union[str, list] = None
    ) -> pd.DataFrame:
        """
        fetches static features of one or more stations
        """
        df = self.static_data()
        df.index = df.index.astype(str)

        static_features = validate_attributes(static_features, self.static_features, 'static_features')
        stations = validate_attributes(stations, self.stations(), 'stations')

        df = df.loc[stations, static_features]

        return df

    def q_mm(
            self,
            stations: Union[str, List[str]] = None
    ) -> pd.DataFrame:
        """
        returns streamflow in the units of milimeter per timestep (e.g. mm/day or mm/hour). This is obtained
        by diving q_cms by the area upstream of the gauge
        (:meth:`LamaHCE.total_upstrm_area`, which equals :meth:`area` for the
        ``total_upstrm`` delineation only).

        parameters
        ----------
        stations : str/list
            name/names of stations. Default is None, which will return
            area of all stations

        Returns
        --------
        pd.DataFrame
            a :obj:`pandas.DataFrame` whose indices are time-steps and columns
            are catchment/station ids.

        """
        if self.timestep.lower().startswith('d'):
            conversion_factor = 86400
        elif self.timestep.lower().startswith('h'):
            conversion_factor = 3600
        else:
            raise ValueError(f"Invalid timestep: {self.timestep}. ")

        stations = validate_attributes(stations, self.stations(), 'stations')
        q = self.fetch_q(stations)
        area_m2 = self.area(stations) * 1e6  # area in m2
        q = (q / area_m2) * conversion_factor  # cms to m
        return q * 1e3  # to mm

    def fetch_q(
            self,
            stations: Union[str, List[str]] = None,
            qc_flag: int = None
    ):
        """
        returns streamflow for one or more stations

        parameters
        -----------
        stations : str/List[str]
            name or names of stations for which streamflow is to be fetched
        qc_flag : int
            following flags are available
            40 Good
            80 Fair
            100 Estimated
            120 suspect
            200 unchecked
            250 missing

        Returns
        --------
        pd.DataFrame
            a :obj:`pandas.DataFrame` whose index is the time and columns are names of stations
            For daily timestep, the dataframe has shape of 32630 rows and 111 columns

        """
        stations = validate_attributes(stations, self.stations(), 'stations')

        cpus = self.processes or min(get_cpus(), 16)
        if len(stations)<=10: cpus=1

        if self.verbosity>1:
            print(f"fetching streamflow for {len(stations)} stations with {cpus} cpus")

        if cpus == 1:
            qs = []
            for stn in stations:
                qs.append(self.fetch_stn_q(stn, qc_flag=qc_flag))
        else:
            qc_flag = [qc_flag for _ in range(len(stations))]
            with  cf.ProcessPoolExecutor(max_workers=cpus) as executor:
                qs = list(executor.map(
                    self.fetch_stn_q,
                    stations,
                    qc_flag
                ))

        df = pd.concat(qs, axis=1)
        df.columns = stations
        return df

    def fetch_stn_q(
            self,
            stn: str,
            qc_flag: int = None
    ) -> pd.Series:
        """returns streamflow for single station"""

        fpath = os.path.join(self.q_path, f"ID_{stn}.csv")

        if not os.path.exists(fpath):
            timestep = {'H': 'h', 'D': 'd'}[self.timestep]

            return pd.Series(dtype=np.float32,
                             index=pd.date_range(self.start, self.end, freq=timestep),
                             name='qobs')

        df = pd.read_csv(fpath, sep=';',
                         dtype={'YYYY': int,
                                'MM': int,
                                'DD': int,
                                'qobs': np.float32,
                                'qc_flag': np.float32
                                })

        # todo : consider quality code!

        # vectorized date parsing from the YYYY/MM/DD integer columns. This is
        # ~200x faster than a per-row datetime.strptime via df.apply(axis=1)
        # (which dominated the hourly read time) and yields identical timestamps.
        index = pd.to_datetime(
            df[['YYYY', 'MM', 'DD']].rename(
                columns={'YYYY': 'year', 'MM': 'month', 'DD': 'day'}))

        if self.timestep == "H":
            # the hourly q file has no explicit hour column; rows are stored
            # sequentially within each (YYYY, MM, DD) group, so derive the
            # hour offset from cumcount within the day.
            hour = df.groupby(['YYYY', 'MM', 'DD']).cumcount()
            df.index = index + pd.to_timedelta(hour, unit='h')
        else:
            df.index = index
        s = df['qobs']
        return s

    def fetch_clim_features(
            self,
            stations: Union[str, List[str]] = None
    ):
        """Returns climate time series data for one or more stations

        Returns
        -------
        pd.DataFrame
        """
        stations = validate_attributes(stations, self.stations(), 'stations')

        dfs = []
        for stn in stations:
            dfs.append(self.fetch_stn_meteo(stn))

        return pd.concat(dfs, axis=1)

    def fetch_stn_meteo(
            self,
            stn: str,
            nrows: int = None,
            usecols: List[str] = None,
    ) -> pd.DataFrame:
        """returns climate/meteorological time series data for one station

        ``usecols`` restricts the columns read from disk (it must include the
        date columns needed to build the index); ``None`` reads all columns.

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` with 23 columns
        """
        fpath = os.path.join(self._clim_ts_path(), f"ID_{stn}.csv")

        dtypes = {
            "YYYY": np.int32,
            "DD": np.int32,
            "MM": np.int32,
            "2m_temp_max": np.float32,
            "2m_temp_mean": np.float32,
            "2m_temp_min": np.float32,
            "2m_dp_temp_max": np.float32,
            "2m_dp_temp_mean": np.float32,
            "2m_dp_temp_min": np.float32,
            "10m_wind_u": np.float32,
            "10m_wind_v": np.float32,
            "fcst_alb": np.float32,
            "lai_high_veg": np.float32,
            "lai_low_veg": np.float32,
            "swe": np.float32,
            "surf_net_solar_rad_max": np.int32,
            "surf_net_solar_rad_mean": np.int32,
            "surf_net_therm_rad_max": np.int32,
            "surf_net_therm_rad_mean": np.int32,
            "surf_press": np.float32,
            "total_et": np.float32,
            "prec": np.float32,
            "volsw_123": np.float32,
            "volsw_4": np.float32,
            "prec_rav": np.float32,
            "prec_carra": np.float32,
            # hourly-timestep columns. The daily aggregates above cover the
            # daily files; the hourly files store the instantaneous variables
            # under un-suffixed names. Declaring an explicit dtype for every
            # hourly column avoids pandas' float64 type-inference on ~15
            # columns, which measurably slows the read of these 631k-row,
            # 97 MB files, and (being half the width of float64) also shrinks
            # the frames shipped back from the worker processes. float32 is
            # used only where its value range was verified safe: the
            # instantaneous variables mirror how the daily aggregates of the
            # same quantities are already read, and the reanalysis (``_rav``)
            # columns all fall well below 1000 in magnitude -- float32 keeps
            # more precision there than the source's own decimals. The one
            # exception is ``surf_press_rav`` (~9.6e4, i.e. 7 significant
            # figures) which is kept at full float64 precision.
            "2m_temp": np.float32,
            "2m_dp_temp": np.float32,
            "surf_net_solar_rad": np.float32,
            "surf_net_therm_rad": np.float32,
            "pet": np.float32,
            "surf_press_rav": np.float64,
            "total_et_rav": np.float32,
            "2m_temp_rav": np.float32,
            "10m_wind_u_rav": np.float32,
            "10m_wind_v_rav": np.float32,
            "surf_dwn_therm_rad_rav": np.float32,
            "surf_outg_therm_rad_rav": np.float32,
            "surf_dwn_solar_rad_rav": np.float32,
            "2m_qv_rav": np.float32,
            "grdflx_rav": np.float32,
        }

        if not os.path.exists(fpath):
            raise FileNotFoundError(f"File not found: {fpath}")

        df = pd.read_csv(fpath, sep=';', dtype=dtypes, nrows=nrows, usecols=usecols)

        # vectorized date parsing from the YYYY/MM/DD integer columns. This is
        # ~200x faster than a per-row datetime.strptime via df.apply(axis=1)
        # (which dominated the hourly read time) and yields identical timestamps.
        index = pd.to_datetime(
            df[['YYYY', 'MM', 'DD']].rename(
                columns={'YYYY': 'year', 'MM': 'month', 'DD': 'day'}))

        if self.timestep == "H":
            df.index = index + pd.to_timedelta(df['HOD'], unit='h')
            for col in ['YYYY', 'MM', 'DD', 'DOY', 'hh', 'mm', 'HOD']:
                if col in df:
                    df.pop(col)
        else:
            df.index = index
            for col in ['YYYY', 'MM', 'DD', 'DOY']:
                if col in df:
                    df.pop(col)

        return df

    @property
    def data_type_dir(self):
        p = "lamah_ice"
        if self.timestep == "H":
            p = "lamah_ice_hourly"
        return os.path.join(self.path, p, p, self.DTYPES[self.data_type])

    def _read_dynamic(
            self,
            stations,
            dynamic_features: Union[str, list] = 'all',
            st=None,
            en=None,
    ):
        """Reads features of one or more station"""

        cpus = self.processes or min(get_cpus(), 16)
        st, en = self._check_length(st, en)

        # spinning up a process pool (and pickling ``self`` to every worker)
        # only pays off once there are >=2 stations to read in parallel; for a
        # single station the pool startup is pure overhead, so read it inline.
        if len(stations) < 2:
            cpus = 1

        if self.verbosity>1:
            print(f"reading dynamic data for {len(stations)} stations with {cpus} cpus")

        # when the caller wants only a strict subset of the dynamic features,
        # tell the per-station reader so it can read just those columns off
        # disk (a meteo file has 34 columns; a single-feature fetch otherwise
        # parses and ships all of them just to throw 33 away). Requesting the
        # full feature set leaves the read path exactly as before.
        subset = None
        if dynamic_features != 'all' and len(dynamic_features) < len(self.dynamic_features):
            subset = list(dynamic_features)

        if cpus > 1:

            reader = self._read_stn_dyn if subset is None else partial(
                self._read_stn_dyn, dynamic_features=subset)

            with  cf.ProcessPoolExecutor(max_workers=cpus) as executor:
                results = executor.map(
                    reader,
                    stations
                )

            if dynamic_features == 'all':
                results = {stn: data for stn, data in zip(stations, results)}
            else:
                results = {stn: data.loc[st:en, dynamic_features] for stn, data in zip(stations, results)}
        else:
            results = {}
            for idx, stn in enumerate(stations):
                if dynamic_features == 'all':
                    results[stn] = self._read_stn_dyn(stn)
                else:
                    results[stn] = self._read_stn_dyn(stn, dynamic_features=subset).loc[st:en, dynamic_features]

                if self.verbosity and idx % 10 == 0:
                    print(f"processed {idx} stations")

        return results

    def _read_stn_dyn(
            self,
            station: str,
            dynamic_features: Union[str, List[str]] = None,
    ) -> pd.DataFrame:
        """
        Reads daily dynamic (meteorological + streamflow) data for one catchment
        and returns as DataFrame.

        ``dynamic_features`` (a list of the standardized/renamed feature names)
        restricts the read to just those columns; ``None`` reads everything.
        """

        if self.verbosity>2:
            print(f"reading data for {station}")

        # translate the requested (renamed) feature names back to the on-disk
        # column names so we can hand read_csv a ``usecols`` and avoid parsing
        # the ~30 columns we would only discard. ``None`` -> read everything.
        meteo_usecols = None
        want_q = True
        if dynamic_features is not None:
            feats = [dynamic_features] if isinstance(dynamic_features, str) else list(dynamic_features)
            renamed_to_src = {ren: src for src, ren in self.dyn_map[self.timestep].items()}
            want_q = self._q_name in feats
            date_cols = ['YYYY', 'MM', 'DD', 'HOD'] if self.timestep == 'H' else ['YYYY', 'MM', 'DD']
            meteo_src = [renamed_to_src.get(f, f) for f in feats if f != self._q_name]
            meteo_usecols = date_cols + [c for c in meteo_src if c not in date_cols]

        # fetch_stn_q / fetch_stn_meteo already return freshly-built objects and
        # the concat below copies again, so defensive .copy() calls here only
        # duplicated the 631k-row meteo frame (~68 MB) for nothing.
        met = self.fetch_stn_meteo(station, usecols=meteo_usecols)

        # drop duplicated index from met (boolean indexing already returns a copy)
        met = met.loc[~met.index.duplicated(keep='first')]

        if want_q:
            q = self.fetch_stn_q(station)
            df = pd.concat([met, q], axis=1).loc[self.start:self.end, :].copy()
        else:
            df = met.loc[self.start:self.end, :].copy()

        for col in self.dyn_map[self.timestep]:
            if col in df.columns:
                df.rename(columns={col: self.dyn_map[self.timestep][col]}, inplace=True)

        df.columns.name = "dynamic_features"
        df.index.name = "time"

        df = df.sort_index()
        # Ensure df always extends to self.end
        if df.index[-1] < self.end or df.index[0] > self.start:
            timestep = {'H': 'h', 'D': 'd'}[self.timestep]
            # Create complete date range from start of existing data to self.end
            complete_range = pd.date_range(start=self.start, end=self.end, freq=timestep)
            # Reindex to fill missing dates with NaN
            df = df.reindex(complete_range)
        return df

    @property
    def dynamic_fnames(self):
        return [f"{feature}.nc" for feature in self.dynamic_features]
