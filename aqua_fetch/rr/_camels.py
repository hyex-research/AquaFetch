import io
import os
import csv
import json
import glob
import time
import zlib
import shutil
import zipfile
import warnings
import functools
from pathlib import Path
import concurrent.futures as cf
from typing import Union, List, Dict, Tuple

import numpy as np
import pandas as pd

from .utils import _RainfallRunoff, apply_dyn_factors, n_workers, cache_name, ymd_index
from .._geom_utils import (epsg25832_to_wgs84, epsg2056_point_to_wgs84, laea_to_wgs84,
                           osgb36_to_wgs84, tmerc_to_wgs84, world_mercator_to_wgs84)
from ..utils import get_cpus, download_and_unzip, BROWSER_HEADERS
from ..utils import validate_attributes, download, unzip

from .._backend import netCDF4, xarray as xr, fiona

if netCDF4 is not None:
    from netCDF4 import date2num

from ._map import (
    observed_streamflow_cms,
    observed_streamflow_mm,
    observed_water_level_cm,
    observed_water_level_m,
    cloud_cover,
    mean_air_temp,
    min_air_temp_with_specifier,
    max_air_temp_with_specifier,
    max_air_temp,
    min_air_temp,
    mean_air_temp_with_specifier,
    total_precipitation,
    total_precipitation_with_specifier,
    total_potential_evapotranspiration,
    total_potential_evapotranspiration_with_specifier,
    simulated_streamflow_cms,
    actual_evapotranspiration,
    actual_evapotranspiration_with_specifier,
    solar_radiation_with_specifier,
    mean_vapor_pressure,
    mean_vapor_pressure_with_specifier,
    mean_rel_hum,
    mean_rel_hum_with_specifier,
    rel_hum_with_specifier,
    mean_windspeed,
    max_wind_gust,
    u_component_of_wind,
    v_component_of_wind,
    u_component_of_wind_at_10m,
    v_component_of_wind_at_10m,
    mean_air_pressure,
    solar_radiation,
    solar_radiation_with_spatial_stat,
    daylight_solar_radiation,
    downward_longwave_radiation,
    MJ_M2_DAY_TO_WM2,
    KJ_M2_DAY_TO_WM2,
    J_CM2_DAY_TO_WM2,
    snow_water_equivalent,
    mean_specific_humidity,
    soil_moisture_layer1,
    soil_moisture_layer2,
    soil_moisture_layer3,
    soil_moisture_layer4,
    mean_dewpoint_temperature_at_2m,
    catchment_area_with_specifier,
    snow_water_equivalent_with_specifier,
    snow_depth,
)

from ._map import (
    catchment_area,
    gauge_latitude,
    gauge_longitude,
    slope,
    gauge_elevation_meters,
    catchment_elevation_meters,
    urban_fraction,
    urban_fraction_with_specifier,
    grass_fraction,
    grass_fraction_with_specifier,
    crop_fraction,
    crop_fraction_with_specifier,
    catchment_perimeter,
    aridity_index,
    med_catchment_elevation_meters,
    min_catchment_elevation_meters,
    max_catchment_elevation_meters,
    soil_depth,
    population_density,
    )

# directory separator
SEP = os.sep


class CAMELS_US(_RainfallRunoff):
    """
    This is a dataset of 671 US catchments with 59 static catchment features
    and 9 catchment averaged dynamic features for each catchment. The dynamic features are
    daily timeseries from 1980-01-01 to 2014-12-31. The data is downloaded
    from its `zenodo repository <https://zenodo.org/records/15529996>`_ . For more details
    on data refer to `Newman et al., 2015 <https://doi.org/10.5194/hess-19-209-2015>`_ ,
    `Newman et al., 2022 <https://gdex.ucar.edu/dataset/camels.html.>`_ and
    `Addor et al., 2017 <https://hess.copernicus.org/articles/21/5293/2017/>`_.

    Please note this data is also known as "CAMELS" however, we have named it CAMELS_US
    to differentiate it from other CAMELS like datasts from other parts of the world.

    Examples
    --------
    >>> from aqua_fetch import CAMELS_US
    >>> dataset = CAMELS_US()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='11478500', as_dataframe=True)
    >>> df = dynamic['11478500'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (12784, 9)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       671
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (67 out of 671)
       67
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(12784, 9), (12784, 9), (12784, 9),... (12784, 9), (12784, 9)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('11478500', as_dataframe=True,
    ...  dynamic_features=['pcp_mm', 'swdownrad_wm2', 'airtemp_C_max', 'airtemp_C_min', 'q_cms_obs'])
    >>> dynamic['11478500'].shape
       (12784, 5)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='11478500', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['11478500'].shape
    ((1, 59), 1, (12784, 9))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 12784, 'dynamic_features': 9})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (671, 2)
    >>> dataset.stn_coords('11478500')  # returns coordinates of station whose id is 11478500
        40.480419	-123.890877
    >>> dataset.stn_coords(['11478500', '14020000'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('11478500')
    # get coordinates of two stations
    >>> dataset.area(['11478500', '14020000'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('11478500')

    """
    DATASETS = ['CAMELS_US']

    url = {
        'camels_attributes_v2.0.pdf': 'https://zenodo.org/records/15529996/files/',
        'camels_attributes_v2.0.xlsx': 'https://zenodo.org/records/15529996/files/',
        'camels_clim.txt': 'https://zenodo.org/records/15529996/files/',
        'camels_geol.txt': 'https://zenodo.org/records/15529996/files/',
        'camels_hydro.txt': 'https://zenodo.org/records/15529996/files/',
        'camels_name.txt': 'https://zenodo.org/records/15529996/files/',
        'camels_soil.txt': 'https://zenodo.org/records/15529996/files/',
        'camels_topo.txt': 'https://zenodo.org/records/15529996/files/',
        'camels_vege.txt': 'https://zenodo.org/records/15529996/files/',
        'readme.txt': 'https://zenodo.org/records/15529996/files/',
        'basin_timeseries_v1p2_metForcing_obsFlow.zip': 'https://zenodo.org/records/15529996/files/',
        'basin_set_full_res.zip': 'https://zenodo.org/records/15529996/files/',
    }

    folders = {'daymet': f'basin_mean_forcing{SEP}daymet',
               'maurer': f'basin_mean_forcing{SEP}maurer',
               'nldas': f'basin_mean_forcing{SEP}nldas',
               'v1p15_daymet': f'basin_mean_forcing{SEP}v1p15{SEP}daymet',
               'v1p15_nldas': f'basin_mean_forcing{SEP}v1p15{SEP}nldas',
               'elev_bands': f'elev{SEP}daymet',
               'hru': f'hru_forcing{SEP}daymet'}

    dynamic_features_ = ['dayl(s)', 'prcp(mm/day)', 'srad(W/m2)',
                         'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)', 'Flow']

    def __init__(
            self,
            path:Union[str, os.PathLike]=None,
            data_source: str = 'daymet',
            **kwargs
    ):

        """
        parameters
        ----------
        path : str
            If the data is alredy downloaded then provide the complete
            path to it. If None, then the data will be downloaded.
            The data is downloaded once and therefore susbsequent
            calls to this class will not download the data unless
            ``overwrite`` is set to True.
        data_source : str
            source of meteorological timeseries data. Allowed values are
    
                - daymet
                - maurer
                - nldas
                - v1p15_daymet
                - v1p15_nldas
        """
        assert data_source in self.folders, f'allowed data sources are {self.folders.keys()}'
        self.data_source = data_source

        super().__init__(path=path, name="CAMELS_US", **kwargs)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        # An archive is satisfied (not needed) when *any* of these alternatives
        # exists on disk: the archive itself, its extracted sibling folder, or a
        # consolidated cache that fully captures its content. The third entry
        # is what allows free_disk_space("redundant") to delete the per-station
        # tree without triggering a re-download next time.
        satisfied_by_cache = {
            'basin_timeseries_v1p2_metForcing_obsFlow.zip': self.dyn_fpath,
        }

        for fname, url in self.url.items():

            fpath = os.path.join(self.path, fname)
            furl = f"{url}{fname}"

            if fname.endswith('.zip'):
                extracted = os.path.join(self.path, fname[:-len('.zip')])
                cache = satisfied_by_cache.get(fname)
                already_have = (
                    os.path.exists(fpath)
                    or os.path.exists(extracted)
                    or (cache is not None and os.path.exists(cache))
                )
            else:
                already_have = os.path.exists(fpath)

            if already_have and not self.overwrite:
                continue

            if self.verbosity:
                print(f"downloading {fname} from {url}")

            download(furl, self.path, fname, verbosity=self.verbosity)

            unzip(self.path, verbosity=self.verbosity)

        self.dataset_dir = os.path.join(
            self.path,
            f'basin_timeseries_v1p2_metForcing_obsFlow{SEP}basin_dataset_public_v1p2')

        self._static_features = self._static_data().columns.tolist()
        self._maybe_to_netcdf()

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(
            self.path,
            "basin_set_full_res",
            "HCDN_nhru_final_671.shp"
        )

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'area_gages2': catchment_area(),
                'slope_mean': slope('mkm-1'),
                'gauge_lat': gauge_latitude(),
                'gauge_lon': gauge_longitude(),
        }

    @property
    def dyn_map(self):
        return {
            'Flow': observed_streamflow_cms(),  # todo : check units
            'tmin(C)': min_air_temp(),
            'tmax(C)': max_air_temp(),
            'prcp(mm/day)': total_precipitation(),
            'swe(mm)': snow_water_equivalent(),
            'pet_mean': total_potential_evapotranspiration(),
            'vp(Pa)': mean_vapor_pressure(),  # todo: convert frmo Pa to hpa
            # Daymet's ``srad`` is averaged over the DAYLIGHT period, not over the
            # 24-h day, so it is not comparable with the daily-mean W m-2 that
            # every other dataset serves as ``swdownrad_wm2``. The native value is
            # kept faithfully under its own name and the 24-h mean is derived
            # from it in _read_stn_dyn(). The maurer and nldas forcings were
            # resampled into the Daymet format and share the convention
            # (verified: raw Kt 1.22 and 1.07, i.e. both impossible as 24-h
            # means). Note _read_stn_dyn currently hardcodes the daymet
            # filename, so those two sources cannot actually be read yet;
            # that is a pre-existing limitation, not one introduced here.
            'srad(W/m2)': daylight_solar_radiation(),
        }

    @property
    def dynamic_features(self) -> List[str]:
        # swdownrad_wm2 is derived (see _read_stn_dyn) so it has no raw counterpart
        # in dynamic_features_
        return [self.dyn_map.get(feat, feat) for feat in self.dynamic_features_] + [solar_radiation()]

    @property
    def dyn_factors(self) -> Dict[str, float]:
        return {
            observed_streamflow_cms(): 0.0283168,
        }

    @property
    def start(self):
        return pd.Timestamp("19800101")

    @property
    def end(self):
        return pd.Timestamp("20141231")

    @property
    def static_features(self)->List[str]:
        return self._static_features

    def stations(self) -> list:
        streamflow_dir = os.path.join(self.dataset_dir, 'usgs_streamflow')
        if os.path.exists(streamflow_dir):
            stns = []
            for _dir in os.listdir(streamflow_dir):
                cat = os.path.join(streamflow_dir, _dir)
                stns += [fname.split('_')[0] for fname in os.listdir(cat)]
        else:
            # Fallback when the per-station tree has been removed by
            # free_disk_space("redundant"): read station ids from the cached
            # static_features.csv, which contains them as the gauge_id index.
            static_fpath = os.path.join(self.path, 'static_features.csv')
            df = pd.read_csv(static_fpath, dtype={'gauge_id': str}, usecols=['gauge_id'])
            stns = df['gauge_id'].astype(str).tolist()

        # remove stations for which static values are not available
        for stn in ['06775500', '06846500', '09535100']:
            if stn in stns:
                stns.remove(stn)

        return stns

    def _redundant_after_consolidation(self) -> List[Tuple[str, str]]:
        """
        The per-station forcing and streamflow text files inside
        ``basin_timeseries_v1p2_metForcing_obsFlow/`` become redundant once
        ``camels_us_D_v2.nc`` exists, since all dynamic data is baked into that
        consolidated NetCDF.

        Caveat: the cache reflects whichever ``data_source`` was selected when
        it was first built. Once ``free_disk_space("redundant")`` removes the
        per-station tree, switching ``data_source`` (e.g. ``daymet`` ?
        ``nldas``) requires re-downloading the source archive to rebuild the
        cache. Run cleanup only after settling on a data source.
        """
        return [
            (os.path.join(self.path, 'basin_timeseries_v1p2_metForcing_obsFlow'),
             self.dyn_fpath),
        ]

    def _read_stn_dyn(
            self,
            stn:str,
    ):

        assert isinstance(stn, str)
        df = None
        dir_name = self.folders[self.data_source]
        for cat in os.listdir(os.path.join(self.dataset_dir, dir_name)):
            cat_dirs = os.listdir(os.path.join(self.dataset_dir, f'{dir_name}{SEP}{cat}'))
            stn_file = f'{stn}_lump_cida_forcing_leap.txt'
            if stn_file in cat_dirs:
                df = pd.read_csv(os.path.join(self.dataset_dir,
                                                f'{dir_name}{SEP}{cat}{SEP}{stn_file}'),
                                    sep="\s+|;|:",
                                    skiprows=4,
                                    engine='python',
                                    names=['Year', 'Mnth', 'Day', 'Hr', 'dayl(s)', 'prcp(mm/day)', 'srad(W/m2)',
                                        'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)'],
                                    )
                df.index = pd.to_datetime(
                    df['Year'].map(str) + '-' + df['Mnth'].map(str) + '-' + df['Day'].map(str))

        flow_dir = os.path.join(self.dataset_dir, 'usgs_streamflow')
        for cat in os.listdir(flow_dir):
            cat_dirs = os.listdir(os.path.join(flow_dir, cat))
            stn_file = f'{stn}_streamflow_qc.txt'
            if stn_file in cat_dirs:
                fpath = os.path.join(flow_dir, f'{cat}{SEP}{stn_file}')
                q_df = pd.read_csv(fpath,
                                    sep=r"\s+",
                                    names=['station', 'Year', 'Month', 'Day', 'Flow', 'Flag'],
                                    engine='python')
                q_df.index = pd.to_datetime(
                    q_df['Year'].map(str) + '-' + q_df['Month'].map(str) + '-' + q_df['Day'].map(str))

        stn_df = pd.concat([
            df[['dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)']],
            q_df['Flow']],
            axis=1)

        stn_df.rename(columns=self.dyn_map, inplace=True)

        self._apply_dyn_factors(stn_df)

        # Daymet reports srad as the mean over the daylight hours only. Weighting
        # by the daylight fraction of the day gives the 24-h mean that the
        # canonical ``swdownrad_wm2`` promises and that every other dataset serves.
        stn_df[solar_radiation()] = (
            stn_df[daylight_solar_radiation()] * stn_df['dayl(s)'] / 86400.0
        )

        return stn_df

    def _static_data(self)->pd.DataFrame:
        static_fpath = os.path.join(self.path, 'static_features.csv')
        if not os.path.exists(static_fpath):
            files = glob.glob(f"{self.path}/*.txt")

            static_dfs = []
            for f in files:
                if not f.endswith('readme.txt'):
                    if self.verbosity>2:
                        print(f"reading {f}")
                    # index should be read as string

                    _df = pd.read_csv(f, sep=';', index_col='gauge_id', dtype={'gauge_id': str})

                    static_dfs.append(_df)
            static_df = pd.concat(static_dfs, axis=1)
            static_df.to_csv(static_fpath, index_label='gauge_id')
        else:  # index should be read as string bcs it has 0s at the start
            static_df = pd.read_csv(static_fpath, index_col='gauge_id', dtype={'gauge_id': str})

        static_df.index = static_df.index.astype(str)

        static_df = static_df#.loc[stn_id][features]
        if isinstance(static_df, pd.Series):
            static_df = pd.DataFrame(static_df).transpose()

        static_df.rename(columns=self.static_map, inplace=True)
        
        return static_df


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


class CAMELS_AUS(_RainfallRunoff):
    """
    This is a dataset of 561 Australian catchments with 187 static features and
    28 dyanmic features for each catchment. The dyanmic features are timeseries
    from 1950-01-01 to 2022-03-31. By default this class reads version 2 of CAMELS-AUS dataset
    following `Fowler et al., 2024 <https://doi.org/10.5194/essd-2024-263>`_ .

    If ``version`` is 1 then this class reads data following `Fowler et al., 2021 <https://doi.org/10.5194/essd-13-3847-2021>`_
    which is a dataset of 222 Australian catchments with 161 static features
    and 26 dyanmic features for each catchment. The dyanmic features are
    timeseries from 1957-01-01 to 2018-12-31.

    Examples
    --------
    >>> from aqua_fetch import CAMELS_AUS
    >>> dataset = CAMELS_AUS()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='912101A', as_dataframe=True)
    >>> df = dynamic['912101A'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (26388, 28)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       561
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (56 out of 561)
       56
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(26388, 28), (26388, 28), (26388, 28),... (26388, 28), (26388, 28)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('912101A', as_dataframe=True,
    ...  dynamic_features=['airtemp_C_awap_max', 'pcp_mm_awap', 'et_morton_actual_SILO', 'q_cms_obs'])
    >>> dynamic['912101A'].shape
       (26388, 4)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='912101A', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['912101A'].shape
    ((1, 187), 1, (26388, 28))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 26388, 'dynamic_features': 28})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (561, 2)
    >>> dataset.stn_coords('912101A')  # returns coordinates of station whose id is 912101A
        -38.214199	-71.8283
    >>> dataset.stn_coords(['912101A', '912105A'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('912101A')
    # get coordinates of two stations
    >>> dataset.area(['912101A', '912105A'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('912101A')
    ...
    # The version 1 can be of CAMELS_AUS can be accessed as below
    >>> dataset = CAMELS_AUS(version=1)
    >>> len(dataset.stations())
    222
    >>> _, dynamic = dataset.fetch(stations='912101A', as_dataframe=True)
    >>> dynamic['912101A'].shape
    (23376, 26)
    """

    # todo : v1 and v2 in the same folder with the same cache name, so switching 
    # version there can read the other version's files.

    url = 'https://doi.pangaea.de/10.1594/PANGAEA.921850'
    url_v2 = "https://zenodo.org/records/14289037"
    urls = {1: {
        "01_id_name_metadata.zip": "https://download.pangaea.de/dataset/921850/files/",
        "02_location_boundary_area.zip": "https://download.pangaea.de/dataset/921850/files/",
        "03_streamflow.zip": "https://download.pangaea.de/dataset/921850/files/",
        "04_attributes.zip": "https://download.pangaea.de/dataset/921850/files/",
        "05_hydrometeorology.zip": "https://download.pangaea.de/dataset/921850/files/",
        "CAMELS_AUS_Attributes&Indices_MasterTable.csv": "https://download.pangaea.de/dataset/921850/files/",
        # "Units_01_TimeseriesData.pdf": "https://download.pangaea.de/dataset/921850/files/",
        # "Units_02_AttributeMasterTable.pdf": "https://download.pangaea.de/dataset/921850/files/",
    },
        2: {
            "01_id_name_metadata.zip": "https://zenodo.org/records/14289037/files/",
            "02_location_boundary_area.zip": "https://zenodo.org/records/14289037/files/",
            "03_streamflow.zip": "https://zenodo.org/records/14289037/files/",
            "04_attributes.zip": "https://zenodo.org/records/14289037/files/",
            "05_hydrometeorology.zip": "https://zenodo.org/records/14289037/files/",
            "CAMELS_AUS_Attributes&Indices_MasterTable.csv": "https://zenodo.org/records/14289037/files/",
            "CAMELS_AUS_v2_Data_Description.pdf": "https://zenodo.org/records/14289037/files/",
        }
    }

    folders = {1: {
        'streamflow_MLd': f'03_streamflow{SEP}03_streamflow',
        'streamflow_MLd_inclInfilled': f'03_streamflow{SEP}03_streamflow',
        'streamflow_mmd': f'03_streamflow{SEP}03_streamflow',

        'et_morton_actual_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
        'et_morton_point_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
        'et_morton_wet_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
        'et_short_crop_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
        'et_tall_crop_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
        'evap_morton_lake_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
        'evap_pan_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
        'evap_syn_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',

        'precipitation_AWAP': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}01_precipitation_timeseries',
        'precipitation_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}01_precipitation_timeseries',
        'precipitation_var_AWAP': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}01_precipitation_timeseries',

        'solarrad_AWAP': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AWAP',
        'tmax_AWAP': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AWAP',
        'tmin_AWAP': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AWAP',
        'vprp_AWAP': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AWAP',

        'mslp_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        'radiation_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        'rh_tmax_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        'rh_tmin_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        'tmax_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        'tmin_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        'vp_deficit_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        'vp_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
    },
        2: {
            'streamflow_MLd': '03_streamflow',
            'streamflow_MLd_inclInfilled': '03_streamflow',
            'streamflow_mmd': '03_streamflow',

            'et_morton_actual_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
            'et_morton_point_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
            'et_morton_wet_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
            'et_short_crop_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
            'et_tall_crop_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
            'evap_morton_lake_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
            'evap_pan_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',
            'evap_syn_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}02_EvaporativeDemand_timeseries',

            'precipitation_AGCD': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}01_precipitation_timeseries',
            'precipitation_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}01_precipitation_timeseries',
            'precipitation_var_AGCD': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}01_precipitation_timeseries',

            # 'solarrad_AWAP': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AGCD{SEP}solarrad_AWAP',
            'tmax_AGCD': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AGCD',
            'tmin_AGCD': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AGCD',
            'vapourpres_h09_AGCD': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AGCD',
            'vapourpres_h15_AGCD': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}AGCD',

            'mslp_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
            'radiation_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
            'rh_tmax_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
            'rh_tmin_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
            'tmax_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
            'tmin_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
            'vp_deficit_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
            'vp_SILO': f'05_hydrometeorology{SEP}05_hydrometeorology{SEP}03_Other{SEP}SILO',
        }
    }

    def __init__(
            self,
            path: str = None,
            version: int = 2,
            to_netcdf: bool = True,
            overwrite: bool = False,
            verbosity: int = 1,
            **kwargs
    ):
        """
        Arguments:
            path: path where the CAMELS_AUS dataset has been downloaded. This path
                must contain five zip files and one xlsx file. If None, then the
                data will be downloaded.
            version: version of the dataset to download. Allowed values are 1 and 2.
            to_netcdf :
        """
        self.version = version

        super().__init__(path=path, verbosity=verbosity, **kwargs)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        for _file, url in self.urls[version].items():
            fpath = os.path.join(self.path, _file)

            if os.path.exists(fpath) and overwrite:
                os.remove(fpath)
                if verbosity > 0: print(f"Re-downloading {_file} from {url + _file} at {fpath}")
                download(url + _file, outdir=self.path, fname=_file, verbosity=self.verbosity)

            elif not os.path.exists(fpath):
                if verbosity > 0:
                    print(f"Downloading {_file} from {url + _file} at {fpath}")
                download(url + _file, outdir=self.path, fname=_file, verbosity=self.verbosity)
            elif verbosity > 0:
                print(f"{_file} already exists at {self.path}")

        # maybe the .zip file has been downloaded previously but not unzipped
        unzip(self.path, verbosity=verbosity, overwrite=overwrite)

        if netCDF4 is None:
            to_netcdf = False

        # if to_netcdf:
        self._maybe_to_netcdf()

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(
            self.path,
            "02_location_boundary_area",
            "02_location_boundary_area",
            "shp",
            "CAMELS_AUS_Boundaries_adopted.shp" if self.version == 1 else "CAMELS_AUS_v2_Boundaries_adopted.shp"
        )

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'catchment_area': catchment_area(),
                'maen_slope_pct': slope('%'),
                'lat_outlet': gauge_latitude(),
                'long_outlet': gauge_longitude(),
        }

    @property
    def dyn_map(self):
        # table 2 in https://essd.copernicus.org/articles/13/3847/2021/#&gid=1&pid=1
        return {
            'streamflow_MLd': observed_streamflow_cms(),
            'streamflow_mmd': observed_streamflow_mm(),
            'tmin_SILO': min_air_temp_with_specifier('silo'),
            'tmax_SILO': max_air_temp_with_specifier('silo'),
            'tmin_AWAP': min_air_temp_with_specifier('awap'),
            'tmax_AWAP': max_air_temp_with_specifier('awap'),
            'et_morton_actual_SILO': actual_evapotranspiration_with_specifier('silo_morton'),
            'et_morton_point_SILO': actual_evapotranspiration_with_specifier('silo_morton_point'),
            'et_short_crop_SILO': actual_evapotranspiration_with_specifier('silo_short_crop'),
            'et_tall_crop_SILO': actual_evapotranspiration_with_specifier('silo_tall_crop'),
            'precipitation_AWAP': total_precipitation_with_specifier('awap'),
            'precipitation_SILO': total_precipitation_with_specifier('silo'),
            'solarrad_AWAP': solar_radiation_with_specifier('awap'),  # MJ/m2/day -> W/m2 in dyn_factors
            'radiation_SILO': solar_radiation_with_specifier('silo'),  # MJ/m2/day -> W/m2 in dyn_factors
            'vp_SILO': mean_vapor_pressure_with_specifier('silo'),
            'vprp_AWAP': mean_vapor_pressure_with_specifier('awap'),
            'rh_tmax_SILO': mean_rel_hum_with_specifier('silo_tmax'),
            'rh_tmin_SILO': mean_rel_hum_with_specifier('silo_tmin'),
            'vapourpres_h09_AGCD': mean_vapor_pressure_with_specifier('agcd_h09'),
            'vapourpres_h15_AGCD': mean_vapor_pressure_with_specifier('agcd_h15'),
            'tmin_AGCD': min_air_temp_with_specifier('agcd'),
            'tmax_AGCD': max_air_temp_with_specifier('agcd'),
            'precipitation_AGCD': total_precipitation_with_specifier('agcd'),
            #'mslp_SILO': mean_sea_level_pressure_with_specifier('silo'),
        }

    @property
    def dyn_factors(self):
        # Solar radiation is distributed as MJ m-2 day-1 ("Solar radiation
        # MJ m-2" in the v2 Data Description, Table `Other variables`) while the
        # canonical name promises W m-2. The AWAP series exists in v1 only; the
        # key is harmless for v2 because absent columns are skipped.
        return {
            observed_streamflow_cms(): 0.01157,
            solar_radiation_with_specifier('silo'): MJ_M2_DAY_TO_WM2,
            solar_radiation_with_specifier('awap'): MJ_M2_DAY_TO_WM2,
        }

    @property
    def dyn_generators(self):
        if self.version == 1:
            return {
    # new column to be created : function to be applied, inputs
    mean_air_temp_with_specifier('silo'): (self.mean_temp, (min_air_temp_with_specifier('silo'), max_air_temp_with_specifier('silo'))), 
    mean_air_temp_with_specifier('awap'): (self.mean_temp, (min_air_temp_with_specifier('awap'), max_air_temp_with_specifier('awap'))),
        }
        else:
            return {
    # new column to be created : function to be applied, inputs
    mean_air_temp_with_specifier('silo'): (self.mean_temp, (min_air_temp_with_specifier('silo'), max_air_temp_with_specifier('silo'))),
    mean_air_temp_with_specifier('agcd'): (self.mean_temp, (min_air_temp_with_specifier('agcd'), max_air_temp_with_specifier('agcd'))),
        }

    @property
    def start(self):
        return "19500101"

    @property
    def end(self):
        return "20181231" if self.version == 1 else "20220331"

    @property
    def location(self):
        return "Australia"

    def stations(self, as_list=True) -> list:
        fname = os.path.join(self.path, f"01_id_name_metadata{SEP}01_id_name_metadata{SEP}id_name_metadata.csv")
        df = pd.read_csv(fname)
        if as_list:
            return df['station_id'].to_list()
        else:
            return df

    @property
    def static_attribute_categories(self):
        features = []
        path = os.path.join(self.path, f'04_attributes{SEP}04_attributes')
        for f in os.listdir(path):
            if os.path.isfile(os.path.join(path, f)) and f.endswith('csv'):
                f = str(f.split('.csv')[0])
                features.append(''.join(f.split('_')[2:]))
        return features

    @property
    def static_features(self) -> List[str]:

        return self._static_data().columns.tolist()

    @property
    def dynamic_features(self) -> list:
        return [self.dyn_map.get(feat, feat) for feat in list(self.folders[self.version].keys())] + list(self.dyn_generators.keys())

    def _static_data(self, #stations, #features,
                     st=None, en=None):

        #features = validate_attributes(features, self.static_features, 'static_features')
        static_fname = 'CAMELS_AUS_Attributes&Indices_MasterTable.csv'
        static_fpath = os.path.join(self.path, static_fname)
        static_df = pd.read_csv(static_fpath, index_col='station_id')

        static_df.index = static_df.index.astype(str)
        #static_df = static_df.loc[stations]#[features]
        if isinstance(static_df, pd.Series):
            static_df = pd.DataFrame(static_df).transpose()
        
        static_df.rename(columns=self.static_map, inplace=True)

        return static_df

    def _read_dynamic(
            self, 
            stations, 
            dynamic_features, 
            st=None,
            en=None,
            ):

        st, en = self._check_length(st, en)
        dynamic_features = validate_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')

        dyn_attrs = {}
        dyn = {}
        dtype = {stn: np.float32 for stn in stations}
        dtype.update({'year': str, 'month': str, 'day': str})

        for _attr in list(self.folders[self.version].keys()):

            if self.dyn_map.get(_attr, _attr) not in dynamic_features:
                continue

            _path = os.path.join(self.path, f'{self.folders[self.version][_attr]}{SEP}{_attr}.csv')
            
            attr_df = pd.read_csv(_path, na_values=['-99.99'], usecols=['year', 'month', 'day'] + stations,
                                  dtype=dtype)
            attr_df.index = pd.to_datetime(attr_df[['year', 'month', 'day']])

            dyn_attrs[_attr] = attr_df[stations]

        # making one separate dataframe for one station
        for idx, stn in enumerate(stations):
            stn_df = pd.DataFrame()
            for attr, attr_df in dyn_attrs.items():
                # if attr in dynamic_features:
                stn_df[attr] = attr_df[stn]

            stn_df.rename(columns=self.dyn_map, inplace=True)

            self._apply_dyn_factors(stn_df)

            for new_col, (func, old_col) in self.dyn_generators.items():
                if isinstance(old_col, str):
                    if old_col in stn_df.columns:
                        # name of Series to func should be same as station id
                        stn_df[new_col] = func(pd.Series(stn_df[old_col], name=stn))
                else:
                    assert isinstance(old_col, tuple)
                    if all([col in stn_df.columns for col in old_col]):
                        # feed all old_cols to the function
                        stn_df[new_col] = func(*[pd.Series(stn_df[col], name=stn) for col in old_col])
            
            if self.verbosity>1 and idx % 100 == 0:
                print(f"processed {idx} stations")

            stn_df.index.name = 'time'
            stn_df.columns.name = 'dynamic_features'
            dyn[stn] = stn_df.loc[st:en, dynamic_features]

        return dyn


_CL_PANGAEA_URL = "https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/"

# the 2022 archive extracts into a folder holding another folder of the same name
_CL_2022_DIR = os.path.join("CAMELS_CL_v202201", "CAMELS_CL_v202201")


def _read_camels_cl_ts(
        fpath: str,
        stations: List[str],
        dtype,
        float_precision: str,
        sep: str,
        date_col: str,
        na_values,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads the columns of ``stations`` from one CAMELS-CL time series file, whose
    rows are days and columns are gauges. Returns the dates and a
    (days, stations) array. A module-level function, so that a process pool
    does not have to pickle the dataset.
    """
    df = pd.read_csv(fpath, sep=sep, usecols=[date_col] + stations, na_values=na_values,
                     dtype={stn: dtype for stn in stations}, float_precision=float_precision)
    return _checked_dates(df[date_col], fpath), df[stations].to_numpy()


def _checked_dates(dates, fpath: str) -> pd.DatetimeIndex:
    """the dates of a CAMELS-CL time series file, which must be sorted, because
    CAMELS_CL.start/end are read from the first and last rows only, and unique,
    because the values of a repeated date would overwrite each other"""
    dates = pd.DatetimeIndex(pd.to_datetime(dates, format="%Y-%m-%d"))
    if not (dates.is_monotonic_increasing and dates.is_unique):
        raise ValueError(f"dates in {fpath} are not sorted or not unique")
    return dates


def _row_date(row: bytes, sep: bytes) -> str:
    """the date in the first field of a row of a CAMELS-CL time series file"""
    return row[:row.find(sep)].strip(b' "').decode()


def _read_camels_cl_dates(fpath: str, sep: bytes) -> pd.DatetimeIndex:
    """dates in the first column of a CAMELS-CL time series file, without parsing its values"""
    with open(fpath, 'rb') as f:
        f.readline()  # header
        return _checked_dates([_row_date(row, sep) for row in f if row.strip()], fpath)


def _first_and_last_row(fpath: str) -> Tuple[bytes, bytes]:
    """first row after the header and last row of a text file"""
    with open(fpath, 'rb') as f:
        f.readline()  # header
        first = f.readline()
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 2**16))  # a row is at most a few kB long
        last = [row for row in f.read().splitlines() if row.strip()][-1]
    return first, last


class CAMELS_CL(_RainfallRunoff):
    """
    Daily data of 516 Chilean catchments following
    `Alvarez-Garreton et al., 2018 <https://doi.org/10.5194/hess-22-5817-2018>`_ .
    Two releases are available through ``version``:

    - ``2022`` (default): the January 2022 release from
      `cr2.cl <https://www.cr2.cl/datos-informacion-integrada-por-cuencas/>`_ ,
      with 10 dynamic and 110 static features on a time index from 1900-01-01
      to 2021-06-22.
    - ``2018``: the August 2018 release from
      `PANGAEA <https://store.pangaea.de/Publications/Alvarez-Garreton-etal_2018/>`_ ,
      with 12 dynamic and 104 static features from 1913-02-15 to 2018-03-09.

    The time index is the union of the dates in the files, and each feature is
    NaN outside its record. In 2022, observed streamflow runs from 1913-02-15 to
    2020-06-06, CR2MET precipitation, air temperature and Hargreaves PET from
    1979-01-01 to 2020-04-30, MSWEP to 2019-12-31, CHIRPS from 1981-01-01 to
    2019-12-31 and TMPA from 1998-01-01 to 2018-12-31.

    Compared with 2018, the 2022 release extends the records, uses newer CR2MET
    and MSWEP versions, redraws 27 catchment boundaries and corrects 570 daily
    flows of 1000 m3/s or more (15 gauges, 2016-2018) that 2018 stores as 1 to 4.
    Snow water equivalent (``swe``, mm) and MODIS PET (``pet_mm_modis``, 8-day
    totals in mm) exist only in 2018.

    Units: ``q_cms_obs`` m3/s; ``q_mm_obs``, ``pcp_mm_*`` and
    ``pet_mm_hargreaves`` mm/day; ``airtemp_C_*`` degree Celsius. Mean slope is
    ``slope_mkm-1`` (m/km) in 2018 and ``slope_%`` in 2022. Dynamic values are
    cast to ``float_precision`` (float32 by default, relative error below 1e-7).
    Not provided: the monthly aggregates, the yearly water-rights series and the
    water-rights register of the 2022 archive, and the 2018 catchment hierarchy.

    Timings for 2022 on a 48-core machine: the first initialization downloads
    288 MB, then extracts it and builds a 918 MB netCDF cache in about 6 s (2 GB
    of disk in total). Afterwards all 516 stations with all features are fetched
    in 0.7 s from the cache (2 s from the csv files) and one station in 0.1 s
    (0.6 s). The csv files are read in parallel processes for 100 or more
    stations or with ``float_precision=np.float64``, so on Windows and macOS run
    such scripts, including the first initialization, under
    ``if __name__ == "__main__":``.

    Examples
    ---------
    >>> from aqua_fetch import CAMELS_CL
    >>> dataset = CAMELS_CL()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='8350001', as_dataframe=True)
    >>> df = dynamic['8350001'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (44368, 10)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
    516
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (51 out of 516)
    51
    ...
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ['q_cms_obs', 'q_mm_obs', 'pcp_mm_cr2met', 'pcp_mm_chirps', 'pcp_mm_mswep', 'pcp_mm_tmpa', 'airtemp_C_min', 'airtemp_C_max', 'airtemp_C_mean', 'pet_mm_hargreaves']
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('8350001', as_dataframe=True,
    ...  dynamic_features=['pet_mm_hargreaves', 'pcp_mm_mswep', 'airtemp_C_mean', 'q_cms_obs'])
    >>> dynamic['8350001'].shape
    (44368, 4)
    ...
    ... # get data of a selected period
    >>> _, dynamic = dataset.fetch('8350001', st='2000-01-01', en='2000-12-31', as_dataframe=True)
    >>> dynamic['8350001'].shape
    (366, 10)
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='8350001', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['8350001'].shape
    ((1, 110), 1, (44368, 10))
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
    (516, 2)
    >>> dataset.stn_coords('8350001')  # returns coordinates of station whose id is 8350001
                    lat     long
    gauge_id
    8350001  -38.214199 -71.8283
    ...
    # get area (km2) of two stations
    >>> dataset.area(['8350001', '3820003'])
    gauge_id
    8350001      46.104347
    3820003    7383.890625
    Name: area_km2, dtype: float32
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('8350001')
    ...
    # the 2018 release
    >>> dataset = CAMELS_CL(version=2018)
    >>> _, dynamic = dataset.fetch(stations='8350001', as_dataframe=True)
    >>> dynamic['8350001'].shape
    (38374, 12)
    """

    # todo : Duplicated code: CAMELS_GB, CAMELS_KR and CAMELS_CL each have their own "extract to a temporary folder"

    # archives of each release: file name -> url
    urls = {
        2018: {f"{stem}.zip": f"{_CL_PANGAEA_URL}{stem}.zip" for stem in (
            "1_CAMELScl_attributes", "2_CAMELScl_streamflow_m3s", "3_CAMELScl_streamflow_mm",
            "4_CAMELScl_precip_cr2met", "5_CAMELScl_precip_chirps", "6_CAMELScl_precip_mswep",
            "7_CAMELScl_precip_tmpa", "8_CAMELScl_tmin_cr2met", "9_CAMELScl_tmax_cr2met",
            "10_CAMELScl_tmean_cr2met", "11_CAMELScl_pet_8d_modis", "12_CAMELScl_pet_hargreaves",
            "13_CAMELScl_swe", "14_CAMELScl_catch_hierarchy", "CAMELScl_catchment_boundaries",
        )},
        2022: {"CAMELS_CL_v202201.zip": "https://www.cr2.cl/download/camels-cl-v202201/?wpdmdl=35317"},
    }

    # daily time series of each release: raw feature name -> file, relative to ``path``
    _dyn_files = {
        2018: {feature: os.path.join(stem, f"{stem}.txt") for feature, stem in (
            ('streamflow_m3s', '2_CAMELScl_streamflow_m3s'),
            ('streamflow_mm', '3_CAMELScl_streamflow_mm'),
            ('precip_cr2met', '4_CAMELScl_precip_cr2met'),
            ('precip_chirps', '5_CAMELScl_precip_chirps'),
            ('precip_mswep', '6_CAMELScl_precip_mswep'),
            ('precip_tmpa', '7_CAMELScl_precip_tmpa'),
            ('tmin_cr2met', '8_CAMELScl_tmin_cr2met'),
            ('tmax_cr2met', '9_CAMELScl_tmax_cr2met'),
            ('tmean_cr2met', '10_CAMELScl_tmean_cr2met'),
            ('pet_8d_modis', '11_CAMELScl_pet_8d_modis'),
            ('pet_hargreaves', '12_CAMELScl_pet_hargreaves'),
            ('swe', '13_CAMELScl_swe'),
        )},
        2022: {feature: os.path.join(_CL_2022_DIR, fname) for feature, fname in (
            ('streamflow_m3s', 'q_m3s_day.csv'),
            ('streamflow_mm', 'q_mm_day.csv'),
            ('precip_cr2met', 'precip_cr2met_mm_day.csv'),
            ('precip_chirps', 'precip_chirps_mm_day.csv'),
            ('precip_mswep', 'precip_mswep_mm_day.csv'),
            ('precip_tmpa', 'precip_tmpa_mm_day.csv'),
            ('tmin_cr2met', 'tmin_cr2met_C_day.csv'),
            ('tmax_cr2met', 'tmax_cr2met_C_day.csv'),
            ('tmean_cr2met', 'tmean_cr2met_C_day.csv'),
            ('pet_hargreaves', 'pet_hargreaves_mm_day.csv'),
        )},
    }

    def __init__(
            self,
            path: str = None,
            version: int = 2022,
            overwrite: bool = False,
            to_netcdf: bool = True,
            verbosity: int = 1,
            **kwargs
    ):
        """
        Parameters
        ----------
        path : str
            directory under which the data is (or will be) saved in a
            ``CAMELS_CL`` folder. Both releases can share it. If None, the
            default data directory of aqua_fetch is used.
        version : int
            ``2022`` (default) or ``2018``.
        overwrite : bool
            if True, the archives, extracted files and netCDF caches (of all
            precisions) of this ``version`` are deleted and downloaded again.
        to_netcdf : bool
            whether to save the dynamic data of this ``version`` in a netCDF
            cache for faster reading. Requires netCDF4 and xarray.
        verbosity : int
            0 prints nothing.
        **kwargs :
            any keyword argument of :py:class:`aqua_fetch.rr._RainfallRunoff`
            such as ``processes``.
        """
        # isinstance, because 2022.0 == 2022 would name a second cache camels_cl_D_2022.0_v2.nc
        if not isinstance(version, (int, np.integer)) or version not in self.urls:
            raise ValueError(f"version must be one of {list(self.urls)} but is {version!r}")
        self.version = int(version)
        self._all_dates = None  # see _dates_of_all_files

        super().__init__(path=path, overwrite=overwrite, to_netcdf=to_netcdf,
                         verbosity=verbosity, **kwargs)

        self._download_camels_cl(overwrite)

        self._check_manifest()

        self._check_duplicates()

        self._maybe_to_netcdf()

    def _download_camels_cl(self, overwrite: bool = False):
        """
        Downloads and extracts the archives of ``self.version`` whose extracted
        folders do not exist, so an archive deleted after extraction is not
        downloaded again. ``overwrite=True`` first deletes this version's
        archives, extracted folders and netCDF caches.
        """
        archives = {os.path.join(self.path, fname): url for fname, url in self.urls[self.version].items()}
        folders = [archive[:-len(".zip")] for archive in archives]

        if overwrite:
            caches = glob.glob(os.path.join(glob.escape(self.path),
                                            f"{self.name.lower()}_{self.timestep}_{self.version}*.nc"))
            _remove_stale([*archives, *folders, *caches], self.verbosity)

        os.makedirs(self.path, exist_ok=True)
        for (archive, url), folder in zip(archives.items(), folders):
            if os.path.exists(folder):
                continue
            if not os.path.exists(archive):
                if self.verbosity:
                    print(f"downloading {url} to {archive}")
                # cr2.cl answers the default user agent of urllib with 403 Forbidden
                download(url, outdir=self.path, fname=os.path.basename(archive),
                         verbosity=self.verbosity,
                         headers=BROWSER_HEADERS if self.version == 2022 else None)

            # into a temporary folder that is renamed once complete, so that an
            # interrupted extraction is redone instead of being taken as complete
            if self.verbosity:
                print(f"extracting {archive}")
            partial = f"{folder}_extracting"
            shutil.rmtree(partial, ignore_errors=True)
            try:
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(partial)
            except (zipfile.BadZipFile, zlib.error, EOFError):  # e.g. an error page saved as the archive
                shutil.rmtree(partial, ignore_errors=True)
                os.remove(archive)
                raise ValueError(f"{archive} is corrupt and was deleted. "
                                 f"Initialize CAMELS_CL again to download it again.") from None
            os.replace(partial, folder)

        if self.remove_zip:
            for archive in archives:
                if os.path.exists(archive):
                    if self.verbosity:
                        print(f"remove_zip=True: removing {archive}")
                    os.remove(archive)
        return

    def _check_manifest(self):
        """warns if files of ``self.version`` are missing, e.g. deleted by hand"""
        shapefile = [self.boundary_file[:-len(".shp")] + ext for ext in (".shp", ".shx", ".dbf", ".prj")]
        files = [self._static_file, *shapefile, *self._dyn_paths.values()]
        missing = [fpath for fpath in files if not os.path.exists(fpath)]
        if missing:
            warnings.warn(
                f"CAMELS_CL {self.version}: {len(missing)} of {len(files)} files are missing: "
                f"{missing}. Use overwrite=True to download them again.", UserWarning)
        return

    def _check_duplicates(self):
        """warns if two gauges share a name and rounded coordinates"""
        meta = self._attributes[['gauge_name', gauge_latitude(), gauge_longitude()]].reset_index()
        meta.columns = ['gauge_id', 'gauge_name', 'gauge_lat', 'gauge_lon']
        _warn_duplicate_gauges(self.name, meta)
        return

    @property
    def _static_file(self) -> os.PathLike:
        if self.version == 2018:
            return os.path.join(self.path, "1_CAMELScl_attributes", "1_CAMELScl_attributes.txt")
        return os.path.join(self.path, _CL_2022_DIR, "catchment_attributes.csv")

    @property
    def boundary_file(self) -> os.PathLike:
        if self.version == 2018:
            return os.path.join(self.path, "CAMELScl_catchment_boundaries",
                                "CAMELScl_catchment_boundaries", "catchments_camels_cl_v1.3.shp")
        return os.path.join(self.path, _CL_2022_DIR, "camels_cl_boundaries", "camels_cl_boundaries.shp")

    @property
    def boundary_id_map(self) -> str:
        return "gauge_id"

    @property
    def _dyn_paths(self) -> Dict[str, os.PathLike]:
        """standardized name of each dynamic feature -> path of its file"""
        return {self.dyn_map.get(raw, raw): os.path.join(self.path, fname)
                for raw, fname in self._dyn_files[self.version].items()}

    @property
    def _ts_format(self) -> Dict:
        """how the time series files of ``self.version`` are written"""
        if self.version == 2018:
            return dict(sep='\t', date_col='gauge_id', na_values=[' '])
        return dict(sep=',', date_col='date', na_values=None)

    @property
    def dyn_fname(self) -> Union[str, os.PathLike]:
        """name of the netCDF cache of the dynamic data, one per ``version`` and
        precision, e.g. camels_cl_D_2022_v2.nc or camels_cl_D_2022_float64_v2.nc"""
        precision = "" if np.dtype(self.fp) == np.float32 else f"_{np.dtype(self.fp).name}"
        return cache_name(f"{self.name.lower()}_{self.timestep}_{self.version}{precision}.nc")

    @property
    def static_map(self) -> Dict[str, str]:
        if self.version == 2018:
            return {
                'area': catchment_area(),                         # km2
                'slope_mean': slope('mkm-1'),                     # m/km
                'gauge_lat': gauge_latitude(),
                'gauge_lon': gauge_longitude(),
            }
        # area_km2 already has the standardized name
        return {
            'mean_slope_perc': slope('%'),                        # % (the 2018 m/km value / 10)
            'mean_elev': catchment_elevation_meters(),            # m a.s.l.
            'med_elev': med_catchment_elevation_meters(),
            'min_elev': min_catchment_elevation_meters(),
            'max_elev': max_catchment_elevation_meters(),
            'gauge_lat': gauge_latitude(),
            'gauge_lon': gauge_longitude(),
        }

    @property
    def dyn_map(self) -> Dict[str, str]:
        # swe (mm) keeps its raw name; pet_8d_modis is mm per 8 days
        return {
            'streamflow_m3s': observed_streamflow_cms(),
            'streamflow_mm': observed_streamflow_mm(),
            'tmin_cr2met': min_air_temp(),
            'tmax_cr2met': max_air_temp(),
            'tmean_cr2met': mean_air_temp(),
            'precip_mswep': total_precipitation_with_specifier('mswep'),
            'precip_tmpa': total_precipitation_with_specifier('tmpa'),
            'precip_cr2met': total_precipitation_with_specifier('cr2met'),
            'precip_chirps': total_precipitation_with_specifier('chirps'),
            'pet_hargreaves': total_potential_evapotranspiration_with_specifier('hargreaves'),
            'pet_8d_modis': total_potential_evapotranspiration_with_specifier('modis'),
        }

    @functools.cached_property
    def _time_extent(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """first and last date over the time series files of ``self.version``.
        A missing file raises, rather than silently narrowing the dates."""
        sep = self._ts_format['sep'].encode()
        dates = [pd.Timestamp(_row_date(row, sep)) for fpath in self._dyn_paths.values()
                 for row in _first_and_last_row(fpath)]
        return min(dates), max(dates)

    def _dates_of_all_files(self, known: Dict[str, pd.DatetimeIndex]) -> pd.DatetimeIndex:
        """union of the dates in the time series files of ``self.version``, found
        once; the dates of the files in ``known`` are not read again"""
        if self._all_dates is None:
            sep = self._ts_format['sep'].encode()
            dates = [known[fpath] if fpath in known else _read_camels_cl_dates(fpath, sep)
                     for fpath in self._dyn_paths.values()]
            self._all_dates = functools.reduce(pd.DatetimeIndex.union, dates).rename('time')
        return self._all_dates

    @property
    def start(self) -> pd.Timestamp:
        return self._time_extent[0]

    @property
    def end(self) -> pd.Timestamp:
        return self._time_extent[1]

    @property
    def location(self):
        return "Chile"

    @functools.cached_property
    def _attributes(self) -> pd.DataFrame:
        """catchment attributes of all gauges with gauge_id as index, read once"""
        if self.version == 2018:
            # gauges are columns, and ids and numbers are padded with spaces
            df = pd.read_csv(self._static_file, sep='\t', index_col='gauge_id', dtype=str).T
            df.index = df.index.str.strip()
            df.columns.name = None
            for col in df.columns:
                text = df[col].str.strip()
                numbers = pd.to_numeric(text, errors='coerce')
                if numbers.notna().sum() == text.notna().sum():  # every value is a number
                    # float() reads e.g. 0.00000000000000000003 exactly, to_numeric does not
                    df[col] = numbers if numbers.dtype.kind == 'i' else text.astype(float)
        else:
            df = pd.read_csv(self._static_file, index_col='gauge_id', dtype={'gauge_id': str},
                             float_precision='round_trip')
        df.index.name = 'gauge_id'
        return df.rename(columns=self.static_map)

    def _static_data(self) -> pd.DataFrame:
        return self._attributes.copy()

    def stations(self) -> List[str]:
        """ids of the 516 gauges, as listed in the catchment attributes table"""
        return self._attributes.index.tolist()

    @property
    def static_features(self) -> List[str]:
        return self._attributes.columns.tolist()

    @property
    def dynamic_features(self) -> List[str]:
        return list(self._dyn_paths)

    @property
    def _mm_feature_name(self) -> str:
        return observed_streamflow_mm()

    def _read_stn_dyn(self, stn: str) -> pd.DataFrame:
        return self._read_dynamic([stn], 'all')[stn]

    def _read_dynamic(
            self,
            stations,
            dynamic_features,
            st: Union[str, pd.Timestamp] = None,
            en: Union[str, pd.Timestamp] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Reads each file of ``dynamic_features`` once for all ``stations``, in
        parallel when the files are large. Returns one DataFrame per station on
        the dates of all the files of ``self.version`` between ``st`` and ``en``.
        """
        st, en = self._check_length(st, en)
        stations = validate_attributes(stations, self.stations(), 'stations')
        features = validate_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')

        # pandas' default parser reads e.g. 0.0000000000000000016 as 0. 'round_trip'
        # is exact but holds the GIL. 'legacy' releases it and, cast to float32,
        # gives the same values as 'round_trip' for every value of both releases.
        exact = np.dtype(self.fp).itemsize > 4
        paths = [self._dyn_paths[feature] for feature in features]
        read = functools.partial(_read_camels_cl_ts, stations=stations, dtype=self.fp,
                                 float_precision='round_trip' if exact else 'legacy', **self._ts_format)

        workers = n_workers(sum(map(os.path.getsize, paths)), len(paths), self.processes)
        if workers == 1:
            results = [read(fpath) for fpath in paths]
        else:
            # Threads need no ``if __name__ == "__main__"`` guard on Windows or
            # macOS and are as fast as processes for a few stations; processes
            # are faster for many stations and for the GIL-holding 'round_trip'.
            pool = cf.ProcessPoolExecutor if len(stations) >= 100 or exact else cf.ThreadPoolExecutor
            with pool(workers) as executor:
                results = list(executor.map(read, paths))

        # the same dates whichever features are read, as in the netCDF cache
        time = self._dates_of_all_files(dict(zip(paths, (dates for dates, _ in results))))
        time = time[(time >= st) & (time <= en)]

        # (stations, time, features), so that each station is a contiguous block. The
        # frames share this array, which is 3 times faster to fill than one per station.
        data = np.full((len(stations), len(time), len(features)), np.nan, dtype=self.fp)
        for k, (dates, values) in enumerate(results):
            rows = time.get_indexer(dates)
            inside = rows >= 0
            data[:, rows[inside], k] = values[inside].T

        columns = pd.Index(features, name='dynamic_features')
        return {stn: pd.DataFrame(data[j], index=time, columns=columns)
                for j, stn in enumerate(stations)}


class CAMELS_CH(_RainfallRunoff):
    """
    Data of 331 Swiss catchments from
    `Hoege et al., 2023 <https://doi.org/10.5194/essd-15-5755-2023>`_ .
    The dataset consists of 209 static catchment features and 9 dynamic features.
    The dynamic features span from 19810101 to 20201231 with daily timestep.
    For daily (``D``) ``timestep``, only streamflow is available for 170 swiss catchments.
    The hourly (``H``) streamflow data is obtained from `Kauzlaric et al., 2023 <https://zenodo.org/records/7691294>`_ .

    Examples
    ---------
    >>> from aqua_fetch import CAMELS_CH
    >>> dataset = CAMELS_CH()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='2004', as_dataframe=True)
    >>> df = dynamic['2004'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (14610, 9)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       331
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (33 out of 331)
       33
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(14610, 9), (14610, 9), (14610, 9),... (14610, 9), (14610, 9)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('2004', as_dataframe=True,
    ...  dynamic_features=['pcp_mm', 'airtemp_C_mean', 'q_cms_obs'])
    >>> dynamic['2004'].shape
       (14610, 3)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='2004', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['2004'].shape
    ((1, 209), 1, (14610, 9))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 14610, 'dynamic_features': 9})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns WGS84 coordinates of all stations
    >>> coords.shape
        (331, 2)
    >>> dataset.stn_coords('2004')  # returns coordinates of station whose id is 2004
                    lat      long
        gauge_id
        2004      46.930752  7.116924
    >>> dataset.stn_coords(['2004', '2007'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('2004')
    # get coordinates of two stations
    >>> dataset.area(['2004', '2007'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('2004')

    """
    url = {
        'camels_ch.zip': "https://zenodo.org/record/7957061",
        'DischargeDBHydroCH.zip': 'https://zenodo.org/records/7691294'
    }

    def __init__(
            self,
            path=None,
            overwrite: bool = False,
            to_netcdf: bool = True,
            timestep: str = 'D',
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
        overwrite : bool
            If the data is already down then you can set it to True,
            to make a fresh download.
        to_netcdf : bool
            whether to convert all the data into one netcdf file or not.
            This will fasten repeated calls to fetch etc. but will
            require netCDF4 package as well as xarry.
        """
        super().__init__(path=path, **kwargs)

        self.timestep = timestep

        if timestep == 'D' and 'DischargeDBHydroCH.zip' in self.url:
            self.url.pop('DischargeDBHydroCH.zip')

        self._download(overwrite=overwrite)

        self._dynamic_features = self._read_stn_dyn(self.stations()[0]).columns.tolist()

        # if to_netcdf:
        self._maybe_to_netcdf()

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(
            self.path,
            'camels_ch',
            'camels_ch',
            'catchment_delineations',
            'CAMELS_CH_catchments.shp'
        )

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'area': catchment_area(),
                'slope_mean': slope('degrees'),
                'gauge_lat': gauge_latitude(),
                'gauge_lon': gauge_longitude(),
        }

    @property
    def dyn_map(self):
        # table 1 in https://essd.copernicus.org/articles/15/5755/2023/
        return {
            'discharge_vol(m3/s)': observed_streamflow_cms(),
            # 'discharge_vol(m3/s)': 'sim_q_cms',
            'discharge_spec(mm/d)': observed_streamflow_mm(),
            'temperature_min(°C)': min_air_temp(),
            'temperature_max(°C)': max_air_temp(),
            'temperature_mean(°C)': mean_air_temp(),
            'precipitation(mm/d)': total_precipitation(),
            'swe(mm)': snow_water_equivalent(),
        }

    @property
    def camels_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.path, 'camels_ch', 'camels_ch')

    @property
    def static_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.camels_path, 'static_attributes')

    @property
    def dynamic_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.camels_path, 'time_series', 'observation_based')

    @property
    def glacier_attr_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_glacier_attributes.csv')

    @property
    def clim_attr_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_climate_attributes_obs.csv')

    @property
    def geol_attr_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_geology_attributes.csv')

    @property
    def supp_geol_attr_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_geology_attributes_supplement.csv')

    @property
    def hum_inf_attr_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_humaninfluence_attributes.csv')

    @property
    def hydrogeol_attr_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_hydrogeology_attributes.csv')

    @property
    def hydrol_attr_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_hydrology_attributes_obs.csv')

    @property
    def lc_attr_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_landcover_attributes.csv')

    @property
    def soil_attr_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_soil_attributes.csv')

    @property
    def topo_attr_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.static_path, 'CAMELS_CH_topographic_attributes.csv')

    @property
    def static_features(self):
        return self._static_data().columns.tolist()

    @property
    def dynamic_features(self) -> List[str]:
        return self._dynamic_features

    def all_hourly_stations(self) -> List[str]:
        """Names of all stations which have hourly data"""
        return pd.read_excel(
            os.path.join(self.path, 'Inventory_discharge_hydroCH.xlsx'), dtype={'ID': str}
        )['ID'].values.tolist()

    def hourly_stations(self) -> List[str]:
        """
        IDs of those stations which have hourly data and which are also part of
        CAMELS-CH dataset
        """
        return [stn for stn in self.all_hourly_stations() if stn in self.stations()]

    @property
    def start(self):  # start of data
        return pd.Timestamp('1981-01-01')

    @property
    def end(self):  # end of data
        return pd.Timestamp('2020-12-31')

    def stations(self) -> List[str]:
        """Returns station ids for catchments"""
        stns = pd.read_csv(
            self.glacier_attr_path,
            sep=';',
            skiprows=1
        )['gauge_id'].values.tolist()
        return [str(stn) for stn in stns]

    @property
    def foen_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.path, 'DischargeDBHydroCH', 'DischargeDBHydroCH', 'CH', 'FOEN')

    def foen_stations(self) -> List[str]:
        """Returns all the stations in the FOEN folder"""
        return os.listdir(self.foen_path)

    def read_hourly_q_ch(self, stn: str) -> pd.DataFrame:
        stn = f"Q_{stn}_hourly.asc"
        fname = [fname for fname in self.foen_stations() if stn in fname][0]
        fpath = os.path.join(self.foen_path, fname)

        q = pd.read_csv(fpath,
                        sep="\t",
                        parse_dates=[['YYYY', 'MM', 'DD', 'HH']],
                        index_col='YYYY_MM_DD_HH',
                        )
        q.index = pd.to_datetime(q.index)
        q.columns = ['q_cms']
        q.index.name = "time"
        return q

    def glacier_attrs(self) -> pd.DataFrame:
        """
        returns a dataframe with four columns
            - 'glac_area'
            - 'glac_vol'
            - 'glac_mass'
            - 'glac_area_neighbours'
        """
        df = pd.read_csv(
            self.glacier_attr_path,
            sep=';',
            skiprows=1,
            index_col='gauge_id',
            dtype=np.float32
        )
        df.index = df.index.astype(int).astype(str)
        return df

    def climate_attrs(self) -> pd.DataFrame:
        """returns 14 climate attributes of catchments.
        """
        df = pd.read_csv(
            self.clim_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id',
            dtype={
                'gauge_id': str,
                'p_mean': float,
                'aridity': float,
                'pet_mean': float,
                'p_seasonality': float,
                'frac_snow': float,
                'high_prec_freq': float,
                'high_prec_dur': float,
                'high_prec_timing': str,
                'low_prec_timing': str
            }
        )
        return df

    def geol_attrs(self) -> pd.DataFrame:
        """15 geological features"""
        df = pd.read_csv(
            self.geol_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id',
            dtype=np.float32
        )
        df.index = df.index.astype(int).astype(str)
        return df

    def supp_geol_attrs(self) -> pd.DataFrame:
        """supplimentary geological features"""
        df = pd.read_csv(
            self.supp_geol_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id',
            dtype=np.float32
        )

        df.index = df.index.astype(int).astype(str)
        return df

    def human_inf_attrs(self) -> pd.DataFrame:
        """
        14 athropogenic factors
        """
        df = pd.read_csv(
            self.hum_inf_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id',
            dtype={
                'gauge_id': str,
                'n_inhabitants': int,
                'dens_inhabitants': float,
                'hp_count': int,
                'hp_qturb': float,
                'hp_inst_turb': float,
                'hp_max_power': float,
                'num_reservoir': int,
                'reservoir_cap': float,
                'reservoir_he': float,
                'reservoir_fs': float,
                'reservoir_irr': float,
                'reservoir_nousedata': float,
                # 'reservoir_year_first': int,
                # 'reservoir_year_last': int
            }
        )
        return df

    def hydrogeol_attrs(self) -> pd.DataFrame:
        """10 hydrogeological factors"""
        df = pd.read_csv(
            self.hydrogeol_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id',
            dtype=float
        )
        df.index = df.index.astype(int).astype(str)
        return df

    def hydrol_attrs(self) -> pd.DataFrame:
        """14 hydrological parameters + 2 useful infos"""
        df = pd.read_csv(
            self.hydrol_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id',
            dtype={
                'gauge_id': str,
                'sign_number_of_years': int,
                'q_mean': float,
                'runoff_ratio': float, 'stream_elas': float, 'slope_fdc': float,
                'baseflow_index_landson': float,
                'hfd_mean': float,
                'Q5': float, 'Q95': float, 'high_q_freq': float, 'high_q_dur': float,
                'low_q_freq': float
            }
        )
        return df

    def landcolover_attrs(self) -> pd.DataFrame:
        """13 landcover parameters"""
        return pd.read_csv(
            self.lc_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id',
            dtype={
                'gauge_id': str,
                'crop_perc': float,
                'grass_perc': float,
                'scrub_perc': float,
                'dwood_perc': float,
                'mixed_wood_perc': float,
                'ewood_perc': float,
                'wetlands_perc': float,
                'inwater_perc': float,
                'ice_perc': float,
                'loose_rock_perc': float,
                'rock_perc': float,
                'urban_perc': float,
                'dom_land_cover': str
            }
        )

    def soil_attrs(self) -> pd.DataFrame:
        """80 soil parameters"""
        df = pd.read_csv(
            self.soil_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id'
        )
        df.index = df.index.astype(int).astype(str)
        return df

    def topo_attrs(self) -> pd.DataFrame:
        """topographic parameters"""
        df = pd.read_csv(
            self.topo_attr_path,
            skiprows=1,
            sep=';',
            index_col='gauge_id',
            encoding="unicode_escape"
        )

        df.index = df.index.astype(int).astype(str)
        return df

    def _static_data(self)->pd.DataFrame:
        df = pd.concat(
            [
                self.climate_attrs(),
                self.geol_attrs(),
                self.supp_geol_attrs(),
                self.glacier_attrs(),
                self.human_inf_attrs(),
                self.hydrogeol_attrs(),
                self.hydrol_attrs(),
                self.landcolover_attrs(),
                self.soil_attrs(),
                self.topo_attrs(),
            ],
            axis=1)
        df.index = df.index.astype(str)

        df.rename(columns=self.static_map, inplace=True)

        return df

    def _read_stn_dyn(self, station: str) -> pd.DataFrame:
        """
        Reads daily dynamic (meteorological + streamflow) data for one catchment
        and returns as DataFrame
        """

        df = pd.read_csv(
            os.path.join(self.dynamic_path, f"CAMELS_CH_obs_based_{station}.csv"),
            sep=';',
            index_col='date',
            parse_dates=True,
            dtype=np.float32
        )

        df.rename(columns=self.dyn_map, inplace=True)

        return df

    def stn_coords(
            self,
            stations: Union[str, List[str]] = 'all'
    ) -> pd.DataFrame:
        """
        Returns WGS84 (EPSG:4326) coordinates of stations as a DataFrame with
        ``lat`` and ``long`` columns.

        The coordinates are derived from the precise ``gauge_easting`` /
        ``gauge_northing`` columns (CH1903+ / LV95, EPSG:2056) and converted to
        WGS84 with the pyproj-free :func:`epsg2056_point_to_wgs84` helper. They
        are used in preference to the dataset's own ``gauge_lat`` / ``gauge_lon``
        columns, which are rounded to two decimals (~500 m). The conversion was
        verified against pyproj (EPSG:2056 -> EPSG:4326): the error is under
        3 m (mean 0.7 m) over all 331 stations.

        Parameters
        ----------
        stations :
            name/names of stations. If not given, coordinates of all stations
            will be returned.

        Returns
        -------
        pd.DataFrame
            with ``lat`` and ``long`` columns, indexed by station id.

        Examples
        --------
        >>> from aqua_fetch import CAMELS_CH
        >>> dataset = CAMELS_CH()
        >>> dataset.stn_coords('2004')
                        lat      long
            gauge_id
            2004      46.930752  7.116924
        """
        en = self.fetch_static_features(
            static_features=['gauge_easting', 'gauge_northing'])
        stations = validate_attributes(stations, self.stations(), 'stations')
        en = en.loc[stations, :].astype(float)
        lat, long = epsg2056_point_to_wgs84(
            en['gauge_easting'].values, en['gauge_northing'].values)
        return pd.DataFrame(
            {'lat': lat, 'long': long}, index=en.index).astype(self.fp)

    def transform_boundary(self, boundary):
        """
        Transform a catchment boundary from CH1903+ / LV95 (EPSG:2056, the CRS
        of the CAMELS-CH shapefile) to WGS84 (EPSG:4326) lon/lat.

        Uses the pyproj-free :func:`epsg2056_point_to_wgs84` helper (swisstopo's
        approximate LV95 -> WGS84 formula). Verified against pyproj
        (EPSG:2056 -> EPSG:4326): the per-vertex error is below 3 m across all
        331 catchments (~3.2 M vertices) - finer than the dataset's own
        two-decimal (~500 m) gauge coordinates. Polygons with interior rings
        (holes) and MultiPolygons are handled; the geometry type and ring
        structure are preserved. The conversion is vectorised per ring (one
        array call rather than one call per vertex).
        """
        if fiona is None:
            return boundary

        def _ring_to_wgs84(ring):
            arr = np.asarray(ring, dtype=float)
            # fiona stores each vertex as (x=easting, y=northing[, z]); output
            # is (lon, lat) to keep the (x, y) ordering of the geometry.
            lat, long = epsg2056_point_to_wgs84(arr[:, 0], arr[:, 1])
            return list(zip(long.tolist(), lat.tolist()))

        if boundary.type == 'MultiPolygon':
            coords = [[_ring_to_wgs84(ring) for ring in polygon]
                      for polygon in boundary.coordinates]
        else:  # Polygon, possibly with interior rings (holes)
            coords = [_ring_to_wgs84(ring) for ring in boundary.coordinates]

        return fiona.Geometry(type=boundary.type, coordinates=coords)


def _read_camels_de_dynamic(
        ts_dir: str,
        prefix: str,
        dyn_map_items: Tuple[Tuple[str, str], ...],
        dyn_feats: List[str],
        st: pd.Timestamp,
        en: pd.Timestamp,
        station: str,
) -> pd.DataFrame:
    """Read the dynamic timeseries of a single CAMELS-DE station.

    This is a module-level function (not a method) on purpose: it is dispatched
    to a :class:`concurrent.futures.ProcessPoolExecutor` by
    :meth:`CAMELS_DE._read_dynamic`. Handing a *bound* method to a process pool
    would pickle ``self`` — and hence every cached heavy attribute — to every
    worker on every task. Carrying only small, picklable arguments avoids that
    and roughly halves the wall-clock of a large hourly fetch. The window/feature
    slicing is done here (in the worker) so the parent does not repeat it
    serially over hundreds of stations.

    The result is byte-for-byte identical to
    ``CAMELS_DE._read_stn_dyn(station).loc[st:en, dyn_feats]`` with the axis
    names set — no values or dtypes are changed.
    """
    df = pd.read_csv(
        os.path.join(ts_dir, f"{prefix}_hydromet_timeseries_{station}.csv"),
        index_col='date',
        parse_dates=True,
    )
    df.rename(columns=dict(dyn_map_items), inplace=True)
    df = df.loc[st:en, list(dyn_feats)]
    df.columns.name = 'dynamic_features'
    df.index.name = 'time'
    return df


class CAMELS_DE(_RainfallRunoff):
    """
    This is the data from 1582 German catchments following the work of
    `Loritz et al., 2024 <https://doi.org/10.5194/essd-16-5625-2024>`_ .
    The data is downloaded from `zenodo <https://zenodo.org/record/12733968>`_ .
    This data consists of 111 static and 21 dynamic features. The dynamic features
    span from 1951-01-01 to 2020-12-31 with daily timestep.

    Hourly data (CAMELS-DE-1h) is available by setting ``timestep='H'``. It has
    1611 catchments with 26 dynamic and 109 static features spanning
    2001-01-01 to 2024-12-31, following `Dolich et al., 2026
    <https://doi.org/10.5194/essd-2026-289>`_ and downloaded from
    `GFZ dataservices <https://doi.org/10.5880/fidgeo.2026.045>`_ .
    Only observed streamflow, meteorological forcing, static attributes and
    catchment boundaries are provided; the modelled benchmark simulations and
    the weather-forecast files are not downloaded. The netcdf consolidation is
    skipped by default for the hourly data (``to_netcdf`` defaults to False).

    Examples
    --------
    >>> from aqua_fetch import CAMELS_DE
    >>> dataset = CAMELS_DE()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='DE110260', as_dataframe=True)
    >>> df = dynamic['DE110260'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (25568, 21)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       1582
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (155 out of 1582)
       155
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(25568, 21), (25568, 21), (25568, 21),... (25568, 21), (25568, 21)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('DE110260', as_dataframe=True,
    ...  dynamic_features=['airtemp_C_mean', 'rh_%', 'pcp_mm_mean', 'q_cms_obs'])
    >>> dynamic['DE110260'].shape
       (25568, 4)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='DE110260', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['DE110260'].shape
    ((1, 111), 1, (25568, 21))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 25568, 'dynamic_features': 21})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (1582, 2)
    >>> dataset.stn_coords('DE110260')  # returns coordinates of station whose id is DE110260
        47.925221       8.191595
    >>> dataset.stn_coords(['DE110260', 'DE110250'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('DE110260')
    # get coordinates of two stations
    >>> dataset.area(['DE110260', 'DE110250'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('DE110260')
    ...
    # the hourly (CAMELS-DE-1h) data can be accessed by setting timestep to 'H'
    >>> dataset = CAMELS_DE(timestep='H')
    >>> len(dataset.stations())
    1611
    >>> _, dynamic = dataset.fetch(stations='DE110000', as_dataframe=True)
    >>> dynamic['DE110000'].shape
    (210383, 26)
    """
    url = "https://zenodo.org/record/16755906"

    # direct download of the hourly (CAMELS-DE-1h) archive from GFZ dataservices.
    # only the ~14 GB CAMELS-DE-1h.zip is listed here (it bundles the timeseries,
    # static attributes and shapefiles). The two weather-forecast .zarr.zip files
    # (deterministic ~15 GB and ensemble ~51 GB) which live at the same URL base
    # are deliberately NOT included so they are never downloaded.
    hourly_url = {
        "CAMELS-DE-1h.zip":
            "https://datapub.gfz.de/download/10.5880.FIDGEO.2026.045-Trfghbv/"
            "2026-045_Dolich-et-al_data/CAMELS-DE-1h.zip",
    }

    time_steps = ['D', 'H']

    def __init__(
            self,
            path=None,
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
            If the data is alredy downloaded then provide the complete
            path to it. If None, then the data will be downloaded.
            The data is downloaded once and therefore susbsequent
            calls to this class will not download the data unless
            ``overwrite`` is set to True.
        timestep : str
            possible values are ``D`` for daily (default) or ``H`` for the
            hourly (CAMELS-DE-1h) data.
        overwrite : bool
            If the data is already down then you can set it to True,
            to make a fresh download.
        to_netcdf : bool
            whether to convert all the data into one netcdf file or not.
            This will fasten repeated calls to fetch etc. but will
            require netCDF4 package as well as xarray. If not given, it
            defaults to True for the daily timestep and False for the hourly
            timestep. The hourly data is too large to consolidate into a single
            netcdf file and reading it directly from the csv files is already
            fast, so the conversion is skipped by default.
        """
        assert timestep in self.time_steps, \
            f"invalid timestep '{timestep}' given, choose from {self.time_steps}"

        if to_netcdf is None:
            # hourly data is too large for the single-file netcdf consolidation
            # and raw-csv fetching is already fast, so skip it by default.
            to_netcdf = (timestep == 'D')

        if to_netcdf and netCDF4 is None:
            warnings.warn("netCDF4 is not installed. Therefore, the data will not be converted to netcdf format.")
            to_netcdf = False

        # forward timestep and to_netcdf so the parent stores them before we use
        # them below (otherwise self.to_netcdf keeps the parent's default of True)
        super().__init__(path=path, timestep=timestep, to_netcdf=to_netcdf,
                         verbosity=verbosity, **kwargs)

        # Lazy caches for the heavy accessors. The base implementations recompute
        # these on every call, so a single bulk ``fetch`` ends up re-listing the
        # timeseries directory and re-reading a station csv hundreds of times
        # (``fetch``/``check_*`` reference ``dynamic_features`` and ``stations()``
        # once per requested station). Caching them here makes repeated fetches
        # fast without touching any values. They are populated on first use so
        # class initialization stays quick (unless it triggers the download).
        self._stations_cache = None
        self._dyn_features_cache = None
        self._static_data_cache = None

        if timestep == 'D':
            self._download_daily(overwrite=overwrite)
        else:
            # CAMELS-DE-1h corresponds to an ESSD preprint (essd-2026-289) that is
            # still under open review; the final accepted dataset may differ.
            warnings.warn(
                "CAMELS-DE-1h (timestep='H') corresponds to a preprint "
                "(https://doi.org/10.5194/essd-2026-289) which is under open "
                "review, not the final peer-reviewed dataset. Values may change "
                "in the accepted version; verify before using for critical work.",
                UserWarning,
            )
            self._download_hourly(overwrite=overwrite)

        self._maybe_to_netcdf()

        self.bbox = {'llcrnrlat': 47.0, 'urcrnrlat': 55.0, 'llcrnrlon': 5.0, 'urcrnrlon': 16.0}
        self.parallels = range(47, 55, 2)
        self.meridians = range(5, 16, 2)

    def _download_daily(self, overwrite: bool = False):
        """Downloads the daily CAMELS-DE data from zenodo.

        The guard is on the daily ``camels_de`` folder (not just a non-empty
        ``CAMELS_DE`` directory) so that the presence of the sibling hourly
        ``CAMELS-DE-1h`` folder does not fool the download check when both
        timesteps share the same ``path``. When the daily folder is missing we
        download directly (bypassing the parent-directory-non-empty check that
        the sibling hourly folder would otherwise satisfy).
        """
        if os.path.exists(self.ts_dir) and not overwrite:
            if self.verbosity:
                print(f"daily data already exists at {self._root}")
            return
        download_and_unzip(self.path, url=self.url, verbosity=self.verbosity)

    def _download_hourly(self, overwrite: bool = False):
        """Downloads and extracts only the CAMELS-DE-1h.zip archive from GFZ.

        The weather-forecast archives (the two large ``.zarr.zip`` files that
        sit at the same URL base) are never requested because ``hourly_url``
        only lists the main archive. The extracted (modelled) ``benchmark_models``
        folder is removed right after extraction since it is never read.

        If the extracted data is already present, nothing is downloaded or
        extracted (unless ``overwrite=True``).
        """
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        fname, url = next(iter(self.hourly_url.items()))
        zip_path = os.path.join(self.path, fname)

        # already have the extracted data -> nothing to do (unless overwrite)
        if os.path.exists(self.ts_dir) and not overwrite:
            return

        # download only if the archive is not already on disk (or overwrite).
        # This way, deleting the .zip after extraction does not re-trigger a
        # download as long as the extracted timeseries folder is present.
        if overwrite or not os.path.exists(zip_path):
            if self.verbosity:
                print(f"Downloading {fname} from {url}")
            download(url, outdir=self.path, fname=fname, verbosity=self.verbosity)

        # forward overwrite so that a fresh download also replaces a stale
        # extracted tree (unzip skips extraction when the target folder exists
        # unless overwrite=True is passed).
        unzip(self.path, overwrite=overwrite, verbosity=self.verbosity)

        # the extracted archive ships a modelled ``benchmark_models`` folder
        # (LSTM/HBV simulations, ~1.8 GB) which we do not expose; remove it so it
        # does not sit on disk. The ~14 GB CAMELS-DE-1h.zip is left in place (a
        # later `free_disk_space("archives")` call removes it) so that deleting it
        # does not force a re-download.
        benchmark_dir = os.path.join(self._root, 'benchmark_models')
        if os.path.exists(benchmark_dir):
            if self.verbosity:
                print(f"removing modelled benchmark data at {benchmark_dir}")
            shutil.rmtree(benchmark_dir)

    @property
    def _prefix(self) -> str:
        """filename prefix used by the dataset files for the current timestep"""
        return "CAMELS_DE_1h" if self.timestep == 'H' else "CAMELS_DE"

    @property
    def _root(self) -> os.PathLike:
        """folder into which the archive extracts for the current timestep.

        The hourly archive ``CAMELS-DE-1h.zip`` extracts into a doubly-nested
        ``CAMELS-DE-1h/CAMELS-DE-1h/`` folder which holds the timeseries, the
        attribute csvs and the catchment boundaries.
        """
        if self.timestep == 'H':
            return os.path.join(self.path, "CAMELS-DE-1h", "CAMELS-DE-1h")
        return os.path.join(self.path, "camels_de")

    @property
    def boundary_file(self) -> os.PathLike:
        if self.timestep == 'H':
            return os.path.join(self._root,
                                "CAMELS_DE_1h_catchment_boundaries",
                                "catchments",
                                "CAMELS_DE_1h_catchments.shp")
        return os.path.join(self._root,
                            "CAMELS_DE_catchment_boundaries",
                            "catchments",
                            "CAMELS_DE_catchments.shp")

    def transform_boundary(self, boundary):
        """
        The hourly (CAMELS-DE-1h) catchment boundaries are in ETRS89-LAEA
        (EPSG:3035, meters); transform them to WGS84 (lat/lon) so they align
        with the gauge coordinates. The shared ``laea_to_wgs84`` helper performs
        the ellipsoidal (GRS80) inverse projection, verified against pyproj to
        better than 1e-8 m. The daily boundaries are left untouched (the base
        no-op) to preserve existing behaviour.
        """
        if self.timestep != 'H':
            return super().transform_boundary(boundary)

        # EPSG:3035 parameters (from the shapefile .prj)
        lon_0, lat_0 = 10.0, 52.0
        false_easting, false_northing = 4321000.0, 3210000.0

        assert len(boundary.coordinates) == 1  # only one polygon
        longs, lats = [], []
        for x, y, *_ in boundary.coordinates[0]:
            lat_, long_ = laea_to_wgs84(x, y, lon_0, lat_0, false_easting, false_northing)
            longs.append(long_)
            lats.append(lat_)

        if fiona is not None:
            boundary = fiona.Geometry(type='Polygon',
                                      coordinates=[list(zip(longs, lats))])
        return boundary

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'area': catchment_area(),
                'slope_fdc': slope(''),
                'gauge_lat': gauge_latitude(),
                'gauge_lon': gauge_longitude(),
        }

    @property
    def dyn_map(self):
        if self.timestep == 'H':
            return self._dyn_map_hourly
        # table 1 in https://essd.copernicus.org/articles/16/5625/2024/#&gid=1&pid=1
        return {
            'discharge_vol_obs': observed_streamflow_cms(),
            'discharge_spec_obs': observed_streamflow_mm(),
            'temperature_min': min_air_temp(),
            'temperature_max': max_air_temp(),
            'temperature_mean': mean_air_temp(),
            # 'precipitation_mean': 'pcp_mm',
            'precipitation_mean': total_precipitation_with_specifier('mean'), # todo: is it mean or total?
            'precipitation_median': total_precipitation_with_specifier('median'),
            'precipitation_stdev': total_precipitation_with_specifier('std'),
            'precipitation_min': total_precipitation_with_specifier('min'),
            'precipitation_max': total_precipitation_with_specifier('max'),
            'humidity_mean': mean_rel_hum(),
            'humidity_median': rel_hum_with_specifier('med'),
            'humidity_stdev': rel_hum_with_specifier('std'),
            'humidity_min': rel_hum_with_specifier('min'),
            'humidity_max': rel_hum_with_specifier('max'),
            # 'water_level':  # observed daily water level,
            # The Data Description defines these as the "spatial mean, median,
            # minimum, maximum and standard deviation of the global radiation",
            # i.e. the spread of the gridded forcing ACROSS the catchment, not
            # a within-day range. They are therefore marked `spat`. The spatial
            # mean is what the bare canonical name already denotes, so
            # radiation_global_mean carries no token.
            'radiation_global_mean': solar_radiation(),
            'radiation_global_stdev': solar_radiation_with_spatial_stat('std'),
            'radiation_global_min': solar_radiation_with_spatial_stat('min'),
            'radiation_global_median': solar_radiation_with_spatial_stat('med'),
            'radiation_global_max': solar_radiation_with_spatial_stat('max'),
        }

    @property
    def _dyn_map_hourly(self) -> Dict[str, str]:
        # Table 1 of Dolich et al., 2026 (CAMELS-DE-1h). All observed and
        # meteorological-forcing columns are kept; those with a canonical name in
        # the aqua_fetch vocabulary are renamed here, the rest keep their original
        # dataset name (surface pressure, wind direction and the precip-coverage
        # diagnostic). The modelled/benchmark data lives in separate files/folders
        # and is never read.
        return {
            'discharge_vol_obs': observed_streamflow_cms(),
            'discharge_spec_obs': observed_streamflow_mm(),  # mm/hour
            'water_level_obs': observed_water_level_cm(),  # cm
            'precipitation_mean': total_precipitation_with_specifier('mean'),
            'precipitation_min': total_precipitation_with_specifier('min'),
            'precipitation_max': total_precipitation_with_specifier('max'),
            'precipitation_stdev': total_precipitation_with_specifier('std'),
            'precipitation_mean_gapfilled': total_precipitation_with_specifier('mean_gapfilled'),
            'precipitation_min_gapfilled': total_precipitation_with_specifier('min_gapfilled'),
            'precipitation_max_gapfilled': total_precipitation_with_specifier('max_gapfilled'),
            'precipitation_stdev_gapfilled': total_precipitation_with_specifier('std_gapfilled'),
            'air_temperature_mean': mean_air_temp(),
            'air_temperature_min': min_air_temp(),
            'air_temperature_max': max_air_temp(),
            'relative_humidity_mean': mean_rel_hum(),
            'water_vapor_mixing_ratio_mean': mean_specific_humidity(),  # g/kg
            'global_radiation_mean': solar_radiation(),
            'air_pressure_sea_level_mean': mean_air_pressure(),  # hPa
            'cloud_cover_mean': cloud_cover(),
            'dew_point_temperature_mean': mean_dewpoint_temperature_at_2m(),
            'wind_speed_eastward_mean': u_component_of_wind_at_10m(),
            'wind_speed_northward_mean': v_component_of_wind_at_10m(),
            'wind_speed_mean': mean_windspeed(),
            # kept with original names (no canonical mapping):
            #   air_pressure_surface_mean (hPa)
            #   wind_direction_mean (degrees)
            #   perc_nan_precipitation_original (% catchment w/o original radar precip)
        }

    @property
    def ts_dir(self) -> str:
        return os.path.join(self._root, 'timeseries')

    def _attr_path(self, name: str) -> str:
        """path to a static-attribute csv for the current timestep"""
        return os.path.join(self._root, f"{self._prefix}_{name}.csv")

    @property
    def clim_attr_path(self) -> str:
        return self._attr_path("climatic_attributes")

    @property
    def hum_infl_path(self) -> str:
        return self._attr_path("humaninfluence_attributes")

    @property
    def hydrogeol_attr_path(self) -> str:
        return self._attr_path("hydrogeology_attributes")

    @property
    def hydrol_attr_path(self) -> str:
        return self._attr_path("hydrologic_attributes")

    @property
    def lc_attr_path(self) -> str:
        return self._attr_path("landcover_attributes")

    @property
    def sim_attr_path(self) -> str:
        return self._attr_path("simulation_benchmark")

    @property
    def soil_attr_path(self) -> str:
        return self._attr_path("soil_attributes")

    @property
    def topo_attr_path(self) -> str:
        return self._attr_path("topographic_attributes")

    def stations(self) -> List[str]:
        # daily file: CAMELS_DE_hydromet_timeseries_<id>.csv  -> id at split index 4
        # hourly file: CAMELS_DE_1h_hydromet_timeseries_<id>.csv -> id at split index 5
        if self._stations_cache is None:
            self._stations_cache = [os.path.splitext(f)[0].split('_')[-1]
                                    for f in os.listdir(self.ts_dir)
                                    if f.endswith('.csv')]
        # return a copy so a caller's in-place edit cannot corrupt the cache
        return list(self._stations_cache)

    def clim_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self.clim_attr_path, index_col='gauge_id',
                           # dtype=np.float32
                           )

    def hum_infl_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self.hum_infl_path, index_col='gauge_id',
                           # dtype=np.float32
                           )

    def hydrogeol_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self.hydrogeol_attr_path, index_col='gauge_id',
                           # dtype=np.float32
                           )

    def hydrol_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self.hydrol_attr_path, index_col='gauge_id',  # dtype=np.float32
                           )

    def lc_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self.lc_attr_path, index_col='gauge_id',  # dtype=np.float32
                           )

    def sim_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self.sim_attr_path, index_col='gauge_id',  # dtype=np.float32
                           )

    def soil_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self.soil_attr_path, index_col='gauge_id',  # dtype=np.float32
                           )

    def topo_attrs(self) -> pd.DataFrame:
        return pd.read_csv(self.topo_attr_path, index_col='gauge_id',  # dtype=np.float32
                           )

    def _static_data(self) -> pd.DataFrame:
        # the concatenated static table (8 csv reads) is cached because it is
        # requested repeatedly (static_features, area, stn_coords, fetch_static_*).
        # Callers only read from it (``.loc`` returns copies), so returning the
        # cached frame directly is safe and does not change any values.
        if self._static_data_cache is not None:
            return self._static_data_cache

        attrs = [
            self.clim_attrs(),
            self.hum_infl_attrs(),
            self.hydrogeol_attrs(),
            self.hydrol_attrs(),
            self.lc_attrs(),
            self.soil_attrs(),
            self.topo_attrs(),
        ]
        # the daily dataset ships a simulation_benchmark file with observed-data
        # signatures. For the hourly (CAMELS-DE-1h) dataset that file only holds
        # modelled benchmark scores (NSE_lstm, NSE_hbv, ...) which we do not make
        # available, so it is excluded from the static attributes.
        if self.timestep == 'D':
            attrs.append(self.sim_attrs())

        df = pd.concat(attrs, axis=1)

        df.rename(columns=self.static_map, inplace=True)

        self._static_data_cache = df

        return df

    def _read_stn_dyn(self, station) -> pd.DataFrame:
        """
        Reads dynamic (meteorological + streamflow) data for one catchment
        and returns as DataFrame
        """

        df = pd.read_csv(
            os.path.join(self.ts_dir, f"{self._prefix}_hydromet_timeseries_{station}.csv"),
            # sep=';',
            index_col='date',
            parse_dates=True,
            # dtype=np.float32
        )

        df.rename(columns=self.dyn_map, inplace=True)

        return df

    def _read_dynamic(
            self,
            stations,
            dynamic_features,
            st: Union[str, pd.Timestamp] = None,
            en: Union[str, pd.Timestamp] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Read dynamic data of many stations, in parallel where it pays off.

        Behaves exactly like the base implementation (same dict of
        ``{station: DataFrame}`` sliced to ``[st:en, dyn_feats]`` with the axis
        names set) but, for the parallel branch, dispatches the module-level
        :func:`_read_camels_de_dynamic` instead of the bound
        ``self._read_stn_dyn``. The station csv files here are large (the hourly
        files are ~31 MB each), so avoiding the per-task pickling of ``self`` —
        which the base incurs by handing a bound method to the pool — roughly
        halves the wall-clock of a big fetch. Values and dtypes are unchanged.
        """
        st, en = self._check_length(st, en)
        dyn_feats = validate_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')
        stations = validate_attributes(stations, self.stations(), 'stations')

        cpus = self.processes or min(get_cpus(), 16)
        start = time.time()
        if len(stations) < cpus:
            cpus = 1

        if cpus == 1:
            dyn = {}
            for idx, stn in enumerate(stations):
                stn_df = self._read_stn_dyn(stn).loc[st:en, dyn_feats]
                stn_df.columns.name = 'dynamic_features'
                stn_df.index.name = 'time'
                dyn[stn] = stn_df

                if self.verbosity and idx % 100 == 0:
                    print(f"Read {idx+1}/{len(stations)} stations.")
        else:
            # a plain function carrying only small, picklable arguments (paths,
            # the rename map and the requested window/features) — NOT the bound
            # method — so the pool does not ship a copy of ``self`` per task.
            reader = functools.partial(
                _read_camels_de_dynamic,
                self.ts_dir, self._prefix, tuple(self.dyn_map.items()),
                list(dyn_feats), st, en,
            )
            with cf.ProcessPoolExecutor(cpus) as executor:
                results = executor.map(reader, stations)

            dyn = {stn: stn_df for stn, stn_df in zip(stations, results)}

        if self.verbosity:
            total = time.time() - start
            print(f"Read {len(dyn)} stations for {len(dyn_feats)} dyn features "
                  f"in {total:.2f} seconds with {cpus} cpus.")

        return dyn

    @property
    def start(self):
        if self.timestep == 'H':
            return pd.Timestamp('2001-01-01 01:00:00')
        return pd.Timestamp('1951-01-01')

    @property
    def end(self):
        if self.timestep == 'H':
            return pd.Timestamp('2024-12-31 23:00:00')
        return pd.Timestamp('2020-12-31')

    @property
    def dynamic_features(self) -> List[str]:
        # cached: the base reads (and renames) a full station csv on every access,
        # and fetch()/check_* touch this once per requested station. The column
        # set is identical for every station, so read it once.
        if self._dyn_features_cache is None:
            self._dyn_features_cache = self._read_stn_dyn(self.stations()[0]).columns.tolist()
        # return a copy so a caller's in-place edit cannot corrupt the cache
        return list(self._dyn_features_cache)

    @property
    def static_features(self) -> List[str]:
        return self._static_data().columns.tolist()

    @property
    def _coords_name(self) -> List[str]:
        return ['gauge_lat', 'gauge_lon']

    @property
    def _area_name(self) -> str:
        return 'area'

    @property
    def _mm_feature_name(self) -> str:
        """Observed catchment-specific discharge (converted to millimetres per day
        using catchment areas"""
        return observed_streamflow_mm()


class CAMELS_SE(_RainfallRunoff):
    """
    Dataset of 50 Swedish catchments following the works of
    `Teutschbein et al., 2024 <https://doi.org/10.1002/gdj3.239>`_ . The data is downloaded
    from Swedish National Data Service `website <https://snd.se/en/catalogue/dataset/2023-173>`_ .
    The dataset consists of 76 static catchment features and 4 dynamic features.
    The dynamic features span from 19610101 to 20201231 with daily timestep.

    Examples
    --------
    >>> from aqua_fetch import CAMELS_SE
    >>> dataset = CAMELS_SE()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='5', as_dataframe=True)
    >>> df = dynamic['5'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (21915, 4)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       50
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (5 out of 50)
       5
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(21915, 4), (21915, 4), (21915, 4),... (21915, 4), (21915, 4)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('5', as_dataframe=True,
    ...  dynamic_features=['q_cms_obs', 'q_mm_obs', 'pcp_mm', 'airtemp_C_mean'])
    >>> dynamic['5'].shape
       (21915, 5)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='5', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['5'].shape
    ((1, 76), 1, (21915, 4))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 21915, 'dynamic_features': 4})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (50, 2)
    >>> dataset.stn_coords('5')  # returns coordinates of station whose id is 5
        68.0356 21.9758
    >>> dataset.stn_coords(['5', '200'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('5')
    # get coordinates of two stations
    >>> dataset.area(['5', '200'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('5')

    """

    url = {
 'catchment properties.zip': 'https://api.researchdata.se/dataset/2023-173/1/file/data?filePath=catchment+properties.zip',
 'catchment time series.zip': 'https://api.researchdata.se/dataset/2023-173/1/file/data?filePath=catchment+time+series.zip',
 'catchment_GIS_shapefiles.zip': 'https://api.researchdata.se/dataset/2023-173/1/file/data?filePath=catchment_GIS_shapefiles.zip',
 'Documentation_2024-01-02.pdf': 'https://api.researchdata.se/dataset/2023-173/1/file/documentation?filePath=Documentation_2024-01-02.pdf'
    }

    def __init__(
            self,
            path: str = None,
            to_netcdf: bool = True,
            overwrite: bool = False,
            verbosity: int = 1,
            **kwargs
    ):
        """
        Arguments:
            path: path where the CAMELS_SE dataset has been downloaded. This path
                must contain five zip files and one xlsx file. If None, then the
                data will be downloaded.
            to_netcdf :
        """
        super().__init__(path=path, verbosity=verbosity, **kwargs)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        for _file, url in self.url.items():
            fpath = os.path.join(self.path, _file)
            if not os.path.exists(fpath) and not overwrite:
                if verbosity > 0:
                    print(f"Downloading {_file} from {url}")
                download(url, outdir=self.path, fname=_file, verbosity=self.verbosity)
                unzip(self.path, verbosity=self.verbosity)
            else:
                if self.verbosity > 0: print(f"{_file} at {self.path} already exists")

        self._static_features = list(set(self._static_data().columns.tolist()))
        self._stations = self.physical_properties().index.to_list()
        self._dynamic_features = self._read_stn_dyn(self.stations()[0], nrows=2).columns.tolist()

        if to_netcdf and netCDF4 is None:
            warnings.warn("netCDF4 is not installed. Therefore, the data will not be converted to netcdf format.")
            to_netcdf = False

        # if to_netcdf:
        self._maybe_to_netcdf()

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(self.path,
                            'catchment_GIS_shapefiles',
                            'catchment_GIS_shapefiles',
                            'Sweden_catchments_50_boundaries_WGS84.shp')

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'Area_km2': catchment_area(),
                'slope_mean_degree': slope('degrees'),
                'Latitude_WGS84': gauge_latitude(),
                'Longitude_WGS84': gauge_longitude(),
        }

    @property
    def dyn_map(self):
        return {
            'Qobs_m3s': observed_streamflow_cms(),
            'Qobs_mm': observed_streamflow_mm(),
            'Tobs_C': mean_air_temp(),
            'Pobs_mm': total_precipitation(),
        }

    @property
    def static_features(self):
        return self._static_features

    @property
    def dynamic_features(self) -> List[str]:
        return self._dynamic_features

    @property
    def properties_path(self):
        return os.path.join(self.path, 'catchment properties', 'catchment properties')

    @property
    def ts_dir(self) -> os.PathLike:
        return os.path.join(self.path, 'catchment time series', 'catchment time series')

    @property
    def _mm_feature_name(self) -> str:
        return observed_streamflow_cms()

    @property
    def start(self):
        return pd.Timestamp("19610101")

    @property
    def end(self):
        return pd.Timestamp("20201231")

    def stations(self) -> List[str]:
        return self._stations

    def landcover(self) -> pd.DataFrame:
        return pd.read_csv(
            os.path.join(self.properties_path, 'catchments_landcover.csv'),
            index_col='ID', dtype={'ID': str})

    def physical_properties(self) -> pd.DataFrame:
        return pd.read_csv(
            os.path.join(self.properties_path, 'catchments_physical_properties.csv'),
            index_col='ID', dtype={'ID': str})

    def soil_classes(self) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(self.properties_path, 'catchments_soil_classes.csv'),
            index_col='ID', dtype={'ID': str})
        df.columns = [f"{c}_sc" for c in df.columns]
        return df

    def hydro_signatures_1961_2020(self) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(self.properties_path, 'catchments_hydrological_signatures_1961_2020.csv'),
            index_col='ID', dtype={'ID': str})
        df.columns = [f"{c}_hs" for c in df.columns]
        return df

    def hydro_signatures_CNP_1961_1990(self) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(self.properties_path, 'catchments_hydrological_signatures_CNP1_1961_1990.csv'),
            index_col='ID', dtype={'ID': str})
        df.columns = [f"{c}_CNP_61_90" for c in df.columns]
        return df

    def hydro_signatures_CNP_1990_2020(self) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(self.properties_path, 'catchments_hydrological_signatures_CNP2_1991_2020.csv'),
            index_col='ID', dtype={'ID': str})
        df.columns = [f"{c}_CNP_91_20" for c in df.columns]
        return df

    def _static_data(self) -> pd.DataFrame:
        df = pd.concat([
            self.landcover(),
            self.physical_properties(),
            self.soil_classes(),
            self.hydro_signatures_1961_2020(),
            self.hydro_signatures_CNP_1961_1990(),
            self.hydro_signatures_CNP_1990_2020()
        ], axis=1)

        df.rename(columns=self.static_map, inplace=True)

        return df

    def _read_stn_dyn(self, station, nrows=None) -> pd.DataFrame:
        """
        Reads daily dynamic (meteorological + streamflow) data for one catchment
        and returns as DataFrame
        """
        # find file starting with 'catchment_id_stn_id_' in self.path
        stn_id = f'catchment_id_{station}_'
        fname = [f for f in os.listdir(self.ts_dir) if f.startswith(stn_id)]
        assert len(fname) == 1
        fname = fname[0]

        df = pd.read_csv(
            os.path.join(self.ts_dir, fname),
            index_col='Year_Month_Day',
            parse_dates=[['Year', 'Month', 'Day']],
            dtype={'Qobs_m3s': np.float32, 'Qobs_mm': np.float32, 'Pobs_mm': np.float32, 'Tobs_C': np.float32},
            nrows=nrows,
        )

        for old_name, new_name in self.dyn_map.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)

        return df


class CAMELS_DK(_RainfallRunoff):
    """
    This is an updated version of :py class:`aqua_fetch.rr.Caravan_DK`
    dataset . This dataset was presented
    by `Liu et al., 2024 <https://doi.org/10.5194/essd-2024-292>`_ and is
    available at `dataverse <https://dataverse.geus.dk/dataset.xhtml?persistentId=doi:10.22008/FK2/AZXSYP>`_ .
    This dataset consists of 119 static and 13 dynamic features from 3330 Danish catchments.
    The dynamic (time series) features span from 1989-01-02 to 2023-12-31 with daily timestep.
    However, the streamflow observations are available for only 304 catchments.

    Examples
    ---------
    >>> from aqua_fetch import CAMELS_DK
    >>> dataset = CAMELS_DK()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='54130033', as_dataframe=True)
    >>> df = dynamic['54130033'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (12782, 13)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       304
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (30 out of 304)
       30
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(12782, 13), (12782, 13), (12782, 13),... (12782, 13), (12782, 13)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('54130033', as_dataframe=True,
    ...  dynamic_features=['Abstraction', 'pet_mm', 'airtemp_C_mean', 'pcp_mm', 'q_cms_obs'])
    >>> dynamic['54130033'].shape
       (12782, 5)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='54130033', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['54130033'].shape
    ((1, 119), 1, (12782, 13))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 12782, 'dynamic_features': 13})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (304, 2)
    >>> dataset.stn_coords('54130033')  # returns coordinates of station whose id is 54130033
        55.325242	9.93079
    >>> dataset.stn_coords(['54130033', '13210113'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('54130033')
    # get coordinates of two stations
    >>> dataset.area(['54130033', '13210113'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('54130033')
    """

    url = {
        'CAMELS_DK_304_gauging_catchment_boundaries.cpg': 'https://dataverse.geus.dk/api/access/datafile/83017',
        'CAMELS_DK_304_gauging_catchment_boundaries.prj': 'https://dataverse.geus.dk/api/access/datafile/83019',
        'CAMELS_DK_304_gauging_catchment_boundaries.shp': 'https://dataverse.geus.dk/api/access/datafile/83021',
        'CAMELS_DK_304_gauging_catchment_boundaries.dbf': 'https://dataverse.geus.dk/api/access/datafile/83020',
        'CAMELS_DK_304_gauging_catchment_boundaries.shx': 'https://dataverse.geus.dk/api/access/datafile/83018',
        'CAMELS_DK_304_gauging_stations.cpg': 'https://dataverse.geus.dk/api/access/datafile/83008',
        'CAMELS_DK_304_gauging_stations.dbf': 'https://dataverse.geus.dk/api/access/datafile/83010',
        'CAMELS_DK_304_gauging_stations.prj': 'https://dataverse.geus.dk/api/access/datafile/83009',
        'CAMELS_DK_304_gauging_stations.shp': 'https://dataverse.geus.dk/api/access/datafile/83011',
        'CAMELS_DK_304_gauging_stations.shx': 'https://dataverse.geus.dk/api/access/datafile/83007',
        'CAMELS_DK_climate.csv': 'https://dataverse.geus.dk/api/access/datafile/83123',
        'CAMELS_DK_geology.csv': 'https://dataverse.geus.dk/api/access/datafile/83124',
        'CAMELS_DK_georegion.dbf': 'https://dataverse.geus.dk/api/access/datafile/83030',
        'CAMELS_DK_georegion.prj': 'https://dataverse.geus.dk/api/access/datafile/83026',
        'CAMELS_DK_georegion.sbn': 'https://dataverse.geus.dk/api/access/datafile/83027',
        'CAMELS_DK_georegion.sbx': 'https://dataverse.geus.dk/api/access/datafile/83028',
        'CAMELS_DK_georegion.shp': 'https://dataverse.geus.dk/api/access/datafile/83029',
        'CAMELS_DK_georegion.shx': 'https://dataverse.geus.dk/api/access/datafile/83031',
        'CAMELS_DK_landuse.csv': 'https://dataverse.geus.dk/api/access/datafile/83125',
        'CAMELS_DK_script.py': 'https://dataverse.geus.dk/api/access/datafile/83135',
        'CAMELS_DK_signature_obs_based.csv': 'https://dataverse.geus.dk/api/access/datafile/83131',
        'CAMELS_DK_signature_sim_based.csv': 'https://dataverse.geus.dk/api/access/datafile/83132',
        'CAMELS_DK_soil.csv': 'https://dataverse.geus.dk/api/access/datafile/83126',
        'CAMELS_DK_topography.csv': 'https://dataverse.geus.dk/api/access/datafile/83127',
        'Data_description.pdf': 'https://dataverse.geus.dk/api/access/datafile/83138',
        'Gauged_catchments.zip': 'https://dataverse.geus.dk/api/access/datafile/83022',
        'Ungauged_catchments.zip': 'https://dataverse.geus.dk/api/access/datafile/83025',
    }

    def __init__(self,
                 path=None,
                 overwrite=False,
                 to_netcdf: bool = True,
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
            require netCDF4 package as well as xarray.
        """
        super(CAMELS_DK, self).__init__(path=path, **kwargs)
        self._download(overwrite=overwrite)

        # self.dyn_fname = os.path.join(self.path, 'camelsdk_dyn.nc')
        self._static_features = self._static_data().columns.to_list()
        self._dynamic_features = self._read_csv(self.stations()[0]).columns.to_list()

        # if to_netcdf:
        self._maybe_to_netcdf()

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(
            self.path,
            "CAMELS_DK_304_gauging_catchment_boundaries.shp"
        )

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'catch_area': catchment_area(),
                'slope_mean': slope('mkm-1'),
                'catch_outlet_lat': gauge_latitude(),
                'catch_outlet_lon': gauge_longitude(),
        }

    @property
    def dyn_map(self):
        # table 1 in https://essd.copernicus.org/preprints/essd-2024-292/essd-2024-292.pdf
        return {
            'Qobs': observed_streamflow_cms(),
            'temperature': mean_air_temp(),
            'precipitation': total_precipitation(),
            'pet': total_potential_evapotranspiration(),  # todo: should we write method (makkink)
            'Qsim': simulated_streamflow_cms(),
            "DKM_eta": actual_evapotranspiration()
        }

    @property
    def gaug_catch_path(self):
        return os.path.join(self.path, "Gauged_catchments", "Gauged_catchments")

    @property
    def climate_fpath(self):
        return os.path.join(self.path, "CAMELS_DK_climate.csv")

    @property
    def geology_fpath(self):
        return os.path.join(self.path, "CAMELS_DK_geology.csv")

    @property
    def landuse_fpath(self):
        return os.path.join(self.path, "CAMELS_DK_landuse.csv")

    @property
    def soil_fpath(self):
        return os.path.join(self.path, "CAMELS_DK_soil.csv")

    @property
    def topography_fpath(self):
        return os.path.join(self.path, "CAMELS_DK_topography.csv")

    @property
    def signature_obs_fpath(self):
        return os.path.join(self.path, "CAMELS_DK_signature_obs_based.csv")

    @property
    def signature_sim_fpath(self):
        return os.path.join(self.path, "CAMELS_DK_signature_sim_based.csv")

    def climate_data(self):
        df = pd.read_csv(self.climate_fpath, index_col=0)
        df.index = df.index.astype(str)
        return df

    def geology_data(self):
        df = pd.read_csv(self.geology_fpath, index_col=0)
        df.index = df.index.astype(str)
        return df

    def landuse_data(self):
        df = pd.read_csv(self.landuse_fpath, index_col=0)
        df.index = df.index.astype(str)
        return df

    def soil_data(self):
        df = pd.read_csv(self.soil_fpath, index_col=0)
        df.index = df.index.astype(str)
        return df

    def topography_data(self):
        df = pd.read_csv(self.topography_fpath, index_col=0)
        df.index = df.index.astype(str)
        return df

    def signature_obs_data(self):
        df = pd.read_csv(self.signature_obs_fpath, index_col=0)
        df.index = df.index.astype(str)
        return df

    def signature_sim_data(self):
        df = pd.read_csv(self.signature_sim_fpath, index_col=0)
        df.index = df.index.astype(str)
        return df

    def _static_data(self) -> pd.DataFrame:
        """combination of topographic + soil + landuse + geology + climate features

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of static features of all catchments of shape (3330, 119)
        """
        df = pd.concat([self.climate_data(),
                          self.geology_data(),
                          self.landuse_data(),
                          self.soil_data(),
                          self.topography_data()
                          ], axis=1)
        
        df.rename(columns=self.static_map, inplace=True)

        return df

    def stations(self) -> List[str]:
        return [fname.split(".csv")[0].split('_')[4] for fname in os.listdir(self.gaug_catch_path)]

    def _read_csv(self, stn: str) -> pd.DataFrame:
        fpath = os.path.join(self.gaug_catch_path, f"CAMELS_DK_obs_based_{stn}.csv")
        df = pd.read_csv(os.path.join(fpath), parse_dates=True, index_col='time')
        df.columns.name = 'dynamic_features'
        df.pop('catch_id')
        df = df.astype(np.float32)

        df.rename(columns=self.dyn_map, inplace=True)
        # for old_name, new_name in self.dyn_map.items():
        #     if old_name in df.columns:
        #         df.rename(columns={old_name: new_name}, inplace=True)
        return df

    @property
    def dynamic_features(self) -> List[str]:
        """returns names of dynamic features"""
        return self._dynamic_features

    @property
    def static_features(self) -> List[str]:
        """returns static features for Denmark catchments"""
        return self._static_features

    @property
    def _coords_name(self) -> List[str]:
        return ['catch_outlet_lat', 'catch_outlet_lon']

    @property
    def _area_name(self) -> str:
        return 'catch_area'

    @property
    def _q_name(self) -> str:
        return observed_streamflow_cms()

    @property
    def start(self) -> pd.Timestamp:  # start of data
        return pd.Timestamp('1989-01-02 00:00:00')

    @property
    def end(self) -> pd.Timestamp:  # end of data
        return pd.Timestamp('2023-12-31 00:00:00')

    def _read_dynamic(
            self,
            stations,
            dynamic_features,
            st=None,
            en=None) -> dict:

        st, en = self._check_length(st, en)
        features = validate_attributes(dynamic_features, self.dynamic_features)

        dyn = {stn: self._read_csv(stn).loc[st:en, features] for stn in stations}

        return dyn

    def transform_stn_coords(self, df:pd.DataFrame)->pd.DataFrame:

        ct_m = pd.DataFrame(columns=['lat', 'long'], index=df.index)
        # Test the function using lat, long in c DataFrame
        for i in range(0, len(df)):
            lat, lon = epsg25832_to_wgs84(df.iloc[i, 1], df.iloc[i, 0], 32)
            ct_m.iloc[i] = [lat, lon]
        
        return ct_m

    def transform_boundary(self, boundary):
        """
        Transforms the coordinates to the required format.
        """
        # from EPSG:25832 - ETRS89 / UTM zone 32N to WGS84

        assert len(boundary.coordinates) == 1  # only one polygon
        longs, lats = [], []
        for i in range(0, len(boundary.coordinates[0])):
            # assuming that coordinates in fiona.Geometry are in long, lat order
            lat_, long_ = epsg25832_to_wgs84(boundary.coordinates[0][i][0], boundary.coordinates[0][i][1], 32)
            longs.append(long_)
            lats.append(lat_)
        longs = np.array(longs)
        lats = np.array(lats)

        if fiona is not None:
            boundary = fiona.Geometry(type='Polygon', 
                                      coordinates=[list(zip(longs, lats))])
        return boundary


def _read_camels_ind_forcings(fpath: str) -> pd.DataFrame:
    """
    Reads one catchment mean forcing file of CAMELS_IND. A module level
    function, so that the process pool of :meth:`CAMELS_IND._read_dynamic`
    pickles a path instead of the dataset.
    """
    df = pd.read_csv(fpath)
    df.index = ymd_index(df.pop('year'), df.pop('month'), df.pop('day'))
    return df.astype(np.float32)


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
            from ..download_zenodo import download_from_zenodo
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


class CAMELS_FR(_RainfallRunoff):
    """
    Dataset of 654 catchments from France following the works of
    `Delaigue et al., 2025 <https://doi.org/10.5194/essd-17-1461-2025>`_.
    The dataset consists of 344 static catchment features and 22 dynamic features.
    The dynamic features span from 1970101 to 20211231 with daily timestep.

    This is release 3.2 of the `Recherche Data Gouv record
    <https://doi.org/10.57745/WH7FJR>`_. All releases hold the same 654 stations,
    22 dynamic features, 344 static features and time span; release 3.0 corrected
    ``hym_q_questionable``, ``hym_q_unqualified`` and ``hym_q_anomaly_inrae``
    (percentages of streamflow values flagged as doubtful), which is the only
    difference in the data. Data downloaded by an earlier version of aqua_fetch is
    release 2.1; it is detected at initialization, and only its 9.4 MB attributes
    archive is downloaded again.

    Not provided: the monthly and yearly aggregates of the time series archive.

    Timings on a 48-core machine: the first initialization downloads 372 MB and
    builds a 2.2 GB netCDF cache, for which it reads the 654 daily csv files in
    18 s. An initialization that only upgrades release 2.1 takes 5 s. Afterwards
    initialization takes 0.5 s, fetching all 654 stations with all 22 dynamic
    features 0.15 s and with all 344 static features 0.45 s.

    Examples
    ---------
    >>> from aqua_fetch import CAMELS_FR
    >>> dataset = CAMELS_FR()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='J421191001', as_dataframe=True)
    >>> df = dynamic['J421191001'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (18993, 22)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       654
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (65 out of 654)
       65
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(18993, 22), (18993, 22), (18993, 22),... (18993, 22), (18993, 22)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('J421191001', as_dataframe=True,
    ...  dynamic_features=['pcp_mm', 'spechum_gkg', 'airtemp_C_mean', 'pet_mm_pm', 'q_cms_obs'])
    >>> dynamic['J421191001'].shape
       (18993, 5)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='J421191001', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['J421191001'].shape
    ((1, 344), 1, (18993, 22))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 18993, 'dynamic_features': 22})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (654, 2)
    >>> dataset.stn_coords('J421191001')  # returns coordinates of station whose id is J421191001
        48.006298   -4.063848
    >>> dataset.stn_coords(['J421191001', '802'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('J421191001')
    # get coordinates of two stations
    >>> dataset.area(['J421191001', '802'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('J421191001')
    """
    # file ids of the Recherche Data Gouv record doi:10.57745/WH7FJR, release 3.2.
    # Only the attributes archive and the README differ from release 2.1; the time
    # series, geography, licenses and description files are the same files.
    _DATAFILE = "https://entrepot.recherche.data.gouv.fr/api/access/datafile/"
    url = {
        "ADDITIONAL_LICENSES.zip": f"{_DATAFILE}343463",
        "CAMELS_FR_attributes.zip": f"{_DATAFILE}621683",
        'CAMELS_FR_geography.zip': f"{_DATAFILE}343465",
        'CAMELS_FR_time_series.zip': f"{_DATAFILE}343470",
        'README.md': f"{_DATAFILE}621685",
        'NEWS.md': f"{_DATAFILE}621689",
        'CAMELS-FR_description.ods': f"{_DATAFILE}348740",
    }

    # files of CAMELS_FR_attributes.zip and CAMELS_FR_geography.zip, from the
    # "File hierarchy convention" section of the README of the dataset. Used by
    # _check_manifest, so that an incomplete extraction is reported instead of
    # being taken for a smaller dataset.
    _STATIC_ATTR_FILES = (
        "00_description_geology_classes.txt",
        "00_description_land_cover_classes.txt",
        "CAMELS_FR_geology_attributes.csv",
        "CAMELS_FR_human_influences_dams.csv",
        "CAMELS_FR_hydrogeology_attributes.csv",
        "CAMELS_FR_land_cover_attributes.csv",
        "CAMELS_FR_site_general_attributes.csv",
        "CAMELS_FR_soil_general_attributes.csv",
        "CAMELS_FR_soil_quantiles_attributes.csv",
        "CAMELS_FR_station_general_attributes.csv",
        "CAMELS_FR_topography_general_attributes.csv",
        "CAMELS_FR_topography_quantiles_attributes.csv",
    )
    _TS_STAT_FILES = (
        "CAMELS_FR_climatic_statistics.csv",
        "CAMELS_FR_hydroclimatic_quantiles.csv",
        "CAMELS_FR_hydroclimatic_regimes_daily.csv",
        "CAMELS_FR_hydroclimatic_statistics_joint_availability_yearly.csv",
        "CAMELS_FR_hydroclimatic_statistics_timeseries_yearly.csv",
        "CAMELS_FR_hydrological_signatures.csv",
        "CAMELS_FR_hydrometry_statistics.csv",
    )
    _GEOG_FILES = (
        "CAMELS_FR_catchment_boundaries.gpkg",
        "CAMELS_FR_catchment_nestedness_information.csv",
        "CAMELS_FR_gauge_outlet.gpkg",
    )

    # size in bytes of CAMELS_FR_hydrometry_statistics.csv in release 2.1, which
    # aqua_fetch downloaded until now. Release 3.0 recomputed hym_q_questionable,
    # hym_q_unqualified and hym_q_anomaly_inrae, making this the only file of the
    # attributes archive whose content, and size, changed.
    _V21_HYDROMETRY_BYTES = 38976

    def __init__(self,
                 path=None,
                 overwrite=False,
                 **kwargs):
        """
        Parameters
        ----------
        path : str
            directory under which the data is (or will be) saved in a
            ``CAMELS_FR`` folder. If None, the default data directory of
            aqua_fetch is used.
        overwrite : bool
            if True, the archives, the extracted folders and the netCDF caches
            are deleted and downloaded/built again.
        **kwargs :
            any keyword argument of :py:class:`aqua_fetch.rr._RainfallRunoff`
            such as ``processes``, ``verbosity``, ``to_netcdf`` or ``remove_zip``.
        """
        super().__init__(path=path, overwrite=overwrite, **kwargs)

        self._download_camels_fr(overwrite=overwrite)

        self._stations = self.__stations()

        self._check_manifest()

        self._static_features = list(set(self._static_data().columns.to_list()))

        self._dynamic_features = self._read_stn_dyn(self.stations()[0]).columns.to_list()

        # if self.to_netcdf:
        self._maybe_to_netcdf()

    def _download_camels_fr(self, overwrite: bool = False):
        """
        Downloads and extracts only those archives whose extracted folder does
        not exist, so that an archive deleted after extraction
        (``remove_zip=True``) is not downloaded again, and downloads the plain
        files that are missing.

        When the attributes on disk are those of the superseded release 2.1 (see
        :meth:`_stale_attributes`), the 9.4 MB attributes archive and the README
        are downloaded again so that an existing installation is brought to
        release 3.2. Nothing else is touched, because the 361 MB time series
        archive, the geography archive and the netCDF cache, which holds dynamic
        data only, are the same in both releases.

        ``overwrite=True`` first deletes this dataset's archives, extracted
        folders, plain files and netCDF caches.
        """
        os.makedirs(self.path, exist_ok=True)

        archives = {fname: link for fname, link in self.url.items() if fname.endswith('.zip')}
        plain = {fname: link for fname, link in self.url.items() if not fname.endswith('.zip')}
        # each archive holds a folder of its own name, so CAMELS_FR_attributes.zip
        # is extracted to path/CAMELS_FR_attributes/CAMELS_FR_attributes/
        folder_of = {fname: os.path.join(self.path, fname[:-len('.zip')]) for fname in archives}

        if overwrite:
            caches = glob.glob(os.path.join(glob.escape(self.path),
                                            f"{self.name.lower()}_{self.timestep}*.nc"))
            _remove_stale([*(os.path.join(self.path, fname) for fname in self.url),
                           *folder_of.values(), *caches], self.verbosity)
        elif self._stale_attributes():
            # unconditional, because this silently changes the values that
            # static_features returns between two runs of the same code
            warnings.warn(
                f"The CAMELS-FR attributes in {self.path} are those of release 2.1, "
                "in which hym_q_questionable, hym_q_unqualified and "
                "hym_q_anomaly_inrae were miscalculated. Downloading the 9.4 MB "
                "attributes archive of release 3.2 to replace them; the time "
                "series, the boundaries and the netCDF cache are unaffected.",
                UserWarning)
            _remove_stale([folder_of['CAMELS_FR_attributes.zip'],
                           os.path.join(self.path, 'CAMELS_FR_attributes.zip'),
                           os.path.join(self.path, 'README.md')],
                          self.verbosity, reason="superseded release 2.1")

        for fname, link in archives.items():
            folder = folder_of[fname]
            if os.path.exists(folder):
                continue

            archive = os.path.join(self.path, fname)
            if not os.path.exists(archive):
                if self.verbosity:
                    print(f"downloading {link} to {archive}")
                download(link, outdir=self.path, fname=fname, verbosity=self.verbosity)

            # extracted into a temporary folder that is renamed once complete, so
            # that an interrupted extraction is redone instead of being taken as
            # complete at the next initialization
            if self.verbosity:
                print(f"extracting {archive}")
            partial = f"{folder}_extracting"
            shutil.rmtree(partial, ignore_errors=True)
            try:
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(partial)
            except (zipfile.BadZipFile, zlib.error, EOFError):  # e.g. an error page saved as the archive
                shutil.rmtree(partial, ignore_errors=True)
                os.remove(archive)
                raise ValueError(f"{archive} is corrupt and was deleted. "
                                 f"Initialize CAMELS_FR again to download it again.") from None
            os.replace(partial, folder)

        for fname, link in plain.items():
            fpath = os.path.join(self.path, fname)
            if not os.path.exists(fpath):
                if self.verbosity:
                    print(f"downloading {link} to {fpath}")
                download(link, outdir=self.path, fname=fname, verbosity=self.verbosity)

        if self.remove_zip:
            for fname in archives:
                archive = os.path.join(self.path, fname)
                if os.path.exists(archive):
                    if self.verbosity:
                        print(f"remove_zip=True: removing {archive}")
                    os.remove(archive)
        return

    @property
    def _hydrometry_file(self) -> os.PathLike:
        """the only file whose content differs between releases 2.1 and 3.2"""
        return os.path.join(self.ts_stat_path, "CAMELS_FR_hydrometry_statistics.csv")

    def _stale_attributes(self) -> bool:
        """
        Whether the extracted attributes are those of release 2.1.

        The two releases differ in one file only, whose size is 38976 bytes in
        2.1 and 39467 bytes in 3.2, so one ``stat`` call tells them apart without
        reading the file or contacting the server.
        """
        fpath = self._hydrometry_file
        return os.path.exists(fpath) and os.path.getsize(fpath) == self._V21_HYDROMETRY_BYTES

    def _check_manifest(self):
        """
        warns if files that the README of the dataset lists are missing, e.g.
        left out by an interrupted extraction or deleted by hand
        """
        files = [os.path.join(self.static_attr_path, fname) for fname in self._STATIC_ATTR_FILES]
        files += [os.path.join(self.ts_stat_path, fname) for fname in self._TS_STAT_FILES]
        files += [os.path.join(self.geog_path, fname) for fname in self._GEOG_FILES]
        missing = [fpath for fpath in files if not os.path.exists(fpath)]

        n_daily = len(glob.glob(os.path.join(glob.escape(self.daily_ts_path),
                                             "CAMELS_FR_tsd_*.csv")))
        if n_daily != len(self._stations):
            missing.append(f"{len(self._stations) - n_daily} of the {len(self._stations)} "
                           f"daily time series files in {self.daily_ts_path}")

        if missing:
            warnings.warn(
                f"CAMELS_FR: {len(missing)} expected files are missing: {missing}. "
                f"Use overwrite=True to download them again.", UserWarning)
        return

    @property
    def boundary_file(self) -> os.PathLike:        
        return os.path.join(
            self.path, 
            'CAMELS_FR_geography', 
            'CAMELS_FR_geography', 
            'CAMELS_FR_catchment_boundaries.gpkg')

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'hyd_slope_fdc': slope(''),
                # 'sit_latitude', 'sit_longitude', todo : what is difference between site and guage lat/lon?
                # gauge latitude/longitude in WGS84 (Hydroportail coordinates)
                'sta_x_w84': gauge_longitude(),
                'sta_y_w84': gauge_latitude(),
                # todo: should we use sta_x_w84_snap and sta_y_w84_snap which are 
                # gauge longitude in WGS84 (INRAE's own estimation, snapped on thorical river network)
                'sit_area_topo': catchment_area(),
        }

    @property
    def dyn_map(self)->Dict[str, str]:
        return {
            # streamflow in liters per second
            'tsd_q_l': observed_streamflow_cms(),
            # streamflow in milimeters per day
            'tsd_q_mm': observed_streamflow_mm(),
            'tsd_wind': mean_windspeed(),
            'tsd_temp_min': min_air_temp(),  # minimum air temperature over the period (18h day-1, 18h day]
            'tsd_temp_max': max_air_temp(),  # maximum air temperature over the period (18h day-1, 18h day]
            'tsd_temp': mean_air_temp(),  # mean air temperature over the period (18h day-1, 18h day]
            # short wave visible radiation over the period (0h day, 0h day+1]
            'tsd_rad_ssi': solar_radiation(),  # J cm-2 day-1 -> W m-2 in dyn_factors
            # long wave atmospheric radiation over the period (0h day, 0h day+1]
            'tsd_rad_dli': downward_longwave_radiation(),  # J cm-2 day-1 -> W m-2 in dyn_factors
            # specific air humidity over the period (0h day, 0h day+1]
            'tsd_humid': mean_specific_humidity(),
            # PET over the period (0h day, 0h day+1] (Penman-Monteith method with a modified albedo when snow lies on the ground)
            'tsd_pet_pm': total_potential_evapotranspiration_with_specifier('pm'),
            'tsd_pet_pe': total_potential_evapotranspiration_with_specifier('pe'),
            'tsd_pet_ou': total_potential_evapotranspiration_with_specifier('ou'),
            # total precipitation (liquid + solid) over the period (6h day, 6h day+1]
            'tsd_prec': total_precipitation(),
            # solid fraction of precipitation over the period (6h day, 6h day+1]
            'tsd_prec_solid_frac': total_precipitation_with_specifier('solfrac'),  # todo : check its units?
        }

    @property
    def dyn_factors(self) -> Dict[str, float]:
        # ``CAMELS-FR_description.ods`` gives both radiation series in J cm-2
        # accumulated over the day, while the canonical names promise W m-2.
        return {
            solar_radiation(): J_CM2_DAY_TO_WM2,
            downward_longwave_radiation(): J_CM2_DAY_TO_WM2,
        }

    @property
    def daily_ts_path(self) -> os.PathLike:
        return os.path.join(self.path, "CAMELS_FR_time_series", "CAMELS_FR_time_series", "daily")

    @property
    def attr_path(self) -> os.PathLike:
        return os.path.join(self.path, "CAMELS_FR_attributes", "CAMELS_FR_attributes")

    @property
    def static_attr_path(self) -> os.PathLike:
        return os.path.join(self.attr_path, "static_attributes")

    @property
    def ts_stat_path(self) -> os.PathLike:
        return os.path.join(self.attr_path, "time_series_statistics")

    @property
    def static_features(self) -> List[str]:
        """returns static features for Denmark catchments"""
        return self._static_features

    @property
    def dynamic_features(self) -> List[str]:
        """returns names of dynamic features"""
        return self._dynamic_features

    @property
    def start(self) -> pd.Timestamp:  # start of data
        return pd.Timestamp('1970-01-01')

    @property
    def end(self) -> pd.Timestamp:  # end of data
        return pd.Timestamp('2021-12-31')

    def __stations(self) -> List[str]:
        return pd.read_csv(os.path.join(
            self.static_attr_path,
            "CAMELS_FR_human_influences_dams.csv"),
            sep=";",
            index_col=0).index.to_list()

    def stations(self) -> List[str]:
        return self._stations

    @property
    def geog_path(self) -> os.PathLike:
        return os.path.join(self.path, "CAMELS_FR_geography", "CAMELS_FR_geography")

    def static_attrs(self) -> pd.DataFrame:
        """
        combination of topographic + soil + landuse + geology + climate + hydro
        + climate + anthropogenic features

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of static features of all catchments of shape (654, xxxx)
        """
        files = glob.glob(f"{self.static_attr_path}/*.csv")
        dfs = []
        for f in files:
            df = pd.read_csv(f, sep=";", index_col=0)
            df.index = df.index.astype(str)
            if len(df) == 654:
                dfs.append(df)
            elif self.verbosity > 1:
                print(f"skipping {os.path.basename(f)} as it has {len(df)} rows")

        static_attrs = pd.concat(dfs, axis=1)

        gen_attrs = pd.read_csv(
            os.path.join(self.static_attr_path, "CAMELS_FR_site_general_attributes.csv"),
            sep=";",
            index_col=0,
        )

        # in gen_attrs the stn_id has lenght of 8 while in static_attrs it is 10
        # so adding the last two digits to the gen_attrs
        _map = {stn[0:-2]: stn for stn in static_attrs.index}
        gen_attrs = gen_attrs.rename(index=_map)

        if self.verbosity:
            for stn in static_attrs.index:
                if stn not in gen_attrs.index:
                    print(stn, " not found in site_general_attributes.csv")

        static_attrs = pd.concat([gen_attrs, static_attrs], axis=1)
        return static_attrs

    def ts_attrs(self) -> pd.DataFrame:
        """
        daily_timeseries statistics of all catchments

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of static features of all catchments of shape (654, xxxx)
        """
        files = glob.glob(f"{self.ts_stat_path}/*.csv")
        dfs = []
        for f in files:
            df = pd.read_csv(f, sep=";", index_col=0)
            df.index = df.index.astype(str)
            if len(df) == 654:
                dfs.append(df)
            elif self.verbosity > 1:
                print(f"skipping {os.path.basename(f)} as it has {len(df)} rows")

        return pd.concat(dfs, axis=1)

    def _static_data(self) -> pd.DataFrame:
        """
        static attributes plus timeseries statistics

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of static features of all catchments of shape (654, xxxx)
        """

        static_data = pd.concat([
            self.static_attrs(),
            self.ts_attrs()
        ], axis=1)
        # remove duplicated columns
        df = static_data.loc[:, ~static_data.columns.duplicated()].copy()

        df.rename(columns=self.static_map, inplace=True)

        return df

    def _read_stn_dyn(self, station: str) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(self.daily_ts_path, f"CAMELS_FR_tsd_{station}.csv"),
            sep=";",
            index_col=0,
            parse_dates=True,
            comment="#",
        )

        df.rename(columns=self.dyn_map, inplace=True)

        self._apply_dyn_factors(df)

        return df


class CAMELS_SPAT(_RainfallRunoff):
    """
    Dataset of 1426 catchments from North America (USA and Canada) following the works of
    `Knoben et al., 2025 <https://doi.org/10.5194/egusphere-2025-893>`_.
    """
    stn_name = 'USA_14141500'
    time_step = 'obs-hourly'  # or obs_daily
    scale = 'macro-scale' # or headwater or meso-scale
    data_type = 'observations'  # or 
    url = {
        f'https://www.frdr-dfdr.ca/repo/files/1/published/publication_1211/submitted_data/{data_type}/{scale}/{time_step}/{stn_name}_hourly_flow_observations.nc',

        }


class CAMELS_NZ(_RainfallRunoff):
    """
    Dataset of 369 catchments from New Zealand following the works of
    `Harrigan et al., 2025 <https://doi.org/10.5194/essd-2025-244>`_.
    The dataset consists of 40 static catchment features and 5 dynamic features.
    The dynamic features span from 19720101 to 20240802 with hourly timestep.
    The data is downloaded from `figshare <https://doi.org/10.26021/canterburynz.28827644>`_.
    This data comes with daily and hourly timesteps and the each can be accessed by
    specifying value of `tiemstep` argument to ``D`` or ``H`` respectively during 
    initialization.
    
    Examples
    ---------
    >>> from aqua_fetch import CAMELS_NZ
    >>> dataset = CAMELS_NZ()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='74321', as_dataframe=True)
    >>> df = dynamic['74321'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (19208, 5)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       369
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (36 out of 369)
       36
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(19208, 5), (19208, 5), (19208, 5),... (19208, 5), (19208, 5)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('74321', as_dataframe=True,
    ...  dynamic_features=['pcp_mm', 'rh_%', 'airtemp_C_mean', 'pet_mm', 'q_cms_obs'])
    >>> dynamic['74321'].shape
       (19208, 4)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='74321', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['74321'].shape
    ((1, 40), 1, (19208, 5))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 19208, 'dynamic_features': 5})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (369, 2)
    >>> dataset.stn_coords('74321')  # returns coordinates of station whose id is 74321
        -45.945599      170.101486
    >>> dataset.stn_coords(['74321', '802'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('74321')
    # get coordinates of two stations
    >>> dataset.area(['74321', '802'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('74321')
    # The hourly data can be accessed by specifyng the timestep to 'H'
    >>> dataset = CAMELS_NZ(timestep='H')
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='74321', as_dataframe=True)
    >>> df = dynamic['74321'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (460978, 5)    
    """
    url = "https://figshare.canterbury.ac.nz/ndownloader/articles/28827644/versions/2"

    def __init__(self,
                 path:Union[str, os.PathLike]=None,
                 **kwargs):

        super().__init__(name="CAMELS_NZ", path=path, **kwargs)

        if self.timestep == 'H':
            self.timestep_ = 'hourly'
        else:
            self.timestep_ = 'daily'

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        zip_path = os.path.join(self.path, 'camels_nz.zip')
        unzipped_dir = os.path.join(self.path, 'camels_nz')

        # Download only if neither the archive nor the extracted folder exists.
        # This way, deleting camels_nz.zip after extraction does not re-trigger
        # a download on subsequent class instantiation.
        if not (os.path.exists(zip_path) or os.path.exists(unzipped_dir)) and not self.overwrite:
            download(
                outdir=self.path,
                url=self.url,
                fname="camels_nz.zip",
                verbosity=self.verbosity,
            )

        # Outer extract: only if the unzipped folder is not already there.
        if not os.path.exists(unzipped_dir):
            unzip(self.path, verbosity=self.verbosity)

        # Inner extract: idempotent when no inner .zip files remain.
        if os.path.exists(unzipped_dir):
            unzip(unzipped_dir, verbosity=self.verbosity)

        # if self.to_netcdf:
        self._maybe_to_netcdf()

    @property
    def boundary_file(self)-> os.PathLike:
        return os.path.join(
            self.shapefile_path,
            "All_Nested_Catchments.shp"
        )

    @property
    def dyn_map(self)->Dict[str, str]:
        return {
            'flow': observed_streamflow_cms(),
            'temperature': mean_air_temp(),
            'Relative_humidity': mean_rel_hum(),
            'precipitation': total_precipitation(),
            'PET': total_potential_evapotranspiration(),
        }

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'latitude': gauge_latitude(),
                'longitude': gauge_longitude(),
                'uparea': catchment_area(),
                'elevation': gauge_elevation_meters(),
                'usAveSlope': slope('degrees')
        }
   
    @property
    def start(self) -> pd.Timestamp:
        return  pd.Timestamp('1972-01-01 00:00:00')
    
    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp('2024-08-02 09:00:00')

    @property
    def dynamic_features(self) -> List[str]:
        """returns names of dynamic features"""
        return [self.dyn_map[feature] for feature in self._path_map]
    
    @property
    def static_features(self) -> List[str]:
        """returns static features for New Zealand catchments"""
        return self._static_data(nrows=2).columns.to_list()

    def stations(self)->List[str]:
        fpath = os.path.join(self.static_path, '4.CAMELS_NZ_Geology.csv')
        df = pd.read_csv(fpath, index_col=0, usecols=[0, 1])
        return df.index.astype(str).tolist()

    @property
    def temp_path(self) -> os.PathLike:
        return os.path.join(self.path, 'camels_nz', f'CAMELS_NZ_{self.timestep_}_Temperature')
    
    @property
    def precip_path(self) -> os.PathLike:
        return os.path.join(self.path, 'camels_nz', f'CAMELS_NZ_{self.timestep_}_Precipitation')
    
    @property
    def q_path(self) -> os.PathLike:
        return os.path.join(self.path, 'camels_nz', f'CAMELS_NZ_{self.timestep_}_Streamflow')
    
    @property
    def shapefile_path(self) -> os.PathLike:
        return os.path.join(self.path, 'camels_nz', 'CAMELS_NZ_Shapefiles')
    
    @property
    def pet_path(self) -> os.PathLike:
        return os.path.join(self.path, 'camels_nz', f'CAMELS_NZ_{self.timestep_}_PET')

    @property
    def rh_path(self) -> os.PathLike:
        return os.path.join(self.path, 'camels_nz', f'CAMELS_NZ_{self.timestep_}_Relative_Humidity')

    @property
    def static_path(self) -> os.PathLike:
        return os.path.join(self.path, 'camels_nz', 'CAMELS_NZ_Catchment_Atrributes')

    def _redundant_after_consolidation(self) -> List[Tuple[str, str]]:
        # The five per-feature folders for each timestep become redundant once
        # the corresponding consolidated NetCDF (camels_nz_D_v2.nc / camels_nz_H_v2.nc)
        # exists. Both timesteps are listed regardless of self.timestep so that
        # cleanup works whichever instance the user invokes free_disk_space on.
        inner = os.path.join(self.path, 'camels_nz')
        name_lc = self.name.lower()
        pairs: List[Tuple[str, str]] = []
        for ts_code, ts_word in (("D", "daily"), ("H", "hourly")):
            cache = os.path.join(self.path, f"{name_lc}_{ts_code}.nc")
            for feat in ("Temperature", "Precipitation", "Streamflow",
                         "PET", "Relative_Humidity"):
                pairs.append(
                    (os.path.join(inner, f"CAMELS_NZ_{ts_word}_{feat}"), cache)
                )
        return pairs

    def _static_data(self, nrows:int = None) -> pd.DataFrame:
        """
        static attributes of catchments

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of static features of all catchments of shape (369, 40)
        """

        dfs = []
        idx = 0

        # read all .csv files in static_path
        for csv_file in glob.glob(os.path.join(self.static_path, '*.csv')):
            df = pd.read_csv(csv_file, index_col=0, nrows=nrows)

            df.index = df.index.astype(str)

            if idx > 0:
                df.drop(columns=['RID', 'StationName', 'latitude', 'longitude'], inplace=True, errors='ignore')

            dfs.append(df)

            idx += 1
        
        static_data = pd.concat(dfs, axis=1)

        static_data.rename(columns=self.static_map, inplace=True)

        return static_data
    
    @property
    def _nodata_stns(self):
        """data from following stations is not available. The corresponding files are empty."""
        return ['75253', "75261", "75265", "75276", "75294", "15408", 
                "15410", "15453", "33356", "52916", "74318", "74321", "1114629"]

    def _read_dynamic_para(
            self, 
            stations:Union[str, List[str]] = "all",
            para_name:str = "PET",
            )-> pd.DataFrame:
        """
        reads dynamic data for a given parameter for given stations.
        """
        assert para_name in list(self._path_map.keys())
        cpus = self.processes or min(get_cpus(), 32)

        stations = validate_attributes(stations, self.stations(), 'stations')

        start = time.time()

        if cpus == 1:
            q_dfs = []

            for _, stn in enumerate(stations):

                stn_q = self._read_stn_dyn_para(stn, para_name)
                q_dfs.append(stn_q)    

                if self.verbosity and _ % 100 == 0:
                    print(f"Read {len(q_dfs)} stations so far...")
        else:
            with cf.ProcessPoolExecutor(cpus) as executor:
                results = executor.map(
                    self._read_stn_dyn_para, 
                    stations, 
                    (para_name for _ in range(len(stations)))
                    )
            
            q_dfs = [stn_q for stn_q in results]

        total = time.time() -  start
        if self.verbosity:
            print(f"Read {len(q_dfs)} stations for {para_name} in {total:.2f} seconds with {cpus} cpus.")

        q_df = pd.concat(q_dfs, axis=1)
        return q_df
    
    @property
    def _path_map(self) -> Dict[str, os.PathLike]:
         return {
            'PET': self.pet_path,
            'precipitation': self.precip_path,
            'Relative_humidity': self.rh_path,
            'temperature': self.temp_path,
            'flow': self.q_path,
        }

    def _read_stn_dyn_para(self, stn:str, para_name:str) -> pd.Series:
        """
        read dynamic data for a given station and parameter.
        """
        stn_q = pd.Series(dtype=np.float32, name=stn)

        fname = {
            'Relative_humidity': 'RH'
        }
        if self.timestep == 'D':
            fpath = os.path.join(
                self._path_map[para_name],
                f'{self.timestep_}_{fname.get(para_name, para_name)}_station_id_{stn}.csv')
        else:
            fpath = os.path.join(
                self._path_map[para_name], 
                f'{fname.get(para_name, para_name)}_station_id_{stn}.csv')
        if os.path.exists(fpath):
            if para_name == 'flow' and stn in self._nodata_stns:
                return stn_q
                        
            try:
                stn_q = pd.read_csv(fpath, index_col=0, parse_dates=True, na_values=['NA  '])
            except pd.errors.EmptyDataError:
                warnings.warn(f"{para_name}_station_id_{stn}.csv is empty. Skipping station {stn}.")
                return stn_q

            if self.timestep == 'H':
                format = '%m/%d/%Y %H:%M'
                if para_name == 'flow' and stn == '57521':
                    format = '%d/%m/%Y %H:%M'            
            else:
                format = '%m/%d/%Y'
                if para_name == 'flow' and stn == '57521':
                    format = '%d/%m/%Y'

            stn_q.index = pd.to_datetime(stn_q.index, format=format)


            stn_q = stn_q[para_name].astype(np.float32).rename(stn)
        else:
            if self.verbosity>1:
                print(f"Warning: {para_name}_station_id_{stn}.csv does not exist. Skipping station {stn}.")
            stn_q = pd.Series(dtype=np.float32, name=stn)
        
        # remove rows with duplicated index, ideally there should not be any
        stn_q = stn_q[~stn_q.index.duplicated(keep='first')]

        return stn_q

    def _read_stn_dyn(self, stn:str)->pd.DataFrame:
        """
        reads dynamic data for a given station
        """
        stn_dfs = []
        for para in self._path_map.keys():

            stn_para = self._read_stn_dyn_para(stn, para)
            stn_dfs.append(stn_para.rename(para, inplace=True))
        stn_df = pd.concat(stn_dfs, axis=1)
        stn_df.index = pd.to_datetime(stn_df.index)

        # convert the temperature to Celcius from Kelvin
        stn_df['temperature'] = stn_df['temperature'] - 273.15

        stn_df.rename(columns=self.dyn_map, inplace=True)
        
        return stn_df
    

class CAMELS_COL(_RainfallRunoff):
    """
    Dataset of 347 catchments from Colombia following the works of
    `Jimenez et al., 2025 <https://doi.org/10.5194/essd-2025-200>`_.
    The dataset consists of 255 static catchment features and 6 dynamic features.
    The dynamic features span from 19810101 to 20221231 with daily timestep.
    The data is downloaded from `Zenodo <https://zenodo.org/records/15554735>`_.

    Examples
    ---------
    >>> from aqua_fetch import CAMELS_COL
    >>> dataset = CAMELS_COL()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='35067040', as_dataframe=True)
    >>> df = dynamic['35067040'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (15340, 6)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       347
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (34 out of 347)
       34
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(15340, 6), (15340, 6), (15340, 6),... (15340, 6), (15340, 6)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('35067040', as_dataframe=True,
    ...  dynamic_features=['pcp_mm', 'airtemp_C_mean', 'pet_mm', 'q_cms_obs'])
    >>> dynamic['35067040'].shape
       (15340, 4)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='35067040', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['35067040'].shape
    ((1, 255), 1, (15340, 6))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 15340, 'dynamic_features': 6})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (347, 2)
    >>> dataset.stn_coords('35067040')  # returns coordinates of station whose id is 35067040
        4.778274        -73.587807
    >>> dataset.stn_coords(['35067040', '21187030'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('35067040')
    # get coordinates of two stations
    >>> dataset.area(['35067040', '21187030'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('35067040')

    """
    url = "https://zenodo.org/records/15554735"

    def __init__(self,
                 path=None,
                 overwrite=False,
                 to_netcdf: bool = True,
                 **kwargs):

        super(CAMELS_COL, self).__init__(
            path=path, 
            to_netcdf=to_netcdf, 
            **kwargs)

        self._download(overwrite=overwrite)

        # if self.to_netcdf:
        self._maybe_to_netcdf()

        self.bbox = {'llcrnrlat': -5.0, 'urcrnrlat': 15.0, 'llcrnrlon': -80.0, 'urcrnrlon': -65.0}
        self.parallels = range(-5, 15, 5)
        self.meridians = range(5, 6, 1),

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(
            self.path,
            "03_CAMELS_COL_Basin_boundary",
            "03_CAMELS_COL_Basin_boundary",
            "CAMELS_COL_catchments_boundaries.shp"
        )

    @property
    def dyn_map(self) -> Dict[str, str]:
        """
        dynamic features map for CAMELS-LUX catchments
        """
        return {
            'streamflow': observed_streamflow_cms(),
            'pr': total_precipitation(),  # CHIRPS V2
            't_mean': mean_air_temp(),
            't_min': min_air_temp(),
            't_max': max_air_temp(),
            'poten_evapo': total_potential_evapotranspiration(),
        }

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'gauge_lat': gauge_latitude(),
                'gauge_lon': gauge_longitude(),
                'area': catchment_area(),
                'gauge_elev': gauge_elevation_meters(),
                'perimeter': catchment_perimeter(),
        }

    def transform_boundary(self, boundary):
        """
        Transforms a catchment boundary from WGS 84 / World Mercator
        (EPSG:3395, the CRS of the shapefile) to WGS84 (EPSG:4326) lon/lat, so
        that it matches the gauge coordinates.

        Uses the pyproj-free :func:`world_mercator_to_wgs84` helper. Verified
        against pyproj (EPSG:3395 -> EPSG:4326) on all catchments: the
        per-vertex error is below 0.01 mm. The 70 MultiPolygons and the 21
        Polygons with interior rings (holes) are handled, and the geometry type
        and ring structure are kept. The conversion is vectorised per ring.
        """
        if fiona is None:
            return boundary

        def _ring_to_wgs84(ring):
            arr = np.asarray(ring, dtype=float)
            # fiona stores each vertex as (x=easting, y=northing[, z]); output
            # is (lon, lat) to keep the (x, y) ordering of the geometry.
            lat, long = world_mercator_to_wgs84(arr[:, 0], arr[:, 1])
            return list(zip(long.tolist(), lat.tolist()))

        if boundary.type == 'MultiPolygon':
            coords = [[_ring_to_wgs84(ring) for ring in polygon]
                      for polygon in boundary.coordinates]
        else:  # Polygon, possibly with interior rings (holes)
            coords = [_ring_to_wgs84(ring) for ring in boundary.coordinates]

        return fiona.Geometry(type=boundary.type, coordinates=coords)

    @property
    def ts_path(self) -> os.PathLike:
        return os.path.join(
            self.path,
            "04_CAMELS_COL_Hydrometeorological_data",
            "04_CAMELS_COL_Hydrometeorological_data",
        )

    def stations(self) -> List[str]:
        return [fname[14:22] for fname in os.listdir(self.ts_path)]

    @property
    def dynamic_features(self) -> List[str]:
        df = self._read_stn_dyn(self.stations()[0], nrows=2)
        return df.columns.to_list()

    @property
    def static_features(self) -> List[str]:  # todo : calling this method again and again can be slow
        """
        returns static features for Colombia catchments
        """
        df = self._static_data()
        return df.columns.to_list()

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp('1981-01-01')
    
    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp('2022-12-31')

    def _read_stn_dyn(self, stn:str, nrows=None)->pd.DataFrame:
        """
        reads dynamic data for a given station
        """
        stn_df = pd.read_csv(
            os.path.join(self.ts_path, f"Hydromet_data_{stn}.txt.txt"), 
            sep='\t',
            index_col=0, 
            parse_dates=True,
            nrows=nrows,
            )
        
        stn_df.index = pd.to_datetime(stn_df.index)

        if stn_df.index.has_duplicates:
            warnings.warn(f"{stn} has duplicated index. Removing duplicates.")
        
        stn_df.rename(columns=self.dyn_map, inplace=True)
        
        return stn_df

    def _soil_data(self) -> pd.DataFrame:
        """
        reads 07_CAMELS_COL_Soil_characteristics.xlsx file
        """
        df = pd.read_excel(
            os.path.join(self.path, "07_CAMELS_COL_Soil_characteristics.xlsx"),
            index_col=0,
            dtype={0: str},
        ).T

        df.index = [name.split('_')[1] for name in df.index]

        return df

    def _lc_data(self) -> pd.DataFrame:
        """
        reads 06_CAMELS_COL_Land_cover_characteristics.xlsx file
        """
        df = pd.read_excel(
            os.path.join(self.path, "06_CAMELS_COL_Land_cover_characteristics.xlsx"),
            index_col=0,
            dtype={0: str},
        ).T

        df.index = [name.split('_')[1] for name in df.index]

        df = df.dropna(axis=1, how='all')

        return df
    
    def _geol_data(self) -> pd.DataFrame:
        """
        reads 05_CAMELS_COL_Geology_characteristics.xlsx file
        """
        df = pd.read_excel(
            os.path.join(self.path, "05_CAMELS_COL_Geologic_characteristics.xlsx"),
            index_col=0,
            dtype={0: str},
            usecols="D:MM",
        ).T

        df.index = [name.split('_')[1] for name in df.index]

        df = df.dropna(axis=1, how='all')
        return df

    def _static_data(self) -> pd.DataFrame:
        """
        static attributes of catchments

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of static features of all catchments of shape (347, 255)
        """

        dfs = []
        idx = 0

        # read all .csv files
        for xlsx_file in [
            '02_CAMELS_COL_Catchment_information',
            '08_CAMELS_COL_Climatic_indices', 
            '09_CAMELS_COL_Hydrological_signatures', 
            '10_CAMELS_COL_Physiograpic_characteristics']:

            #if not  csv_file.endswith('basin_id.csv'):
            df = pd.read_excel(os.path.join(self.path, f"{xlsx_file}.xlsx"), 
                               index_col=0, dtype={0: str})

            df.index = df.index.astype(str)

            dfs.append(df)

            idx += 1
        
        static_data = pd.concat(dfs, axis=1)

        soil = self._soil_data()
        lc = self._lc_data()
        geol = self._geol_data()

        static_data = pd.concat([static_data, soil, lc, geol], axis=1)

        static_data.rename(columns=self.static_map, inplace=True)

        for col, fac in  self.static_factors.items():
            if col in static_data.columns:
                static_data[col] *= fac

        # the file gives the gauge position in EPSG:3395 meters: the column
        # named gauge_lat holds the northing and gauge_lon the easting
        lat, lon = world_mercator_to_wgs84(
            static_data[gauge_longitude()].values.astype(float),
            static_data[gauge_latitude()].values.astype(float))
        static_data[gauge_latitude()] = lat
        static_data[gauge_longitude()] = lon

        return static_data    


class CAMELS_SK(_RainfallRunoff):
    """
    Dataset of 178 catchments from South Korea following the work of 
    `Kim et al., 2025 <https://doi.org/10.5281/zenodo.15073263>`_.
    The dataset consists of 215 static catchment features and 17 dynamic features.
    The dynamic features span from 20000101 to 20191231 with hourly timestep.

    Examples
    ---------
    >>> from aqua_fetch import CAMELS_SK
    >>> dataset = CAMELS_SK()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='2013615', as_dataframe=True)
    >>> df = dynamic['2013615'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (175320, 17)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       178
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (17 out of 178)
       17
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(175320, 17), (175320, 17), (175320, 17),... (175320, 17), (175320, 17)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('2013615', as_dataframe=True,
    ...  dynamic_features=['total_precipitation', 'snow_depth', 'air_temp_obs', 'potential_evaporation', 'q_cms_obs'])
    >>> dynamic['2013615'].shape
       (175320, 17)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='2013615', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['2013615'].shape
    ((1, 215), 1, (175320, 17))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 175320, 'dynamic_features': 17})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (178, 2)
    >>> dataset.stn_coords('2013615')  # returns coordinates of station whose id is 2013615
        35.880798       128.173096
    >>> dataset.stn_coords(['2013615', '2017620'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('2013615')
    # get coordinates of two stations
    >>> dataset.area(['2013615', '2017620'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('2013615')
    
    """

    url = "https://zenodo.org/records/15073264"

    def __init__(self,
                 path=None,
                 timestep:str = "H",
                 to_netcdf: bool = True,
                 **kwargs):
 
        super(CAMELS_SK, self).__init__(
            path=path,  
            timestep=timestep,
            to_netcdf=to_netcdf, 
            **kwargs)
        
        self._download(overwrite=self.overwrite)

        # if self.to_netcdf:
        self._maybe_to_netcdf()
        
        self._unzip_7z_files()

    @property
    def boundary_file(self) -> Union[str, os.PathLike]:
        return os.path.join(
            self.path,
            "shp",
            "KFM_bas.shp"
        )

    @property
    def start(self) -> pd.Timestamp:
        """
        start of data
        """
        return pd.Timestamp('2000-01-01')
    
    @property
    def end(self) -> pd.Timestamp:
        """
        end of data
        """
        return pd.Timestamp('2019-12-31 23:59:59')

    @property
    def ts_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.path, "timeseries", "timeseries")

    @property
    def static_features(self) -> List[str]:
        return self._static_data(nrows=2).columns.to_list()
    
    @property
    def dynamic_features(self) -> List[str]:
        return self._read_stn_dyn(self.stations()[0], nrows=2).columns.to_list()

    def stations(self) -> List[str]:
        """
        returns names of stations as a list
        """
        return [fname.split('.')[0] for fname in os.listdir(self.ts_path)]

    @property
    def static_map(self) -> Dict[str, str]:
        return {
            'Area': catchment_area(),
            'Area_HydroATLAS': catchment_area_with_specifier('hydroatlas'),
            'Lon_gage': gauge_longitude(),
            'Lat_gage': gauge_latitude(),
            }

    @property
    def dyn_map(self) -> Dict[str, str]:
        """
        dynamic features map for CAMELS-SK catchments
        """
        return {
            # todo not sure about the units of these features
            # 'total_precipitation': total_precipitation(),
            # 'temperature_2m': mean_air_temp_with_specifier('2m'),
            # 'dewpoint_temperature_2m': mean_dewpoint_temperature_at_2m(),
            # 'potential_evaporation': total_potential_evapotranspiration(),
            # 'u_component_of_wind_10m': u_component_of_wind(),
            # 'v_component_of_wind_10m': v_component_of_wind(),
            #
            # surface_net_solar_radiation / surface_net_thermal_radiation are
            # deliberately NOT mapped, and no dyn_factor can fix them. They are
            # ERA5-Land RUNNING ACCUMULATIONS in J m-2 carried on hourly rows:
            # the value climbs through the day, resets mid-series, and then sits
            # flat overnight holding the previous total. For 2010-06-21 the
            # series runs 14.9e6 (00:00-05:00, flat) -> 17.6e6 (09:00) ->
            # 1.96e6 (10:00, reset) -> 18.4e6 (20:00) -> flat to midnight.
            # Recovering a per-timestep flux needs differencing consecutive
            # steps and handling the reset -- a derivation, not a unit
            # conversion -- so it is left to the user rather than guessed at.
            # Mapping it to swnetrad_wm2 would assert W m-2 for a J m-2
            # accumulation, which is simply false.
            # 'surface_net_solar_radiation': solar_radiation(),
            # 'air_temp_obs': mean_air_temp(),
            # 'precip_obs': total_precipitation(),
            # 'wind_sp_obs': mean_windspeed(),
            'streamflow': observed_streamflow_cms(),
        }

    def _unzip_7z_files(self):
        # The attributes file is .7z file
        try:
            import py7zr
        except (ModuleNotFoundError, ImportError):
            raise ImportError('py7zr is required to extract the .7z files. Please install it using `pip install py7zr`')

        if not os.path.exists(self.boundary_file):

            fpath = os.path.join(self.path, 'shp.7z')
            with py7zr.SevenZipFile(fpath, mode='r') as z:
                z.extractall(path = self.path)
                if self.verbosity:
                    print(f'Extracted {fpath}')
        return

    def _read_stn_dyn(self, stn:str, nrows=None)->pd.DataFrame:
        """
        reads dynamic data for a given station
        """
        stn_df = pd.read_csv(
            os.path.join(self.ts_path, f"{stn}.csv"),
            index_col=0, 
            parse_dates=True,
            nrows=nrows,
            )
        
        stn_df.index = pd.to_datetime(stn_df.index)

        if stn_df.index.has_duplicates:
            warnings.warn(f"{stn} has duplicated index. Removing duplicates.")
        
        stn_df.rename(columns=self.dyn_map, inplace=True)
        
        return stn_df

    def _static_data(self, nrows:int = None) -> pd.DataFrame:
        """
        static attributes of catchments

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of static features of all catchments of shape (178, 239)
        """

        dfs = []
        idx = 0

        # read all .csv files in static_path
        for csv_file in glob.glob(os.path.join(self.path, '*.csv')):
            df = pd.read_csv(csv_file, index_col=0, nrows=nrows)

            df.index = df.index.astype(str)

            if df.index.duplicated().any():
                if self.verbosity > 1:
                    print(f"Warning: Duplicated indices found in {csv_file}. Dropping duplicates.")
                df = df.drop_duplicates()

            if csv_file.endswith('HydroATLAS.csv'):
                val_stns = [stn for stn in df.index if stn in self.stations()]
                df = df.loc[val_stns, :]

            dfs.append(df)

            idx += 1
        
        static_data = pd.concat(dfs, axis=1)

        # remove duplicated columns
        static_data = static_data.loc[:, ~static_data.columns.duplicated()].copy()

        static_data.rename(columns=self.static_map, inplace=True)

        return static_data    


class CAMELS_LUX(_RainfallRunoff):
    """
    Dataset of 56 catchments from Luxembourg following the work of
    `Nijzink et al., 2025 <https://doi.org/10.5194/essd-2024-482>`_.
    The dataset consists of 61 static catchment features and 25 dynamic features.
    The dynamic features span from 20040101 to 20211231 with daily, hourly, and 15-minute timesteps.
    The data is downloaded from `Zenodo <https://zenodo.org/records/14910359>`_.

    Examples
    ---------
    >>> from aqua_fetch import CAMELS_LUX
    >>> dataset = CAMELS_LUX()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='ID_02', as_dataframe=True)
    >>> df = dynamic['ID_02'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (6209, 25)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       56
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (5)
       5
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(6209, 25), (6209, 25), (6209, 25),... (6209, 25), (6209, 25)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('ID_02', as_dataframe=True,
    ...  dynamic_features=['pcp_mm_station', 'rh_%', 'airtemp_C_mean', 'pet_mm_pm', 'q_cms_obs'])
    >>> dynamic['ID_02'].shape
       (6209, 5)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='ID_02', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['ID_02'].shape
    ((1, 61), 1, (6209, 25))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 6209, 'dynamic_features': 25})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (56, 2)
    >>> dataset.stn_coords('ID_02')  # returns coordinates of station whose id is ID_02
        49.586288       6.14908
    >>> dataset.stn_coords(['ID_02', 'ID_01'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('ID_02')
    # get coordinates of two stations
    >>> dataset.area(['ID_02', 'ID_01'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('ID_02')
    ...
    # if we want to get hourly data we can do as below
    >>> dataset = CAMELS_LUX(timestep='H')
    >>> _, dynamic = dataset.fetch(stations='ID_02', as_dataframe=True)
    >>> df.shape
    (149016, 25)   
    ...
    # if we want to get 15Min data we can do as below
    >>> dataset = CAMELS_LUX(timestep='15Min')
    >>> _, dynamic = dataset.fetch(stations='ID_02', as_dataframe=True)
    >>> df.shape
    (596061, 25) 
    """

    url = "https://zenodo.org/records/14910359"

    def __init__(self,
                 path=None,
                 timestep:str = 'D',
                 overwrite=False,
                 to_netcdf: bool = True,
                 **kwargs):

        assert timestep in ['D', 'H', '15Min'], "timestep must be one of ['D', 'H', '15Min']"

        super(CAMELS_LUX, self).__init__(
            path=path,
            timestep=timestep,
            to_netcdf=to_netcdf,
            **kwargs)

        # Skip download if any proof-of-data is already on disk: the original
        # CAMELS-LUX folder, OR any of the per-timestep consolidated NetCDF
        # caches (camels_lux_{D,H,15Min}.nc). The presence of any .nc proves
        # the source archive was once successfully extracted, which is what
        # the download is for.
        lux_dir = os.path.join(self.path, "CAMELS-LUX")
        name_lc = self.name.lower()
        nc_files = [os.path.join(self.path, f"{name_lc}_{ts}.nc")
                    for ts in ('D', 'H', '15Min')]
        already_have = os.path.exists(lux_dir) or any(os.path.exists(f) for f in nc_files)
        if not already_have or overwrite:
            self._download(overwrite=overwrite)

        # if self.to_netcdf:
        self._maybe_to_netcdf()

    @property
    def static_features(self) -> List[str]:
        return self._static_data().columns.to_list()

    @property
    def dynamic_features(self) -> List[str]:
        ts_path = {
            'D': self.daily_ts_path,
            'H': self.hourly_ts_path,
            '15Min': self.subhourly_ts_path,
        }[self.timestep]
        if os.path.exists(ts_path):
            df = self._read_stn_dyn(self.stations()[0], nrows=2)
            return df.columns.to_list()
        # Fallback when the per-station csv folder has been removed by
        # free_disk_space("redundant"): read feature names from the
        # consolidated NetCDF for the current timestep.
        with netCDF4.Dataset(self.dyn_fpath, "r") as ds:
            return [str(s) for s in ds.variables["dynamic_features"][:]]

    def stations(self) -> List[str]:
        """
        returns names of stations a list
        """
        return pd.read_csv(
            os.path.join(self.path, "CAMELS-LUX", "basin_id.csv"),
            header=None,
            index_col=0,
            dtype={0: str}
        ).index.to_list()

    def _redundant_after_consolidation(self) -> List[Tuple[str, str]]:
        """
        The per-station csv files under ``timeseries/<timestep>/`` become
        redundant once the consolidated NetCDF for that timestep exists.

        All three timesteps are listed regardless of the current instance's
        ``self.timestep``: the parent's per-pair guard skips any folder
        whose backing ``.nc`` cache is missing, so a single
        ``free_disk_space("redundant")`` call cleans up every timestep that
        has been consolidated. After all three NetCDFs are present and
        cleanup runs, the outer ``CAMELS-LUX/timeseries/`` directory will
        be left as empty timestep subdirs (which can be removed manually).
        """
        name_lc = self.name.lower()
        pairs: List[Tuple[str, str]] = []
        for ts_code, ts_word in (('D', 'daily'), ('H', 'hourly'), ('15Min', '15Min')):
            cache = os.path.join(self.path, f"{name_lc}_{ts_code}.nc")
            pairs.append((os.path.join(self.ts_path, ts_word), cache))
        return pairs

    @property
    def boundary_file(self):
        return os.path.join(
            self.path,
            "CAMELS-LUX_shapefiles",
            "catchments_CAMELS-LUX.shp"
        )
    
    @property
    def static_map(self) -> Dict[str, str]:
        return {
            'Lat': gauge_latitude(),
            'Lon': gauge_longitude(),
            'area_km2': catchment_area(),
            'SLOPE_MEAN': slope('degree'),
            'Z_MEAN': catchment_elevation_meters(),
            'grassland': grass_fraction(),
            'agricultural_land': crop_fraction(),
            'urban': urban_fraction(),
            'perimeter_km': catchment_perimeter(),
        }
    
    @property
    def static_factors(self) -> Dict[str, float]:
        """
        static factors for CAMELS-LUX catchments
        """
        return {
            urban_fraction(): 0.01,
            grass_fraction(): 0.01,
            crop_fraction(): 0.01,
        }

    @property
    def dyn_map(self) -> Dict[str, str]:
        """
        dynamic features map for CAMELS-LUX catchments
        """
        return {
            'Q': observed_streamflow_cms(),
            'Qspec': observed_streamflow_mm(),
            'RR_rad': total_precipitation_with_specifier('radar'),
            'RR_stn': total_precipitation_with_specifier('station'),
            'tp': total_precipitation_with_specifier('era5'),
            't2m': mean_air_temp(),
            'PET_Oudin': total_potential_evapotranspiration_with_specifier('oudin'),
            'PET_PM': total_potential_evapotranspiration_with_specifier('pm'),
            'q': mean_specific_humidity(),  # todo : convert from kg/kg -> g/kg
            'rh': mean_rel_hum(),
            'ws10500': mean_windspeed(),
            'swvl1': soil_moisture_layer1(),
            'swvl2': soil_moisture_layer2(),
            'swvl3': soil_moisture_layer3(),
            'swvl4': soil_moisture_layer4(),
        }

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp('2004-01-01')
    
    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp('2021-12-31')
        
    @property
    def ts_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.path, "CAMELS-LUX", "timeseries")
    
    @property
    def topo_fpath(self) -> Union[str, os.PathLike]:
        return os.path.join(self.path, "CAMELS-LUX", "CAMELS_LUX_topographic_attributes.csv")
    
    @property
    def daily_ts_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.ts_path, "daily")
    
    @property
    def hourly_ts_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.ts_path, "hourly")
    
    @property
    def subhourly_ts_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.ts_path, "15Min")

    def _static_data(self) -> pd.DataFrame:
        """
        static attributes of catchments

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of static features of all catchments of shape (56, 61)
        """

        dfs = []
        idx = 0

        # read all .csv files
        for csv_file in glob.glob(os.path.join(self.path, 'CAMELS-LUX', '*.csv')):

            if not  csv_file.endswith('basin_id.csv'):
                df = pd.read_csv(csv_file, index_col=0, dtype={0: str})

                df.index = df.index.astype(str)

                dfs.append(df)

                idx += 1
        
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
        ts_path = {
            'D': self.daily_ts_path,
            'H': self.hourly_ts_path,
            '15Min': self.subhourly_ts_path
        }

        stn_df = pd.read_csv(
            os.path.join(ts_path[self.timestep], f"CAMELS_LUX_hydromet_timeseries_{stn}.csv"), 
            index_col=0, 
            parse_dates=True,
            nrows=nrows,
            )
        
        stn_df.index = pd.to_datetime(stn_df.index)

        if stn_df.index.has_duplicates:
            warnings.warn(f"{stn} has duplicated index. Removing duplicates.")
        
        # drop rows with duplicated index, ideally there should not be any
        if self.timestep == '15Min':
            stn_df = stn_df[~stn_df.index.duplicated(keep='first')]

        stn_df.rename(columns=self.dyn_map, inplace=True)
        
        return stn_df


class CAMELS_DEBY(_RainfallRunoff):
    """
    lumped and gridded data at hourly and daily timestep for 210
    Bavarian (Germany) catchments following the work of
    `Anwar et al., 2025 <https://doi.org/10.5281/zenodo.14893685>`_.
    """


class CAMELS_ES(_RainfallRunoff):
    """
    """
    url = "https://zenodo.org/records/15040948"


class CAMELS_FI(_RainfallRunoff):
    """
    Dataset of 320 Finnish catchments with 16 dynamic features and 106 static features.
    The dynamic features span from 19610101 to 20231231 with daily timestep.
    The data is downloaded from `Zenodo <https://zenodo.org/records/16257216>`_.

    
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
    ((1, 106), 1, (23010, 5))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 23010, 'dynamic_features': 16})
    ...
    >>> len(dynamic.data_vars)   # -> 10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (320, 2)
    >>> dataset.stn_coords('1156')  # returns coordinates of station whose id is 1156
        62.253101       24.444099
    >>> dataset.stn_coords(['1156', '1116'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('1156')
    # get coordinates of two stations
    >>> dataset.area(['1156', '1116'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('1156')
    """

    url = "https://zenodo.org/records/16257216"

    def __init__(self,
                 path=None,
                 overwrite=False,
                 to_netcdf: bool = True,
                 **kwargs):

        super(CAMELS_FI, self).__init__(
            path=path, 
            to_netcdf=to_netcdf, 
            **kwargs)
        
        self._download(overwrite=overwrite)

        self._unzip_boundaries()

        self._maybe_to_netcdf()

    @property
    def data_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.path, 
                            "CAMELS-FI", 
                            "CAMELS-FI",
                            "data")
    
    @property
    def boundary_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.data_path, "CAMELS_FI_catchment_boundaries")

    @property
    def boundary_file(self) -> Union[str, os.PathLike]:
        return os.path.join(
            self.boundary_path,
            "CAMELS_FI_catchment_boundaries.shp"
        )
    
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
            'snow_depth': snow_depth(),  # change from cm to m
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
        static factors for CAMELS-LUX catchments
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
            a :obj:`pandas.DataFrame` of static features of all catchments of shape (320, 106)
        """

        csv_files = glob.glob(os.path.join(self.data_path, '*.csv'))

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


def _remove_stale(paths, verbosity: int = 1, reason: str = "overwrite=True"):
    """
    Deletes every existing file or folder in ``paths``. Called before an
    ``overwrite=True`` re-download so that nothing stale survives: ``download``
    would save a second archive as ``<name>.zip1`` which nothing reads, an
    existing extracted folder would not be re-extracted, and the base
    ``_maybe_to_netcdf`` would rebuild the netCDF cache by reading the old cache.
    """
    for stale in paths:
        if os.path.lexists(stale):
            if verbosity:
                print(f"{reason}: removing stale {stale}")
            # a symlink is unlinked; its target is left alone
            if os.path.isdir(stale) and not os.path.islink(stale):
                shutil.rmtree(stale)
            else:
                os.remove(stale)
    return


def _warn_duplicate_gauges(dataset: str, meta: pd.DataFrame, decimals: int = 3):
    """
    Warns (regardless of ``verbosity``) if two gauges share the same name and
    coordinates rounded to ``decimals``. ``meta`` must have the columns
    ``gauge_id``, ``gauge_name``, ``gauge_lat`` and ``gauge_lon``. Duplicates
    are only reported, never excluded.
    """
    key = (meta['gauge_name'].astype(str) + '_' +
           meta['gauge_lat'].round(decimals).astype(str) + '_' +
           meta['gauge_lon'].round(decimals).astype(str))
    dup_mask = key.duplicated(keep=False)
    if dup_mask.any():
        dups = meta.loc[dup_mask, 'gauge_id'].tolist()
        warnings.warn(
            f"{dataset}: {int(dup_mask.sum())} gauges appear to be "
            f"duplicates (same name and coordinates): {dups}. They are "
            f"kept in the dataset.", UserWarning)
    return


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


def _kr_ts_fname(kind: str, station: str) -> str:
    """name of a CAMELS-KR time-series file; ``kind`` is ``Hydrological`` or
    ``Meteorological``"""
    return f"CAMELS_KR_{kind}_timeseries_{station}.csv"


def _read_camels_kr_stn(spec: Dict, station: str) -> pd.DataFrame:
    """
    Reads the dynamic data of one CAMELS-KR station as described by
    :meth:`CAMELS_KR._reader_spec`. It is a module-level function so that a
    process pool pickles only the small ``spec``, not the dataset instance.
    """
    frames = [
        pd.read_csv(os.path.join(folder, _kr_ts_fname(kind, station)),
                    usecols=usecols, index_col='date', parse_dates=True)
        for kind, (folder, usecols) in spec['files'].items()
    ]
    df = pd.concat(frames, axis=1) if len(frames) > 1 else frames[0]
    df.rename(columns=spec['rename'], inplace=True)
    apply_dyn_factors(df, spec['factors'])
    df = df.loc[spec['st']:spec['en'], spec['features']].astype(spec['fp'])
    df.index.name = 'time'
    df.columns.name = 'dynamic_features'
    return df


class CAMELS_KR(_RainfallRunoff):
    """
    Daily hydrometeorological time series and static catchment attributes for
    282 catchments in South Korea following
    `Lee et al., 2026 <https://doi.org/10.5194/essd-2026-544>`_ (CAMELS-KR v1.1).
    The data is downloaded from its
    `zenodo repository <https://zenodo.org/records/21930882>`_.

    The dataset has 14 dynamic features from 1981-01-01 to 2025-12-31 (16436
    daily steps) and 75 static features. Streamflow and water level come from
    WAMIS and HRFCO (gauge-dependent coverage, missing days are ``NaN``). The
    meteorological series have no gaps: they are KMA station observations,
    gap-filled and interpolated to a 0.1° grid by the authors; ``pet_mm_gleam``
    and ``aet_mm_gleam`` are from GLEAM4. Catchment boundaries are shapefiles in
    WGS84.

    Dynamic features (raw name, unit):

    - ``q_cms_obs`` (discharge_vol, m3 s-1)
    - ``q_mm_obs`` (discharge_spec, mm day-1)
    - ``wl_m_obs`` (water_level, m above the station's zero datum)
    - ``pcp_mm`` (prec, mm day-1)
    - ``airtemp_C_min``, ``airtemp_C_max``, ``airtemp_C_mean`` (temp_min, temp_max, temp_avg, °C)
    - ``rh_%`` (rel_hum, %)
    - ``windspeed_mps`` (wind_speed at 10 m, m s-1)
    - ``windgust_mps_max`` (wind_speed_max, daily maximum gust, m s-1)
    - ``swdownrad_wm2`` (solar_radiation, MJ m-2 day-1 converted to W m-2)
    - ``pet_mm`` (pet, FAO Penman-Monteith, mm day-1)
    - ``pet_mm_gleam``, ``aet_mm_gleam`` (pet_gleam, aet_gleam, mm day-1)

    Static features keep their raw names except ``area_km2`` (basin_area),
    ``lat``/``long`` (gauge_lat/gauge_lon), ``elev_gauge_m`` (gauge_elev) and
    ``pop_density_<year>_km2`` (dens_<year>). ``lat``/``long`` are rounded to
    0.01° in the source (up to 0.55 km from the gauge); unrounded coordinates
    are in ``Catchment_boundaries/Stations/CAMELS_KR_stations.shp``.

    Not provided: the LSTM and HBV simulated streamflow and the HBV parameters
    are model output, so they are not extracted from the archive. Two served
    feature groups are modelled rather than measured: ``pet_mm_gleam`` and
    ``aet_mm_gleam`` come from GLEAM4 (which learns evaporative stress from
    flux-tower data) and the 18 soil attributes from SoilGrids 2.0 (machine
    learning on soil profiles), as in CAMELS_BR, CAMELS_IND and GSHA.

    .. note::
        Examples of source data issues, served unchanged:

        - ``elev_mean``, ``elev_min``, ``elev_5``, ``elev_95`` and ``elev_max``
          are mislabeled or misaligned (``elev_5`` > ``elev_max`` for every
          gauge), hence not renamed. ``gauge_elev`` (6-599 m) is plausible and
          is the only elevation column renamed.
        - ``flow_record`` counts days with water level, not discharge, and
          ``q_mean`` is lower than the mean of ``q_mm_obs`` at every gauge.
        - ``bulk_density_*`` is in g cm-3 (values 0.7-1.5), not kg m-3 as documented.
        - ``q_cms_obs`` is negative on 2350 days at 39 gauges and holds -999 and
          9999 at 3014610, -99.99 at 2005660 and 2013650, and 999.9 at 1019630.
          ``q_mm_obs`` exceeds 500 mm day-1 on 141 days at 32 gauges.
        - ``wl_m_obs`` holds -9999 or -999.9 at 2016650, 4009665 and 4005660;
          values ~100 times too large (probably cm) at 2018645 (1988-06-20 to
          1993-12-31), 1019630 (1997-1998) and 1018683 (1998); one-day spikes
          of 500-4168 m at 2016650, 4006680, 3101645 and 5005680; and 99.0-99.98 m
          (probably negative stages) at 2013615, 5003650, 2301630 and 2021675.
        - ``swdownrad_wm2`` exceeds the top-of-atmosphere irradiance on 0.85% of
          the days.

    :py:class:`aqua_fetch.rr.CAMELS_SK` covers 178 Korean gauges at hourly
    timestep for 2000-2019 (17 dynamic, 215 static features). Both use the
    official Korean gauge codes, so the 115 gauges they share are found by
    :meth:`common_stations`::

        >>> from aqua_fetch import CAMELS_KR, CAMELS_SK
        >>> shared = CAMELS_KR().common_stations(CAMELS_SK().stations())
        >>> len(shared)
        115

    :py:class:`aqua_fetch.rr.GSHA` has 4 Korean GRDC gauges under different ids;
    they are CAMELS_KR gauges 1007635, 2011650, 3012620 and 5004650, which
    ``common_stations(gsha, max_dist_km=2)`` finds.

    The first initialization (download of ~360 MB, extraction and a 260 MB
    netCDF cache) took ~3 minutes. Afterwards, fetching all 14 features of all
    282 stations takes ~0.3 s from the cache (~0.07 s for one station). Without
    a cache it takes ~1 s from the csv files with a process pool, ~8 s with
    ``processes=1`` and ~0.03 s for one station. With the ``spawn`` or
    ``forkserver`` start method (Windows, macOS, Linux from Python 3.14), run
    scripts under ``if __name__ == "__main__":`` because building the cache
    uses a process pool.

    Examples
    --------
    >>> from aqua_fetch import CAMELS_KR
    >>> dataset = CAMELS_KR()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='1001620', as_dataframe=True)
    >>> dynamic['1001620'].shape
    (16436, 14)
    >>> stns = dataset.stations()
    >>> len(stns)
    282
    ... # get data of 10 % of (randomly selected) stations
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)
    28
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('1001620', as_dataframe=True,
    ...  dynamic_features=['pcp_mm', 'airtemp_C_mean', 'pet_mm', 'q_cms_obs'])
    >>> dynamic['1001620'].shape
    (16436, 4)
    ... # get data between two dates
    >>> _, dynamic = dataset.fetch('1001620', st='2020-01-01', en='2020-12-31', as_dataframe=True)
    >>> dynamic['1001620'].shape
    (366, 14)
    ... # get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='1001620', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['1001620'].shape
    ((1, 75), 1, (16436, 14))
    >>> dataset.fetch_static_features('1001620', ['area_km2', 'aridity', 'forest_perc'])
              area_km2   aridity  forest_perc
    gauge_id
    1001620      160.9  0.701389    90.985306
    ... # without as_dataframe=True (and with xarray installed) an xarray Dataset is returned
    >>> _, dynamic = dataset.fetch(10)
    >>> dynamic.sizes
    Frozen({'time': 16436, 'dynamic_features': 14})
    >>> dataset.stn_coords().shape
    (282, 2)
    >>> dataset.area('1001620')  # km2
    gauge_id
    1001620    160.899994
    Name: area_km2, dtype: float32
    >>> dataset.q_mm('1001620').shape  # observed streamflow in mm day-1
    (16436, 1)
    ... # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('1001620')['type']
    'Polygon'
    """
    url = "https://zenodo.org/records/21930882"

    # the only file downloaded from the record (the pdf description is skipped)
    _archive_name = "CAMELS-KR.zip"

    # folders of model output inside the archive, not extracted (observations only)
    _model_output_dirs = ("Simulated hydrological time series", "HBV_model_parameters")

    # attribute files, named CAMELS_KR_<name>_attributes.csv
    _attr_files = ("location", "topography", "climate", "hydrology",
                   "land cover", "soil", "human influence")

    # the two time-series files of each station, CAMELS_KR_<kind>_timeseries_<id>.csv
    _ts_kinds = ("Hydrological", "Meteorological")

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
            folder in which the ``CAMELS_KR`` folder is (or will be) created.
            If None, the default aqua_fetch data folder is used.
        overwrite : bool
            If True, the archive, the extracted data and the netCDF cache are
            deleted, downloaded and built again.
        to_netcdf : bool
            whether to store all dynamic data in a netCDF file (:attr:`dyn_fpath`)
            for faster fetching. Requires netCDF4 and xarray. Once the file
            exists, the data is read from it even with ``to_netcdf=False``.
        verbosity : int
            0: no message will be printed
        kwargs :
            passed to the parent class e.g. ``processes`` (``processes=1``
            disables multiprocessing) or ``remove_zip`` (delete the archive
            after extraction).
        """
        super().__init__(path=path, overwrite=overwrite, to_netcdf=to_netcdf,
                         verbosity=verbosity, **kwargs)

        # lazy caches
        self._location = None
        self._static_df = None
        self._raw_cols = None
        self._extent = None

        self._download_camels_kr(overwrite=overwrite)

        self._check_manifest()

        _warn_duplicate_gauges(self.name, self._location_attrs().reset_index())

        self._maybe_to_netcdf()

        self.bbox = {'llcrnrlat': 34.0, 'urcrnrlat': 39.5,
                     'llcrnrlon': 126.0, 'urcrnrlon': 130.0}
        self.parallels = range(34, 40, 2)
        self.meridians = range(126, 131, 2)

    def _download_camels_kr(self, overwrite: bool = False):
        """
        Downloads ``CAMELS-KR.zip`` and extracts it. Nothing happens if the
        extracted folder exists, even if the archive was deleted
        (``remove_zip=True``).
        """
        if os.path.isdir(self._root) and not overwrite:
            if self.verbosity:
                print(f"CAMELS_KR data already exists at {self._root}")
            self.maybe_remove_zip_files()
            return

        archive = os.path.join(self.path, self._archive_name)
        if overwrite:
            _remove_stale((archive, self._root, self.dyn_fpath), self.verbosity)

        if not os.path.exists(archive):
            # imported here because this module installs a SIGINT handler on import
            from ..download_zenodo import download_from_zenodo
            os.makedirs(self.path, exist_ok=True)
            download_from_zenodo(self.path, doi=self.url, include=[self._archive_name],
                                 verbosity=self.verbosity)

        self._extract(archive)

        self.maybe_remove_zip_files()
        return

    def _extract(self, archive: str):
        """
        Extracts ``archive``, except the model output, into a temporary folder
        which is renamed to :attr:`_root` once complete. An interrupted
        extraction is therefore redone on the next initialization instead of
        being taken as complete.
        """
        tmp = f"{self._root}_extracting"
        if os.path.exists(tmp):  # left over from an interrupted extraction
            shutil.rmtree(tmp)

        if self.verbosity:
            print(f"extracting {archive}")

        # the archive holds a single top-level folder with the same name as _root
        top = os.path.basename(self._root)
        skip = tuple(f"{top}/{folder}/" for folder in self._model_output_dirs)
        with zipfile.ZipFile(archive) as zf:
            members = [m for m in zf.namelist() if not m.startswith(skip)]
            zf.extractall(tmp, members=members)

        os.replace(os.path.join(tmp, top), self._root)
        os.rmdir(tmp)

        warnings.warn(
            f"CAMELS_KR: {' and '.join(self._model_output_dirs)} are model output "
            f"and were not extracted from {archive}.", UserWarning)
        return

    def _check_manifest(self):
        """
        Warns if any attribute, boundary or time-series file is missing. The
        expected station files come from the location attributes, not from
        whatever is on disk.
        """
        location = self._attr_path("location")
        if not os.path.exists(location):
            raise FileNotFoundError(
                f"{location} not found. Re-initialize CAMELS_KR with overwrite=True.")

        # fiona needs the .dbf/.shx/.prj siblings of the boundary .shp as well
        boundary = os.path.splitext(self.boundary_file)[0]
        missing = [f for f in [self._attr_path(name) for name in self._attr_files]
                   + [f"{boundary}{ext}" for ext in ('.shp', '.dbf', '.shx', '.prj')]
                   if not os.path.exists(f)]

        for kind in self._ts_kinds:
            folder = self._ts_dir(kind)
            present = set(os.listdir(folder)) if os.path.isdir(folder) else set()
            missing += [self._ts_path(kind, stn) for stn in self.stations()
                        if _kr_ts_fname(kind, stn) not in present]

        if missing:
            warnings.warn(
                f"CAMELS_KR: {len(missing)} files are missing, e.g. {missing[:3]}. "
                f"The data is incomplete; re-initialize with overwrite=True.",
                UserWarning)
        return

    @property
    def _root(self) -> os.PathLike:
        """folder into which the archive is extracted"""
        return os.path.join(self.path, "CAMELS-KR")

    def _ts_dir(self, kind: str) -> os.PathLike:
        """folder of the ``Hydrological`` or ``Meteorological`` time-series files"""
        return os.path.join(self._root, f"{kind} time series")

    def _ts_path(self, kind: str, station: str) -> os.PathLike:
        return os.path.join(self._ts_dir(kind), _kr_ts_fname(kind, station))

    def _attr_path(self, name: str) -> os.PathLike:
        return os.path.join(self._root, f"CAMELS_KR_{name}_attributes.csv")

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(self._root, "Catchment_boundaries", "Catchments",
                            "CAMELS_KR_catchments.shp")

    @property
    def boundary_id_map(self) -> str:
        return "ID"

    @property
    def dyn_map(self) -> Dict[str, str]:
        # units (Table 1 of Lee et al., 2026) are verified against the data:
        # discharge_spec == discharge_vol * 86.4 / basin_area, and the daily
        # clearness index of solar_radiation (MJ m-2 day-1) has a median of 0.55
        return {
            'discharge_vol': observed_streamflow_cms(),     # m3 s-1
            'discharge_spec': observed_streamflow_mm(),     # mm day-1
            'water_level': observed_water_level_m(),        # m
            'prec': total_precipitation(),                  # mm day-1
            'temp_min': min_air_temp(),                     # deg C
            'temp_max': max_air_temp(),                     # deg C
            'temp_avg': mean_air_temp(),                    # deg C
            'rel_hum': mean_rel_hum(),                      # %
            'wind_speed': mean_windspeed(),                 # m s-1
            'wind_speed_max': max_wind_gust(),              # m s-1, daily maximum gust
            'solar_radiation': solar_radiation(),           # MJ m-2 day-1 -> W m-2
            'pet': total_potential_evapotranspiration(),    # mm day-1
            'pet_gleam': total_potential_evapotranspiration_with_specifier('gleam'),  # mm day-1
            'aet_gleam': actual_evapotranspiration_with_specifier('gleam'),  # mm day-1
        }

    @property
    def dyn_factors(self) -> Dict[str, float]:
        return {solar_radiation(): MJ_M2_DAY_TO_WM2}

    @property
    def static_map(self) -> Dict[str, str]:
        # the elev_* statistics are not renamed, see the class docstring
        return {
            'basin_area': catchment_area(),          # km2
            'gauge_lat': gauge_latitude(),           # deg N (WGS84)
            'gauge_lon': gauge_longitude(),          # deg E (WGS84)
            'gauge_elev': gauge_elevation_meters(),  # m a.s.l.
            'dens_2000': population_density(2000),   # persons km-2
            'dens_2010': population_density(2010),
            'dens_2020': population_density(2020),
            'dens_2024': population_density(2024),
        }

    @property
    def _mm_feature_name(self) -> str:
        return observed_streamflow_mm()

    @property
    def _area_name(self) -> str:
        return catchment_area()

    @property
    def _coords_name(self) -> List[str]:
        return [gauge_latitude(), gauge_longitude()]

    def _read_attr(self, name: str) -> pd.DataFrame:
        # most attribute files start with a byte order mark
        return pd.read_csv(self._attr_path(name), index_col='gauge_id',
                           dtype={'gauge_id': str}, encoding='utf-8-sig')

    def _location_attrs(self) -> pd.DataFrame:
        if self._location is None:
            self._location = self._read_attr("location")
        return self._location

    def stations(self) -> List[str]:
        """ids of the 282 gauges, from the location attributes file"""
        return self._location_attrs().index.tolist()

    def _static_table(self) -> pd.DataFrame:
        """the cached table of all static features"""
        if self._static_df is None:
            # the location file is already cached for the manifest/duplicate checks
            df = pd.concat([self._location_attrs() if name == "location" else self._read_attr(name)
                            for name in self._attr_files], axis=1)
            df.rename(columns=self.static_map, inplace=True)
            self._static_df = df
        return self._static_df

    def _static_data(self) -> pd.DataFrame:
        # a copy, so that a caller's in-place edit cannot corrupt the cache
        return self._static_table().copy()

    @property
    def static_features(self) -> List[str]:
        return self._static_table().columns.tolist()

    def _raw_columns(self) -> Dict[str, List[str]]:
        """raw column names of the two time-series files. They and the time
        extent are read once from the first gauge's files, which have the same
        columns and dates as every other gauge's (checked by the tests)."""
        if self._raw_cols is None:
            stn = self.stations()[0]
            frames = {kind: pd.read_csv(self._ts_path(kind, stn), index_col='date', parse_dates=True)
                      for kind in self._ts_kinds}
            self._extent = (min(df.index.min() for df in frames.values()),
                            max(df.index.max() for df in frames.values()))
            self._raw_cols = {kind: df.columns.tolist() for kind, df in frames.items()}
        return self._raw_cols

    @property
    def dynamic_features(self) -> List[str]:
        return [self.dyn_map.get(col, col)
                for cols in self._raw_columns().values() for col in cols]

    @property
    def start(self) -> pd.Timestamp:
        self._raw_columns()
        return self._extent[0]

    @property
    def end(self) -> pd.Timestamp:
        self._raw_columns()
        return self._extent[1]

    def _reader_spec(self, features: List[str], st=None, en=None) -> Dict:
        """
        Small, picklable description of a read for :func:`_read_camels_kr_stn`.
        Only the raw columns needed for ``features`` are parsed, and a file with
        none of them is not read at all.
        """
        rename = self.dyn_map
        wanted = set(features)
        files = {}
        for kind, cols in self._raw_columns().items():
            cols = [col for col in cols if rename.get(col, col) in wanted]
            if cols:
                files[kind] = (self._ts_dir(kind), ['date'] + cols)
        return dict(files=files, rename=rename, factors=self.dyn_factors,
                    features=list(features), st=st, en=en, fp=self.fp)

    def _read_stn_dyn(self, station: str) -> pd.DataFrame:
        """
        All dynamic features of one station with ``NaN`` for missing values.
        Values are cast to ``float_precision`` (float32 by default), which is
        safe here: the largest magnitude is ~1e5 (``q_cms_obs``) and the
        relative rounding error is below 1e-7. An existing netCDF cache keeps
        the precision it was built with.
        """
        return _read_camels_kr_stn(self._reader_spec(self.dynamic_features), station)

    def _read_dynamic(
            self,
            stations,
            dynamic_features,
            st: Union[str, pd.Timestamp] = None,
            en: Union[str, pd.Timestamp] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Reads dynamic features of several stations, with a process pool if the
        csv files to parse are large enough (see :func:`n_workers`).
        """
        st, en = self._check_length(st, en)
        features = validate_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')
        if not features:
            raise ValueError("no dynamic feature was requested")
        stations = validate_attributes(stations, self.stations(), 'stations')
        spec = self._reader_spec(features, st, en)

        start = time.time()
        nbytes = len(stations) * sum(os.path.getsize(self._ts_path(kind, stations[0]))
                                     for kind in spec['files']) if stations else 0
        cpus = n_workers(nbytes, len(stations), self.processes)

        if cpus == 1:
            dyn = {stn: _read_camels_kr_stn(spec, stn) for stn in stations}
        else:
            with cf.ProcessPoolExecutor(cpus) as executor:
                results = executor.map(functools.partial(_read_camels_kr_stn, spec), stations)
                dyn = dict(zip(stations, results))

        if self.verbosity > 1:
            print(f"Read {len(dyn)} stations for {len(features)} dyn features "
                  f"in {time.time() - start:.2f} seconds with {cpus} cpus.")
        return dyn


# Process-wide, read-only netCDF4 handles for the two consolidated CAMELSH
# caches (``all_stations_q.nc`` ~18 GB and ``all_stn_forcings.nc`` ~67 GB).
# Re-opening these multi-GB HDF5 files on every fetch — or holding an xarray
# handle and a netCDF4 handle on the *same* file at once — can segfault HDF5
# (the same hazard fixed for EStreams' meteorology.nc). Keeping a single shared
# handle per path, opened once and never mixed with xarray, avoids both.
_SHARED_CAMELSH_NC: Dict[str, "netCDF4.Dataset"] = {}


def _shared_camelsh_nc(path: str) -> "netCDF4.Dataset":
    """Returns the process-wide read-only netCDF4 handle for ``path``, opening
    it once on first use."""
    handle = _SHARED_CAMELSH_NC.get(path)
    if handle is None:
        handle = netCDF4.Dataset(path, "r")
        _SHARED_CAMELSH_NC[path] = handle
    return handle


class CAMELSH(_RainfallRunoff):
    """
    Hourly data of 5,767 catchments from United States of America with 13 dynamic
    features and 779 static features for each catchment. For more details on data see
    `Tran et al., (2025) <https://doi.org/10.1038/s41597-025-05612-6>`_ . The dynamic features
    span from 19800101 to 20241231 . The data is downloaded from
    `Zenodo <https://zenodo.org/records/16729675>`_.

    Please note that usage of this dataset requires xarray and netCDF4 libraries.

    Examples
    --------
    >>> from aqua_fetch import CAMELSH
    >>> dataset = CAMELSH()
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       5767
    ... # get data by station id/name
    >>> _, dynamic = dataset.fetch(stations='02342070', as_dataframe=True)
    >>> df = dynamic['02342070'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (394488, 13)
    ...
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (67 out of 5767)
       67
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(394488, 13), (394488, 8), (394488, 13),... (394488, 13), (394488, 13)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('02342070', as_dataframe=True,
    ...  dynamic_features=['swdownrad_wm2', 'pcp_mm', 'pet_mm', 'airtemp_C_mean', 'q_cms_obs'])
    >>> dynamic['02342070'].shape
       (394488, 5)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='02342070', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['02342070'].shape
    ((1, 779), 1, (394488, 13))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 394488, 'dynamic_features': 8})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (5767, 2)
    >>> dataset.stn_coords('02342070')  # returns coordinates of station whose id is 02342070
        32.37431	-84.957993
    >>> dataset.stn_coords(['02342070', '14316700'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('02342070')
    # get coordinates of two stations
    >>> dataset.area(['02342070', '14316700'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('02342070')

    """
    url = {
        "Hourly2.zip": "https://zenodo.org/records/16729675",  # contains observed q and water level
        "timeseries_nonobs.7z": "https://zenodo.org/records/15070091",  # contains NLDAS forcing data
        "timeseries.7z": "https://zenodo.org/records/15066778",
        "attributes.7z": "https://zenodo.org/records/15066778",
        "info.csv": "https://zenodo.org/records/15066778",
        "shapefiles.7z": "https://zenodo.org/records/15066778"
    }

    def __init__(self,
                 path=None,
                 overwrite=False,
                 timestep="H",
                 **kwargs,
    ):

        assert netCDF4 is not None, "netCDF4 library is required for CAMELSH dataset. Please install it using 'pip install netCDF4'"

        super(CAMELSH, self).__init__(
        path=path,
        timestep=timestep,
        **kwargs)

        assert self.timestep == "H", f"CAMELSH dataset only supports hourly timestep but got {self.timestep}."

        # For each remote resource, declare the on-disk paths that fully satisfy
        # it. If any of these exists, the archive is not needed and we skip
        # download + unzip. This prevents re-acquiring data that free_disk_space()
        # (or the user) has deleted because a consolidated NetCDF cache
        # (all_stations_q.nc / all_stn_forcings.nc) covers the same content.
        satisfied_by = {
            "Hourly2.zip":          [self.h2_path,         self.all_stations_q_path],
            "timeseries_nonobs.7z": [self.nonobs_path,     self.all_stn_forcings_path],
            "timeseries.7z":        [self.timeseries_path, self.all_stn_forcings_path],
            "attributes.7z":        [self.attr_path],
            "shapefiles.7z":        [self.sf_path],
            "info.csv":             [os.path.join(self.path, "info.csv")],
        }

        for fname, url in self.url.items():

            if not overwrite and any(os.path.exists(p) for p in satisfied_by.get(fname, [])):
                continue

            dirname = fname.split('.')[0]
            dirpath = os.path.join(self.path, dirname)

            fpath = os.path.join(self.path, fname)

            if not ((os.path.exists(fpath) or os.path.exists(dirpath)) or overwrite):
                download_and_unzip(self.path, url, include=[fname], verbosity=self.verbosity)

            uzipped_dir_path = os.path.join(self.path, fname.split('.')[0])
            if fname.endswith(('.zip', '.7z')) and not os.path.exists(uzipped_dir_path):
                unzip(self.path, keep_parent_dir=True, verbosity=self.verbosity)

        # Discover stations and dynamic features. Prefer the per-station files
        # when Hourly2/ is present; otherwise fall back to the consolidated nc
        # caches. The fallback uses netCDF4 directly (one open per file) instead
        # of xarray, which is much faster for metadata-only lookups when the
        # consolidated file has thousands of data variables.
        if os.path.exists(self.h2_path):
            self.__stations = [fname.split('_')[0] for fname in os.listdir(self.h2_path)]
            self.__dyn_features = self._read_stn_dyn(self.stations()[0]).dynamic_features.data.tolist()
        else:
            with netCDF4.Dataset(self.all_stations_q_path, "r") as _ncq:
                self.__stations = [v for v in _ncq.variables if v not in _ncq.dimensions]
                q_feats_raw = [str(s) for s in _ncq.variables["dynamic_features"][:]]
            with netCDF4.Dataset(self.all_stn_forcings_path, "r") as _ncf:
                f_feats_raw = [str(s) for s in _ncf.variables["dynamic_features"][:]]
            q_feats = [str(self.dyn_map.get(v, v)) for v in q_feats_raw]
            f_feats = [str(self.dyn_map.get(v, v)) for v in f_feats_raw]
            self.__dyn_features = q_feats + f_feats

        self.bbox = {"llcrnrlat": 22, "urcrnrlat": 75,
                     "llcrnrlon": -168.0,  "urcrnrlon": -65.0}
        self.parallels = np.arange(22, 75, 7)
        self.meridians = np.arange(-168, -65, 12)

        # lazily-populated caches for the consolidated-cache fast reader and the
        # static attribute table (both are heavy to build and read many times).
        self._axes_cache = None
        self._static_cache = None
        self._read_order_cache = None

    def stations(self) -> List[str]:
        return self.__stations

    @property
    def static_map(self) -> Dict[str, str]:
        return {
            'LAT_GAGE': gauge_latitude(),
            'LNG_GAGE': gauge_longitude(),
            #'LAT_CENT': centroid_latitude(),
            #'LONG_CENT': centroid_longitude(),
            'ELEV_MEAN_M_BASIN': catchment_elevation_meters(),
            'DRAIN_SQKM': catchment_area(),
            'ELEV_SITE_M': gauge_elevation_meters(),
            'SLOPE_PCT': slope('percent'),
            'PDEN_2000_BLOCK': population_density(2000),
            'PDEN_DAY_LANDSCAN_2007': population_density(2007),
            #'PDEN_NIGHT_LANDSCAN_2007': population_density(2007),
            # 'CLAYAVE': clay_content(),
            # 'SILTAVE': silt_content(),
            # 'SANDAVE': sand_content(),
        }

    @property
    def dyn_map(self) -> Dict[str, str]:
        return {
            #'water_level': water_level('m'),
            'Tair': mean_air_temp(),
            'PotEvap': total_potential_evapotranspiration(),
            'Rainf': total_precipitation(),
            # NLDAS-2 downward shortwave/longwave, already hourly W m-2
            # (verified: shortwave peaks near 1000 and is 0 at night)
            'SWdown': solar_radiation(),
            'LWdown': downward_longwave_radiation(),
            'streamflow': observed_streamflow_cms()
        }

    @property
    def boundary_id_map(self) -> str:
        return "GAGE_ID"

    @property
    def h2_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.path, "Hourly2", "Hourly2")
    
    @property
    def attr_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.path, "attributes")
    
    @property
    def sf_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.path, "shapefiles")
    
    @property
    def boundary_file(self) -> Union[str, os.PathLike]:
        return os.path.join(
            self.sf_path,
            "CAMELSH_shapefile.shp"
        )
    
    @property
    def nonobs_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.path, "timeseries_nonobs", "Data", "CAMELSH", "timeseries_nonobs")

    @property
    def timeseries_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.path, "timeseries", "Data", "CAMELSH", "timeseries")

    @property
    def all_stn_forcings_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.path, 'all_stn_forcings.nc')

    @property
    def all_stations_q_path(self) -> Union[str, os.PathLike]:
        return os.path.join(self.path, 'all_stations_q.nc')

    @property
    def dynamic_features(self) -> List[str]:
        """
        Returns a list of dynamic features that are available in the dataset.

        Returns
        -------
        List[str]
            a list of dynamic features that are available in the dataset.
            The names of the features are the same as the names used in the
            dataset. The names can be used to fetch the data using
            :meth:`fetch_dynamic_features`.
        """
        # overwriting because _read_stn_dyn in this class returns xarray Dataset
        return self.__dyn_features

    # ------------------------------------------------------------------
    # Fast reader for the consolidated NetCDF caches.
    #
    # When ``all_stations_q.nc`` + ``all_stn_forcings.nc`` are present (the
    # standard, disk-consolidated layout), dynamic data is read directly with
    # netCDF4 and assembled with numpy. This replaces xarray's per-variable
    # ``concat``/``sel`` machinery — which took minutes for a few hundred
    # stations because its cost scales with the ~5,767 data variables in each
    # file — with a direct read that is byte-for-byte identical.
    # ------------------------------------------------------------------

    def _consolidated_ready(self) -> bool:
        """True when both consolidated caches exist, so the fast netCDF4 reader
        can be used instead of the per-station files."""
        return (os.path.exists(self.all_stations_q_path)
                and os.path.exists(self.all_stn_forcings_path))

    @property
    def _q_handle(self) -> "netCDF4.Dataset":
        return _shared_camelsh_nc(self.all_stations_q_path)

    @property
    def _forcing_handle(self) -> "netCDF4.Dataset":
        return _shared_camelsh_nc(self.all_stn_forcings_path)

    def _consolidated_axes(self):
        """
        Returns ``(time_index, q_feats, f_feats, q_col, f_col)`` for the
        consolidated caches, decoded once and cached.

        - ``time_index`` : the shared hourly :obj:`pandas.DatetimeIndex`
          (identical in both files).
        - ``q_feats`` / ``f_feats`` : mapped dynamic-feature names held by
          ``all_stations_q.nc`` / ``all_stn_forcings.nc``.
        - ``q_col`` / ``f_col`` : maps each mapped feature name to its column
          index within that file's per-station ``(time, dynamic_features)``
          variable.
        """
        if self._axes_cache is None:
            q_exists = os.path.exists(self.all_stations_q_path)
            f_exists = os.path.exists(self.all_stn_forcings_path)
            # the decoded time axis is identical in both files; use whichever
            # exists to build it once.
            time_handle = self._q_handle if q_exists else self._forcing_handle
            tv = time_handle.variables['time']
            time_index = pd.DatetimeIndex(
                netCDF4.num2date(
                    tv[:], tv.units, getattr(tv, 'calendar', 'standard'),
                    only_use_cftime_datetimes=False,
                    only_use_python_datetimes=True),
                name='time')
            q_feats = ([str(self.dyn_map.get(str(s), str(s)))
                        for s in self._q_handle.variables['dynamic_features'][:]]
                       if q_exists else [])
            f_feats = ([str(self.dyn_map.get(str(s), str(s)))
                        for s in self._forcing_handle.variables['dynamic_features'][:]]
                       if f_exists else [])
            q_col = {name: i for i, name in enumerate(q_feats)}
            f_col = {name: i for i, name in enumerate(f_feats)}
            self._axes_cache = (time_index, q_feats, f_feats, q_col, f_col)
        return self._axes_cache

    def _time_positions(self, st, en):
        """Maps a ``(st, en)`` window onto integer positions in the shared time
        index. ``time_index[i0:i1]`` is inclusive of both endpoints, matching
        :meth:`xarray.Dataset.sel` with a ``slice``."""
        time_index = self._consolidated_axes()[0]
        st, en = self._check_length(st, en)
        i0 = int(time_index.searchsorted(pd.Timestamp(st), side='left'))
        i1 = int(time_index.searchsorted(pd.Timestamp(en), side='right'))
        return time_index, i0, i1

    def _read_order(self) -> Dict[str, int]:
        """
        Cached ``{station: position}`` in the variable-creation order of the
        (compressed, chunked, hence seek-sensitive) forcing file. Reading a
        random station subset in this order turns scattered back-and-forth
        seeks into a mostly-forward scan; because the data lives on a spinning
        disk this measurably speeds up cold reads and never changes the result
        (the caller still receives a station-keyed dict).
        """
        if self._read_order_cache is None:
            handle = (self._forcing_handle
                      if os.path.exists(self.all_stn_forcings_path)
                      else self._q_handle)
            self._read_order_cache = {
                v: i for i, v in enumerate(
                    v for v in handle.variables if v not in handle.dimensions)}
        return self._read_order_cache

    def _read_consolidated(self, stations: List[str], feats: List[str], i0: int, i1: int):
        """
        Reads ``feats`` for ``stations`` over time positions ``[i0:i1]`` directly
        from the consolidated netCDF4 caches.

        Returns ``{station: ndarray(shape=(i1-i0, len(feats)), dtype=float32)}``
        with columns ordered exactly as ``feats``. NaN handling matches xarray:
        ``all_stations_q.nc`` variables carry a ``_FillValue`` (returned as a
        masked array, filled with NaN) while ``all_stn_forcings.nc`` stores NaNs
        directly, and ``np.ma.filled`` is correct for both.

        Reads are issued in on-disk order (see :meth:`_read_order`) to keep the
        spinning-disk access pattern sequential; a process pool is deliberately
        *not* used because the cold read is disk-seek bound (parallel workers do
        not speed it up and the IPC of shipping the decoded arrays back is a net
        loss — both measured).
        """
        _, _, _, q_col, f_col = self._consolidated_axes()
        need_q = any(f in q_col for f in feats)
        need_f = any(f in f_col for f in feats)
        qh = self._q_handle if need_q else None
        fh = self._forcing_handle if need_f else None
        n = i1 - i0
        order = self._read_order()
        read_seq = sorted(stations, key=lambda s: order.get(s, 0))
        out: Dict[str, np.ndarray] = {}
        for stn in read_seq:
            qa = np.ma.filled(qh.variables[stn][i0:i1], np.nan) if need_q else None
            fa = np.ma.filled(fh.variables[stn][i0:i1], np.nan) if need_f else None
            arr = np.empty((n, len(feats)), dtype='float32')
            for j, feat in enumerate(feats):
                col = q_col.get(feat)
                if col is not None:
                    arr[:, j] = qa[:, col]
                else:
                    arr[:, j] = fa[:, f_col[feat]]
            out[stn] = arr
        return out

    def _build_dyn_dataset(self, data: Dict[str, np.ndarray], feats: List[str], time_index):
        """Wraps ``{station: ndarray(time, dynamic_features)}`` into an xarray
        Dataset (data_vars = stations, dims = time × dynamic_features) without
        copying the arrays."""
        return xr.Dataset(
            {stn: (('time', 'dynamic_features'), arr) for stn, arr in data.items()},
            coords={'time': time_index, 'dynamic_features': list(feats)},
        )

    def _fetch_dynamic(self, stations: List[str], feats: List[str], st, en, as_dataframe: bool):
        """Fast entry point for dynamic reads from the consolidated caches.
        Returns a dict of per-station DataFrames (``as_dataframe=True``) or an
        xarray Dataset."""
        time_index, i0, i1 = self._time_positions(st, en)
        data = self._read_consolidated(stations, feats, i0, i1)
        tsel = time_index[i0:i1]
        if as_dataframe:
            out: Dict[str, pd.DataFrame] = {}
            for stn in stations:
                df = pd.DataFrame(data[stn], index=tsel, columns=list(feats))
                df.columns.name = 'dynamic_features'
                df.index.name = 'time'
                out[stn] = df
            return out
        return self._build_dyn_dataset(data, feats, tsel)

    def close(self):
        """Closes the process-wide consolidated-cache handles for this dataset,
        if open, and drops the decoded-axes cache."""
        for path in (self.all_stations_q_path, self.all_stn_forcings_path):
            handle = _SHARED_CAMELSH_NC.pop(path, None)
            if handle is not None:
                handle.close()
        self._axes_cache = None
        self._read_order_cache = None

    def _read_stn_q(self, stn):
        fpath = os.path.join(self.h2_path, f"{stn}_hourly.nc")
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"q data for station {stn} not found in {self.h2_path}")
        
        ds = xr.open_dataset(fpath, engine='netcdf4')
        return ds

    def _read_stn_q1(self, stn):
        fpath = os.path.join(self.h2_path, f"{stn}_hourly.nc")
        if os.path.exists(fpath):
            dyn_map = {k:v for k,v in self.dyn_map.items() if k in ['streamflow']}
            ds = xr.open_dataset(fpath, engine='netcdf4').rename(dyn_map)
            return ds.to_array("dynamic_features").astype('float32').to_dataset(name=stn).transpose()

        # Fall back to the consolidated cache if the per-station file is gone.
        # Read via the shared netCDF4 handle (never xarray) so we never hold two
        # handles on the same multi-GB file at once.
        if os.path.exists(self.all_stations_q_path):
            time_index, q_feats, _, _, _ = self._consolidated_axes()
            data = self._read_consolidated([stn], q_feats, 0, len(time_index))
            return self._build_dyn_dataset(data, q_feats, time_index)

        raise FileNotFoundError(
            f"q data for station {stn} not found in {self.h2_path} "
            f"nor in {self.all_stations_q_path}"
        )

    def fetch_q(
            self, 
            stations:List[str] = "all"
            ):
        """
        Since fetching q from other methods can be slower because of merging with 
        other dynamic (forcing) features, this method fetches only observed streamflow 
        data for given stations using multiprocessing.

        Returns
        --------
        xr.Dataset
            xarray Dataset whose data variables are station names and dimensions 
            are time and dynamic features
        """
        stations = validate_attributes(stations, self.stations(), 'stations')

        all_q_fname = self.all_stations_q_path
        if os.path.exists(all_q_fname) and not self.overwrite:
            if self.verbosity>1:
                print(f"Loading q data for {len(stations)} stations from {all_q_fname}")
            # read the requested stations directly via the shared netCDF4 handle
            # (numpy assembly) instead of xarray, which is orders of magnitude
            # faster for a file holding thousands of data variables.
            time_index, q_feats, _, _, _ = self._consolidated_axes()
            data = self._read_consolidated(stations, q_feats, 0, len(time_index))
            return self._build_dyn_dataset(data, q_feats, time_index)

        cpus = self.processes or min(32, get_cpus())
        if self.verbosity>2: print(f"Using {cpus} processes to read q data of {len(stations)} stations")

        st = time.time()
        with cf.ProcessPoolExecutor(cpus) as executor:
            results = executor.map(self._read_stn_q1, stations)
        et = time.time()
        total = round(et - st, 2)
        if self.verbosity: print(f"read q for {len(stations)} stations in {total} secs with {cpus} processes")

        results = xr.merge(results)

        if len(stations) == len(self.stations()):
            # saving it so that next time we don't have to read all stations separately again
            if self.verbosity>1:
                print(f"Saving all stations q data to {all_q_fname}")
            results.to_netcdf(all_q_fname, engine='netcdf4',
                              encoding={stn: {'dtype': 'float32'} for stn in stations})
        return results

    def _read_stn_forcing1(self, stn):
        """
        Returns
        -------
        xr.Dataset
            xarray Dataset with 'time' and 'dynamic_features' dimensions and 
            stn as data variable
        """
        fpath = os.path.join(self.nonobs_path, f"{stn}.nc")
        if not os.path.exists(fpath):
            fpath = os.path.join(self.timeseries_path, f"{stn}.nc")
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"forcing data for station {stn} not found in {self.nonobs_path}")
        
        ds = xr.open_dataset(fpath, engine='netcdf4')

        # todo : what is difference between Streamflow in forcing and streamflow in Hourly2 path?
        ds = ds.drop_vars("Streamflow", errors="ignore")

        ds = ds.rename({'DateTime': 'time'})
        return ds.to_array("dynamic_features").transpose().astype('float32').to_dataset(name=stn)

    def _read_stns_forcing(self, stations:List[str]):
        """retunrs forcings of multiple stations as xarray Dataset."""
        assert isinstance(stations, list), "stations should be a list of station names/ids"

        all_stn_forcings = self.all_stn_forcings_path
        if os.path.exists(all_stn_forcings) and not self.overwrite:
            if self.verbosity>1:
                print(f"Loading forcing data for {len(stations)} stations from {all_stn_forcings}")
            # direct netCDF4 + numpy assembly (see fetch_q); the feature names are
            # already mapped via dyn_map inside _consolidated_axes.
            time_index, _, f_feats, _, _ = self._consolidated_axes()
            data = self._read_consolidated(stations, f_feats, 0, len(time_index))
            return self._build_dyn_dataset(data, f_feats, time_index)

        cpus = self.processes or min(32, get_cpus())
        st = time.time()

        if self.verbosity>2: print(f"Using {cpus} processes to read forcing data of {len(stations)} stations")

        with cf.ProcessPoolExecutor(cpus) as executor:
            results = executor.map(self._read_stn_forcing1, stations)
        et = time.time()
        total = round(et - st, 2)
        if self.verbosity: print(f"read {len(stations)} stations forcing data in {total} secs with {cpus} processes")
        results = xr.merge(results)

        new_dyn = [str(self.dyn_map.get(v, v)) for v in results["dynamic_features"].data]
        results = results.assign_coords(dynamic_features=("dynamic_features", new_dyn))
        return results

    def _read_stn_forcing(self, stn):
        """prefers reading from all_stn_forcings_path if exists."""
        assert isinstance(stn, str), "station name/id should be a string"
    
        all_stn_forcings = self.all_stn_forcings_path
        if os.path.exists(all_stn_forcings) and not self.overwrite:
            if self.verbosity>1:
                print(f"Loading forcing data for {stn} from {all_stn_forcings}")
            time_index, _, f_feats, _, _ = self._consolidated_axes()
            data = self._read_consolidated([stn], f_feats, 0, len(time_index))
            return self._build_dyn_dataset(data, f_feats, time_index)

        fpath = os.path.join(self.nonobs_path, f"{stn}.nc")
        if not os.path.exists(fpath):
            fpath = os.path.join(self.timeseries_path, f"{stn}.nc")
            if not os.path.exists(fpath):
                raise FileNotFoundError(f"forcing data for station {stn} not found in {self.nonobs_path}")
        
        ds = xr.open_dataset(fpath, engine='netcdf4')

        # todo : what is difference between Streamflow in forcing and streamflow in Hourly2 path?
        ds = ds.drop_vars("Streamflow", errors="ignore")

        ds = ds.rename({'DateTime': 'time'})
    
        new_dyn = {k:str(self.dyn_map.get(k, k)) for k in ds.data_vars}
        return ds.rename(new_dyn).to_array("dynamic_features").transpose().astype('float32').to_dataset(name=stn)

    def _read_stns_dyn(self, stns:List[str]):
        """
        Returns
        -------
        xr.Dataset
        """
        if self._consolidated_ready():
            time_index, q_feats, f_feats, _, _ = self._consolidated_axes()
            feats = q_feats + f_feats
            data = self._read_consolidated(stns, feats, 0, len(time_index))
            return self._build_dyn_dataset(data, feats, time_index)

        q = self.fetch_q(stns)
        forcing = self._read_stns_forcing(stns)

        ds = xr.concat([q, forcing], dim='dynamic_features')

        return ds

    def _read_stn_dyn(self, stn:str, nrows=None) -> pd.DataFrame:
        if self._consolidated_ready():
            time_index, q_feats, f_feats, _, _ = self._consolidated_axes()
            feats = q_feats + f_feats
            data = self._read_consolidated([stn], feats, 0, len(time_index))
            return self._build_dyn_dataset(data, feats, time_index)

        q = self._read_stn_q1(stn)
        forcing = self._read_stn_forcing(stn)

        ds = xr.concat([q, forcing], dim='dynamic_features')

        return ds

    def _static_data(self) -> pd.DataFrame:
        """
        reads static data for all stations. The assembled table is cached after
        the first read because it is consulted many times (static_features,
        stn_coords, area, every static fetch) and re-reading ~30 CSVs each time
        was a needless repeated cost.
        """
        if self._static_cache is not None:
            return self._static_cache

        csv_files = glob.glob(os.path.join(self.attr_path, '*.csv'))

        dfs = []
        for csv_file in csv_files:

            if 'attributes_hydroATLAS.csv' in csv_file:
                df = pd.read_csv(
                csv_file, 
                index_col=0, 
                sep='\t',
                dtype={0: str})
            else:
                df = pd.read_csv(
                csv_file, 
                index_col=0, 
                dtype={0: str})
            df.index = df.index.astype(str)
            dfs.append(df)

        df = pd.concat(dfs, axis=1)

        # drop duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]

        # rename columns using self.static_map
        df = df.rename(columns=self.static_map)
        self._static_cache = df
        return df

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
        """
        Reads features of more than one stations.

        parameters
        ----------
        stations :
            list of stations for which data is to be fetched.
        dynamic_features :
            list of dynamic features to be fetched.
            if ``all``, then all dynamic features will be fetched.
        static_features : list of static features to be fetched.
            If ``all``, then all static features will be fetched. If None,
            `then no static attribute will be fetched.
        st :
            start of data to be fetched.
        en :
            end of data to be fetched.
        as_dataframe :
            whether to return the dynamic data as pandas dataframe. default
            is :obj:`xarray.Dataset` object
        kwargs dict:
            additional keyword arguments

        Returns
        -------
        tuple
            A tuple of static and dynamic features. Static features are always
            returned as :obj:`pandas.DataFrame` with shape (stations, staticfeatures).
            The index of static features is the station/gauge ids while the columns 
            are the static features. Dynamic features are returned as either
            :obj:`xarray.Dataset` or a :obj:`dict` with keys as station names and values as
            :obj:`pandas.DataFrame` depending upon whether `as_dataframe`
            is True or False and whether the xarray module is installed or not.
            If dynamic features are xarray Dataset, then it consists of `data_vars`
            equal to the number of stations and `time` and `dynamic_features` as
            dimensions.

        Raises:
            ValueError, if both dynamic_features and static_features are None

        Examples
        --------
        >>> from aqua_fetch import CAMELSH
        >>> dataset = CAMELSH()
        ... # find out station ids
        >>> dataset.stations()
        ... # get data of selected stations as xarray Dataset
        >>> dataset.fetch_stations_features(['01141800', '02349900', '11062000'])
        ... # get data of selected stations as dictionary of pandas DataFrame
        >>> dataset.fetch_stations_features(['01141800', '02349900', '11062000'],
        ...  as_dataframe=True)
        ... # get both dynamic and static features of selected stations
        >>> dataset.fetch_stations_features(['01141800', '02349900', '11062000'],
        ... dynamic_features=['q_mm_obs', 'air_temp_C', 'pcp_mm'], static_features=['elev_catch_m'])
        """
        # overwriting because _read_stn_dyn in this class returns xarray Dataset

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

            if self._consolidated_ready():
                # fast path: read/assemble directly from the consolidated caches,
                # building DataFrames without an intermediate xarray Dataset.
                dynamic = self._fetch_dynamic(stations, dynamic_features, st, en, as_dataframe)
            else:
                dynamic = self._read_dynamic(stations, dynamic_features, st=st, en=en)
                if as_dataframe:
                    # convert xarray Dataset to dictionary of pandas DataFrame
                    dynamic = {stn: dynamic[stn].to_pandas() for stn in stations}

            if static_features is not None:
                static = self.fetch_static_features(stations, static_features)

        elif static_features is not None:

            return self.fetch_static_features(stations, static_features), dynamic

        else:
            raise ValueError(f"static features are {static_features} and dynamic features are {dynamic_features}")

        return static, dynamic

    def _read_dynamic(
            self, 
            stations, 
            dynamic_features, 
            st:Union[str, pd.Timestamp] = None, 
            en:Union[str, pd.Timestamp] = None
            ):
        """
        Returns
        -------
        xr.Dataset
        """
        # overwriting because here we  always return xarray Dataset

        st, en = self._check_length(st, en)
        dyn_feats = validate_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')
        stations = validate_attributes(stations, self.stations(), 'stations')

        if self._consolidated_ready():
            # fast path: direct netCDF4 read + numpy assembly, already sliced to
            # the requested features and time window.
            return self._fetch_dynamic(stations, dyn_feats, st, en, as_dataframe=False)

        # fallback (per-station files; consolidated caches not built yet). There
        # can be 3 scenarios.
        if len(dyn_feats) == 1 and dyn_feats[0] == observed_streamflow_cms():
            # only q is asked
            results = self.fetch_q(stations)

        elif observed_streamflow_cms() not in dyn_feats and 'water_level' not in dyn_feats:
            # only forcing data is asked
            results = self._read_stns_forcing(stations)
        else:
            # both q and forcing data is asked
            if len(stations) > 1:
                results = self._read_stns_dyn(stations)
            else:
                results = self._read_stn_dyn(stations[0])

        # select required dynamic features and time range
        results = results.sel(dynamic_features=dyn_feats, time=slice(st, en))
        return results

    def collate_forcing_data(self):
        """
        Collate forcing data of all stations into a single NetCDF file using multiprocessing.

        """
        stations = self.stations()

        cpus = self.processes or min(32, get_cpus())

        out_path = self.all_stn_forcings_path
        out_path = Path(out_path)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

        if tmp_path.exists():
            tmp_path.unlink()
        if out_path.exists():
            out_path.unlink()  # start clean; or implement resume logic

        # Launch workers
        first_result = None
        results_queue = []
        st = time.time()

        with cf.ProcessPoolExecutor(max_workers=cpus) as ex:
            futures = [ex.submit(self._worker, i) for i in stations]

            # We’ll create the .nc after the first finished task (to learn dims/dtypes)
            for fut in cf.as_completed(futures):
                run_id, time_int, dyn, arr = fut.result()
                if first_result is None:
                    first_result = (run_id, time_int, dyn, arr)
                    break
                else:
                    results_queue.append((run_id, time_int, dyn, arr))

            # Initialize NetCDF using the first result
            fr_run_id, fr_time, fr_dyn, fr_arr = first_result

            # Create temp file to ensure crash-safety; rename at the end
            with netCDF4.Dataset(tmp_path, "w", format="NETCDF4") as nc:
                # Define dimensions
                T = int(fr_time.shape[0])
                D = int(fr_dyn.shape[0])
                nc.createDimension("time", T)
                nc.createDimension("dynamic_features", D)

                # Coordinate variables
                v_time = nc.createVariable("time", "f8", ("time",))
                v_time[:] = fr_time
                v_time.long_name = "time"
                v_time.units = "hours since 1970-01-01 00:00:00"
                v_time.calendar = "proleptic_gregorian"

                # NetCDF4 supports variable-length strings via dtype=str
                v_dyn = nc.createVariable("dynamic_features", str, ("dynamic_features",))
                v_dyn[:] = fr_dyn.data
                v_dyn.long_name = "dynamic feature names"

                # Create **all** data variables up-front (fast metadata, RAM-free)
                # Compression & chunking recommended
                chunks_t = min(T, 1024)
                var_handles = {}
                for stn in stations:
                    vname = f"{stn}"
                    var_handles[stn] = nc.createVariable(
                        vname,
                        fr_arr.dtype,
                        ("time", "dynamic_features"),
                        zlib=True,
                        complevel=4,
                        shuffle=True,
                        chunksizes=(chunks_t, D),
                    )
                    # Optional attrs (for CF/xarray friendliness)
                    var_handles[stn].coordinates = "dynamic_features time"

                # Write the first result
                var_handles[fr_run_id][:] = fr_arr
                nc.sync()

                # Consume already-completed results
                for run_id, time_f, dyn, arr in results_queue:
                    _validate_coords(run_id, time_f, dyn, T, D)
                    var_handles[run_id][:] = arr
                    nc.sync()

                # Consume the rest as they finish
                for fut in cf.as_completed(futures):
                    # Some futures were already consumed; skip them gracefully
                    try:
                        run_id, time_f, dyn, arr = fut.result()
                    except Exception as e:
                        warnings.warn(f"[worker] a forcing-data task failed and its "
                                      f"station was skipped: {e}")
                        continue
                    # Skip the one we already wrote
                    if run_id == fr_run_id:
                        continue
                    _validate_coords(run_id, time_f, dyn, T, D)
                    var_handles[run_id][:] = arr
                    nc.sync()

            # Atomic finalize
            os.replace(tmp_path, out_path)
        et = time.time()
        total = round(et - st, 2)
        if self.verbosity: print(f"collated {len(stations)} stations in {total} secs with {cpus} processes")
        return

    def _worker(self, stn_id: str):
        out = self._read_stn_forcing1(stn_id)
        # Return minimal payload; main proc writes to disk
        return stn_id, _encode_time_for_nc(out["time"].data), out["dynamic_features"], out[stn_id].values

    def q_mm(
            self,
            stations: Union[str, List[str]] = "all",
            as_dataframe: bool = True
    ) -> pd.DataFrame:
        """
        returns streamflow in the units of milimeter per timestep (mm/hour). This is obtained
        by diving ``q`` by area.

        parameters
        ----------
        stations : str/list
            name/names of stations. Default is ``all``, which will return
            q_mm of all stations
        as_dataframe : bool
            whether to return the data as pandas DataFrame. Default is True.
            Setting it to False will return xarray Dataset and can be faster.

        Returns
        --------
        pd.DataFrame or xr.Dataset
            a :obj:`pandas.DataFrame` whose indices are time-steps and columns
            are catchment/station ids.

        """
        # overwriting because we don't want to call fetch, which can be slow
        # instead we directly call .q method

        stations = validate_attributes(stations, self.stations(), 'stations')

        q = self.fetch_q(stations)
        q = q.sel(dynamic_features='q_cms_obs')
        if as_dataframe:
            q = q.to_pandas().drop(columns=['dynamic_features'], errors='ignore')

        area_m2 = self.area(stations) * 1e6  # area in m2

        time_conversion = 3600  # seconds per hour

        q = (q / area_m2) * time_conversion  # cms to m
        return q * 1e3  # to mm

    def _redundant_after_consolidation(self) -> List[Tuple[str, str]]:
        return [
            (os.path.join(self.path, "Hourly2"),           self.all_stations_q_path),
            (os.path.join(self.path, "timeseries"),        self.all_stn_forcings_path),
            (os.path.join(self.path, "timeseries_nonobs"), self.all_stn_forcings_path),
        ]


def _validate_coords(run_id, time_f, dyn, T, D):
    if time_f.shape[0] != T or dyn.shape[0] != D:
        raise ValueError(
            f"Run {run_id} dims differ: got {(time_f.shape[0], dyn.shape[0])}, expected {(T, D)}"
        )

def _encode_time_for_nc(time64: np.ndarray,
                       units="hours since 1970-01-01 00:00:00",
                       calendar="proleptic_gregorian") -> np.ndarray:
    # convert to Python datetimes (vectorized via pandas)
    py_dt = pd.to_datetime(time64).to_pydatetime()
    return date2num(py_dt, units=units, calendar=calendar).astype("float64")


class HydResponses(_RainfallRunoff):
    """
    See `von Matt et al., 2025 <https://essd.copernicus.org/preprints/essd-2025-383/>`_ .
    """
    url = "https://zenodo.org/records/14713275/files/HYD_RESPONSES.zip"


class ThirdPole(_RainfallRunoff):
    """
    Observed streamflow data from from 11 stations of Third Pole region following work of
    `Liu and Wang (2025) <https://doi.org/10.1080/20964471.2025.2585701>`_ . The data
    is downloaded from its `zenodo repository <https://zenodo.org/records/15853656>`_ .
    """

    _stations = ['Asaraghat', 'Benighat', 'Besham', 'Changdu', 'Chatara', 'Chisapani', 
                 'Devghat', 'Doyain', 'Jiayuqiao', 'Nuxia', 'Yogu']
    url = {
        f"Discharge_{stn}_1981-2020.nc": "https://zenodo.org/records/15853656/files/" for stn in _stations
    }
    url.update({'basin_info.rar': 'https://zenodo.org/records/15853656/files'})
