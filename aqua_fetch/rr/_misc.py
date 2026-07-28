
__all__ = [
    "DraixBleone", 
    "JialingRiverChina", 
    "NamalValleyPakistan"
    ]

import os
import warnings
from typing import Dict, List, Union
import concurrent.futures as cf

import requests
import numpy as np
import pandas as pd

from .utils import validate_attributes
from .utils import _RainfallRunoff
from .utils import get_cpus
from ..utils import merge_shapefiles_fiona
from .._backend import xarray as xr, fiona

from ._map import (
    observed_streamflow_cms,
    snow_depth,
    snow_water_equivalent,
    mean_windspeed,
    total_precipitation,
    mean_dewpoint_temperature,
    observed_water_level_ft,
    )

from ._map import (
    gauge_elevation_meters,
    gauge_latitude,
    gauge_longitude,
    catchment_area,
)


class DraixBleone(_RainfallRunoff):
    """
    A high-frequency, long-term data set of hydrology and sediment yield: the alpine
    badland catchments of Draix-Bléone Observatory following the work of `Klotz et al., 2023 <https://doi.org/10.5194/essd-15-4371-2023>`_.

    """
    url = {
        # "spatial": "https://doi.org/10.57745/RUQLJL",
        # "hydro_sediment": "https://doi.org/10.17180/obs.draix",
        # "climate": "https://doi.org/10.57745/BEYQFQ"
"README.txt": 
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/158242",
"DRAIXBLEONE_DRAIX_BRU_DISCH.txt":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/158223",
"DRAIXBLEONE_DRAIX_BRU_SEDTRAP.txt":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/158225",
"DRAIXBLEONE_DRAIX_BRU_SSC.txt":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/158222",
"DRAIXBLEONE_DRAIX_LAV_DISCH.txt":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/158229",
"DRAIXBLEONE_DRAIX_LAV_SEDTRAP.txt":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/158224",
"DRAIXBLEONE_DRAIX_MOU_DISCH.txt":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/158226",
"DRAIXBLEONE_DRAIX_ROU_DISCH.txt":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/158238",

"Draix_Bleone_instruments.shp":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168716",
"Draix_Bleone_instruments.prj":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168720",
"Draix_Bleone_instruments.dbf":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168715",
"Draix_Bleone_instruments.cpg":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168718",
"Draix_Bleone_instruments.shx":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168717",
"Draix_Bleone_instruments.qpj":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168719",

"Draix_Bleone_catchment_contours.shp":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168844",
"Draix_Bleone_catchment_contours.prj":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168839",
"Draix_Bleone_catchment_contours.dbf":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168843",
"Draix_Bleone_catchment_contours.cpg":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168840",
"Draix_Bleone_catchment_contours.shx":  
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168841",
"Draix_Bleone_catchment_contours.qpj":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168842",

# "DEM_Draix.tif":
#         "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168727",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._download()

    @property
    def boundary_file(self)-> os.PathLike:
        return os.path.join(self.path, "Draix_Bleone_catchment_contours.shp")

    def stations(self)->List[str]:
        return ['BRU', 'LAV', 'MOU', 'ROU']

    def _read_stn_dyn(self, stn:str):

        fpath = os.path.join(self.path, f"DRAIXBLEONE_DRAIX_{stn}_DISCH.txt")
        stn_df = pd.read_csv(fpath, sep=';', index_col=0, parse_dates=False,
                             header=2, usecols=[0, 1, 2],
                             )
        
        stn_df.index = pd.to_datetime(stn_df.index, format="%d/%m/%Y %H:%M:%S")

        stn_df.rename(columns={'Valeur': observed_streamflow_cms()}, inplace=True)
        stn_df.columns.name = 'dynamic_features'
        stn_df.index.name = 'time'

        # convert L/s to m3/s
        # stn_df['runoff'] = stn_df['runoff'] * 1000.0

        return stn_df

    def _static_data(self):
        # from README.txt file
        coords = {'BRU': (965694, 6345789, 801, 1.07, 87), 
                  'LAV': (968818, 6343668, 850, 0.86, 32), 
                  'MOU': (968688, 6343610, 847, 0.086, 46), 
                  'ROU': (968828, 6343644, 852, 0.0013, 21)
                  }
        coords = pd.DataFrame.from_dict(
            coords, orient='index', 
            columns=[gauge_longitude(), gauge_latitude(), gauge_elevation_meters(), 
                     catchment_area(), 
                     'veg_cover_%'])
        return coords


class JialingRiverChina(_RainfallRunoff):
    """
    Dataset of 11 catchments in the upper, middle and lower reaches of the Jialing 
    River mainstream basin, China . For more infromation on data see `Wang et al., 2024 <https://doi.org/10.1016/j.envsoft.2024.106091>`_.
    The data consists of daily observations of weather variables and runoff from 
    2010 to 2022. 

    The dataset is available at `github link <https://github.com/AtrCheema/CVTGR-model>`_.

    Examples
    --------
    >>> from aqua_fetch.rr import JialingRiverChina
    >>> dataset = JialingRiverChina()
    >>> len(dataset.stations())
    11
    >>> df = dataset.fetch_dynamic_features(dataset.stations()[0], as_dataframe=True)
    """
    url = {
        'Beibei.csv': 'https://raw.githubusercontent.com/AtrCheema/CVTGR-model/refs/heads/main/Data/OriginalData/Beibei.csv',
        'Ciba.csv': 'https://raw.githubusercontent.com/AtrCheema/CVTGR-model/refs/heads/main/Data/OriginalData/Ciba.csv',
        'Dongjintuo.csv': 'https://raw.githubusercontent.com/AtrCheema/CVTGR-model/refs/heads/main/Data/OriginalData/Dongjintuo.csv',
        'Fengzhou.csv': 'https://raw.githubusercontent.com/AtrCheema/CVTGR-model/refs/heads/main/Data/OriginalData/Fengzhou.csv',
        'Guangyuan.csv': 'https://raw.githubusercontent.com/AtrCheema/CVTGR-model/refs/heads/main/Data/OriginalData/Guangyuan.csv',
        'Jinxi.csv': 'https://raw.githubusercontent.com/AtrCheema/CVTGR-model/refs/heads/main/Data/OriginalData/Jinxi.csv',
        'Langzhong.csv': 'https://raw.githubusercontent.com/AtrCheema/CVTGR-model/refs/heads/main/Data/OriginalData/Langzhong.csv',
        'Lueyang.csv': 'https://raw.githubusercontent.com/AtrCheema/CVTGR-model/refs/heads/main/Data/OriginalData/Lueyang.csv',
        'Tanjiazhuang.csv': 'https://raw.githubusercontent.com/AtrCheema/CVTGR-model/refs/heads/main/Data/OriginalData/Tanjiazhuang.csv',
        'Tingzikou.csv': 'https://raw.githubusercontent.com/AtrCheema/CVTGR-model/refs/heads/main/Data/OriginalData/Tingzikou.csv',
        'Wusheng.csv': 'https://raw.githubusercontent.com/AtrCheema/CVTGR-model/refs/heads/main/Data/OriginalData/Wusheng.csv',
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._download()

        _dynamic_features = [
            self._read_stn_dyn(stn).columns.tolist()
            for stn in self.stations()
        ]
        # unpack the list of lists into a single list
        self._dynamic_features = list(set([item for sublist in _dynamic_features for item in sublist]))

        self.dyn_fname = ''

    @property
    def dyn_map(self):
        return {
            'runoff': observed_streamflow_cms(),  # as per Table 3 in paper, this is streamflow
            'sd': snow_depth(),
            'sdwe': snow_water_equivalent(),
            'Pre': total_precipitation(),
            'dpt': mean_dewpoint_temperature(),
            'aws': mean_windspeed(),
        }

    def stations(self)->List[str]:
        return [f.split('.')[0] for f in os.listdir(self.path)]
    
    @property
    def dynamic_features(self) -> List[str]:
        return self._dynamic_features

    def _read_stn_dyn(self, stn:str):

        fpath = os.path.join(self.path, f"{stn}.csv")
        stn_df = pd.read_csv(fpath, index_col=0, parse_dates=True)

        stn_df.rename(columns=self.dyn_map, inplace=True)

        stn_df.columns.name = 'dynamic_features'
        stn_df.index.name = 'time'

        return stn_df


class HeiheRiverChina:
    """
     Data of the precipitation, stream discharge, air temperature, dissolved organic 
     carbon concentrations and dissolved inorganic carbon concentrations and DOM 
     optical indices of water at different locations in the Hulugou catchment, upper 
     reaches of Heihe River, Northeastern Tibet Plateau, China. For for on this data
     see `Liu et al., 2025 <https://doi.org/10.1016/j.envsoft.2025.106567>`_ and 
     `Hu et al., 2023 < https://doi.org/10.1029/2022WR032426>`_.
    """
    url = "https://zenodo.org/records/7067158"


class ShyftNorway(_RainfallRunoff):
    """
    The dataset contains  observed streamflow data from 111 Norwegian catchments, 
    as well as catchment boundaries and some catchment specific static data. 
    For more information on this data see `Silantyeva et al., 2025 <https://doi.org/10.5194/egusphere-2025-4071>`_.
    Note that currently only streamflow data is included, other dynamic features 
    may be added in future releases. Also note that observed streamflow data may
    slightly differ from the data from `seriekart.nve.no <https://seriekart.nve.no/>`_ since
    data at seriekart is updated regularly based upon updated rating curves.


    Examples
    --------
    >>> from aqua_fetch import ShyftNorway
    >>> dataset = ShyftNorway()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='2.11.0', as_dataframe=True)
    >>> df = dynamic['2.11.0'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (23376, 1)
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
        [(23376, 1), (23376, 1), (23376, 1),... (23376, 1), (23376, 1)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
        ['observed_streamflow_cms']
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='2.11.0', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['2.11.0'].shape
    ((1, 10), 1, (23376, 1))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 23376, 'dynamic_features': 1})
    ...
    >>> len(dynamic.data_vars)
    10
    # get area of a single station
    >>> dataset.area('2.11.0')
    # get coordinates of two stations
    >>> dataset.area(['2.11.0', '2.28.0'])
    ...
    >>> dataset.get_boundary('2.11.0')    
    
    """
    url = "https://gitlab.com/osilan/shyft-hydro-benchmarking/-/tree/main/shyft-data/Data"

    def __init__(
            self,
            *args,
            **kwargs):
        super().__init__(*args, **kwargs)

        if fiona is None:
            raise ImportError("fiona is required to read shapefiles. Please install fiona.")

        if not os.path.exists(os.path.join(self.path, 'shapefiles')):
            if self.verbosity:
                print("Downloading shapefiles for Norway catchments")
            download_shapefiles(self.path)

        files_to_merge = ['calibration_catchment_86_all_attributes.shp', 'validation_catchment_26_all_attributes.shp']
        
        self.boundary_file = os.path.join(self.path, 'catchment_all_attributes.shp')
        if not os.path.exists(self.boundary_file):
                merge_shapefiles_fiona(
                [os.path.join(self.path, 'shapefiles', f) for f in files_to_merge],
                self.boundary_file,
                gauge_id_attribute_name='stID',
                copy_properties=True
                )

        self._q = self.fetch_q(as_dataframe=True)
        self._stations = self._q.columns.tolist()

    @property
    def boundary_id_map(self):
        return 'gauge_id'

    def stations(self):
        return self._stations

    def fetch_q(
            self, 
            as_dataframe:bool=True,
            ):
        """
        returns the streamflow data of Norway as xarray.Dataset or pandas.DataFrame

        Returns
        -------
        xarray.Dataset or pandas.DataFrame. If as_dataframe is True, returns pandas.DataFrame
        with columns as station codes and index as time. If as_dataframe is False, returns
        xarray.Dataset with station codes as variables and time as dimension.
        """
        fname = 'daily_q.csv' 

        fpath = os.path.join(self.path, fname)

        if not os.path.exists(fpath) or self.overwrite:

            if self.verbosity>1: print(f"Downloading q data at {self.path}")

            q_df = get_shyftnorway_q(outpath=self.path, 
                                     cpus=self.processes or min(get_cpus() - 2, 8),
                                     verbosity=self.verbosity-1
                                     )
        else:
            if self.verbosity: print(f"Reading q data from pre-existing file {fpath}")
            q_df = pd.read_csv(fpath, index_col=0)
            q_df.index = pd.to_datetime(q_df.index)
       
        q_df.index.name = 'time'

        # # because stations are identified by basin_id
        q_df.columns = ['.'.join(col.split('.', 3)[:3]) for col in q_df.columns.tolist()]

        if as_dataframe:
            return q_df
        return xr.Dataset({stn: xr.DataArray(q_df.loc[:, stn]) for stn in q_df.columns})

    def _read_stn_dyn(self, stn:str):
        
        stn_df = self._q.loc[:, [stn]]
        stn_df.rename(columns={stn: observed_streamflow_cms()}, inplace=True)
        stn_df.columns.name = 'dynamic_features'
        stn_df.index.name = 'time'
        return stn_df

    def _static_data(self):
        with fiona.open(self.boundary_file) as src:

            properties = []
            for feature in src:
                prop = feature['properties']

                prop_s = pd.Series({k:v for k,v in prop.items()}, name=prop['gauge_id'])

                properties.append(prop_s)
            static = pd.DataFrame(properties)

        # drop duplicate indices (keep first)
        static = static.loc[~static.index.duplicated(keep='first')]

        # drop WTS_ID_F column
        static.drop(columns=['gauge_id', 'stID',
                             'reference_', 'Station_1',
                             'shyft', 'id_12', 'id_1',
                             'objType', 'stID_1', 'stID2', 
                             'station', 'ID', 'calibratio',
                             'ekspType', 'stSamletID', 'regine_are',
                             'Regulering', 'main_no'], 
                             inplace=True, errors='ignore')

        static.rename(columns={
            'areal_km2': catchment_area(),
            'glacier': 'glacier_%',
        },
            inplace=True)

        return static

def get_shyftnorway_q(
        outpath,
        cpus:int = None,
        verbosity:int=1
):

    project_id=65512664
    folder_path="shyft-data/Data/Q"
    ref="main"

    q_path = os.path.join(outpath, "daily_q")
    os.makedirs(q_path, exist_ok=True)
    api = "https://gitlab.com/api/v4"
    params = {
        "path": folder_path,
        "ref": ref,
        "per_page": 200  # > number of files
    }
    r = requests.get(f"{api}/projects/{project_id}/repository/tree", params=params)
    r.raise_for_status()
    entries = r.json()

    blobs = [e for e in entries if e.get("type") == "blob"]
    if verbosity: print(f"Found {len(blobs)} files.")

    if cpus == 1:
        for e in blobs:
            _download_file(e, q_path, verbosity)
    else:
        cpus = cpus or min(get_cpus() - 2, 8)
        if verbosity:
            print(f"Downloading using {cpus} cpus")
        with cf.ProcessPoolExecutor(max_workers=cpus) as executor:
            futures = [executor.submit(_download_file, e, q_path, verbosity) for e in blobs]
            for i, future in enumerate(cf.as_completed(futures)):
                if verbosity and i % 10 == 0:
                    print(f"Downloaded {i} files")
    
    dfs = []
    for f in os.listdir(q_path):
        fpath = os.path.join(q_path, f)
        df = pd.read_csv(fpath, index_col=0, header=None, sep=' ', dtype={1: 'float32'}, 
                         na_values='-9999.0')

        indexes = []
        for index in df.index:
            index = index.split('/')[0]
            index = pd.Timestamp(index)
            indexes.append(index)
        df.index = pd.DatetimeIndex(indexes)

        # find first valid value
        df = df.iloc[:, 0]

        df = df.loc[df.first_valid_index():df.last_valid_index()]

        df.name = f
        dfs.append(df)
    
    q_df = pd.concat(dfs, axis=1)
    q_df.to_csv(os.path.join(outpath, "daily_q.csv"), index=True, index_label='date')

    return q_df


def _download_file(e, out_dir, verbosity:int=1):
    ref = "main"
    rel_path = e["path"]  # includes folder_path/file
    fname = os.path.basename(rel_path)
    raw_url = f"https://gitlab.com/osilan/shyft-hydro-benchmarking/-/raw/{ref}/{rel_path}"
    resp = requests.get(raw_url, timeout=60)
    if resp.ok:
        with open(os.path.join(out_dir, fname), "wb") as fh:
            fh.write(resp.content)
        if verbosity > 0:
            print("Downloaded", fname)
    else:
        if verbosity > 0:
            print("Failed", fname, resp.status_code)
    return


def download_shapefiles(outpath):

    project_id=65512664
    folder_path="shyft-data/Data/GIS"
    ref="main"

    shp_path = os.path.join(outpath, "shapefiles")
    os.makedirs(shp_path, exist_ok=True)
    api = "https://gitlab.com/api/v4"
    params = {
        "path": folder_path,
        "ref": ref,
        "per_page": 200  # > number of files
    }
    r = requests.get(f"{api}/projects/{project_id}/repository/tree", params=params)
    r.raise_for_status()
    entries = r.json()

    blobs = [e for e in entries if e.get("type") == "blob"]

    for e in blobs:

        ref = "main"
        rel_path = e["path"]  # includes folder_path/file
        fname = os.path.basename(rel_path)
        raw_url = f"https://gitlab.com/osilan/shyft-hydro-benchmarking/-/raw/{ref}/{rel_path}"
        resp = requests.get(raw_url, timeout=60)
        if resp.ok:
            with open(os.path.join(shp_path, fname), "wb") as fh:
                fh.write(resp.content)
    return


class NamalValleyPakistan(_RainfallRunoff):
    """
    Hydro-meteorological sensor network of the Namal Valley watershed, Pakistan,
    following `Sheraz et al., 2025 <https://doi.org/10.1038/s41597-025-05310-3>`_.
    The data is available at `figshare <https://doi.org/10.6084/m9.figshare.28359608>`_
    under a CC-BY-4.0 licence.

    The network has 14 stations. All 14 record **precipitation** and 8 of them
    additionally record **water level**. The two dynamic (time series) features,
    both at a 10-minute time step and kept in their original units, are

        - ``pcp_mm``    : precipitation depth per 10-min interval (mm), 14 stations
        - ``wl_ft_obs`` : water level in feet, 8 stations

    The 8 water-level stations comprise 7 stream gauges (BW, DB, KB, LW, RK, GB, SF)
    reporting stream stage above the local ground/bed, and the Namal Dam gauge (ND)
    reporting the instantaneous lake surface level. Both are in feet but referenced
    to a station-specific datum, so absolute values are not comparable across the
    lake and the streams. This dataset provides no discharge/streamflow, no catchment
    boundaries and no catchment areas (the stations are point sensors, not gauged
    catchments), so :meth:`area`, :meth:`q_mm` and :meth:`get_boundary` are not
    available.

    The 10-minute data spans 2020-11-25 to 2024-09-30. ``pcp_mm`` is the observed
    precipitation depth per 10-minute interval. In the source this incremental
    series is distributed as a "Rate" sheet alongside an equivalent running
    daily-cumulative sheet (``pcp_mm`` is the incremental representation of the same
    measurement) and alongside coarser hourly/daily/monthly aggregates. Only this
    10-minute incremental series and the 10-minute water level are exposed; the
    coarser temporal aggregates are not read. Missing observations (a sensor out of
    service, or gaps in the record) are preserved as NaN, never filled or dropped.

    Each station's series is indexed on the union of that station's precipitation
    and water-level timestamps; rainfall-only stations carry an all-NaN
    ``wl_ft_obs`` column so every station shares the same two-column schema.

    Static features are the published station inventory: ``lat``, ``long``,
    ``elev_gauge_m`` (altitude), the full station name, station type (``R`` for
    rainfall only, ``R, WL`` for rainfall and water level) and operational status.

    Fetching the two dynamic features of all 14 stations (~5.7 million values)
    takes about 1 second.

    Examples
    --------
    >>> from aqua_fetch import NamalValleyPakistan
    >>> ds = NamalValleyPakistan()
    >>> len(ds.stations())
    14
    >>> ds.dynamic_features
    ['pcp_mm', 'wl_ft_obs']
    >>> ds.static_features
    ['station_name', 'station_type', 'lat', 'long', 'elev_gauge_m', 'deploy_date', 'status']
    ... # dynamic data is returned as a dict {station_id: DataFrame}
    >>> _, dyn = ds.fetch('ND', as_dataframe=True)
    >>> dyn['ND'].shape
    (202464, 2)
    ... # only precipitation of two stations
    >>> _, dyn = ds.fetch(['BS', 'LW'], dynamic_features='pcp_mm', as_dataframe=True)
    >>> dyn['BS'].shape
    (202464, 1)
    ... # coordinates of all stations
    >>> ds.stn_coords().shape
    (14, 2)
    """

    url = {
        "Namal_Catchment_Precipitation_Data.xlsx":
            "https://ndownloader.figshare.com/files/52177559",
        "Namal_Catchment_Stream_Levels_Data.xlsx":
            "https://ndownloader.figshare.com/files/52177562",
        "Namal_Catchment_Lake_Level_Data.xlsx":
            "https://ndownloader.figshare.com/files/52177556",
        "Network_Metadata-Namal.xlsx":
            "https://ndownloader.figshare.com/files/54388085",
    }

    # source excel file / sheet holding the observed 10-minute series
    _PRECIP_XLSX = "Namal_Catchment_Precipitation_Data.xlsx"
    _PRECIP_SHEET = "10- minute precipitation Rate"   # incremental mm per 10-min
    _STREAM_XLSX = "Namal_Catchment_Stream_Levels_Data.xlsx"
    _STREAM_SHEET = "10-minutes Stream Level"
    _LAKE_XLSX = "Namal_Catchment_Lake_Level_Data.xlsx"
    _LAKE_SHEET = "10-minutes Lake Level"
    _META_XLSX = "Network_Metadata-Namal.xlsx"
    _LAKE_STATION = "ND"

    def __init__(
            self,
            path: str = None,
            to_netcdf: bool = False,
            overwrite: bool = False,
            verbosity: int = 1,
            **kwargs):
        """
        parameters
        ----------
        path : str
            directory in which the data is/will be stored. If it already contains
            the dataset, it will not be downloaded again.
        to_netcdf : bool (default=False)
            kept for API compatibility with other rainfall-runoff datasets; the
            10-minute data is served from csv files and is not converted to netcdf.
        overwrite : bool (default=False)
            if True, previously downloaded/processed files are removed and the
            data is downloaded and processed afresh.
        verbosity : int (default=1)
        """
        super().__init__(path=path, timestep='10min', to_netcdf=to_netcdf,
                         overwrite=overwrite, verbosity=verbosity, **kwargs)

        self._static_df = None
        self._precip = self._stream = self._lake = None  # lazy dynamic caches

        # gate on the presence of the *processed* csv (not the source xlsx): once
        # the csv exist, reads need only pandas, so we neither re-download nor
        # re-read the excel files (and, if remove_zip, the source xlsx are gone).
        if self.overwrite or not all(os.path.exists(f) for f in self._processed_files):
            self._download(overwrite=self.overwrite)
            self._process(overwrite=self.overwrite)
            if self.remove_zip:
                self._remove_source_files()

    # ------------------------------------------------------------------ #
    # paths of the processed (csv) files
    # ------------------------------------------------------------------ #
    @property
    def _precip_csv(self) -> str:
        return os.path.join(self.path, "precipitation_mm_10min.csv")

    @property
    def _stream_csv(self) -> str:
        return os.path.join(self.path, "stream_level_ft_10min.csv")

    @property
    def _lake_csv(self) -> str:
        return os.path.join(self.path, "lake_level_ft_10min.csv")

    @property
    def _static_csv(self) -> str:
        return os.path.join(self.path, "static.csv")

    @property
    def _daterange_txt(self) -> str:
        return os.path.join(self.path, "date_range.txt")

    @property
    def _processed_files(self) -> List[str]:
        return [self._precip_csv, self._stream_csv, self._lake_csv,
                self._static_csv, self._daterange_txt]

    # ------------------------------------------------------------------ #
    # one-time conversion of the source .xlsx files to .csv
    # ------------------------------------------------------------------ #
    def _process(self, overwrite: bool = False):
        """converts the observed 10-minute sheets of the source excel files into
        csv files so that subsequent reads use only the minimal requirements
        (pandas). Reading the excel files needs ``openpyxl`` which is why this is
        done only once."""

        if overwrite:
            for f in self._processed_files:
                if os.path.exists(f):
                    os.remove(f)

        if all(os.path.exists(f) for f in self._processed_files):
            if self.verbosity > 1:
                print(f"processed csv files already exist in {self.path}")
            return

        try:
            import openpyxl  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "openpyxl is required to read the source .xlsx files of "
                "NamalValleyPakistan for the (one-time) conversion to csv. "
                "Please install it with `pip install openpyxl`.")

        if self.verbosity:
            print("converting Namal Valley excel files to csv (one-time)")

        precip = self._read_10min_sheet(self._PRECIP_XLSX, self._PRECIP_SHEET)
        stream = self._read_10min_sheet(self._STREAM_XLSX, self._STREAM_SHEET)
        lake = self._read_10min_sheet(self._LAKE_XLSX, self._LAKE_SHEET)
        static = self._read_metadata()

        # write atomically (temp file + os.replace) so an interrupted conversion
        # cannot leave a half-written csv that later passes the "exists" gate.
        self._atomic_to_csv(precip, self._precip_csv, "Timestamp")
        self._atomic_to_csv(stream, self._stream_csv, "Timestamp")
        self._atomic_to_csv(lake, self._lake_csv, "Timestamp")
        self._atomic_to_csv(static, self._static_csv, "station")

        # global temporal extent, derived from the data itself. Written last so
        # its presence signals a complete conversion.
        start = min(precip.index.min(), stream.index.min(), lake.index.min())
        end = max(precip.index.max(), stream.index.max(), lake.index.max())
        tmp = self._daterange_txt + ".tmp"
        with open(tmp, "w") as fh:
            fh.write(f"{start.isoformat()}\n{end.isoformat()}\n")
        os.replace(tmp, self._daterange_txt)
        return

    @staticmethod
    def _atomic_to_csv(df: pd.DataFrame, fpath: str, index_label: str):
        tmp = fpath + ".tmp"
        df.to_csv(tmp, index_label=index_label)
        os.replace(tmp, fpath)

    @staticmethod
    def _warn_flat_series(precip: pd.DataFrame):
        """warns (unconditionally) about precipitation gauges whose entire record
        is zero or missing (e.g. a dead/uncalibrated gauge). The data is kept
        unchanged; this only flags it so the user is not misled."""
        flat = []
        for col in precip.columns:
            s = precip[col].dropna()
            if len(s) == 0 or (s == 0).all():
                flat.append(col)
        if flat:
            warnings.warn(
                f"precipitation for station(s) {sorted(flat)} is entirely "
                f"zero or missing; the values are kept unchanged.", UserWarning)

    def _remove_source_files(self):
        """removes the source .xlsx once the csv cache is built (remove_zip)."""
        for fname in self.url:
            fpath = os.path.join(self.path, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
        return

    def _read_10min_sheet(self, fname: str, sheet: str) -> pd.DataFrame:
        """reads a 10-minute sheet whose first data column is the timestamp and
        whose first three rows below the header carry latitude, longitude and the
        measurement unit."""
        df = pd.read_excel(os.path.join(self.path, fname), sheet_name=sheet, header=0)
        # rows 0,1,2 hold Latitude, Longitude, Unit (verified against the file)
        assert str(df.iloc[0, 0]).strip().lower() == "latitude", \
            f"unexpected layout in {fname}:{sheet}"
        df = df.iloc[3:].copy()
        df = df.rename(columns={df.columns[0]: "Timestamp"})
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
        df = df.set_index("Timestamp")
        # measurement values are kept at full (float64) precision
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df[~df.index.duplicated(keep="first")].sort_index()
        return df

    def _read_metadata(self) -> pd.DataFrame:
        """parses the station inventory (14 stations) from the metadata file."""
        import re
        meta = pd.read_excel(os.path.join(self.path, self._META_XLSX),
                             sheet_name="metadata", header=None)
        rows = []
        # station table occupies rows 3..16, columns 1..9 (verified)
        for r in range(3, 17):
            name = meta.iat[r, 2]
            if pd.isna(name):
                continue
            m = re.search(r"\(([^)]+)\)", str(name))
            code = m.group(1).strip()
            rows.append({
                "station": code,
                "station_name": str(name).strip(),
                "station_type": str(meta.iat[r, 3]).strip(),
                gauge_latitude(): float(meta.iat[r, 4]),
                gauge_longitude(): float(meta.iat[r, 5]),
                gauge_elevation_meters(): float(meta.iat[r, 6]),
                "deploy_date": str(meta.iat[r, 7]).strip(),
                "status": str(meta.iat[r, 9]).strip(),
            })
        static = pd.DataFrame(rows).set_index("station")
        self._warn_duplicates(static)
        return static

    @staticmethod
    def _warn_duplicates(static: pd.DataFrame):
        """warns (without excluding) about stations sharing name + rounded
        coordinates. Cheap: only 14 stations."""
        key = list(zip(static["station_name"],
                       static[gauge_latitude()].round(4),
                       static[gauge_longitude()].round(4)))
        seen, dups = set(), set()
        for k in key:
            if k in seen:
                dups.add(k[0])
            seen.add(k)
        if dups:
            warnings.warn(f"Duplicate stations (name + coordinates) found: {sorted(dups)}",
                          UserWarning)

    # ------------------------------------------------------------------ #
    # static data
    # ------------------------------------------------------------------ #
    def _static_data(self) -> pd.DataFrame:
        if self._static_df is None:
            df = pd.read_csv(self._static_csv, index_col="station")
            df.index = df.index.astype(str)
            self._static_df = df
        return self._static_df.copy()

    def stations(self) -> List[str]:
        return self._static_data().index.tolist()

    @property
    def static_features(self) -> List[str]:
        return self._static_data().columns.tolist()

    # ------------------------------------------------------------------ #
    # dynamic data
    # ------------------------------------------------------------------ #
    @property
    def dynamic_features(self) -> List[str]:
        return [total_precipitation(), observed_water_level_ft()]

    @staticmethod
    def _load_csv(fpath: str) -> pd.DataFrame:
        df = pd.read_csv(fpath, index_col="Timestamp")
        # the source is 10-min aligned except for a couple of off-grid sensor
        # timestamps (e.g. ``2024-03-31 23:49:59.995``) which are preserved as-is
        df.index = pd.to_datetime(df.index, format="ISO8601")
        df.index.name = "time"
        return df

    # each big csv is parsed once and cached; per-station reads then slice a
    # single column (avoids re-parsing the ~200k-row timestamp index per station)
    @property
    def _precip_df(self) -> pd.DataFrame:
        if self._precip is None:
            self._precip = self._load_csv(self._precip_csv)
            # surface the advisory on first use, so it is seen on the common
            # cached-read path too (not only during the one-time conversion)
            self._warn_flat_series(self._precip)
        return self._precip

    @property
    def _stream_df(self) -> pd.DataFrame:
        if self._stream is None:
            self._stream = self._load_csv(self._stream_csv)
        return self._stream

    @property
    def _lake_df(self) -> pd.DataFrame:
        if self._lake is None:
            self._lake = self._load_csv(self._lake_csv)
        return self._lake

    @property
    def _stream_stations(self) -> List[str]:
        # water-level stream stations, derived from the stream-level csv columns
        return self._stream_df.columns.tolist()

    def _read_stn_dyn(self, stn: str) -> pd.DataFrame:
        pcp = self._precip_df[[stn]].copy()
        pcp.columns = [total_precipitation()]

        if stn == self._LAKE_STATION:
            wl = self._lake_df[[stn]].copy()
        elif stn in self._stream_stations:
            wl = self._stream_df[[stn]].copy()
        else:
            wl = None

        if wl is not None:
            wl.columns = [observed_water_level_ft()]
            df = pcp.join(wl, how="outer").sort_index()
        else:
            df = pcp
            df[observed_water_level_ft()] = np.nan

        df = df[[total_precipitation(), observed_water_level_ft()]]
        df.index.name = "time"
        df.columns.name = "dynamic_features"
        return df

    def _read_dynamic(
            self,
            stations,
            dynamic_features,
            st: Union[str, pd.Timestamp] = None,
            en: Union[str, pd.Timestamp] = None,
    ) -> Dict[str, pd.DataFrame]:
        """reads the dynamic data of the given stations. Overridden to slice the
        (cached, in-memory) per-variable tables serially: for this small 14-station
        network a bulk read is faster than a process pool, and it avoids pickling
        the cached tables to worker processes."""
        st, en = self._check_length(st, en)
        dyn_feats = validate_attributes(dynamic_features, self.dynamic_features,
                                        'dynamic_features')
        stations = validate_attributes(stations, self.stations(), 'stations')

        dyn = {}
        for stn in stations:
            stn_df = self._read_stn_dyn(stn).loc[st:en, dyn_feats]
            stn_df.columns.name = 'dynamic_features'
            stn_df.index.name = 'time'
            dyn[stn] = stn_df
        return dyn

    # ------------------------------------------------------------------ #
    # temporal extent (derived from the data during processing)
    # ------------------------------------------------------------------ #
    @property
    def start(self) -> pd.Timestamp:
        with open(self._daterange_txt) as fh:
            return pd.Timestamp(fh.readline().strip())

    @property
    def end(self) -> pd.Timestamp:
        with open(self._daterange_txt) as fh:
            fh.readline()
            return pd.Timestamp(fh.readline().strip())

    # ------------------------------------------------------------------ #
    # unavailable for this point-sensor network (no catchments)
    # ------------------------------------------------------------------ #
    def area(self, stations: Union[str, List[str]] = 'all') -> pd.Series:
        raise NotImplementedError(
            "NamalValleyPakistan is a point-sensor network with no catchment "
            "areas.")

    def q_mm(self, stations: Union[str, List[str]] = "all") -> pd.DataFrame:
        raise NotImplementedError(
            "NamalValleyPakistan provides water level (ft), not discharge, so "
            "q_mm is unavailable.")

    def get_boundary(self, catchment_id: str, to_wgs84: bool = True):
        raise NotImplementedError(
            "NamalValleyPakistan provides no catchment boundaries.")