
__all__ = ["DraixBleone", "JialingRiverChina"]

import os
from typing import List
import concurrent.futures as cf

import requests
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


class NamalValleyPakistan:
    """
    Dataset containing hydrological and meteorological data for the Namal Valley catchment 
    in Pakistan. For more information on this data see `Sheraz et al., 2025 <https://doi.org/10.1038/s41597-025-05310-3>`_.
    The data consists of observations of precipitation (from 14 sensors) and water level (from 8 sensors) from 
    2022 to 2024 at 10 minutes intervals.

    The dataset is available at `Figshare link <https://doi.org/10.6084/m9.figshare.28359608>`_.

    """

    url = "https://springernature.figshare.com/ndownloader/articles/28359608/versions/1"