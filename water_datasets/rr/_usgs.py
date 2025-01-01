
import os
import re
import time
import requests
from io import StringIO
from typing import List, Union
from requests.exceptions import JSONDecodeError
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .camels import Camels
from .._backend import netCDF4, shapefile, xarray as xr
from ..utils import get_cpus
from ..utils import check_attributes
from ._hysets import HYSETS

DAILY_START = "1820-01-01"
DAILY_END = "2024-05-30"
HOURLY_START = "1910-01-01"
HOURLY_END = "2024-05-30"


class USGS(Camels):
    """
    This class handles the hydrometeorological data for the USA. The daily and 
    hourly discharge data is downloaded
    from `usgs/nwis website <https://waterservices.usgs.gov/nwis/dv>`_ . 
    The data is optionally stored in a netCDF 
    file if xarray is available. Currently the data is downloaded for only those 
    sites/catchments that are in the `HYSETS database <https://doi.org/10.1038/s41597-020-00583-2>`_. 
    This is because the catchment boundaries 
    are taken from HYSETS database using :py:class:`water_quality.HYSETS`.

    For hourly timestep, "iv" service is used to download the instantaneous data 
    which is then resampled to hourly data. Data with only ``A, [92]``, ``A, [91]``, 
    ``A, [93]``, ``A, e``, ``A`` flags is used.
    For daily streamflow, "dv" service is used to download the data. 
    In this case, the data with only ``A`` and ``A, e`` flags is used.
    """

    def __init__(
            self, 
            path:Union[str, os.PathLike] = None, 
            hysets_path: Union[str, os.PathLike] = None,
            verbosity:int = 1, 
            **kwargs
            ):
        """
        Parameters
        ----------
        path : str
            Path to store the data
        """

        super().__init__(path, verbosity=verbosity, **kwargs)

        if hysets_path is None:
            hysets_path = os.path.join(os.path.dirname(self.path), 'HYSETS')
            if os.path.exists(hysets_path):
                self.hysets_path = hysets_path

        self.hysets = HYSETS(path = hysets_path, verbosity=verbosity)
        self.hysets_path = self.hysets.path

        self._stations = self.__stations()
        self.metadata = maybe_make_and_get_metadata(self.path, self.stations())

        self._static_features = self.__static_features()
        self.boundary_file = self.hysets.boundary_file

    @property
    def start(self)->str:
        return "19500101"

    @property
    def end(self)->str:
        return "20181231"

    @property
    def dynamic_features(self)->List[str]:
        return ['obs_q_cms'] + self.hysets.dynamic_features[1:]

    def stations(self)->List[str]:
        return self._stations

    @property
    def static_features(self)->List[str]:
        return self._static_features

    @property
    def _q_name(self)->str:
        return 'obs_q_cms'

    def get_boundary(
            self,
            catchment_id: str,
            as_type: str = 'numpy'
    ):
        """
        returns boundary of a catchment in a required format

        Parameters
        ----------
        catchment_id : str
            name/id of catchment
        as_type : str
            'numpy' or 'geopandas'
        
        Examples
        --------
        >>> from water_datasets import USGS
        >>> dataset = USGS()
        >>> dataset.get_boundary(dataset.stations()[0])
        """

        if shapefile is None:
            raise ModuleNotFoundError("shapefile module is not installed. Please install it to use boundary file")

        from shapefile import Reader

        bndry_sf = Reader(self.boundary_file)
        bndry_shp = bndry_sf.shape(self.hysets.bndry_id_map[catchment_id])

        bndry_sf.close()

        xyz = np.array(bndry_shp.points)

        xyz = self.transform_coords(xyz)

        return xyz
    
    def area(
            self,
            stations: Union[str, List[str]] = 'all',
    ) ->pd.Series:
        """
        Returns area_gov (Km2) of all catchments as pandas series

        parameters
        ----------
        stations : str/list
            name/names of stations. Default is None, which will return
            area of all stations

        Returns
        --------
        pd.Series
            a pandas series whose indices are catchment ids and values
            are areas of corresponding catchments.

        Examples
        ---------
        >>> from water_datasets import USGS
        >>> dataset = USGS()
        >>> dataset.area()  # returns area of all stations
        >>> dataset.area('912101A')  # returns area of station whose id is 912101A
        >>> dataset.area(['912101A', '12388200'])  # returns area of two stations
        """
        stations = check_attributes(stations, self.stations(), 'stations')

        area = self.metadata['drain_area_va']

        area.name = 'area'
        return area.loc[stations]

    def fetch_static_features(
            self,
            stations: Union[str, List[str]] = "all",
            features:Union[str, List[str]] = "all",
            st=None,
            en=None,
            as_ts=False
    ) -> pd.DataFrame:
        """
        returns static atttributes of one or multiple stations

        Parameters
        ----------
            stations : str
                name/id of station of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.
            st :
            en :
            as_ts :

        Examples
        ---------
        >>> from water_datasets import USGS
        >>> dataset = USGS()
        get the names of stations
        >>> stns = dataset.stations()
        >>> len(stns)
            12004
        get all static data of all stations
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (12004, 27)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('01010070')
        >>> static_data.shape
           (1, 27)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['Drainage_Area_km2', 'Elevation_m'])
        >>> static_data.shape
           (12004, 2)
        """
        stations = check_attributes(stations, self.stations())
        map_ = self.hysets.OfficialID_WatershedID_map
        stations = [int(map_[stn]) for stn in stations]        
        static_feats = self.hysets.fetch_static_features(stations, features, st, en, as_ts)
        static_feats.set_index('Official_ID', inplace=True)
        return static_feats

    def stn_coords(
            self,
            stations:Union[str, List[str]] = 'all'
    ) ->pd.DataFrame:
        """
        returns coordinates of stations as DataFrame
        with ``long`` and ``lat`` as columns.

        Parameters
        ----------
        stations :
            name/names of stations. If not given, coordinates
            of all stations will be returned.

        Returns
        -------
        coords :
            pandas DataFrame with ``long`` and ``lat`` columns.
            The length of dataframe will be equal to number of stations
            wholse coordinates are to be fetched.

        Examples
        --------
        >>> dataset = USGS()
        >>> dataset.stn_coords() # returns coordinates of all stations
        >>> dataset.stn_coords('01010000')  # returns coordinates of station whose id is 912101A
        >>> dataset.stn_coords(['01010000', '01010070'])  # returns coordinates of two stations
        """
        stations = check_attributes(stations, self.stations(), 'stations')
        coords = self.metadata.loc[:, ['dec_long_va', 'dec_lat_va']]
        coords.rename(columns={'dec_long_va': 'long', 'dec_lat_va': 'lat'}, inplace=True)
        return coords.loc[stations, :]

    def fetch_stations_features(
            self,
            stations: list,
            dynamic_features: Union[str, list, None] = 'all',
            static_features: Union[str, list, None] = None,
            st=None,
            en=None,
            as_dataframe: bool = False,
            **kwargs
    ):
        """
        returns features of multiple stations

        Examples
        --------
        >>> from water_datasets import USGS
        >>> dataset = USGS()
        >>> stations = dataset.stations()[0:3]
        >>> features = dataset.fetch_stations_features(stations)
        """
        stations = check_attributes(stations, self.stations(), 'stations')
        #stations = [int(stn) for stn in stations]

        if dynamic_features is not None:

            dyn = self._fetch_dynamic_features(stations=stations,
                                               dynamic_features=dynamic_features,
                                               as_dataframe=as_dataframe,
                                               st=st,
                                               en=en,
                                               **kwargs
                                               )

            if static_features is not None:  # we want both static and dynamic
                to_return = {}
                static = self._fetch_static_features(station=stations,
                                                     static_features=static_features,
                                                     st=st,
                                                     en=en,
                                                     **kwargs
                                                     )
                to_return['static'] = static
                to_return['dynamic'] = dyn
            else:
                to_return = dyn

        elif static_features is not None:
            # we want only static
            to_return = self._fetch_static_features(
                station=stations,
                static_features=static_features,
                **kwargs
            )
        else:
            raise ValueError(f"static features are {static_features} and dynamic features are {dynamic_features}")

        return to_return

    def _fetch_dynamic_features(
            self,
            stations: list,
            dynamic_features = 'all',
            st=None,
            en=None,
            as_dataframe=False,
            as_ts=False
    ):
        """Fetches dynamic features of station."""
        st, en = self._check_length(st, en)
        features = check_attributes(dynamic_features, self.dynamic_features.copy(), 'dynamic_features')

        daily_q = None

        if 'obs_q_cms' in features:
            # todo:  we are reading all file even for one station
            daily_q = self._q(as_dataframe)
            if isinstance(daily_q, xr.Dataset):
                daily_q = daily_q.sel(time=slice(st, en))[stations]
            else:
                daily_q = daily_q.loc[st:en, stations]
            
            features.remove('obs_q_cms')

        if len(features) == 0:
            return daily_q

        stations_ = [int(self.hysets.OfficialID_WatershedID_map[stn]) for stn in stations]
        data = self.hysets._fetch_dynamic_features(stations_, features, st, en, as_dataframe)

        if daily_q is not None:
            if isinstance(daily_q, xr.Dataset):
                assert isinstance(data, xr.Dataset), "xarray dataset not supported"
                data = data.rename({int(k)-1:v for k,v in self.hysets.WatershedID_OfficialID_map.items() if v in stations})

                # first create a new dimension in daily_q named dynamic_features
                daily_q = daily_q.expand_dims({'dynamic_features': ['obs_q_cms']})
                data = xr.concat([data, daily_q], dim='dynamic_features')
            else:
                # -1 because the data in .nc files hysets starts with 0
                data.rename(columns={int(k)-1:v for k,v in self.hysets.WatershedID_OfficialID_map.items()}, inplace=True)
                assert isinstance(data.index, pd.MultiIndex)
                # data is multiindex dataframe but daily_q is not
                # first make daily_q multiindex
                daily_q['dynamic_features'] = 'daily_q'
                daily_q.set_index('dynamic_features', append=True, inplace=True)
                daily_q = daily_q.reorder_levels(['time', 'dynamic_features'])
                data = pd.concat([data, daily_q], axis=0).sort_index()
                #data = data.join(daily_q, how='outer')

        return data

    def _fetch_static_features(
            self,
            station="all",
            static_features: Union[str, list] = 'all',
            st=None,
            en=None,
            as_ts=False
    )->pd.DataFrame:
        """Fetches static features of station."""
        if self.verbosity>1:
            print('fetching static features')
        stations = check_attributes(station, self.stations(), 'stations')
        map_ = self.hysets.OfficialID_WatershedID_map
        stations = [int(map_[stn]) for stn in stations]
        static_feats = self.hysets.fetch_static_features(stations, static_features, st, en, as_ts).copy()
        static_feats = static_feats.set_index('Official_ID')
        return static_feats

    def __static_features(self)->List[str]:
        if self.verbosity>1:
            print('getting static features from hysets')
        static_feats =  self.hysets.static_features
        static_feats.remove('Official_ID')
        return static_feats

    def _q(self, as_dataframe:bool=None, read_csv_kwargs:dict=None):

        if netCDF4 is None:
            ext = 'csv'
        else:
            ext = 'nc'
        
        if self.timestep.lower().startswith('d'):
            fname = f'daily_q.{ext}'
        else:
            fname = f'hourly_q.{ext}'

        if self.verbosity>1:
            print(f'reading {fname}')

        fpath = os.path.join(self.path, fname)

        if not os.path.exists(fpath):
            print(f"{fpath} not found. Downloading data storing it in {fpath}")
            self._make_csv(cpus=None)

        if ext == 'csv':
            return pd.read_csv(fpath, index_col=0, **read_csv_kwargs)
        else:
            if as_dataframe:
                return xr.open_dataset(fpath).to_dataframe()
            return xr.open_dataset(fpath)        

    def __stations(self)->List[str]:
        if self.verbosity>1:
            print('getting stations')
        dataset = self._q(read_csv_kwargs=dict(nrows=2))

        if isinstance(dataset, xr.Dataset):
            # get names of all variables
            return list(dataset.data_vars.keys())
        else:
            return dataset.columns.tolist()

    def _make_csv(
            self,
            cpus:int = None,
            ):

        df = pd.read_csv(os.path.join(self.hysets_path, "HYSETS_watershed_properties.txt"),
                 sep=";")

        sites = df.loc[df['Source']=='USGS']['Official_ID']

        if cpus is None:
            cpus = get_cpus() - 2
        
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
        make_daily_q(self.path, sites, cpus)
        print(f"Downloaded daily data and stored in {self.path}/daily_q.nc")

        #make_hourly_q(self.path, sites[9000:10000], cpus=cpus)
        #print(f"Downloaded hourly data and stored in {self.path}/hourly_q.nc")

        return


def download_metadata(
        site:str
        )->pd.DataFrame:
    
    try:
        metadata = _download_metadata(site)
    except (ValueError, IndexError):
        print(f"Site: {site} ValueError/IndexError")
        # create dataframe with nan values
        metadata = pd.DataFrame(
            np.array([np.nan, np.nan, np.nan]).reshape(1,3),
            columns=['dec_lat_va', 'dec_long_va', 'drain_area_va']
        )

    metadata = metadata[['dec_lat_va', 'dec_long_va', 'drain_area_va']]
    metadata.index = [site]
    return metadata


def maybe_make_and_get_metadata(        
        path:str,
        sites:List[str], 
        cpus:int=None
        )->pd.DataFrame:

    cpus = get_cpus()-2 if cpus is None else cpus

    fpath = os.path.join(path, 'metadata.csv')
    if os.path.exists(fpath):
        return pd.read_csv(
            fpath, 
            index_col='site_no',
            dtype={'site_no': str, "dec_lat_va": float, "dec_long_va": float, "drain_area_va": float}) 

    print(f"Downloading metadata for {len(sites)} sites using {cpus} cpus")

    start = time.time()
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        data = executor.map(download_metadata, sites)

    total = round((time.time() - start)/60, 2)
    print(f"Time taken to download metadata: {total} mins with {cpus} cpus")
    start = time.time()

    metadata = pd.concat(list(data))

    # convert drain_area_va from sq miles to sq km
    metadata['drain_area_va'] = metadata['drain_area_va'] * 2.58999
    
    metadata.index.name = "site_no"

    metadata.to_csv(fpath, index_label='site_no')
    return metadata


def make_daily_q(
        path:str,
        sites:List[str], 
        cpus:int,
        verbosity:int=1
        ):

    if verbosity: print(f"Downloading daily data for {len(sites)} sites using {cpus} cpus")

    daily_files_path = os.path.join(path, 'daily_files')
    if not os.path.exists(daily_files_path):
        os.makedirs(daily_files_path)

    start = time.time()
    with ProcessPoolExecutor(max_workers=cpus) as executor:
        data = executor.map(download_daily_record, sites)

    total = round((time.time() - start)/60, 2)
    
    if verbosity: print(f"Time taken to download data: {total} mins with {cpus} cpus")
    start = time.time()

    if netCDF4 is None:
        save_daily_q_as_csv(path, data)
    else:
        save_daily_q_as_nc(path, data)


    total = round((time.time() - start)/60, 2)
    print(f"Time taken: to store {total} mins")
    return


def save_daily_q_as_csv(path, data):
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(path, 'daily_q.csv'), index=True)
    return

def save_daily_q_as_nc(path, data):
    from netCDF4 import Dataset, date2num

    ncfile = Dataset(os.path.join(path, 'daily_q.nc'), mode='w', format='NETCDF4')

    _ = ncfile.createDimension('time', None)  # unlimited dimension

    new_idx = pd.date_range(start=DAILY_START, end=DAILY_END, freq='D')
    time_var = ncfile.createVariable('time', 'f8', ('time',))
    time_var.units = 'days since 1820-01-01 00:00:00'
    time_var.calendar = 'gregorian'
    time_var[:] = date2num(new_idx.to_pydatetime(), units=time_var.units, calendar=time_var.calendar)

    for idx, ts in enumerate(data):
        # create a variable for each column
        col_var = ncfile.createVariable(str(ts.name), 
                                        datatype='f4', 
                                        dimensions=('time',),
                                        complevel=4, 
                                        compression='zlib',
                                        shuffle=True,
                                        least_significant_digit=4,
                                        )
        # ts may have different index than new_idx/tim_demension, so we need to reindex
        new_ts = ts.reindex(index=new_idx)

        col_var[:] = new_ts.values
        col_var.units = "cms"
        col_var.description = "daily discharge"

        if idx % 100 == 0:
            print(f"Saved data for {idx} sites in .nc file")

    ncfile.close()

    return


def _read_json(json):
    """
    Following code is taken and modified after dataretrieval.utils

    Reads a NWIS Water Services formatted JSON into a ``pandas.DataFrame``.

    Parameters
    ----------
    json: dict
        A JSON dictionary response to be parsed into a ``pandas.DataFrame``

    Returns
    -------
    df: ``pandas.DataFrame``
        Times series data from the NWIS JSON
    md: :obj:`dataretrieval.utils.Metadata`
        A custom metadata object

    """
    merged_df = pd.DataFrame(columns=["site_no", "datetime"])

    site_list = [
        ts["sourceInfo"]["siteCode"][0]["value"] for ts in json["value"]["timeSeries"]
    ]

    # create a list of indexes for each change in site no
    # for example, [0, 21, 22] would be the first and last indeces
    index_list = [0]
    index_list.extend(
        [i + 1 for i, (a, b) in enumerate(zip(site_list[:-1], site_list[1:])) if a != b]
    )
    index_list.append(len(site_list))

    for i in range(len(index_list) - 1):
        start = index_list[i]  # [0]
        end = index_list[i + 1]  # [21]

        # grab a block containing timeseries 0:21,
        # which are all from the same site
        site_block = json["value"]["timeSeries"][start:end]
        if not site_block:
            continue

        site_no = site_block[0]["sourceInfo"]["siteCode"][0]["value"]
        site_df = pd.DataFrame(columns=["datetime"])

        for timeseries in site_block:
            param_cd = timeseries["variable"]["variableCode"][0]["value"]
            # check whether min, max, mean record XXX
            option = timeseries["variable"]["options"]["option"][0].get("value")

            # loop through each parameter in timeseries, then concat to the merged_df
            for parameter in timeseries["values"]:
                col_name = param_cd
                method = parameter["method"][0]["methodDescription"]

                # if len(timeseries['values']) > 1 and method:
                if method:
                    # get method, format it, and append to column name
                    method = method.strip("[]()").lower()
                    col_name = f"{col_name}_{method}"

                if option:
                    col_name = f"{col_name}_{option}"

                record_json = parameter["value"]

                if not record_json:
                    # no data in record
                    continue
                # should be able to avoid this by dumping
                record_json = str(record_json).replace("'", '"')

                # read json, converting all values to float64 and all qualifiers
                # Lists can't be hashed, thus we cannot df.merge on a list column
                record_df = pd.read_json(
                    StringIO(record_json),
                    orient="records",
                    dtype={"value": "float64", "qualifiers": "unicode"},
                    convert_dates=False,
                )

                record_df["qualifiers"] = (
                    record_df["qualifiers"].str.strip("[]").str.replace("'", "")
                )

                record_df.rename(
                    columns={
                        "value": col_name,
                        "dateTime": "datetime",
                        "qualifiers": col_name + "_cd",
                    },
                    inplace=True,
                )

                site_df = site_df.merge(record_df, how="outer", on="datetime")

        # end of site loop
        site_df["site_no"] = site_no
        merged_df = pd.concat([merged_df, site_df])

    # convert to datetime, normalizing the timezone to UTC when doing so
    if "datetime" in merged_df.columns:
        merged_df["datetime"] = pd.to_datetime(merged_df["datetime"], utc=True)

    return merged_df


def download_daily_q_nwis(
        site:str = '14105700', 
        start = '1820-01-01', 
        end='2024-05-30'
        )->pd.DataFrame:

    response = requests.get(
        "https://waterservices.usgs.gov/nwis/dv", 
        params={'format': 'json', 'parameterCd': '00060', 'sites': site, 'startDT': start, 'endDT': end, 'multi_index': None}, 
        headers={"user-agent": f"python-dataretrieval/1.0.11"}, verify=True)

    if response.status_code in [400, 404, 414]:
        raise ValueError(f"Bad Request, check that your parameters are correct. URL: {response.url}")

    try:
        site_data = _read_json(response.json())
    except JSONDecodeError:
        site_data = pd.DataFrame()
        print(f"Site: {site} JSONDecodeError")

    if 'datetime' in site_data.columns: 
        site_data.index = pd.to_datetime(site_data.pop('datetime'))
    
    return site_data


def download_daily_record(
        site:str,
        )->pd.Series:

    # todo: should we store the data in csv files so that we don't have to download it again?
    # but using 110 cores for all 12004 sites takes around 5 minutes which is not that bad
    # given that it has to happen only once!
    #fpath = os.path.join(path, f"{site}.csv")

    # if os.path.exists(fpath):
    #     return pd.read_csv(fpath, index_col=0)

    site_data = download_daily_q_nwis(site, 
                    start="1820-01-01",  # DAILY_START
                    end="2024-05-30",    # DAILY_END
                    )
    if f'00060_Mean' in site_data.columns:
        # get data for stations which have A in 00060_Mean_cd column
        site_data = site_data[site_data['00060_Mean_cd'].isin(["A", "A, e"])]
        site_data = site_data['00060_Mean']
    elif '00060_upstream site_Mean' in site_data.columns:
        site_data = site_data[site_data['00060_upstream site_Mean_cd'].isin(["A", "A, e"])]
        site_data = site_data['00060_upstream site_Mean']
    elif "00060_upstream of dam_Mean" in site_data.columns:
        # "01399100" has this column
        site_data = site_data[site_data['00060_upstream of dam_Mean_cd'].isin(["A", "A, e"])]
        site_data = site_data["00060_upstream of dam_Mean"]
    elif "00060_2_Mean" in site_data.columns:
        # "01401000" has this column
        site_data = site_data[site_data['00060_2_Mean_cd'].isin(["A", "A, e"])]
        site_data = site_data["00060_2_Mean"]
    elif "00060_published_Mean" in site_data.columns:
        # "02129590" has this column
        site_data = site_data[site_data['00060_published_Mean_cd'].isin(["A", "A, e"])]
        site_data = site_data["00060_published_Mean"]
    elif len(site_data) == 0:
        # return empty series
        site_data = pd.Series(name=f"{site}__empty",
                              index = pd.date_range(start="2024-01-01", end="2024-05-30", freq='D')
                              )
    else:
        #print(f"Site: {site} has {site_data.columns}")
        site_data = pd.Series(name=f"{site}__empty",
                            index = pd.date_range(start="2024-01-01", end="2024-05-30", freq='D')
                            )
    
    site_data.name = site
    site_data.index = site_data.index.tz_localize(None)
    return site_data * 0.028316847 # convert cfs to cms


def format_response(
    df: pd.DataFrame, **kwargs
) -> pd.DataFrame:
    """Setup index for response from query.

    This function formats the response from the NWIS web services, in
    particular it sets the index of the data frame. This function tries to
    convert the NWIS response into pandas datetime values localized to UTC,
    and if possible, uses these timestamps to define the data frame index.

    Parameters
    ----------
    df: ``pandas.DataFrame``
        The data frame to format
    service: string, optional, default is None
        The NWIS service that was queried, important because the 'peaks'
        service returns a different format than the other services.
    **kwargs: optional
        Additional keyword arguments, e.g., 'multi_index'

    Returns
    -------
    df: ``pandas.DataFrame``
        The formatted data frame

    """
    mi = kwargs.pop("multi_index", True)

    # check for multiple sites:
    if "datetime" not in df.columns:
        # XXX: consider making site_no index
        return df

    elif len(df["site_no"].unique()) > 1 and mi:
        # setup multi-index
        df.set_index(["site_no", "datetime"], inplace=True)
        if hasattr(df.index.levels[1], "tzinfo") and df.index.levels[1].tzinfo is None:
            df = df.tz_localize("UTC", level=1)

    else:
        df.set_index(["datetime"], inplace=True)
        if hasattr(df.index, "tzinfo") and df.index.tzinfo is None:
            df = df.tz_localize("UTC")

    return df.sort_index()


def _read_rdb(rdb):
    """
    Convert NWIS rdb table into a ``pandas.dataframe``.

    Parameters
    ----------
    rdb: string
        A string representation of an rdb table

    Returns
    -------
    df: ``pandas.dataframe``
        A formatted pandas data frame

    """
    count = 0

    for line in rdb.splitlines():
        # ignore comment lines
        if line.startswith("#"):
            count = count + 1

        else:
            break

    fields = re.split("[\t]", rdb.splitlines()[count])
    fields = [field.replace(",", "") for field in fields]
    dtypes = {
        "site_no": str,
        "dec_long_va": float,
        "dec_lat_va": float,
        "parm_cd": str,
        "parameter_cd": str,
    }

    df = pd.read_csv(
        StringIO(rdb),
        delimiter="\t",
        skiprows=count + 2,
        names=fields,
        na_values="NaN",
        dtype=dtypes,
    )

    df = format_response(df)
    return df

def _download_metadata(site:str)->pd.DataFrame:
    response = requests.get(
        'https://waterservices.usgs.gov/nwis/site', 
        params={'sites': site, 'parameterCd': '00060', 'siteOutput': 'Expanded', 'format': 'rdb'}, 
        headers={'user-agent': 'python-dataretrieval/1.0.11'}, 
        verify=True
        )

    return _read_rdb(response.text)
