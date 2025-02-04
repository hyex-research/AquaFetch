
import os
from typing import Union, List, Dict, Tuple

import numpy as np
import pandas as pd

from .utils import _RainfallRunoff
from .._backend import shapefile, xarray as xr
from ..utils import check_attributes, sanity_check

from ._map import (
    observed_streamflow_cms,
    min_air_temp,
    max_air_temp,
    total_precipitation,
    snow_water_equivalent,
    )

from ._map import (
    catchment_area,
    gauge_latitude,
    gauge_longitude,
    slope
    )


class HYSETS(_RainfallRunoff):
    """
    database for hydrometeorological modeling of 14,425 North American watersheds
    from 1950-2018 following the work of `Arsenault et al., 2020 <https://doi.org/10.1038/s41597-020-00583-2>`_
    The user must manually download the files, unpack them and provide
    the `path` where these files are saved.

    This data comes with multiple sources. Each source having one or more dynamic_features
    Following data_source are available.

    +---------------+------------------------------+
    |sources        | dynamic_features             |
    +===============+==============================+
    |SNODAS_SWE     | dscharge, swe                |
    +---------------+------------------------------+
    |SCDNA          | discharge, pr, tasmin, tasmax|
    +---------------+------------------------------+
    |nonQC_stations | discharge, pr, tasmin, tasmax|
    +---------------+------------------------------+
    |Livneh         | discharge, pr, tasmin, tasmax|
    +---------------+------------------------------+
    |ERA5           | discharge, pr, tasmax, tasmin|
    +---------------+------------------------------+
    |ERAS5Land_SWE  | discharge, swe               |
    +---------------+------------------------------+
    |ERA5Land       | discharge, pr, tasmax, tasmin|
    +---------------+------------------------------+

    all sources contain one or more following dynamic_features
    with following shapes

    +----------------------------+------------------+
    |dynamic_features            |      shape       |
    +============================+==================+
    |time                        |   (25202,)       |
    +----------------------------+------------------+
    |watershedID                 |   (14425,)       |
    +----------------------------+------------------+
    |drainage_area               |   (14425,)       |
    +----------------------------+------------------+
    |drainage_area_GSIM          |   (14425,)       |
    +----------------------------+------------------+
    |flag_GSIM_boundaries        |   (14425,)       |
    +----------------------------+------------------+
    |flag_artificial_boundaries  |   (14425,)       |
    +----------------------------+------------------+
    |centroid_lat                |   (14425,)       |
    +----------------------------+------------------+
    |centroid_lon                |   (14425,)       |
    +----------------------------+------------------+
    |elevation                   |   (14425,)       |
    +----------------------------+------------------+
    |slope                       |   (14425,)       |
    +----------------------------+------------------+
    |discharge                   |   (14425, 25202) |
    +----------------------------+------------------+
    |pr                          |   (14425, 25202) |
    +----------------------------+------------------+
    |tasmax                      |   (14425, 25202) |
    +----------------------------+------------------+
    |tasmin                      |   (14425, 25202) |
    +----------------------------+------------------+

    Examples
    --------
    >>> from aqua_fetch import HYSETS
    >>> dataset = HYSETS(path="path/to/HYSETS")
    ... # fetch data of a random station
    >>> _, df = dataset.fetch(1, as_dataframe=True)
    >>> df.shape
    (25202, 5)
    >>> stations = dataset.stations()
    >>> len(stations)
    14425
    >>> _, df = dataset.fetch('999', as_dataframe=True)
    >>> df.unstack().shape
    (25202, 5)

    """
    doi = "https://doi.org/10.1038/s41597-020-00583-2"
    url = "https://osf.io/rpc3w/"
    Q_SRC = ['ERA5', 'ERA5Land', 'ERA5Land_SWE', 'Livneh', 'nonQC_stations', 'SCDNA', 'SNODAS_SWE']
    SWE_SRC = ['ERA5Land_SWE', 'SNODAS_SWE']
    OTHER_SRC = [src for src in Q_SRC if src not in ['ERA5Land_SWE', 'SNODAS_SWE']]
    dynamic_features = ['discharge', 'swe', 'tasmin', 'tasmax', 'pr']

    def __init__(self,
                 path: str,
                 swe_source: str = "SNODAS_SWE",
                 discharge_source: str = "ERA5",
                 tasmin_source: str = "ERA5",
                 tasmax_source: str = "ERA5",
                 pr_source: str = "ERA5",
                 **kwargs
                 ):
        """
        parameters
        --------------
            path : str
                If the data is alredy downloaded then provide the complete
                path to it. If None, then the data will be downloaded.
                The data is downloaded once and therefore susbsequent
                calls to this class will not download the data unless
                ``overwrite`` is set to True.
            swe_source : str
                source of swe data.
            discharge_source :
                source of discharge data
            tasmin_source :
                source of tasmin data
            tasmax_source :
                source of tasmax data
            pr_source :
                source of pr data
            kwargs :
                arguments for ``Camels`` base class

        """

        assert swe_source in self.SWE_SRC, f'swe source must be one of {self.SWE_SRC}'
        assert discharge_source in self.Q_SRC, f'discharge source must be one of {self.Q_SRC}'
        assert tasmin_source in self.OTHER_SRC, f'tsmin source must be one of {self.OTHER_SRC}'
        assert tasmax_source in self.OTHER_SRC, f'tsmax source must be one of {self.OTHER_SRC}'
        assert pr_source in self.OTHER_SRC, f'pr source must be one of {self.OTHER_SRC}'

        self.sources = {
            'swe': swe_source,
            'discharge': discharge_source,
            'tasmin': tasmin_source,
            'tasmax': tasmax_source,
            'pr': pr_source
        }

        super().__init__(**kwargs)

        self.path = path

        self._stations = self.__stations()

        fpath = os.path.join(self.path, 'hysets_dyn.nc')
        if not os.path.exists(fpath):
            self._maybe_to_netcdf('hysets_dyn')
        
        self.boundary_file = os.path.join(path,  
                                          "HYSETS_watershed_boundaries", 
                                          "HYSETS_watershed_boundaries_20200730.shp")
        self._create_boundary_id_map(self.boundary_file, 2)

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'Drainage_Area_km2': catchment_area(), # todo: why give preference to, Drainage_Area_GSIM_km2
                'Centroid_Lat_deg_N': gauge_latitude(),
                'Slope_deg': slope('degrees'),
                'Centroid_Lon_deg_E': gauge_longitude(),
        }

    @property
    def dyn_map(self):
        return {
        'discharge': observed_streamflow_cms(), 
        'tasmin': min_air_temp(),
        'tasmax': max_air_temp(),
        'pr': total_precipitation(),
        'swe': snow_water_equivalent()
        }

    def _maybe_to_netcdf(self, fname: str):
        # todo saving as one file takes very long time
        oneD_vars = []
        twoD_vars = []

        for src in self.Q_SRC:
            fpath = os.path.join(self.path, f'HYSETS_2020_{src}.nc')
            assert os.path.exists(fpath), f'{fpath} does not exist'
            xds = xr.open_dataset(fpath)

            if self.verbosity>1:
                print(f'processing {src}')

            for idx, var in enumerate(xds.variables):
                if self.verbosity>2: print(f'{idx}: getting {var} from source {src} ')

                if len(xds[var].data.shape) > 1:
                    xar = xds[var]
                    xar.name = f"{xar.name}_{src}"
                    twoD_vars.append(xar)
                else:
                    xar = xds[var]
                    xar.name = f"{xar.name}_{src}"
                    oneD_vars.append(xar)

        if self.verbosity>1:
            print('merging 1D vars')
        oneD_xds = xr.merge(oneD_vars)
        twoD_xds = xr.merge(twoD_vars)

        if self.verbosity>1:
            print('saving to netcdf')
        oneD_xds.to_netcdf(os.path.join(self.path, "hysets_static.nc"))
        twoD_xds.to_netcdf(os.path.join(self.path, "hysets_dyn.nc"))

        return

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, x):
        sanity_check('HYSETS', x)
        self._path = x

    @property
    def static_features(self)->List[str]:
        df = self.read_static_data(nrows=2)
        return df.columns.to_list()

    def stations(self) -> List[str]:
        """
        retuns a list of station names. The ``Watershed_ID`` of the station is used
        as station name instead of ``Official_ID``. This is because in .nc files
        watershed_ID is used for stations instead of Official_ID. ``Official_ID``
        starts with 1, 2, 3 and so on while ``Watershed_ID`` is a code from
        meteo agency such as ``01AD002`` for station 1.

        Returns
        -------
        list
            a list of ids of stations

        Examples
        --------
        >>> from aqua_fetch import HYSETS
        >>> dataset = HYSETS()
        ... # get name of all stations as list
        >>> dataset.stations()

        """
        return self._stations

    def __stations(self)->List[str]:
        return self.read_static_data().index.to_list()

    @property
    def WatershedID_OfficialID_map(self):
        """A dictionary mapping Watershed_ID to Official_ID.
        For example '01AD002': '1'
        """
        return self.read_static_data(
            usecols=['Watershed_ID', 'Official_ID']
            ).loc[:, 'Official_ID'].to_dict()

    @property
    def OfficialID_WatershedID_map(self):
        """A dictionary mapping Official_ID to Watershed_ID.
        For example '1': '01AD002'
        """
        s = self.read_static_data(usecols=['Watershed_ID', 'Official_ID'])
        return {v:k for k,v in s.loc[:, 'Official_ID'].to_dict().items()}
        

    @property
    def start(self)->str:
        return "19500101"

    @property
    def end(self)->str:
        return "20181231"

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
        >>> from aqua_fetch import HYSETS
        >>> dataset = HYSETS()
        >>> dataset.get_boundary(dataset.stations()[0])
        """

        if shapefile is None:
            raise ModuleNotFoundError("shapefile module is not installed. Please install it to use boundary file")

        from shapefile import Reader

        bndry_sf = Reader(self.boundary_file)
        bndry_shp = bndry_sf.shape(self.bndry_id_map[self.WatershedID_OfficialID_map[catchment_id]])

        bndry_sf.close()

        xyz = np.array(bndry_shp.points)

        xyz = self.transform_coords(xyz)

        return xyz

    def q_mmd(
            self,
            stations: Union[str, List[str]] = 'all'
    )->pd.DataFrame:
        """
        returns streamflow in the units of milimeter per day. This is obtained
        by diving q_cms/area

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
        stations = check_attributes(stations, self.stations())
        _, q = self.fetch_stations_features(stations,
                                           dynamic_features='discharge',
                                           as_dataframe=True)
        # todo: this is not good practice. fetch_stations_features should requrn q with correct
        # column names and after correcting it correct it in USGS as well
        q.columns = stations
        
        q.index = q.index.get_level_values(0)
        area_m2 = self.area(stations) * 1e6  # area in m2
        q = (q / area_m2) * 86400  # cms to m/day
        return q * 1e3  # to mm/day

    def area(
            self,
            stations: Union[str, List[str]] = 'all',
            source:str = 'other'
    ) ->pd.Series:
        """
        Returns area_gov (Km2) of all catchments as :obj:`pandas.Series`

        parameters
        ----------
        stations : str/list
            name/names of stations. Default is None, which will return
            area of all stations
        source : str
            source of area calculation. It should be either ``gsim`` or ``other``

        Returns
        --------
        pd.Series
            a :obj:`pandas.Series` whose indices are catchment ids and values
            are areas of corresponding catchments.

        Examples
        ---------
        >>> from aqua_fetch import HYSETS
        >>> dataset = HYSETS()
        >>> dataset.area()  # returns area of all stations
        >>> dataset.area('92')  # returns area of station whose id is 912101A
        >>> dataset.area(['92', '142'])  # returns area of two stations
        """
        stations = check_attributes(stations, self.stations())

        SRC_MAP = {
            'gsim': 'Drainage_Area_GSIM_km2',
            'other': 'area_km2'
        }

        s = self.fetch_static_features(
            static_features=[SRC_MAP[source]],
        )

        s.columns = ['area_km2']
        return s.loc[stations, 'area_km2']

    # def stn_coords(
    #         self,
    #         stations:Union[str, List[str]] = 'all'
    # ) ->pd.DataFrame:
    #     """
    #     returns coordinates of stations as DataFrame
    #     with ``long`` and ``lat`` as columns.

    #     Parameters
    #     ----------
    #     stations :
    #         name/names of stations. If not given, coordinates
    #         of all stations will be returned.

    #     Returns
    #     -------
    #     coords :
    #         pandas DataFrame with ``long`` and ``lat`` columns.
    #         The length of dataframe will be equal to number of stations
    #         wholse coordinates are to be fetched.

    #     Examples
    #     --------
    #     >>> dataset = HYSETS()
    #     >>> dataset.stn_coords() # returns coordinates of all stations
    #     >>> dataset.stn_coords('92')  # returns coordinates of station whose id is 912101A
    #     >>> dataset.stn_coords(['92', '142'])  # returns coordinates of two stations

    #     """
    #     df = self.fetch_static_features(
    #         static_features=['Centroid_Lat_deg_N', 'Centroid_Lon_deg_E'])
    #     df.columns = ['lat', 'long']
    #     stations = check_attributes(stations, self.stations())

    #     return df.loc[stations, :]

    def fetch_stations_features(
            self,
            stations: list,
            dynamic_features: Union[str, list, None] = 'all',
            static_features: Union[str, list, None] = None,
            st=None,
            en=None,
            as_dataframe: bool = False,
            **kwargs
              ) -> Tuple[pd.DataFrame, Union[pd.DataFrame, "Dataset"]]:
        """returns features of multiple stations
        Examples
        --------
        >>> from aqua_fetch import HYSETS
        >>> dataset = HYSETS()
        >>> stations = dataset.stations()[0:3]
        >>> features = dataset.fetch_stations_features(stations)
        """
        stations = check_attributes(stations, self.stations())
        stations = [int(stn) for stn in stations]

        static, dynamic = None, None

        if dynamic_features is not None:

            dynamic = self._fetch_dynamic_features(stations=stations,
                                               dynamic_features=dynamic_features,
                                               as_dataframe=as_dataframe,
                                               st=st,
                                               en=en,
                                               **kwargs
                                               )

            if static_features is not None:  # we want both static and dynamic
                static = self._fetch_static_features(station=stations,
                                                     static_features=static_features,
                                                     st=st,
                                                     en=en,
                                                     **kwargs
                                                     )

        elif static_features is not None:
            # we want only static
            static = self._fetch_static_features(
                station=stations,
                static_features=static_features,
                **kwargs
            )
        else:
            raise ValueError

        return static, dynamic

    def fetch_dynamic_features(
            self,
            station,
            dynamic_features = 'all',
            st=None,
            en=None,
            as_dataframe=False
    ):
        """Fetches dynamic features of one station.

        Examples
        --------
        >>> from aqua_fetch import HYSETS
        >>> dataset = HYSETS()
        >>> dyn_features = dataset.fetch_dynamic_features('station_name')
        """
        station = [int(station)]
        return self._fetch_dynamic_features(
            stations=station,
            dynamic_features=dynamic_features,
            st=st,
            en=en,
            as_dataframe=as_dataframe
        )

    def _fetch_dynamic_features(
            self,
            stations: List[int],
            dynamic_features = 'all',
            st=None,
            en=None,
            as_dataframe=False,
            as_ts=False
    ):
        """Fetches dynamic features of station."""
        st, en = self._check_length(st, en)
        attrs = check_attributes(dynamic_features, self.dynamic_features)

        stations = np.subtract(stations, 1).tolist()
        # maybe we don't need to read all variables
        sources = {k: v for k, v in self.sources.items() if k in attrs}

        # original .nc file contains datasets with dynamic and static features as data_vars
        # however, for uniformity of this API and easy usage, we want a Dataset to have
        # station names/gauge_ids as data_vars and each data_var has
        # dimension (time, dynamic_variables)
        # Therefore, first read all data for each station from .nc file
        # then rearrange it.
        # todo, this operation is slower because of `to_dataframe`
        # also doing this removes all the metadata
        x = {}
        f = os.path.join(self.path, "hysets_dyn.nc")
        xds = xr.open_dataset(f)
        for stn in stations:
            xds1 = xds[[f'{k}_{v}' for k, v in sources.items()]].sel(watershed=stn, time=slice(st, en))
            xds1 = xds1.rename_vars({f'{k}_{v}': k for k, v in sources.items()})
            x[stn] = xds1.to_dataframe(['time'])  # todo, this fails in older xr versions
        xds = xr.Dataset(x)
        xds = xds.rename_dims({'dim_1': 'dynamic_features'})
        xds = xds.rename_vars({'dim_1': 'dynamic_features'})

        if as_dataframe:
            return xds.to_dataframe(['time', 'dynamic_features'])

        return xds

    def _fetch_static_features(
            self,
            station="all",
            static_features: Union[str, list] = 'all',
            st=None,
            en=None,
            as_ts=False
    ):

        df = self.read_static_data()

        static_features = check_attributes(static_features, self.static_features, 'static_features')

        if isinstance(station, list):
            station = [str(stn) for stn in station]

        stations = check_attributes(station, self.stations(), 'stations')

        return df.loc[stations, static_features]

    def fetch_static_features(
            self,
            stations: Union[str, List[str]]="all",
            static_features:Union[str, List[str]]="all",
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
            static_features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.
            st :
            en :
            as_ts :

        Examples
        ---------
        >>> from aqua_fetch import HYSETS
        >>> dataset = HYSETS()
        get the names of stations
        >>> stns = dataset.stations()
        >>> len(stns)
            14425
        get all static data of all stations
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (14425, 28)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('991')
        >>> static_data.shape
           (1, 28)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['area_km2', 'Elevation_m'])
        >>> static_data.shape
           (14425, 2)
        """
        return self._fetch_static_features(stations, static_features, st, en, as_ts)

    def read_static_data(self, usecols=None, nrows=None):
        """
        reads the HYSETS_watershed_properties.txt file while using `Watershed_ID`
        as index instead of ``Official_ID``. Watershed_ID starts with 1,2,3 and so on
        while ``Official_ID`` is code from meteo agency such as ``01AD002`` for station 1.
        """
        fname = os.path.join(self.path, 'HYSETS_watershed_properties.txt')
        static_df = pd.read_csv(fname, index_col='Watershed_ID', sep=';', usecols=usecols, nrows=nrows)
        static_df.index = static_df.index.astype(str)

        static_df.rename(columns=self.static_map, inplace=True)
        return static_df
