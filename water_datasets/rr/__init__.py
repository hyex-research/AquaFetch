"""
Rainfall Runoff datasets
"""

# ExtendinG SUb-DAily River Discharge data over INdia (GUARDIAN)
# https://springernature.figshare.com/articles/dataset/ExtendinG_SUb-DAily_River_Discharge_data_over_INdia_GUARDIAN_/27004282

import os
from typing import Union, List

import pandas as pd
from .._backend import plt, plt_Axes

from .camels import Camels
from ._camels import CAMELS_AUS
from ._camels import CAMELS_CL
from ._camels import CAMELS_GB
from ._camels import CAMELS_US
from ._lamah import LamaHCE
from ._brazil import CAMELS_BR
from ._brazil import CABra
from ._hysets import HYSETS
from ._hype import HYPE
from ._camels import CAMELS_DK
from ._waterbenchiowa import WaterBenchIowa
from ._gsha import GSHA
from ._ccam import CCAM
from ._rrluleasweden import RRLuleaSweden
from ._camels import CAMELS_CH
from ._lamah import LamaHIce
from ._camels import CAMELS_DE
from ._grdccaravan import GRDCCaravan
from ._camels import CAMELS_SE
from ._simbi import Simbi
from ._denmark import CAMELS_DK as CAMELS_DK0
from ._bull import Bull
from ._camels import CAMELS_IND
from ._gsha import Arcticnet
from ._usgs import USGS
from ._estreams import EStreams
from ._gsha import Japan
from ._gsha import Thailand
from ._gsha import Spain
from ._estreams import Ireland
from ._estreams import Finland
from ._estreams import Finland
from ._estreams import Poland
from ._estreams import Italy
from ._camels import CAMELS_FR
from ._estreams import Portugal
# following are not available with RainfallRunoff class yet
from ._npctr import NPCTRCatchments
from .mtropics import MtropicsLaos
from .mtropics import MtropcsThailand
from .mtropics import MtropicsVietnam


DATASETS = {
    "camels": Camels,
    "CAMELS_AUS": CAMELS_AUS,
    "CAMELS_CL": CAMELS_CL,
    "CAMELS_GB": CAMELS_GB,
    "CAMELS_US": CAMELS_US,
    "LamaHCE": LamaHCE,
    "CAMELS_BR": CAMELS_BR,
    "CABra": CABra,
    "HYSETS": HYSETS,
    "HYPE": HYPE,
    "CAMELS_DK": CAMELS_DK,
    "WaterBenchIowa": WaterBenchIowa,
    "GSHA": GSHA,
    "EStreams": EStreams,
    "CCAM": CCAM,
    "RRLuleaSweden": RRLuleaSweden,
    "CAMELS_CH": CAMELS_CH,
    "LamaHIce": LamaHIce,
    "CAMELS_DE": CAMELS_DE,
    "GRDCCaravan": GRDCCaravan,
    "CAMELS_SE": CAMELS_SE,
    "Simbi": Simbi,
    "CAMELS_DK0": CAMELS_DK0,
    "Bull": Bull,
    "CAMELS_IND": CAMELS_IND,
    "USGS": USGS,
    "Arcticnet": Arcticnet,
    'Japan': Japan,
    'Spain': Spain,
    'Thailand': Thailand,
    'Ireland': Ireland,
    'Finland': Finland,
    'Poland': Poland,
    'Italy': Italy,
    'CAMELS_FR': CAMELS_FR,
    'Portugal': Portugal,
}

"""
    .. list-table:: Naming Convention for dynamic features
       :widths: 20 30
       :header-rows: 1

       * - Feature Name
         - Description
       * - obs_q_cms
         - observed streamflow in cms
       * - obs_q_mmd
         - observed streamflow in mm/day
       * - pcp_mm
         - precipitation in mm
       * - max_temp_C
         - maximum air temperature in degree celcius
       * - min_temp_C
         - minimum air temperature in degree celcius
       * - mean_temp_C
         - mean temperature in degree celcius
       * - method_pet_mm
         - potential evapotranspiration in mm
       * - et_mm
         - evapotranspiration in mm
       * - rh_%
         - relative humidity in percentage
       * - min_rh_%
         - minimum relative humidity
       * - max_rh_%
         - maximum relative humidity
       * - swe_mm
         - snow water equivalent
       * - solrad_wm2
         - solar radiation watt per meter square
       * - windspeed_ms
         - wind speed in meter per second
       * - sim_q_cms
         - simulated streamflow in cms
       * - dwn_lw_rad_wm2
         - downward long wave radiation in watt per meter square
       * - dwn_sw_rad_wm2
         - downward short wave radiation in watt per meter square
       * - airpres_hpa
         - Mean air pressure at sea level in hectopascal


    .. list-table:: Naming Convention for static features
       :widths: 20 30
       :header-rows: 1

       * - Feature Name
         - Description
       * - guage_lat
         - Latitude of the guage station
       * - guage_long
         - Longitude of the guage station
       * - area_km2
         - catchment area in km2
       * - mean_elev
         - mean elevation in meters
"""
class RainfallRunoff(object):
    """
    This  class provides access to all the rainfall-runoff
    datasets. For simiplity and resusability, use this class 
    instead of using the individual dataset classes.

    Examples
    --------
    >>> from water_datasets import RainfallRunoff
    >>> dataset = RainfallRunoff('CAMELS_AUS')  # instead of CAMELS_AUS, you can provide any other dataset name
    >>> df = dataset.fetch(stations=1, as_dataframe=True)
    >>> df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
    >>> df.columns = df.columns.get_level_values('dynamic_features')
    >>> df.shape
       (21184, 26)
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       222
    ... # get data of 10 % of stations as dataframe
    >>> df = dataset.fetch(0.1, as_dataframe=True)
    >>> df.shape
       (550784, 22)
    ... # The returned dataframe is a multi-indexed data
    >>> df.index.names == ['time', 'dynamic_features']
        True
    ... # get data by station id
    >>> df = dataset.fetch(stations='224214A', as_dataframe=True).unstack()
    >>> df.shape
        (21184, 26)
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> data = dataset.fetch(1, as_dataframe=True,
    ...  dynamic_features=['tmax_AWAP', 'precipitation_AWAP', 'et_morton_actual_SILO', 'streamflow_MLd']).unstack()
    >>> data.shape
       (21184, 4)
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> df = dataset.fetch(10, as_dataframe=True)
    >>> df.shape  # remember this is a multiindexed dataframe
       (21184, 260)
    # when we get both static and dynamic data, the returned data is a dictionary
    # with ``static`` and ``dyanic`` keys.
    >>> data = dataset.fetch(stations='224214A', static_features="all", as_dataframe=True)
    >>> data['static'].shape, data['dynamic'].shape
    ((1, 166), (550784, 1))
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (472, 2)
    >>> dataset.stn_coords('3001')  # returns coordinates of station whose id is 3001
        18.3861	80.3917
    >>> dataset.stn_coords(['3001', '17021'])  # returns coordinates of two stations

    See :ref:`sphx_glr_auto_examples_camels_australia.py` for more comprehensive usage example.

    """

    def __init__(
            self,
            dataset: str,
            path: Union[str, os.PathLike] = None,
            overwrite: bool = False,
            to_netcdf: bool = True,
            processes: int = None,
            remove_zip: bool = True,
            verbosity: int = 1,
            **kwargs
    ):
        """
        Rainfall Runoff datasets

        Parameters
        ----------
        dataset: str
            dataset name. This must be one of the following:

            - ``Arcticnet``
            - ``Bull``
            - ``CABra``
            - ``CCAM``
            - ``CAMELS_AUS``
            - ``CAMELS_BR``
            - ``CAMELS_CH``
            - ``CAMELS_CL``
            - ``CAMELS_DE``
            - ``CAMELS_DK0``
            - ``CAMELS_DK``
            - ``CAMELS_FR``
            - ``CAMELS_GB``
            - ``CAMELS_IND``
            - ``CAMELS_SE``
            - ``CAMELS_US``
            - ``EStreams``
            - ``Finland``
            - ``GRDCCaravan``
            - ``GSHA``
            - ``HYSETS``
            - ``HYPE``
            - ``Ireland``
            - ``Italy``
            - ``Japan``
            - ``LamaHCE``
            - ``LamaHIce``
            - ``Poland``
            - ``Portugal``
            - ``RRLuleaSweden``
            - ``Simbi``
            - ``Spain``
            - ``Thailand``
            - ``USGS``
            - ``WaterBenchIowa``

        path : str
            path to directory inside which data is located/downloaded.
            If provided and the path/dataset exists, then the data will be read
            from this path. If provided and the path/dataset does not exist,
            then the data will be downloaded at this path. If not provided,
            then the data will be downloaded in the default path which is
            ``.../water-datasts/data/``.
        overwrite : bool
            If the data is already downloaded then you can set it to True,
            to make a fresh download.
        to_netcdf : bool
            whether to convert all the data into one netcdf file or not.
            This will fasten repeated calls to fetch etc but will
            require netcdf5 package as well as xarray.
        verbosity : int
            0: no message will be printed
        kwargs :
            additional keyword arguments for the underlying dataset class
            For example ``version`` for :py:class:`water_quality.rr.CAMELS_AUS` or ``timestep`` for
            :py:class:`water_quality.rr.LamaHCE` dataset or ``met_src`` for ``CAMELS_BR``
        """

        if dataset not in DATASETS:
            raise ValueError(f"Dataset {dataset} not available")

        self.dataset = DATASETS[dataset](
            path=path,
            overwrite=overwrite,
            to_netcdf=to_netcdf,
            processes=processes,
            remove_zip=remove_zip,
            verbosity=verbosity,
            **kwargs
        )

    def __str__(self):
        return f"{self.name} with {len(self.stations())} stations, {self.num_dynamic()} dynamic and {self.num_static()} static features"

    def __len__(self):
        return len(self.stations())

    def num_dynamic(self) -> int:
        """number of dynamic features associated with the dataset"""
        return len(self.dynamic_features)

    def num_static(self) -> int:
        """number of static features associated with the dataset"""
        return len(self.static_features)

    @property
    def name(self) -> str:
        """
        returns name of dataset
        """
        return self.dataset.name

    @property
    def path(self) -> str:
        """
        returns path where the data is stored. The default path is
        ~../water_quality/data
        """
        return self.dataset.path

    @property
    def static_features(self) -> List[str]:
        """
        returns names of static features as python list of strings

        Examples
        --------
        >>> from water_datasets import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_AUS')
        >>> dataset.static_features
        """
        return self.dataset.static_features

    @property
    def dynamic_features(self) -> List[str]:
        """
        returns names of dynamic features as python list of strings

        Examples
        --------
        >>> from water_datasets import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_AUS')
        >>> dataset.dynamic_features
        """
        return self.dataset.dynamic_features

    def fetch_static_features(
            self,
            stations: Union[str, list] = "all",
            static_features: Union[str, list] = "all"
    ) -> pd.DataFrame:
        """Fetches all or selected static attributes of one or more stations.

        Parameters
        ----------
            stations : str
                name/id of station of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Returns
        -------
        pd.DataFrame
            a pandas dataframe

        Examples
        --------
            >>> from water_datasets import RainfallRunoff
            >>> camels = RainfallRunoff('CAMELS_AUS')
            >>> camels.fetch_static_features('224214A')
            >>> camels.static_features
            >>> camels.fetch_static_features('224214A',
            ... features=['elev_mean', 'relief', 'ksat', 'pop_mean'])
        """

        return self.dataset.fetch_static_features(stations, static_features)

    def area(
            self,
            stations: Union[str, List[str]] = "all"
    ) -> pd.Series:
        """
        Returns area (Km2) of all/selected catchments as pandas series

        parameters
        ----------
        stations : str/list (default=``all``)
            name/names of stations. Default is ``all``, which will return
            area of all stations

        Returns
        --------
        pd.Series
            a pandas series whose indices are catchment ids and values
            are areas of corresponding catchments.

        Examples
        ---------
        >>> from water_datasets import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_CH')
        >>> dataset.area()  # returns area of all stations
        >>> dataset.area('2004')  # returns area of station whose id is 2004
        >>> dataset.area(['2004', '6004'])  # returns area of two stations
        """
        return self.dataset.area(stations)

    def fetch(self,
              stations: Union[str, List[str], int, float] = "all",
              dynamic_features: Union[List[str], str, None] = 'all',
              static_features: Union[str, List[str], None] = None,
              st: Union[None, str] = None,
              en: Union[None, str] = None,
              as_dataframe: bool = False,
              **kwargs  # todo, where do these keyword args go?
              ) -> Union[dict, pd.DataFrame]:
        """
        Fetches the features of one or more stations.

        parameters
        ----------
        stations :
            It can have following values:

                - int : number of (randomly selected) stations to fetch
                - float : fraction of (randomly selected) stations to fetch
                - str : name/id of station to fetch. However, if ``all`` is
                  provided, then all stations will be fetched.
                - list : list of names/ids of stations to fetch
        dynamic_features : (default=``all``)
            It can have following values:

                - str : name of dynamic feature to fetch. If ``all`` is
                  provided, then all dynamic features will be fetched.
                - list : list of dynamic features to fetch.
                - None : No dynamic feature will be fetched.
        static_features : (default=None)
            It can have following values:

                - str : name of static feature to fetch. If ``all`` is
                  provided, then all static features will be fetched.
                - list : list of static features to fetch.
                - None : No static feature will be fetched.
        st :
            starting date of data to be returned. If None, the data will be
            returned from where it is available.
        en :
            end date of data to be returned. If None, then the data will be
            returned till the date data is available.
        as_dataframe :
            whether to return dynamic attributes as pandas
            dataframe or as xarray dataset.
        kwargs :
            keyword arguments

        returns
        -------
            If both static  and dynamic features are obtained then it returns a
            dictionary whose keys are station/gauge_ids and values are the
            attributes and dataframes.
            Otherwise either dynamic or static features are returned.

        Examples
        --------
        >>> from water_datasets import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_AUS')
        >>> # get data of 10% of stations
        >>> df = dataset.fetch(stations=0.1, as_dataframe=True)  # returns a multiindex dataframe
        ...  # fetch data of 5 (randomly selected) stations
        >>> five_random_stn_data = dataset.fetch(stations=5, as_dataframe=True)
        ... # fetch data of 3 selected stations
        >>> three_selec_stn_data = dataset.fetch(stations=['912101A','912105A','915011A'], as_dataframe=True)
        ... # fetch data of a single stations
        >>> single_stn_data = dataset.fetch(stations='318076', as_dataframe=True)
        ... # get both static and dynamic features as dictionary
        >>> data = dataset.fetch(1, static_features="all", as_dataframe=True)  # -> dict
        >>> data['dynamic']
        ... # get only selected dynamic features
        >>> sel_dyn_features = dataset.fetch(stations='318076',
        ...     dynamic_features=['streamflow_MLd', 'solarrad_AWAP'], as_dataframe=True)
        ... # fetch data between selected periods
        >>> data = dataset.fetch(stations='318076', st="20010101", en="20101231", as_dataframe=True)

        """
        return self.dataset.fetch(stations, dynamic_features, static_features, st, en, as_dataframe, **kwargs)

    def fetch_stations_features(
            self,
            stations: Union[str, List[str]],
            dynamic_features: Union[str, List[str], None] = 'all',
            static_features: Union[str, List[str], None] = None,
            st=None,
            en=None,
            as_dataframe: bool = False,
            **kwargs
    ):
        """
        Reads attributes of more than one stations.

        parameters
        ----------
        stations :
            list of stations for which data is to be fetched.
        dynamic_features :
            list of dynamic features to be fetched.
                if 'all', then all dynamic features will be fetched.
        static_features :
            list of static features to be fetched.
            If `all`, then all static features will be fetched. If None,
            then no static attribute will be fetched.
        st :
            start of data to be fetched.
        en :
            end of data to be fetched.
        as_dataframe : whether to return the data as pandas dataframe. default
                is xr.Dataset object
        kwargs dict:
            additional keyword arguments

        Returns
        -------
        pd.DataFrame or xr.Dataset or dict
            Dynamic and static features of one or multiple stations. Dynamic features
            are by default returned as xr.Dataset unless `as_dataframe` is True or xarray is not installed, in
            such a case, it is a pandas dataframe with multiindex. If xr.Dataset,
            it consists of `data_vars` equal to number of stations and for each
            station, the `DataArray` is of dimensions (time, dynamic_features).
            where `time` is defined by `st` and `en` i.e. length of `DataArray`.
            In case, when the returned object is pandas DataFrame, the first index
            is `time` and second index is `dyanamic_features`. Static attributes
            are always returned as pandas DataFrame and have following shape
            `(stations, static_features). If `dynamic_features` is None,
            then they are not returned and the returned value only consists of
            static features. Same holds true for `static_features`.
            If both are not None, then the returned type is a dictionary with
            `static` and `dynamic` keys.

        Raises
        ------
        ValueError, if both ``dynamic_features`` and ``static_features`` are None

        Examples
        --------
            >>> from water_datasets import RainfallRunoff
            >>> dataset = RainfallRunoff('CAMELS_AUS')
            ... # find out station ids
            >>> dataset.stations()
            ... # get data of selected stations
            >>> dataset.fetch_stations_features(['912101A', '912105A', '915011A'],
            ...  as_dataframe=True)
        """
        return self.dataset.fetch_stations_features(stations, dynamic_features, static_features, st, en, as_dataframe,
                                                    **kwargs)

    def fetch_dynamic_features(
            self,
            stn_id: str,
            dynamic_features='all',
            st=None,
            en=None,
            as_dataframe=False
    ):
        """Fetches all or selected dynamic attributes of one station.

        Parameters
        ----------
            stn_id : str
                name/id of station of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                dynamic features are returned.
            st : Optional (default=None)
                start time from where to fetch the data.
            en : Optional (default=None)
                end time untill where to fetch the data
            as_dataframe : bool, optional (default=False)
                if true, the returned data is pandas DataFrame otherwise it
                is xarray dataset

        Examples
        --------
            >>> from water_datasets import RainfallRunoff
            >>> camels = RainfallRunoff('CAMELS_AUS')
            >>> camels.fetch_dynamic_features('224214A', as_dataframe=True).unstack()
            >>> camels.dynamic_features
            >>> camels.fetch_dynamic_features('224214A',
            ... features=['tmax_AWAP', 'vprp_AWAP', 'streamflow_mmd'],
            ... as_dataframe=True).unstack()
        """
        return self.dataset.fetch_dynamic_features(
            stn_id, dynamic_features, st, en, as_dataframe)

    def fetch_station_features(
            self,
            stn_id: str,
            dynamic_features: Union[str, list, None] = 'all',
            static_features: Union[str, list, None] = None,
            as_ts: bool = False,
            st: Union[str, None] = None,
            en: Union[str, None] = None,
            **kwargs
    ) -> pd.DataFrame:
        """
        Fetches features for one station.

        Parameters
        -----------
            station :
                station id/gauge id for which the data is to be fetched.
            dynamic_features : str/list, optional
                names of dynamic features/attributes to fetch
            static_features :
                names of static features/attributes to be fetches
            as_ts : bool
                whether static features are to be converted into a time
                series or not. If yes then the returned time series will be of
                same length as that of dynamic attribtues.
            st : str,optional
                starting point from which the data to be fetched. By default,
                the data will be fetched from where it is available.
            en : str, optional
                end point of data to be fetched. By default the dat will be fetched

        Returns
        -------
        pd.DataFrame
            dataframe if as_ts is True else it returns a dictionary of static and
            dynamic features for a station/gauge_id

        Examples
        --------
            >>> from water_datasets import RainfallRunoff
            >>> dataset = RainfallRunoff('CAMELS_AUS')
            >>> dataset.fetch_station_features('912101A')

        """
        return self.dataset.fetch_station_features(stn_id, dynamic_features, static_features, as_ts, st, en, **kwargs)

    def plot_stations(
            self,
            stations: List[str] = 'all',
            marker='.',
            ax: plt_Axes = None,
            show: bool = True,
            **kwargs
    ) -> plt_Axes:
        """
        plots coordinates of stations

        Parameters
        ----------
        stations :
            name/names of stations. If not given, all stations will be plotted
        marker :
            marker to use.
        ax : plt.Axes
            matplotlib axes to draw the plot. If not given, then
            new axes will be created.
        show : bool
        **kwargs

        Returns
        -------
        plt.Axes

        Examples
        --------
        >>> from water_datasets import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_AUS')
        >>> dataset.plot_stations()
        >>> dataset.plot_stations(['1', '2', '3'])
        >>> dataset.plot_stations(marker='o', ms=0.3)
        >>> ax = dataset.plot_stations(marker='o', ms=0.3, show=False)
        >>> ax.set_title("Stations")
        >>> plt.show()

        """
        return self.dataset.plot_stations(stations, marker, ax, show, **kwargs)

    def q_mmd(
            self,
            stations: Union[str, List[str]] = 'all'
    ) -> pd.DataFrame:
        """
        returns streamflow in the units of milimeter per day. This is obtained
        by diving ``q``/area

        parameters
        ----------
        stations : str/list
            name/names of stations. Default is ``all``, which will return
            area of all stations

        Returns
        --------
        pd.DataFrame
            a pandas DataFrame whose indices are time-steps and columns
            are catchment/station ids.

        """
        return self.dataset.q_mmd(stations)

    def stn_coords(
            self,
            stations: Union[str, List[str]] = "all"
    ) -> pd.DataFrame:
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
        >>> from water_datasets import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_CH')
        >>> dataset.stn_coords() # returns coordinates of all stations
        >>> dataset.stn_coords('2004')  # returns coordinates of station whose id is 2004
        >>> dataset.stn_coords(['2004', '6004'])  # returns coordinates of two stations

        >>> from water_datasets import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_AUS')
        >>> dataset.stn_coords() # returns coordinates of all stations
        >>> dataset.stn_coords('912101A')  # returns coordinates of station whose id is 912101A
        >>> dataset.stn_coords(['G0050115', '912101A'])  # returns coordinates of two stations

        """
        return self.dataset.stn_coords(stations)

    def get_boundary(
            self,
            stn_id: str,
            as_type: str = 'numpy'
    ):
        """
        returns boundary of a catchment in a required format

        Parameters
        ----------
        stn_id : str
            name/id of catchment
        as_type : str
            'numpy' or 'geopandas'

        Examples
        --------
        >>> from water_datasets import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_SE')
        >>> dataset.get_boundary(dataset.stations()[0])
        """
        return self.dataset.get_boundary(stn_id, as_type)

    def plot_catchment(
            self,
            stn_id: str,
            ax: plt_Axes = None,
            show: bool = True,
            **kwargs
    ) -> plt.Axes:
        """
        plots catchment boundaries

        Parameters
        ----------
        ax : plt.Axes
            matplotlib axes to draw the plot. If not given, then
            new axes will be created.
        show : bool
        **kwargs

        Returns
        -------
        plt.Axes

        Examples
        --------
        >>> from water_datasets import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_AUS')
        >>> dataset.plot_catchment()
        >>> dataset.plot_catchment(marker='o', ms=0.3)
        >>> ax = dataset.plot_catchment(marker='o', ms=0.3, show=False)
        >>> ax.set_title("Catchment Boundaries")
        >>> plt.show()

        """
        return self.dataset.plot_catchment(stn_id, ax, show, **kwargs)

    def stations(self) -> List[str]:
        """
        returns names of all stations

        Examples
        --------
        >>> from water_datasets import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_AUS')
        >>> dataset.stations()
        """
        return self.dataset.stations()

    @property
    def start(self) -> str:
        """
        returns starting date of data

        Examples
        --------
        >>> from water_datasets import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_AUS')
        >>> dataset.start()
        """
        return self.dataset.start

    @property
    def end(self) -> str:
        """
        returns end date of data

        Examples
        --------
        >>> from water_datasets import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_AUS')
        >>> dataset.end()
        """
        return self.dataset.end
