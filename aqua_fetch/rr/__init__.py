"""
Rainfall Runoff datasets
"""

# ExtendinG SUb-DAily River Discharge data over INdia (GUARDIAN)
# https://springernature.figshare.com/articles/dataset/ExtendinG_SUb-DAily_River_Discharge_data_over_INdia_GUARDIAN_/27004282

import os
from typing import Union, List

import pandas as pd
from .._backend import plt, plt_Axes

from .utils import _RainfallRunoff
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
from ._denmark import Caravan_DK
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
from ._camels import CAMELS_NZ
# following are not available with RainfallRunoff class yet
from ._npctr import NPCTRCatchments
from .mtropics import MtropicsLaos
from .mtropics import MtropcsThailand
from .mtropics import MtropicsVietnam
from ._misc import DraixBleone


DATASETS = {
    "camels": _RainfallRunoff,
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
    "Caravan_DK": Caravan_DK,
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
    'CAMELS_NZ': CAMELS_NZ,
}


class RainfallRunoff(object):
    """
    This  class provides access to all the rainfall-runoff
    datasets. For simiplity and resusability, use this class 
    instead of using the individual dataset classes.

    Examples
    --------
    >>> from aqua_fetch import RainfallRunoff
    >>> dataset = RainfallRunoff('CAMELS_AUS')  # instead of CAMELS_AUS, you can provide any other dataset name
    >>> _, df = dataset.fetch(stations=1, as_dataframe=True)
    >>> df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
    >>> df.columns = df.columns.get_level_values('dynamic_features')
    >>> df.shape
       (26388, 28)
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       561
    ... # get data of 10 % of stations as dataframe
    >>> _, df = dataset.fetch(0.1, as_dataframe=True)
    >>> df.shape
       (738864, 56)
    ... # The returned dataframe is a multi-indexed data
    >>> df.index.names == ['time', 'dynamic_features']
        True
    ... # get data by station id
    >>> _, df = dataset.fetch(stations='912101A', as_dataframe=True)
    >>> df.unstack().shape
        (26388, 28)
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, data = dataset.fetch(1, as_dataframe=True,
    ...  dynamic_features=['airtemp_C_silo_max', 'pcp_mm_silo', 'aet_mm_silo_morton', 'q_cms_obs'])
    >>> data.unstack().shape
       (26388, 4)
    ... # get names of available static features
    >>> dataset.static_features
    ... # get all static features of all stations
    >>> df = dataset.fetch_static_features()
    >>> df.shape
         (561, 187)
    ... # get area of a single station
    >>> area = dataset.area('912101A')
    >>> type(area), area.shape
        (pandas.core.series.Series, (1,))
    ... # get data of 10 random stations
    >>> _, df = dataset.fetch(10, as_dataframe=True)
    >>> df.shape  # remember this is a multiindexed dataframe
       (26388, 280)
    # If we want to get both static and dynamic data 
    >>> static, dynamic = dataset.fetch(stations='912101A', static_features="all", as_dataframe=True)
    >>> static.shape, dynamic.unstack().shape
    ((1, 187), (26388, 28))
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (561, 2)
    >>> dataset.stn_coords('912101A')  # returns coordinates of station whose id is 912101A
       -18.643612	139.253052
    >>> dataset.stn_coords(['912101A', '912105A'])  # returns coordinates of two stations
    ... # get boundary of the catchment
    >>> boundary = dataset.get_boundary('912101A')
    >>> b.shape
    >>> (20086, 2)

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
            - ``CAMELS_NZ``
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
            ``.../aqua_fetch/data/``.
        overwrite : bool
            If the data is already downloaded then you can set it to True,
            to make a fresh download.
        to_netcdf : bool
            whether to convert all the data into one netcdf file or not.
            This will fasten repeated calls to fetch etc but will
            require netcdf5 package as well as :obj:`xarray`.
        verbosity : int
            0: no message will be printed
        kwargs :
            additional keyword arguments for the underlying dataset class
            For example ``version`` for :py:class:`aqua_fetch.rr.CAMELS_AUS` or ``timestep`` for
            :py:class:`aqua_fetch.rr.LamaHCE` dataset or ``met_src`` for :py:class:`aqua_fetch.rr.CAMELS_BR`
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

    def __getattr__(self, item):
        """
        Although we are using most attributes of the underlying dataset class,
        by directly accessing them, but there still can be some dataset specific
        attributes that are not directly accessed. In that case, we can use this
        method to access those attributes.
        """
        if hasattr(self.dataset, item):
            return getattr(self.dataset, item)
        raise AttributeError(f"{item} not found in {self.name} dataset")

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
        ~../aqua_fetch/data
        """
        return self.dataset.path

    @property
    def static_features(self) -> List[str]:
        """
        returns names of static features as python list of strings

        Examples
        --------
        >>> from aqua_fetch import RainfallRunoff
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
        >>> from aqua_fetch import RainfallRunoff
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
                name/id of station of which to extract the data . For names of stations
                see :meth:`stations` .
            static_features : list/str, optional (default="all")
                The name/names of static features to fetch. By default, all available
                static features are returned. For names of static features, see
                :meth:`static_features` .

        Returns
        -------
        pd.DataFrame
            a pandas :obj:`pandas.DataFrame`

        Examples
        --------
        >>> from aqua_fetch import RainfallRunoff
        >>> camels = RainfallRunoff('CAMELS_AUS')
        >>> camels.fetch_static_features('912101A')
        >>> camels.static_features
        >>> camels.fetch_static_features('912101A',
        ... features=['elev_mean', 'relief', 'ksat', 'pop_mean'])
        """

        return self.dataset.fetch_static_features(stations, static_features)

    def area(
            self,
            stations: Union[str, List[str]] = "all"
    ) -> pd.Series:
        """
        Returns area (Km2) of all/selected catchments as :obj:`pandas.Series`

        parameters
        ----------
        stations : str/list (default=``all``)
            name/names of stations. Default is ``all``, which will return
            area of all stations. For names of stations, see :meth:`stations`.

        Returns
        --------
        pd.Series
            a :obj:`pandas.Series` whose indices are catchment ids and values
            are areas of corresponding catchments.

        Examples
        ---------
        >>> from aqua_fetch import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_CH')
        >>> dataset.area()  # returns area of all stations
        >>> dataset.area('2004')  # returns area of station whose id is 2004
        >>> dataset.area(['2004', '6004'])  # returns area of two stations
        """
        return self.dataset.area(stations)

    def fetch(
            self,
            stations: Union[str, List[str], int, float] = "all",
            dynamic_features: Union[List[str], str, None] = 'all',
            static_features: Union[str, List[str], None] = None,
            st: Union[None, str] = None,
            en: Union[None, str] = None,
            as_dataframe: bool = False,
            **kwargs  # todo, where do these keyword args go?
            ) -> tuple[pd.DataFrame, Union[pd.DataFrame, "Dataset"]]:
        """
        Fetches the features of one or more stations.

        parameters
        ----------
        stations :
            It can have following values:

                - :obj:`int` : number of (randomly selected) stations to fetch
                - :obj:`float` : fraction of (randomly selected) stations to fetch
                - :obj:`str` : name/id of station to fetch. However, if ``all`` is
                  provided, then all stations will be fetched. For names of stations,
                  see :meth:`stations`.
                - :obj:`list` : list of names/ids of stations to fetch
        dynamic_features : (default=``all``)
            It can have following values:

                - :obj:`str` : name of dynamic feature to fetch. If ``all`` is
                  provided, then all dynamic features will be fetched. For names
                  of dynamic features, see :meth:`dynamic_features`.
                - :obj:`list` : list of dynamic features to fetch.
                - None : No dynamic feature will be fetched. The second returned value will be None.
        static_features : (default=None)
            It can have following values:

                - :obj:`str` : name of static feature to fetch. If ``all`` is
                  provided, then all static features will be fetched. For names
                  of static features, see :meth:`static_features`.
                - :obj:`list` : list of static features to fetch.
                - None : No static feature will be fetched. The first returned value will be None.
        st :
            starting date of data to be returned. If None, the data will be
            returned from where it is available.
        en :
            end date of data to be returned. If None, then the data will be
            returned till the date data is available.
        as_dataframe :
            whether to return dynamic attributes as :obj:`pandas.DataFrame`
            or as :obj:`xarray.Dataset`. if :obj:`xarray` library is not
            installed, then this parameter will be ignored and the data will
            be returned as :obj:`pandas.DataFrame`.
        kwargs :
            keyword arguments

        Returns
        -------
        tuple
            A tuple of static and dynamic features. Static features are always
            returned as :obj:`pandas.DataFrame` with shape (stations, static features).
            The index of static features' DataFrame is the station/gauge ids while the columns 
            are names of the static features. Dynamic features are returned either as
            :obj:`xarray.Dataset` or :obj:`pandas.DataFrame` depending upon whether `as_dataframe`
            is True or False and whether the :obj:`xarray` library is installed or not.
            If dynamic features are :obj:`xarray.Dataset`, then this dataset consists of `data_vars`
            equal to the number of stations and station names as :obj:`xarray.Dataset.variables`  
            and `time` and `dynamic_features` as dimensions and coordinates. If 
            dynamic features are returned as :obj:`pandas.DataFrame`, then
            the first index is `time` and the second index is `dynamic_features`.

        Examples
        --------
        >>> from aqua_fetch import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_AUS')
        >>> # get data of 10% of stations
        >>> _, df = dataset.fetch(stations=0.1, as_dataframe=True)  # returns a multiindex dataframe
        ...  # fetch data of 5 (randomly selected) stations
        >>> _, five_random_stn_data = dataset.fetch(stations=5, as_dataframe=True)
        ... # fetch data of 3 selected stations
        >>> _, three_selec_stn_data = dataset.fetch(stations=['912101A','912105A','915011A'], as_dataframe=True)
        ... # fetch data of a single stations
        >>> _, single_stn_data = dataset.fetch(stations='318076', as_dataframe=True)
        ... # get both static and dynamic features as dictionary
        >>> static, dyanmic = dataset.fetch(1, static_features="all", as_dataframe=True)  # -> dict
        >>> dynamic
        ... # get only selected dynamic features
        >>> _, sel_dyn_features = dataset.fetch(stations='318076',
        ...     dynamic_features=['q_cms_obs', 'solrad_wm2_silo'], as_dataframe=True)
        ... # fetch data between selected periods
        >>> _, data = dataset.fetch(stations='318076', st="20010101", en="20101231", as_dataframe=True)

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
              ) -> tuple[pd.DataFrame, Union[pd.DataFrame, "Dataset"]]:
        """
        Reads attributes of more than one stations.

        parameters
        ----------
        stations :
            name/ids of stations for which data is to be fetched. For names
            of stations, see :meth:`stations`.
        dynamic_features :
            list of dynamic features to be fetched. For names of dynamic features,
            see :meth:`dynamic_features`. if ``all``, then all dynamic features 
            will be fetched. If None, then no dynamic attribute will be fetched 
            and the second returned value will be None.
        static_features :
            list of static features to be fetched.
            If `all`, then all static features will be fetched. If None,
            then no static attribute will be fetched. For names of static features,
            see :meth:`static_features`.
        st :
            start of data to be fetched.
        en :
            end of data to be fetched.
        as_dataframe : whether to return the data as :obj:`pandas.DataFrame`. default
                is :obj:`xarray.Dataset` object
        kwargs dict:
            additional keyword arguments

        Returns
        -------
        tuple
            A tuple of static and dynamic features. Static features are always
            returned as :obj:`pandas.DataFrame` with shape (stations, static features).
            The index of static features' DataFrame is the station/gauge ids while the columns 
            are names of the static features. Dynamic features are returned either as
            :obj:`xarray.Dataset` or :obj:`pandas.DataFrame` depending upon whether `as_dataframe`
            is True or False and whether the :obj:`xarray` library is installed or not.
            If dynamic features are :obj:`xarray.Dataset`, then this dataset consists of `data_vars`
            equal to the number of stations and station names as :obj:`xarray.Dataset.variables`  
            and `time` and `dynamic_features` as dimensions and coordinates. If 
            dynamic features are returned as :obj:`pandas.DataFrame`, then
            the first index is `time` and the second index is `dynamic_features`.

        Raises
        ------
        ValueError
            if both ``dynamic_features`` and ``static_features`` are None

        Examples
        --------
        >>> from aqua_fetch import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_AUS')
        ... # find out station ids
        >>> dataset.stations()
        ... # get data of selected stations
        >>> static, dynamic = dataset.fetch_stations_features(['912101A', '912105A', '915011A'],
        ...  as_dataframe=True)
        """
        return self.dataset.fetch_stations_features(
            stations, 
            dynamic_features, 
            static_features, 
            st, 
            en, 
            as_dataframe,
            **kwargs)

    def fetch_dynamic_features(
            self,
            station: str,
            dynamic_features='all',
            st=None,
            en=None,
            as_dataframe=False
    )->Union[pd.DataFrame, "Dataset"]:
        """
        Fetches all or selected dynamic attributes of one station.

        Parameters
        ----------
            station : str
                name/id of station of which to extract the data. For names of stations
                see :meth:`stations`
            dynamic_features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                dynamic features are returned. For names of dynamic features, see
                :meth:`dynamic_features`
            st : Optional (default=None)
                start time from where to fetch the data.
            en : Optional (default=None)
                end time untill where to fetch the data
            as_dataframe : bool, optional (default=False)
                if true, the returned data is :obj:`pandas.DataFrame` otherwise it
                is :obj:`xarray.Dataset`
        
        Returns
        -------
        pd.DataFrame or xr.Dataset
            a :obj:`pandas.DataFrame` or :obj:`xarray.Dataset` depending upon the value of
            `as_dataframe` and whether :obj:`xarray` is installed or not.

        Examples
        --------
        >>> from aqua_fetch import RainfallRunoff
        >>> camels = RainfallRunoff('CAMELS_AUS')
        >>> camels.fetch_dynamic_features('912101A', as_dataframe=True).unstack()
        >>> camels.dynamic_features
        >>> camels.fetch_dynamic_features('912101A',
        ... features=['airtemp_C_silo_max', 'vp_hpa_silo', 'q_cms_obs'],
        ... as_dataframe=True).unstack()
        """
        return self.dataset.fetch_dynamic_features(
            station, dynamic_features, st, en, as_dataframe)

    def fetch_station_features(
            self,
            station: str,
            dynamic_features: Union[str, list, None] = 'all',
            static_features: Union[str, list, None] = None,
            st: Union[str, None] = None,
            en: Union[str, None] = None,
            **kwargs
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetches static and dynamic features for one station.

        Parameters
        -----------
            station : str
                station id/gauge id for which the data is to be fetched.
                For names of stations, see :meth:`stations`
            dynamic_features : str/list, optional
                names of dynamic features/attributes to fetch. For names of dynamic
                features, check the output of :meth:`dynamic_features`
            static_features :
                names of static features/attributes to be fetches. For names of
                static features, check the output of :meth:`static_features`
            st : str,optional
                starting point from which the data to be fetched. By default,
                the data will be fetched from where it is available.
            en : str, optional
                end point of data to be fetched. By default the dat will be fetched

        Returns
        -------
        tuple
            A tuple of static and dynamic features, both as :obj:`pandas.DataFrame`.
            The dataframe of static features will be of single row while the dynamic
            features will be of shape (time, dynamic features).

        Examples
        --------
        >>> from aqua_fetch import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_AUS')
        >>> static, dynamic = dataset.fetch_station_features('912101A')
        >>> static.shape

        >>> dynamic.shape

        """
        return self.dataset.fetch_station_features(station, dynamic_features, static_features, st, en, **kwargs)

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
            name/names of stations. If not given, all stations will be plotted.
            For names of stations, see :meth:`stations`.
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
        >>> from aqua_fetch import RainfallRunoff
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
            area of all stations. For names of stations, see :meth:`stations`.

        Returns
        --------
        pd.DataFrame
            a :obj:`pandas.DataFrame` whose indices are time-steps and columns
            are catchment/station ids.

        """
        return self.dataset.q_mmd(stations)

    def stn_coords(
            self,
            stations: Union[str, List[str]] = "all"
    ) -> pd.DataFrame:
        """
        returns coordinates of stations as :obj:`pandas.DataFrame`
        with ``long`` and ``lat`` as columns.

        Parameters
        ----------
        stations :
            name/names of stations. If not given, coordinates
            of all stations will be returned. For names of stations,
            see :meth:`stations`.

        Returns
        -------
        pd.DataFrame
            :obj:`pandas.DataFrame` with ``long`` and ``lat`` columns.
            The length of dataframe will be equal to number of stations
            wholse coordinates are to be fetched.

        Examples
        --------
        >>> from aqua_fetch import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_CH')
        >>> dataset.stn_coords() # returns coordinates of all stations
        >>> dataset.stn_coords('2004')  # returns coordinates of station whose id is 2004
        >>> dataset.stn_coords(['2004', '6004'])  # returns coordinates of two stations

        >>> from aqua_fetch import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_AUS')
        >>> dataset.stn_coords() # returns coordinates of all stations
        >>> dataset.stn_coords('912101A')  # returns coordinates of station whose id is 912101A
        >>> dataset.stn_coords(['G0050115', '912101A'])  # returns coordinates of two stations

        """
        return self.dataset.stn_coords(stations)

    def get_boundary(
            self,
            station: str,
            as_type: str = 'numpy'
    ):
        """
        returns boundary of a catchment in a required format

        Parameters
        ----------
        station : str
            name/id of catchment. For names of catchments, see :meth:`stations`.
        as_type : str
            'numpy' or 'geopandas'

        Examples
        --------
        >>> from aqua_fetch import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_SE')
        >>> dataset.get_boundary(dataset.stations()[0])
        """
        return self.dataset.get_boundary(station, as_type)

    def plot_catchment(
            self,
            station: str,
            ax: plt_Axes = None,
            show: bool = True,
            **kwargs
    ):
        """
        plots catchment boundaries

        Parameters
        ----------
        station : str
            name/id of station. For names of stations, see :meth:`stations`
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
        >>> from aqua_fetch import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_AUS')
        >>> dataset.plot_catchment()
        >>> dataset.plot_catchment(marker='o', ms=0.3)
        >>> ax = dataset.plot_catchment(marker='o', ms=0.3, show=False)
        >>> ax.set_title("Catchment Boundaries")
        >>> plt.show()

        """
        return self.dataset.plot_catchment(station, ax, show, **kwargs)

    def stations(self) -> List[str]:
        """
        returns names/ids of all stations. This can be either gauge id or basin id.
        Every catchment has a unique name/id which can be used to fetch its data.

        Examples
        --------
        >>> from aqua_fetch import RainfallRunoff
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
        >>> from aqua_fetch import RainfallRunoff
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
        >>> from aqua_fetch import RainfallRunoff
        >>> dataset = RainfallRunoff('CAMELS_AUS')
        >>> dataset.end()
        """
        return self.dataset.end
