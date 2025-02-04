
__all__ = ["RC4USCoast"]

import os
from typing import Union, List

import numpy as np
import pandas as pd

from .._backend import xarray as xr

from .._datasets import Datasets


class RC4USCoast(Datasets):
    """
    Monthly river water chemistry (N, P, SIO2, DO, ... etc), discharge and
    temperature of 140 monitoring sites of US coasts from 1950 to 2020
    following the work of
    `Gomez et al., 2022 <https://doi.org/10.5194/essd-15-2223-2023>`_.

    Examples
    --------
    >>> from aqua_fetch import RC4USCoast
    >>> dataset = RC4USCoast()
    >>> len(dataset.stations)
    140
    >>> len(dataset.parameters)
    27
    >>> stn_coords = dataset.stn_coords()
    >>> stn_coords.shape
    (140, 2)

    """
    url = {
'dataset_info.xlsx':
    'https://www.ncei.noaa.gov/data/oceans/archive/arc0207/0260455/3.3/data/0-data/dataset_info.xlsx',
'mclim_19501989_chem.nc':
    'https://www.ncei.noaa.gov/data/oceans/archive/arc0207/0260455/3.3/data/0-data/mclim_19501989_chem.nc',
'mclim_19501989_disc.nc':
    'https://www.ncei.noaa.gov/data/oceans/archive/arc0207/0260455/3.3/data/0-data/mclim_19501989_disc.nc',
'mclim_19502022_chem.nc':
    'https://www.ncei.noaa.gov/data/oceans/archive/arc0207/0260455/3.3/data/0-data/mclim_19502022_chem.nc',
'mclim_19502022_disc.nc':
    'https://www.ncei.noaa.gov/data/oceans/archive/arc0207/0260455/3.3/data/0-data/mclim_19502022_disc.nc',
'mclim_19902022_chem.nc':
    'https://www.ncei.noaa.gov/data/oceans/archive/arc0207/0260455/3.3/data/0-data/mclim_19902022_chem.nc',
'mclim_19902022_disc.nc':
    'https://www.ncei.noaa.gov/data/oceans/archive/arc0207/0260455/3.3/data/0-data/mclim_19902022_disc.nc',
'series_chem.nc':
    'https://www.ncei.noaa.gov/data/oceans/archive/arc0207/0260455/3.3/data/0-data/series_chem.nc',
'series_disc.nc':
    'https://www.ncei.noaa.gov/data/oceans/archive/arc0207/0260455/3.3/data/0-data/series_disc.nc',
    }

    def __init__(self, path=None, *args, **kwargs):
        """

        Parameters
        ----------
        path :
            path where the data is already downloaded. If None, the data will
            be downloaded into the disk.
        """
        super(RC4USCoast, self).__init__(path=path, *args, **kwargs)
        self.path = path
        self._download()

    @property
    def chem_fname(self)->str:
        return os.path.join(self.path, "series_chem.nc")

    @property
    def q_fname(self) -> str:
        return os.path.join(self.path, "series_disc.nc")

    @property
    def info_fname(self) -> str:
        return os.path.join(self.path, "info.xlsx")

    @property
    def stations(self)->List[str]:
        """
        Examples
        --------
        >>> from aqua_fetch import RC4USCoast
        >>> ds = RC4USCoast(path=r'F:\data\RC4USCoast')
        >>> len(ds.stations)
        140
        """
        return xr.open_dataset(self.q_fname).RC4USCoast_ID.data.astype(str).tolist()

    @property
    def parameters(self)->List[str]:
        """
        returns names of parameters

        Examples
        --------
        >>> from aqua_fetch import RC4USCoast
        >>> ds = RC4USCoast()
        >>> len(ds.parameters)
        27
        """
        df = xr.open_dataset(self.chem_fname)
        return list(df.data_vars.keys())

    def stn_coords(self)->pd.DataFrame:
        """
        Returns the coordinates of all the stations in the dataset in wgs84
        projection.

        Returns
        -------
        pd.DataFrame
            A dataframe with columns 'lat', 'long'
        """

        fpath = os.path.join(self.path, 'series_chem.nc')
        data = xr.open_dataset(fpath)

        # get the coordinates named 'lat' and 'lon' from data
        coord = data['lat'].to_dataframe()#.reset_index()
        coord =  coord.iloc[:, 0:2].astype(np.float32)
        coord.columns = ['lat', 'long']
        return coord

    @property
    def start(self)->pd.Timestamp:
        return pd.Timestamp(xr.open_dataset(self.q_fname).time.data[0])

    @property
    def end(self)->pd.Timestamp:
        return pd.Timestamp(xr.open_dataset(self.q_fname).time.data[-1])

    def fetch_chem(
            self,
            parameter,
            stations: Union[List[int], int, str] = "all",
            as_dataframe:bool = False,
            st: Union[int, str, pd.DatetimeIndex] = None,
            en: Union[int, str, pd.DatetimeIndex] = None,
    ):
        """
        Returns water chemistry parameters from one or more stations.

        parameters
        ----------
        parameter : list, str
            name/names of parameters to fetch
        stations : list, str
            name/names of stations from which the parameters are to be fetched
        as_dataframe : bool (default=False)
            whether to return data as pandas.DataFrame or xarray.Dataset
        st :
            start time of data to be fetched. The default starting
            date is 19500101
        en :
            end time of data to be fetched. The default end date is
            20201201

        Returns
        -------
        pd.DataFrame or xarray Dataset

        Examples
        --------
        >>> from aqua_fetch import RC4USCoast
        >>> ds = RC4USCoast()
        >>> data = ds.fetch_chem(['temp', 'do'])
        >>> data
        >>> data = ds.fetch_chem(['temp', 'do'], as_dataframe=True)
        >>> data.shape  # this is a multi-indexed dataframe
        (119280, 4)
        >>> data = ds.fetch_chem(['temp', 'do'], st="19800101", en="20181230")
        """
        if isinstance(parameter, str):
            parameter = [parameter]

        ds = xr.open_dataset(self.chem_fname)[parameter]
        if stations == "all":
            pass
        elif not isinstance(stations, list):
            stations = [stations]
            ds = ds.sel(RC4USCoast_ID=stations)
        elif isinstance(stations, list):
            ds = ds.sel(RC4USCoast_ID = stations)
        else:
            assert stations is None

        ds = ds.sel(time=slice(st or self.start, en or self.end))

        if as_dataframe:
            return ds.to_dataframe()
        return ds

    def fetch_q(
            self,
            stations:Union[int, List[int], str, np.ndarray] = "all",
            as_dataframe:bool=True,
            nv=0,
            st: Union[int, str, pd.DatetimeIndex] = None,
            en: Union[int, str, pd.DatetimeIndex] = None,
    ):
        """returns discharge data

        parameters
        -----------
        stations :
            stations for which q is to be fetched
        as_dataframe : bool (default=True)
            whether to return the data as pd.DataFrame or as xarray.Dataset
        nv : int (default=0)
        st :
            start time of data to be fetched. The default starting
            date is 19500101
        en :
            end time of data to be fetched. The default end date is
            20201201

        Examples
        --------
        >>> from aqua_fetch import RC4USCoast
        >>> ds = RC4USCoast()
        # get data of all stations as DataFrame
        >>> q = ds.fetch_q("all")
        >>> q.shape
        (852, 140)  # where 140 is the number of stations
        # get data of only two stations
        >>> q = ds.fetch_q([1,10])
        >>> q.shape
        (852, 2)
        # get data as xarray Dataset
        >>> q = ds.fetch_q("all", as_dataframe=False)
        >>> type(q)
        xarray.core.dataset.Dataset
        # getting data between specific periods
        >>> data = ds.fetch_q("all", st="20000101", en="20181230")
        """
        q = xr.open_dataset(self.q_fname)
        if stations:
            if stations == "all":
                q = q.sel(nv=nv)
            elif not isinstance(stations, list):
                stations = [stations]
                q = q.sel(RC4USCoast_ID=stations, nv=nv)
            elif isinstance(stations, list):
                q = q.sel(RC4USCoast_ID=stations, nv=nv)
            else:
                raise ValueError(f"invalid {stations}")

        q = q.sel(time=slice(st or self.start, en or self.end))

        if as_dataframe:
            return q.to_dataframe()['disc'].unstack()
        return q
