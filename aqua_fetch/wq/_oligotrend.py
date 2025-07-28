
__all__ = ["Oligotrend"]

import os
from typing import Union, List, Dict

import pandas as pd
import numpy as np

from .._datasets import Datasets
from ..utils import check_attributes


class Oligotrend(Datasets):
    """
    A global database of multi-decadal (1986-2023) timeseries of chlorophyll-a and 16 others including N and P,
    from 1846 unique monitoring locations across estuaries (n=238), lakes (n=687), and rivers (969).
    The datasets consists of 4.3 million observations and most timeseries cover the period 1986-2022
    and comprise at least 15 years of Chl-a observations.
    For more details, see `Minaudo et al., 2025 <https://doi.org/10.5194/essd-17-3411-2025>_`. 
    The data is fetched from `EDI data portal <https://portal.edirepository.org/nis/mapbrowse?packageid=edi.1778.3>`_.

    Examples
    --------
    >>> from aqua_fetch import Oligotrend
    >>> ds = Oligotrend(path='/path/to/data')
    get names of parameters in the dataset
    >>> ds.parameters()
    >>> len(ds.parameters())
    17
    get list of stations in the dataset
    >>> ds.stations()
    >>> len(ds.stations())
    1846
    >>> len(ds.lakes())
    685
    >>> len(ds.rivers())
    924
    >>> len(ds.estuaries())
    237
    get parameters of a single station
    >>> data = ds.fetch_stn_parameters('lake_atlanticoceanseaboard_usa12721')
    >>> data.shape
    (303, 3)
    get all parameters for specific stations
    >>> data = ds.fetch_stns_parameters(['river_ebro_9027', 'river_elbe_elbe_10'])
    >>> data['river_ebro_9027'].shape
    (287, 8)
    >>> data['river_elbe_elbe_10'].shape
    (8154, 12)
    Get only 'chla' parameter for the stations
    >>> data1 = ds.fetch_stns_parameters(['river_ebro_9027', 'river_elbe_elbe_10'],
    ...                                 parameters=['chla'])
    >>> data1['river_ebro_9027'].shape
    (177, 1)
    >>> data1['river_elbe_elbe_10'].shape
    (413, 1)
    """

    # todo : why stations are 1846 and not 1894 as mentioned in the paper?

    url = {
'data_sources_oligotrend.csv': "https://pasta.lternet.edu/package/data/eml/edi/1778/3/c828f5056b9c46b8e120bc2c9406de05",
'oligotrend_L1.csv': "https://pasta.lternet.edu/package/data/eml/edi/1778/3/cc48f89ff50a51a6e9dbf9e35fc16c20",
"oligotrend_L1_xy_gis.csv": "https://pasta.lternet.edu/package/data/eml/edi/1778/3/4fb9081418e2f8112f094bb284197dde",
"oligotrend_L2_trends.csv": "https://pasta.lternet.edu/package/data/eml/edi/1778/3/00446b372fabf8e97bb23c2d47482a0d",
    }

    def __init__(self, path=None, **kwargs):
        super().__init__(path=path, **kwargs)
        self.ds_dir = path
        self._download()

        self._stations = self.gis_data().index.tolist()
        self._parameters = self.l1_data()['variable'].unique().tolist()

    def stn_coords(self)->pd.DataFrame:
        """
        Returns the coordinates of all the stations in the dataset in wgs84
        projection.

        Returns
        -------
        pd.DataFrame
            A dataframe with columns 'lat', 'long'
        """
        coords =  self.gis_data()[['Y', 'X']]
        coords.columns = ['lat', 'long']
        return coords

    def parameters(self)->List[str]:
        """
        Returns the list of names of parameters in the dataset.
        """
        return self._parameters

    def stations(self)->List[str]:
        """
        returns the list of stations in the dataset
        """
        return self._stations
    
    def lakes(self)->List[str]:
        """
        Returns the list of stations which are lakes in the dataset.
        """
        df = self.gis_data()
        return df.loc[df['ecsystm'] == 'lake', :].index.tolist()
    
    def rivers(self)->List[str]:
        """
        Returns the list of stations which are rivers in the dataset.
        """
        df = self.gis_data()
        return df.loc[df['ecsystm'] == 'river', :].index.tolist()
    
    def estuaries(self)->List[str]:
        """
        Returns the list of stations which are estuaries in the dataset.
        """
        df = self.gis_data()
        return df.loc[df['ecsystm'] == 'estuary', :].index.tolist()

    def sources(self):
        """
        Returns the sources of the dataset.
        """
        df = pd.read_csv(os.path.join(self.path, 'data_sources_oligotrend.csv'))
        return df

    def gis_data(self)->pd.DataFrame:
        """
        Returns the GIS data of the dataset.
        """
        df = pd.read_csv(os.path.join(self.path, 'oligotrend_L1_xy_gis.csv'),
                         index_col='uniqID')
        return df

    def l1_data(self)->pd.DataFrame:
        """
        Returns the oligotrend_L1.csv file and returns as dataframe of shape
        5056630, 7.
        """
        df = pd.read_csv(os.path.join(self.path, 'oligotrend_L1.csv'),
                         dtype={
                             'basin': str,
                             'ecosystm': str,
                             'id': str,
                             'variable': str,
                             'value': np.float32,
                             'flat': int,
                         },
                         parse_dates=['date'],
                         date_format=lambda x: pd.to_datetime(x, format='%Y-%m-%d')
                         )
        # the vlaues in id column have .csv at the end, we remove it
        df['id'] = df['id'].str.replace('.csv', '', regex=False)
        return df

    def fetch_stns_parameters(
            self,
            stns: Union[str, List[str]],
            parameters: Union[str, List[str]] = "all",
    )-> Dict[str, pd.DataFrame]:
        """
        Fetches the parameters for the given stations.

        Parameters
        ----------
        stns : str or list of str
            The station(s) to fetch the parameters for.
        parameters : str or list of str, optional
            The parameter(s) to fetch. If 'all', all parameters are fetched.

        Returns
        -------
        Dict[str, pd.DataFrame]
            A dictionary with the station id as key and a dataframe of parameters as value.
        
        Examples
        --------
        >>> data = ds.fetch_stns_parameters(['river_ebro_9027', 'river_elbe_elbe_10'])
        >>> data['river_ebro_9027'].shape
        (287, 8)
        >>> data['river_elbe_elbe_10'].shape
        (8154, 12)
        >>> data = ds.fetch_stns_parameters(['river_ebro_9027', 'river_elbe_elbe_10'], 'chla')
        >>> data['river_ebro_9027'].shape
        (177, 1)
        >>> data['river_elbe_elbe_10'].shape
        (413, 1)
        """

        df = self.l1_data()

        stns = check_attributes(stns, self.stations(), 'stns')
        parameters = check_attributes(parameters, self.parameters(), 'parameters')

        cond1 = df['id'].isin(stns)
        cond2 = df['variable'].isin(parameters)

        df = df.loc[cond1 & cond2, :]

        # make a dictionary with the station id as key and a dataframe of parameters as value
        results = {}
        for stn in stns:
            stn_df = df.loc[df['id'] == stn, ['variable', 'date', 'value']]
            # for some date, variable paris there are more than 1 value which means there are duplicates
            # for example for stn river_ebro_9027 for chla on date 1991-07-01, there are 2 values
            # so we have use aggfunc, ideally we should not have duplicates
            stn_df = stn_df.pivot_table(index='date', columns='variable', values='value', aggfunc='mean')
            results[stn] = stn_df
        
        return results

    def fetch_stn_parameters(
            self,
            stn:str,
            parameters: Union[str, List[str]] = "all",
    ):
        """

        Examples
        --------
        >>> stn_df = ds.fetch_stn_parameters('lake_atlanticoceanseaboard_usa12721')
        >>> stn_df.shape 
        (303, 3)
        """
        assert isinstance(stn, str), "stn should be a string"

        df = self.l1_data()

        stn = check_attributes(stn, self.stations(), 'stn')

        parameters = check_attributes(parameters, self.parameters(), 'parameters')

        cond1 = df['id'].isin(stn)

        cond2 = df['variable'].isin(parameters)

        df = df.loc[cond1 & cond2, :]

        df = df[['variable', 'date', 'value']].pivot(index='date', columns='variable', values='value')

        return df

    def get_stations(
            self,
            parameter:str,
            ecosystm: str = 'river',
    ) -> pd.Series:
        """
        Returns a list of stations that have the specified parameter.

        Examples
        --------
        >>>> chla_stns = ds.get_stations('chla')
        >>>> len(chla_stns)
        969
        """
        df = self.l1_data()
        cond1 = df['variable'] == parameter
        cond2 = df['ecosystem'] == ecosystm
        return df.loc[cond1 & cond2, 'id'].unique().tolist()

    def num_obs(self, parameter:str):

        assert isinstance(parameter, str), "parameter should be a string"
        assert parameter in self.parameters(), f"parameter {parameter} not found in {self.parameters()}"

        l1_data = self.l1_data()
        obs_counts = l1_data.groupby(['id', 'variable']).size().reset_index(name='count')

        para_counts = obs_counts.loc[obs_counts['variable']==parameter]

        # use the 'id' column as index
        para_counts = para_counts.set_index('id')

        para_counts.pop('variable')
        return para_counts.loc[self.stations(), 'count']
