__all__ = ["SyltRoads"]

import os
from typing import Union, List

import pandas as pd

from .._datasets import Datasets
from ..utils import check_attributes, download_and_unzip

# todo: entrance and entrace are same

class SyltRoads(Datasets):
    """
    Dataset of physico-hydro-chemical time series data at Sylt Roads from 1973 - 2019
    following `Rick et al., 2023 <https://doi.org/10.5194/essd-15-1037-2023>`_ .
    Following parameters are available

        - ``location``
        - ``Depth water [m]``
        - ``Sal``
        - ``Temp [°C]``
        - ``[PO4]3- [µmol/l]``
        - ``[NH4]+ [µmol/l]``
        - ``[NO2]- [µmol/l]``
        - ``[NO3]- [µmol/l]``
        - ``Si(OH)4 [µmol/l]``
        - ``SPM [mg/l]``
        - ``pH``
        - ``O2 [µmol/l]``
        - ``Chl a [µg/l]``
        - ``DON [µmol/l]``
        - ``DOP [µmol/l]``
        - ``DIN [µmol/l]``

    Examples
    --------
    >>> from aqua_fetch import SyltRoads
    >>> ds = SyltRoads()

    """
    url = {
        "list_entrance_2014.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.873545?format=textfile",
        "list_reede_2014.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.873549?format=textfile",
        "list_ferry_2014.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.873547?format=textfile",
        "list_reede_2015.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918018?format=textfile",
        "list_entrance_2015.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918032?format=textfile",
        "list_ferry_2015.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918027?format=textfile",
        "list_reede_2016.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918023?format=textfile",
        "list_entrance_2016.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918033?format=textfile",
        "list_ferry_2016.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918028?format=textfile",
        "list_reede_2017.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918024?format=textfile",
        "list_entrace_2017.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918034?format=textfile",
        "list_ferry_2017.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918029?format=textfile",
        "list_reede_2018.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918025?format=textfile",
        "list_entrace_2018.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918035?format=textfile",
        "list_ferry_2018.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918030?format=textfile",
        "list_reede_2019.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918026?format=textfile",
        "list_entrace_2019.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918036?format=textfile",
        "list_ferry_2019.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.918031?format=textfile",
        "1973_2013.txt":
            "https://doi.pangaea.de/10.1594/PANGAEA.150032?format=textfile"
    }

    def __init__(self, path=None, **kwargs):
        super().__init__(path=path, **kwargs)
        self.ds_dir = path
        self._download()

        self._parameters = self._get_data().columns.tolist()

    @property
    def parameters(self)->List[str]:
        """returns names of parameters in the dataset"""
        return self._parameters

    @property
    def raw_data_path(self):
        path = os.path.join(self.path, 'raw_data')
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def stn_coords(self)->pd.DataFrame:
        """
        Returns the coordinates of all the stations in the dataset in wgs84
        projection.

        Returns
        -------
        pd.DataFrame
            A dataframe with columns 'lat', 'long'
        """
        entrace = 55.038300 , 8.438300
        ferry = 55.015530 , 8.439900
        reede = 55.030000 , 8.460000

        return pd.DataFrame([entrace, ferry, reede], 
                            columns=['lat', 'long'],
                    index=['entrace', 'ferry', 'reede'])

    def fetch(
            self,
            parameters: Union[str, List[str]] = "all",
    )->pd.DataFrame:
        """
        Fetch the data from the dataset

        Parameters
        ----------
        parameters : str or List[str], optional
            Parameters to fetch. Default is None which will fetch all parameters

        Returns
        -------
        pd.DataFrame
            DataFrame containing the data

        Examples
        --------
        >>> from aqua_fetch import SyltRoads
        >>> ds = SyltRoads()
        >>> df = ds.fetch()
        >>> df.shape
        (5710, 16)
        >>> len(ds.parameters)
        16
        >>> ds.fetch(['Sal', 'Temp [°C]', 'pH']).shape
        (5710, 3)
        """

        parameters = check_attributes(parameters, self.parameters, 'parameters')
        return self._get_data()[parameters]
    
    def _get_data(self)->pd.DataFrame:
        df = self._get_historical_data()
        df1 = self._read_data_2014_2019()

        df = pd.concat([df, df1])

        return df

    def _read_files(self, location:str)->pd.DataFrame:
        entrace_files = [f for f in os.listdir(self.path) if f.startswith(location)]
        # read first file and skip all rows in the file until the row which starts with Date
        # then read the rest of the file and use the rown which starts with Date as header
        entrace = []
        for file in entrace_files:
            with open(os.path.join(self.path, file)) as f:
                for line in f:
                    if line.startswith('*/'):
                        break
                df = pd.read_csv(f, sep='\t', index_col=0)
            
            entrace.append(df)

        entrace = pd.concat(entrace)

        return entrace
    
    def _read_data_2014_2019(self)->pd.DataFrame:

        dfs = []
        for loc in ["list_entrace", "list_reede", "list_ferry", "list_entrance"]:
            df = self._read_files(loc)
            df.insert(0, 'location', loc)

            dfs.append(df)

        return pd.concat(dfs)
    
    def _get_historical_data(self)->pd.DataFrame:
        """gets data from 1973 - 2013"""

        fpath = os.path.join(self.path, "1973_2013.csv")

        if os.path.exists(fpath):
            if self.verbosity:
                print(f"Reading data from pre-existing {fpath}")
            return pd.read_csv(fpath, index_col=0)
        
        if self.verbosity:
            print(f"Downloading data from 1973 - 2013")

        dfs = []
        for stn, col_idx in {"list_reede": 0, "list_entrance": 1,  "list_ferry": 3}.items():
            df = self._get_historical_data_stn(stn, col_idx)
        
            dfs.append(df)
        
        df = pd.concat(dfs)

        df.to_csv(fpath, index_label='Date/Time')

        return df            
    
    def _get_historical_data_stn(self, stn, col_idx):
        # read the file 1973_2013.txt
        with open(os.path.join(self.path, "1973_2013.txt")) as f:
            for line in f:
                if line.startswith('*/'):
                    break
            doi_df = pd.read_csv(f, sep='\t', index_col=0)

        dfs = []
        for year, doi in zip(doi_df.index, doi_df.iloc[:, col_idx]):
            if isinstance(doi, str):
                f = doi.split('.')[-1]
                url =  f"https://doi.pangaea.de/10.1594/PANGAEA.{f}?format=textfile"
                fname = f'{year}_{stn}.txt'
                download_and_unzip(self.raw_data_path, {fname: url}, verbosity=0)

                fpath = os.path.join(self.raw_data_path, fname)

                with open(fpath) as f:
                    for line in f:
                        if line.startswith('*/'):
                            break
                    df_ = pd.read_csv(f, sep='\t', index_col=0)
                    df_.insert(0, 'location', stn)

                dfs.append(df_)

        df = pd.concat(dfs)
        return df
