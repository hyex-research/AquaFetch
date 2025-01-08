

from typing import Union, List

import pandas as pd

from .._datasets import Datasets


class SanFranciscoBay(Datasets):
    """
    Time series of water quality parameters from 1969 - 2015.
        chl-a
        dissolved oxygen
        suspended particulate matter
        nitrate
        nitrite
        ammonium
        silicate
        phosphate
    """
    url = "https://static-content.springer.com/esm/art%3A10.1038%2Fsdata.2017.98/MediaObjects/41597_2017_BFsdata201798_MOESM18_ESM.zip"

    def __init__(self, path=None, **kwargs):
        super().__init__(path=path, **kwargs)
        self.ds_dir = path
        self._download()

    def fetch(
            self,
            parameters:Union[str, List[str]]='all',
    )->pd.DataFrame:
        """



        Parameters
        ----------
        parameters : Union[str, List[str]], optional
            The parameters to return. The default is 'all'.

        Returns
        -------
        pd.DataFrame
            DESCRIPTION.

        """
        
        raise NotImplementedError
