

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


def white_clay_creek(
        parameters:Union[str, List[str]]='all',
):
    """
    Time series of water quality parameters from 2001 - 2012.
        
        - chl-a
        - Dissolved Organic Carbon
    """
    url = {
        "chla": "https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/",
        "doc": "https://portal.edirepository.org/nis/dataviewer?packageid=edi.386.1&entityid=3f802081eda955b2b0b405b55b85d11c"
        }

    raise NotImplementedError