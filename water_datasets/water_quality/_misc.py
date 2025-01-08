

from typing import Union, List

import pandas as pd

def SanFranciscoBay(
        parameters:Union[str, List[str]]='all',
)->pd.DataFrame:
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

    Parameters
    ----------
    parameters : Union[str, List[str]], optional
        The parameters to return. The default is 'all'.

    Returns
    -------
    pd.DataFrame
        DESCRIPTION.

    """
    url = "https://static-content.springer.com/esm/art%3A10.1038%2Fsdata.2017.98/MediaObjects/41597_2017_BFsdata201798_MOESM18_ESM.zip"
    data = pd.read_csv(url)
    if parameters != 'all':
        data = data[parameters]
    return data