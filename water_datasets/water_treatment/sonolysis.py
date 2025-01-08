
__all__ = ["cyanobacteria_disinfection"]

from typing import Union, Tuple, Any, List, Dict

import pandas as pd

from ..utils import (
    check_attributes,
    LabelEncoder,
    OneHotEncoder,
    maybe_download_and_read_data,
    encode_cols
)

def cyanobacteria_disinfection():
    """
    `Jaffari et al., 2024 <https://doi.org/10.1016/j.jhazmat.2024.133762>`_
    """

    url = "https://gitlab.com/atrcheema/flowcam/-/raw/03cfdc0ac32e31b44eae2e0abb3cd6770517ce06/scripts/data_0315.csv"

    data = maybe_download_and_read_data(url, "cyanobacteria_disinfection.csv")

    return data, {}
