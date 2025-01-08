
__all__ = ["micropollutant_removal_osmosis", "ion_transport_via_reverse_osmosis"]

from typing import Union, Tuple, Any, List, Dict

import pandas as pd

from ..utils import (
    check_attributes,
    LabelEncoder,
    OneHotEncoder,
    maybe_download_and_read_data,
    encode_cols
)


def micropollutant_removal_osmosis():
    """
    `Jeong et al., 2021 <https://doi.org/10.1021/acs.est.1c04041>`_
    """

    url = "https://pubs.acs.org/doi/suppl/10.1021/acs.est.1c04041/suppl_file/es1c04041_si_002.xlsx"

    data = maybe_download_and_read_data(url, "micropollutant_removal_osmosis.csv")
    return data, {}


def ion_transport_via_reverse_osmosis():
    """
    `Jeong et al., 2023 <https://doi.org/10.1021/acs.est.2c08384>`_
    """

    url = "https://pubs.acs.org/doi/suppl/10.1021/acs.est.2c08384/suppl_file/es2c08384_si_002.xlsx"

    data = maybe_download_and_read_data(url, "ion_transport_via_reverse_osmosis.csv")

    return data, {}
