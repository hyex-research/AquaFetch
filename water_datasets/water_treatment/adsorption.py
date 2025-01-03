
import os
from typing import Union, Tuple, Any, List, Dict

import numpy as np
import pandas as pd

from ..utils import encode_column, LabelEncoder, OneHotEncoder



def ec_removal_biochar(
        input_features:List[str]=None,
        encoding:str = None
)->Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Data of removal of emerging contaminants/pollutants from wastewater
    using biochar. The data consists of three types of features,
    1) adsorption experimental conditions, 2) elemental composition of
    adsorbent (biochar) and parameters representing
    physical and synthesis conditions of biochar.
    For more description of this data see `Jaffari et al., 2023 <https://doi.org/10.1016/j.cej.2023.143073>`_


    Parameters
    ----------
    input_features :
        By default following features are used as input
            - ``Adsorbent``
            - ``Pyrolysis temperature``
            - ``Pyrolysis time``
            - ``C``
            - ``H``
            - ``O``
            - ``N``
            - ``(O+N)/C``
            - ``Ash``
            - ``H/C``
            - ``O/C``
            - ``Surface area``
            - ``Pore volume``
            - ``Average pore size``
            - ``Pollutant``
            - ``Adsorption time``
            - ``concentration``
            - ``Solution pH``
            - ``RPM``
            - ``Volume``
            - ``Adsorbent dosage``
            - ``Adsorption temperature``
            - ``Ion concentration``
            - ``Humid acid``
            - ``Wastewater type``
            - ``Adsorption type``

    encoding : str, default=None
        the type of encoding to use for categorical features. If not None, it should
        be either ``ohe`` or ``le``.

    Returns
    --------
    tuple
        A tuple of length two. The first element is a DataFrame while the
        second element is a dictionary consisting of encoders with ``adsorbent``
        ``pollutant``, ``ww_type`` and ``adsorption_type`` as keys.

    Examples
    --------
    >>> from water_datasets import ec_removal_biochar
    >>> data, *_ = ec_removal_biochar()
    >>> data.shape
    (3757, 27)
    >>> data, encoders = ec_removal_biochar(encoding="le")
    >>> data.shape
    (3757, 27)
    >>> len(set(encoders['adsorbent'].inverse_transform(data.iloc[:, 22])))
    15
    >>> len(set(encoders['pollutant'].inverse_transform(data.iloc[:, 23])))
    14
    >>> set(encoders['ww_type'].inverse_transform(data.iloc[:, 24]))
    {'Ground water', 'Lake water', 'Secondary effluent', 'Synthetic'}
    >>> set(encoders['adsorption_type'].inverse_transform(data.iloc[:, 25]))
    {'Competative', 'Single'}

    We can also use one hot encoding to convert categorical features into
    numerical features. This will obviously increase the number of features/columns in DataFrame

    >>> data, encoders = ec_removal_biochar(encoding="ohe")
    >>> data.shape
    (3757, 58)
    >>> len(set(encoders['adsorption_type'].inverse_transform(data.iloc[:, 22:37].values)))
    15
    >>> len(set(encoders['pollutant'].inverse_transform(data.iloc[:, 37:51].values)))
    14
    >>> set(encoders['ww_type'].inverse_transform(data.iloc[:, 51:55].values))
    {'Ground water', 'Lake water', 'Secondary effluent', 'Synthetic'}
    >>> set(encoders['adsorption_type'].inverse_transform(data.iloc[:, 55:-1].values))
    {'Competative', 'Single'}

    """
    fpath = os.path.join(os.path.dirname(__file__), "data", 'qe_biochar_ec.csv')
    url = 'https://raw.githubusercontent.com/ZeeshanHJ/Adsorption-capacity-prediction-for-ECs/main/Raw_data.csv'

    if os.path.exists(fpath):
        data = pd.read_csv(fpath)
    else:
        data = pd.read_csv(url)
        # remove space in 'Pyrolysis temperature '
        data['Pyrolysis temperature'] = data.pop('Pyrolysis temperature ')

        data['Adsorbent'] = data.pop('Adsorbent')
        data['Pollutant'] = data.pop('Pollutant')
        data['Wastewater type'] = data.pop('Wastewater type')
        data['Adsorption type'] = data.pop('Adsorption type')

        data['Capacity'] = data.pop('Capacity')

        data.to_csv(fpath, index=False)

    def_inputs = [
        'Pyrolysis temperature',
        'Pyrolysis time',
        'C',
        'H',
        'O',
        'N',
        '(O+N)/C',
        'Ash',
        'H/C',
        'O/C',
        'Surface area',
        'Pore volume',
        'Average pore size',
        'Adsorption time',
        'Initial concentration',
        'Solution pH',
        'RPM',
        'Volume',
        'Adsorbent dosage',
        'Adsorption temperature',
        'Ion concentration',
        'Humic acid',
        'Adsorbent',
        'Pollutant',
        'Wastewater type',
        'Adsorption type',
    ]

    if input_features is not None:
        assert isinstance(input_features, list)
        assert all([feature in def_inputs for feature in input_features])
    else:
        input_features = def_inputs

    data = data[input_features + ['Capacity']]

    ads_enc, pol_enc, wwt_enc, adspt_enc = None, None, None, None
    if encoding:
        data, _, ads_enc = encode_column(data, 'Adsorbent', encoding)
        data, _, pol_enc = encode_column(data, 'Pollutant', encoding)
        data, _, wwt_enc = encode_column(data, 'Wastewater type', encoding)
        data, _, adspt_enc = encode_column(data, 'Adsorption type', encoding)

        # putting capacity at the end
        data['Capacity'] = data.pop('Capacity')

    encoders = {
        "adsorbent": ads_enc,
        "pollutant": pol_enc,
        "ww_type": wwt_enc,
        "adsorption_type": adspt_enc
    }
    return data, encoders


def po4_removal_biochar():
    """
    `Iftikhar et al., 2023 <https://doi.org/10.1016/j.chemosphere.2024.144031>`_
    """
    url = "https://github.com/Sara-Iftikhar/po4_removal_ml/blob/main/scripts/master_sheet_0802.xlsx"
    return


def cr_removal():
    """
    `Ishtiaq et al., 2024 <https://doi.org/10.1016/j.jece.2024.112238>`_
    """

    url = "https://gitlab.com/atrcheema/envai103/-/blob/main/data/data.csv"
    return


def heavy_metal_removal():
    """
    `Jaffari et al., 2024 <https://doi.org/10.1016/j.jhazmat.2023.132773>`_
    """
    return


def heavy_metal_removal_Shen():
    """
    `Shen et al., 2024 <https://doi.org/10.1016/j.jhazmat.2024.133442>`_
    """
    return


def industrial_dye_removal():
    """
    `Iftikhar et al., 2023 <https://doi.org/10.1016/j.seppur.2023.124891>`_
    """
    url = "https://github.com/Sara-Iftikhar/ai4adsorption/blob/main/scripts/Dyes%20data.xlsx"
    return


def P_recovery():
    """
    `Leng et al., 2024 <https://doi.org/10.1016/j.jwpe.2024.104896>`_
    """
    url = "https://zenodo.org/records/14586314/files/P_recovery.csv?download=1"
    return


def N_recovery():
    """
    `Leng et al., 2024 <https://doi.org/10.1016/j.jwpe.2024.104896>`_
    """
    url = "https://zenodo.org/records/14586314/files/N_recovery.csv?download=1"
    return
