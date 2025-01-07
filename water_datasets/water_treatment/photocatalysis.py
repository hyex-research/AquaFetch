import os
from typing import Union, Tuple, Any, List

import numpy as np
import pandas as pd

from ..utils import encode_column, LabelEncoder, OneHotEncoder, encode_cols


def mg_degradation(
        inputs: List[str] = None,
        target: str = "Efficiency (%)",
        encoding: str = None
) -> Tuple[pd.DataFrame,
Union[LabelEncoder, OneHotEncoder, Any],
Union[LabelEncoder, OneHotEncoder, Any]]:
    """
    This data is about photocatalytic degradation of melachite green dye using
    nobel metal dobe BiFeO3. For further description of this data see
    `Jafari et al., 2023 <https://doi.org/10.1016/j.jhazmat.2022.130031>`_ and
    for the use of this data for removal efficiency prediction `see <https://github.com/ZeeshanHJ/Photocatalytic_Performance_Prediction>`_ .
    This dataset consists of 1200 points collected during ~135 experiments.

    Parameters
    ----------
        inputs : list, optional
            features to use as input. By default following features are used as input

                - ``Catalyst_type``
                - ``Surface area``
                - ``Pore Volume``
                - ``Catalyst_loading (g/L)``
                - ``Light_intensity (W)``
                - ``time (min)``
                - ``solution_pH``
                - ``HA (mg/L)``
                - ``Anions``
                - ``Ci (mg/L)``
                - ``Cf (mg/L)``

        target : str, optional, default="Efficiency (%)"
            features to use as target. By default ``Efficiency (%)`` is used as target
            which is photodegradation removal efficiency of dye from wastewater. Following
            are valid target names

                - ``Efficiency (%)``
                - ``k_first``
                - ``k_2nd``

        encoding : str, default=None
            type of encoding to use for the two categorical features i.e., ``Catalyst_type``
            and ``Anions``, to convert them into numberical. Available options are ``ohe``,
            ``le`` and None. If ohe is selected the original input columns are replaced
            with ohe hot encoded columns. This will result in 6 columns for Anions and
            15 columns for Catalyst_type.

    Returns
    -------
    data : pd.DataFrame
        a pandas dataframe consisting of input and output features. The default
        setting will result in dataframe shape of (1200, 12)
    cat_encoder :
        catalyst encoder
    an_encoder :
        encoder for anions

    Examples
    --------
    >>> from ai4water.datasets import mg_degradation
    >>> mg_data, catalyst_encoder, anion_encoder = mg_degradation()
    >>> mg_data.shape
    (1200, 12)
    ... # the default encoding is None, but if we want to use one hot encoder
    >>> mg_data_ohe, cat_enc, an_enc = mg_degradation(encoding="ohe")
    >>> mg_data_ohe.shape
    (1200, 31)
    >>> cat_enc.inverse_transform(mg_data_ohe.iloc[:, 9:24].values)
    >>> an_enc.inverse_transform(mg_data_ohe.iloc[:, 24:30].values)
    ... # if we want to use label encoder
    >>> mg_data_le, cat_enc, an_enc = mg_degradation(encoding="le")
    >>> mg_data_le.shape
    (1200, 12)
    >>> cat_enc.inverse_transform(mg_data_le.iloc[:, 9].values.astype(int))
    >>> an_enc.inverse_transform(mg_data_le.iloc[:, 10].values.astype(int))
    ... # By default the target is efficiency but if we want
    ... # to use first order k as target
    >>> mg_data_k, _, _ = mg_degradation(target="k_first")
    ... # if we want to use 2nd order k as target
    >>> mg_data_k2, _, _ = mg_degradation(target="k_2nd")

    """

    df = pd.read_csv(
        "https://raw.githubusercontent.com/ZeeshanHJ/Photocatalytic_Performance_Prediction/main/Raw%20data.csv"
    )
    default_inputs = ['Surface area', 'Pore Volume', 'Catalyst_loading (g/L)',
                      'Light_intensity (W)', 'time (min)', 'solution_pH', 'HA (mg/L)',
                      'Ci (mg/L)', 'Cf (mg/L)', 'Catalyst_type', 'Anions',
                      ]
    default_targets = ['Efficiency (%)', 'k_first', 'k_2nd']

    # first order
    df["k_first"] = np.log(df["Ci (mg/L)"] / df["Cf (mg/L)"]) / df["time (min)"]

    # k second order
    df["k_2nd"] = ((1 / df["Cf (mg/L)"]) - (1 / df["Ci (mg/L)"])) / df["time (min)"]

    if inputs is None:
        inputs = default_inputs

    if not isinstance(target, list):
        if isinstance(target, str):
            target = [target]
    elif isinstance(target, list):
        pass
    else:
        target = default_targets

    assert isinstance(target, list)

    assert all(trgt in default_targets for trgt in target)

    df = df[inputs + target]

    # consider encoding of categorical features
    cat_encoder, an_encoder = None, None
    if encoding:
        df, cols_added, cat_encoder = encode_column(df, "Catalyst_type", encoding)
        df, an_added, an_encoder = encode_column(df, "Anions", encoding)

        # move the target to the end
        for t in target:
            df[t] = df.pop(t)

    return df, cat_encoder, an_encoder


def dye_removal(
        inputs: List[str] = None,
        outputs: Union[str, List[str]] = "Efficiency (%)",
        encoding: str = None
):
    """
    `Kim et al., 2024 <https://doi.org/10.1016/j.jhazmat.2023.132995>`_ .

    Parameters
    ----------
    inputs : list, optional
        features to use as input. It must be a subset of the following features

            - ``Catalyst``
            - ``Hydrothermal synthesis time (min)``
            - ``Energy Band gap (Eg) eV', 'C (At%)', 'O (At%)', 'Fe (At%)', 'Al (At%)',
            - ``Ni (At%)', 'Mo (At%)', 'S (At%)', 'Bi', 'Ag', 'Pd', 'Pt',
            - ``Surface area (m2/g)', 'Pore volume (cm3/g)', 'Pore size (nm)',
            - ``volume (L)',
            - ``loading (g)``,
            - ``Catalyst_loading_mg``,
            - ``Light intensity (watt)``
            - ``Light source distance (cm)``
            - ``Time (m)``
            - ``Dye``
            - ``log_Kw``
            - ``hydrogen_bonding_acceptor_count``
            - ``hydrogen_bonding_donor_count``
            - ``solubility (g/L)``
            - ``molecular_wt (g/mol)``
            - ``pka1``
            - ``pka2``
            - ``Dye concentration (mg/L)``
            - ``Solution pH``
            - ``HA (mg/L)``
            - ``Anions``

    """

    url = "https://gitlab.com/atrcheema/bajwachor/-/blob/main/scripts/data/230613_Photocatalysis_with_Zeeshan_data_CMKim_Updated.csv"
    df = maybe_download_and_read_data(url, "dye_removal_data.csv")

    # first order k following https://doi.org/10.1016/j.seppur.2019.116195
    k = np.log(df["Ci"] / df["Cf"]) / df["Time (m)"]
    df["k"] = k

    k_2nd = ((1 / df["Cf"]) - (1 / df["Ci"])) / df["Time (m)"]
    df["k_2nd"] = k_2nd

    # at Time 0, let k==0
    df.loc[df['Time (m)'] <= 0.0, "k"] = 0.0

    # when final concentration is very low, k is not calculable (will be inf)
    # therefore inserting very small value of k
    df.loc[df['Cf'] == 0.0, "k"] = 0.001

    # mass_ratio = (loading / volume )/dye_conc.

    # when no anions are present, represent them as N/A
    df.loc[df['Anions'].isin(['0', 'without Anion']), "Anions"] = "N/A"

    if inputs is None:
        inputs = default_inputs

    if outputs is None:
        outputs = ['Efficiency']
    else:
        if not isinstance(outputs, list):
            outputs = [outputs]

    df = df[inputs + outputs]

    # consider encoding of categorical features
    df, encoders = encode_cols(df, ["Catalyst", "Dye", "Anions"], encoding)

    return df, encoders


def dichlorophenoxyacetic_acid_removal(
    parameters: Union[str, List[str]] = "all",
    encoding: str = None,
):
    """
    Data for photodegradation of 2,4-dichlorophenoxyacetic acid using gold-doped bismuth ferrite
    """
    return
