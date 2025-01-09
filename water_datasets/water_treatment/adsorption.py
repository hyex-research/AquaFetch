
__all__ = ["ec_removal_biochar", "po4_removal_biochar", "cr_removal", "heavy_metal_removal"]

from typing import Union, Tuple, Any, List, Dict

import pandas as pd

from ..utils import (
    check_attributes,
    LabelEncoder,
    OneHotEncoder,
    maybe_download_and_read_data,
    encode_cols
)


def ec_removal_biochar(
        parameters: Union[str, List[str]] = "all",
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
    #fpath = os.path.join(os.path.dirname(__file__), "data", 'qe_biochar_ec.csv')
    url = 'https://raw.githubusercontent.com/ZeeshanHJ/Adsorption-capacity-prediction-for-ECs/main/Raw_data.csv'

    data = maybe_download_and_read_data(url, 'ec_removal_biochar.csv')

    # remove space in 'Pyrolysis temperature '
    data['Pyrolysis temperature'] = data.pop('Pyrolysis temperature ')

    data['Adsorbent'] = data.pop('Adsorbent')
    data['Pollutant'] = data.pop('Pollutant')
    data['Wastewater type'] = data.pop('Wastewater type')
    data['Adsorption type'] = data.pop('Adsorption type')

    data['Capacity'] = data.pop('Capacity')

    def_paras = [
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
        'Capacity'
    ]

    parameters = check_attributes(parameters, def_paras, 'parameters')

    data, encoders = encode_cols(data, ['Adsorbent', 'Pollutant', 'Wastewater type', 'Adsorption type'], encoding)

    data = data[parameters]

    return data, encoders


def po4_removal_biochar(
        parameters:Union[str, List[str]] = "all",
        encoding:str = None,
)->Tuple[pd.DataFrame, Dict[str, Union[LabelEncoder, OneHotEncoder, Any]]]:
    """
    `Iftikhar et al., 2023 <https://doi.org/10.1016/j.chemosphere.2024.144031>`_
    """
    url = "https://github.com/Sara-Iftikhar/po4_removal_ml/raw/main/scripts/master_sheet_0802.xlsx"
    data = maybe_download_and_read_data(url, "po4_removal_biochar.csv")

    def_paras = ['Adsorbent', 'Feedstock', 'Pyrolysis_temp',
                              'Heating rate (oC)', 'Pyrolysis_time (min)',
                              'C', 'O', 'Surface area', 'Adsorption_time (min)',
                              'Ci_ppm', 'solution pH', 'rpm', 'Volume (L)',
                              'loading (g)', 'adsorption_temp',
                              'Ion Concentration (mM)', 'ion_type']

    parameters = check_attributes(parameters, def_paras, 'parameters')

    data, encoders = encode_cols(data, ['Adsorbent', 'Feedstock', 'ion_type'], encoding)

    data = data[parameters]

    return data, encoders


def cr_removal(
        parameters: Union[str, List[str]] = "all",
        encoding:str = None
):
    """
    `Ishtiaq et al., 2024 <https://doi.org/10.1016/j.jece.2024.112238>`_
    """

    url = "https://gitlab.com/atrcheema/envai103/-/raw/main/data/data.csv"
    data = maybe_download_and_read_data(url, "cr_removal.csv")

    cf = data['final conc.']
    ci = data['Initial conc.']
    v = data['Volume (L)']
    m = data['loading (g)']
    qe = ((ci - cf) * v) / m
    qe = qe.fillna(0.0)
    data['Adsorption capacity'] = qe

    data["Removal Efficiency"] = ((ci - cf) / ci) * 100

    def_paras = ['Adsorbent', 'NaOH conc. (M)', 'Surface area',
                      'Pore volume', 'C (At%)',
                  'Al (At%)', 'Nb (At%)', 'O (At%)', 'Na (At%)', 'Pore size ',
                  'Adsorption time', 'Initial conc.',
                  'loading (g)', 'Solution pH', 'Cycle number']

    parameters = check_attributes(parameters, def_paras, 'parameters')

    data, encoders = encode_cols(data, ['Adsorbent'], encoding)

    data = data[parameters]

    return data, encoders


def heavy_metal_removal(
        parameters: Union[str, List[str]] = "all",
        encoding: str = None
):
    """
    `Jaffari et al., 2024 <https://doi.org/10.1016/j.jhazmat.2023.132773>`_
    """

    url = "https://gitlab.com/atrcheema/envai103/-/raw/main/data/data.csv"
    data = maybe_download_and_read_data(url, "heavy_metal_removal.csv")

    # data.columns = ['Adsorbent', 'Feedstock', 'Pyrolysis_temp', 'Heating rate (oC)',
    #        'Pyrolysis_time (min)', 'C', 'H', 'O', 'N', 'Ash', 'H/C', 'O/C', 'N/C',
    #        '(O+N/C)', 'Surface area', 'Pore volume', 'Average pore size',
    #        'inorganics',
    #                 'radius (pm)', 'hydra_radius_pm', 'First_ionic_IE_KJ/mol',
    #                 'Adsorption_time (min)', 'Ci', 'solution pH', 'rpm',
    #        'Volume (L)', 'loading (g)', 'g/L', 'adsorption_temp',
    #        'Ion Concentration (M)', 'Anion_type', 'DOM', 'Cf', 'qe']

    def_paras = ['Adsorbent', 'NaOH conc. (M)', 'Surface area', 'Pore volume', 'C (At%)',
       'Al (At%)', 'Nb (At%)', 'O (At%)', 'Na (At%)', 'Pore size ',
       'Adsorption time', 'Initial conc.', 'loading (g/L)', 'Volume (L)',
       'loading (g)', 'Solution pH', 'Cycle number', 'final conc.']

    parameters = check_attributes(parameters, def_paras, 'parameters')

    data, encoders = encode_cols(data, ['Adsorbent', 'Feedstock', 'inorganics', 'Anion_type'], encoding)

    data = data[parameters]

    return data, encoders


def heavy_metal_removal_Shen(
        parameters: Union[str, List[str]] = "all",
        encoding: str = None
):
    """
    `Shen et al., 2024 <https://doi.org/10.1016/j.jhazmat.2024.133442>`_
    """

    url = "https://ars.els-cdn.com/content/image/1-s2.0-S0304389424000207-mmc2.xlsx"

    data = maybe_download_and_read_data(url, "heavy_metal_removal_Shen.csv")
    return data, {}


def industrial_dye_removal(
        parameters: Union[str, List[str]] = "all",
        encoding: str = None
):
    """
    `Iftikhar et al., 2023 <https://doi.org/10.1016/j.seppur.2023.124891>`_
    """
    url = "https://github.com/Sara-Iftikhar/ai4adsorption/raw/main/scripts/Dyes%20data.xlsx"
    data = maybe_download_and_read_data(url, "industrial_dye_removal.csv")

    def_paras = ['Adsorbent', 'Dye']

    parameters = check_attributes(parameters, def_paras, 'parameters')

    data, encoders = encode_cols(data, ['Adsorbent', 'Dye'], encoding)

    data = data[parameters]

    return data, encoders


def P_recovery(
        parameters: Union[str, List[str]] = "all",
        encoding: str = None
):
    """
    `Leng et al., 2024 <https://doi.org/10.1016/j.jwpe.2024.104896>`_
    """
    url = "https://zenodo.org/records/14586314/files/P_recovery.csv"
    data = maybe_download_and_read_data(url, "P_recovery.csv")

    def_paras = ['stir(rpm)', 't(min)', 'T(℃)', 'pH', 'N:P', 'Mg:P', 'P_initial(mg/L)',
       'P_recovery(%)']

    parameters = check_attributes(parameters, def_paras, 'parameters')

    data, encoders = encode_cols(data, [], encoding)

    data = data[parameters]

    return data, encoders


def N_recovery(
        parameters: Union[str, List[str]] = "all",
        encoding: str = None
):
    """
    `Leng et al., 2024 <https://doi.org/10.1016/j.jwpe.2024.104896>`_
    """
    url = "https://zenodo.org/records/14586314/files/N_recovery.csv"

    data = maybe_download_and_read_data(url, "N_recovery.csv")

    def_paras = ['stir(rpm)', 't(min)', 'T(℃)', 'pH', 'N:P', 'Mg: N', 'P_initial(mg/L)',
       'N_recovery(%)']

    parameters = check_attributes(parameters, def_paras, 'parameters')

    data, encoders = encode_cols(data, ['Adsorbent', 'Pollutant', 'Wastewater type', 'Adsorption type'], encoding)

    data = data[parameters]

    return data, encoders


def As_recovery(
        parameters: Union[str, List[str]] = "all",
        encoding: str = None
)->Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    `Huang et al., 2023 <https://doi.org/10.1016/j.watres.2024.122815>`_
    """
    url = "https://ars.els-cdn.com/content/image/1-s2.0-S0043135424017147-mmc2.xlsx"
    data = maybe_download_and_read_data(url, "As_recovery.csv")
    return data, {}
