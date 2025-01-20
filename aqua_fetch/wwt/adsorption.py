
__all__ = [
    "ec_removal_biochar", 
    "po4_removal_biochar", 
    "cr_removal", 
    "heavy_metal_removal",
    "heavy_metal_removal_Shen",
    "industrial_dye_removal",
    "P_recovery",
    "N_recovery",
    "As_recovery"
    ]

from typing import Union, Tuple, Any, List, Dict

import numpy as np
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
)->Tuple[pd.DataFrame, Dict[str, Union[OneHotEncoder, LabelEncoder, Any]]]:
    """
    Data of removal of emerging contaminants/pollutants from wastewater
    using biochar. The data consists of three types of features,
    1) adsorption experimental conditions, 2) elemental composition of
    adsorbent (biochar) and  3) parameters representing
    physical and synthesis conditions of biochar.
    For more description of this data see `Jaffari et al., 2023 <https://doi.org/10.1016/j.cej.2023.143073>`_

    Parameters
    ----------
    parameters :
        By default following features are used as input

            - ``adsorbent``
            - ``pyrolysis_temperature``
            - ``pyrolysis_time``
            - ``C``
            - ``H``
            - ``O``
            - ``N``
            - ``(O+N)/C``
            - ``ash``
            - ``H/C``
            - ``O/C``
            - ``N/C``
            - ``surface_area``
            - ``pore_volume``
            - ``average_pore_size``
            - ``pollutant``
            - ``adsorption_time``
            - ``concentration``
            - ``Solution_ph``
            - ``rpm``
            - ``volume``
            - ``adsorbent_dosage``
            - ``adsorption_temperature``
            - ``ion_concentration``
            - ``humid_acid``
            - ``wastewater_type``
            - ``adsorption_type``
            - ``final_concentration``
            - ``capacity``

    encoding : str, default=None
        the type of encoding to use for categorical features. If not None, it should
        be either ``ohe`` or ``le``.

    Returns
    --------
    tuple
        A tuple of length two. The first element is a DataFrame while the
        second element is a dictionary consisting of encoders with ``adsorbent``
        ``pollutant``, ``wastewater_type`` and ``adsorption_type`` as keys.

    Examples
    --------
    >>> from aqua_fetch import ec_removal_biochar
    >>> data, _ = ec_removal_biochar()
    >>> data.shape
    (3757, 29)
    >>> data, encoders = ec_removal_biochar(encoding="le")
    >>> data.shape
    (3757, 29)
    >>> len(set(encoders['adsorbent'].inverse_transform(data.loc[:, "adsorbent"])))
    15
    >>> len(set(encoders['pollutant'].inverse_transform(data.iloc[:, "Pollutant"])))
    14
    >>> set(encoders['wastewater_type'].inverse_transform(data.loc[:, "wastewater_type"]))
    {'Ground water', 'Lake water', 'Secondary effluent', 'Synthetic'}
    >>> set(encoders['adsorption_type'].inverse_transform(data.loc[:, "adsorption_type"]))
    {'Competative', 'Single'}

    We can also use one hot encoding to convert categorical features into
    numerical features. This will obviously increase the number of features/columns in DataFrame

    >>> data, encoders = ec_removal_biochar(encoding="ohe")
    >>> data.shape
    (3757, 60)
    >>> len(set(encoders['adsorbent'].inverse_transform(data.loc[:, [col for col in data.columns if col.startswith('adsorbent')]].values)))
    15
    >>> len(set(encoders['pollutant'].inverse_transform(data.loc[:, [col for col in data.columns if col.startswith('pollutant')]].values)))
    14
    >>> set(encoders['wastewater_type'].inverse_transform(data.loc[:, [col for col in data.columns if col.startswith('wastewater_type')]].values))
    {'Ground water', 'Lake water', 'Secondary effluent', 'Synthetic'}
    >>> set(encoders['adsorption_type'].inverse_transform(data.iloc[:, [col for col in data.columns if col.startswith('adsorption_type')]].values))
    {'Competative', 'Single'}

    """
    url = 'https://raw.githubusercontent.com/ZeeshanHJ/Adsorption-capacity-prediction-for-ECs/main/Raw_data.csv'

    data = maybe_download_and_read_data(url, 'ec_removal_biochar.csv')

    capitals = [        
        'C',
        'H',
        'O',
        'N',
        '(O+N)/C',
        'H/C',
        'O/C',
        'N/C']

    # remove trailing space, make everything lower and rplace space with _
    data.columns = [col.strip(' ').lower().replace(' ', '_') if col not in capitals else col for col in data.columns]

    def_paras = [
        'pyrolysis_temperature',
        'pyrolysis_time',
        'C',
        'H',
        'O',
        'N',
        '(O+N)/C',
        'ash',
        'H/C',
        'O/C',
        'N/C',
        'surface_area',
        'pore_volume',
        'average_pore_size',
        'adsorption_time',
        'initial_concentration',
        'solution_ph',
        'rpm',
        'volume',
        'adsorbent_dosage',
        'adsorption_temperature',
        'ion_concentration',
        'humic_acid',
        'adsorbent',
        'pollutant',
        'wastewater_type',
        'adsorption_type',
        'capacity',
        'final_concentration'
    ]

    parameters = check_attributes(parameters, def_paras, 'parameters')

    data = data[parameters]

    data, encoders = encode_cols(
        data,
  ['adsorbent', 'pollutant', 'wastewater_type', 'adsorption_type'],
        encoding)

    return data, encoders


def po4_removal_biochar(
        parameters:Union[str, List[str]] = "all",
        encoding:str = None,
)->Tuple[pd.DataFrame, Dict[str, Union[LabelEncoder, OneHotEncoder, Any]]]:
    """
    Data from adsorption experiments conducted for Cr removal from wastewater using biochar.
    For details on data see `Iftikhar et al., 2023 <https://doi.org/10.1016/j.chemosphere.2024.144031>`_

    Parameters
    ----------
    parameters :
        The parameters of the adsorption. It must be one of the following:

            - ``adsorbent``
            - ``feedstock``
            - ``activation``
            - ``pyrolysis_temp``
            - ``heating_rate``
            - ``pyrolysis_time``
            - ``C_%``
            - ``H_%``
            - ``O_%``
            - ``N_%``
            - ``S_%``
            - ``Ca_%``
            - ``ash``
            - ``H/C``
            - ``O/C``
            - ``N/C``
            - ``(O+N/C)``
            - ``surface_area``
            - ``pore_volume``
            - ``avg_pore_size``
            - ``adsorption_time_min``
            - ``Ci_ppm``
            - ``solution_pH``
            - ``rpm``
            - ``volume_l``
            - ``loading_g``
            - ``loading_g/L``
            - ``adsorption_temp``
            - ``ion_concentration_mM``
            - ``ion_type``
            - ``final_conf``
            - ``qe``
            - ``efficiency``

    encoding : str, default=None
        the type of encoding to use for categorical parameters. If not None, it should
        be either ``ohe`` or ``le``.

    """
    url = "https://github.com/Sara-Iftikhar/po4_removal_ml/raw/main/scripts/master_sheet_0802.xlsx"
    data = maybe_download_and_read_data(url, "po4_removal_biochar.csv")

    columns = {
        "Adsorbent": "adsorbent",
        "Feedstock": "feedstock",
        "Activation": "activation",
        "Pyrolysis_temp": "pyrolysis_temp",
    "Heating rate (oC)": "heating_rate",
    "Pyrolysis_time (min)": "pyrolysis_time",
    "C": "C_%",
    "H": "H_%",
    "O": "O_%",
    "N": "N_%",
    "S": "S_%",
    "Ca": "Ca_%",
    "Ash": "ash",
    "H/C": "H/C",
        "O/C": "O/C",
        "N/C": "N/C",
        "(O+N/C)": "(O+N/C)",
    "Surface area": "surface_area",
        "Pore volume": "pore_volume",
    "Average pore size": "avg_pore_size",
        "Adsorption_time (min)": "adsorption_time_min",
        "Ci_ppm": "Ci_ppm",
        "solution pH": "solution_pH",
    "rpm": "rpm",
    "Volume (L)":"volume_l",
    "loading (g)": "loading_g",
    "g/L": "loading_g/L",
    "adsorption_temp": "adsorption_temp",
    "Ion Concentration (mM)": "ion_concentration_mM",
    "ion_type": "ion_type",
    "Cf": "final_conf",
    "qe": "qe",
    "efficiency": "efficiency"
    }

    data.rename(columns=columns, inplace=True)

    data['feedstock'] = data['feedstock'].replace(np.nan, 'None')
    data['ion_type'] = data['ion_type'].replace(np.nan, 'None')

    parameters = check_attributes(parameters, list(columns.values()), 'parameters')

    data = data[parameters]

    data, encoders = encode_cols(data, ['adsorbent', 'feedstock', 'ion_type'], encoding)

    return data, encoders


def cr_removal(
        parameters: Union[str, List[str]] = "all",
        encoding:str = None
)->Tuple[pd.DataFrame, Dict[str, Union[LabelEncoder, OneHotEncoder, Any]]]:
    """
    Data from experiments conducted for Cr removal from wastewater using adsorption
    `Ishtiaq et al., 2024 <https://doi.org/10.1016/j.jece.2024.112238>`_

    Parameters
    ----------
    parameters :
        By default following parameters are used

            - ``adsorbent``
            - ``NaOH_conc_M``
            - ``surface_area``
            - ``pore_volume``
            - ``C_%``
            - ``Al_%``
            - ``Nb_%``
            - ``O_%``
            - ``Na_%``
            - ``pore_size``
            - ``adsorption_time``
            - ``initial_conc``
            - ``loading_g/L``
            - ``volume_l``
            - ``loading_g``
            - ``solution_ph``
            -  ``cycle_number``
            - ``final_conc``
            - ``adsorption_capacity``
            - ``removal_efficiency``

    encoding : str, default=None
        the type of encoding to use for categorical parameters. If not None, it should
        be either ``ohe`` or ``le``.

    Returns
    --------
    tuple
        A tuple of length two. The first element is a DataFrame while the
        second element is a dictionary consisting of encoder with ``adsorbent``
        as key.

    Examples
    --------
    >>> from aqua_fetch import cr_removal
    >>> data, _ = cr_removal()
    >>> data.shape
    (219, 20)
    >>> data, encoders = cr_removal(encoding="le")
    >>> data.shape
    (219, 20)
    >>> len(set(encoders['adsorbent'].inverse_transform(data.loc[:, "adsorbent"])))
    5
    >>> set(encoders['adsorbent'].inverse_transform(data.loc[:, "adsorbent"]))
    {'5M Nb2CTx', '20M Nb2CTx', '15M Nb2CTx', 'Nb2AlC', '10M Nb2CTx'}
    >>> data, encoders = cr_removal(encoding="ohe")
    >>> data.shape
    (219, 24)

    We can also use one hot encoding to convert categorical features into
    numerical features. This will obviously increase the number of features/columns in DataFrame

    >>> data, encoders = ec_removal_biochar(encoding="ohe")
    >>> data.shape
    (3757, 60)
    """

    url = "https://gitlab.com/atrcheema/envai103/-/raw/main/data/data.csv"
    data = maybe_download_and_read_data(url, "cr_removal.csv")

    columns = {"Adsorbent": "adsorbent",
                        "NaOH conc. (M)": "NaOH_conc_M",
                         "Surface area": "surface_area",
                         "Pore volume": "pore_volume",
                         "C (At%)": "C_%",
                "Al (At%)": "Al_%",
                         "Nb (At%)": "Nb_%",
                         "O (At%)": "O_%",
                         "Na (At%)": "Na_%",
                         "Pore size ": "pore_size",
                         "Adsorption time": "adsorption_time",
                         "Initial conc.": "initial_conc",
                         "loading (g/L)": "loading_g/L",
                         "Volume (L)": "volume_l",
                         "loading (g)": "loading_g",
                         "Solution pH": "solution_ph",
                         "Cycle number": "cycle_number",
                        "final conc.": "final_conc"}
    data.rename(columns=columns,
                inplace=True)

    cf = data['final_conc']
    ci = data['initial_conc']
    v = data['volume_l']
    m = data['loading_g']
    qe = ((ci - cf) * v) / m
    qe = qe.fillna(0.0)
    data['adsorption_capacity'] = qe

    data["removal_efficiency"] = ((ci - cf) / ci) * 100

    def_paras = list(columns.values()) + ['adsorption_capacity', 'removal_efficiency']

    parameters = check_attributes(parameters, def_paras, 'parameters')

    data = data[parameters]

    data, encoders = encode_cols(data, ['adsorbent'], encoding)

    return data, encoders


def heavy_metal_removal(
        parameters: Union[str, List[str]] = "all",
        encoding: str = None
)->Tuple[pd.DataFrame, Dict[str, Union[LabelEncoder, OneHotEncoder, Any]]]:
    """
    Data from experiments conducted for heavy metal removal from wastewater using adsorption.
    For more details on data see `Jaffari et al., 2024 <https://doi.org/10.1016/j.jhazmat.2023.132773>`_ .

    Parameters
    ----------
    parameters :
        By default following parameters are used

            - ``adsorbent``
            - ``NaOH_conc_M``
            - ``surface_area``
            - ``pore_volume``
            - ``C_%``
            - ``Al_%``
            - ``Nb_%``
            - ``O_%``
            - ``Na_%``
            - ``pore_size``
            - ``adsorption_time``
            - ``initial_conc``
            - ``loading_g/L``
            - ``volume_l``
            - ``loading_g``
            - ``solution_ph``
            - ``cycle_number``
            - ``final_conc``
        
    encoding : str, default=None
        the type of encoding to use for categorical parameters. If not None, it should
        be either ``ohe`` or ``le``.
    
    Returns
    --------
    tuple
        A tuple of length two. The first element is a DataFrame while the
        second element is a dictionary consisting of encoder with ``adsorbent``
        as key.
    
    Examples
    --------
    >>> from aqua_fetch import heavy_metal_removal
    >>> data, _ = heavy_metal_removal()
    >>> data.shape
    (219, 18)

    >>> data, encoders = heavy_metal_removal(encoding="le")
    >>> data.shape
    (219, 18)
    >>> len(set(encoders['adsorbent'].inverse_transform(data.loc[:, "adsorbent"])))
    5
    >>> set(encoders['adsorbent'].inverse_transform(data.loc[:, "adsorbent"]))
    {'5M Nb2CTx', '20M Nb2CTx', '15M Nb2CTx', 'Nb2AlC', '10M Nb2CTx'}
    >>> data, encoders = heavy_metal_removal(encoding="ohe")
    >>> data.shape
    (219, 22)
    >>> len(set(encoders['adsorbent'].inverse_transform(data.loc[:, [col for col in data.columns if col.startswith('adsorbent')]].values)))
    5
    """

    url = "https://gitlab.com/atrcheema/envai103/-/raw/main/data/data.csv"
    data = maybe_download_and_read_data(url, "heavy_metal_removal.csv")

    columns = {
        "Adsorbent": "adsorbent",
        "NaOH conc. (M)": "NaOH_conc_M",
        "Surface area": "surface_area",
        "Pore volume": "pore_volume",
        "C (At%)": "C_%",
        "Al (At%)": "Al_%",
        "Nb (At%)": "Nb_%",
        "O (At%)": "O_%",
        "Na (At%)": "Na_%",
        "Pore size ": "pore_size",
        "Adsorption time": "adsorption_time",
        "Initial conc.": "initial_conc",
        "loading (g/L)": "loading_g/L",
        "Volume (L)": "volume_l",
        "loading (g)": "loading_g",
        "Solution pH": "solution_ph",
        "Cycle number": "cycle_number",
        "final conc.": "final_conc"
    }

    data.rename(columns=columns, inplace=True)

    def_paras = list(columns.values())

    parameters = check_attributes(parameters, def_paras, 'parameters')

    data = data[parameters]

    data, encoders = encode_cols(data, ['adsorbent'], encoding)

    return data, encoders


def heavy_metal_removal_Shen(
        parameters: Union[str, List[str]] = "all",
        encoding: str = None
)->Tuple[pd.DataFrame, Dict[str, Union[LabelEncoder, OneHotEncoder, Any]]]:
    """
    Data from experiments conducted for heavy metal removal from wastewater using adsorption.
    For more details on data see `Shen et al., 2024 <https://doi.org/10.1016/j.jhazmat.2024.133442>`_

    Parameters
    ----------
        parameters :
            By default following parameters are used

                - ``heavy_metal``
                - ``hm_label``
                - ``ph_bichar``
                - ``C_%``
                - ``(O+N)/C``
                - ``O/C``
                - ``H/C``
                - ``ash``
                - ``PS``
                - ``SA``
                - ``CEC``
                - ``temperature``
                - ``solution_ph``
                - ``C0``
                - ``χ``
                - ``r``
                - ``Ncharge``
                - ``n``
            
        encoding : str, default=None
            the type of encoding to use for categorical parameters. If not None, it should
            be either ``ohe`` or ``le``.
        
    Returns
    --------
    tuple
        A tuple of length two. The first element is a DataFrame while the
        second element is a dictionary consisting of encoders with ``heavy_metal``
        and ``hm_label`` as keys.
    
    Examples
    --------
    >>> from aqua_fetch import heavy_metal_removal_Shen
    >>> data, _ = heavy_metal_removal_Shen()
    >>> data.shape
    (353, 18)
    >>> data, encoders = heavy_metal_removal_Shen(encoding="le")
    >>> data.shape
    (353, 18)
    >>> len(set(encoders['heavy_metal'].inverse_transform(data.loc[:, "heavy_metal"])))
    10
    >>> len(set(encoders['hm_label'].inverse_transform(data.loc[:, "hm_label"])))
    42
    >>> data, encoders = heavy_metal_removal_Shen(encoding="ohe")
    >>> data.shape
    (353, 68)
    >>> len(set(encoders['heavy_metal'].inverse_transform(data.loc[:, [col for col in data.columns if col.startswith('heavy_metal')]].values)))
    10
    >>> len(set(encoders['hm_label'].inverse_transform(data.loc[:, [col for col in data.columns if col.startswith('hm_label')]].values)))
    42
    """

    url = "https://ars.els-cdn.com/content/image/1-s2.0-S0304389424000207-mmc2.xlsx"

    data = maybe_download_and_read_data(url, "heavy_metal_removal_Shen.csv")

    columns = {
        "HM": "heavy_metal",
        "Label": "hm_label",
        "pH_biochar": "ph_bichar",
        "C": "C_%",
        "(O+N)/C": "(O+N)/C",
        "O/C": "O/C",
        "H/C": "H/C",
        "Ash": "ash",
        "PS": "PS",
        "SA": "SA",
        "CEC": "CEC",
        "T": "temperature",
        "pH_solution": "solution_ph",
        "C0": "C0",
        "χ": "χ",
        "r": "r",
        "Ncharge": "Ncharge",
        "η": "n"
    }

    data.rename(columns=columns, inplace=True)

    parameters = check_attributes(parameters, list(columns.values()), 'parameters')

    data = data[parameters]

    data, encoders = encode_cols(data, ['heavy_metal', 'hm_label'], encoding)

    return data, encoders


def industrial_dye_removal(
        parameters: Union[str, List[str]] = "all",
        encoding: str = None
)->Tuple[pd.DataFrame, Dict[str, Union[LabelEncoder, OneHotEncoder, Any]]]:
    """
    Data from experiments conducted for industrial dye removal from wastewater using adsorption.
    For more details on data see `Iftikhar et al., 2023 <https://doi.org/10.1016/j.seppur.2023.124891>`_ .

    Parameters
    ----------
        parameters : 
            By default following parameters are used

                - ``adsorbent``
                - ``calcination_temperature``
                - ``calcination_time_min``
                - ``C_%``
                - ``H_%``
                - ``O_%``
                - ``N_%``
                - ``ash``
                - ``H/C``
                - ``O/C``
                - ``N/C``
                - ``surface_area``
                - ``pore_volume``
                - ``average_pore_size``
                - ``dye``
                - ``adsorption_time_min``
                - ``initial_concentration``
                - ``solution_ph``
                - ``rpm``
                - ``volume_l``
                - ``loading_g/l``
                - ``adsorption_temperature``
                - ``ion_concentration_M``
                - ``humic_acid``
                - ``wastewater_type``
                - ``adsorption_type``
                - ``final_concentration``
                - ``qe``
                - ``adsorbent_loading``
            
        encoding : str, default=None
            the type of encoding to use for categorical parameters. If not None, it should
            be either ``ohe`` or ``le``.
        
    Returns
    --------
    tuple
        A tuple of length two. The first element is a DataFrame while the
        second element is a dictionary consisting of encoders with ``adsorbent``
        and ``dye`` as keys.
    
    Examples
    --------
    >>> from aqua_fetch import industrial_dye_removal
    >>> data, _ = industrial_dye_removal()
    >>> data.shape
    (680, 29)
    >>> data, encoders = industrial_dye_removal(encoding="le")
    >>> data.shape
    (680, 29)
    >>> len(set(encoders['adsorbent'].inverse_transform(data.loc[:, "adsorbent"])))
    7
    >>> len(set(encoders['dye'].inverse_transform(data.loc[:, "dye"])))
    4
    >>> data, encoders = industrial_dye_removal(encoding="ohe")
    >>> data.shape
    (680, 38)
    >>> len(set(encoders['adsorbent'].inverse_transform(data.loc[:, [col for col in data.columns if col.startswith('adsorbent')]].values)))
    7
    >>> len(set(encoders['dye'].inverse_transform(data.loc[:, [col for col in data.columns if col.startswith('dye')]].values)))
    4
    """
    url = "https://github.com/Sara-Iftikhar/ai4adsorption/raw/main/scripts/Dyes%20data.xlsx"
    data = maybe_download_and_read_data(url, "industrial_dye_removal.csv")

    columns = {
        "Adsorbent": "adsorbent",
        "calcination_temperature": "calcination_temperature",
        "calcination (min)": "calcination_time_min",
        "C": "C_%",
        "H": "H_%",
        "O": "O_%",
        "N": "N_%",
        "Ash": "ash",
        "H/C": "H/C",
        "O/C": "O/C",
        "N/C": "N/C",
        "Surface area": "surface_area",
        "Pore volume": "pore_volume",
        "Average pore size": "average_pore_size",
        "Dye": "dye",
        "Adsorption_time (min)": "adsorption_time_min",
        "initial concentration": "initial_concentration",
        "solution pH": "solution_ph",
        "rpm": "rpm",
        "Volume (L)": "volume_l",
        "g/L": "loading_g/l",
        "adsorption_temperature": "adsorption_temperature",
        "Ion Concentration (M)": "ion_concentration_M",
        "Humic acid": "humic_acid",
        "wastewater type": "wastewater_type",
        "Adsorption type": "adsorption_type",
        "Cf": "final_concentration",
        "qe": "qe",
        #"Ref": "ref",
        "adsorbent loading": "adsorbent_loading"
    }

    data.rename(columns=columns, inplace=True)

    parameters = check_attributes(parameters, list(columns.values()), 'parameters')

    data = data[parameters]

    data, encoders = encode_cols(data, ['adsorbent', 'dye'], encoding)

    return data, encoders


def P_recovery(
        parameters: Union[str, List[str]] = "all",
        encoding: str = None
):
    """
    Data from experiments conducted for P recovery from wastewater using adsorption.
    For more details on data see `Leng et al., 2024 <https://doi.org/10.1016/j.jwpe.2024.104896>`_ .

    Parameters
    ----------
    parameters :
        parameters to use as input. By default following parameters are used

            - ``stir(rpm)``
            - ``t(min)``
            - ``T(℃)``
            - ``pH``
            - ``N:P``
            - ``Mg:P``
            - ``P_initial(mg/L)``
            - ``P_recovery(%)``

    encoding : str, default=None
        the type of encoding to use for categorical parameters. If not None, it should
        be either ``ohe`` or ``le``.
    
    Returns
    --------
    tuple
        A tuple of length two. The first element is a DataFrame while the
        second element is an empty dictionary.
    
    Examples
    --------
    >>> from aqua_fetch import P_recovery
    >>> data, _ = P_recovery()
    >>> data.shape
    (504, 8)
    """
    url = "https://zenodo.org/records/14586314/files/P_recovery.csv"
    data = maybe_download_and_read_data(url, "P_recovery.csv")

    columns = {
        'stir(rpm)': 'stir_rpm',
        't(min)': 'time_min',
        'T(℃)': 'temperature_C',
        'pH': 'pH',
        'N:P': 'N:P',
        'Mg:P': 'Mg:P',
        'P_initial(mg/L)': 'P_initial_mgl',
        'P_recovery(%)': 'P_recovery_%'
    }

    data.rename(columns=columns, inplace=True)

    parameters = check_attributes(parameters, list(columns.values()), 'parameters')

    data = data[parameters]

    data, encoders = encode_cols(data, [], encoding)

    return data, encoders


def N_recovery(
        parameters: Union[str, List[str]] = "all",
        encoding: str = None
)->Tuple[pd.DataFrame, dict]:
    """
    Data from experiments conducted for N recovery from wastewater using adsorption.
    For more details on data see `Leng et al., 2024 <https://doi.org/10.1016/j.jwpe.2024.104896>`_ .

    Parameters
    ----------
    parameters :
        parameters to use as input. By default following parameters are used

            - ``stir(rpm)``
            - ``t(min)``
            - ``T(℃)``
            - ``pH``
            - ``N:P``
            - ``Mg: N``
            - ``P_initial(mg/L)``
            - ``N_recovery(%)``
        
    encoding : str, default=None
        the type of encoding to use for categorical parameters. If not None, it should
        be either ``ohe`` or ``le``.
    
    Returns
    --------
    tuple
        A tuple of length two. The first element is a DataFrame while the
        second element is an empty dictionary.
    
    Examples
    --------
    >>> from aqua_fetch import N_recovery
    >>> data, _ = N_recovery()
    >>> data.shape
    (210, 8)
    """
    url = "https://zenodo.org/records/14586314/files/N_recovery.csv"

    data = maybe_download_and_read_data(url, "N_recovery.csv")

    columns = {
        'stir(rpm)': 'stir_rpm',
        't(min)': 'time_min',
        'T(℃)': 'temperature_C',
        'pH': 'pH',
        'N:P': 'N:P',
        'Mg: N': 'Mg:N',
        'P_initial(mg/L)': 'P_initial_mgl',
        'N_recovery(%)': 'N_recovery_%'
    }

    data.rename(columns=columns, inplace=True)

    parameters = check_attributes(parameters, list(columns.values()), 'parameters')

    data = data[parameters]

    data, encoders = encode_cols(data, [], encoding)

    return data, encoders


def As_recovery(
        parameters: Union[str, List[str]] = "all",
        encoding: str = None
)->Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Data from experiments conducted for As recovery from wastewater using adsorption.
    For more details on data see `Huang et al., 2023 <https://doi.org/10.1016/j.watres.2024.122815>`_ .

    Parameters
    ----------
    parameters :
        parameters to use as input. By default following parameters are used

            - ``material``
            - ``biochar_modification``
            - ``biochar_type``
            - ``BET_surface_area``
            - ``pore_volume``
            - ``solution_pH``
            - ``reactor_temperature``
            - ``initial_As_concentration_mg_L``
            - ``adsorbent_dosage``
            - ``equilibrium_reaction_time_h``
            - ``pyrolysis_temperature``
            - ``As_mg_g``
            - ``As_type``
        
    encoding : str, default=None
        the type of encoding to use for categorical parameters. If not None, it should
        be either ``ohe`` or ``le``.
    
    Returns
    --------
    tuple
        A tuple of length two. The first element is a DataFrame while the
        second element is a dictionary consisting of encoders with ``material``,
        ``biochar_modification``, ``biochar_type`` and ``As_type`` as keys.
    
    Examples
    --------
    >>> from aqua_fetch import As_recovery
    ... # Using default parameters
    >>> data, _ = As_recovery()
    >>> data.shape
    (1605, 13)
    ... # Using label encoding
    >>> data, encoders = As_recovery(encoding="le")
    >>> data.shape
    (1605, 13)
    >>> len(set(encoders['material'].inverse_transform(data.loc[:, "material"])))
    72
    >>> len(set(encoders['biochar_modification'].inverse_transform(data.loc[:, "biochar_modification"])))
    2
    >>> len(set(encoders['biochar_type'].inverse_transform(data.loc[:, "biochar_type"])))
    159
    >>> len(set(encoders['As_type'].inverse_transform(data.loc[:, "As_type"])))
    2
    ... # Using one hot encoding
    >>> data, encoders = As_recovery(encoding="ohe")
    >>> data.shape
    (1605, 244)
    >>> len(set(encoders['material'].inverse_transform(data.loc[:, [col for col in data.columns if col.startswith('material')]].values)))
    72
    >>> len(set(encoders['biochar_modification'].inverse_transform(data.loc[:, [col for col in data.columns if col.startswith('biochar_modification')]].values)))
    2
    >>> len(set(encoders['biochar_type'].inverse_transform(data.loc[:, [col for col in data.columns if col.startswith('biochar_type')]].values)))
    159
    >>> len(set(encoders['As_type'].inverse_transform(data.loc[:, [col for col in data.columns if col.startswith('As_type')]].values)))
    2
    """
    url = "https://ars.els-cdn.com/content/image/1-s2.0-S0043135424017147-mmc2.xlsx"
    data = maybe_download_and_read_data(
        url,
        "As_recovery.csv",
    )

    columns = {
        'Materialstype': 'material',
        'Biochartype(Unmodifiedormodified)': 'biochar_modification',
        'Biochartype': 'biochar_type',
        'BETsurfacearea(m2/g)': 'BET_surface_area',
        'Porevolume(cm3/g)': 'pore_volume',
        'solutionpH(pHsol)': 'solution_pH',
        'Reactortemperature': 'reactor_temperature',
        'InitialAsconcentration(Total)mg/L': 'initial_As_concentration_mg_L',
        'Adsorbentdosage(g/L)': 'adsorbent_dosage',
        'Equilibrium/Reactiontime(h)': 'equilibrium_reaction_time_h',
        'Pyrolysistemperature(◦C)': 'pyrolysis_temperature',
        'As_mg/g': 'As_mg_g',
        'As_type': 'As_type'
    }

    data.rename(columns=columns, inplace=True)

    parameters = check_attributes(parameters, list(columns.values()), 'parameters')

    data = data[parameters]

    data, encoders = encode_cols(data, ['material', 'biochar_modification', 'biochar_type', 'As_type'], encoding)

    return data, encoders
