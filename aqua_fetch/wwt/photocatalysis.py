
__all__ = [
    "mg_degradation",
    "dye_removal",
    "dichlorophenoxyacetic_acid_removal",
    "pms_removal",
    "tetracycline_degradation",
    "tio2_degradation"
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

def mg_degradation(
        parameters: Union[str, List[str]] = "all",
        encoding: str = None
)->Tuple[pd.DataFrame, Dict[str, Union[OneHotEncoder, LabelEncoder, Any]]]:
    """
    This data is about photocatalytic degradation of melachite green dye using
    nobel metal dobe BiFeO3. For further description of this data see
    `Jafari et al., 2023 <https://doi.org/10.1016/j.jhazmat.2022.130031>`_ and
    for the use of this data for removal efficiency prediction `see <https://github.com/ZeeshanHJ/Photocatalytic_Performance_Prediction>`_ .
    This dataset consists of 1200 points collected during ~135 experiments.

    Parameters
    ----------
        parameters : list, optional
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
                - ``Efficiency (%)``
                - ``k_first``
                - ``k_2nd``

        encoding : str, default=None
            type of encoding to use for the two categorical features i.e., ``catalyst_type``
            and ``anions``, to convert them into numberical. Available options are ``ohe``,
            ``le`` and None. If ``ohe`` is selected the original input columns are replaced
            with ohe hot encoded columns. This will result in 6 columns for Anions and
            15 columns for catalyst_type.

    Returns
    --------
    tuple
        A tuple of length two. The first element is a DataFrame of shape (1200, len(parameters))
        while the second element is a dictionary consisting of encoders with
        ``catalyst_type`` and ``anions`` as keys.

    Examples
    --------
    >>> from aqua_fetch import mg_degradation
    >>> mg_data, encoders = mg_degradation()
    >>> mg_data.shape
    (1200, 14)
    ... # the default encoding is None, but if we want to use one hot encoder
    >>> mg_data_ohe, encoders = mg_degradation(encoding="ohe")
    >>> mg_data_ohe.shape
    (1200, 33)
    >>> encoders['catalyst_type'].inverse_transform(mg_data_ohe.loc[:, [col for col in data.columns if col.startswith('catalyst_type')]].values)
    >>> encoders['anions'].inverse_transform(mg_data_ohe.loc[:, [col for col in data.columns if col.startswith('anions')]].values)
    ... # if we want to use label encoder
    >>> mg_data_le, cat_enc, an_enc = mg_degradation(encoding="le")
    >>> mg_data_le.shape
    (1200, 14)
    >>> encoders['catalyst_type'].inverse_transform(mg_data_le.loc[:, 'catalyst_type'].values.astype(int))
    >>> encoders['anions'].inverse_transform(mg_data_le.loc[:, 'anions'].values.astype(int))
    ... # By default the target is efficiency but if we want
    ... # to use first order k as target
    >>> mg_data_k, _ = mg_degradation()
    ... # if we want to use 2nd order k as target
    >>> mg_data_k2, _ = mg_degradation()

    """

    url = "https://raw.githubusercontent.com/ZeeshanHJ/Photocatalytic_Performance_Prediction/main/Raw%20data.csv"
    data = maybe_download_and_read_data(url, "mg_degradation.csv")

    columns = {
        'Catalyst_type': 'catalyst_type',
        'Anions': 'anions',
        'Ci (mg/L)': 'ini_conc_mg/l',
        "Cf (mg/L)": 'final_conc_mg/l',
        "time (min)": 'time_min',
        'Catalyst_loading (g/L)': 'catalyst_loading_g/l',
        'Surface area': 'surface_area',
        'Pore Volume': 'pore_volume',
    }

    data.rename(columns=columns, inplace=True)

    # first order
    data["k_first"] = np.log(data['ini_conc_mg/l'] / data['final_conc_mg/l']) / data["time_min"]

    # k second order
    data["k_2nd"] = ((1 / data['final_conc_mg/l']) - (1 / data['ini_conc_mg/l'])) / data["time_min"]

    def_paras = ['surface_area', 'pore_volume', 'catalyst_loading_g/l',
                      'Light_intensity (W)', 'time_min', 'solution_pH', 'HA (mg/L)',
                      'ini_conc_mg/l', 'final_conc_mg/l', 'catalyst_type', 'anions',
                      ] + ['Efficiency (%)', 'k_first', 'k_2nd']

    parameters = check_attributes(parameters, def_paras, "parameters")

    data = data[parameters]

    # consider encoding of categorical features
    data, encoders = encode_cols(data, ['catalyst_type', 'anions'], encoding)

    return data, encoders


def dye_removal(
        parameters: Union[str, List[str]] = "all",
        encoding: str = None
)->Tuple[pd.DataFrame, Dict[str, Union[OneHotEncoder, LabelEncoder, Any]]]:
    """
    Data from experiments conducted to measure dye removal rate from wastewater
    treatment using photocatalysis method. For more information on data see
    `Kim et al., 2024 <https://doi.org/10.1016/j.jhazmat.2023.132995>`_ .

    Parameters
    ----------
    parameters : list, optional
        features to use as input. It must be a subset of the following features

            - ``catalyst``
            - ``hydrothermal_synthesis_time_min)``
            - ``energy_Band_gap_Eg) eV``
            - ``C_%``
            - ``O_%``
            - ``Fe_%``
            - ``Al_%``
            - ``Ni_%``
            - ``Mo_%``
            - ``S_%``
            - ``Bi``
            - ``Ag``
            - ``Pd``
            - ``Pt``
            - ``surface_area_m2/g``
            - ``pore_volume_cm3/g``
            - ``pore_size_nm``
            - ``volume_L``
            - ``loading_g``
            - ``catalyst_loading_mg``
            - ``light_intensity_watt``
            - ``light_source_distance_cm``
            - ``time_m``
            - ``dye``
            - ``log_Kw``
            - ``hydrogen_bonding_acceptor_count``
            - ``hydrogen_bonding_donor_count``
            - ``solubility_g/L``
            - ``molecular_wt_g/mol``
            - ``pka1``
            - ``pka2``
            - ``dye_concentration_mg/L``
            - ``solution_pH``
            - ``HA_mg/L``
            - ``anions``

        encoding : str, default=None
            type of encoding to use for the two categorical features i.e., ``Catalyst_type``
            ``dye`` and ``Anions``, to convert them into numberical. Available options are ``ohe``,
            ``le`` and None.

    Returns
    --------
    tuple
        A tuple of length two. The first element is a DataFrame of shape (1200, len(parameters))
        while the second element is a dictionary consisting of encoders with
        ``catalyst_type`` and ``anions`` as keys.

    Examples
    --------
    >>> from aqua_fetch import dye_removal

    >>> data, encoders = dye_removal()
    >>> assert data.shape == (1527, 36)
    # using label encoding to encode the categorical variables
    >>> data, encoders = dye_removal(encoding='le')
    >>> assert data.shape == (1527, 36), data.shape
    >>> catalysts = encoders['catalyst'].inverse_transform(data.loc[:, 'catalyst'].values)
    >>> len(set(catalysts.tolist()))
    18
    >>> dye = encoders['dye'].inverse_transform(data.loc[:, "dye"].values)
    >>> set(dye.tolist())
    {'Melachite Green', 'Indigo'}
    >>> anions = encoders['anions'].inverse_transform(data.loc[:,'anions'].values)
    >>> set(anions.tolist())
    {'NaCO3', 'N/A', 'Na2SO4', 'Na2HPO4', 'NaHCO3', 'NaCl'}
    # using one hot encoding for categroicla parameters
    >>> data, encoders = dye_removal(encoding='ohe')
    >>> assert data.shape == (1527, 59), data.shape
    >>> catalysts = encoders['catalyst'].inverse_transform(data.loc[:, [col for col in data.columns if col.startswith('catalyst')]].values)
    >>> len(set(catalysts.tolist()))
    18
    >>> dye = encoders['dye'].inverse_transform(data.loc[:, ["dye_0", "dye_1"]].values)
    >>> set(dye.tolist())
    {'Melachite Green', 'Indigo'}
    >>> anions = encoders['anions'].inverse_transform(data.loc[:, [col for col in data.columns if col.startswith('anions')]].values)
    >>> set(anions.tolist())
    {'NaCO3', 'N/A', 'Na2SO4', 'Na2HPO4', 'NaHCO3', 'NaCl'}

    """

    url = "https://gitlab.com/atrcheema/bajwachor/-/raw/main/scripts/data/230613_Photocatalysis_with_Zeeshan_data_CMKim_Updated.csv"
    df = maybe_download_and_read_data(url, "dye_removal.csv")

    columns = {
        'Catalyst': 'catalyst',
        'Hydrothermal synthesis time (min)': 'hydrothermal_synthesis_time_min',
    'Energy Band gap (Eg) eV': 'energy_band_gap_eV',
        'C (At%)': 'C_%',
        'O (At%)': 'O_%',
        'Fe (At%)': 'Fe_%',
        'Al (At%)': 'Al_%',
    'Ni (At%)': "Ni_%",
    'Mo (At%)': 'Mo_%',
        'S (At%)': 'S_%',
        'Bi': 'Bi', 'Ag': 'Ag', 'Pd': 'Pd', 'Pt': 'Pt',
    'Surface area (m2/g)': "surface_area_m2/g",
        'Pore volume (cm3/g)': 'pore_volume_cm3/g',
        'Pore size (nm)': 'pore_size_nm',
    'volume (L)': 'volume_l',
        # consider one of loading or catalysing loadnig
    'loading (g)': 'loading_g',  # 'Catalyst_loading_mg',
    'Light intensity (watt)': 'light_intensity_watt',
        'Light source distance (cm)': 'light_source_dist_cm',
        'Time (m)': 'time_m',
    'Dye': 'dye',
        # pollutant (dye) properties)
    'log_Kw': 'log_kw',
        'hydrogen_bonding_acceptor_count': 'hydrogen_bonding_accep_count',
        'hydrogen_bonding_donor_count': 'hydrogen_bonding_donor_count',
    'solubility (g/L)': 'solubility_g/l',
        'molecular_wt (g/mol)': 'molecular_wt_g/M',
        'pka1': 'pka1',
        'pka2': 'pka2',
        # instead of Ci we consider Dye Concentration
    'Dye concentration (mg/L)': 'dye_conc_mg/l',
    'Solution pH': 'solution_ph',  # 'Ci',
    'HA (mg/L)': 'ha_mg/l',
    'Anions': 'anions',
    }

    df.rename(columns=columns, inplace=True)

    # first order k following https://doi.org/10.1016/j.seppur.2019.116195
    k = np.log(df["Ci"] / df["Cf"]) / df["time_m"]
    df["k_1st"] = k

    k_2nd = ((1 / df["Cf"]) - (1 / df["Ci"])) / df["time_m"]
    df["k_2nd"] = k_2nd

    # at Time 0, let k==0
    df.loc[df['time_m'] <= 0.0, "k"] = 0.0

    # when final concentration is very low, k is not calculable (will be inf)
    # therefore inserting very small value of k
    df.loc[df['Cf'] == 0.0, "k"] = 0.001

    # mass_ratio = (loading / volume )/dye_conc.

    # when no anions are present, represent them as N/A
    df.loc[df['anions'].isin(['0', 'without Anion']), "anions"] = "N/A"

    default_paras = list(columns.values()) + ['k_1st', 'k_2nd']

    parameters = check_attributes(parameters, default_paras, 'parameters')

    df = df[parameters]

    # consider encoding of categorical features
    df, encoders = encode_cols(df, ["catalyst", "dye", "anions"], encoding)

    return df, encoders


def dichlorophenoxyacetic_acid_removal(
    parameters: Union[str, List[str]] = "all",
    encoding: str = None,
)->Tuple[pd.DataFrame, Dict[str, Union[OneHotEncoder, LabelEncoder, Any]]]:

    """
    Data for photodegradation of 2,4-dichlorophenoxyacetic acid using gold-doped bismuth ferrite

    Parameters
    ----------
    parameters : list, optional
        features to use as input. It must be a subset of the following features

            - ``catalyst``
            - ``surface_area``
            - ``pore_volume``
            - ``energy_band_gap_eV``
            - ``Au_%``
            - ``Bi_%``
            - ``Fe_%``
            - ``O_%``
            - ``catalyst_loading_g/l``
            - ``light_intensity_watt``
            - ``time_min
            - ``solution_ph``
            - ``anions``
            - ``ini_conc_mg/l``
            - ``final_conc_mg/l``
            - ``efficiency_%``

        encoding : str, default=None
            type of encoding to use for the two categorical features i.e., ``Catalyst_type``
            ``dye`` and ``Anions``, to convert them into numberical. Available options are ``ohe``,
            ``le`` and None.

    Returns
    --------
    tuple
        A tuple of length two. The first element is a DataFrame of shape (1200, len(parameters))
        while the second element is a dictionary consisting of encoders with
        ``catalyst_type`` and ``anions`` as keys.

    Examples
    --------
    >>> from aqua_fetch import dichlorophenoxyacetic_acid_removal
    ... # by default all parameters are returned
    >>> data, encoders = dichlorophenoxyacetic_acid_removal()
    >>> assert data.shape == (1044, 16), data.shape
    # using label encoding for categorical parameters
    >>> data, encoders = dichlorophenoxyacetic_acid_removal(encoding='le')
    >>> assert data.shape == (1044, 16), data.shape
    >>> catalysts = encoders['catalyst'].inverse_transform(data.loc[:, 'catalyst'].values)
    >>> assert len(set(catalysts.tolist())) == 7
    >>> anions = encoders['anions'].inverse_transform(data.loc[:,'anions'].values)
    >>> set(anions.tolist())
    {'Na2SO4', 'Without Anions', 'Na2HPO4', 'NaHCO3', 'NaCO3', 'NaCl'}
    # using one hot encoding for categorical parameters
    >>> data, encoders = dichlorophenoxyacetic_acid_removal(encoding='ohe')
    >>> assert data.shape == (1044, 27), data.shape
    >>> catalysts = encoders['catalyst'].inverse_transform(data.loc[:, ['catalyst_0', 'catalyst_1', 'catalyst_2',
       'catalyst_3', 'catalyst_4', 'catalyst_5', 'catalyst_6']].values)
    >>> assert len(set(catalysts.tolist())) == 7
    >>> anions = encoders['anions'].inverse_transform(data.loc[:, [col for col in data.columns if col.startswith('anions')]].values)
    >>> set(anions.tolist())
    {'Na2SO4', 'Without Anions', 'Na2HPO4', 'NaHCO3', 'NaCO3', 'NaCl'}

    """

    url = "https://gitlab.com/atrcheema/envai106/-/raw/main/data/data.xlsx"
    data = maybe_download_and_read_data(url, "dichlorophenoxyacetic_acid_removal.xlsx")

    columns = {
        'Catalyst type': 'catalyst',
        'Surface area': 'surface_area',
        'Pore volume': 'pore_volume',
        'BandGap (eV)': 'energy_band_gap_eV',
        'Au': 'Au_%',
        'Bi': 'Bi_%',
        'Fe': 'Fe_%',
        'O': 'O_%',
        'Catalyst loading (g/L)': 'catalyst_loading_g/l',
        'Light intensity (W)': 'light_intensity_watt',
        'time (min)': 'time_min',
        'solution pH': 'solution_ph',
        'Anions': 'anions',
        'Ci (mg/L)': 'ini_conc_mg/l',
        'Cf (mg/L)': 'final_conc_mg/l',
        'Efficiency (%)': 'efficiency_%',
    }

    data.rename(columns=columns, inplace=True)

    default_parameters = list(columns.values())

    parameters = check_attributes(parameters, default_parameters, 'parameters')

    data = data[parameters]

    data, encoders = encode_cols(data, ['catalyst', 'anions'], encoding)
    return data, encoders


def pms_removal(
    parameters: Union[str, List[str]] = "all",
    encoding: str = None,
)->Tuple[pd.DataFrame, Dict[str, Union[OneHotEncoder, LabelEncoder, Any]]]:

    """
    Data for photodegradation of phenol using peroxymonosulfate.

    Parameters
    ----------
        parameters : list, optional
            Names of the parameters to use. By default following parameters are used

                - ``time_min``
                - ``catalyst_type``
                - ``magnetization_Ms_emu/g``
                - ``energy_band_gap_eV``
                - ``calcination_temp_C``
                - ``min_calcination_time``
                - ``surface_area``
                - ``pore_size``
                - ``pollutant``
                - ``poll_mol_formula``
                - ``pms_concentration_g/l``
                - ``light_intensity_watt``
                - ``light_type``
                - ``catalyst_dosage_g/l``
                - ``ini_conc_ppm``
                - ``solution_ph``
                - ``H2O2_Conc_ppm``
                - ``volume_ml``
                - ``stirring_speed_rpm``
                - ``radical_scavenger``
                - ``inorganic anions``
                - ``water_type``
                - ``cycle_num``
                - ``final_conc_ppm``
                - ``removal_efficiency_%``
        encoding : str, default=None
            type of encoding to use for the two categorical features i.e., ``Catalyst_type``
            ``dye`` and ``Anions``, to convert them into numberical. Available options are ``ohe``,
            ``le`` and None.
        
    Returns
    --------
    tuple
        A tuple of length two. The first element is a DataFrame of shape (2078, len(parameters))
        while the second element is a dictionary consisting of encoders with
        ``catalyst_type``, ``pollutant``, ``poll_mol_formula`` and ``water_type`` as keys.
    
    Examples
    --------
    >>> from aqua_fetch import pms_removal
    >>> data, encoders = pms_removal()
    >>> data.shape
    (2078, 25)
    ... # the default encoding is None, but if we want to use one hot encoder
    >>> data_ohe, encoders = pms_removal(encoding="ohe")
    >>> data_ohe.shape
    (2078, 100)
    >>> catalysts = encoders['catalyst_type'].inverse_transform(data_ohe.loc[:, [col for col in data.columns if col.startswith('catalyst_type')]].values)
    >>> len(set(catalysts))
    42
    >>> pollutants = encoders['pollutant'].inverse_transform(data_ohe.loc[:, [col for col in data.columns if col.startswith('pollutant')]].values)
    >>> len(set(pollutants))
    14
    >>> poll_mol_formula = encoders['poll_mol_formula'].inverse_transform(data_ohe.loc[:, [col for col in data.columns if col.startswith('poll_mol_formula')]].values)
    >>> len(set(poll_mol_formula))
    14
    >>> water_type = encoders['water_type'].inverse_transform(data_ohe.loc[:, [col for col in data.columns if col.startswith('water_type')]].values)
    >>> len(set(water_type))
    9
    ... # if we want to use label encoder
    >>> data_le, encoders = pms_removal(encoding="le")
    >>> data_le.shape
    (2078, 25)
    >>> catalysts = encoders['catalyst_type'].inverse_transform(data_le.loc[:, 'catalyst_type'].values)
    >>> len(set(catalysts))
    42
    >>> pollutants = encoders['pollutant'].inverse_transform(data_le.loc[:, 'pollutant'].values)
    >>> len(set(pollutants))
    14
    >>> poll_mol_formula = encoders['poll_mol_formula'].inverse_transform(data_le.loc[:, 'poll_mol_formula'].values)
    >>> len(set(poll_mol_formula))
    14
    >>> water_type = encoders['water_type'].inverse_transform(data_le.loc[:, 'water_type'].values)
    >>> len(set(water_type))
    9
    """

    url = "https://gitlab.com/atrcheema/envai105/-/raw/main/data/Final_data_sheet_0716.xlsx"
    data = maybe_download_and_read_data(url, "pms_removal.xlsx")

    columns = {
        'time (min)': 'time_min',
        'Photocatalyst': 'catalyst_type',
        'Magnetization (Ms) (emu/g)': 'magnetization_Ms_emu/g',
        'band gap energy Eg (eV)': 'energy_band_gap_eV',
        'Calcination Temp. (oC)': 'calcination_temp_C',
        'Calcination Time (min)': 'min_calcination_time',
    'Surface area': 'surface_area',
        'Pore size': 'pore_size',
    'Pollutant': 'pollutant',
        'Pollutant molecular formula': 'poll_mol_formula',
    'PMS concentration (g/L)': 'pms_concentration_g/l',
        'Light intensity (W)': 'light_intensity_watt',
        'Light type': 'light_type',
        'Catalyst dosage (g/L)': 'catalyst_dosage_g/l',
        'Initial concentration (ppm)': 'ini_conc_ppm',
        'Solution pH': 'solution_ph',
        'H2O2 Concentration (mM)': 'H2O2_Conc_ppm',
        'Volume (mL)': 'volume_ml',
        'stirring speed (rpm)': 'stirring_speed_rpm',
        'Radical Scavenger': 'radical_scavenger',
        'Inorganic Anions': 'inorganic anions',
    'Type of the water': 'water_type',
        'No of Cycle': 'cycle_num',
        'Final Concentration (ppm)': 'final_conc_ppm',
        'Removal efficiency (%)': 'removal_efficiency_%'
    }
    data.rename(columns=columns, inplace=True)

    default_parameters = list(columns.values())

    parameters = check_attributes(parameters, default_parameters, 'parameters')

    data = data[parameters]

    data, encoders = encode_cols(
        data,
        ['catalyst_type', 'pollutant', 'poll_mol_formula', 'water_type'],
        encoding)
    return data, encoders


def tetracycline_degradation(
    parameters: Union[str, List[str]] = "all",
    encoding: str = None,
)->Tuple[pd.DataFrame, dict]:
    """
    Data for photodegradation of tetracycline. For details on data see
    `Abdi et al., 2022 <https://doi.org/10.1016/j.chemosphere.2021.132135>`_ .

    Parameters
    ----------
    parameters : list, optional
        Names of the parameters to use. By default, following parameters are used

            - ``surf_area_m2g``
            - ``pore_vol_cm3g``
            - ``catalyst_dosage_gL``
            - ``antibiotic_dosage_mgL``
            - ``illumination_time_min``
            - ``pH``
            - ``metallic_org_framework``
            - ``efficiency_%``

    encoding : str, default=None
            type of encoding to use for the categorical features. It can be either
            'ohe', 'le' or None. If 'ohe' is selected the original categroical column
            (``metallic_org_framework``) is replaced with one hot encoded columns.
            If 'le' is selected the original column is replaced with a label encoded column.
            If None is selected, the original column is not replaced.
        
    Returns
    --------
    tuple
        A tuple of length two. The first element is a DataFrame of shape (474, len(parameters))
        while the second element is a dictionary consisting of encoders with
        ``metallic_org_framework`` as key.
    
    Examples
    --------
    >>> from aqua_fetch import tetracycline_degradation
    >>> data, encoders = tetracycline_degradation()
    >>> data.shape
    (374, 8)

    >>> data, encoders = tetracycline_degradation(encoding='le')
    >>> data.shape
    (374, 8)
    >>> mofs = encoders['metallic_org_framework'].inverse_transform(data.loc[:, 'metallic_org_framework'].values)
    >>> len(set(mofs))
    10

    >>> data, encoders = tetracycline_degradation(encoding='ohe')
    >>> data.shape
    (374, 17)
    >>> mofs = encoders['metallic_org_framework'].inverse_transform(data.loc[:, [col for col in data.columns if col.startswith('metallic_org_framework')]].values)
    >>> len(set(mofs))
    10

    """

    url = "https://ars.els-cdn.com/content/image/1-s2.0-S0045653521026072-mmc1.zip"

    data = maybe_download_and_read_data(url, "tetracycline_degradation.csv")

    columns = {
        'Surface area (m2/g)': 'surf_area_m2g',
        'Pore Volume (cm3/g)': 'pore_vol_cm3g',
        'Catalyst dosage (g/L)': 'catalyst_dosage_gL',
        'Antibiotic dosage (mg/L)': 'antibiotic_dosage_mgL',
        'Illumination time (min)': 'illumination_time_min',
        'pH': 'pH',
        'Metallic organic framework': 'metallic_org_framework',
        'Degradation efficiency (%)': 'efficiency_%',
    }

    data.rename(columns=columns, inplace=True)

    parameters = check_attributes(parameters, list(columns.values()), 'parameters')

    data = data[parameters]

    data, encoders = encode_cols(data, ['metallic_org_framework'], encoding)

    return data, encoders


def tio2_degradation(
    parameters: Union[str, List[str]] = "all",
    encoding: str = None,
)->Tuple[pd.DataFrame, dict]:
    """
    Data for photodegradation of tio2

    For details on data see `Jiang et al., 2020 <https://doi.org/10.1016/j.envres.2020.109697>`_ .

    Parameters
    ----------
    parameters : list, optional
        Names of the parameters to use. By default following parameters are used

            - ``OC``
            - ``i_mWpercm2``
            - ``temp_C``
            - ``D_gl``
            - ``C0_mgl``
            - ``pH``
            - ``neglog_k_permin``

    encoding : str, default=None
        type of encoding to use for the categorical features.

    Returns
    --------
    tuple
        A tuple of length two. The first element is a DataFrame of shape (446, len(parameters))
        while the second element is an empty dictionary.
    
    Examples
    --------
    >>> from aqua_fetch import tio2_degradation
    >>> data, encoders = tio2_degradation()
    >>> data.shape
    (446, 7)
    """

    url = "https://ars.els-cdn.com/content/image/1-s2.0-S0045653521026072-mmc1.zip"

    data = maybe_download_and_read_data(url, "tio2_degradation.csv")

    columns = {
        'OC': 'OC',
        'I_mW/cm2': 'i_mWpercm2',
        'T_C': 'temp_C',
        'D_gl': 'D_gl',
        'C0_mgl': 'C0_mgl',
        'pH': 'pH',
        'neglog_k_permin': 'neglog_k_permin',
    }

    data.rename(columns=columns, inplace=True)

    parameters = check_attributes(parameters, list(columns.values()), 'parameters')

    data = data[parameters]

    data, encoders = encode_cols(data, [], encoding)

    return data, encoders


def photodegradation_Jiang(
        parameters: Union[str, List[str]] = "all",
        encoding: str = None,
)->Tuple[pd.DataFrame, Dict[str, Union[OneHotEncoder, LabelEncoder, Any]]]:
    """
    Data for photodegradation of multiple pollutants using various photocatalysts.
    For details on data see `Jiang et al., 2021 <https://doi.org/10.3390/catal11091107>`_ .

    Parameters
    ----------
        parameters : list, optional
            Names of the parameters to use. By default following parameters are used

                - ``photocatalyst``
                - ``contaminants``
                - ``photocat_dosage_gl``
                - ``photocat_size_nm``
                - ``initial_conc_mgl``
                - ``pH``
                - ``light_type``
                - ``k_min-1``
        
        encoding : str, default=None
            type of encoding to use for the categorical features.
            It can be either ``ohe``, ``le`` or None. If ``ohe`` is selected the original categroical column
            is replaced with one hot encoded columns. If ``le`` is selected the original column is replaced
            with a label encoded column. If None is selected, the original column is not replaced.
    
    Returns
    --------
    tuple
        A tuple of length two. The first element is a DataFrame of shape (446, len(parameters))
        while the second element is a dictionary consisting of encoders with
        ``photocatalyst`` and ``contaminants`` as keys.
    
    Examples
    --------
    >>> from aqua_fetch import photodegradation_Jiang
    >>> data, encoders = photodegradation_Jiang()
    >>> data.shape
    (449, 8)
    ... # the default encoding is None, but if we want to use one hot encoder
    >>> data_ohe, encoders = photodegradation_Jiang(encoding="ohe")
    >>> data_ohe.shape
    (449, 16)
    >>> photocatalysts = encoders['photocatalyst'].inverse_transform(data_ohe.loc[:, [col for col in data.columns if col.startswith('photocatalyst')]].values)
    >>> len(set(photocatalysts))
    100
    >>> contaminants = encoders['contaminants'].inverse_transform(data_ohe.loc[:, [col for col in data.columns if col.startswith('contaminants')]].values)
    >>> len(set(contaminants))
    47
    ... # if we want to use label encoder
    >>> data_le, encoders = photodegradation_Jiang(encoding="le")
    >>> data_le.shape
    (449, 8)
    >>> photocatalysts = encoders['photocatalyst'].inverse_transform(data_le.loc[:, 'photocatalyst'].values)
    >>> len(set(photocatalysts))
    100
    >>> contaminants = encoders['contaminants'].inverse_transform(data_le.loc[:, 'contaminants'].values)
    >>> len(set(contaminants))
    47    
    """

    url = "https://www.mdpi.com/2073-4344/11/9/1107#app1-catalysts-11-01107"

    data = maybe_download_and_read_data(url, "photodegradation_Jiang.csv")

    # replace "N/A" with np.nan in photocat_size_nm column
    data.loc[data["Photocat. size (nm)"] == "N/A", "Photocat. size (nm)"] = np.nan
    data.loc[data["Photocat. size (nm)"] == "N/A ", "Photocat. size (nm)"] = np.nan
    # convert '100-4000 ' to np.nan
    data.loc[data["Photocat. size (nm)"] == '100-4000 ', "Photocat. size (nm)"] = np.nan
    # convert '~200 ' to 200
    data.loc[data["Photocat. size (nm)"] == '~200 ', "Photocat. size (nm)"] = 200
    # convert '>1000 ' to np.nan
    data.loc[data["Photocat. size (nm)"] == '>1000 ', "Photocat. size (nm)"] = np.nan
    # convert '20-50 ' to np.nan
    data.loc[data["Photocat. size (nm)"] == '20-50 ', "Photocat. size (nm)"] = np.nan
    # convert '<44000 ' to np.nan
    data.loc[data["Photocat. size (nm)"] == '<44000 ', "Photocat. size (nm)"] = np.nan
    data.loc[data["pH"] == "N/A", "pH"] = np.nan
    data.loc[data["pH"] == "N/A ", "pH"] = np.nan
    # convert "Photocat. size (nm)" and 'pH' to float
    #data["Photocat. size (nm)"] = data["Photocat. size (nm)"].astype(np.float32)
    data['pH'] = data['pH'].astype(np.float32)

    # remove trailing empty space in contaiminant column
    data["Contaminants"] = data["Contaminants"].str.strip()

    columns = {
        "Photocatalyst": "photocatalyst",
        "Contaminants": "contaminants",
        "Photocat. dosage (g/L)": "photocat_dosage_gl",
        "Photocat. size (nm)": "photocat_size_nm",
        "Initial conc. (mg/L)": "initial_conc_mgl",
        "pH": "pH",
        "Light type": "light_type",
        "k (min-1)": "k_min-1"
    }

    data.rename(columns=columns, inplace=True)

    parameters = check_attributes(parameters, list(columns.values()), 'parameters')

    data = data[parameters]


    data, encoders = encode_cols(data, ['photocatalyst', 
                                        'contaminants'
                                        ], encoding)

    return data, encoders
