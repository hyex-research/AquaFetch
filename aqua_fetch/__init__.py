
import os
from typing import Union

import pandas as pd

from .rr import _RainfallRunoff
from .rr import CAMELS_AUS
from .rr import CAMELS_CL
from .rr import CAMELS_BR
from .rr import CAMELS_GB
from .rr import CAMELS_US
from .rr import LamaHCE
from .rr import HYSETS
from .rr import HYPE
from .rr import WaterBenchIowa
from .rr import CAMELS_DK
from .rr import GSHA
from .rr import CCAM
from .rr import RRLuleaSweden
from .rr import CABra
from .rr import CAMELS_CH
from .rr import LamaHIce
from .rr import CAMELS_DE
from .rr import GRDCCaravan
from .rr import CAMELS_SE
from .rr import Simbi
from .rr import Bull
from .rr import CAMELS_IND
from .rr import RainfallRunoff
from .rr import Arcticnet
from .rr import USGS
from .rr import EStreams
from .rr import Japan
from .rr import Thailand
from .rr import Spain
from .rr import Ireland
from .rr import Finland
from .rr import Poland
from .rr import Italy
from .rr import CAMELS_FR
from .rr import Portugal
from .rr import Caravan_DK
from .rr import CAMELS_NZ
from .rr import CAMELS_LUX
from .rr import CAMELS_COL
from .rr import CAMELS_SK
from .rr import CAMELS_FI
from .rr import CAMELS_PL
from .rr import CAMELS_PE
from .rr import Slovenia
from .rr import UKFlow15
from .rr import CAMELSH
from .rr import ShyftNorway
from .rr import NamalValleyPakistan

from .rr import MtropicsLaos
from .rr import MtropcsThailand
from .rr import MtropicsVietnam
from .rr import NPCTRCatchments


# *** Waste Water Treatment ***
from .wwt import ec_removal_biochar
from .wwt import cr_removal
from .wwt import po4_removal_biochar
from .wwt import heavy_metal_removal
from .wwt import industrial_dye_removal
from .wwt import heavy_metal_removal_Shen
from .wwt import P_recovery
from .wwt import N_recovery
from .wwt import As_recovery

from .wwt import mg_degradation
from .wwt import dye_removal
from .wwt import dichlorophenoxyacetic_acid_removal
from .wwt import pms_removal
from .wwt import tetracycline_degradation
from .wwt import tio2_degradation
from .wwt import photodegradation_Jiang

from .wwt import micropollutant_removal_osmosis
from .wwt import ion_transport_via_reverse_osmosis

from .wwt import cyanobacteria_disinfection


# *** Water Quality ***
from .wq import Quadica
from .wq import GRQA
from .wq import SWatCh
from .wq import RC4USCoast
from .wq import DoceRiver
from .wq import SeluneRiver
from .wq import busan_beach
from .wq import SyltRoads
from .wq import ecoli_mekong_laos
from .wq import ecoli_houay_pano
from .wq import ecoli_mekong_2016
from .wq import ecoli_mekong
from .wq import CamelsChem
from .wq import SanFranciscoBay
from .wq import GRiMeDB
from .wq import BuzzardsBay
from .wq import WhiteClayCreek
from .wq import RiverChemSiberia
from .wq import CamelsCHChem
from .wq import Oligotrend
from .wq import CaravanQual
from .wq import GlobalRiverNutrients

# *** Miscellaneous ***

from .misc import Weisssee
from .misc import WaterChemEcuador
from .misc import WaterChemVictoriaLakes
from .misc import WeatherJena
from .misc import WQCantareira
from .misc import WQJordan
from .misc import FlowSamoylov
from .misc import FlowSedDenmark
from .misc import StreamTempSpain
from .misc import RiverTempEroo
from .misc import HoloceneTemp
from .misc import FlowTetRiver
from .misc import SedimentAmersee
from .misc import HydrocarbonsGabes
from .misc import HydroChemJava
from .misc import PrecipBerlin
from .misc import GeoChemMatane
from .misc import WQJordan2
from .misc import YamaguchiClimateJp
from .misc import FlowBenin
from .misc import HydrometricParana
from .misc import RiverTempSpain
from .misc import RiverIsotope
from .misc import EtpPcpSamoylov
from .misc import SWECanada
from .misc import gw_punjab
from .misc import RRAlpineCatchments
from .misc import SoilPhosphorus


ALL_DATASETS = [
    CAMELS_AUS.__name__,
    CAMELS_BR.__name__,
    CAMELS_CL.__name__,
    CAMELS_GB.__name__,
    CAMELS_US.__name__,
    CAMELS_DK.__name__,
    CAMELS_CH.__name__,
    CAMELS_DE.__name__,
    CAMELS_FR.__name__,
    CAMELS_IND.__name__,
    CAMELS_SE.__name__,
    GSHA.__name__,
    CCAM.__name__,
    RRLuleaSweden.__name__,
    CABra.__name__,
    LamaHIce.__name__,
    LamaHCE.__name__,
    HYSETS.__name__,
    HYPE.__name__,
    WaterBenchIowa.__name__,
    Simbi.__name__,
    Bull.__name__,
    RainfallRunoff.__name__,
    Arcticnet.__name__,
    USGS.__name__,
    EStreams.__name__,
    Japan.__name__,
    Thailand.__name__,
    Spain.__name__,
    Ireland.__name__,
    Finland.__name__,
    Poland.__name__,
    Italy.__name__,
    Portugal.__name__,
    Caravan_DK.__name__,
    MtropicsLaos.__name__,
    MtropcsThailand.__name__,
    MtropicsVietnam.__name__,
    NPCTRCatchments.__name__,
    GRDCCaravan.__name__,
    CAMELS_NZ.__name__,
    CAMELS_LUX.__name__,
    CAMELS_COL.__name__,
    CAMELS_SK.__name__,
    CAMELS_FI.__name__,
    CAMELS_PL.__name__,
    CAMELS_PE.__name__,
    Slovenia.__name__,
    UKFlow15.__name__,
    CAMELSH.__name__,
    NamalValleyPakistan.__name__,

    Quadica.__name__,
    GRQA.__name__,
    SWatCh.__name__,
    RC4USCoast.__name__,
    DoceRiver.__name__,
    SeluneRiver.__name__,
    busan_beach.__name__,
    SyltRoads.__name__,
    ecoli_mekong_laos.__name__,
    ecoli_houay_pano.__name__,
    ecoli_mekong_2016.__name__,
    ecoli_mekong.__name__,
    CamelsChem.__name__,
    SanFranciscoBay.__name__,
    GRiMeDB.__name__,
    BuzzardsBay.__name__,
    WhiteClayCreek.__name__,
    RiverChemSiberia.__name__,
    CamelsCHChem.__name__,
    Oligotrend.__name__,
    CaravanQual.__name__,
    GlobalRiverNutrients.__name__,

    ec_removal_biochar.__name__,
    cr_removal.__name__,
    po4_removal_biochar.__name__,
    heavy_metal_removal.__name__,
    industrial_dye_removal.__name__,
    heavy_metal_removal_Shen.__name__,
    P_recovery.__name__,
    N_recovery.__name__,
    As_recovery.__name__,
    mg_degradation.__name__,
    dye_removal.__name__,
    dichlorophenoxyacetic_acid_removal.__name__,
    pms_removal.__name__,
    tetracycline_degradation.__name__,
    tio2_degradation.__name__,
    photodegradation_Jiang.__name__,
    micropollutant_removal_osmosis.__name__,
    ion_transport_via_reverse_osmosis.__name__,
    cyanobacteria_disinfection.__name__,

    Weisssee.__name__,
    WaterChemEcuador.__name__,
    WaterChemVictoriaLakes.__name__,
    WeatherJena.__name__,
    WQCantareira.__name__,
    WQJordan.__name__,
    FlowSamoylov.__name__,
    FlowSedDenmark.__name__,
    StreamTempSpain.__name__,
    RiverTempEroo.__name__,
    HoloceneTemp.__name__,
    FlowTetRiver.__name__,
    SedimentAmersee.__name__,
    HydrocarbonsGabes.__name__,
    HydroChemJava.__name__,
    PrecipBerlin.__name__,
    GeoChemMatane.__name__,
    WQJordan2.__name__,
    YamaguchiClimateJp.__name__,
    FlowBenin.__name__,
    HydrometricParana.__name__,
    RiverTempSpain.__name__,
    RiverIsotope.__name__,
    EtpPcpSamoylov.__name__,
    SWECanada.__name__,
    gw_punjab.__name__,
    RRAlpineCatchments.__name__,
    SoilPhosphorus.__name__
]




def load_nasdaq(inputs: Union[str, list, None] = None, target: str = 'NDX', verbosity: int = 1):
    """Loads Nasdaq100 by downloading it if it is not already downloaded."""

    DeprecationWarning("load_nasdaq is deprecated and will be removed in future versions."
                       "See aqua_fetch to get an appropriate dataset")

    fname = os.path.join(os.path.dirname(__file__), "data", "nasdaq100_padding.csv")

    if not os.path.exists(fname):
        if verbosity:
            print(f"downloading file to {fname}")
        df = pd.read_csv("https://raw.githubusercontent.com/KurochkinAlexey/DA-RNN/master/nasdaq100_padding.csv")
        df.to_csv(fname)

    df = pd.read_csv(fname)
    in_cols = list(df.columns)
    in_cols.remove(target)
    if inputs is None:
        inputs = in_cols
    target = [target]

    return df[inputs + target]


__version__ = "1.0.2"
