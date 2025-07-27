
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
from .rr import Slovenia

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
from .wq import Camels_Ch_Chem

# *** Miscellaneous ***

from ._datasets import Weisssee
from ._datasets import WaterChemEcuador
from ._datasets import WaterChemVictoriaLakes
from ._datasets import WeatherJena
from ._datasets import WQCantareira
from ._datasets import WQJordan
from ._datasets import FlowSamoylov
from ._datasets import FlowSedDenmark
from ._datasets import StreamTempSpain
from ._datasets import RiverTempEroo
from ._datasets import HoloceneTemp
from ._datasets import FlowTetRiver
from ._datasets import SedimentAmersee
from ._datasets import HydrocarbonsGabes
from ._datasets import HydroChemJava
from ._datasets import PrecipBerlin
from ._datasets import GeoChemMatane
from ._datasets import WQJordan2
from ._datasets import YamaguchiClimateJp
from ._datasets import FlowBenin
from ._datasets import HydrometricParana
from ._datasets import RiverTempSpain
from ._datasets import RiverIsotope
from ._datasets import EtpPcpSamoylov
from ._datasets import SWECanada
from ._datasets import gw_punjab
from ._datasets import RRAlpineCatchments
from ._hyperspectral import SoilPhosphorus


ALL_DATASETS = [
    CAMELS_AUS.__class__.__name__,
    CAMELS_BR.__class__.__name__,
    CAMELS_CL.__class__.__name__,
    CAMELS_GB.__class__.__name__,
    CAMELS_US.__class__.__name__,
    CAMELS_DK.__class__.__name__,
    CAMELS_CH.__class__.__name__,
    CAMELS_DE.__class__.__name__,
    CAMELS_FR.__class__.__name__,
    CAMELS_IND.__class__.__name__,
    CAMELS_SE.__class__.__name__,
    GSHA.__class__.__name__,
    CCAM.__class__.__name__,
    RRLuleaSweden.__class__.__name__,
    CABra.__class__.__name__,
    LamaHIce.__class__.__name__,
    LamaHCE.__class__.__name__,
    HYSETS.__class__.__name__,
    HYPE.__class__.__name__,
    WaterBenchIowa.__class__.__name__,
    Simbi.__class__.__name__,
    Bull.__class__.__name__,
    RainfallRunoff.__class__.__name__,
    Arcticnet.__class__.__name__,
    USGS.__class__.__name__,
    EStreams.__class__.__name__,
    Japan.__class__.__name__,
    Thailand.__class__.__name__,
    Spain.__class__.__name__,
    Ireland.__class__.__name__,
    Finland.__class__.__name__,
    Poland.__class__.__name__,
    Italy.__class__.__name__,
    Portugal.__class__.__name__,
    Caravan_DK.__class__.__name__,
    MtropicsLaos.__class__.__name__,
    MtropcsThailand.__class__.__name__,
    MtropicsVietnam.__class__.__name__,
    NPCTRCatchments.__class__.__name__,
    GRDCCaravan.__class__.__name__,
    CAMELS_NZ.__class__.__name__,
    CAMELS_LUX.__class__.__name__,
    CAMELS_COL.__class__.__name__,
    CAMELS_SK.__class__.__name__,
    CAMELS_FI.__class__.__name__,
    Slovenia.__class__.__name__,

    Quadica.__class__.__name__,
    GRQA.__class__.__name__,
    SWatCh.__class__.__name__,
    RC4USCoast.__class__.__name__,
    DoceRiver.__class__.__name__,
    SeluneRiver.__class__.__name__,
    busan_beach.__name__,
    SyltRoads.__class__.__name__,
    ecoli_mekong_laos.__name__,
    ecoli_houay_pano.__name__,
    ecoli_mekong_2016.__name__,
    ecoli_mekong.__name__,
    CamelsChem.__class__.__name__,
    SanFranciscoBay.__class__.__name__,
    GRiMeDB.__class__.__name__,
    BuzzardsBay.__class__.__name__,
    WhiteClayCreek.__class__.__name__,
    RiverChemSiberia.__class__.__name__,
    Camels_Ch_Chem.__class__.__name__,
    
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

    Weisssee.__class__.__name__,
    WaterChemEcuador.__class__.__name__,
    WaterChemVictoriaLakes.__class__.__name__,
    WeatherJena.__class__.__name__,
    WQCantareira.__class__.__name__,
    WQJordan.__class__.__name__,
    FlowSamoylov.__class__.__name__,
    FlowSedDenmark.__class__.__name__,
    StreamTempSpain.__class__.__name__,
    RiverTempEroo.__class__.__name__,
    HoloceneTemp.__class__.__name__,
    FlowTetRiver.__class__.__name__,
    SedimentAmersee.__class__.__name__,
    HydrocarbonsGabes.__class__.__name__,
    HydroChemJava.__class__.__name__,
    PrecipBerlin.__class__.__name__,
    GeoChemMatane.__class__.__name__,
    WQJordan2.__class__.__name__,
    YamaguchiClimateJp.__class__.__name__,
    FlowBenin.__class__.__name__,
    HydrometricParana.__class__.__name__,
    RiverTempSpain.__class__.__name__,
    RiverIsotope.__class__.__name__,
    EtpPcpSamoylov.__class__.__name__,
    SWECanada.__class__.__name__,
    gw_punjab.__name__,
    RRAlpineCatchments.__class__.__name__,
    SoilPhosphorus.__class__.__name__
]




def load_nasdaq(inputs: Union[str, list, None] = None, target: str = 'NDX'):
    """Loads Nasdaq100 by downloading it if it is not already downloaded."""

    DeprecationWarning("load_nasdaq is deprecated and will be removed in future versions."
                       "See aqua_fetch to get an appropriate dataset")

    fname = os.path.join(os.path.dirname(__file__), "data", "nasdaq100_padding.csv")

    if not os.path.exists(fname):
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


__version__ = "1.0.0rc3"
