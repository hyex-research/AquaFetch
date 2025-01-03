
import os
from typing import Union

import pandas as pd

from .rr import Camels
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


from .water_treatment import ec_removal_biochar
from .water_treatment import cr_removal
from .water_treatment import po4_removal_biochar
from .water_treatment import heavy_metal_removal
from .water_treatment import industrial_dye_removal
from .water_treatment import heavy_metal_removal_Shen
from .water_treatment import P_recovery
from .water_treatment import N_recovery

from .water_treatment import mg_degradation
from .water_treatment import dye_removal
from .water_treatment import dichlorophenoxyacetic_acid_removal

from .water_treatment import micropollutant_removal_osmosis
from .water_treatment import ion_transport_via_reverse_osmosis

from .water_treatment import cyanobacteria_disinfection

from .mtropics import MtropicsLaos
from .mtropics import MtropcsThailand
from .mtropics import MtropicsVietnam


from ._grimedb import GRiMeDB
from ._npctr import NPCTRCatchments
from ._hyperspectral import SoilPhosphorus


from .water_quality import Quadica
from .water_quality import GRQA
from .water_quality import Swatch
from .water_quality import RC4USCoast
from .water_quality import DoceRiver
from .water_quality import SeluneRiver
from .water_quality import busan_beach
from .water_quality import RiverChemSiberia
from .water_quality import SyltRoads
from .water_quality import ecoli_mekong_laos
from .water_quality import ecoli_houay_pano
from .water_quality import ecoli_mekong_2016
from .water_quality import ecoli_mekong
from .water_quality import CamelsChem


def load_nasdaq(inputs: Union[str, list, None] = None, target: str = 'NDX'):
    """Loads Nasdaq100 by downloading it if it is not already downloaded."""

    DeprecationWarning("load_nasdaq is deprecated and will be removed in future versions."
                       "See ai4water.datasets to get an appropriate dataset")

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


__version__ = '0.1.0'