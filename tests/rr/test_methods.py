import os
import site   # so that aqua_fetch directory is in path
import logging
import unittest

# add the parent directory in the path
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

gscad_path = '/mnt/datawaha/hyex/atr/gscad_database/raw'

if __name__ == "__main__":
    logging.basicConfig(filename='test_methods.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)


from aqua_fetch import CCAM
from aqua_fetch import CAMELSH
from aqua_fetch import CAMELS_CH
from aqua_fetch import CAMELS_GB
from aqua_fetch import CAMELS_AUS
from aqua_fetch import CAMELS_CL
from aqua_fetch import CAMELS_US
from aqua_fetch import HYPE
from aqua_fetch import WaterBenchIowa
from aqua_fetch import CAMELS_DE
from aqua_fetch import CAMELS_SE
from aqua_fetch import CAMELS_IND
from aqua_fetch import CAMELS_FR
from aqua_fetch import RainfallRunoff
from aqua_fetch import RRLuleaSweden
from aqua_fetch import CAMELS_NZ
from aqua_fetch import CAMELS_LUX
from aqua_fetch import CAMELS_COL
from aqua_fetch import CAMELS_SK
from aqua_fetch.rr import CAMELS_FI
from aqua_fetch import Thailand
from aqua_fetch import GSHA
from aqua_fetch import Japan
from aqua_fetch import Arcticnet
from aqua_fetch import Spain
from aqua_fetch import HYSETS
from aqua_fetch import NPCTRCatchments
from aqua_fetch import Simbi
from aqua_fetch import USGS
from aqua_fetch import CAMELS_BR
from aqua_fetch import CABra
from aqua_fetch import Bull
from aqua_fetch import CAMELS_DK
from aqua_fetch import Caravan_DK
from aqua_fetch import EStreams
from aqua_fetch import Ireland
from aqua_fetch import Finland
from aqua_fetch import Italy
from aqua_fetch import Poland
from aqua_fetch import Portugal
from aqua_fetch import Slovenia
from aqua_fetch import GRDCCaravan
from aqua_fetch import LamaHCE
from aqua_fetch import LamaHIce

from utils import test_stations
from utils import test_boundary
from utils import test_plot_catchment
from utils import test_q_mm


DATASETS = {
'Arcticnet': Arcticnet(path=gscad_path, verbosity=0),
"Bull": Bull(path=gscad_path, verbosity=0),
"CABra" : CABra(path=gscad_path, met_src='era5', verbosity=0),
"CABra_ens" : CABra(path=gscad_path, met_src='ens', verbosity=0),
"CABra_ref" : CABra(path=gscad_path, met_src='ref', verbosity=0),
'CCAM': CCAM(path=gscad_path, verbosity=0),
'CAMELSH': CAMELSH(path=os.path.join(gscad_path, 'CAMELS'), verbosity=0),
'CAMELS_AUS': CAMELS_AUS(path=os.path.join(gscad_path, 'CAMELS_AUS_V1'), version=1, verbosity=0),
'CAMELS_AUS_V2': CAMELS_AUS(path=os.path.join(gscad_path, 'CAMELS'), version=2, verbosity=0),
'CAMELS_BR': CAMELS_BR(path=os.path.join(gscad_path, 'CAMELS'), verbosity=0),
'CAMELS_CH': CAMELS_CH(path=os.path.join(gscad_path, 'CAMELS'), verbosity=0),
'CAMELS_CL': CAMELS_CL(path=os.path.join(gscad_path, 'CAMELS'), verbosity=0),
'CAMELS_COL': CAMELS_COL(path=os.path.join(gscad_path, 'CAMELS'), verbosity=0),
'CAMELS_DE': CAMELS_DE(path=os.path.join(gscad_path, 'CAMELS'), verbosity=0),
"CAMELS_DK": CAMELS_DK(path=os.path.join(gscad_path, 'CAMELS'), verbosity=0),
'CAMELS_FR': CAMELS_FR(path=os.path.join(gscad_path, 'CAMELS'), verbosity=0),
'CAMELS_GB': CAMELS_GB(path=os.path.join(gscad_path, 'CAMELS'), verbosity=0),
'CAMELS_IND': CAMELS_IND(path=os.path.join(gscad_path, 'CAMELS'), verbosity=0),
'CAMELS_SE': CAMELS_SE(path=os.path.join(gscad_path, 'CAMELS'), verbosity=0),
'CAMELS_US': CAMELS_US(path=os.path.join(gscad_path, 'CAMELS'), verbosity=0),
'CAMELS_NZ': CAMELS_NZ(path=os.path.join(gscad_path, 'CAMELS'), verbosity=0),
'CAMELS_LUX': CAMELS_LUX(path=os.path.join(gscad_path, 'CAMELS'), verbosity=0),
'CAMELS_SK': CAMELS_SK(path=os.path.join(gscad_path, 'CAMELS'), verbosity=0),
'CAMELS_FI': CAMELS_FI(path=os.path.join(gscad_path, 'CAMELS'), verbosity=0),
"Caravan_DK": Caravan_DK(path= gscad_path, verbosity=0),
"EStreams": EStreams(path=gscad_path, verbosity=0),
"Finland": Finland(path=gscad_path, verbosity=0),
'GSHA': GSHA(path=gscad_path, verbosity=0),
"GRDCCaravan": GRDCCaravan(path=gscad_path, verbosity=0),
'HYSETS': HYSETS(path=gscad_path, verbosity=0),
'HYPE': HYPE(path=gscad_path, verbosity=0),
"Ireland": Ireland(path=gscad_path, verbosity=0),
"Italy": Italy(path=gscad_path, verbosity=0),
'Japan': Japan(path=gscad_path, verbosity=0),
"LamaHCE_tu": LamaHCE(timestep='D', data_type="total_upstrm", path=os.path.join(gscad_path, 'LamaHCE_daily'), verbosity=10),
"LamaHCE_ia": LamaHCE(timestep='D', data_type="intermediate_all", path=os.path.join(gscad_path, 'LamaHCE_daily'), verbosity=10),
"LamaHCE_il": LamaHCE(timestep='D', data_type="intermediate_lowimp", path=os.path.join(gscad_path, 'LamaHCE_daily'), verbosity=10),
"LamaHCE_h_tu": LamaHCE(timestep='H', data_type="total_upstrm", path=gscad_path, verbosity=10),
"LamaHCE_h_ia": LamaHCE(timestep='H', data_type="intermediate_all", path=gscad_path, verbosity=10),
"LamaHCE_h_il": LamaHCE(timestep='H', data_type="intermediate_lowimp", path=gscad_path, verbosity=10),
"LamaHIce_h_tu": LamaHIce(path=gscad_path, timestep="H", data_type="total_upstrm", verbosity=0),
"LamaHIce_h_ia": LamaHIce(path=gscad_path, timestep="H", data_type="intermediate_all", verbosity=0),
"LamaHIce_h_il": LamaHIce(path=gscad_path, timestep="H", data_type="intermediate_lowimp", verbosity=0),
"LamaHIce_d_tu": LamaHIce(path=os.path.join(gscad_path, 'LamaHIce_daily'), timestep="D", data_type="total_upstrm", verbosity=0),
"LamaHIce_d_ia": LamaHIce(path=os.path.join(gscad_path, 'LamaHIce_daily'), timestep="D", data_type="intermediate_all", verbosity=0),
"LamaHIce_d_il": LamaHIce(path=os.path.join(gscad_path, 'LamaHIce_daily'), timestep="D", data_type="intermediate_lowimp", verbosity=0),
'NPCTRCatchments': NPCTRCatchments(path='/mnt/datawaha/hyex/atr/data', verbosity=0),
"Poland": Poland(path=gscad_path, verbosity=0),
"Portugal": Portugal(path=gscad_path, verbosity=0),
'RainfallRunoff': RainfallRunoff('CAMELS_AUS', path=os.path.join(gscad_path, 'CAMELS'), verbosity=0),
'RRLuleaSweden': RRLuleaSweden(path=os.path.join(gscad_path, 'RRLuleaSweden'), verbosity=0),
'Simbi' : Simbi(path=gscad_path, verbosity=0),
"Slovenia": Slovenia(path=gscad_path, verbosity=0),
'Spain': Spain(path=gscad_path, verbosity=0),
'Thailand': Thailand(path=gscad_path, verbosity=0),
'USGS' : USGS(path=gscad_path, verbosity=0),
'WaterBenchIowa': WaterBenchIowa(path=gscad_path, verbosity=0),
}


def test_get_boundary():

    for ds_name, ds in DATASETS.items():

        if ds_name not in ['HYPE', 'WaterBenchIowa', 'RRLuleaSweden', 'CAMELS_NZ',
                       ]:
            test_boundary(ds)
    return


def test_plot_catchment_method():

    for ds_name, ds in DATASETS.items():

        if ds_name not in ['HYPE', 'WaterBenchIowa', 'RRLuleaSweden', 
                           'CAMELS_NZ',
                       ]:
            test_plot_catchment(ds)
    return


def test_qmm_method():

    for ds_name, ds in DATASETS.items():

        if ds_name not in ["EStreams", "GSHA", "RRLuleaSweden", "WaterBenchIowa"]:
            test_q_mm(ds)
    return


def test_stations_method():

    numbers = {
        'Estreams': 17130,
        'Ireland': 464,
        'Finland': 669,
        'Italy': 294,
        'Poland': 1287,
        'Portugal': 280,
        'Slovenia': 117,
        'CCAM': 102,
        'CAMELS_CH': 331,
        'CAMELS_GB': 671,
        'CAMELS_AUS': 222,
        'CAMELS_AUS_V2': 561,
        'CAMELS_CL': 516,
        'CAMELS_US': 671,
        'HYPE': 564,
        'WaterBenchIowa': 125,
        'CAMELS_DE': 1555,
        'CAMELS_SE': 50,
        'CAMELS_IND': 472,
        'CAMELS_FR': 654,
        'RainfallRunoff': 561,
        'CAMELS_NZ': 369,
        'CAMELS_LUX': 56,
        'CAMELS_COL': 347,
        'CAMELS_SK': 178,
        'CAMELS_FI': 320,
        'Thailand': 73,
        'GSHA': 21568,
        'Japan': 751,
        'Arcticnet': 106,
        'Spain': 889,
        'HYSETS': 14425,
        'NPCTRCatchments': 7,
        'Simbi' : 24,
        'USGS' : 12004,
        "CABra" : 735,
        "CABra_ens" : 735,
        "CABra_ref" : 735,
        'CAMELS_BR': 897,
        "Bull": 484,
        "CAMELS_DK": 304,
        "Caravan_DK": 308,
        "Estreams": 17130,
        "Ireland": 464,
        "Finland": 669,
        "Italy": 294,
        "Poland": 1287,
        "Portugal": 280,
        "Slovenia": 117,
        "GRDCCaravan": 5357,
        "LamaHCE_tu": 859,
        "LamaHCE_ia": 859,
        "LamaHCE_il": 454,
        "LamaHCE_h_tu": 859,
        "LamaHCE_h_ia": 859,
        "LamaHCE_h_il": 454,
        "LamaHIce_h_tu": 111,
        "LamaHIce_h_ia": 107,
        "LamaHIce_h_il": 86,
        "LamaHIce_d_tu": 111,
        "LamaHIce_d_ia": 107,
        "LamaHIce_d_il": 86
    }

    for ds_name, num_stn in numbers.items():

        if ds_name not in ['RRLuleaSweden']:
            test_stations(DATASETS[ds_name], num_stn)
    return


test_get_boundary()

test_plot_catchment_method()

test_stations_method()

test_qmm_method()
