
# https://doi.org/10.1016/j.cej.2021.130649
# https://doi.org/10.1007/s42773-022-00183-w
# https://doi.org/10.1007/s42773-024-00303-8
# https://doi.org/10.1016/j.scitotenv.2024.173939  hydrothermal liquefaction
# https://doi.org/10.1016/j.energy.2024.133707
# https://doi.org/10.1016/j.scitotenv.2024.176780

# parameter naming conventions
# - no white space
# provide units wherever possible
# avoid capital except for element symbols
# avoid ( ) : ;

from .adsorption import ec_removal_biochar
from .adsorption import cr_removal
from .adsorption import po4_removal_biochar
from .adsorption import heavy_metal_removal
from .adsorption import industrial_dye_removal
from .adsorption import heavy_metal_removal_Shen
from .adsorption import P_recovery
from .adsorption import N_recovery
from .adsorption import As_recovery

from .photocatalysis import mg_degradation
from .photocatalysis import dye_removal
from .photocatalysis import dichlorophenoxyacetic_acid_removal
from .photocatalysis import pms_removal
from .photocatalysis import tio2_degradation
from .photocatalysis import tetracycline_degradation

from .membrane import micropollutant_removal_osmosis
from .membrane import ion_transport_via_reverse_osmosis

from .sonolysis import cyanobacteria_disinfection
