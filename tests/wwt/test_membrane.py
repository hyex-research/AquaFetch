
import os
import site

# add the parent directory in the path
wd_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(wd_dir)

import unittest

from aqua_fetch import (
    micropollutant_removal_osmosis,
    ion_transport_via_reverse_osmosis,
)


class TestMembrane(unittest.TestCase):

    pass


#data, _ = micropollutant_removal_osmosis()
