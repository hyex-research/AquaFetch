import os
import site   # so that water_datasets directory is in path
wd_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(wd_dir)

import unittest

from aqua_fetch import mg_degradation
from aqua_fetch.utils import LabelEncoder, OneHotEncoder

data_path = '/mnt/datawaha/hyex/atr/data'

class TestEncoders(unittest.TestCase):

    def test_labelencoder(self):
        data, _, _ = mg_degradation()
        cat_enc1 = LabelEncoder()
        cat_ = cat_enc1.fit_transform(data['Catalyst_type'].values)
        _cat = cat_enc1.inverse_transform(cat_)
        all([a == b for a, b in zip(data['Catalyst_type'].values, _cat)])
        return

    def test_ohe(self):
        data, _, _ = mg_degradation()
        cat_enc1 = OneHotEncoder()
        cat_ = cat_enc1.fit_transform(data['Catalyst_type'].values)
        _cat = cat_enc1.inverse_transform(cat_)
        all([a==b for a,b in zip(data['Catalyst_type'].values, _cat)])
        return


if __name__ == "__main__":
    unittest.main()