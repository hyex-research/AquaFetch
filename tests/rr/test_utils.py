import os
import site   # so that aqua_fetch directory is in path
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

import unittest

import matplotlib.pyplot as plt

from aqua_fetch import mg_degradation
from aqua_fetch import RainfallRunoff
from aqua_fetch.utils import LabelEncoder, OneHotEncoder

data_path = '/mnt/datawaha/hyex/atr/data'
gscad_path = '/mnt/datawaha/hyex/atr/gscad_database/raw'


dataset = RainfallRunoff('CAMELS_COL', path=os.path.join(gscad_path, 'CAMELS'), verbosity=0)
dataset1 = RainfallRunoff('CAMELS_AUS', path=os.path.join(gscad_path, 'CAMELS'), verbosity=0)


class TestEncoders(unittest.TestCase):

    def test_labelencoder(self):
        data, _ = mg_degradation()
        cat_enc1 = LabelEncoder()
        cat_ = cat_enc1.fit_transform(data['catalyst_type'].values)
        _cat = cat_enc1.inverse_transform(cat_)
        assert all([a == b for a, b in zip(data['catalyst_type'].values, _cat)])
        return

    def test_ohe(self):
        data, _ = mg_degradation()
        cat_enc1 = OneHotEncoder()
        cat_ = cat_enc1.fit_transform(data['catalyst_type'].values)
        _cat = cat_enc1.inverse_transform(cat_)
        assert all([a==b for a,b in zip(data['catalyst_type'].values, _cat)])
        return


class Testplotnumobservations(unittest.TestCase):

    def test_vanilla(self):
        ax = dataset.plot_num_observations(show=False)
        assert isinstance(ax, plt.Axes)
        plt.close('all')
        return

    def test_with_axes(self):
        fig, ax = plt.subplots()
        ax = dataset.plot_num_observations(ax=ax, show=False)
        assert isinstance(ax, plt.Axes)
        plt.close('all')
        return

    def test_with_dyn_features(self):
        ax = dataset1.plot_num_observations(
            dynamic_features=['q_cms_obs', 'q_mm_obs', 'streamflow_MLd_inclInfilled'], show=False)
        assert isinstance(ax, plt.Axes)
        plt.close('all')
        return

    def test_with_show_constants(self):
        ax = dataset.plot_num_observations(show_constant=True, show=False)
        assert isinstance(ax, plt.Axes)
        plt.close('all')
        return
    
    def test_with_st_en(self):
        _, ax = plt.subplots()
        for idx, period in enumerate([("19810101", "19901231"), ("19910101", "20001231"), ("20010101", "20101231")]):
            start, end = period
            ax = dataset.plot_num_observations(
                dynamic_features=['q_cms_obs'],
                ax=ax,
                start=start, end=end, show=False)
            ax.lines[idx].set_label(f'{start} to {end}')
            assert isinstance(ax, plt.Axes)
        ax.legend()
        plt.close('all')
        return


if __name__ == "__main__":
    unittest.main()