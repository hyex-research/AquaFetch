import os
import site   # so that aqua_fetch directory is in path
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

import unittest
from unittest.mock import MagicMock

import matplotlib.pyplot as plt

from aqua_fetch import mg_degradation
from aqua_fetch import RainfallRunoff
from aqua_fetch.rr.utils import _RainfallRunoff
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


class TestFetchSeed(unittest.TestCase):
    """Reproducible (seeded) random station selection in ``fetch``.

    These are data-free: the two data-touching hooks (``stations`` and
    ``fetch_stations_features``) are stubbed so the real sampling code in
    ``fetch`` runs in isolation without downloading or reading any dataset.
    """

    POOL = [str(i) for i in range(100)]

    def _fetch(self, stations, seed=None):
        # invoke the real fetch() with data access stubbed out; return the list
        # of station ids that fetch forwarded to fetch_stations_features
        inst = MagicMock(spec=_RainfallRunoff)
        inst.stations.return_value = list(self.POOL)
        inst.fetch_stations_features.side_effect = lambda selected, *a, **k: selected
        return _RainfallRunoff.fetch(inst, stations=stations, seed=seed)

    def test_int_reproducible_with_seed(self):
        a = self._fetch(5, seed=313)
        b = self._fetch(5, seed=313)
        assert a == b, (a, b)
        assert len(a) == 5
        assert set(a).issubset(set(self.POOL))
        return

    def test_float_reproducible_with_seed(self):
        a = self._fetch(0.1, seed=7)
        b = self._fetch(0.1, seed=7)
        assert a == b, (a, b)
        assert len(a) == 10  # 10% of 100
        return

    def test_different_seeds_differ(self):
        # extremely unlikely to coincide for a 5-of-100 draw
        assert self._fetch(5, seed=1) != self._fetch(5, seed=2)
        return

    def test_seed_none_is_valid(self):
        sel = self._fetch(5, seed=None)
        assert len(sel) == 5
        assert set(sel).issubset(set(self.POOL))
        return

    def test_global_random_state_untouched(self):
        import random
        random.seed(1234)
        before = [random.random() for _ in range(3)]
        random.seed(1234)
        self._fetch(5, seed=313)          # must not consume from the global RNG
        after = [random.random() for _ in range(3)]
        assert before == after, (before, after)
        return

    def test_explicit_list_passes_through(self):
        picked = ['3', '9', '27']
        assert self._fetch(picked, seed=42) == picked
        return


if __name__ == "__main__":
    unittest.main()