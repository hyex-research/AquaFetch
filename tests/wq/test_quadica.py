
import os
import site   # so that aqua_fetch directory is in path
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

import unittest

from aqua_fetch import Quadica


DS = Quadica(path='/mnt/datawaha/hyex/atr/data')


class TestQuadica(unittest.TestCase):

    def test_avg_temp(self):
        assert DS.avg_temp().shape == (828, 1386)
        return
    
    def test_pet(self):
        assert DS.pet().shape == (828, 1386)

        data = DS.pet(stations=['1', '2'])

        assert data.shape == (828, 2), data.shape
        return

    def test_precipitation(self):
        assert DS.precipitation().shape == (828, 1386)

        data = DS.precipitation(stations=['1', '2'])

        assert data.shape == (828, 2)
        return

    def test_monthly_medians(self):
        data = DS.monthly_medians()
        assert data.shape == (16629, 17), data.shape

        data = DS.monthly_medians(stations=['1', '2'])

        assert data.shape == (24, 17)
        return

    def test_wrtds_monthly(self):
        #assert DS.wrtds_monthly().shape == (50186, 47)

        data = DS.wrtds_monthly(stations=['1', '2'])

        assert data.shape == (0, 47), data.shape

        data = DS.wrtds_monthly(stations=['218', '226'])

        assert data.shape == (530, 47), data.shape        
        return

    def test_catchment_attrs(self):
        assert DS.catchment_attributes().shape == (1386, 112), DS.catchment_attributes().shape
        assert DS.catchment_attributes(stations=['1', '2', '3']).shape == (3, 112)
        return

    def test_fetch_monthly(self):
        dyn, cat = DS.fetch_monthly(max_nan_tol=None)
        assert dyn.shape == (29484, 33)
        assert cat.shape == (29484, 112), cat.shape
        mon_dyn_tn, mon_cat_tn = DS.fetch_monthly(parameters="TN", max_nan_tol=0)
        assert mon_dyn_tn.shape == (6300, 9)
        assert mon_cat_tn.shape == (6300, 112)
        mon_dyn_tp, mon_cat_tp = DS.fetch_monthly(parameters="TP", max_nan_tol=0)
        assert mon_dyn_tp.shape == (21420, 9)
        assert mon_cat_tp.shape == (21420, 112)
        mon_dyn_toc, mon_cat_toc = DS.fetch_monthly(parameters="TOC", max_nan_tol=0)
        assert mon_dyn_toc.shape == (5796, 9)
        assert mon_cat_toc.shape == (5796, 112)
        mon_dyn_doc, mon_cat_doc = DS.fetch_monthly(parameters="DOC", max_nan_tol=0)
        assert mon_dyn_doc.shape == (6804, 9)
        assert mon_cat_doc.shape == (6804, 112)

        # mon_dyn_doc, mon_cat_doc = DS.fetch_monthly(parameters="DOC", max_nan_tol=None, stations=['218', '226'])
        # assert mon_dyn_doc.shape == (6804, 9), mon_dyn_doc.shape
        # assert mon_cat_doc.shape == (6804, 112), mon_cat_doc.shape

        return

    def test_stn_coords(self):
        coords = DS.stn_coords()
        assert coords.shape == (1386, 2), coords.shape
        assert 'lat' in coords.columns
        assert 'long' in coords.columns
        return


if __name__=="__main__":
    unittest.main()