
import os
import site

# add the parent directory in the path
wd_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(wd_dir)

import unittest

from water_datasets import (
    ec_removal_biochar,
    cr_removal,
    po4_removal_biochar,
    heavy_metal_removal,
    heavy_metal_removal_Shen,
    industrial_dye_removal,
    N_recovery,
    P_recovery,
    As_recovery,
)


class TestAdsorption(unittest.TestCase):

    def test_ec_removal_biochar(self):
        data, _ = ec_removal_biochar()
        assert data.shape == (3757, 29), data.shape
        data, encoders = ec_removal_biochar(encoding="le")
        assert data.shape == (3757, 29)
        assert data.sum().sum() >= 10346311.47, data.sum().sum()
        adsorbents = encoders['adsorbent'].inverse_transform(data.loc[:, 'adsorbent'])
        assert len(set(adsorbents)) == 15
        pollutants = encoders['pollutant'].inverse_transform(data.loc[:, 'pollutant'])
        assert len(set(pollutants)) == 14
        ww_types = encoders['wastewater_type'].inverse_transform(data.loc[:, 'wastewater_type'])
        assert len(set(ww_types)) == 4
        adsorption_types = encoders['adsorption_type'].inverse_transform(data.loc[:, 'adsorption_type'])
        assert len(set(adsorption_types)) == 2
        data, encoders = ec_removal_biochar(encoding="ohe")
        assert data.shape == (3757, 60), data.shape
        # adsorbents = encoders['adsorbent'].inverse_transform(
        #     data.loc[:, [col for col in data.columns if col.startswith('adsorbent')]].values
        # )
        # assert len(set(adsorbents)) == 15
        pollutants = encoders['pollutant'].inverse_transform(
            data.loc[:, [col for col in data.columns if col.startswith('pollutant')]].values)
        assert len(set(pollutants)) == 14
        ww_types = encoders['wastewater_type'].inverse_transform(
            data.loc[:, [col for col in data.columns if col.startswith('wastewater_type')]].values)
        assert len(set(ww_types)) == 4
        adsorption_types = encoders['adsorption_type'].inverse_transform(
            data.loc[:, [col for col in data.columns if col.startswith('adsorption_type')]].values)
        assert len(set(adsorption_types)) == 2
        return

    def test_cr_removal(self):

        data, _ = cr_removal()
        assert data.shape == (219, 20)

        data, encoders = cr_removal(encoding='le')
        assert data.shape == (219, 20)
        adsorbents = len(set(encoders['adsorbent'].inverse_transform(data.loc[:, 'adsorbent']).tolist()))
        assert adsorbents == 5, adsorbents

        data, encoders = cr_removal(encoding="ohe")
        assert data.shape == (219, 24)
        adsorbents = len(set(
            encoders['adsorbent'].inverse_transform(
                data.loc[:, [col for col in data.columns if col.startswith('adsorbent')]].astype(int).values
            ).tolist()))
        assert adsorbents == 5, adsorbents

        return

    def test_po4_removal_biochar(self):
        data, _ = po4_removal_biochar()
        assert data.shape == (5014, 33)

        data, encoders = po4_removal_biochar(encoding='le')
        assert data.shape == (5014, 33)
        assert len(set(encoders['adsorbent'].inverse_transform(data.loc[:, 'adsorbent']).tolist())) == 253
        assert len(set(encoders['feedstock'].inverse_transform(data.loc[:, 'feedstock']).tolist())) == 88
        assert len(set(encoders['ion_type'].inverse_transform(data.loc[:, 'ion_type']).tolist())) == 18

        data, encoders = po4_removal_biochar(encoding='ohe')
        assert data.shape == (5014, 389)
        assert len(set(
            encoders['adsorbent'].inverse_transform(
                data.loc[:, [col for col in data.columns if col.startswith('adsorbent')]].astype(int).values
            ).tolist())) == 253

        assert len(set(
            encoders['feedstock'].inverse_transform(
                data.loc[:, [col for col in data.columns if col.startswith('feedstock')]].astype(int).values
            ).tolist())) == 88

        assert len(set(
            encoders['ion_type'].inverse_transform(
                data.loc[:, [col for col in data.columns if col.startswith('ion_type')]].astype(int).values
            ).tolist())) == 18
        
        return

    def test_heavy_metal_removal(self):
        data, _ = heavy_metal_removal()
        assert data.shape == (219, 18)

        data, encoders = heavy_metal_removal(encoding='le')
        assert data.shape == (219, 18)
        adsorbents = set(encoders['adsorbent'].inverse_transform(data.loc[:, 'adsorbent']).tolist())
        assert len(adsorbents) == 5, adsorbents

        data, encoders = heavy_metal_removal(encoding='ohe')
        assert data.shape == (219, 22), data.shape
        adsorbents = set(encoders['adsorbent'].inverse_transform(
            data.loc[:, [col for col in data.columns if col.startswith('adsorbent')]].astype(int).values
        ).tolist())
        assert len(adsorbents) == 5, adsorbents
        return 

    def test_heavy_metal_removal_shen(self):
        data, encoders = heavy_metal_removal_Shen()
        assert data.shape == (353, 18)

        data, encoders = heavy_metal_removal_Shen(encoding="le")
        assert data.shape == (353, 18), data.shape
        adsorbents = set(encoders['heavy_metal'].inverse_transform(data.loc[:, 'heavy_metal']).tolist())
        assert len(adsorbents) == 10, len(adsorbents)
        adsorbents = set(encoders['hm_label'].inverse_transform(data.loc[:, 'hm_label']).tolist())
        assert len(adsorbents) == 42, len(adsorbents)

        data, encoders = heavy_metal_removal_Shen(encoding="ohe")
        assert data.shape == (353, 68), data.shape
        adsorbents = set(encoders['heavy_metal'].inverse_transform(
            data.loc[:, [col for col in data.columns if col.startswith('heavy_metal')]].astype(int).values).tolist())
        assert len(adsorbents) == 10, len(adsorbents)
        adsorbents = set(encoders['hm_label'].inverse_transform(
            data.loc[:, [col for col in data.columns if col.startswith('hm_label')]].astype(int).values).tolist())
        assert len(adsorbents) == 42, len(adsorbents)

        return

    def test_industrial_dye_removal(self):
        data, encoders = industrial_dye_removal()
        assert data.shape == (680, 29)

        data, encoders = industrial_dye_removal(encoding="le")
        assert data.shape == (680, 29), data.shape
        adsorbents = set(encoders['adsorbent'].inverse_transform(data.loc[:, 'adsorbent']).tolist())
        assert len(adsorbents) == 7, len(adsorbents)
        dyes = set(encoders['dye'].inverse_transform(data.loc[:, 'dye']).tolist())
        assert len(dyes) == 4, len(dyes)

        data, encoders = industrial_dye_removal(encoding="ohe")
        assert data.shape == (680, 38), data.shape
        # adsorbents = set(encoders['adsorbent'].inverse_transform(
        #     data.loc[:, [col for col in data.columns if col.startswith('adsorbent')]].astype(int).values).tolist())
        # assert len(adsorbents) == 7, len(adsorbents)
        dyes = set(encoders['dye'].inverse_transform(
            data.loc[:, [col for col in data.columns if col.startswith('dye')]].astype(int).values).tolist())
        assert len(dyes) == 4, len(dyes)
        return

    def test_N_recovery(self):
        data, encoders = N_recovery()
        assert data.shape == (210, 8)

        assert len(encoders) == 0

        return

    def test_P_recovery(self):
        data, encoders = P_recovery()
        assert data.shape == (504, 8), data.shape

        assert len(encoders) == 0

        return

    def test_As_recovery(self):
        data, encoders = As_recovery()
        assert data.shape == (684, 14), data.shape

        return


if __name__ == '__main__':
    unittest.main()
