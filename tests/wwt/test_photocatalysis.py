import os
import site

# add the parent directory in the path
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

import unittest

from aqua_fetch import (
    mg_degradation,
    dye_removal,
    dichlorophenoxyacetic_acid_removal,
    pms_removal,
    tio2_degradation,
    tetracycline_degradation,
    photodegradation_Jiang,
)


class TestPhotacatalysis(unittest.TestCase):

    def test_mg_photodegradation(self):

        data, _ = mg_degradation()
        assert data.shape == (1200, 14), data.shape
        data, encoders = mg_degradation(encoding="le")
        assert data.shape == (1200, 14)
        assert data.sum().sum() >= 406354.95, data.sum().sum()
        encoders['catalyst_type'].inverse_transform(data.loc[:, 'catalyst_type'].values.astype(int))
        encoders['anions'].inverse_transform(data.loc[:, 'anions'].values.astype(int))
        data, encoders = mg_degradation(encoding="ohe")
        assert data.shape == (1200, 33), data.shape
        encoders['catalyst_type'].inverse_transform(data.loc[:, [col for col in data.columns if col.startswith('catalyst_type')]].values)
        encoders['anions'].inverse_transform(data.loc[:, [col for col in data.columns if col.startswith('anions')]].values)

        return

    def test_dye_removal(self):

        data, encoders = dye_removal()
        assert data.shape == (1527, 38)

        data, encoders = dye_removal(encoding='le')
        assert data.shape == (1527, 38), data.shape
        catalysts = encoders['catalyst'].inverse_transform(data.loc[:, 'catalyst'].values)
        assert len(set(catalysts.tolist())) == 18
        dye = encoders['dye'].inverse_transform(data.loc[:, "dye"].values)
        set(dye.tolist())
        anions = encoders['anions'].inverse_transform(data.loc[:,'anions'].values)
        set(anions.tolist())

        data, encoders = dye_removal(encoding='ohe')
        assert data.shape == (1527, 61), data.shape
        catalysts = encoders['catalyst'].inverse_transform(data.loc[:, [col for col in data.columns if col.startswith('catalyst')]].values)
        assert len(set(catalysts.tolist())) == 18
        dye = encoders['dye'].inverse_transform(data.loc[:, ["dye_0", "dye_1"]].values)
        set(dye.tolist())
        anions = encoders['anions'].inverse_transform(data.loc[:, [col for col in data.columns if col.startswith('anions')]].values)
        set(anions.tolist())

        return

    def test_dichlorophenoxyacetic_acid_removal(self):

        data, encoders = dichlorophenoxyacetic_acid_removal()
        assert data.shape == (1044, 16), data.shape

        data, encoders = dichlorophenoxyacetic_acid_removal(encoding='le')
        assert data.shape == (1044, 16), data.shape
        catalysts = encoders['catalyst'].inverse_transform(data.loc[:, 'catalyst'].values)
        assert len(set(catalysts.tolist())) == 7
        anions = encoders['anions'].inverse_transform(data.loc[:,'anions'].values)
        set(anions.tolist())
        {'Na2SO4', 'Without Anions', 'Na2HPO4', 'NaHCO3', 'NaCO3', 'NaCl'}


        data, encoders = dichlorophenoxyacetic_acid_removal(encoding='ohe')
        assert data.shape == (1044, 27), data.shape
        catalysts = encoders['catalyst'].inverse_transform(data.loc[:, ['catalyst_0', 'catalyst_1', 'catalyst_2',
               'catalyst_3', 'catalyst_4', 'catalyst_5', 'catalyst_6']].values)
        assert len(set(catalysts.tolist())) == 7
        anions = encoders['anions'].inverse_transform(data.loc[:, [col for col in data.columns if col.startswith('anions')]].values)
        set(anions.tolist())
        {'Na2SO4', 'Without Anions', 'Na2HPO4', 'NaHCO3', 'NaCO3', 'NaCl'}
        return

    def test_pms_removal(self):

        data, encoders = pms_removal()
        assert data.shape == (2078, 25), data.shape

        data, encoders = pms_removal(encoding='le')
        assert data.shape == (2078, 25), data.shape
        catalysts = encoders['catalyst_type'].inverse_transform(data.loc[:, 'catalyst_type'].values)
        assert len(set(catalysts.tolist())) == 42, len(set(catalysts.tolist()))
        pollutant = encoders['pollutant'].inverse_transform(data.loc[:, 'pollutant'].values)
        assert len(set(pollutant.tolist())) == 14, len(set(pollutant.tolist()))
        poll_mol_formula = encoders['poll_mol_formula'].inverse_transform(data.loc[:, 'poll_mol_formula'].values)
        assert len(set(poll_mol_formula.tolist())) == 14, len(set(poll_mol_formula.tolist()))
        water_type = encoders['water_type'].inverse_transform(data.loc[:, 'water_type'].values)
        assert len(set(water_type.tolist())) == 9, len(set(water_type.tolist()))

        data, encoders = pms_removal(encoding='ohe')
        assert data.shape == (2078, 100), data.shape
        catalysts = encoders['catalyst_type'].inverse_transform(data.loc[:, [col for col in data.columns if col.startswith('catalyst_type')]].values)
        assert len(set(catalysts.tolist())) == 42, len(set(catalysts.tolist()))
        pollutant = encoders['pollutant'].inverse_transform(data.loc[:, [col for col in data.columns if col.startswith('pollutant')]].values)
        assert len(set(pollutant.tolist())) == 14, len(set(pollutant.tolist()))
        poll_mol_formula = encoders['poll_mol_formula'].inverse_transform(
            data.loc[:, [col for col in data.columns if col.startswith('poll_mol_formula')]].values)
        assert len(set(poll_mol_formula.tolist())) == 14, len(set(poll_mol_formula.tolist()))
        water_type = encoders['water_type'].inverse_transform(data.loc[:, [col for col in data.columns if col.startswith('water_type')]].values)
        assert len(set(water_type.tolist())) == 9, len(set(water_type.tolist()))

        return

    def test_tio2_degradation(self):

        data, encoders = tio2_degradation()
        assert data.shape == (446, 7), data.shape

        assert len(encoders) == 0

        return

    def test_tetracycline_degradation(self):

        data, encoders = tetracycline_degradation()
        assert data.shape == (374, 8), data.shape
        assert len(encoders) == 0

        data, encoders = tetracycline_degradation(encoding='le')
        assert data.shape == (374, 8), data.shape
        mofs = encoders['metallic_org_framework'].inverse_transform(data.loc[:, 'metallic_org_framework'].values)
        assert len(set(mofs.tolist())) == 10, len(set(mofs.tolist()))

        data, encoders = tetracycline_degradation(encoding='ohe')
        assert data.shape == (374, 17), data.shape
        mofs = encoders['metallic_org_framework'].inverse_transform(data.loc[:, [col for col in data.columns if col.startswith('metallic_org_framework')]].values)
        assert len(set(mofs.tolist())) == 10, len(set(mofs.tolist()))

        return

    def test_photodegradation_Jiang(self):
        data, encoders = photodegradation_Jiang()
        assert data.shape == (449, 8), data.shape
        assert len(encoders) == 0

        data, encoders = photodegradation_Jiang(encoding='le')
        assert data.shape == (449, 8), data.shape
        contaiminants = encoders['contaminants'].inverse_transform(data.loc[:, 'contaminants'].values)
        assert len(set(contaiminants.tolist())) == 47, len(set(contaiminants.tolist()))
        catalysts = encoders['photocatalyst'].inverse_transform(data.loc[:, 'photocatalyst'].values)
        assert len(set(catalysts.tolist())) == 100, len(set(catalysts.tolist()))

        data, encoders = photodegradation_Jiang(encoding='ohe')
        assert data.shape == (449, 153), data.shape
        contaiminants = encoders['contaminants'].inverse_transform(data.loc[:, [col for col in data.columns if col.startswith('contaminants')]].values)
        assert len(set(contaiminants.tolist())) == 47, len(set(contaiminants.tolist()))
        catalysts = encoders['photocatalyst'].inverse_transform(data.loc[:, [col for col in data.columns if col.startswith('photocatalyst')]].values)
        assert len(set(catalysts.tolist())) == 100, len(set(catalysts.tolist()))

        return


if __name__ == '__main__':
    unittest.main()
