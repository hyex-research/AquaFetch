
import os
import site   # so that aqua_fetch directory is in path
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

from aqua_fetch import CamelsChem
from aqua_fetch import CamelsCHChem


def test_camels_chem():

    dataset = CamelsChem(
        path='/mnt/datawaha/hyex/atr/data',
        verbosity=3
        )

    metrics = dataset.metrics()
    assert metrics.shape == (41, 6)

    data = dataset.data()
    assert data.shape == (76284, 45)

    ## ** testing fetch
    data = dataset.fetch('1591400', 'cl_mg/l')['1591400']
    assert data.shape == (55, 1)        

    data = dataset.fetch(stations=['1591400', '6350000'], parameters=['cl_mg/l', 'na_mg/l'])
    assert data['1591400'].shape == (55, 2)
    assert data['6350000'].shape == (328, 2)

    data = dataset.fetch('1591400')['1591400']
    assert data.shape == (55, 28)

    all_data = dataset.fetch()
    assert len(all_data) == 516

    # # ** testing atm_dep_data
    atm_data = dataset.atm_dep_data()
    assert atm_data.shape == (22814, 77)

    data = dataset.fetch_atm_dep()
    assert len(data) == 671

    data = dataset.fetch_atm_dep(stations='1591400', parameters='cl')
    assert data['1591400'].shape == (34, 8)

    data = dataset.fetch_atm_dep(stations=['1591400', '6350000'], parameters=['cl', 'na'])
    assert data['1591400'].shape == (34, 16)
    assert data['6350000'].shape == (34, 16)

    return  


def test_camels_ch_chem():

    dataset = CamelsCHChem(
        path='/mnt/datawaha/hyex/atr/data',
        verbosity=3
        )

    stns = dataset.stations()

    assert len(stns) == 115

    coords = dataset.stn_coords()
    assert coords.shape == (115, 2)

    data = dataset.fetch_catch_avg(stations=['2009', '2011'])
    assert data['2009'].shape == (209, 32)
    assert data['2011'].shape == (209, 32)
    data = dataset.fetch_catch_avg()
    assert len(data) == 115

    data = dataset.fetch_wq_ts(stations=['2009', '2011'])
    assert data['2009'].shape == (14610, 4)
    assert data['2011'].shape == (14610, 4)
    data = dataset.fetch_wq_ts()
    assert len(data) == 115

    data = dataset.fetch_isotope(stations=['2009', '2016'])
    assert data['2009'].shape == (452, 4)
    assert data['2016'].shape == (450, 4)

    return


test_camels_chem()

test_camels_ch_chem()

print('All tests passed!')
