
import os
import site   # so that aqua_fetch directory is in path
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

from aqua_fetch import CamelsChem
from aqua_fetch.wq._camels_chem import Camels_Ch_Chem


ds = CamelsChem(
    path='/mnt/datawaha/hyex/atr/data',
    verbosity=3
    )

metrics = ds.metrics()
assert metrics.shape == (41, 6)

data = ds.data()
assert data.shape == (76284, 45)


atm_data = ds.atm_dep_data()
assert atm_data.shape == (22814, 77)


def test_fetch():
    data = ds.fetch('1591400', 'cl_mg/l')['1591400']
    assert data.shape == (55, 1)        


    data = ds.fetch(stations=['1591400', '6350000'], parameters=['cl_mg/l', 'na_mg/l'])
    assert data['1591400'].shape == (55, 2)
    assert data['6350000'].shape == (328, 2)

    data = ds.fetch('1591400')['1591400']
    assert data.shape == (55, 28)

    all_data = ds.fetch()
    assert len(all_data) == 516

    return 


def test_fetch_atm_dep():

    data = ds.fetch_atm_dep(stations='1591400', parameters='cl')
    assert data['1591400'].shape == (34, 8)

    data = ds.fetch_atm_dep(stations=['1591400', '6350000'], parameters=['cl', 'na'])
    assert data['1591400'].shape == (34, 16)
    assert data['6350000'].shape == (34, 16)

    data = ds.fetch_atm_dep()
    assert len(data) == 671

    return  


def test_camels_ch_chem():
    
    ds = Camels_Ch_Chem(
        path='/mnt/datawaha/hyex/atr/data',
        verbosity=3
        )

    stns = ds.stations()

    assert len(stns) == 115

    coords = ds.stn_coords()
    assert coords.shape == (115, 2)

    data = ds.fetch_catch_avg(stations=['2009', '2011'])
    assert data['2009'].shape == (209, 32)
    assert data['2011'].shape == (209, 32)
    data = ds.fetch_catch_avg()
    assert len(data) == 115

    data = ds.fetch_wq_ts(stations=['2009', '2011'])
    assert data['2009'].shape == (14610, 4)
    assert data['2011'].shape == (14610, 4)
    data = ds.fetch_wq_ts()
    assert len(data) == 115

    data = ds.fetch_isotope(stations=['2009', '2016'])
    assert data['2009'].shape == (452, 4)
    assert data['2016'].shape == (450, 4)

    return

test_fetch()

test_fetch_atm_dep()

test_camels_ch_chem()

print('All tests passed!')
