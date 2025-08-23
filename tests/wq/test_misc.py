import os
import site   # so that aqua_fetch directory is in path
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#wd_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(wd_dir)

from aqua_fetch import (
    SanFranciscoBay, BuzzardsBay, WhiteClayCreek,
    RiverChemSiberia, SeluneRiver
    )
from aqua_fetch.wq._misc import NinAfrica

def test_SanFranciscoBay():

    ds = SanFranciscoBay(
        path='/mnt/datawaha/hyex/atr/data',
        verbosity=4
        )

    stations = ds.stations()
    assert len(stations) == 59

    data = ds.data()
    assert data.shape == (212472, 19)

    parameters = ds.parameters()
    assert len(parameters) == 18
    # fetch data for station 18
    stn18 = ds.fetch(stations='18')
    assert stn18.shape == (13944, 18)

    return


def test_BuzzardsBay():
    ds = BuzzardsBay(
        path='/mnt/datawaha/hyex/atr/data',
        verbosity=4
        )

    assert len(ds.stations()) == 401

    assert len(ds.parameters) == 64

    data = ds.data()
    assert data.shape == (99670, 64)

    stations = ds.read_stations()
    assert stations.shape == (401, 10)

    return


def test_WhiteClayCreek():
    ds = WhiteClayCreek(
        path='/mnt/datawaha/hyex/atr/data',
        verbosity=4
        )

    df = ds.doc()
    assert df.shape == (11092, 4)
    assert df['site'].nunique() == 2

    chla = ds.chla()
    assert chla.shape == (1028, 10)

    return


def test_RiverChemSiberia():
    ds = RiverChemSiberia(
        path='/mnt/datawaha/hyex/atr/data',
        verbosity=4
        )

    database = ds.database()

    assert database.shape == (1434, 34)
    assert database['Sample_ID'].nunique() == 1422

    database['Basin'].nunique() == 7

    return


def test_SeluneRiver():

    ds = SeluneRiver(
        path='/mnt/datawaha/hyex/atr/data',
        verbosity=4
        )

    data = ds.data()
    #assert data.shape == 
    return


test_SanFranciscoBay()
     
test_BuzzardsBay()

test_WhiteClayCreek()

test_RiverChemSiberia()

test_SeluneRiver()

print("All tests passed!")


dataset = NinAfrica(verbosity=3)

data = dataset.data()

data.shape