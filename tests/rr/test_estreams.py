
import os
import site
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

import logging

if __name__ == "__main__":
    logging.basicConfig(filename='test_estreams.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ignore all warnings
import warnings
warnings.filterwarnings("ignore")

from unittest import mock

import numpy as np
import pandas as pd
import xarray as xr

from aqua_fetch import EStreams
from aqua_fetch import Ireland
from aqua_fetch import Finland
from aqua_fetch import Italy
from aqua_fetch import Poland
from aqua_fetch import Portugal
from aqua_fetch import Slovenia
from aqua_fetch.rr._estreams import _imgw_dates

from utils import (
    test_dataset as run_shared_tests,
    test_coords as check_coords,
    test_stations as check_stations,
    test_area as check_area,
    test_boundary as check_boundary,
    test_plot_stations as check_plot_stations,
    test_fetch_static_feature as check_fetch_static_feature,
    test_fetch_dynamic_features as check_fetch_dynamic_features,
    test_plot_catchment as check_plot_catchment,
    )

raw_data_path = '/path/to/your/raw/data'  # change this path accordingly

ds = EStreams(path=raw_data_path, verbosity=3)

assert ds.md.shape == (17130, 29), ds.md.shape

assert len(ds.countries) == 39, len(ds.agencies)

assert len(ds.country_stations('IT')) == 767, len(ds.country_stations('IT'))

assert len(ds.country_stations('ES')) == 1440, len(ds.country_stations('ES'))

assert len(ds.country_stations('IE')) == 464, len(ds.country_stations('IE'))

assert len(ds.country_stations('PL')) == 1287, len(ds.country_stations('PL'))


def test_coords_():

    check_coords(ds)

    assert ds.stn_coords(countries='IE').shape == (464, 2), ds.stn_coords(countries='IE').shape

    assert ds.stn_coords(stations='IEEP0281').shape == (1, 2), ds.stn_coords(stations='IEEP0281').shape

    return


def test_area_():

    check_area(ds)

    assert ds.area(countries='IE').shape == (464, ), ds.area(countries='IE').shape

    assert ds.area(stations='IEEP0281').shape == (1, ), ds.area(stations='IEEP0281').shape

    return


def test_fetch_static_features():
    # out = ds.fetch_static_features(countries='IE')  # todo : why countries argument is not working
    # assert out.shape == (464, 251), out.shape

    check_fetch_static_feature(ds, 'IEEP0281', 17130, 251)
    return


def test_meteo():
    out = ds.meteo_data_station('IEEP0281')

    assert isinstance(out, pd.DataFrame), type(out)

    assert out.shape[1] == 11, out.shape

    out = ds.meteo_data(countries='IE')
    assert len(out) == 464, len(out)

    out = ds.meteo_data()

    assert len(out) == 17130, len(out)

    return


def test_fetch_dynamic_features_():
    out = ds.fetch_dynamic_features(countries='IE')

    assert len(out) == 464, len(out)

    check_fetch_dynamic_features(ds, 'IEEP0281', 17130)

    return


def test_fetch_():
    """fetch() returns what fetch_dynamic_features and fetch_static_features
    return (it used to raise NotImplementedError)"""
    stns = ds.country_stations('IE')[:20]
    st, en = '2000-01-01', '2000-12-31'
    static, dynamic = ds.fetch(stns, static_features='all', st=st, en=en)
    assert dict(dynamic.sizes) == {'time': 366, 'dynamic_features': 11}, dict(dynamic.sizes)
    xr.testing.assert_identical(dynamic, ds.fetch_dynamic_features(stns, st=st, en=en))
    pd.testing.assert_frame_equal(static, ds.fetch_static_features(stns, 'all'))
    return


def test_release_update_():
    """
    An older release on disk is replaced by release 1.4 once, with a warning,
    and the weather cache built from it is deleted. Zenodo is stubbed.
    """
    import shutil, tempfile, zipfile
    import aqua_fetch.download_zenodo as dz

    def fake_download(outdir, doi, include, verbosity):
        for archive in include:
            with zipfile.ZipFile(os.path.join(outdir, archive), 'w') as zf:
                zf.writestr(f'{archive[:-4]}/estreams_{archive[:-4]}_AT000001.csv', 'date,p_mean,petmk_mean\n')

    tmp = tempfile.mkdtemp()
    try:
        new = EStreams.__new__(EStreams)
        new.verbosity, new.remove_zip, new.overwrite, new._meteo_axes_cache = 0, True, False, None
        new.path = tmp   # the setter appends 'EStreams'
        os.makedirs(os.path.join(new.path2, 'meteorology'))
        with open(os.path.join(new.path2, 'meteorology', 'estreams_meteorology_AT000001.csv'), 'w') as fp:
            fp.write('date,p_mean,pet_mean\n')   # release 1.1 has no petmk_mean column
        with open(new.nc_path, 'w') as fp:
            fp.write('cache of release 1.1')

        with mock.patch.object(dz, 'download_from_zenodo', fake_download), \
                warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            new._download()
        assert any('older than 1.4' in str(w.message) for w in caught)
        assert sorted(os.listdir(new.path2)) == sorted(a[:-4] for a in EStreams.archives)
        assert not os.path.exists(new.nc_path)

        # release 1.4 is on disk, so nothing is downloaded again
        with mock.patch.object(dz, 'download_from_zenodo', side_effect=AssertionError("downloaded")):
            new._download()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return


def test_release_values_():
    """the weather of a station is served as published in release 1.4"""
    stn = 'AT000001'
    out = ds.meteo_data_station(stn)
    raw = pd.read_csv(os.path.join(ds.path2, 'meteorology', f'estreams_meteorology_{stn}.csv'),
                      index_col='date', parse_dates=True)
    assert out.index.equals(raw.index) and len(out) == 27210, len(out)
    # float32 when read from the weather cache, float64 from the csv files
    np.testing.assert_array_equal(out.to_numpy(), raw.to_numpy(out.to_numpy().dtype))
    # the static features of different files do not share a name
    assert ds.fetch_static_features('all', 'all').columns.is_unique
    return


test_coords_()

test_area_()

test_fetch_static_features()

test_meteo()

test_fetch_dynamic_features_()

test_fetch_()

test_release_update_()

test_release_values_()

check_boundary(ds)

check_plot_stations(ds)

check_stations(ds, 17130)

check_plot_catchment(ds)

print('All tests passed!')


ds = Ireland(path=raw_data_path, verbosity=3)

ds.basin_id_gauge_id_map()['IEOP0126'] == '20002'
ds.gauge_id_basin_id_map()['20002'] == 'IEOP0126'

run_shared_tests(ds, 
             num_stations=464, 
             dyn_data_len=27210, 
             num_static_attrs=251,
              num_dyn_attrs=12
              )

_, dynamic = ds.fetch('all', dynamic_features='q_cms_obs', as_dataframe=True)
pd.concat(list(dynamic.values()), axis=1).count().sum() >= 3303345


ds = Ireland(path=raw_data_path, verbosity=3, processes=32, timestep='H')

h_q = ds.fetch_q()

# only 'D' and 'H' are accepted; other spellings used to crash the first
# streamflow download with a KeyError, and 'd' read the hourly file
for timestep in ('daily', 'hourly', 'd', 'h'):
    try:
        Ireland(path=raw_data_path, timestep=timestep, verbosity=0)
    except ValueError:
        pass
    else:
        raise AssertionError(f"Ireland accepted timestep={timestep!r}")

ds = Finland(path=raw_data_path, processes=1, 
             verbosity=3)

run_shared_tests(ds, 
             num_stations=669, 
             dyn_data_len=4565, 
             num_static_attrs=251,
              num_dyn_attrs=12,
              st="20200101",
              en="20201231"
              )
_, dynamic = ds.fetch('all', dynamic_features='q_cms_obs', as_dataframe=True)
pd.concat(list(dynamic.values()), axis=1).count().sum() >= 814277

ds = Italy(path=raw_data_path, verbosity=3)

run_shared_tests(ds, 
             num_stations=294, 
             dyn_data_len=27210, 
             num_static_attrs=251,
              num_dyn_attrs=12
              )



ds = Poland(path=raw_data_path, verbosity=3)

run_shared_tests(ds, 
             num_stations=1287, 
             dyn_data_len=27210, 
             num_static_attrs=251,
              num_dyn_attrs=12
              )

_, dynamic = ds.fetch('all', dynamic_features='q_cms_obs', as_dataframe=True)
pd.concat(list(dynamic.values()), axis=1).count().sum() >= 16319627

q = ds.fetch_q()

assert q.shape[1]>1287

# IMGW files carry the hydrological year, which starts on 1 November, so
# November and December belong to the calendar year before it
assert list(_imgw_dates(pd.Series([1951, 1951, 1951]), pd.Series([11, 12, 1]), pd.Series([1, 31, 1]))) == \
    [pd.Timestamp('1950-11-01'), pd.Timestamp('1950-12-31'), pd.Timestamp('1951-01-01')]
# so the flow runs on smoothly from 31 October into 1 November (with Nov/Dec
# a year late the median day-to-day change there was 43 %, elsewhere ~5 %)
oct31, nov1 = q[(q.index.month == 10) & (q.index.day == 31)], q[(q.index.month == 11) & (q.index.day == 1)]
oct31.index = oct31.index + pd.Timedelta(days=1)
change = ((nov1 - oct31).abs() / oct31.abs().clip(lower=1e-3)).stack()
assert change.median() < 0.15, change.median()

ds = Portugal(path=raw_data_path, verbosity=3)

q = ds.fetch_q()

assert q.shape == (18628, 280)

assert isinstance(q.index, pd.DatetimeIndex)

# find columns in q dataframe which are all NaN
nan_cols = q.columns[q.isna().all()]

run_shared_tests(ds, 
             num_stations=280, 
             dyn_data_len=18628, 
             num_static_attrs=251,
              num_dyn_attrs=12
              )


## Slovenia

ds = Slovenia(path=raw_data_path, verbosity=3)

q = ds.fetch_q().loc[:"2023-12-31"]

assert q.shape == (27028, 117)

assert len(q.columns[q.isna().all()]) == 0, len(q.columns[q.isna().all()])

run_shared_tests(ds, 
             num_stations=117, 
             dyn_data_len=27210, 
             num_static_attrs=251,
              num_dyn_attrs=12
              )
