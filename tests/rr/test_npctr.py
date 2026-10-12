
import os
import site   # so that AquaFetch directory is in path
import logging

# add the parent directory in the path
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

if __name__ == "__main__":
    logging.basicConfig(filename='test_npctr.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from utils import test_dataset as run_shared_tests

import numpy as np
import pandas as pd

from aqua_fetch.rr import NPCTRCatchments
from aqua_fetch._geom_utils import web_mercator_to_wgs84
from aqua_fetch._backend import fiona


def check_web_mercator_known_points():
    """the origin, the antimeridian and the poles-side limit of EPSG:3857"""
    a = 6378137.0
    lat, lon = web_mercator_to_wgs84([0.0, np.pi * a, -np.pi * a, 0.0],
                                     [0.0, 0.0, 0.0, np.pi * a])
    np.testing.assert_allclose(lon, [0.0, 180.0, -180.0, 0.0], atol=1e-12)
    # the northing of pi*a is the edge of the square Web Mercator world
    np.testing.assert_allclose(lat, [0.0, 0.0, 0.0, 85.0511287798066], atol=1e-10)
    return


def check_boundaries_in_wgs84(dataset):
    """every raw EPSG:3857 vertex converts as pyproj does (when pyproj is
    installed), and every gauge lies inside the lon/lat bounding box of its own
    catchment"""
    try:
        import pyproj
    except ImportError:
        pyproj = None

    if pyproj is not None:
        with fiona.open(dataset.boundary_file) as src:
            assert src.crs.to_epsg() == 3857, src.crs
            xy = []
            for feat in src:
                g = feat['geometry']
                polys = g['coordinates'] if g['type'] == 'MultiPolygon' else [g['coordinates']]
                xy += [np.asarray(ring, dtype=float)[:, :2] for poly in polys for ring in poly]
        xy = np.concatenate(xy)
        lat, lon = web_mercator_to_wgs84(xy[:, 0], xy[:, 1])
        plon, plat = pyproj.Transformer.from_crs(3857, 4326, always_xy=True).transform(xy[:, 0], xy[:, 1])
        assert np.abs(lon - plon).max() < 1e-9 and np.abs(lat - plat).max() < 1e-9

    coords = dataset.stn_coords()
    for stn in dataset.stations():
        geom = dataset.get_boundary(stn)
        polys = geom.coordinates if geom.type == 'MultiPolygon' else [geom.coordinates]
        pts = np.concatenate([np.asarray(ring, dtype=float) for poly in polys for ring in poly])
        glat, glon = coords.loc[stn, 'lat'], coords.loc[stn, 'long']
        # 0.005 deg (~500 m) slack: the gauge sits at the outlet, on the edge
        assert pts[:, 0].min() - 0.005 <= glon <= pts[:, 0].max() + 0.005, (stn, glon)
        assert pts[:, 1].min() - 0.005 <= glat <= pts[:, 1].max() + 0.005, (stn, glat)
    return


# Table A1 of Korver et al. (2022), https://doi.org/10.5194/essd-14-4231-2022
PAPER_COORDS = {
    'RefStn': (51.6520, -128.1287), 'SSN626': (51.6408, -128.1219),
    'WSN626': (51.6262, -128.1018), 'SSN693': (51.6442, -127.9978),
    'WSN693_703': (51.6106, -127.9871), 'SSN703': (51.6466, -128.0257),
    'WSN703': (51.6433, -128.0228), 'WSN703_708': (51.6222, -128.0507),
    'SSN708': (51.6486, -128.0684), 'SSN819': (51.6619, -128.0419),
    'WSN819_1015': (51.6827, -128.0433), 'SSN844': (51.6608, -128.0025),
    'WSN844': (51.6614, -127.9975), 'SSN1015': (51.6906, -128.0653),
    'East Buxton': (51.5899, -127.9752), 'Hecate': (51.6826, -128.0228),
}


def check_coords_match_paper(dataset):
    coords = dataset.all_stn_coords()
    for stn, lat_lon in PAPER_COORDS.items():
        assert tuple(coords.loc[stn]) == lat_lon, (stn, tuple(coords.loc[stn]))
    return


def check_commented_flags_are_kept(dataset):
    """a flag followed by a comment, e.g. "AV: QC'd by EH", is still AV"""
    raw = pd.read_csv(os.path.join(dataset.path, '2013-2019_Discharge_Hourly.csv'),
                      usecols=['Qflag', 'Qrate', 'Qvol', 'Qmm'])
    raw = raw.dropna(subset=['Qrate', 'Qvol', 'Qmm'], how='all')
    expected = raw['Qflag'].str.match(r'\s*(AV|EV)\s*(:|$)', na=False).sum()
    served = sum(len(q) for q in dataset.read_hourly_q().values())
    assert served == expected, (served, expected)
    return


check_web_mercator_known_points()

ds = NPCTRCatchments(path='/mnt/datawaha/hyex/atr/data', verbosity=4)

check_coords_match_paper(ds)
check_commented_flags_are_kept(ds)
check_boundaries_in_wgs84(ds)

run_shared_tests(ds, 7, 53072, 14, 14, st="20140101", en="20141231")


sh_q = ds.read_5min_q()

for k,v in sh_q.items():
    assert k in ds.stations()
    assert isinstance(v, pd.DataFrame)
    assert isinstance(v.index, pd.DatetimeIndex)


ds = NPCTRCatchments(path='/mnt/datawaha/hyex/atr/data', timestep='5min')

pcp_5m = ds.read_pcp()
rh_5m = ds.read_rel_hum()
#temp_5m = ds.read_temp()
solrad_5m = ds.read_sol_rad()
ws_5m = ds.read_wind_speed()
snowdepth_5m = ds.read_snow_depth()
winddir_5m = ds.read_wind_dir()


