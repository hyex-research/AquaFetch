
import os
import site
# add the parent directory in the path
wd_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(wd_dir)

import fiona
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from easy_mpl.utils import despine_axes
from mpl_toolkits.basemap import Basemap

from pyproj import Transformer
from shapely.geometry import shape

from aqua_fetch import RainfallRunoff, Quadica

from aqua_fetch._geom_utils import calc_centroid
from aqua_fetch._geom_utils import utm_to_lat_lon, laea_to_wgs84, lcc_to_wgs84


DATA_PATH = '/mnt/datawaha/hyex/atr/gscad_database/raw'


def test_calc_centroid():

    stns_file = os.path.join(
        ds.path, 
        "Boundary data", "Boundary data", 
        "Basin_boundary.shp")

    # Read Basin_boundary.shp
    with fiona.open(stns_file) as shp:
        for _, feature in enumerate(shp):
            # Convert fiona geometry to shapely geometry
            geometry = feature['geometry']

            polygon = shape(feature['geometry'])
            centroid = polygon.centroid

            shp_cent = np.array(centroid.coords.xy).T[0]

            centroid1 = calc_centroid(geometry)
            est_cent = np.array(centroid1)

            np.testing.assert_allclose(est_cent, shp_cent, rtol=1e-5, atol=1e-5)
    return


def test_25832_to_4326():

    ds = RainfallRunoff(
        "CAMELS_DK", 
        path=os.path.join(DATA_PATH, 'CAMELS'), 
        verbosity=3)

    c = ds.fetch_static_features(static_features=['catch_outlet_lat', 'catch_outlet_lon'])

    transformer = Transformer.from_crs("EPSG:25832", "EPSG:4326")
    lat, long = transformer.transform(c.iloc[:, 1], c.iloc[:, 0])
    ct = pd.DataFrame(np.column_stack([lat, long]), index=c.index,
                                        columns=['lat', 'long'])

    ct_m = pd.DataFrame(columns=['lat', 'long'], index=ct.index)
    # Test the function using lat, long in c DataFrame
    for i in range(0, len(c)):
        lat, lon = utm_to_lat_lon(c.iloc[i, 1], c.iloc[i, 0], 32)
        ct_m.iloc[i] = [lat, lon]

    np.testing.assert_allclose(ct.values, ct_m.values.astype(float), atol=1e-5)

    return


def test_laea_to_wgs84():
    ds = Quadica(
        path='/mnt/datawaha/hyex/atr/data', 
        verbosity=3
        )

    stn_coords = ds._stn_coords()

    transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326")

    lat, long = transformer.transform(stn_coords.loc[:, 'lat'], stn_coords.loc[:, 'long'])
    coord = pd.DataFrame(np.column_stack([lat, long]), index=stn_coords.index,
                            columns=['lat', 'long'])

    false_easting, false_northing = 4321000.0, 3210000.0
    lat_0, lon_0 = 52, 10
    x, y = laea_to_wgs84(stn_coords.loc[:, 'long'], stn_coords.loc[:, 'lat'], lon_0, lat_0, false_easting, false_northing)

    coord_m = pd.concat([x, y], axis=1)
    coord_m.columns = ['lat', 'long']

    # todo : atol value is too high, need to reduce it
    np.testing.assert_allclose(coord.values, coord_m.values, atol=1e-1)

    return


def plot_test_25832_to_4326():
    ds = RainfallRunoff(
        "CAMELS_FR", 
        path=os.path.join(DATA_PATH, 'CAMELS'), 
        verbosity=3)

    c = ds.stn_coords()

    coords_data = {
        "CAMELS_FR": c,
    }

    colors = plt.cm.tab20.colors + plt.cm.tab20b.colors

    rets = {}
    items = {}

    # draw the figure
    _, ax = plt.subplots(figsize=(10, 12))

    map = Basemap(ax=ax, resolution='l', 
                **{'llcrnrlat': -40, 'urcrnrlat': 78.0, 'llcrnrlon': -10.0, 'urcrnrlon': 50.0})
    map.drawcoastlines(linewidth=0.3, ax=ax, color="gray", zorder=0)
    #map.drawcounties(linewidth=0.3, ax=ax, color="gray", zorder=0)
    short = ['CAMELS_FR', #'CAMELS_DE', 'CAMELS_DK'
            ]
    for idx, src in enumerate(short):

        coords = coords_data[src]

        ret = map.scatter(coords['long'].values, coords['lat'].values, 
                    marker=".", 
                    s=2, 
                    linewidths=0.0,
                    color = colors[idx],
                    alpha=1.0,
                    label=f"{src} (n={coords.shape[0]})")
        
        rets[src] = ret
        items[src] = coords.shape[0]

    leg2 = ax.legend([rets[src] for src in short], 
                    [f"{src} (n={items[src]})" for src in short], 
            markerscale=12,
            fontsize=8,
            borderpad=0.2,
            labelspacing=0.5,
            title="Datasets",  
            title_fontproperties={'weight': 'bold', 'size': 8+2},
            bbox_to_anchor=(0.34, 0.33))
    ax.add_artist(leg2)

    despine_axes(ax)

    plt.show()

    return


def test_lcc_to_wgs84():
    coords = pd.DataFrame({
        'lat': [397683.0, 383164.0, 359057.0, 451317.0, 621984.0],
        'long': [403449.0, 401601.0, 551036.0, 370531.0, 375499.0]
    })

    transformer = Transformer.from_crs("EPSG:3057", "EPSG:4326")
    lat_true, long_true = transformer.transform(coords.loc[:, 'long'], coords.loc[:, 'lat'])

    # Parameters for EPSG:3057
    lon_0 = -19.0        # Central Meridian
    lat_0 = 65.0         # Latitude of Origin
    lat_1 = 64.25        # First standard parallel
    lat_2 = 65.75        # Second standard parallel
    false_easting = 500000.0
    false_northing = 500000.0

    lat, lon = lcc_to_wgs84(
        coords['long'].values, coords['lat'].values, 
        lon_0, lat_0, lat_1, lat_2, false_easting, 
        false_northing)

    np.testing.assert_allclose(lat, lat_true, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(lon, long_true, rtol=1e-5, atol=1e-5)

    return


test_25832_to_4326()

test_laea_to_wgs84()

test_lcc_to_wgs84()
