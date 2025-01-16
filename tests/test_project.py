import os
import site

# add the parent directory in the path
wd_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
site.addsitedir(wd_dir)

import pandas as pd
import numpy as np

from pyproj import Transformer

from water_datasets._project import utm_to_lat_lon
from water_datasets import RainfallRunoff

DATA_PATH = '/mnt/datawaha/hyex/atr/gscad_database/raw'


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


# test_25832_to_4326()


import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

from easy_mpl.utils import despine_axes

from water_datasets import RainfallRunoff

DATA_PATH = '/mnt/datawaha/hyex/atr/gscad_database/raw'


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