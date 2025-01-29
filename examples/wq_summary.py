"""
===================================
Summary of water quality datasets
===================================
This file shows summary of all'multi-station' water quality datasets available in the package
and how to access these datasets.

At the time of running this script, the datasets have been previosly downloaded. Therefore,
if you run this script for the first time, it may take days to run or may even not
run successfully till the end due to internet connection issues. 

"""
import os
import site

wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath('__file__')))))
#wd_dir = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
#wd_dir = os.path.dirname(os.path.realpath('__file__'))
print(wd_dir)
site.addsitedir(wd_dir)

import matplotlib
nice_fonts = {
    #"text.usetex": True,
    "font.family": "sans-serif",  #sans -serif
    #"font.serif" : "Times New Roman",
}
matplotlib.rcParams.update(nice_fonts)

from easy_mpl.utils import despine_axes

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

from aqua_fetch import (
    SWatCh, 
    GRQA, 
    Quadica, 
    GRiMeDB, 
    RC4USCoast, 
    CamelsChem, 
    SyltRoads, 
    RiverChemSiberia,
    ecoli_mekong
)

from aqua_fetch.utils import print_info

# %%

print_info()

# %%

coords_data = {}
for ds in [SWatCh,
           # GRQA, 
           Quadica, 
           GRiMeDB, 
           RC4USCoast, 
           CamelsChem, 
           SyltRoads, ecoli_mekong, RiverChemSiberia
           ]:

    if ds == ecoli_mekong:
        df = ecoli_mekong(parameters=['lat', 'long', 'station_name']).set_index('station_name')
        # drop rows with duplicate index
        df = df[~df.index.duplicated(keep='first')]
        coords_data['ecoli_mekong'] = df[['lat', 'long']]
    else:
        ds = ds()
        coords = ds.stn_coords()
        coords_data[ds.name] = coords

# %%
colors = plt.cm.tab20.colors

block1 = ['SWatCh', 
          #'GRQA', 
          'Quadica', 
          'GRiMeDB', 
          'RC4USCoast', 
          'CamelsChem'
          ]
block2 = ['SyltRoads', 'ecoli_mekong', 'RiverChemSiberia']

# draw the figure
_, ax = plt.subplots(figsize=(10, 12))

map = Basemap(ax=ax, resolution='l',
              **{'llcrnrlat': -60, 'urcrnrlat': 78.0, 'llcrnrlon': -180.0, 'urcrnrlon': 180.0})
map.drawcoastlines(linewidth=0.3, ax=ax, color="gray", zorder=0)
#map.drawrivers(linewidth=0.1, ax=ax, color="blue", zorder=0)

s = 2

rets = {}
items = {}
for idx, (ds_name, coords) in enumerate(coords_data.items()):

    if ds_name == 'SyltRoads':
        s = 10

    ret = map.scatter(coords['long'].values, coords['lat'].values,
                marker=".",
                s=s,
                linewidths=0.0,
                color = colors[idx],
                alpha=1.0,
                label=f"{ds_name} (n={coords.shape[0]})")
    
    rets[ds_name] = ret
    items[ds_name] = coords.shape[0]

leg1 = ax.legend(
    [rets[src] for src in sorted(block1)],
    [f"{src} (n={items[src]})" for src in sorted(block1)],
    markerscale=12,
    fontsize=8,
    borderpad=0.2,
    labelspacing=0.5,
    title_fontproperties={'weight': 'bold', 'size': 8+2},
    bbox_to_anchor=(0.001, 0.05),
    loc="lower left",
    framealpha=0.6
    )

leg2 = ax.legend(
    [rets[src] for src in sorted(block2)],
    [f"{src} (n={items[src]})" for src in sorted(block2)],
    markerscale=5,
    fontsize=8,
    borderpad=0.2,
    labelspacing=0.5,
    title_fontproperties={'weight': 'bold', 'size': 8+2},
    bbox_to_anchor=(0.585, 0.05),
    loc="lower left",
    framealpha=0.6
    )

ax.add_artist(leg1)

despine_axes(ax)
# plt.savefig("wq_summary.png", dpi=600, bbox_inches="tight")
plt.show()

