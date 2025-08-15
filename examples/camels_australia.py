"""
====================
CAMELS Australia
====================

.. currentmodule:: aqua_fetch

This example demonstrates how to use the `aqua_fetch` package to download and
explore the `CAMELS Australia <https://doi.org/10.5194/essd-2024-263>`_ dataset 
using the :py:class:`aqua_fetch.RainfallRunoff` class. Although we show it
for CAMELS Australia, the same can be done for all other rainfall runoff datasets.

**Note:** This file runs online on readthedocs everytime the documentation is built.
The server to download the CAMELS_AUS data is sometimes down and gives `HTTPError: HTTP Error 500: Internal Server Error`.

"""

import os
import site

if __name__ == '__main__':
    wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath('__file__')))))
    #wd_dir = os.path.dirname(os.path.realpath('__file__'))
    #wd_dir = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
    print(wd_dir)
    site.addsitedir(wd_dir)

from tabulight import EDA
import matplotlib.pyplot as plt
from easy_mpl import scatter, hist
from easy_mpl.utils import process_cbar
from aqua_fetch import RainfallRunoff
from aqua_fetch.utils import print_info

# %%

print_info()

# %%

dataset = RainfallRunoff('CAMELS_AUS', version=1, 
                         #overwrite=True,
                         #path='/mnt/datawaha/hyex/atr/gscad_database/raw/CAMELS_AUS_V1'
                         )

# %%
dataset.start

# %%
dataset.end

# %%

stations = dataset.stations()
len(stations)

# %%

stations[0:10]

# %%
# Static Features
# ---------------

dataset.static_features

# %%

len(dataset.static_features)

# %%

mrvbf = 'proportion of catchment occupied by classes of MultiResolution Valley Bottom Flatness'
lc01 = 'land cover codes'
nvis = 'vegetation sub-groups'
anngro = 'Average annual growth index value for some plants'
gromega = 'Seasonality of growth index value'
npp = 'net primary productivity'


# %%

static = dataset.fetch_static_features(stations=stations)
static.shape

# %%

# EDA(data=static, save=False).heatmap()

# %%

physical_features = []
soil_features = []
geological_features = []
flow_characteristics = []

static = static.dropna(axis=1)
static.shape

# %%
coords = dataset.stn_coords()
coords

# %%

dataset.plot_stations(color='area_km2')

# %%

dataset.plot_catchment(dataset.area().sort_values(ascending=False).index[0], show_outlet=True)

# %%

lat = coords['lat'].astype(float).values.reshape(-1,)
long = coords['long'].astype(float).values.reshape(-1,)

# %%

idx = 0
ax_num = 0

fig, axes = plt.subplots(5, 5, figsize=(15, 12))
axes = axes.flatten()

while ax_num < 25:

    val = static.iloc[:, idx]
    idx += 1

    try:
        c = val.astype(float).values.reshape(-1,)

        en = 222
        ax = axes[ax_num]
        ax, sc = scatter(long[0:en], lat[0:en], c=c[0:en], cmap="hot", show=False, ax=ax)

        process_cbar(ax, sc, border=False, title=val.name, #title_kws ={"fontsize": 14}
                    )
        ax_num += 1
    except ValueError:
        continue

plt.tight_layout()
plt.show()
print(idx)

# %%

idx = 32
ax_num = 0

fig, axes = plt.subplots(5, 5, figsize=(15, 12))
axes = axes.flatten()

while ax_num < 25:

    val = static.iloc[:, idx]
    idx += 1

    try:
        c = val.astype(float).values.reshape(-1,)

        en = 222
        ax = axes[ax_num]
        ax, sc = scatter(long[0:en], lat[0:en], c=c[0:en], cmap="hot", show=False, ax=ax)

        process_cbar(ax, sc, border=False, title=val.name, #title_kws ={"fontsize": 14}
                    )
        ax_num += 1
    except ValueError:
        continue

plt.tight_layout()
plt.show()
print(idx)

# %%

idx = 59
ax_num = 0

fig, axes = plt.subplots(5, 5, figsize=(15, 12))
axes = axes.flatten()

while ax_num < 25:

    val = static.iloc[:, idx]
    idx += 1

    try:
        c = val.astype(float).values.reshape(-1,)

        en = 222
        ax = axes[ax_num]
        ax, sc = scatter(long[0:en], lat[0:en], c=c[0:en], cmap="hot", show=False, ax=ax)

        process_cbar(ax, sc, border=False, title=val.name, #title_kws ={"fontsize": 14}
                    )
        ax_num += 1
    except ValueError:
        continue

plt.tight_layout()
plt.show()
print(idx)


# %%
# Dyanmic Features
# ----------------
dataset.dynamic_features

# %%
# Streamflow
# ==================
streamflow = dataset.q_mmd()

streamflow.shape

# %%
streamflow

# %%

# EDA(data=streamflow, save=False).heatmap()

# %%

fig, axes = plt.subplots(7, 7, figsize=(10, 10), sharey="all")

for idx, ax in enumerate(axes.flat):

    hist(streamflow.iloc[:, idx].values.reshape(-1,),
         bins=20,
         ax=ax,
         show=False
        )

plt.show()

# %%

_ = hist(streamflow.skew().values.reshape(-1,), bins=50)

# %%
_, dynamic = dataset.fetch(stations=1, as_dataframe=True)
print(len(dynamic))
df = dynamic.popitem()[1] # the key in dynamic is the station name and value is stn dynamic data as DataFrame
df.shape

# %%
df

# %%

# get name of all stations as list
stns = dataset.stations()
len(stns)

# %%
# get data of 10 % of stations as dataframe
_, dynamic = dataset.fetch(0.1, as_dataframe=True)
len(dynamic)  # remember this is a dictionary of dataframes

# %%
# The keys in dynamic dictionary are names of stations
dynamic.keys()

# %%
# get data by station id
dynamic = dataset.fetch(stations='224214A', as_dataframe=True)[1]
dynamic['224214A'].shape

# %%

dynamic['224214A']

# %%
# get names of available dynamic features
dataset.dynamic_features
# get only selected dynamic features
dynamic = dataset.fetch(1, as_dataframe=True,
dynamic_features=['airtemp_C_awap_max', 'pcp_mm_awap', 'aet_mm_silo_morton', 'q_cms_obs'])[1]
print(type(dynamic))
data = dynamic.popitem()[1]
data.shape

# %%

data

# %%

# get names of available static features
dataset.static_features
# get data of 10 random stations
_, dynamic = dataset.fetch(10, as_dataframe=True)
len(dynamic)  # remember this is a dictioanry of dataframes

# %%

# static data is always a pandas DataFrame while dynamic is a dictionary of dataframes
# with keys as station names.
static, dynamic = dataset.fetch(stations='224214A', static_features="all", as_dataframe=True)
static.shape, dynamic['224214A'].shape

# %%
static

# %%

dynamic

# %%
# get data data of all stations as xarray dataset
static, dynamic = dataset.fetch()
static

# %%

dynamic
