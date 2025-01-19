"""
=========================
Quadica dataset
=========================
"""

# sphinx_gallery_thumbnail_number = 3

import os
import site

if __name__ == '__main__':
    wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath('__file__')))))
    # wd_dir = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
    print(wd_dir)
    site.addsitedir(wd_dir)

import pandas as pd
import matplotlib.pyplot as plt
from easy_mpl import hist, ridge
from easy_mpl.utils import create_subplots

from aqua_fetch import Quadica
from aqua_fetch.utils import print_info
# %%

print_info()

# %%

dataset = Quadica()

# %%

avg_temp = dataset.avg_temp()
print(avg_temp.shape)

# %%

avg_temp.head()

# %%
# pet
# ----

pet = dataset.pet()
print(pet.shape)

# %%
# precipitation
# -------------
pcp = dataset.precipitation()
print(pcp.shape)

# %%
# monthly median values
# -------------------------
mon_medians = dataset.monthly_medians()
print(mon_medians.shape)

# %%

mon_medians.head()

# %%
wrtds_mon = dataset.wrtds_monthly()
print(wrtds_mon.shape)

# %%
# catchment attributes
# ---------------------
cat_attrs = dataset.catchment_attributes()
print(cat_attrs.shape)

# %%
print(cat_attrs.columns)

# %%
dataset.catchment_attributes(stations=[1,2,3])

# %%
# monthly data
# ------------
dyn, cat = dataset.fetch_monthly(max_nan_tol=None)
print(dyn.shape)

# %%
dyn['OBJECTID'].unique()

# %%

print(dyn.columns)

# %%
print(dyn.isna().sum())

# %%
print(cat.shape)

# %%
# monthly TN
# -----------
dyn, cat = dataset.fetch_monthly(features="TN", max_nan_tol=0)
print(dyn.shape)

# %%
dyn.head()

# %%
dyn.tail()

# %%
print(dyn.isna().sum())

# %%
dyn['OBJECTID'].unique()

# %%
print(len(dyn['OBJECTID'].unique()))

# %%
print(cat.shape)
# %%

df = pd.concat([grp['median_C_TN'] for idx,grp in dyn.groupby('OBJECTID')], axis=1)
df.columns = dyn['OBJECTID'].unique()
ridge(df, figsize=(10, 10), color="GnBu", title="median_C_TN")


# %%
# monthly TP
# ------------
dyn, cat = dataset.fetch_monthly(features="TP", max_nan_tol=0)
print(dyn.shape)

# %%
dyn['OBJECTID'].unique()

# %%
print(len(dyn['OBJECTID'].unique()))

# %%
dyn.head()

# %%
dyn.tail()

# %%
print(dyn.isna().sum())

# %%
print(cat.shape)

# %%
# monthly TOC
# ------------

dyn, cat = dataset.fetch_monthly(features="TOC", max_nan_tol=0)
print(dyn.shape)

# %%
dyn['OBJECTID'].unique()

# %%
print(len(dyn['OBJECTID'].unique()))

grouper = dyn.groupby("OBJECTID")



fig, axes = create_subplots(grouper.ngroups, figsize=(12, 10))
for (idx, grp), ax in zip(grouper, axes.flat):
    hist(grp['median_C_TOC'], ax=ax, show=False, ax_kws=dict(title=idx))
plt.show()

# %%

df = pd.concat([grp['median_C_TOC'] for idx,grp in dyn.groupby('OBJECTID')], axis=1)
df.columns = dyn['OBJECTID'].unique()

ridge(df, figsize=(10, 10), color="GnBu", title="median_C_TOC")

# %%
dyn.head()

# %%
dyn.tail()

# %%
print(dyn.isna().sum())

# %%
print(cat.shape)

# %%
# monthly DOC
# ------------
dyn, cat = dataset.fetch_monthly(features="DOC", max_nan_tol=0)
print(dyn.shape)

# %%
dyn['OBJECTID'].unique()

# %%
print(len(dyn['OBJECTID'].unique()))

# %%
dyn.head()

# %%
dyn.tail()

# %%
print(dyn.isna().sum())

# %%
print(cat.shape)

# %%

