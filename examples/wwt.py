"""
=========================================
Summary of wastewater treatment datasets
=========================================
This file describes summary of datasets available in water_datasets package
for wastewater treatment. The datasets are divided into following categories:

1. Adsorption
2. Photocatalysis
3. Membrane processes
4. Sonolysis

"""

import os
import site

if __name__ == '__main__':
    wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath('__file__')))))
    # wd_dir = os.path.dirname(os.path.dirname(os.path.realpath('__file__')))
    #wd_dir = os.path.dirname(os.path.realpath('__file__'))  # for debugging
    print(wd_dir)
    site.addsitedir(wd_dir)

from aqua_fetch import (
    ec_removal_biochar,
    cr_removal,
    po4_removal_biochar,
    heavy_metal_removal,
    industrial_dye_removal,
    heavy_metal_removal_Shen,
    P_recovery,
    N_recovery,
    As_recovery,
    mg_degradation,
    dye_removal,
    dichlorophenoxyacetic_acid_removal,
    pms_removal,
    micropollutant_removal_osmosis,
    ion_transport_via_reverse_osmosis,
    cyanobacteria_disinfection
)


# %%
# Adsorption
# ----------

data, _ = ec_removal_biochar()
print(data.shape)

# %%
print(data.columns)

# %%
data, _ = cr_removal()
print(data.shape)

# %%

print(data.columns)

# %%
data, _ = po4_removal_biochar()
print(data.shape)

# %%

print(data.columns)

# %%
data, _ = heavy_metal_removal()
print(data.shape)

# %%

print(data.columns)

# %%
data, _ = industrial_dye_removal()
print(data.shape)

# %%

print(data.columns)

# %%
data, _ = heavy_metal_removal_Shen()
print(data.shape)

# %%

print(data.columns)

# %%
data, _ = P_recovery()
print(data.shape)

# %%

print(data.columns)

# %%
data, _ = N_recovery()
print(data.shape)

# %%

print(data.columns)

# %%
data, _ = As_recovery()
print(data.shape)

# %%

print(data.columns)


# %%
# Photocatalysis
# --------------

data, _ = mg_degradation()
print(data.shape)

# %%

print(data.columns)

# %%
data, _ = dye_removal()
print(data.shape)

# %%

print(data.columns)

# %%
data, _ = dichlorophenoxyacetic_acid_removal()
print(data.shape)

# %%

print(data.columns)

# %%
data, _ = pms_removal()
print(data.shape)

# %%

print(data.columns)

# %%
# Membrane processes
# ------------------

# data, _ = micropollutant_removal_osmosis()
# print(data.shape)

# # %%
# data, _ = ion_transport_via_reverse_osmosis()
# print(data.shape)

# %%
data, _ = cyanobacteria_disinfection()
print(data.shape)

# %%
print(data.columns)
