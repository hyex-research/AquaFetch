import os
import site

# add the parent directory in the path
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

import unittest


from aqua_fetch import Oligotrend

ds = Oligotrend(path='/mnt/datawaha/hyex/atr/data')

# assert len(ds.parameters()) == 17, "Number of parameters should be 17"

# assert len(ds.stations()) == 1846, "Number of stations should be 1846"

# assert len(ds.lakes()) == 685, "Number of lakes should be 685"

# assert len(ds.rivers()) == 924, "Number of rivers should be 924"

# assert len(ds.estuaries()) == 237, "Number of estuaries should be 237"

# df = ds.gis_data()

# assert df.shape == (1846, 4), "GIS data shape should be (1846, 4)"

# l1_data = ds.l1_data()

# assert l1_data.shape == (5056630, 7), "L1 data shape should be (5056630, 7)"

# stn_df = ds.fetch_stn_parameters('lake_atlanticoceanseaboard_usa12721')

# assert stn_df.shape == (303, 3), "Station data shape should be (303, 3)"

# ds.get_stations('chla')

# data = ds.fetch_stns_parameters(['river_ebro_9027', 'river_elbe_elbe_10'])
# assert data['river_ebro_9027'].shape == (287, 8)
# assert data['river_elbe_elbe_10'].shape == (8154, 12)

# data1 = ds.fetch_stns_parameters(['river_ebro_9027', 'river_elbe_elbe_10'],
#                                 parameters=['chla'])
# assert data1['river_ebro_9027'].shape == (177, 1)
# assert data1['river_elbe_elbe_10'].shape == (413, 1)

chla_counts = ds.num_obs('chla')

assert len(chla_counts) == 1846
