
import os
import site   # so that water_datasets directory is in path
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

from water_datasets import GRiMeDB


ds = GRiMeDB(
    path='/mnt/datawaha/hyex/atr/data',
    verbosity=4
    )

conc = ds.concentrations()
assert conc.shape == (25052, 59)
assert conc['Site_ID'].nunique() == 4964
conc = ds.concentrations(parameters=['Q', 'NO3', 'NH4', 'TN', 'SRP', 'TP', 'DOC'])
assert conc.shape == (25052, 7)

conc = ds.concentrations(streams=['Indus River'])
assert conc.shape == (2, 59)

conc = ds.concentrations(streams=['Ganges'])
assert conc.shape == (11, 59)

conc = ds.concentrations(streams=['Brahmaputra'])
assert conc.shape == (2, 59)

conc = ds.concentrations(streams=['Han River', 'North Han', 'South Han', 'Lower Han River'])
assert conc.shape == (8, 59)

fluxes = ds.fluxes()
assert fluxes.shape == (7298, 52)
assert fluxes['Site_ID'].nunique() == 1903

sites = ds.sites()
sites['Site_ID'].nunique() == 5029
sites['Stream_Name'].nunique() == 2722

sources = ds.sources()
assert sources.shape == (298, 10)
