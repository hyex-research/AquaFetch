.. water-datasets documentation master file, created by
   sphinx-quickstart on Sat Oct  1 11:53:01 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

water-datasets
==============
**A Unified Python Interface for Water Resource Data Acquisition**

The water-datasets is a Python package designed for the automated downloading, parsing, and cleaning of water resources datasets.
It allows the users to directly utilize these datasets after saving them locally as Comma Separated Values (CSV) or netCDF files. 
It provides users with analysis-ready data by acquiring open-source data from the web and minimizing the preprocessing steps required for this purpose. 
Therefore, it bridges the gap between online water resources data and Python, facilitating access in as few steps as possible.
The package comprises three submodules, each representing a different type of water 
resource data: `rr` for rainfall-runoff processes, `wq` for surface water quality, 
and `wwt` for wastewater treatment. The rr submodule offers data for 47,716 catchments 
worldwide, encompassing both dynamic and static features for each catchment. The 
dynamic features consist of observed streamflow and meteorological time series, 
averaged over the catchment area, and available at daily or hourly timesteps. 
Static features include constant parameters such as land use, soil, topography, 
and other physiographical characteristics, along with catchment boundaries. 
This submodule not only provides access to established rainfall-runoff datasets 
but also introduces new datasets compiled for the first time from open-source web 
data. The `wq` submodule offers access to 12 surface water quality datasets, each 
containing various water quality parameters measured across different spaces and 
times. Meanwhile, the `wwt` submodule provides access to over 20,000 experimental 
data points for wastewater treatment techniques such as adsorption, photocatalysis, 
membrane filtration, and sonolysis.


.. toctree::
   :maxdepth: 2

   installation
   quickstart


.. toctree::
   :maxdepth: 2
   :caption: API

   rr
   wq
   wwt
   misc


.. toctree::
   :maxdepth: 2
   :caption: Examples

   auto_examples/index
   _notebooks/main


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
