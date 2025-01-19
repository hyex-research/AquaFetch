.. AquaFetch documentation master file, created by
   sphinx-quickstart on Sat Oct  1 11:53:01 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AquaFetch
==============
**A Unified Python Interface for Water Resource Dataset Acquisition and Harmonization**

AquaFetch is a Python package designed for the automated downloading, parsing, cleaning, and harmonization of freely available water resource datasets related to rainfall-runoff processes, surface water quality, and wastewater treatment. The package currently supports approximately 70 datasets, each containing between 1 to hundreds of parameters. It facilitates the downloading and transformation of raw data into consistent, easy-to-use, analysis-ready formats. This allows users to directly access and utilize the data without labor-intensive and time-consuming preprocessing.

The package comprises three submodules, each representing a different type of water resource data: `rr` for rainfall-runoff processes, `wq` for surface water quality, and `wwt` for wastewater treatment. The rr submodule offers data for 47,716 catchments worldwide, encompassing both dynamic and static features for each catchment. The dynamic features consist of observed streamflow and meteorological time series, averaged over the catchment area, available at daily or hourly time steps. Static features include constant parameters such as land use, soil, topography, and other physiographical characteristics, along with catchment boundaries. This submodule not only provides access to established rainfall-runoff datasets such as CAMELS and LamaH but also introduces new datasets compiled for the first time from publicly accessible online data sources. The `wq` submodule offers access to `16 surface water quality datasets <https://water-datasets.readthedocs.io/en/latest/wq.html#list-of-datasets>`_, each containing various water quality parameters measured across different spaces and times. The `wwt` submodule provides access to over 20,000 experimental measurements related to wastewater treatment techniques such as adsorption, photocatalysis, membrane filtration, and sonolysis.

The development of water-datasets was inspired by the growing availability of diverse water resource datasets in recent years. As a community-driven project, the codebase is structured to allow contributors to easily add new datasets, ensuring the package continues to expand and evolve to meet future needs.



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
