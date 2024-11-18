Rainfall Runoff datasets
*************************
This section include datasets which can be used for rainfall runoff modeling.
They all contain observed streamflow and meteological data as time series.
These are named as ``dynamic features``. The physical catchment properties
are included as ``static features``. Although each data source has a dedicated
class, however :py:class:`water_datasets.rr.RainfallRunoff` class can be used to access all the datasets.


.. list-table:: Stations per Source
   :widths: 10 15 10 10 30
   :header-rows: 1

   * - Source Name
     - Class
     - Number of Daily Stations
     - Number of Hourly Stations
     - Reference
   * - ``Arcticnet``
     - :py:class:`water_datasets.rr.Arcticnet`
     - 106
     - 
     - `R-Arcticnet <https://www.r-arcticnet.sr.unh.edu/v4.0/AllData/index.html>`_     
   * - ``CAMELS_AUS``
     - :py:class:`water_datasets.rr.CAMELS_AUS`
     - 222, 561
     - 
     - `Flower et al., 2021 <https://doi.org/10.5194/essd-13-3847-2021>`_
   * - ``CAMELS_GB``
     - :py:class:`water_datasets.rr.CAMELS_GB`
     - 671
     - 
     - `Coxon et al., 2020 <https://doi.org/10.5194/essd-12-2459-2020>`_
   * - ``CAMELS_BR``
     - :py:class:`water_datasets.rr.CAMELS_BR`
     - 897
     - 
     - `Chagas et al., 2020 <https://doi.org/10.5194/essd-12-2075-2020>`_
   * - ``CAMELS_US``
     - :py:class:`water_datasets.rr.CAMELS_US`
     - 671
     - 
     - `Newman et al., 2014 <https://gdex.ucar.edu/dataset/camels.html>`_
   * - ``CAMELS_CL``
     - :py:class:`water_datasets.rr.CAMELS_CL`
     - 516
     - 
     - `Alvarez-Garreton et al., 2018 <https://doi.org/10.5194/hess-22-5817-2018>`_
   * - ``CAMELS_DK``
     - :py:class:`water_datasets.rr.CAMELS_DK`
     - 304
     - 
     - `Liu et al., 2024 <https://doi.org/10.5194/essd-2024-292>`_
   * - ``CAMELS_CH``
     - :py:class:`water_datasets.rr.CAMELS_CH`
     - 331
     - 
     - `Hoege et al., 2023 <https://doi.org/10.5194/essd-15-5755-2023>`_
   * - ``CAMELS_DE``
     - :py:class:`water_datasets.rr.CAMELS_DE`
     - 1555
     - 
     - `Loritz et al., 2024 <https://essd.copernicus.org/preprints/essd-2024-318/>`_
   * - ``CAMELS_SE``
     - :py:class:`water_datasets.rr.CAMELS_SE`
     - 50
     -
     - `Teutschbein et al., 2024 <https://doi.org/10.1002/gdj3.239>`_
   * - ``LamaHCE``
     - :py:class:`water_datasets.rr.LamaHCE`
     - 859
     - 859
     - `Klingler et al., 2021 <https://doi.org/10.5194/essd-13-4529-2021>`_
   * - ``LamaHIce``
     - :py:class:`water_datasets.rr.LamaHIce`
     - 111
     -
     - `Helgason and Nijssen 2024 <https://doi.org/10.5194/essd-16-2741-2024>`_
   * - ``HYSETS``
     - :py:class:`water_datasets.rr.HYSETS`
     - 14425
     -
     - `Arsenault et al., 2020 <https://doi.org/10.1038/s41597-020-00583-2>`_
   * - ``GRDCCaravan``
     - :py:class:`water_datasets.rr.GRDCCaravan`
     - 5357
     -
     - `Faerber et al., 2023 <https://zenodo.org/records/10074416>`_
   * - ``Bull``
     - :py:class:`water_datasets.Bull`
     - 484
     -
     - `Aparicio et al., 2024 <https://doi.org/10.1038/s41597-024-03594-5>`_    
   * - ``WaterBenchIowa``
     - :py:class:`water_datasets.rr.WaterBenchIowa`
     - 125
     -
     - `Demir et al., 2022 <https://doi.org/10.5194/essd-14-5605-2022>`_
   * - ``CCAM``
     - :py:class:`water_datasets.rr.CCAM`
     - 111
     -
     - `Hao et al., 2021 <https://doi.org/10.5194/essd-13-5591-2021>`_
   * - ``RRLuleaSweden``
     - :py:class:`water_datasets.RRLuleaSweden`
     - 1
     -
     - `Broekhuizen et al., 2020 <https://doi.org/10.5194/hess-24-869-2020>`_
   * - ``CABra``
     - :py:class:`water_datasets.rr.CABra`
     - 735
     - 
     - `Almagro et al., 2021 <https://doi.org/10.5194/hess-25-3105-2021>`_ 
   * - ``HYPE``
     - :py:class:`water_datasets.rr.HYPE`
     - 561
     - 
     - `Arciniega-Esparza and Birkel, 2020 <https://zenodo.org/records/4029572>`_
   * - ``Simbi``
     - :py:class:`water_datasets.rr.Simbi`
     - 24
     -
     - `Bathelemy et al., 2024 <doi: 10.5194/essd-16-2073-2024>`_
   * - ``CAMELS_IND``
     - :py:class:`water_datasets.rr.CAMELS_IND`
     - 472
     -
     - `Mangukiya et al., 2024 <https://doi.org/10.5194/essd-2024-379>`_
   * - ``USGS``
     - :py:class:`water_datasets.rr.USGS`
     - 12004
     -
     - `USGS nwis <https://waterdata.usgs.gov/nwis/>`_


High Level API
==============
It consists of a unified interface to access all the datasets. The datasets are accessed by their names.

.. autoclass:: water_datasets.rr.RainfallRunoff
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__


Low Level API
=============
The datasets can be accessed individually by their names.


.. autoclass:: water_datasets.rr.Camels
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.CAMELS_AUS
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.CAMELS_GB
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.CAMELS_BR
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.CAMELS_US
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.CAMELS_CL
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.CAMELS_DK
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.LamaHCE
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.LamaHIce
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.HYSETS
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.HYPE
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.CCAM
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.CABra
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.CAMELS_CH
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.CAMELS_DE
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.GRDCCaravan
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.CAMELS_SE
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.WaterBenchIowa
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr.Simbi
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.RRLuleaSweden
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.Bull
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.CAMELS_IND
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.USGS
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.rr._denmark.CAMELS_DK
   :members:
   :show-inheritance:

   .. automethod:: __init__
