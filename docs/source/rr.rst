Rainfall Runoff datasets
*************************
This section include datasets which can be used for rainfall runoff modeling.
They all contain observed streamflow and meteological data as time series.
These are named as ``dynamic features``. The physical catchment properties
are included as ``static features``. Although each data source has a dedicated
class, however :py:class:`aqua_fetch.rr.RainfallRunoff` class can be used to access all the datasets.

List of datasets
================
.. list-table:: Stations per Source
   :widths: 10 15 10 10 30
   :header-rows: 1

   * - Source Name
     - Class
     - Number of Daily Stations
     - Number of Hourly Stations
     - Reference
   * - ``Arcticnet``
     - :py:class:`aqua_fetch.rr.Arcticnet`
     - 106
     - 
     - `R-Arcticnet <https://www.r-arcticnet.sr.unh.edu/v4.0/AllData/index.html>`_
   * - ``Bull``
     - :py:class:`aqua_fetch.Bull`
     - 484
     -
     - `Aparicio et al., 2024 <https://doi.org/10.1038/s41597-024-03594-5>`_
   * - ``CABra``
     - :py:class:`aqua_fetch.rr.CABra`
     - 735
     - 
     - `Almagro et al., 2021 <https://doi.org/10.5194/hess-25-3105-2021>`_ 
   * - ``CAMELS_AUS``
     - :py:class:`aqua_fetch.rr.CAMELS_AUS`
     - 222, 561
     - 
     - `Flower et al., 2021 <https://doi.org/10.5194/essd-13-3847-2021>`_
   * - ``CAMELS_GB``
     - :py:class:`aqua_fetch.rr.CAMELS_GB`
     - 671
     - 
     - `Coxon et al., 2020 <https://doi.org/10.5194/essd-12-2459-2020>`_
   * - ``CAMELS_BR``
     - :py:class:`aqua_fetch.rr.CAMELS_BR`
     - 897
     - 
     - `Chagas et al., 2020 <https://doi.org/10.5194/essd-12-2075-2020>`_
   * - ``CAMELS_CH``
     - :py:class:`aqua_fetch.rr.CAMELS_CH`
     - 331
     - 
     - `Hoege et al., 2023 <https://doi.org/10.5194/essd-15-5755-2023>`_
   * - ``CAMELS_CL``
     - :py:class:`aqua_fetch.rr.CAMELS_CL`
     - 516
     - 
     - `Alvarez-Garreton et al., 2018 <https://doi.org/10.5194/hess-22-5817-2018>`_
   * - ``CAMELS_DK``
     - :py:class:`aqua_fetch.rr.CAMELS_DK`
     - 304
     - 
     - `Liu et al., 2024 <https://doi.org/10.5194/essd-2024-292>`_
   * - ``CAMELS_DE``
     - :py:class:`aqua_fetch.rr.CAMELS_DE`
     - 1555
     - 
     - `Loritz et al., 2024 <https://essd.copernicus.org/preprints/essd-2024-318/>`_
   * - ``CAMELS_FR``
     - :py:class:`aqua_fetch.rr.CAMELS_FR`
     - 654
     - 
     - `Delaigue et al., 2024 <https://doi.org/10.5194/essd-2024-415>`_
   * - ``CAMELS_IND``
     - :py:class:`aqua_fetch.rr.CAMELS_IND`
     - 472
     -
     - `Mangukiya et al., 2024 <https://doi.org/10.5194/essd-2024-379>`_
   * - ``CAMELS_SE``
     - :py:class:`aqua_fetch.rr.CAMELS_SE`
     - 50
     -
     - `Teutschbein et al., 2024 <https://doi.org/10.1002/gdj3.239>`_
   * - ``CAMELS_US``
     - :py:class:`aqua_fetch.rr.CAMELS_US`
     - 671
     - 
     - `Newman et al., 2014 <https://gdex.ucar.edu/dataset/camels.html>`_
   * - ``Caravan_DK``
     - :py:class:`aqua_fetch.rr.Caravan_DK`
     - 304
     - 
     - `Koch 2022 <https://doi.org/10.5281/zenodo.7962379>`_     
   * - ``CCAM``
     - :py:class:`aqua_fetch.rr.CCAM`
     - 111
     -
     - `Hao et al., 2021 <https://doi.org/10.5194/essd-13-5591-2021>`_
   * - ``Finland``
     - :py:class:`aqua_fetch.rr.Finland`
     - 669
     -
     - `ymparisto.fi <https://wwwi3.ymparisto.fi>`_
   * - ``GRDCCaravan``
     - :py:class:`aqua_fetch.rr.GRDCCaravan`
     - 5357
     -
     - `Faerber et al., 2023 <https://zenodo.org/records/10074416>`_
   * - ``HYPE``
     - :py:class:`aqua_fetch.rr.HYPE`
     - 561
     - 
     - `Arciniega-Esparza and Birkel, 2020 <https://zenodo.org/records/4029572>`_     
   * - ``HYSETS``
     - :py:class:`aqua_fetch.rr.HYSETS`
     - 14425
     -
     - `Arsenault et al., 2020 <https://doi.org/10.1038/s41597-020-00583-2>`_
   * - ``Ireland``
     - :py:class:`aqua_fetch.rr.Ireland`
     - 464
     -
     - `EPA Ireland <https://epawebapp.epa.ie>`_  
   * - ``Italy``
     - :py:class:`aqua_fetch.rr.Italy`
     - 294
     -
     - `EPA Ireland <https://epawebapp.epa.ie>`_  
   * - ``Japan``
     - :py:class:`aqua_fetch.rr.Japan`
     - 751
     -
     - `river.go.jp <http://www1.river.go.jp>`_           
   * - ``LamaHCE``
     - :py:class:`aqua_fetch.rr.LamaHCE`
     - 859
     - 859
     - `Klingler et al., 2021 <https://doi.org/10.5194/essd-13-4529-2021>`_
   * - ``LamaHIce``
     - :py:class:`aqua_fetch.rr.LamaHIce`
     - 111
     -
     - `Helgason and Nijssen 2024 <https://doi.org/10.5194/essd-16-2741-2024>`_
   * - ``Poland``
     - :py:class:`aqua_fetch.rr.Poland`
     - 1287
     -
     - `imgw.pl <https://danepubliczne.imgw.pl>`_     
   * - ``Portugal``
     - :py:class:`aqua_fetch.rr.Portugal`
     - 280
     -
     - `snirh <https://snirh.apambiente.pt>`_       
   * - ``RRLuleaSweden``
     - :py:class:`aqua_fetch.RRLuleaSweden`
     - 1
     -
     - `Broekhuizen et al., 2020 <https://doi.org/10.5194/hess-24-869-2020>`_   
   * - ``Spain``
     - :py:class:`aqua_fetch.rr.Spain`
     - 889
     -
     - `ceh-flumen64 <https://ceh-flumen64.cedex.es>`_     
   * - ``Simbi``
     - :py:class:`aqua_fetch.rr.Simbi`
     - 24
     -
     - `Bathelemy et al., 2024 <https://doi.org/10.23708/02POK6>`_   
   * - ``Thailand``
     - :py:class:`aqua_fetch.rr.Thailand`
     - 73
     -
     - `RID project <https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/rid-river/disc_d.html>`_
   * - ``USGS``
     - :py:class:`aqua_fetch.rr.USGS`
     - 12004
     -
     - `USGS nwis <https://waterdata.usgs.gov/nwis/>`_
   * - ``WaterBenchIowa``
     - :py:class:`aqua_fetch.rr.WaterBenchIowa`
     - 125
     -
     - `Demir et al., 2022 <https://doi.org/10.5194/essd-14-5605-2022>`_


High Level API
==============
The high level API is provided by :py:class:`aqua_fetch.rr.RainfallRunoff` 
class to provide a unified and easy-to-use interface to access all the datasets. 
The datasets are accessed by their names.

.. autoclass:: aqua_fetch.rr.RainfallRunoff
   :members:
   :show-inheritance:

   .. automethod:: __init__


Low Level API
=============
The low level API provides access to each individual dataset classes.
This provides more control over the datasets.


.. autoclass:: aqua_fetch.rr.Camels
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.rr._gsha._GSHA
   :members:
   :show-inheritance:

   .. automethod:: __init__

.. autoclass:: aqua_fetch.rr._misc._EStreams
   :members:
   :show-inheritance:

   .. automethod:: __init__

.. autoclass:: aqua_fetch.Arcticnet
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.Bull
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.rr.CABra
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.rr.CAMELS_AUS
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.rr.CAMELS_BR
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.rr.CAMELS_CH
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.rr.CAMELS_CL
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.rr.CAMELS_GB
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.rr.CAMELS_US
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.rr.CAMELS_DE
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.rr.CAMELS_DK
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.rr.Caravan_DK
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.rr.CAMELS_FR
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.CAMELS_IND
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.rr.CAMELS_SE
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.rr.CCAM
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.rr.Finland
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.rr.GRDCCaravan
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.rr.HYSETS
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.rr.HYPE
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.Ireland
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.rr.Italy
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.Japan
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.rr.LamaHCE
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.rr.LamaHIce
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.rr.Poland
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.rr.Portugal
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.RRLuleaSweden
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.rr.Simbi
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.rr.Spain
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.Thailand
   :members:
   :show-inheritance:

   .. automethod:: __init__

.. autoclass:: aqua_fetch.USGS
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.rr.WaterBenchIowa
   :members:
   :show-inheritance:

   .. automethod:: __init__


The following datasets are very much similar to RainfallRunoff datasets,
but they do not have observed streamflow data. They are used
to provide static and dynamic features to other datasets.


.. autoclass:: aqua_fetch.GSHA
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.EStreams
   :members:
   :show-inheritance:

   .. automethod:: __init__
