Rainfall Runoff datasets
*************************
This section include datasets which can be used for rainfall runoff modeling.
They all contain observed streamflow and meteological data as time series.
These are named as ``dynamic features``. The physical catchment properties
are included as ``static features`` as tabular data, where each row corresponds 
to one catchment and each column to one static feature.

In addition to published datasets, this package introduces 10 new datasets for 
rainfall-runoff modeling. These datasets have not yet been published but follow 
the CAMELS dataset series convention. They include Ireland, Finland, Italy, Poland, 
Portugal, Japan, Thailand, Arcticnet, Spain, and the USGS. The observed streamflow 
data are sourced from the national meteorological or hydrological websites of the 
respective countries. Catchment boundaries and meteorological data for Ireland, 
Finland, Italy, Poland, and Portugal are obtained from EStreams (`Nascimento et al., 2024 <https://doi.org/10.1038/s41597-024-03706-1>`_), 
and similarly for Japan, Thailand, Arcticnet, and Spain from GSHA (`Peirong et al., 2023 <https://doi.org/10.5194/essd-16-1559-2024>`_). 
For USGS, the catchment boundaries are sourced from HYSETS (`Arsenault et al., 2020 <https://doi.org/10.1038/s41597-020-00583-2>`_).

Although each data source has a dedicated, however
all datasets listed in Table :ref:`rr_table` are accessible via the :py:class:`aqua_fetch.rr.RainfallRunoff` 
class, which allows for a unified and consistent approach to each dataset. The class 
provides several methods to access static features, dynamic features, or catchment 
boundaries. Although the raw data files for each dataset may come in different formats, 
the methods to access these features through the :py:class:`aqua_fetch.rr.RainfallRunoff` class remain the same. 
Individual classes for each dataset are also available and may offer more control to 
users over specific datasets. However, for most cases, the use of the :py:class:`aqua_fetch.rr.RainfallRunoff` 
class will suffice.

The naming and units of dynamic features in each dataset may vary. However, we have 
standardized these features using the formula ``name_unit_specifier`` for each dynamic 
feature across all datasets. In this formula, the specifier can indicate the source 
(such as ERA5 or MSWEP for precipitation), the method used to calculate the feature 
(like makkink or penman for evapotranspiration), or the aggregation type (min, max, mean). 
For example, a precipitation dynamic feature from MSWEP would be labeled as pcp_mm_mswep. 
This approach ensures that feature names are representative and understandable. 
Dynamic features for which this method is inapplicable retain their original names.

Another feature of the AquaFetch is the optional inclusion of static and dynamic 
features from EStreams and GSHA for all datasets listed in Table :ref:`rr_table`. 
This is beneficial as EStreams and GSHA include several static and dynamic features 
calculated for the catchments, which are not included in other datasets. For instance, 
EStreams provides information on annual variation in land use for all European catchments, 
a feature not available in CAMELS-GB (`Coxon et al., 2020 <https://doi.org/10.5194/essd-12-2459-2020>`_) or other European datasets. This step 
is optional since it initiaties the download of GSHA and EStreams datasets which can 
be time-consuming and may not always be necessary.

Certain datasets in this package feature overlapping stations from the same region. 
For example, both the ``Bull`` and ``Spain`` datasets cover Spain. 
However, the Bull dataset was introduced by by `Aparicio et al., 2024 <https://doi.org/10.1038/s41597-024-03594-5>`_ , 
whereas the Spain dataset was introduced in this work. The Spain dataset contains 
more stations, totaling 889, while the Bull dataset includes 484 stations.
Similarly, both the CABra (`Almagro et al., 2021 <https://doi.org/10.5194/hess-25-3105-2021>`_) and CAMELS_BR (`Chagas et al., 2020 <https://doi.org/10.5194/essd-12-2075-2020>`_) datasets 
cover Brazil and have been published in peer-reviewed journals. However, they differ 
in their temporal coverage and the number of static and dynamic features. Furthermore, 
Denmark is covered by two datasets, Caravan_DK (`Koch 2022 <https://doi.org/10.5281/zenodo.7962379>`_) and CAMELS_DK (`Liu et al., 2024 <https://doi.org/10.5194/essd-2024-292>`_), 
which differ in temporal coverage and the number of static and dynamic features. 
The HYSETS dataset (`Arsenault et al., 2020 <https://doi.org/10.1038/s41597-020-00583-2>`_) covers Mexico, the US, and Canada. However, 
we identified issues with the observed streamflow data for the US in HYSETS. As a 
result, we introduced the ``USGS`` dataset, which focuses specifically on the US region. 
The catchment boundaries, static features, and meteorological data for USGS, however, 
are still obtained from HYSETS.

.. _rr_table:
List of datasets
================
.. list-table:: Stations per Source
   :widths: 10 15 10 10 10 10 10 10 30
   :header-rows: 1

   * - Source Name
     - Class
     - Number of Daily Stations
     - Number of Hourly Stations
     - Dynamic features
     - Static features
     - Temporal Coverage
     - Spatial Coverage
     - Reference
   * - ``Arcticnet``
     - :py:class:`aqua_fetch.rr.Arcticnet`
     - 106
     - 
     - 27
     - 35
     - 1979 - 2003
     - Arctic (Russia)
     - `R-Arcticnet <https://www.r-arcticnet.sr.unh.edu/v4.0/AllData/index.html>`_
   * - ``Bull``
     - :py:class:`aqua_fetch.Bull`
     - 484
     -
     - 55
     - 214
     - 1990 - 2020
     - Spain
     - `Aparicio et al., 2024 <https://doi.org/10.1038/s41597-024-03594-5>`_
   * - ``CABra``
     - :py:class:`aqua_fetch.rr.CABra`
     - 735
     - 
     - 12
     - 97
     - 1980 - 2010
     - Brazil
     - `Almagro et al., 2021 <https://doi.org/10.5194/hess-25-3105-2021>`_ 
   * - ``CAMELS_AUS``
     - :py:class:`aqua_fetch.rr.CAMELS_AUS`
     - 222, 561
     - 
     - 26
     - 166, 187
     - 1900 - 2018
     - Australia
     - `Flower et al., 2021 <https://doi.org/10.5194/essd-13-3847-2021>`_
   * - ``CAMELS_BR``
     - :py:class:`aqua_fetch.rr.CAMELS_BR`
     - 897
     - 
     - 10
     - 67
     - 1920 - 2019
     - Brazil
     - `Chagas et al., 2020 <https://doi.org/10.5194/essd-12-2075-2020>`_
   * - ``CAMELS_CH``
     - :py:class:`aqua_fetch.rr.CAMELS_CH`
     - 331
     - 
     - 9
     - 209
     - 1981 - 2020
     - Switzerland
     - `Hoege et al., 2023 <https://doi.org/10.5194/essd-15-5755-2023>`_
   * - ``CAMELS_CL``
     - :py:class:`aqua_fetch.rr.CAMELS_CL`
     - 516
     - 
     - 12
     - 104
     - 1913 - 2018
     - Chile
     - `Alvarez-Garreton et al., 2018 <https://doi.org/10.5194/hess-22-5817-2018>`_
   * - ``CAMELS_DE``
     - :py:class:`aqua_fetch.rr.CAMELS_DE`
     - 1555
     - 
     - 21
     - 111
     - 1951 - 2020
     - Germany
     - `Loritz et al., 2024 <https://essd.copernicus.org/preprints/essd-2024-318/>`_
   * - ``CAMELS_DK``
     - :py:class:`aqua_fetch.rr.CAMELS_DK`
     - 304
     - 
     - 13
     - 119
     - 1989 - 2023
     - Denmark
     - `Liu et al., 2024 <https://doi.org/10.5194/essd-2024-292>`_
   * - ``CAMELS_FR``
     - :py:class:`aqua_fetch.rr.CAMELS_FR`
     - 654
     - 
     - 22
     - 344
     - 1970 - 2021
     - France
     - `Delaigue et al., 2024 <https://doi.org/10.5194/essd-2024-415>`_
   * - ``CAMELS_GB``
     - :py:class:`aqua_fetch.rr.CAMELS_GB`
     - 671
     - 
     - 10
     - 145
     - 1970 - 2015
     - Britain
     - `Coxon et al., 2020 <https://doi.org/10.5194/essd-12-2459-2020>`_
   * - ``CAMELS_IND``
     - :py:class:`aqua_fetch.rr.CAMELS_IND`
     - 472
     -
     - 20
     - 210
     - 1980 - 2020
     - Republic of India
     - `Mangukiya et al., 2024 <https://doi.org/10.5194/essd-2024-379>`_
   * - ``CAMELS_LUX``
     - :py:class:`aqua_fetch.rr.CAMELS_LUX`
     - 56
     - 56
     - 25
     - 61
     - 2004 - 2021
     - Luxumbourg
     - `Nijzink et al., 2025 <https://doi.org/10.5194/essd-2024-482>`_
   * - ``CAMELS_NZ``
     - :py:class:`aqua_fetch.rr.CAMELS_NZ`
     - 
     - 369
     - 5
     - 39
     - 1972 - 2024
     - New Zealand
     - `Harrigan et al., 2025 <https://doi.org/10.5194/essd-2025-244>`_
   * - ``CAMELS_SE``
     - :py:class:`aqua_fetch.rr.CAMELS_SE`
     - 50
     -
     - 4
     - 76
     - 1961 - 2020
     - Sweden
     - `Teutschbein et al., 2024 <https://doi.org/10.1002/gdj3.239>`_
   * - ``CAMELS_US``
     - :py:class:`aqua_fetch.rr.CAMELS_US`
     - 671
     - 
     - 8
     - 59
     - 1980 - 2014
     - United States
     - `Newman et al., 2014 <https://gdex.ucar.edu/dataset/camels.html>`_
   * - ``Caravan_DK``
     - :py:class:`aqua_fetch.rr.Caravan_DK`
     - 304
     - 
     - 38
     - 211
     - 1981 - 2020
     - Denmark
     - `Koch 2022 <https://doi.org/10.5281/zenodo.7962379>`_     
   * - ``CCAM``
     - :py:class:`aqua_fetch.rr.CCAM`
     - 111
     -
     - 16
     - 124
     - 1990 - 2020
     - China
     - `Hao et al., 2021 <https://doi.org/10.5194/essd-13-5591-2021>`_
   * - ``Finland``
     - :py:class:`aqua_fetch.rr.Finland`
     - 669
     -
     - 27
     - 35
     - 2012 - 2023
     - Finland
     - `ymparisto.fi <https://wwwi3.ymparisto.fi>`_
   * - ``GRDCCaravan``
     - :py:class:`aqua_fetch.rr.GRDCCaravan`
     - 5357
     -
     - 39
     - 211
     - 1950 - 2023
     - Global
     - `Faerber et al., 2023 <https://zenodo.org/records/10074416>`_
   * - ``HYPE``
     - :py:class:`aqua_fetch.rr.HYPE`
     - 561
     - 
     -
     - 
     - 
     - 
     - `Arciniega-Esparza and Birkel, 2020 <https://zenodo.org/records/4029572>`_     
   * - ``HYSETS``
     - :py:class:`aqua_fetch.rr.HYSETS`
     - 14425
     -
     - 5
     - 28
     - 1950 - 2018
     - North America (Mexico, Canada, USA)
     - `Arsenault et al., 2020 <https://doi.org/10.1038/s41597-020-00583-2>`_
   * - ``Ireland``
     - :py:class:`aqua_fetch.rr.Ireland`
     - 464
     -
     - 27
     - 35
     - 1992 - 2020
     - Ireland
     - `EPA Ireland <https://epawebapp.epa.ie>`_  
   * - ``Italy``
     - :py:class:`aqua_fetch.rr.Italy`
     - 294
     -
     - 37
     - 35
     - 1992 - 2020
     - Italy
     - `EPA Ireland <https://epawebapp.epa.ie>`_  
   * - ``Japan``
     - :py:class:`aqua_fetch.rr.Japan`
     - 751
     - 696
     - 27
     - 35
     - 1979 - 2022
     - Japan
     - `river.go.jp <http://www1.river.go.jp>`_           
   * - ``LamaHCE``
     - :py:class:`aqua_fetch.rr.LamaHCE`
     - 859
     - 859
     - 22
     - 80
     - 1981 - 2019
     - Central Europe
     - `Klingler et al., 2021 <https://doi.org/10.5194/essd-13-4529-2021>`_
   * - ``LamaHIce``
     - :py:class:`aqua_fetch.rr.LamaHIce`
     - 111
     - 111
     - 36
     - 154
     - 1950 - 2021
     - Iceland
     - `Helgason and Nijssen 2024 <https://doi.org/10.5194/essd-16-2741-2024>`_
   * - ``Poland``
     - :py:class:`aqua_fetch.rr.Poland`
     - 1287
     -
     - 27
     - 35
     - 1992 - 2020
     - Poland
     - `imgw.pl <https://danepubliczne.imgw.pl>`_     
   * - ``Portugal``
     - :py:class:`aqua_fetch.rr.Portugal`
     - 280
     -
     - 27
     - 35
     - 1992 - 2020
     - Portugal
     - `snirh <https://snirh.apambiente.pt>`_       
   * - ``RRLuleaSweden``
     - :py:class:`aqua_fetch.RRLuleaSweden`
     - 1
     - 
     - 2
     - 0
     - 2016 - 2019
     - Lulea (Sweden)
     - `Broekhuizen et al., 2020 <https://doi.org/10.5194/hess-24-869-2020>`_   
   * - ``Spain``
     - :py:class:`aqua_fetch.rr.Spain`
     - 889
     -
     - 27
     - 35
     - 1979 - 2020
     - Spain
     - `ceh-flumen64 <https://ceh-flumen64.cedex.es>`_     
   * - ``Simbi``
     - :py:class:`aqua_fetch.rr.Simbi`
     - 24
     -
     - 3
     - 232
     - 1920 - 1940
     - Haiti
     - `Bathelemy et al., 2024 <https://doi.org/10.23708/02POK6>`_   
   * - ``Thailand``
     - :py:class:`aqua_fetch.rr.Thailand`
     - 73
     -
     - 27
     - 35
     - 1980 - 1999
     - Thailand
     - `RID project <https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/rid-river/disc_d.html>`_
   * - ``USGS``
     - :py:class:`aqua_fetch.rr.USGS`
     - 12004
     -
     - 5
     - 27
     - 1950 - 2018
     - United States
     - `USGS nwis <https://waterdata.usgs.gov/nwis/>`_
   * - ``WaterBenchIowa``
     - :py:class:`aqua_fetch.rr.WaterBenchIowa`
     - 125
     -
     - 3
     - 7
     - 2011 - 2018
     - Iowa (USA)
     - `Demir et al., 2022 <https://doi.org/10.5194/essd-14-5605-2022>`_


High Level API
==============
The :py:class:`aqua_fetch.rr.RainfallRunoff` class represents high level API
which provides a unified and easy-to-use interface to access all the datasets. 
It is recommended to use this class to access the datasets.

.. autoclass:: aqua_fetch.rr.RainfallRunoff
   :members:
   :show-inheritance:

   .. automethod:: __init__


Low Level API
=============
The low level API provides access to each individual dataset classes.
This provides more control over the datasets.


.. autoclass:: aqua_fetch.rr._RainfallRunoff
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


.. autoclass:: aqua_fetch.rr.CAMELS_GB
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


.. autoclass:: aqua_fetch.rr.CAMELS_NZ
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.rr.CAMELS_LUX
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.rr.CAMELS_US
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


.. autoclass:: aqua_fetch.rr.NPCTRCatchments
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
