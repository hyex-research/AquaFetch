Quick Start
************

The following sections describe brief usage of datasets from each of the three submodules i.e. ``rr``, ``wq`` and ``wwt``.
For detailed usage examples see `examples which run online <https://aquafetch.readthedocs.io/en/latest/auto_examples/index.html>`_
and `pre-run jupyter notebooks <https://aquafetch.readthedocs.io/en/latest/_notebooks/main.html>`_

Rainfall-Runoff Datasets
========================
The package provides a unified interface to access multiple rainfall-runoff datasets.
A rainfall runoff dataset consists of observed streamflow, meterological time series 
data averaged over the catchments, static features of the catchments and catchment
boundary as shapefile. The following example shows how to acess `CAMELS_SE <https://snd.se/en/catalogue/dataset/2023-173>`_ dataset
however, the same interface can be used to access `all other datasets <https://aquafetch.readthedocs.io/en/latest/rr.html#list-of-datasets>`_ as well.

.. code-block:: python

    >>> from aqua_fetch import RainfallRunoff
    >>> dataset = RainfallRunoff('CAMELS_SE')  # instead of CAMELS_SE, you can provide any other dataset name
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='5', as_dataframe=True)
    >>> df = dynamic['5'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (21915, 4)

    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       50
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (5)
       5

    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(21915, 4), (21915, 4), (21915, 4), (21915, 4), (21915, 4)]

    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('5', as_dataframe=True,
    ...  dynamic_features=['pcp_mm', 'airtemp_C_mean', 'q_cms_obs'])
    >>> dynamic['5'].shape
       (21915, 3)

    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10

    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='5', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['5'].shape
    ((1, 76), 1, (21915, 4))

    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   # -> xarray.core.dataset.Dataset

    >>> dynamic.dims   # -> FrozenMappingWarningOnValuesAccess({'time': 21915, 'dynamic_features': 4})

    >>> len(dynamic.data_vars)   # -> 10

    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (50, 2)
    >>> dataset.stn_coords('5')  # returns coordinates of station whose id is 5
        68.035599	21.9758
    >>> dataset.stn_coords(['5', '736'])  # returns coordinates of two stations

    # get area of a single station
    >>> dataset.area('5')
    # get coordinates of two stations
    >>> dataset.area(['5', '736'])

    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('5')

Water Quality Datasets
=======================

.. code-block:: python

    >>> from aquafetch import busan_beach
    >>> dataframe = busan_beach()
    >>> dataframe.shape
    (1446, 14)
    >>> dataframe = busan_beach(target=['tetx_coppml', 'sul1_coppml'])
    >>> dataframe.shape
    (1446, 15)

    >>> from aqua_fetch import GRQA
    >>> ds = GRQA(path="/mnt/datawaha/hyex/atr/data")
    >>> print(ds.parameters)
    >>> len(ds.parameters)
    >>> country = "Pakistan"
    >>> len(ds.fetch_parameter('TEMP', country=country))


Wastewater Treatment Datasets
==============================
The package provides easy access to data from ~20,000 experiments conducted
on the removal of various pollutants from wastewater using photocatalysis,
adsorption, sonolysis and membrane filteration.

.. code-block:: python

    >>> from aqua_fetch import ec_removal_biochar
    >>> data, _ = ec_removal_biochar()
    >>> data.shape
    (3757, 27)
    >>> data, encoders = ec_removal_biochar(encoding="le")
    >>> data.shape
    (3757, 27)

    >>> from aqua_fetch import mg_degradation
    >>> mg_data, encoders = mg_degradation()
    >>> mg_data.shape
    (1200, 12)
    ... # the default encoding is None, but if we want to use one hot encoder
    >>> mg_data_ohe, encoders = mg_degradation(encoding="ohe")
    >>> mg_data_ohe.shape
    (1200, 31)
