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
boundary as shapefile. The following example shows how to acess `CAMELS_AUS <https://doi.org/10.5194/essd-2024-263>`_ dataset
however, the same interface can be used to access `all other datasets <https://aquafetch.readthedocs.io/en/latest/rr.html#list-of-datasets>`_ as well.

.. code-block:: python

    >>> from aqua_fetch import RainfallRunoff
    >>> dataset = RainfallRunoff('CAMELS_AUS')  # instead of CAMELS_AUS, you can provide any other dataset name
    >>> _, df = dataset.fetch(stations=1, as_dataframe=True)
    >>> df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
    >>> df.columns = df.columns.get_level_values('dynamic_features')
    >>> df.shape
       (21184, 26)
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       222
    ... # get data of 10 % of stations as dataframe
    >>> _, df = dataset.fetch(0.1, as_dataframe=True)
    >>> df.shape
       (550784, 22)
    ... # The returned dataframe is a multi-indexed data
    >>> df.index.names == ['time', 'dynamic_features']
        True
    ... # get data by station id
    >>> _, df = dataset.fetch(stations='224214A', as_dataframe=True).unstack()
    >>> df.shape
        (21184, 26)
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, data = dataset.fetch(1, as_dataframe=True,
    ...  dynamic_features=['tmax_AWAP', 'precipitation_AWAP', 'et_morton_actual_SILO', 'streamflow_MLd']).unstack()
    >>> data.shape
       (21184, 4)
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, df = dataset.fetch(10, as_dataframe=True)
    >>> df.shape  # remember this is a multiindexed dataframe
       (21184, 260)
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='224214A', static_features="all", as_dataframe=True)
    >>> static.shape, dynamic.shape
    ((1, 166), (550784, 1))
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (472, 2)
    >>> dataset.stn_coords('3001')  # returns coordinates of station whose id is 3001
        18.3861	80.3917
    >>> dataset.stn_coords(['3001', '17021'])  # returns coordinates of two stations


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
    >>> data, *_ = ec_removal_biochar()
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
