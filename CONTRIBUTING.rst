
=========================
Contributing to AquaFetch
=========================

Contributions are always welcome and highly appreciated. The collection and arrangement of dataset
is a never-ending task and we need your help to keep this repository up-to-date and useful. Any change to the 
code must be made in ``dev`` branch and ``pull request`` must be submitted to the ``dev`` branch. If you
want to contribute you can contribute in following ways

- add a new dataset
- update an existing dataset
- improve documentation
- fix a bug
- write new tests
- improving the code base


adding new dataset
-----------------------
If you find a new dataset which is not covered in this dataset, you can add it by following
the steps given for each of the submodule below.

rainfall-runoff
==================
Consider that you want to add a new rainfall-runoff dataset named ``NewDataSource``.

- create a new file in ``aqua_fetch/rr/_newdatasource.py`` file
- create a new class in this file which must inherit from ``_RainfallRunoff`` class
- implement the following methods. The signature of these methods/attributes must be same as the parent class.

    - ``__init__``  -> within this method, the logic to download the data must be implemented. The data must be downloaded within `path`
    - ``_read_stn_dyn`` -> must return a pandas DataFrame with with columns of time series of /dynamic/hydrometeorological data for a single station and index as time
    - ``_read_static``  -> must return a pandas DataFrame with columns as names of static_features and index as station ids
- The implementation/creation of following methods/attributes is optional but recommended

    - ``dynamic_features`` -> must return a list of dynamic features that are available in the dataset, if not implemented then the column names of _read_stn_dyn() will be used
    - ``static_features`` -> must return a list of static features that are available in the dataset. If not implemented then the column names of _read_static() will be used
    - ``stations`` -> must return a list of station ids that are available in the dataset, If not defined then index of _read_static() will be used
    - ``end``  -> return the end date of the dataset
    - ``start`` -> return the start date of the dataset
    - ``stn_coords``
    - ``q_mmd``
    - ``boundary_file``  -> the path to the boundary file for the dataset.
- import this class in ``aqua_fetch/__init__.py`` file in aqua_fetch folder and change the ``ALL_DATASETS`` lists to include this new class
- update the ``DATASETS`` dictionary in ``aqua_fetch/rr/__init__.py`` file to include this new class
- Write tests for this new data source in tests directory where you compare the files generated by this new data source with the raw files.
- update the `table <https://github.com/hyex-research/AquaFetch/blob/master/docs/source/rr.rst#list-of-datasets>`_ as well as entry under `low level api <https://github.com/hyex-research/AquaFetch/blob/master/docs/source/rr.rst#low-level-api>`_ in docs/source/rr.rst file
- update the `README <https://github.com/hyex-research/AquaFetch/tree/master?tab=readme-ov-file#summary-of-rainfall-runoff-datasets>`_ file to include this new data source

Update an existing data source
-------------------------------
For some data sources, the data is updated regularly. If the new data on the web breaks the code-base,
you can update the code to handle the new data. 

improve documentation
----------------------
If you find any mistakes in the documentation or you think that the documentation can be improved, please
feel free to make the changes and submit a pull request. The documentation is built with sphinx and 
sphinx gallery packages. Please see docs/requirements.txt for complete list of dependencies. You must
run install all these dependencies and then run ``make html`` in docs directory
to see the changes in the documentation.

Fix a bug
---------
If you find a bug in the code, please report it by creating an issue in the github repository. If you
want to fix the bug, please create a pull request with the fix. You must also write a test for this bug
in tests directory. Please also ensure that existing tests for the data source are not failing. It is possible
that some tests fail because you have obtained the updated data for the given data source. In this case, please
discuss.

write new tests
---------------
Currently there are only few ``tests`` under tests directory. We need more tests to ensure that the code
is working correctly. If you want to contribute, you can write new tests for the existing dataset.
This will require that you download the raw data files for the dataset and then compare the output
of aqua_fetch with raw data files. 

improving the code base
-----------------------
There are many ways the code can be improved. For example, the downloading and 
parsing of certain datasets takes a long time. Although we are employing parallel 
processing but still there are sections where parallel processing
can be employed or existing implementation of parallel processing can be improved. Another way to improve
the code base is by adding new features in the ``Datasets`` class. You can also write code to add new static
features.