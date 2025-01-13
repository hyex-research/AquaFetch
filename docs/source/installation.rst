Installation
*************
Currently the library can be installed either using the github link or the setup.py file.
We will soon release our package on pypi for easier installation.

using github link
=================
You can also use github link to install water-datasets.
::
    python -m pip install git+https://github.com/AtrCheema/water-datasets.git

The latest code however (possibly with less bugs and more features) can be installed from ``dev`` branch instead
::
    python -m pip install git+https://github.com/AtrCheema/water-datasets.git@dev

To install the latest branch (`dev`) with all requirements use ``all`` keyword
::
    python -m pip install "water-datasets[all] @ git+https://github.com/AtrCheema/water-datasets.git@dev"

This will install `xarray <https://docs.xarray.dev/en/stable/>`_, `netCDF4 <https://github.com/Unidata/netcdf4-python>`_, 
`easy_mpl <https://easy-mpl.readthedocs.io/>`_
and `pyshp <https://github.com/GeospatialPython/pyshp>`_

You can also install water-datasets from a specific commit using the commit code (SHA) as below
::
    pip install git+https://github.com/AtrCheema/water-datasets.git@e2c0a9825bb987e16c3c29d5e124203829ef3802


using setup.py file
===================
This involves cloning the respotory, changing directory to the cloned folder and then running the setup.py file.
::
    git clone https://github.com/AtrCheema/water-datasets.git
    cd water-datasets
    python setup.py install