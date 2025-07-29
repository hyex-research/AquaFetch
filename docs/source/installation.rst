Installation
*************
Thehe library can be installed using `pip`, or the github link or the setup.py file.

using pip
=========
The most easy way to install aqua-fetch is using ``pip``
::
    pip install aqua-fetch

However, if you are interested in installing all dependencies of AquaFetch, you can
choose to install all of them as well.
::
    pip install aqua-fetch[all]

This will install `xarray <https://docs.xarray.dev/en/stable/>`_, `netCDF4 <https://unidata.github.io/netcdf4-python/>`_, 
`fiona <https://fiona.readthedocs.io/en/stable/>`_ and
`easy_mpl <https://easy-mpl.readthedocs.io/>`_ libraries. The xarray 
library is used to save the data in netCDF4 format for faster I/O operations. 
fiona is used to process shapefiles while easy_mpl is used for plotting purpose.

We can also specify the AquaFetch version that we want to install as below
::
    pip install aqua-fetch==1.0.0

To updated the installation run
::
    pip install --upgrade aqua-fetch

using github link
=================

You can also use github link to install AquaFetch.
::
    python -m pip install git+https://github.com/hyex-research/AquaFetch.git

The latest code however (possibly with less bugs and more features) can be installed from ``dev`` branch instead
::
    python -m pip install git+https://github.com/hyex-research/AquaFetch.git@dev

To install the latest branch (`dev`) with all requirements use ``all`` keyword
::
    python -m pip install "aqua-fetch[all] @ git+https://github.com/hyex-research/AquaFetch.git@dev"

This will install `xarray <https://docs.xarray.dev/en/stable/>`_, `netCDF4 <https://github.com/Unidata/netcdf4-python>`_, 
`easy_mpl <https://easy-mpl.readthedocs.io/>`_
and `fiona <https://fiona.readthedocs.io/en/stable/>`_ libraries.

You can also install AquaFetch from a specific commit using the commit code (SHA) as below
::
    pip install git+https://github.com/hyex-research/AquaFetch.git@e2c0a9825bb987e16c3c29d5e124203829ef3802


using setup.py file
===================
This involves cloning the respotory, changing directory to the cloned folder and then running the setup.py file.
::
    git clone https://github.com/hyex-research/AquaFetch.git
    cd AquaFetch
    python setup.py install
