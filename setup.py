# -*- coding: utf-8 -*-
# Don't know which rights should be reserved  Ather Abbas
from setuptools import setup


import os
fpath = os.path.join(os.getcwd(), "readme.md")
if os.path.exists(fpath):
    with open(fpath, "r") as fd:
        long_desc = fd.read()
else:
    long_desc = "https://github.com/hyex-research/AquaFetch"


pandas_ver = 'pandas>=0.25.0, <= 2.1.4'


min_requirements = [
    pandas_ver,
    'requests',
    ]

extra_requires = [
# for plotting
"easy_mpl",

# spatial processing
# maybe manually download the wheel file and install
# shapely, # for spatial processing in mtropics
'pyshp',
# fiona, # processing of shapefiles in mtropics

# for reading data
'netCDF4',
'xarray',

# todo : following libraries are required by read_html
#lxml for reading html
#html5lib for reading html
]


all_requirements = min_requirements + extra_requires

setup(

    name='aqua_fetch',

    version="0.0.1rc1",

    description='A Unified Python Interface for Water Resource Data Acquisition and harmonization',
    long_description=long_desc,
    long_description_content_type="text/markdown",

    url='https://github.com/hyex-research/AquaFetch',

    author='Ather Abbas',
    author_email='ather_abbas786@yahoo.com',

    package_data={'data': ['portugal_stn_codes.csv']},
    include_package_data=True,

    classifiers=[
        'Development Status :: 4 - Beta',

        'Natural Language :: English',

        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',

        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],

    packages=['aqua_fetch',
              'aqua_fetch/wq',
              'aqua_fetch/rr',
              'aqua_fetch/wwt',
              ],

    install_requires=min_requirements,

    extras_require={
        'all': extra_requires,
    }
)
