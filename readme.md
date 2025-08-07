
[![Documentation Status](https://readthedocs.org/projects/aquafetch/badge/?version=latest)](https://aquafetch.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/aqua-fetch.svg)](https://badge.fury.io/py/aqua-fetch)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/aqua-fetch)](https://pypi.org/project/aqua-fetch/)
[![status](https://joss.theoj.org/papers/a53b0c03154da4b953b04cdb43de2387/status.svg)](https://joss.theoj.org/papers/a53b0c03154da4b953b04cdb43de2387)

<p float="left">
  <img src="/docs/source/imgs/logo.png"/>
</p>


# A Unified Python Interface for Water Resource Dataset Acquisition and Harmonization

AquaFetch is a Python package designed for the automated downloading, parsing, cleaning, and harmonization of freely available water resource datasets related to rainfall-runoff processes, surface water quality, and wastewater treatment. The package currently supports approximately 70 datasets, each containing between 1 to hundreds of parameters. It facilitates the downloading and transformation of raw data into consistent, easy-to-use, analysis-ready formats. This allows users to directly access and utilize the data without labor-intensive and time-consuming preprocessing.

The package comprises three submodules, each representing a different type of water resource data: `rr` for rainfall-runoff processes, `wq` for surface water quality, and `wwt` for wastewater treatment. The rr submodule offers data for 47,291 catchments worldwide, encompassing both dynamic and static features for each catchment. The dynamic features consist of observed streamflow and meteorological time series, averaged over the catchment area, available at daily and/or hourly time steps. Static features include constant parameters such as land use, soil, topography, and other physiographical characteristics, along with catchment boundaries. This submodule not only provides access to established rainfall-runoff datasets such as CAMELS and LamaH but also introduces new datasets compiled for the first time from publicly accessible online data sources. The `wq` submodule offers access to [17 surface water quality datasets](https://aquafetch.readthedocs.io/en/latest/wq.html#list-of-datasets), each containing various water quality parameters measured across different spaces and times. The `wwt` submodule provides access to over 20,000 experimental measurements related to wastewater treatment techniques such as adsorption, photocatalysis, membrane filtration, and sonolysis.

The development of AquaFetch was inspired by the growing availability of diverse water resource datasets in recent years. As a community-driven project, the codebase is structured to allow contributors to easily add new datasets, ensuring the package continues to expand and evolve to meet future needs.


## Installation

You can install AquaFetch using `pip`

    pip install aqua-fetch

The package can be installed using GitHub link from the master branch

	python -m pip install git+https://github.com/hyex-research/AquaFetch.git

To install from a specific branch such as ``dev`` branch which contains more recent code

	python -m pip install git+https://github.com/hyex-research/AquaFetch.git@dev

The above code will install minimal depencies required to use the library which include
numpy, pandas and requests. To install the library with full list of dependencies use the
``all`` option during installation.

	python -m pip install "aqua-fetch[all] @ git+https://github.com/hyex-research/AquaFetch.git"

This will install addtional optional depencdies which include [xarray](https://docs.xarray.dev/en/stable/), [fiona](https://fiona.readthedocs.io/en/stable/), [netCDF4](https://github.com/Unidata/netcdf4-python) and [easy_mpl](https://easy-mpl.readthedocs.io/).

## Usage
The following sections describe brief usage of datasets from each of the three submodules i.e. ``rr``, ``wq`` and ``wwt``.
For detailed usage examples see [docs](https://aquafetch.readthedocs.io/en/latest/index.html)

The core of ``rr`` sub-module is the [``RainfallRunoff``](https://aquafetch.readthedocs.io/en/latest/rainfall_runoff.html#aquafetch.rr.RainfallRunoff) class. This class
fetches dynamic features (catchment averaged hydrometeorological data at daily or sub-daily timesteps),
static features (catchment characteristics related to topography, soil, land use-land cover, or hydrological indices that have constant values over time)
and the catchment boundary. The following example demonstrates how to fetch data for [CAMELS_AUS](https://aquafetch.readthedocs.io/en/latest/rainfall_runoff.html#aquafetch.rr.CAMELS_AUS). However, the method is the same for all [available rainfall-runoff datasets](https://aquafetch.readthedocs.io/en/latest/rainfall_runoff.html#id36).

```python
from aqua_fetch import RainfallRunoff
dataset = RainfallRunoff('CAMELS_SE')  # instead of CAMELS_SE, you can provide any other dataset name

# get the data of a single (randomly selected) station
_, df = dataset.fetch(stations=1, as_dataframe=True)
df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
df.columns = df.columns.levels[1]
df.shape   # ->    (21915, 4)

# get name of all stations as list
stns = dataset.stations()
len(stns)  # -> 50

# get data of 10 % of stations as dataframe
_, df = dataset.fetch(0.1, as_dataframe=True)
df.shape  # (87660, 5)

# The returned dataframe is a multi-indexed data
df.index.names   # ['time', 'dynamic_features'] 

# get data by station id
_, df = dataset.fetch(stations='5', as_dataframe=True)
df.unstack().shape  # (21915, 4)

# get names of available dynamic features
dataset.dynamic_features

# get only selected dynamic features
_, data = dataset.fetch(1, as_dataframe=True,
...  dynamic_features=['pcp_mm', 'airtemp_C_mean', 'q_cms_obs'])
data.unstack().shape  # (21915, 3)

# get names of available static features
dataset.static_features

# get data of 10 random stations
_, df = dataset.fetch(10, as_dataframe=True)
df.shape  # remember this is a multiindexed dataframe  with shape (87660, 10)

# If we want to get both static and dynamic data
static, dynamic = dataset.fetch(stations='5', static_features="all", as_dataframe=True)
static.shape, dynamic.unstack().shape   # ((1, 76), (21915, 4))

# get coordinates of all stations
coords = dataset.stn_coords()
coords.shape  #     (50, 2)
# get coordinates of station whose id is 5
dataset.stn_coords('5')       # 68.035599	21.9758
# get coordinates of two stations
dataset.stn_coords(['5', '736'])
```

The datasets related to surface water quality are available using functional or objected-oriented API
depending upon the complexity of the dataset. The following example shows usage of two surface water
quality related datasets. For complete name of Python functions and classes see [documentation](https://aquafetch.readthedocs.io/en/latest/water_quality.html)

```python
from aqua_fetch import busan_beach
dataframe = busan_beach()
dataframe.shape  # (1446, 14)

dataframe = busan_beach(target=['tetx_coppml', 'sul1_coppml'])
dataframe.shape  # (1446, 15)

from aqua_fetch import GRQA
ds = GRQA(path="/path/to/data")
print(ds.parameters)

len(ds.parameters)    # 42
country = "Pakistan"
len(ds.fetch_parameter('TEMP', country=country))
```

The datasets for wastewater treatment are all available in function API design. These datasets consist of experimental conducted
to remove certain pollutants from wastewater. For complete list of functions, see [documentation](https://aquafetch.readthedocs.io/en/latest/wwt.html)

```python
from aqua_fetch import ec_removal_biochar
data, *_ = ec_removal_biochar()
data.shape  # -> (3757, 27)

data, encoders = ec_removal_biochar(encoding="le")
data.shape  # -> (3757, 27)


from aqua_fetch import mg_degradation
mg_data, encoders = mg_degradation()
mg_data.shape  # -> (1200, 12)

# the default encoding is None, but if we want to use one hot encoder
mg_data_ohe, encoders = mg_degradation(encoding="ohe")
mg_data_ohe.shape  # -> (1200, 31)

```

## Summary of rainfall runoff Datasets

| Name           | Num. of daily stations | Num. of hourly stations | Num. of dynamic features | Num. of static features | Temporal Coverage | Spatial Coverage                            | Ref.                                                                                                        |
|----------------|------------------------|-------------------------|--------------------------|-------------------------|-------------------|---------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| Arcticnet      | 106                    |                         | 27                       | 35                      | 1979 - 2003       | Arctic (Russia)                             | [R-Arcticnet](https://www.r-arcticnet.sr.unh.edu/v4.0/AllData/index.html)                                   |
| Bull           | 484                    |                         | 55                       | 214                     | 1990 - 2020       | Spain                                       | [Aparicio et al., 2024](https://doi.org/10.1038/s41597-024-03594-5)                                         |
| CABra          | 735                    |                         | 13                       | 87                      | 1980 - 2010       | Brazil                                      | [Almagro et al., 2021](https://doi.org/10.5194/hess-25-3105-2021)                                           |
| CAMELS_AUS     | 222, 561               |                         | 28                       | 166, 187                | 1900 - 2018       | Australia                                   | [Flower et al., 2021](https://doi.org/10.5194/essd-13-3847-2021)                                            |
| CAMELS_BR      | 897                    |                         | 10                       | 67                      | 1920 - 2019       | Brazil                                      | [Chagas et al., 2020](https://doi.org/10.5194/essd-12-2075-2020)                                            |
| CAMELS_COL     | 347                    |                         | 6                        | 255                     | 1981 - 2022       | Columbia                                    | [Jimenez et al., 2025](https://doi.org/10.5194/essd-2025-200)                                               |
| CAMELS_CH      | 331                    |                         | 9                        | 209                     | 1981 - 2020       | Switzerland, Austria, France, Germany Italy | [Hoege et al., 2023](https://doi.org/10.5194/essd-15-5755-2023)                                             |
| CAMELS_CL      | 516                    |                         | 12                       | 104                     | 1913 - 2018       | Chile                                       | [Alvarez-Garreton et al., 2018](https://doi.org/10.5194/hess-22-5817-2018)                                  |
| CAMELS_DK      | 304                    |                         | 13                       | 119                     | 1989 - 2023       | Denmark                                     | [Liu et al., 2024](https://doi.org/10.5194/essd-2024-292)                                                   |
| CAMELS_DE      | 1555                   |                         | 21                       | 111                     | 1951 - 2020       | Germany                                     | [Loritz et al., 2024](https://essd.copernicus.org/preprints/essd-2024-318)                                  |
| CAMELS_FI      | 320                    |                         |                          | 111                     | 1963 - 2023       | Finland                                     | [Seppä, I et al., 2025](https://doi.org/10.5281/zenodo.16257216)                                            |
| CAMELS_FR      | 654                    |                         | 22                       | 344                     | 1970 - 2021       | France                                      | [Delaigue et al., 2024](https://essd.copernicus.org/preprints/essd-2024-415/)                               |
| CAMELS_GB      | 671                    |                         | 10                       | 145                     | 1970 - 2015       | Britain                                     | [Coxon et al., 2020](https://doi.org/10.5194/essd-12-2459-2020)                                             |
| CAMELS_IND     | 472                    |                         | 20                       | 210                     | 1980 - 2020       | India                                       | [Mangukiya et al., 2024](https://doi.org/10.5194/essd-2024-379)                                             |
| CAMELS_LUX     | 56                     | 56                      | 25                       | 61                      | 2004 - 2021       | Luxumbourg                                  | [Nijzink et al., 2025](https://doi.org/10.5194/essd-2024-482)                                               |
| CAMELS_SE      | 50                     |                         | 4                        | 76                      | 1961 - 2020       | Sweden                                      | [Teutschbein et al., 2024](https://doi.org/10.1002/gdj3.239)                                                |
| CAMELS_SK      |                        | 178                     | 17                       | 215                     | 2000 - 2019       | South Korea                                 | [Kim et al., 2025](https://doi.org/10.5281/zenodo.15073263)                                                 |
| CAMELS_NZ      | 369                    | 369                     | 5                        | 39                      | 1972 - 2024       | New Zealand                                 | [Bushra, et al., 2025](https://doi.org/10.5194/essd-2025-244)                                               |
| CAMELS_US      | 671                    |                         | 8                        | 59                      | 1980 - 2014       | USA                                         | [Newman et al., 2014](https://gdex.ucar.edu/dataset/camels.html)                                            |
| Caravan_DK     | 308                    |                         | 38                       | 211                     | 1981 - 2020       | Denmark                                     | [Koch, J. (2022)](https://doi.org/10.5281/zenodo.7962379)                                                   |
| CCAM           | 102                    |                         | 16                       | 124                     | 1990 - 2020       | China                                       | [Hao et al., 2021](https://doi.org/10.5194/essd-13-5591-2021)                                               |
| Finland        | 669                    |                         | 10                       | 214                     | 2012 - 2023       | Finland                                     | [Nascimento et al., 2024](https://doi.org/10.5194/essd-2024-379) & [ymparisto.fi](https://wwwi3.ymparisto.fi) |
| GRDCCaravan    | 5357                   |                         | 39                       | 211                     | 1950 - 2023       | Global                                      | [Faerber et al., 2023](https://zenodo.org/records/10074416)                                                 |
| HYSETS         | 14425                  |                         | 20                       | 30                      | 1950 - 2018       | North America                               | [Arsenault et al., 2020](https://doi.org/10.1038/s41597-020-00583-2)                                        |
| HYPE           | 561                    |                         | 9                        | 3                       | 1985 - 2019       | Costa Rica                                  | [Arciniega-Esparza and Birkel, 2020](https://zenodo.org/records/4029572)                                    |
| Ireland        | 464                    |                         | 10                       | 214                     | 1992 - 2020       | Ireland                                     | [Nascimento et al., 2024](https://doi.org/10.5194/essd-2024-379) & [EPA Ireland](https://epawebapp.epa.ie)  |
| Italy          | 294                    |                         | 10                       | 214                     | 1992 - 2020       | Italy                                       | [Nascimento et al., 2024](https://doi.org/10.5194/essd-2024-379) & [hiscentral.isprambiente.gov.it](http://www.hiscentral.isprambiente.gov.it/hiscentral/hydromap.aspx?map=obsclient) |
| Japan          | 751                    | 696                     | 27                       | 35                      | 1979 - 2022       | Japan                                       | [Peirong et al., 2023](https://doi.org/10.5194/essd-16-1559-2024) & [river.go.jp](http://www1.river.go.jp)  |
| LamaHCE        | 859                    | 859                     | 22                       | 80                      | 1981 - 2019       | Central Europe                              | [Klingler et al., 2021](https://doi.org/10.5194/essd-13-4529-2021)                                          |
| LamaHIce       | 111                    | 111                     | 36                       | 154                     | 1950 - 2021       | Iceland                                     | [Helgason and Nijssen 2024](https://doi.org/10.5194/essd-16-2741-2024)                                      |
| NPCTRCatchments| -                      | 7                       | 8                        | 14                      | 2013 - 2019       | Canada                                      | [Korver et al., 2022](https://doi.org/10.5194/essd-14-4231-2022)                                            |
| Poland         | 1287                   |                         | 10                       | 214                     | 1992 - 2020       | Poland                                      | [Nascimento et al., 2024](https://doi.org/10.5194/essd-2024-379) & [danepubliczne.imgw.pl](https://danepubliczne.imgw.pl) |
| Portugal       | 280                    |                         | 10                       | 214                     | 1992 - 2020       | Portugal                                    | [Nascimento et al., 2024](https://doi.org/10.5194/essd-2024-379) & [SNIRH Portugal](https://snirh.apambiente.pt) |
| RRLuleaSweden  | 1                      |                         | 2                        | 0                       | 2016 - 2019       | Lulea (Sweden)                              | [Broekhuizen et al., 2020](https://doi.org/10.5194/hess-24-869-2020)                                        |
| Simbi          | 70                     |                         | 3                        | 232                     | 1920 - 1940       | Haiti                                       | [Bathelemy et al., 2024](https://doi.org/10.23708/02POK6)                                                   |
| Slovenia       | 117                    |                         | 3                        | 214                     | 1950 - 2023       | Slovenia                                    | [Nascimento et al., 2024](https://doi.org/10.5194/essd-2024-379) & [vode.arso.gov.si](https://vode.arso.gov.si) |
| Spain          | 889                    |                         | 27                       | 35                      | 1979 - 2020       | Spain                                       | [Peirong et al., 2023](https://doi.org/10.5194/essd-16-1559-2024) & [ceh-flumen64](https://ceh-flumen64.cedex.es) |
| Thailand       | 73                     |                         | 27                       | 35                      | 1980 - 1999       | Thailand                                    | [Peirong et al., 2023](https://doi.org/10.5194/essd-16-1559-2024) & [RID project](https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/rid-river/disc_d.html) |
| USGS           | 12004                  | 1541                    | 5                        | 27                      | 1950 - 2018       | USA                                         | [USGS nwis](https://waterdata.usgs.gov/nwis)                                                                |
| WaterBenchIowa | 125                    |                         | 3                        | 7                       | 2011 - 2018       | Iowa (USA)                                  | [Demir et al., 2022](https://doi.org/10.5194/essd-14-5605-2022)                                             |

## Summary of Water Quality Datasets

| Name                      | Variables Covered | Number of Stations | Temporal Coverage | Spatial Coverage          | Ref.                                                                         |
|---------------------------|-------------------|--------------------|-------------------|---------------------------|------------------------------------------------------------------------------|
| Busan Beach               | 14                |     1              | 2018 - 2019       | Busan, South Korea        | [Jang et al., 2021](https://doi.org/10.1016/j.watres.2021.117001)            |
| Buzzards Bay              | 64                |                    | 1992 - 2018       | Buzzards Bay (USA)        | [Jakuba et al., 2021](https://doi.org/10.1038/s41597-021-00856-4)            |
| CamelsChem                | 28                |     671            | 1980 - 2018       | Conterminous USA          | [Sterle et al., 2024](https://doi.org/10.5194/hess-28-611-2024)              |
| Camels_Ch_Chem            | 40                |     115            | 1980 - 2020       | Swtizerland               | [Nascimento et al., 2025](https://eartharxiv.org/repository/view/9046/)      |
| Ecoli Mekong River        | 10                |                    | 2011 - 2021       | Mekong river (Houay Pano) | [Boithias et al., 2022](https://essd.copernicus.org/articles/14/2883/2022/)  |
| Ecoli Mekong River (Laos) | 10                |                    | 2011 - 2021       | Mekong River (Laos)       | [Boithias et al., 2022](https://essd.copernicus.org/articles/14/2883/2022/)  |
| Ecoli Houay Pano (Laos)   | 10                |                    | 2011 - 2021       | Houay Pano (Laos)         | [Boithias et al., 2022](https://essd.copernicus.org/articles/14/2883/2022/)  |
| GRQA                      | 42                |                    | 1898 - 2020       | Global                    | [Virro et al., 2021](https://essd.copernicus.org/articles/13/5483/2021/)     |
| GRiMeDB                   | 1                 |     5029           | 1973 - 2021       | Global                    | [Stanley et al., 2023](https://doi.org/10.5194/essd-15-2879-2023)            |
| Oligotrend                | 17                |     1846           | 1986 - 2022       | Global                    | [Minaudo et al., 2025 ](https://doi.org/10.5194/essd-17-3411-2025)  |
| Quadica                   | 10                |     1386           | 1950 - 2018       | Germany                   | [Ebeling et al., 2022 ](https://essd.copernicus.org/articles/14/3715/2022/)  |
| RC4USCoast                | 21                |     140            | 1850 - 2020       | USA                       | [Gomez et al., 2022](https://essd.copernicus.org/articles/15/2223/2023/)     |
| SanFrancisco Bay          | 18                |                    | 1969 - 2015       | Sans Francisco Bay (USA)  | [Cloern et al., 2017](https://doi.org/10.1002/lno.10537)                     |
| Selune River              | 5                 |                    | 2021 - 2022       | Selune River (France)     | [Moustapha Ba et al., 2023](https://doi.org/10.1016/j.dib.2022.10883)        |
| Sylt Roads                | 15                |      3             | 1973 - 2019       | North Sea (Arctic)        | [Rick et al., 2023](https://doi.org/10.5194/essd-15-1037-2023)               |
| SWatCh                    | 24                |     26322          | 1960 - 2022       | Global                    | [Lobke et al., 2022](https://doi.org/10.5194/essd-14-4667-2022)              |
| White Clay Creek          | 2                 |                    | 1977 - 2017       | White Clay Creek (USA)    | [Newbold and  Damiano 2013](https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/) |

## Summary of datasets related to wastewater treatment

| Treatment Process | Parameters | Target Pollutant               | Data Points | Reference                                                                  |
|-------------------|------------|--------------------------------|-------------|----------------------------------------------------------------------------|
| Adsorption        | 26         | Emerg. Contaminants            | 3,757       | [Jaffari et al., 2023](https://doi.org/10.1016/j.cej.2023.143073)          |
| Adsorption        | 15         | Cr                             | 219         | [Ishtiaq et al., 2024](https://doi.org/10.1016/j.jece.2024.112238)         |
| Adsorption        | 30         | (Cr(VI), Co(II), Sr(II), Ba(II), I, and Fe ) | 1,518  | [Jaffari et al., 2023 ](https://doi.org/10.1016/j.jhazmat.2023.132773)     |
| Adsorption        | 30         | po4                            | 5,014       | [Iftikhar et al., 2024](https://doi.org/10.1016/j.chemosphere.2024.144031) |
| Adsorption        | 12         | Industrial Dye                 | 1,514       | [Iftikhar et al., 2023](https://doi.org/10.1016/j.seppur.2023.124891)      |
| Adsorption        | 17         | Cu, Zn, Pb, Cd, Ni, and As     | 689         | [Shen et al., 2023](https://doi.org/10.1016/j.jhazmat.2024.133442)         |
| Adsorption        | 8          | P                              | 504         | [Leng et al., 2024](https://doi.org/10.1016/j.jwpe.2024.104896)            |
| Adsorption        | 8          | N                              | 211         | [Leng et al., 2024](https://doi.org/10.1016/j.jwpe.2024.104896)            |
| Adsorption        | 13         | As                             | 1,605       | [Huang et al., 2024](https://doi.org/10.1016/j.watres.2024.122815)         |
| Photocatalysis    | 11         | Melachite Green                | 1,200       | [Jaffari et a., 2023](https://doi.org/10.1016/j.jhazmat.2022.130031)       |
| Photocatalysis    | 23         | Dyes                           | 1,527       | [Kim et al., 2024](https://doi.org/10.1016/j.jhazmat.2023.132995)          |
| Photocatalysis    | 15         | 2,4,Dichlorophenoxyacetic acid | 1,044       | [Kim et al., 2024](https://doi.org/10.1016/j.jhazmat.2023.132995)          |
| Photocatalysis    | -          | -                              | 2,078       | [submitted et al., 2024](https://doi.org/10.1016/j.jhazmat.2023.132995)    |
| Photocatalysis    | 8          | Tetracycline                   | 374         | [Abdi et al., 2022](https://doi.org/10.1016/j.chemosphere.2021.132135)     |
| Photocatalysis    | 7          | TiO2                           | 446         | [Jiang et al., 2020](https://doi.org/10.1016/j.envres.2020.109697)         |
| Photocatalysis    | 8          | multiple                       | 457         | [Jiang et al., 2020](https://doi.org/10.3390/catal11091107)                |
| membrane          | 18         | micropollutants                | 1,906       | [Jeong et al., 2021](https://doi.org/10.1021/acs.est.1c04041)              |
| membrane          | 18         | salts                          | 1,586       | [Jeong et al., 2023](https://doi.org/10.1021/acs.est.2c08384)              |
| sonolysis         | 6          | Cyanobacteria                  | 314         | [Jaffari et al., 2024](https://doi.org/10.1016/j.jhazmat.2024.133762)      |
