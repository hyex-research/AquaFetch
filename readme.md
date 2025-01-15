
[![Documentation Status](https://readthedocs.org/projects/water-datasets/badge/?version=latest)](https://water-datasets.readthedocs.io/en/latest/?badge=latest)

<p float="left">
  <img src="/docs/source/imgs/logo.png"/>
</p>


# A Unified Python Interface for Water Resource Dataset Acquisition and Harmonization

water-datasets is a Python package designed for the automated downloading, parsing, cleaning, and harmonization of freely available water resource datasets related to rainfall-runoff processes, surface water quality, and wastewater treatment. The package currently supports approximately 70 datasets, each containing between 1 to hundreds of parameters. It facilitates the downloading and transformation of raw data into consistent, easy-to-use, analysis-ready formats. This allows users to directly access and utilize the data without labor-intensive and time-consuming preprocessing.

The package comprises three submodules, each representing a different type of water resource data: `rr` for rainfall-runoff processes, `wq` for surface water quality, and `wwt` for wastewater treatment. The rr submodule offers data for 47,716 catchments worldwide, encompassing both dynamic and static features for each catchment. The dynamic features consist of observed streamflow and meteorological time series, averaged over the catchment area, available at daily or hourly time steps. Static features include constant parameters such as land use, soil, topography, and other physiographical characteristics, along with catchment boundaries. This submodule not only provides access to established rainfall-runoff datasets such as CAMELS and LamaH but also introduces new datasets compiled for the first time from publicly accessible online data sources. The `wq` submodule offers access to [16 surface water quality datasets](https://water-datasets.readthedocs.io/en/latest/wq.html#list-of-datasets), each containing various water quality parameters measured across different spaces and times. The `wwt` submodule provides access to over 20,000 experimental measurements related to wastewater treatment techniques such as adsorption, photocatalysis, membrane filtration, and sonolysis.

The development of water-datasets was inspired by the growing availability of diverse water resource datasets in recent years. As a community-driven project, the codebase is structured to allow contributors to easily add new datasets, ensuring the package continues to expand and evolve to meet future needs.


## Installation

The package can be installed using GitHub link from the master branch

	python -m pip install git+https://github.com/AtrCheema/water-datasets.git

To install from a specific branch such as ``dev`` branch which contains more recent code

	python -m pip install git+https://github.com/AtrCheema/water-datasets.git@dev

The above code will install minimal depencies required to use the library which include
numpy, pandas and requests. To install the library with full list of dependencies use the
``all`` option during installation.

	python -m pip install "water_datasets[all] @ git+https://github.com/AtrCheema/water_datasets.git"

This will install addtional optional depencdies which include xarray, pyshp, netCDF and easy_mpl.

## Usage
The following sections describe brief usage of datasets from each of the three submodules i.e. ``rr``, ``wq`` and ``wwt``.
For detailed usage examples see [docs](https://water-datasets.readthedocs.io/en/latest/index.html)

The core of ``rr`` sub-module is the [``RainfallRunoff``](https://water-datasets.readthedocs.io/en/latest/rainfall_runoff.html#water_datasets.rr.RainfallRunoff) class. This class
fetches dynamic features (catchment averaged hydrometeorological data at daily or sub-daily timesteps),
static features (catchment characteristics related to topography, soil, land use-land cover, or hydrological indices that have constant values over time)
and the catchment boundary. The following example demonstrates how to fetch data for [CAMELS_AUS](https://water-datasets.readthedocs.io/en/latest/rainfall_runoff.html#water_datasets.rr.CAMELS_AUS). However, the method is the same for all [available rainfall-runoff datasets](https://water-datasets.readthedocs.io/en/latest/rainfall_runoff.html#id36).

```python
from water_datasets import RainfallRunoff
dataset = RainfallRunoff('CAMELS_AUS')  # instead of CAMELS_AUS, you can provide any other dataset name
df = dataset.fetch(stations=1, as_dataframe=True)
df = df.unstack() # the returned dataframe is a multi-indexed dataframe so we have to unstack it
df.columns = df.columns.get_level_values('dynamic_features')
df.shape
   (21184, 26)
# get name of all stations as list
stns = dataset.stations()
len(stns)
   222
# get data of 10 % of stations as dataframe
df = dataset.fetch(0.1, as_dataframe=True)
df.shape
   (550784, 22)
# The returned dataframe is a multi-indexed data
df.index.names == ['time', 'dynamic_features']
    True
# get data by station id
df = dataset.fetch(stations='224214A', as_dataframe=True).unstack()
df.shape
    (21184, 26)
# get names of available dynamic features
dataset.dynamic_features
# get only selected dynamic features
data = dataset.fetch(1, as_dataframe=True,
...  dynamic_features=['tmax_AWAP', 'precipitation_AWAP', 'et_morton_actual_SILO', 'streamflow_MLd']).unstack()
data.shape
   (21184, 4)
# get names of available static features
dataset.static_features
# get data of 10 random stations
df = dataset.fetch(10, as_dataframe=True)
df.shape  # remember this is a multiindexed dataframe
   (21184, 260)
# when we get both static and dynamic data, the returned data is a dictionary
# with ``static`` and ``dyanic`` keys.
data = dataset.fetch(stations='224214A', static_features="all", as_dataframe=True)
data['static'].shape, data['dynamic'].shape
((1, 166), (550784, 1))
coords = dataset.stn_coords() # returns coordinates of all stations
coords.shape
    (472, 2)
dataset.stn_coords('3001')  # returns coordinates of station whose id is 3001
    18.3861 80.3917
dataset.stn_coords(['3001', '17021'])  # returns coordinates of two stations
```

The datasets related to surface water quality are available using functional or objected-oriented API
depending upon the complexity of the dataset. The following example shows usage of two surface water
quality related datasets. For complete name of Python functions and classes see [documentation](https://water-datasets.readthedocs.io/en/latest/water_quality.html)

```python
from water_datasets import busan_beach
dataframe = busan_beach()
dataframe.shape
(1446, 14)
dataframe = busan_beach(target=['tetx_coppml', 'sul1_coppml'])
dataframe.shape
(1446, 15)

from water_datasets import GRQA
ds = GRQA(path="/mnt/datawaha/hyex/atr/data")
print(ds.parameters)
len(ds.parameters)
country = "Pakistan"
len(ds.fetch_parameter('TEMP', country=country))
```

The datasets for wastewater treatment are all available in function API design. These datasets consist of experimental conducted
to remove certain pollutants from wastewater. For complete list of functions, see [documentation](https://water-datasets.readthedocs.io/en/latest/wwt.html)

```python
from water_datasets import ec_removal_biochar
data, *_ = ec_removal_biochar()
data.shape
(3757, 27)
data, encoders = ec_removal_biochar(encoding="le")
data.shape
(3757, 27)

from water_datasets import mg_degradation
mg_data, catalyst_encoder, anion_encoder = mg_degradation()
mg_data.shape
(1200, 12)
# the default encoding is None, but if we want to use one hot encoder
mg_data_ohe, cat_enc, an_enc = mg_degradation(encoding="ohe")
mg_data_ohe.shape
(1200, 31)
```

## Summary of rainfall runoff Datasets

| Name           | Num. of daily stations | Num. of hourly stations | Num. of dynamic features | Num. of static features | Temporal Coverage | Spatial Coverage                            | Ref.                                                                                                        |
|----------------|------------------------|-------------------------|--------------------------|-------------------------|-------------------|---------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| Arcticnet      | 106                    |                         | 27                       | 35                      | 1979 - 2003       | Arctic (Russia)                             | [R-Arcticnet](https://www.r-arcticnet.sr.unh.edu/v4.0/AllData/index.html)                                   |
| CAMELS_AUS     | 222, 561               |                         | 26                       | 166, 187                | 1900 - 2018       | Australia                                   | [Flower et al., 2021](https://doi.org/10.5194/essd-13-3847-2021)                                            |
| CAMELS_GB      | 671                    |                         | 10                       | 145                     | 1970 - 2015       | Britain                                     | [Coxon et al., 2020](https://doi.org/10.5194/essd-12-2459-2020)                                             |
| CAMELS_BR      | 897                    |                         | 10                       | 67                      | 1920 - 2019       | Brazil                                      | [Chagas et al., 2020](https://doi.org/10.5194/essd-12-2075-2020)                                            |
| CAMELS_US      | 671                    |                         | 8                        | 59                      | 1980 - 2014       | USA                                         | [Newman et al., 2014](https://gdex.ucar.edu/dataset/camels.html)                                            |
| CAMELS_CL      | 516                    |                         | 12                       | 104                     | 1913 - 2018       | Chile                                       | [Alvarez-Garreton et al., 2018](https://doi.org/10.5194/hess-22-5817-2018)                                  |
| CAMELS_DK      | 304                    |                         | 13                       | 119                     | 1989 - 2023       | Denmark                                     | [Liu et al., 2024](https://doi.org/10.5194/essd-2024-292)                                                   |
| CAMELS_CH      | 331                    |                         | 9                        | 209                     | 1981 - 2020       | Switzerland, Austria, France, Germany Italy | [Hoege et al., 2023](https://doi.org/10.5194/essd-15-5755-2023)                                             |
| CAMELS_DE      | 1555                   |                         | 21                       | 111                     | 1951 - 2020       | Germany                                     | [Loritz et al., 2024](https://essd.copernicus.org/preprints/essd-2024-318)                                  |
| CAMELS_SE      | 50                     |                         | 4                        | 76                      | 1961 - 2020       | Sweden                                      | [Teutschbein et al., 2024](https://doi.org/10.1002/gdj3.239)                                                |
| Caravan_DK     | 308                    |                         | 38                       | 211                     | 1981 - 2020       | Denmark                                     | [Koch, J. (2022)](https://doi.org/10.5281/zenodo.7962379)                                                   |
| LamaHCE        | 859                    | 859                     | 22                       | 80                      | 1981 - 2019       | Central Europe                              | [Klingler et al., 2021](https://doi.org/10.5194/essd-13-4529-2021)                                          |
| LamaHIce       | 111                    | 111                     | 36                       | 154                     | 1950 - 2021       | Iceland                                     | [Helgason and Nijssen 2024](https://doi.org/10.5194/essd-16-2741-2024)                                      |
| HYSETS         | 14425                  |                         | 5                        | 28                      | 1950 - 2018       | North America                               | [Arsenault et al., 2020](https://doi.org/10.1038/s41597-020-00583-2)                                        |
| GRDCCaravan    | 5357                   |                         | 39                       | 211                     | 1950 - 2023       | Global                                      | [Faerber et al., 2023](https://zenodo.org/records/10074416)                                                 |
| Bull           | 484                    |                         | 55                       | 214                     | 1990 - 2020       | Spain                                       | [Aparicio et al., 2024](https://doi.org/10.1038/s41597-024-03594-5)                                         |
| WaterBenchIowa | 125                    |                         | 3                        | 7                       | 2011 - 2018       | Iowa (USA)                                  | [Demir et al., 2022](https://doi.org/10.5194/essd-14-5605-2022)                                             |
| CCAM           | 102                    |                         | 16                       | 124                     | 1990 - 2020       | China                                       | [Hao et al., 2021](https://doi.org/10.5194/essd-13-5591-2021)                                               |
| RRLuleaSweden  | 1                      |                         | 2                        | 0                       | 2016 - 2019       | Lulea (Sweden)                              | [Broekhuizen et al., 2020](https://doi.org/10.5194/hess-24-869-2020)                                        |
| CABra          | 735                    |                         | 12                       | 97                      | 1980 - 2010       | Brazil                                      | [Almagro et al., 2021](https://doi.org/10.5194/hess-25-3105-2021)                                           |
| HYPE           | 561                    |                         | 9                        | 3                       | 1985 - 2019       | Costa Rica                                  | [Arciniega-Esparza and Birkel, 2020](https://zenodo.org/records/4029572)                                    |
| Ireland        | 464                    |                         | 27                       | 35                      | 1992 - 2020       | Ireland                                     | [EPA Ireland](https://epawebapp.epa.ie)                                                                     |
| Spain          | 889                    |                         | 27                       | 35                      | 1979 - 2020       | Spain                                       | [ceh-flumen64](https://ceh-flumen64.cedex.es)                                                               |
| Simbi          | 70                     |                         | 3                        | 232                     | 1920 - 1940       | Haiti                                       | [Bathelemy et al., 2024](https://doi.org/10.23708/02POK6)                                                   |
| CAMELS_IND     | 472                    |                         | 20                       | 210                     | 1980 - 2020       | India                                       | [Mangukiya et al., 2024](https://doi.org/10.5194/essd-2024-379)                                             |
| Japan          | 751                    | 696                     | 27                       | 35                      | 1979 - 2022       | Japan                                       | [river.go.jp](http://www1.river.go.jp)                                                                      |
| Thailand       | 73                     |                         | 27                       | 35                      | 1980 - 1999       | Thailand                                    | [RID project](https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/rid-river/disc_d.html)                  |
| USGS           | 12004                  | 1541                    | 5                        | 27                      | 1950 - 2018       | USA                                         | [USGS nwis](https://waterdata.usgs.gov/nwis)                                                                |
| Finland        | 669                    |                         | 27                       | 35                      | 2012 - 2023       | Finland                                     | [Nascimento et al., 2024](https://doi.org/10.5194/essd-2024-379)                                            |
| Poland         | 1287                   |                         | 27                       | 35                      | 1992 - 2020       | Poland                                      | [Nascimento et al., 2024](https://danepubliczne.imgw.pl)                                                    |
| Portugal       | 280                    |                         | 27                       | 35                      | 1992 - 2020       | Portugal                                    | [SNIRH Portugal](https://snirh.apambiente.pt)                                                               |
| Italy          | 294                    |                         | 27                       | 35                      | 1992 - 2020       | Italy                                       | [Nascimento et al., 2024](http://www.hiscentral.isprambiente.gov.it/hiscentral/hydromap.aspx?map=obsclient) |
| CAMELS_FR      | 654                    |                         | 22                       | 344                     | 1970 - 2021       | France                                      | [Delaigue et al., 2024](https://essd.copernicus.org/preprints/essd-2024-415/)                               |

## Summary of Water Quality Datasets

| Name                      | Variables Covered | Temporal Coverage | Spatial Coverage          | Ref.                                                                         |
|---------------------------|-------------------|-------------------|---------------------------|------------------------------------------------------------------------------|
| SWatCh                    | 24                | 1960 - 2022       | Global                    | [Lobke et al., 2022](https://doi.org/10.5194/essd-14-4667-2022)              |
| GRQA                      | 42                | 1898 - 2020       | Global                    | [Virro et al., 2021](https://essd.copernicus.org/articles/13/5483/2021/)     |
| Quadica                   |                   | 1950 - 2018       | Germany                   | [Ebeling et al., 2022 ](https://essd.copernicus.org/articles/14/3715/2022/)  |
| RC4USCoast                | 21                | 1850 - 2020       | USA                       | [Gomez et al., 2022](https://essd.copernicus.org/articles/15/2223/2023/)     |
| Busan Beach               |                   | 2018 - 2019       | Busan, South Korea        | [Jang et al](https://doi.org/10.1016/j.watres.2021.117001)                   |
| Ecoli Mekong River        | 10                | 2011 - 2021       | Mekong river (Houay Pano) | [Boithias et al., 2022](https://essd.copernicus.org/articles/14/2883/2022/)  |
| Ecoli Mekong River (Laos) | 10                | 2011 - 2021       | Mekong River (Laos)       | [Boithias et al., 2022](https://essd.copernicus.org/articles/14/2883/2022/)  |
| Ecoli Houay Pano (Laos)   | 10                | 2011 - 2021       | Houay Pano (Laos)         | [Boithias et al., 2022](https://essd.copernicus.org/articles/14/2883/2022/)  |
| CamelsChem                | 18                | 1980 - 2018       | Conterminous USA          | [Sterle et al., 2024](https://doi.org/10.5194/hess-28-611-2024)              |
| GRiMeDB                   | 18                | -                 | Global                    | [Stanley et al., 2023](https://doi.org/10.5194/essd-15-2879-2023)            |
| SanFrancisco Bay          | 18                | 1969 - 2015       | Sans Francisco Bay (USA)  | [Cloern et al., 2017](https://doi.org/10.1002/lno.10537)                     |
| Sylt Roads                | 18                | 1973 - 2019       | North Sea (Arctic)        | [Rick et al., 2023](https://doi.org/10.5194/essd-15-1037-2023)               |
| Buzzards Bay              | 64                | 1992 - 2018       | Buzzards Bay (USA)        | [Jakuba et al., 2021](https://doi.org/10.1038/s41597-021-00856-4)            |
| White Clay Creek          | 2                 | 1977 - 2017       | White Clay Creek (USA)    | [Newbold and  Damiano 2013](https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/) |
| Selune River              | 5                 | 2021 - 2022       | Selune River (France)     | [Moustapha Ba et al., 2023](https://doi.org/10.1016/j.dib.2022.10883)        |

## Summary of datasets related to wastewater treatment

| Treatment Process | Parameters | Target Pollutant               | Data Points | Reference                                                                  |
|-------------------|------------|--------------------------------|-------------|----------------------------------------------------------------------------|
| Adsorption        | 26         | Emerg. Contaminants            | 3,757       | [Jaffari et al., 2023](https://doi.org/10.1016/j.cej.2023.143073)          |
| Adsorption        | 15         | Cr                             | 219         | [Ishtiaq et al., 2024](https://doi.org/10.1016/j.jece.2024.112238)         |
| Adsorption        | 30         | heavy metals                   | 1,518       | [Jaffari et al., 2023 ](https://doi.org/10.1016/j.jhazmat.2023.132773)     |
| Adsorption        | 30         | po4                            | 5,014       | [Iftikhar et al., 2024](https://doi.org/10.1016/j.chemosphere.2024.144031) |
| Adsorption        | 12         | Industrial Dye                 | 1,514       | [Iftikhar et al., 2023](https://doi.org/10.1016/j.seppur.2023.124891)      |
| Adsorption        | 17         | Heavy metals                   | 689         | [Shen et al., 2023](https://doi.org/10.1016/j.jhazmat.2024.133442)         |
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
| membrane          | 18         | heavy metals                   | 1,586       | [Jeong et al., 2023](https://doi.org/10.1021/acs.est.2c08384)              |
| sonolysis         | 6          | Cyanobacteria                  | 314         | [Jaffari et al., 2024](https://doi.org/10.1016/j.jhazmat.2024.133762)      |
