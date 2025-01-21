---
title: 'AquaFetch: A Unified Python Interface for Water Resource Dataset Acquisition and Harmonization'
tags:
  - Python
  - modeling
  - hydrology
  - data
  - water
authors:
  - name: Ather Abbas
    orcid: 0000-0002-0031-745X
    affiliation: 1
  - name: Sara Iftikhar
    affiliation: 1
    orcid: 0000-0001-7446-6805
  - name: Hylke E. Beck
    corresponding: true
    orcid: 0000-0002-0031-745X
    affiliation: 1
affiliations:
 - name: King Abdullah University of Science and Technology, Thuwal, Saudi Arabia
   index: 1
date: 21 January 2025
bibliography: paper.bib
---


# Summary 
AquaFetch is a Python package designed for the automated downloading, parsing, cleaning, and harmonization of freely available water resource datasets related to rainfall-runoff processes, surface water quality, and wastewater treatment. The package currently supports 70 datasets, downloading and transforming raw data into consistent, easy-to-use analysis-ready data. This allows users to directly access and utilize the data without labor-intensive and time-consuming preprocessing.

The package comprises three submodules, each representing a different type of water resource data: `rr` for rainfall-runoff processes, `wq` for surface water quality, and `wwt` for wastewater treatment. The rr submodule offers data for 47,716 catchments worldwide, encompassing both dynamic and static features for each catchment. The dynamic features consist of observed streamflow and meteorological time series, averaged over the catchment area, available at daily or hourly time steps. Static features include constant parameters such as land use, soil, topography, and other physiographical characteristics, along with catchment boundaries. This submodule not only provides access to established rainfall-runoff datasets such as CAMELS and LamaH but also introduces new datasets compiled for the first time from publicly accessible online data sources. The `wq` submodule offers access to 16 surface water quality datasets, each containing various water quality parameters measured across different spaces and times. The `wwt` submodule provides access to 22,471 experimental measurements related to wastewater treatment techniques such as adsorption, photocatalysis, and sonolysis.

The development of AquaFetch was inspired by the growing availability of diverse water resource datasets in recent years. As a community-driven project, the codebase is structured to allow contributors to easily add new datasets, ensuring the package continues to expand and evolve to meet future needs.


# Statement of need
In recent years, there has been a significant increase in the release of datasets across various domains, including water resources. This surge is driven by advancements in computational and storage technologies, as well as the growing need to develop robust, accurate data-driven solutions to address challenges such as climate change, water scarcity, and environmental pollution. As a result, a wealth of national and global spatio-temporal datasets has become freely accessible online. These datasets are invaluable for applications like flood forecasting, climate change analysis, aquatic ecosystem management, improving drinking water safety, and optimizing wastewater treatment processes.

Despite the availability of these datasets, importing them into Python remains cumbersome. Researchers must often sift through multiple sources, including search engines, GitHub repositories, and various websites, to locate the necessary data. The diversity of data providers means datasets are frequently presented in inconsistent units and stored in varying formats. Additionally, many datasets require extensive preprocessing before they can be used for analysis or modeling. This makes acquiring, cleaning, organizing, and managing data a complex task requiring advanced data handling skills.

These challenges highlight the need for a unified, consistent, automated, and reusable framework for extracting hydrological and environmental data. The AquaFetch package addresses this gap by leveraging data-handling tools such as Pandas [@Pandas], NumPy [@NumPy2020], xarray [@xarray2017], and Shapely to offer a streamlined workflow for automatic data extraction from multiple sources in various formats.

Several other packages also aim to facilitate the acquisition and processing of water resource data. For example, tools like dataretrieval [@dataretrieval], hydrofunctions [@hydrofunctions], and harmonize-wq [@harmozie_wq] provide streamlined access to hydrological and water quality data, though they are primarily limited to data from U.S. Geological Survey (USGS) and Environmental Protection Agency (EPA) stations. Data Retriever [@retriever_Senyondo2017] offers access to over 200 parameters, including water quality, but lacks comprehensive rainfall-runoff datasets and wastewater treatment data. Meanwhile, tsp [@tsp] focuses on managing ground temperature data, and the HyRiver suite [@HyRiver] provides access to hydro-climate data exclusively from USGS’s National Water Information Service (NWIS). Geodata-Harvester [@GeodataHarvester] offers reusable and automated workflows for extracting diverse geospatial and environmental data for Australia.

In the R programming language, packages like weathercan [@weathercan], weatherOz [@weatherOz], and riversCentralAsia [@riversCentralAsia] provide hydro-climate data for regions such as Canada, Australia, and Central Asia, respectively. However, these packages often lack the static catchment features essential for modeling rainfall-runoff processes and do not offer comprehensive datasets for this purpose.

The Caravan initiative [@Caravan] stands out as a global platform for hydro-meteorological data with a distinct focus. While it aims to create large-sample hydrology datasets globally, AquaFetch facilitates access to a broad range of data via a unified interface, including processed rainfall-runoff datasets from countries like Japan, Thailand, Ireland, and Poland --- regions not covered by Caravan. Additionally, Caravan emphasizes cloud-based processing using the proprietary Google Earth Engine and does not provide a standalone Python package, requiring users to upload catchment boundaries for extracting static features. This highlights the distinct operational focuses of Caravan and AquaFetch.

Ultimately, we hope this package will foster the development of benchmark datasets in hydrological and environmental sciences, enhancing the comparability and reproducibility of data-driven solutions for water resource management.



# Implementation and Archiecture
The AquaFetch package is structured using both functional and Object-Oriented Programming (OOP) designs. The OOP design is employed for handling more complex datasets within the `rr` and `wq` submodules, while simpler datasets are managed through a functional interface. The package's code is logically organized, with `rr`, `wq`, and `wwt` subdirectories present in both the source code (aqua_fetch) and tests directories. All public classes and functions are accessible from the parent directory, allowing for straightforward imports as shown below:

```python
    from aqua_fetch import RainfallRUnoff
    from aqua_fetch import SWatch
    from aqua_fetch import mg_degradation
```

Datasets are downloaded upon the first call to the respective function or class. The data is saved locally and will not be redownloaded unless the user explicitly requests it to be overwritten. The package also leverages parallel processing to expedite the downloading and parsing of large datasets in the `rr` submodule, significantly speeding up data retrieval when multiple CPU cores are available.

# Rainfall Runoff datasets
Table \ref{tbl:table1} provides a comprehensive list of the 32 rainfall-runoff datasets currently covered by the package. These datasets include series such as CAMELS [@camels_addor_2017] and LamaH [@lamahce_klingler], published over the last decade and offering data for specific countries or regions. The GRDC-Caravan is the only dataset providing data from catchments globally (\autoref{fig1}). These datasets consist of three types of data: time series, tabular, and catchment shapefiles. The static catchment features are presented as tabular data, where each row corresponds to one catchment and each column to one static feature. The dynamic features of catchments include time series of meteorological data and observed streamflow, available at daily or hourly timesteps.

![Locations of catchment gauge stations covered by each of the 32 rainfall-runoff datasets\label{fig1}](rr_stations.png)

In addition to published datasets, this package introduces 10 new datasets for rainfall-runoff modeling. These datasets have not yet been published but follow the CAMELS dataset series convention. They include Ireland, Finland, Italy, Poland, Portugal, Japan, Thailand, Arcticnet, Spain, and the USGS. The observed streamflow data are sourced from the national meteorological or hydrological websites of the respective countries. Catchment boundaries and meteorological data for Ireland, Finland, Italy, Poland, and Portugal are obtained from EStreams [@estreams_do2024], and similarly for Japan, Thailand, Arcticnet, and Spain from GSHA [@gsha_yin2024]. For USGS, the catchment boundaries are sourced from HYSETS [@hysets_arsenault2020].

All datasets listed in Table \ref{tbl:table1} are accessible via the `RainfallRunoff` class, which allows for a unified and consistent approach to each dataset. The class provides several methods to access static features, dynamic features, or catchment boundaries. Although the raw data files for each dataset may come in different formats, the methods to access these features through the `RainfallRunoff` class remain the same. Individual classes for each dataset are also available and may offer more control to users over specific datasets. However, for most cases, the use of the `RainfallRunoff` class will suffice.

The naming and units of dynamic features in each dataset may vary. However, we have standardized these features using the formula `name_unit_specifier` for each dynamic feature across all datasets. In this formula, the specifier can indicate the source (such as ERA5 or MSWEP for precipitation), the method used to calculate the feature (like makkink or penman for evapotranspiration), or the aggregation type (min, max, mean). For example, a precipitation dynamic feature from MSWEP would be labeled as pcp_mm_mswep. This approach ensures that feature names are representative and understandable. Dynamic features for which this method is inapplicable retain their original names.

Another feature of the AquaFetch is the optional inclusion of static and dynamic features from EStreams and GSHA for all datasets listed in Table \ref{tbl:table1}. This is beneficial as EStreams and GSHA include several static and dynamic features calculated for the catchments, which are not included in other datasets. For instance, EStreams provides information on annual variation in land use for all European catchments, a feature not available in CAMELS-GB [@camels_gb] or other European datasets. This step is optional since it initiaties the download of GSHA and EStreams datasets which can be time-consuming and may not always be necessary.

## Duplicate datasets
Certain datasets in our package feature overlapping stations from the same region. For example, both the Bull [@bull_senent2024] and Spain datasets both cover Spain. However, the Bull dataset was introduced by by Aparicio et al. (2024) [@bull_senent2024], whereas the Spain dataset was introduced in our work. The Spain dataset contains more stations, totaling 889, while the Bull dataset includes 484 stations.
Similarly, both the CABra [@cabra_almagro2021] and CAMELS_BR [@camels_br] datasets cover Brazil and have been published in peer-reviewed journals. However, they differ in their temporal coverage and the number of static and dynamic features. Furthermore, Denmark is covered by two datasets, Caravan_DK [@caravan_dk_koch2023] and CAMELS_DK [@camels_dk_Liu2024], which differ in temporal coverage and the number of static and dynamic features. 
The HYSETS dataset [@hysets_arsenault2020] covers Mexico, the US, and Canada. However, we identified issues with the observed streamflow data for the US in HYSETS. As a result, we introduced the USGS class, which focuses specifically on the US region. The catchment boundaries, static features, and meteorological data for USGS, however, are still obtained from HYSETS.

| Name |Daily stations|Hourly stations|Dynamic features|Static features|Temporal Coverage|Spatial Coverage|Reference|
|---------------|--------:|:--------:|:--------:|:--------:|------------|------------|------------------------------|
| Arcticnet      | 106    |          | 27       | 35       | 1979--2003 | Arctic (Russia)| [R-Arcticnet](https://www.r-arcticnet.sr.unh.edu/v4.0/AllData/index.html)  |
| Bull           | 484    |          | 55       | 214      | 1990--2020 | Spain          | [Aparicio et al., 2024](https://doi.org/10.1038/s41597-024-03594-5)        |
| CABra          | 735    |          | 12       | 97       | 1980--2010 | Brazil         | [Almagro et al., 2021](https://doi.org/10.5194/hess-25-3105-2021)                                           |
| CAMELS_AUS     | 222, 561|         | 26       | 166, 187 | 1900--2018 | Australia      | [Flower et al., 2021](https://doi.org/10.5194/essd-13-3847-2021)        |
| CAMELS_BR      | 897    |          | 10       | 67       | 1920--2019 | Brazil         | [Chagas et al., 2020](https://doi.org/10.5194/essd-12-2075-2020)                   |
| CAMELS_CH      | 331    |          | 9        | 209      | 1981--2020 | Switzerland, Austria, France, Germany Italy | [Hoege et al., 2023](https://doi.org/10.5194/essd-15-5755-2023)  |
| CAMELS_CL      | 516    |          | 12       | 104      | 1913--2018 | Chile          | [Alvarez-Garreton et al., 2018](https://doi.org/10.5194/hess-22-5817-2018) |
| CAMELS_DE      | 1555   |          | 21       | 111      | 1951--2020 | Germany        | [Loritz et al., 2024](https://essd.copernicus.org/preprints/essd-2024-318)                                  |
| CAMELS_DK      | 304    |          | 13       | 119      | 1989--2023 | Denmark  | [Liu et al., 2024](https://doi.org/10.5194/essd-2024-292)            |
| CAMELS_FR      | 654    |          | 22       | 344      | 1970--2021 | France   | [Delaigue et al., 2024](https://essd.copernicus.org/preprints/essd-2024-415/)  |
| CAMELS_GB      | 671    |          | 10       | 145      | 1970--2015 | Britain                                     | [Coxon et al., 2020](https://doi.org/10.5194/essd-12-2459-2020)      |
| CAMELS_IND     | 472    |          | 20       | 210      | 1980--2020 | India                                       | [Mangukiya et al., 2024](https://doi.org/10.5194/essd-2024-379)   |
| CAMELS_SE      | 50     |          | 4        | 76       | 1961--2020 | Sweden    | [Teutschbein et al., 2024](https://doi.org/10.1002/gdj3.239)                                                |
| CAMELS_US      | 671    |          | 8        | 59       | 1980--2014 | USA       | [Newman et al., 2014](https://gdex.ucar.edu/dataset/camels.html)                                            |
| Caravan_DK     | 308    |          | 38       | 211      | 1981--2020 | Denmark        | [Koch, J. (2022)](https://doi.org/10.5281/zenodo.7962379)                                                   |
| CCAM           | 102    |          | 16       | 124      | 1990--2020 | China                                       | [Hao et al., 2021](https://doi.org/10.5194/essd-13-5591-2021) |
| Finland        | 669    |          | 27       | 35       | 2012--2023 | Finland                                     | [Nascimento et al., 2024](https://doi.org/10.5194/essd-2024-379)   |
| GRDCCaravan    | 5357   |          | 39       | 211      | 1950--2023 | Global                                      | [Faerber et al., 2023](https://zenodo.org/records/10074416)   |
| HYSETS         | 14425  |          | 5        | 28       | 1950--2018 | North America    | [Arsenault et al., 2020](https://doi.org/10.1038/s41597-020-00583-2)                                        |
| Italy          | 294    |          | 27       | 35       | 1992--2020 | Italy            | [Nascimento et al., 2024](http://www.hiscentral.isprambiente.gov.it/hiscentral/hydromap.aspx?map=obsclient) |
| Ireland        | 464    |          | 27       | 35       | 1992--2020 | Ireland          | [EPA Ireland](https://epawebapp.epa.ie)                                                                     |
| Japan          | 751    | 696      | 27       | 35       | 1979--2022 | Japan                                       | [river.go.jp](http://www1.river.go.jp)  |
| LamaHCE        | 859    | 859      | 22       | 80       | 1981--2019 | Central Europe                              | [Klingler et al., 2021](https://doi.org/10.5194/essd-13-4529-2021) |
| LamaHIce       | 111    | 111      | 36       | 154      | 1950--2021 | Iceland           | [Helgason and Nijssen 2024](https://doi.org/10.5194/essd-16-2741-2024) |
| Poland         | 1287   |          | 27       | 35       | 1992--2020 | Poland                                      | [Nascimento et al., 2024](https://danepubliczne.imgw.pl)   |
| Portugal       | 280    |          | 27       | 35       | 1992--2020 | Portugal                                    | [SNIRH Portugal](https://snirh.apambiente.pt) |
| RRLuleaSweden  | 1      |          | 2        | 0        | 2016--2019 | Lulea (Sweden)                              | [Broekhuizen et al., 2020](https://doi.org/10.5194/hess-24-869-2020) |
| Spain          | 889    |          | 27       | 35       | 1979--2020 | Spain                                       | [ceh-flumen64](https://ceh-flumen64.cedex.es) |
| Simbi          | 70     |          | 3        | 232      | 1920--1940 | Haiti                           | [Bathelemy et al., 2024](https://doi.org/10.23708/02POK6)          |
| Thailand       | 73     |          | 27       | 35       | 1980--1999 | Thailand    | [RID project](https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/rid-river/disc_d.html) |
| USGS           | 12004  | 1541     | 5        | 27       | 1950--2018 | USA                                         | [USGS nwis](https://waterdata.usgs.gov/nwis) |
| WaterBenchIowa | 125   |           | 3        | 7        | 2011--2018 | Iowa (USA)                                  | [Demir et al., 2022](https://doi.org/10.5194/essd-14-5605-2022)|

Table:  Summary of rainfall-runoff datasets covered in AquaFetch package \label{tbl:table1}

# Water Quality datasets
The `wq` submodule contains datasets that represent surface water chemistry at various locations worldwide. Currently, it includes 16 water quality datasets, but we anticipate this number will increase in the future. The spatial and temporal coverage of these datasets are detailed in Table \ref{tbl:table2} while the exact location of measuring stations are depicted in \autoref{fig2}.

| Name                    | Variables Covered | Temporal Coverage | Spatial Coverage      | Reference               |
|-------------------------|:-----------------:|-------------------|-----------------------|-------------------------|
| SWatCh                    | 24                | 1960--2022        | Global                    | [Lobke et al., 2022](https://doi.org/10.5194/essd-14-4667-2022)                    |
| GRQA                      | 42                | 1898--2020        | Global                    | [Virro et al., 2021](https://essd.copernicus.org/articles/13/5483/2021/)           |
| Quadica                   | 10                | 1950--2018        | Germany                   | [Ebeling et al., 2022 ](https://essd.copernicus.org/articles/14/3715/2022/)        |
| RC4USCoast                | 21                | 1850--2020        | USA                       | [Gomez et al., 2022](https://essd.copernicus.org/articles/15/2223/2023/)           |
| Busan Beach               | 14                | 2018--2019        | Busan (South Korea)       | [Jang et al](https://doi.org/10.1016/j.watres.2021.117001)                         |
| Ecoli Mekong River        | 10                | 2011--2021        | Mekong river (Houay Pano) | [Boithias et al., 2022](https://essd.copernicus.org/articles/14/2883/2022/)        |
| Ecoli Mekong River (Laos) | 10                | 2011--2021        | Mekong River (Laos)       | [Boithias et al., 2022](https://essd.copernicus.org/articles/14/2883/2022/)        |
| Ecoli Houay Pano (Laos)   | 10                | 2011--2021        | Houay Pano (Laos)         | [Boithias et al., 2022](https://essd.copernicus.org/articles/14/2883/2022/)        |
| CamelsChem                | 18                | 1980--2018        | Conterminous USA          | [Sterle et al., 2024](https://doi.org/10.5194/hess-28-611-2024)                    |
| Sylt Roads                | 14                | 1973--2019        | Continental USA           | [Rick., 2023](https://doi.org/10.5194/essd-15-1037-2023)                           |
| SanFranciscoBay           | 8                 | 1969--2015        | San Francisco (USA)       | [Sterle et al., 2024](https://doi.org/10.5194/hess-28-611-2024)                    |
| White Clay Creek          | 2                 | 2001--2012        | White Creek (USA)         | [Hydroshare](http://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b)  |
| Selune River              | 5                 | 2021--2022        | Selune River (France)     | [Moustapha Ba et al., 2023](https://doi.org/10.1016/j.dib.2022.10883)              |
| Buzzards Bay              | 64                | 1992--2018        | Buzzards Bay (USA)        | [Jakuba et al., 2021](https://doi.org/10.1038/s41597-021-00856-4)                  |
| GRiMeDB                   | 18                | -                 | Global                    | [Stanley et al., 2023](https://doi.org/10.5194/essd-15-2879-2023)                  |
| River Chemistry Siberia   | 30                | 1991--2012        | Siberia                   | [Liu et al., 2022](https://doi.org/10.1038/s41597-022-01844-y)                     |

Table:  Summary of water quality datasets covered in AquaFetch package \label{tbl:table2}

![Locations of measuring stations of water quality datasets\label{fig2}](wq_stations.png)

# Wastewater treatment datasets
The `wwt` submodule contains data from approximately 22,471 experiments focused on the removal of various contaminants from wastewater using treatment strategies such as adsorption, photocatalysis, and sonolysis. This submodule provides a unified interface to access all this data, which is scattered across the literature, in a standardized format using a few Python functions. It is important to note that we do not introduce this data since this data has already been utilized and analyzed in various peer-reviewed scientific publications. However, we offer a simple and easy-to-use interface to access this existing data. The availability of such a large corpus of experimental data can significantly aid in data-driven modeling and material discovery. A summary of these datasets is provided in Table \ref{tbl:table3}.

| Treatment Process        | Parameters        |        Target Pollutant      | Data Points | Reference             |
|--------------------------|:-----------------:|------------------------------|:-----------:|-----------------------|
| Adsorption        | 26         | Emerg. Contaminants            | 3,757       | [Jaffari et al., 2023](https://doi.org/10.1016/j.cej.2023.143073)          |
| Adsorption        | 15         | Cr                             | 219         | [Ishtiaq et al., 2024](https://doi.org/10.1016/j.jece.2024.112238)         |
| Adsorption        | 30         | Heavy Metals                   | 1,518       | [Jaffari et al., 2023 ](https://doi.org/10.1016/j.jhazmat.2023.132773)     |
| Adsorption        | 30         | PO4                            | 5,014       | [Iftikhar et al., 2024](https://doi.org/10.1016/j.chemosphere.2024.144031) |
| Adsorption        | 12         | Industrial Dye                 | 1,514       | [Iftikhar et al., 2023](https://doi.org/10.1016/j.seppur.2023.124891)      |
| Adsorption        | 17         | Heavy Metals                   | 689         | [Shen et al., 2023](https://doi.org/10.1016/j.jhazmat.2024.133442)         |
| Adsorption        | 8          | P                              | 504         | [Leng et al., 2024](https://doi.org/10.1016/j.jwpe.2024.104896)            |
| Adsorption        | 8          | N                              | 211         | [Leng et al., 2024](https://doi.org/10.1016/j.jwpe.2024.104896)            |
| Adsorption        | 13         | As                             | 1,605       | [Huang et al., 2024](https://doi.org/10.1016/j.watres.2024.122815)         |
| Photocatalysis    | 11         | Melachite Green                | 1,200       | [Jaffari et a., 2023](https://doi.org/10.1016/j.jhazmat.2022.130031)       |
| Photocatalysis    | 23         | Dyes                           | 1,527       | [Kim et al., 2024](https://doi.org/10.1016/j.jhazmat.2023.132995)          |
| Photocatalysis    | 15         | 2,4,Dichlorophenoxyacetic acid | 1,044       | [Kim et al., 2024](https://doi.org/10.1016/j.jhazmat.2023.132995)          |
| Photocatalysis    | 25         | Multiple                       | 2,078       | [GitHub](https://gitlab.com/atrcheema/envai105)                            |
| Photocatalysis    | 8          | Tetracycline                   | 374         | [Abdi et al., 2022](https://doi.org/10.1016/j.chemosphere.2021.132135)     |
| Photocatalysis    | 7          | TiO2                           | 446         | [Jiang et al., 2020](https://doi.org/10.1016/j.envres.2020.109697)         |
| Photocatalysis    | 8          | Multiple                       | 457         | [Jiang et al., 2020](https://doi.org/10.3390/catal11091107)                |
| Sonolysis         | 6          | Cyanobacteria                  | 314         | [Jaffari et al., 2024](https://doi.org/10.1016/j.jhazmat.2024.133762)      |

Table: Summary of wastewater treatment datasets covered in the package \label{tbl:table3}

# Testing and dependencies
The AquaFetch code is hosted on GitHub at https://github.com/hyex-research/AquaFetch. The package's core dependencies include requests [@requests], NumPy, and Pandas. xarray is utilized for saving data in netCDF5 [@NSF_Unidata_and_Davis_NetCDF-C] format, which is efficient for handling large datasets; however, this step is an optional dependency. Other optional dependencies include Matplotlib [@matplotlib] for plotting and visualization, openpyxl [@openpyxl] for parsing Microsoft Excel files, along with Shapely, Shapefile, and Fiona for processing shapefiles.
The documentation for the package is available on ReadTheDocs at https://aquafetch.readthedocs.io, featuring several tutorials that outline its usage and capabilities.

Adhering to the 'unit test' protocol, comprehensive testing has been implemented for all data classes and functions. Since downloading the datasets is time-consuming, the tests are conducted offline under the assumption that the datasets are already downloaded. These unit tests verify the number and types of parameters returned by the data functions.

# Acknowledgements
For part of the analysis, we utilized the Shaheen~III supercomputer, managed by the Supercomputing Core Laboratory at King Abdullah University of Science and Technology (KAUST) in Thuwal, Saudi Arabia. Part of the research was supported by the KAUST/MEWA Strategic Partnership Agreement (SPA) for Water, under award numbers 6110 and 6111.

# References