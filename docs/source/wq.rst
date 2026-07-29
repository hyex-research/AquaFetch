Water Quality
*************
The `wq` submodule contains datasets that represent surface water chemistry at
various locations worldwide. Currently, it includes 18 water quality datasets,
but we anticipate this number will increase in the future. The spatial and temporal
coverage of these datasets are detailed in following table.

List of datasets
================
.. list-table:: Summary of datasets
   :widths: 10 10 10 15 30 15
   :header-rows: 1
   :class: sphinx-datatable

   * - Dataset
     - Variables Covered
     - Temporal Coverage
     - Spatial Coverage
     - Reference
     - Class / Function Name
   * - Busan Beach
     - 14
     - 2018 - 2019
     - Busan, S.Korea
     - `Jang et al <https://www.sciencedirect.com/science/article/pii/S0043135421001998?via%3Dihub/>`_
     - :py:class:`aqua_fetch.busan_beach`
   * - Buzzards Bay
     - 16
     - 1992 - 2018
     - Buzzards Bay (USA)
     - `Jakuba et al., <https://doi.org/10.1038/s41597-021-00856-4>`_
     - :py:class:`aqua_fetch.BuzzardsBay`
   * - CamelsChem
     - 28
     - 1980 - 2018
     - Continental USA
     - `Sterle et al., 2024 <https://doi.org/10.5194/hess-28-611-2024>`_
     - :py:class:`aqua_fetch.CamelsChem`
   * - CamelsCHChem
     - 40
     - 1980 - 2020
     - Swtizerland
     - `Nascimento et al., 2025 <https://eartharxiv.org/repository/view/9046/>`_
     - :py:class:`aqua_fetch.CamelsCHChem`
   * - CaravanQual
     - 100
     - 1894 - 2025
     - Global
     - `Jones et al., 2026 <https://doi.org/10.1038/s41597-026-07352-7>`_
     - :py:class:`aqua_fetch.CaravanQual`
   * - Surface Water Chemistry
     - 24
     - 1960 - 2022
     - Global
     - `Lobke et al., 2022 <https://doi.org/10.5194/essd-14-4667-2022>`_
     - :py:class:`aqua_fetch.SWatCh`
   * - Global River Water Quality Archive
     - 42
     - 1898 - 2020
     - Global
     - `Virro et al., 2021 <https://essd.copernicus.org/articles/13/5483/2021/>`_
     - :py:class:`aqua_fetch.GRQA`
   * - water QUAlity, DIscharge and Catchment Attributes
     - 10
     - 1950 - 2018
     - Germany
     - `Ebeling et al., 2022 <https://essd.copernicus.org/articles/14/3715/2022/>`_
     - :py:class:`aqua_fetch.Quadica`
   * - river chemistry for US coasts
     - 21
     - 1850 - 2020
     - USA
     - `Gomez et al., 2022 <https://essd.copernicus.org/articles/15/2223/2023/>`_
     - :py:class:`aqua_fetch.RC4USCoast`
   * - Ecoli Mekong River
     - 10
     - 2011 - 2021
     - Mekong river (Houay Pano)
     - `Boithias et al., 2022 <https://essd.copernicus.org/articles/14/2883/2022/>`_
     - :py:class:`aqua_fetch.ecoli_mekong`
   * - Ecoli Mekong River (Laos)
     - 10
     - 2011 - 2021
     - Mekong River (Laos)
     - `Boithias et al., 2022 <https://essd.copernicus.org/articles/14/2883/2022/>`_
     - :py:class:`aqua_fetch.ecoli_mekong_laos`
   * - Ecoli Houay Pano (Laos)
     - 10
     - 2011 - 2021
     - Houay Pano (Laos)
     - `Boithias et al., 2022 <https://essd.copernicus.org/articles/14/2883/2022/>`_
     - :py:class:`aqua_fetch.ecoli_houay_pano`
   * - Global River Methane
     - 1
     - 1973 - 2021
     - Global
     - `Stanley et al., 2024 <https://doi.org/10.5194/essd-15-2879-2023>`_
     - :py:class:`aqua_fetch.GRiMeDB`
   * - Oligotrend
     - 17
     - 1986 - 2022
     - Global
     - `Minaudo et al., 2025 <https://doi.org/10.5194/essd-17-3411-2025>`_
     - :py:class:`aqua_fetch.Oligotrend`
   * - Global River Nutrients
     - 3
     - 1806 - 2025
     - Global
     - `Peters et al., 2026 <https://doi.org/10.1038/s41597-026-07625-1>`_
     - :py:class:`aqua_fetch.GlobalRiverNutrients`
   * - Sylt Roads
     - 15
     - 1973 - 2019
     - Red Sea (Arctic)
     - `Rick et al., 2023 <https://doi.org/10.5194/essd-15-1037-2023>`_
     - :py:class:`aqua_fetch.SyltRoads`
   * - San Francisco Bay
     - 18
     - 1969 - 2015
     - San Francisco (USA)
     - `Schraga et al., 2017 <https://doi.org/10.1038/sdata.2017.98>`_
     - :py:class:`aqua_fetch.SanFranciscoBay`
   * - Selune River, France
     - 5
     - 2021 - 2022
     - Selune River, (France)
     - `Moustapha Ba et al., 2023 <https://doi.org/10.1016/j.dib.2022.108837>`_
     - :py:class:`aqua_fetch.SeluneRiver`
   * - Siberian Rivers Chemistry
     - 30
     - 1991--2012
     - Siberian Rivers, (Russia)
     - `Moustapha Ba et al., 2023 <https://doi.org/10.1016/j.dib.2022.108837>`_
     - :py:class:`aqua_fetch.RiverChemSiberia`
   * - White Clay Creek
     - 2
     - 1973 - 2019
     - White Clay Creek (USA)
     - `Newbold and  Damiano 2013 <https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/>`_
     - :py:class:`aqua_fetch.WhiteClayCreek`


Functions and Classes
=======================

.. autoclass:: aqua_fetch.BuzzardsBay
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.CamelsChem
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.CamelsCHChem
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.CaravanQual
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.GlobalRiverNutrients
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.GRiMeDB
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.GRQA
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.Oligotrend
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.Quadica
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.RC4USCoast
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.RiverChemSiberia
    :members:
    :show-inheritance:
  
    .. automethod:: __init__


.. autoclass:: aqua_fetch.SyltRoads
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.SanFranciscoBay
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.SeluneRiver
    :members:
    :show-inheritance:
  
    .. automethod:: __init__


.. autoclass:: aqua_fetch.SWatCh
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.WhiteClayCreek
    :members:
    :show-inheritance:
  
    .. automethod:: __init__


.. autofunction:: aqua_fetch.busan_beach


.. autofunction:: aqua_fetch.ecoli_mekong


.. autofunction:: aqua_fetch.ecoli_mekong_laos


.. autofunction:: aqua_fetch.ecoli_houay_pano


.. autofunction:: aqua_fetch.ecoli_mekong_2016


.. autofunction:: aqua_fetch.white_clay_creek   
