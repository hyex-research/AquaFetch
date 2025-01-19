Water Quality
*************
The `wq` submodule contains datasets that represent surface water chemistry at 
various locations worldwide. Currently, it includes 16 water quality datasets, 
but we anticipate this number will increase in the future. The spatial and temporal 
coverage of these datasets are detailed in following table.

List of datasets
================
.. list-table:: Summary of datasets
   :widths: 10 15 10 10 15 30
   :header-rows: 1

   * - Dataset
     - Class / Function Name
     - Variables Covered
     - Temporal Coverage
     - Spatial Coverage
     - Reference
   * - Surface Water Chemistry
     - :py:class:`aqua_fetch.SWatCh`
     - 24
     - 1960 - 2022
     - Global
     - `Lobke et al., 2022 <https://doi.org/10.5194/essd-14-4667-2022>`_
   * - Global River Water Quality Archive
     - :py:class:`aqua_fetch.GRQA`
     - 42
     - 1898 - 2020
     - Global
     - `Virro et al., 2021 <https://essd.copernicus.org/articles/13/5483/2021/>`_
   * - water QUAlity, DIscharge and Catchment Attributes
     - :py:class:`aqua_fetch.Quadica`
     - 10
     - 1950 - 2018
     - Germany
     - `Ebeling et al., 2022 <https://essd.copernicus.org/articles/14/3715/2022/>`_
   * - river chemistry for US coasts
     - :py:class:`aqua_fetch.RC4USCoast`
     - 21
     - 1850 - 2020
     - USA
     - `Gomez et al., 2022 <https://essd.copernicus.org/articles/15/2223/2023/>`_
   * - Busan Beach
     - :py:class:`aqua_fetch.busan_beach`
     - 14
     - 2018 - 2019
     - Busan, S.Korea
     - `Jang et al <https://www.sciencedirect.com/science/article/pii/S0043135421001998?via%3Dihub/>`_
   * - Ecoli Mekong River
     - :py:class:`aqua_fetch.ecoli_mekong`
     - 10
     - 2011 - 2021
     - Mekong river (Houay Pano)
     - `Boithias et al., 2022 <https://essd.copernicus.org/articles/14/2883/2022/>`_
   * - Ecoli Mekong River (Laos)
     - :py:class:`aqua_fetch.ecoli_mekong_laos`
     - 10
     - 2011 - 2021
     - Mekong River (Laos)
     - `Boithias et al., 2022 <https://essd.copernicus.org/articles/14/2883/2022/>`_
   * - Ecoli Houay Pano (Laos)
     - :py:class:`aqua_fetch.ecoli_houay_pano`
     - 10
     - 2011 - 2021
     - Houay Pano (Laos)
     - `Boithias et al., 2022 <https://essd.copernicus.org/articles/14/2883/2022/>`_
   * - CamelsChem
     - :py:class:`aqua_fetch.CamelsChem`
     - 18
     - 1980 - 2018
     - Continental USA
     - `Sterle et al., 2024 <https://doi.org/10.5194/hess-28-611-2024>`_
   * - Global River Methane
     - :py:class:`aqua_fetch.GRiMeDB`
     - 18
     - 
     - Global
     - `Stanley et al., 2024 <https://doi.org/10.5194/essd-15-2879-2023>`_
   * - Sylt Roads
     - :py:class:`aqua_fetch.SyltRoads`
     - 18
     - 1973 - 2019
     - Red Sea (Arctic)
     - `Rick et al., 2023 <https://doi.org/10.5194/essd-15-1037-2023>`_
   * - San Francisco Bay
     - :py:class:`aqua_fetch.SanFranciscoBay`
     - 18
     - 1973 - 2019
     - San Francisco (USA)
     - `Schraga et al., 2017 <https://doi.org/10.1038/sdata.2017.98>`_
   * - Buzzards Bay
     - :py:class:`aqua_fetch.BuzzardsBay`
     - 18
     - 1992 - 2018
     - Buzzards Bay (USA)
     - `Jakuba et al., <https://doi.org/10.1038/s41597-021-00856-4>`_
   * - White Clay Creek
     - :py:class:`aqua_fetch.WhiteClayCreek`
     - 2
     - 1973 - 2019
     - White Clay Creek (USA)
     - `Newbold and  Damiano 2013 <https://www.hydroshare.org/resource/d841f99381424ebc850842a1dbb5630b/>`_
   * - Selune River, France
     - :py:class:`aqua_fetch.SeluneRiver`
     - 5
     - 2021 - 2022
     - Selune River, (France)
     - `Moustapha Ba et al., 2023 <https://doi.org/10.1016/j.dib.2022.108837>`_ 



Functions and Classes
=======================
.. autoclass:: aqua_fetch.SWatCh
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.GRQA
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


.. autoclass:: aqua_fetch.CamelsChem
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


.. autoclass:: aqua_fetch.GRiMeDB
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.BuzzardsBay
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: aqua_fetch.WhiteClayCreek
    :members:
    :show-inheritance:
  
    .. automethod:: __init__


.. autoclass:: aqua_fetch.SeluneRiver
    :members:
    :show-inheritance:
  
    .. automethod:: __init__



.. autofunction:: aqua_fetch.busan_beach


.. autofunction:: aqua_fetch.ecoli_mekong


.. autofunction:: aqua_fetch.ecoli_mekong_laos


.. autofunction:: aqua_fetch.ecoli_houay_pano


.. autofunction:: aqua_fetch.ecoli_mekong_2016


.. autofunction:: aqua_fetch.white_clay_creek   
