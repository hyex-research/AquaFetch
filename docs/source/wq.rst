Water Quality
*************
The `wq` submodule contains datasets that represent surface water chemistry at 
various locations worldwide. Currently, it includes 12 water quality datasets, 
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
   * - ``SWatCh``
     - :py:class:`water_datasets.SWatCh`
     - 24
     - 1960 - 2022
     - Global
     - `Lobke et al., 2022 <https://doi.org/10.5194/essd-14-4667-2022>`_
   * - ``GRQA``
     - :py:class:`water_datasets.GRQA`
     - 42
     - 1898 - 2020
     - Global
     - `Virro et al., 2021 <https://essd.copernicus.org/articles/13/5483/2021/>`_
   * - ``Quadica``
     - :py:class:`water_datasets.Quadica`
     -
     - 1950 - 2018
     - Germany
     - `Ebeling et al., 2022 <https://essd.copernicus.org/articles/14/3715/2022/>`_
   * - ``RC4USCoast``
     - :py:class:`water_datasets.RC4USCoast`
     - 21
     - 1850 - 2020
     - USA
     - `Gomez et al., 2022 <https://essd.copernicus.org/articles/15/2223/2023/>`_
   * - ``Busan Beach``
     - :py:class:`water_datasets.busan_beach`
     -
     - 2018 - 2019
     - Busan, S.Korea
     - `Jang et al <https://www.sciencedirect.com/science/article/pii/S0043135421001998?via%3Dihub/>`_
   * - ``Ecoli Mekong River``
     - :py:class:`water_datasets.ecoli_mekong`
     - 10
     - 2011 - 2021
     - Mekong river (Houay Pano)
     - `Boithias et al., 2022 <https://essd.copernicus.org/articles/14/2883/2022/>`_
   * - ``Ecoli Mekong River (Laos)``
     - :py:class:`water_datasets.ecoli_mekong_laos`
     - 10
     - 2011 - 2021
     - Mekong River (Laos)
     - `Boithias et al., 2022 <https://essd.copernicus.org/articles/14/2883/2022/>`_
   * - ``Ecoli Houay Pano (Laos)``
     - :py:class:`water_datasets.ecoli_houay_pano`
     - 10
     - 2011 - 2021
     - Houay Pano (Laos)
     - `Boithias et al., 2022 <https://essd.copernicus.org/articles/14/2883/2022/>`_
   * - ``CamelsChem``
     - :py:class:`water_datasets.CamelsChem`
     - 18
     - 1980 - 2018
     - Continental USA
     - `Sterle et al., 2024 <https://doi.org/10.5194/hess-28-611-2024>`_


Functions and Classes
=======================
.. autoclass:: water_datasets.SWatCh
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.GRQA
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.Quadica
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.RC4USCoast
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.CamelsChem
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.SyltRoads
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autoclass:: water_datasets.SanFranciscoBay
   :members:
   :show-inheritance:

   .. automethod:: __init__


.. autofunction:: water_datasets.busan_beach


.. autofunction:: water_datasets.ecoli_mekong


.. autofunction:: water_datasets.ecoli_mekong_laos


.. autofunction:: water_datasets.ecoli_houay_pano


.. autofunction:: water_datasets.ecoli_mekong_2016


.. autofunction:: water_datasets.white_clay_creek   
