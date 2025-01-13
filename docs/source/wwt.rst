Waste Water Treatment
**********************
This sub-module contains datasets related to wastewater treatment 
with various methods like adsorption, photocatalysis, membrane, sonolysis etc.

List of datasets
================
.. list-table:: Stations per Source
   :widths: 10 15 10 15 10 15
   :header-rows: 1

   * - Treatment Process
     - Class
     - Parameters
     - Target Pollutant
     - Data Points
     - Reference
   * - ``Adsorption``
     - :py:class:`water_datasets.water_treatment.ec_removal_biochar`
     - 26
     - Emerg. Contaminants
     - 3,757
     - `Jaffari et al., 2023 <https://doi.org/10.1016/j.cej.2023.143073>`_
   * - ``Adsorption``
     - :py:class:`water_datasets.water_treatment.cr_removal`
     - 15
     - Cr
     - 219
     - `Ishtiaq et al., 2024 <https://doi.org/10.1016/j.jece.2024.112238>`_
   * - ``Adsorption``
     - :py:class:`water_datasets.water_treatment.heavy_metal_removal`
     - 30
     - heavy metals
     - 1518
     - `Jaffari et al., 2023 <https://doi.org/10.1016/j.jhazmat.2023.132773>`_
   * - ``Adsorption``
     - :py:class:`water_datasets.water_treatment.po4_removal_biochar`
     - 30
     - po4
     - 5014
     - `Iftikhar et al., 2024 <https://doi.org/10.1016/j.chemosphere.2024.144031>`_
   * - ``Adsorption``
     - :py:class:`water_datasets.water_treatment.industrial_dye_removal`
     - 12
     - Industrial Dye
     - 1514
     - `Iftikhar et al., 2023 <https://doi.org/10.1016/j.seppur.2023.124891>`_
   * - ``Adsorption``
     - :py:class:`water_datasets.water_treatment.heavy_metal_removal_Shen`
     - 17
     - Heavy Metals
     - 689
     - `Shen et al., 2023 <https://doi.org/10.1016/j.jhazmat.2024.133442>`_
   * - ``Adsorption``
     - :py:class:`water_datasets.water_treatment.P_recovery`
     - 8
     - P
     - 504
     - `Leng et al., 2024 <https://doi.org/10.1016/j.jwpe.2024.104896>`_
   * - ``Adsorption``
     - :py:class:`water_datasets.water_treatment.N_recovery`
     - 8
     - N
     - 211
     - `Leng et al., 2024 <https://doi.org/10.1016/j.jwpe.2024.104896>`_
   * - ``Adsorption``
     - :py:class:`water_datasets.water_treatment.As_recovery`
     - 13
     - As
     - 1605
     - `Huang et al., 2024 <https://doi.org/10.1016/j.watres.2024.122815>`_
   * - ``Photocatalysis``
     - :py:class:`water_datasets.water_treatment.mg_degradation`
     - 11
     - Melachite Green
     - 1200
     - `Jaffari et a., 2023 <https://doi.org/10.1016/j.jhazmat.2022.130031>`_
   * - ``Photocatalysis``
     - :py:class:`water_datasets.water_treatment.dye_removal`
     - 23
     - Dyes
     - 1527
     - `Kim et al., 2024 <https://doi.org/10.1016/j.jhazmat.2023.132995>`_
   * - ``Photocatalysis``
     - :py:class:`water_datasets.water_treatment.dichlorophenoxyacetic_acid_removal`
     - 15
     - 2,4,Dichlorophenoxyacetic acid
     - 1044
     - `Kim et al., 2024 <https://doi.org/10.1016/j.jhazmat.2023.132995>`_
   * - ``Photocatalysis``
     - :py:class:`water_datasets.water_treatment.pms_removal`
     -
     -
     - 2078
     - `submitted et al., 2024 <https://doi.org/10.1016/j.jhazmat.2023.132995>`_
   * - ``Photocatalysis``
     - :py:class:`water_datasets.water_treatment.tetracycline_degradation`
     - 8
     - Tetracycline
     - 374
     - `Abdi et al., 2022 <https://doi.org/10.1016/j.chemosphere.2021.132135>`_
   * - ``Photocatalysis``
     - :py:class:`water_datasets.water_treatment.tio2_degradation`
     - 7
     - TiO2
     - 446
     - `Jiang et al., 2020 <https://doi.org/10.1016/j.envres.2020.109697>`_
   * - ``Membrane``
     - :py:class:`water_datasets.water_treatment.micropollutant_removal_osmosis`
     - 18
     - micropollutants
     - 1906
     - `Jeong et al., 2021 <https://doi.org/10.1021/acs.est.1c04041>`_
   * - ``sonolysis``
     - :py:class:`water_datasets.water_treatment.cyanobacteria_disinfection`
     - 6
     - Cyanobacteria
     - 314
     - `Jaffari et al., 2024 <https://doi.org/10.1016/j.jhazmat.2024.133762>`_

Adsorption
==============
.. autofunction:: water_datasets.ec_removal_biochar
.. autofunction:: water_datasets.cr_removal
.. autofunction:: water_datasets.po4_removal_biochar
.. autofunction:: water_datasets.heavy_metal_removal
.. autofunction:: water_datasets.industrial_dye_removal
.. autofunction:: water_datasets.heavy_metal_removal_Shen
.. autofunction:: water_datasets.P_recovery
.. autofunction:: water_datasets.N_recovery
.. autofunction:: water_datasets.As_recovery


Photocatalysis
=================
.. autofunction:: water_datasets.mg_degradation
.. autofunction:: water_datasets.dye_removal
.. autofunction:: water_datasets.dichlorophenoxyacetic_acid_removal
.. autofunction:: water_datasets.pms_removal
.. autofunction:: water_datasets.tetracycline_degradation
.. autofunction:: water_datasets.tio2_degradation


Membrane
=========
.. autofunction:: water_datasets.micropollutant_removal_osmosis
.. autofunction:: water_datasets.ion_transport_via_reverse_osmosis


Sonolysis
=========
.. autofunction:: water_datasets.cyanobacteria_disinfection
