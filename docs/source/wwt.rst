Waste Water Treatment
**********************
The `wwt` submodule contains data from approximately 20,000 experiments focused 
on the removal of various contaminants from wastewater using treatment strategies 
such as adsorption, photocatalysis, membrane filtration, and sonolysis. This 
submodule provides a unified interface to access all this data, which is scattered 
across the literature, in a standardized format using a few Python functions. It 
is important to note that we do not introduce this data since this data has already 
been utilized and analyzed in various peer-reviewed scientific publications. However, 
we offer a simple and easy-to-use interface to access this existing data. The availability 
of such a large corpus of experimental data can significantly aid in data-driven 
modeling and material discovery. A summary of these datasets is provided in following table.

List of datasets
================
.. list-table:: Summary of datasets
   :widths: 10 15 10 15 10 15
   :header-rows: 1

   * - Treatment Process
     - Class
     - Parameters
     - Target Pollutant
     - Data Points
     - Reference
   * - ``Adsorption``
     - :py:func:`water_datasets.ec_removal_biochar`
     - 26
     - Emerg. Contaminants
     - 3,757
     - `Jaffari et al., 2023 <https://doi.org/10.1016/j.cej.2023.143073>`_
   * - ``Adsorption``
     - :py:func:`water_datasets.cr_removal`
     - 15
     - Cr
     - 219
     - `Ishtiaq et al., 2024 <https://doi.org/10.1016/j.jece.2024.112238>`_
   * - ``Adsorption``
     - :py:func:`water_datasets.heavy_metal_removal`
     - 30
     - heavy metals
     - 1518
     - `Jaffari et al., 2023 <https://doi.org/10.1016/j.jhazmat.2023.132773>`_
   * - ``Adsorption``
     - :py:func:`water_datasets.po4_removal_biochar`
     - 30
     - po4
     - 5014
     - `Iftikhar et al., 2024 <https://doi.org/10.1016/j.chemosphere.2024.144031>`_
   * - ``Adsorption``
     - :py:func:`water_datasets.industrial_dye_removal`
     - 12
     - Industrial Dye
     - 1514
     - `Iftikhar et al., 2023 <https://doi.org/10.1016/j.seppur.2023.124891>`_
   * - ``Adsorption``
     - :py:func:`water_datasets.heavy_metal_removal_Shen`
     - 17
     - Heavy Metals
     - 689
     - `Shen et al., 2023 <https://doi.org/10.1016/j.jhazmat.2024.133442>`_
   * - ``Adsorption``
     - :py:func:`water_datasets.P_recovery`
     - 8
     - P
     - 504
     - `Leng et al., 2024 <https://doi.org/10.1016/j.jwpe.2024.104896>`_
   * - ``Adsorption``
     - :py:func:`water_datasets.N_recovery`
     - 8
     - N
     - 211
     - `Leng et al., 2024 <https://doi.org/10.1016/j.jwpe.2024.104896>`_
   * - ``Adsorption``
     - :py:func:`water_datasets.As_recovery`
     - 13
     - As
     - 1605
     - `Huang et al., 2024 <https://doi.org/10.1016/j.watres.2024.122815>`_
   * - ``Photocatalysis``
     - :py:func:`water_datasets.mg_degradation`
     - 11
     - Melachite Green
     - 1200
     - `Jaffari et a., 2023 <https://doi.org/10.1016/j.jhazmat.2022.130031>`_
   * - ``Photocatalysis``
     - :py:func:`water_datasets.dye_removal`
     - 23
     - Dyes
     - 1527
     - `Kim et al., 2024 <https://doi.org/10.1016/j.jhazmat.2023.132995>`_
   * - ``Photocatalysis``
     - :py:func:`water_datasets.dichlorophenoxyacetic_acid_removal`
     - 15
     - 2,4,Dichlorophenoxyacetic acid
     - 1044
     - `Kim et al., 2024 <https://doi.org/10.1016/j.jhazmat.2023.132995>`_
   * - ``Photocatalysis``
     - :py:func:`water_datasets.pms_removal`
     -
     -
     - 2078
     - `submitted et al., 2024 <https://doi.org/10.1016/j.jhazmat.2023.132995>`_
   * - ``Photocatalysis``
     - :py:func:`water_datasets.tetracycline_degradation`
     - 8
     - Tetracycline
     - 374
     - `Abdi et al., 2022 <https://doi.org/10.1016/j.chemosphere.2021.132135>`_
   * - ``Photocatalysis``
     - :py:func:`water_datasets.tio2_degradation`
     - 7
     - TiO2
     - 446
     - `Jiang et al., 2020 <https://doi.org/10.1016/j.envres.2020.109697>`_
   * - ``Photocatalysis``
     - :py:func:`water_datasets.photodegradation_Jiang`
     - 8
     - multiple
     - 457
     - `Jiang et al., 2021 <https://doi.org/10.3390/catal11091107>`_     
   * - ``Membrane``
     - :py:func:`water_datasets.micropollutant_removal_osmosis`
     - 18
     - micropollutants
     - 1906
     - `Jeong et al., 2021 <https://doi.org/10.1021/acs.est.1c04041>`_
   * - ``sonolysis``
     - :py:func:`water_datasets.cyanobacteria_disinfection`
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
.. autofunction:: water_datasets.photodegradation_Jiang  


Membrane
=========
.. autofunction:: water_datasets.micropollutant_removal_osmosis
.. autofunction:: water_datasets.ion_transport_via_reverse_osmosis


Sonolysis
=========
.. autofunction:: water_datasets.cyanobacteria_disinfection
