from ..utils import _RainfallRunoff


class CAMELS_SPAT(_RainfallRunoff):
    """
    Dataset of 1426 catchments from North America (USA and Canada) following the works of
    `Knoben et al., 2025 <https://doi.org/10.5194/egusphere-2025-893>`_.
    """
    stn_name = 'USA_14141500'
    time_step = 'obs-hourly'  # or obs_daily
    scale = 'macro-scale' # or headwater or meso-scale
    data_type = 'observations'  # or 
    url = {
        f'https://www.frdr-dfdr.ca/repo/files/1/published/publication_1211/submitted_data/{data_type}/{scale}/{time_step}/{stn_name}_hourly_flow_observations.nc',

        }


class CAMELS_DEBY(_RainfallRunoff):
    """
    lumped and gridded data at hourly and daily timestep for 210
    Bavarian (Germany) catchments following the work of
    `Anwar et al., 2025 <https://doi.org/10.5281/zenodo.14893685>`_.
    """


class CAMELS_ES(_RainfallRunoff):
    """
    """
    url = "https://zenodo.org/records/15040948"


class HydResponses(_RainfallRunoff):
    """
    See `von Matt et al., 2025 <https://essd.copernicus.org/preprints/essd-2025-383/>`_ .
    """
    url = "https://zenodo.org/records/14713275/files/HYD_RESPONSES.zip"


class ThirdPole(_RainfallRunoff):
    """
    Observed streamflow data from from 11 stations of Third Pole region following work of
    `Liu and Wang (2025) <https://doi.org/10.1080/20964471.2025.2585701>`_ . The data
    is downloaded from its `zenodo repository <https://zenodo.org/records/15853656>`_ .
    """

    _stations = ['Asaraghat', 'Benighat', 'Besham', 'Changdu', 'Chatara', 'Chisapani', 
                 'Devghat', 'Doyain', 'Jiayuqiao', 'Nuxia', 'Yogu']
    url = {
        f"Discharge_{stn}_1981-2020.nc": "https://zenodo.org/records/15853656/files/" for stn in _stations
    }
    url.update({'basin_info.rar': 'https://zenodo.org/records/15853656/files'})
