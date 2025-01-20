
__all__ = ["DraixBleone"]

from .utils import _RainfallRunoff


class DraixBleone(_RainfallRunoff):
    """
    A high-frequency, long-term data set of hydrology and sediment yield: the alpine
    badland catchments of Draix-Bléone Observatory

    """
    url = {
        "spatial": "https://doi.org/10.57745/RUQLJL",
        "hydro_sediment": "https://doi.org/10.17180/obs.draix",
        "climate": "https://doi.org/10.57745/BEYQFQ"
    }
