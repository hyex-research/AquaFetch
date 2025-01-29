
__all__ = ["DraixBleone"]

from .utils import _RainfallRunoff


class DraixBleone(_RainfallRunoff):
    """
    A high-frequency, long-term data set of hydrology and sediment yield: the alpine
    badland catchments of Draix-Bléone Observatory following the work of `Klotz et al., 2023 <https://doi.org/10.5194/essd-15-4371-2023>`_.

    """
    url = {
        # "spatial": "https://doi.org/10.57745/RUQLJL",
        # "hydro_sediment": "https://doi.org/10.17180/obs.draix",
        # "climate": "https://doi.org/10.57745/BEYQFQ"
"README.txt": 
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/158242",
"DRAIXBLEONE_DRAIX_BRU_DISCH.txt":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/158223",
"DRAIXBLEONE_DRAIX_BRU_SEDTRAP.txt":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/158225",
"DRAIXBLEONE_DRAIX_BRU_SSC.txt":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/158222",
"DRAIXBLEONE_DRAIX_LAV_DISCH.txt":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/158229",
"DRAIXBLEONE_DRAIX_LAV_SEDTRAP.txt":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/158224",
"DRAIXBLEONE_DRAIX_MOU_DISCH.txt":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/158226",
"DRAIXBLEONE_DRAIX_ROU_DISCH.txt":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/158238",

"Draix_Bleone_instruments.shp":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168716",
"Draix_Bleone_instruments.prj":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168720",
"Draix_Bleone_instruments.dbf":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168715",
"Draix_Bleone_instruments.cpg":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168718",
"Draix_Bleone_instruments.shx":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168717",
"Draix_Bleone_instruments.qpj":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168719",

"Draix_Bleone_catchment_contours.shp":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168844",
"Draix_Bleone_catchment_contours.prj":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168839",
"Draix_Bleone_catchment_contours.dbf":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168843",
"Draix_Bleone_catchment_contours.cpg":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168840",
"Draix_Bleone_catchment_contours.shx":  
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168841",
"Draix_Bleone_catchment_contours.qpj":
        "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168842",

# "DEM_Draix.tif":
#         "https://entrepot.recherche.data.gouv.fr/api/access/datafile/168727",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._download()