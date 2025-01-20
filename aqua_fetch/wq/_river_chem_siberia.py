
__all__ = ["RiverChemSiberia"]

import os
from typing import List

import pandas as pd

from .._datasets import Datasets


class RiverChemSiberia(Datasets):
    """
    A database of water chemistry in eastern Siberian rivers following
    `Liu et al., 2022 <https://doi.org/10.1038/s41597-022-01844-y>`_ .
    The dataset consists of meteorological data, water chemistry data, and
    shapefiles of 7 basins in eastern Siberia. The data is collected from 1991
    to 2012. The dataset is available at `figshare <https://doi.org/10.6084/m9.figshare.c.5831975.v1>`_ .
    Following parameters are available in the dataset:

        - ``La``
        - ``Lo``
        - ``Ca2+``
        - ``Mg2+``
        - ``K+``
        - ``Na+``
        - ``Cl-``
        - ``SO42-``
        - ``HCO3-``
        - ``TDS``
        - ``pH``
        - ``River``
        - ``Basin``
        - ``Subbasin``
        - ``Tannual``
        - ``Tmonthly``
        - ``Pannual``
        - ``Pmonthly``
        - ``Lithology``
        - ``Permafrost type``
        - ``IB``
        - ``Discharge``
        - ``Ori_ID``
        - ``Li``
        - ``Sr``
        - ``As``
        - ``Ba``
        - ``Si``
        - ``87Sr/86Sr``
        - ``¦Ä18O-H2O``
        - ``¦Ä2H-H2O``
    
    Examples
    --------
    >>> from aqua_fetch import RiverChemSiberia
    >>> ds = RiverChemSiberia()
    >>> ds.stations()
    ['Selenga-Baikal', 'Angara', 'Lena', 'Eastern-Siberia', 'Kolyma', 'Yana', 'Indigirka']
    """
    url = {
        "Sample data.zip": "https://springernature.figshare.com/ndownloader/files/37706754",
        "Boundary data.zip": "https://springernature.figshare.com/ndownloader/files/37706622"
    }

    def __init__(self, path=None, **kwargs):
        super().__init__(path=path, **kwargs)
        self.ds_dir = path
        self._download()

    @property
    def parameters(self)->List[str]:
        """
        Returns the parameters available in the dataset.
        """
        return self.database().columns.tolist()

    def stn_coords(self)->pd.DataFrame:
        """
        Returns the coordinates of the stations.
        """

        from shapefile import Reader
        from shapely.geometry import shape, Point

        stns_file = os.path.join(self.path, "Boundary data", "Boundary data", "Basin_boundary.shp")

        sf = Reader(stns_file)

        coords = []
        # Iterate through the shapes in the shapefile
        for shaperec in sf.iterShapeRecords():
            # Convert shapefile geometries into shapely geometries
            polygon = shape(shaperec.shape.__geo_interface__)
            centroid = polygon.centroid
            
            # Print or process the centroid
            coords.append([centroid.x, centroid.y, shaperec.record.Basin])

        sf.close()

        sf = os.path.join(self.path, "Boundary data", "Boundary data", "Eastern_Siberia_boundary.shp")
        sf = Reader(sf)

        # Iterate through the shapes in the shapefile
        for shaperec in sf.iterShapeRecords():
            # Convert shapefile geometries into shapely geometries
            polygon = shape(shaperec.shape.__geo_interface__)
            centroid = polygon.centroid
            
            # Print or process the centroid
            coords.append([centroid.x, centroid.y, 'Eastern_Siberia'])

        sf.close()

        return pd.DataFrame(coords, columns=['long', 'lat', 'index']).set_index('index')

    def stations(self)->List[str]:
        """
        Returns the names of (7) stations available in the dataset.
        """
        return self.database()['Basin'].unique().tolist()

    def database(self)->pd.DataFrame:
        """
        Returns the database of the water chemistry in eastern Siberian rivers.
        """
        fpath = os.path.join(self.path, "Sample data", "Sample data", "Samples_database.csv")

        # read the data file with encoding which can handle the special characters
        df = pd.read_csv(fpath, encoding='latin1', index_col=0)

        [df.pop(col) for col in ('Year', 'Month')]

        df.index = pd.to_datetime(df.pop('Data'))

        return df
    
    def boundary(self)->pd.DataFrame:
        """
        Returns the boundary data of the water chemistry in eastern Siberian rivers.
        """
        fpath = os.path.join(self.path, "Boundary data", "Boundary data", "Boundary_data.csv")

        raise NotImplementedError("The method is not implemented yet.")

    def meteorology(self):
        raise NotImplementedError("The method is not implemented yet.")