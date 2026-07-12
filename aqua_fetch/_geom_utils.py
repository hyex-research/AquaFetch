
from typing import List, Tuple

import math
import numpy as np


def _make_boundary_2d(geometry)-> List[np.ndarray]:
    """
    Converts a fiona.Geometry object to a list of 2D polygons (NumPy arrays).
    """
    def make_polygon_2d(polygon):
        """
        converts a 3D polygon to 2D polygon by removing the z coordinate
        """
        if polygon.ndim == 3:
            assert polygon.shape[0] == 1, "Only one polygon is expected for a catchment"
            polygon = polygon[0]
        
        if polygon.shape[1] == 3:
            # if the polygon has 3 coordinates, then we will remove the z coordinate
            assert polygon[:, -1].sum() == 0, "Z coordinate is not zero for the polygon"
            polygon = polygon[:, :-1]

        return polygon

    rings = []
    if geometry.type == 'MultiPolygon':
        for polygon in geometry.coordinates:
            if len(polygon) == 1:
                polygon = np.array(polygon)
                rings.append(make_polygon_2d(polygon))
            else:
                for p in polygon:  # for GRDC_1159100, # there are multiple polygons
                    p = np.array(p)
                    rings.append(make_polygon_2d(p))
    else:
        if len(geometry.coordinates) > 1:
            for polygon in geometry.coordinates:
                polygon = np.array(polygon)
                rings.append(make_polygon_2d(polygon))
        else:
            polygon = np.array(geometry.coordinates)
            rings.append(make_polygon_2d(polygon))
    return rings


def polygon_centroid(polygon:np.ndarray):
    """
    Calculates the centroid of a polygon represented as a NumPy array.

    Args:
        polygon (np.ndarray): A NumPy array of shape (n, 2) representing the polygon's vertices,
                              where n is the number of vertices.

    Returns:
        tuple: A tuple (centroid_x, centroid_y) representing the centroid coordinates.
    """
    assert polygon.ndim == 2 and polygon.shape[1] == 2, "Polygon must be a 2D array with shape (n, 2)"
    assert len(polygon) > 2, "Polygon must have at least 3 vertices to calculate centroid"

    x = polygon[:, 0]
    y = polygon[:, 1]
    n = len(polygon)

    # Calculate area using the shoelace formula
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += x[i] * y[j] - x[j] * y[i]
    area = 0.5 * area

    # Calculate centroid
    centroid_x = 0.0
    centroid_y = 0.0
    for i in range(n):
        j = (i + 1) % n
        centroid_x += (x[i] + x[j]) * (x[i] * y[j] - x[j] * y[i])
        centroid_y += (y[i] + y[j]) * (x[i] * y[j] - x[j] * y[i])

    centroid_x /= (6.0 * area)
    centroid_y /= (6.0 * area)

    return area, centroid_x, centroid_y  # Return area as well


def multipolygon_centroid(polygons: List[np.ndarray]):
    """
    Calculates the centroid of a MultiPolygon given a list of NumPy polygon arrays.

    Args:
        polygons: A list of NumPy arrays, where each array represents a polygon
                  with shape (n, 2).

    Returns:
        tuple: (centroid_x, centroid_y) representing the area-weighted centroid
               of the MultiPolygon.
    """
    total_area = 0.0
    total_centroid_x = 0.0
    total_centroid_y = 0.0

    for polygon in polygons:
        area, centroid_x, centroid_y = polygon_centroid(polygon)
        total_area += area
        total_centroid_x += area * centroid_x
        total_centroid_y += area * centroid_y

    if abs(total_area) < 1e-10:
        return 0.0, 0.0  # Handle zero area case

    centroid_x = total_centroid_x / total_area
    centroid_y = total_centroid_y / total_area

    return total_area, centroid_x, centroid_y


def calc_centroid(geometry)->Tuple[float, float]:
    """
    Calculates the centroid of a geometry object, which can be a Polygon or MultiPolygon.

    Args:
        geometry: A geometry object of fiona.Geometry (Polygon or MultiPolygon).

    Returns:
        tuple: (centroid_x, centroid_y) representing the centroid coordinates.
    """
    rings = _make_boundary_2d(geometry)

    if geometry.type == 'Polygon':    
        assert len(rings) == 1
        return polygon_centroid(rings[0])[1:]
    elif geometry.type == 'MultiPolygon':
        assert len(rings) > 1
        return multipolygon_centroid(rings)[1:]
    else:
        raise ValueError("Unsupported geometry type for centroid calculation.")


def epsg25832_to_wgs84(
        easting,  # longitude like but in projected coordinate
        northing,  # latitude like but in projected coordinate
        zone:int):
    # Constants
    a = 6378137.0  # WGS 84 major axis
    # Eccentricity : how much the ellipsoid deviates from being a perfect sphere
    e = 0.081819190842622  
    x = easting - 500000  # Correct for 500,000 meter offset
    y = northing
    # Scale factor, coefficient that scales the metric units in the projection to real-world distances
    k0 = 0.9996  
    
    # Calculate the Meridian Arc
    m = y / k0
    mu = m / (a * (1 - math.pow(e, 2) / 4 - 3 * math.pow(e, 4) / 64 - 5 * math.pow(e, 6) / 256))
    
    # Calculate Footprint Latitude
    e1 = (1 - math.sqrt(1 - e ** 2)) / (1 + math.sqrt(1 - e ** 2))
    phi1 = mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu)
    phi1 += (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu)
    phi1 += (151 * e1 ** 3 / 96) * math.sin(6 * mu)
    phi1 += (1097 * e1 ** 4 / 512) * math.sin(8 * mu)
    
    # Latitude and Longitude
    n1 = a / math.sqrt(1 - e ** 2 * math.sin(phi1) ** 2)
    t1 = math.tan(phi1) ** 2
    c1 = e ** 2 / (1 - e ** 2) * math.cos(phi1) ** 2
    r1 = a * (1 - e ** 2) / math.pow(1 - e ** 2 * math.sin(phi1) ** 2, 1.5)
    d = x / (n1 * k0)
    
    lat = phi1 - (n1 * math.tan(phi1) / r1) * (d ** 2 / 2 - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * e ** 2) * d ** 4 / 24)
    lat += (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 3 * c1 ** 2 - 252 * e ** 2) * d ** 6 / 720
    lat = lat * 180 / math.pi  # Convert to degrees
    
    lon = (d - (1 + 2 * t1 + c1) * d ** 3 / 6 + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * e ** 2 + 24 * t1 ** 2) * d ** 5 / 120) / math.cos(phi1)
    lon = lon * 180 / math.pi + (zone * 6 - 183)  # Convert to degrees
    
    return lat, lon


def tmerc_to_wgs84(
        easting,
        northing,
        lon_0: float,
        k0: float,
        false_easting: float,
        false_northing: float,
):
    """
    Inverse Transverse Mercator projection to WGS84 (latitude/longitude).

    This is a generalisation of :func:`epsg25832_to_wgs84` (which is hard-wired
    to the UTM parameters). It works for any Transverse Mercator projection
    defined on the WGS84/GRS80 ellipsoid, e.g. ETRS89 / Poland CS92
    (EPSG:2180) which is used by :py:class:`aqua_fetch.rr.CAMELS_PL`.

    The GRS80 and WGS84 ellipsoids differ only in the flattening at the ~1e-11
    level, so a single (WGS84) eccentricity is used for both. Validated against
    ``pyproj`` on the CAMELS-PL catchment boundaries with a maximum positional
    error of ~5 cm, which is negligible for catchment-scale geometry.

    Parameters
    ----------
    easting : float
        projected easting (x) in meters.
    northing : float
        projected northing (y) in meters.
    lon_0 : float
        longitude of the central meridian in degrees (e.g. ``19.0`` for EPSG:2180).
    k0 : float
        scale factor at the central meridian (e.g. ``0.9993`` for EPSG:2180).
    false_easting : float
        false easting in meters (e.g. ``500000`` for EPSG:2180).
    false_northing : float
        false northing in meters (e.g. ``-5300000`` for EPSG:2180).

    Returns
    -------
    tuple
        ``(lat, lon)`` in degrees.
    """
    a = 6378137.0                    # WGS84/GRS80 semi-major axis
    e = 0.081819190842622            # WGS84/GRS80 first eccentricity

    x = easting - false_easting
    y = northing - false_northing

    # meridional arc -> footprint latitude
    m = y / k0
    mu = m / (a * (1 - e ** 2 / 4 - 3 * e ** 4 / 64 - 5 * e ** 6 / 256))
    e1 = (1 - math.sqrt(1 - e ** 2)) / (1 + math.sqrt(1 - e ** 2))
    phi1 = (mu
            + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu)
            + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu)
            + (151 * e1 ** 3 / 96) * math.sin(6 * mu)
            + (1097 * e1 ** 4 / 512) * math.sin(8 * mu))

    n1 = a / math.sqrt(1 - e ** 2 * math.sin(phi1) ** 2)
    t1 = math.tan(phi1) ** 2
    c1 = (e ** 2 / (1 - e ** 2)) * math.cos(phi1) ** 2
    r1 = a * (1 - e ** 2) / math.pow(1 - e ** 2 * math.sin(phi1) ** 2, 1.5)
    d = x / (n1 * k0)

    lat = phi1 - (n1 * math.tan(phi1) / r1) * (
        d ** 2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * e ** 2) * d ** 4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * e ** 2 - 3 * c1 ** 2) * d ** 6 / 720)
    lat = math.degrees(lat)

    lon = (d
           - (1 + 2 * t1 + c1) * d ** 3 / 6
           + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * e ** 2 + 24 * t1 ** 2) * d ** 5 / 120) / math.cos(phi1)
    lon = math.degrees(math.radians(lon_0) + lon)

    return lat, lon


def laea_to_wgs84(x, y, lon_0, lat_0, false_easting, false_northing):
    # converts from Lambert Azimuthal Equal Area (LAEA) to WGS84

    R = 6378137.0  # Radius of the Earth in meters (WGS84)
    lat_0 = np.deg2rad(lat_0)  # Convert origin latitude to radians
    lon_0 = np.deg2rad(lon_0)  # Convert origin longitude to radians

    # Adjust for false easting and northing
    x_adj = x - false_easting
    y_adj = y - false_northing

    # Cartesian to spherical conversion
    p = np.sqrt(x_adj**2 + y_adj**2)
    c = 2 * np.arcsin(p / (2 * R))

    lat = np.arcsin(np.cos(c) * np.sin(lat_0) + y_adj * np.sin(c) * np.cos(lat_0) / p)
    lon = lon_0 + np.arctan2(x_adj * np.sin(c), p * np.cos(lat_0) * np.cos(c) - y_adj * np.sin(lat_0) * np.sin(c))

    return (np.rad2deg(lat), np.rad2deg(lon))


def lcc_to_wgs84(x, y, lon_0, lat_0, lat_1, lat_2, false_easting, false_northing):
    """
    Converts coordinates from a Lambert Conformal Conic (LCC) projection (EPSG:3057)
    to WGS84 (latitude/longitude).

    Args:
        x (np.ndarray): Easting coordinates.
        y (np.ndarray): Northing coordinates.
        lon_0 (float): Longitude of origin / Central Meridian in degrees.
        lat_0 (float): Latitude of origin in degrees.
        lat_1 (float): First standard parallel in degrees.
        lat_2 (float): Second standard parallel in degrees.
        false_easting (float): False easting value.
        false_northing (float): False northing value.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing latitude and longitude arrays.
    """
    # GRS80 Ellipsoid parameters (used by ISN93/EPSG:3057)
    a = 6378137.0
    f_inv = 298.257222101
    f = 1 / f_inv
    e2 = 2 * f - f**2
    e = np.sqrt(e2)

    # Convert degrees to radians
    lon_0_rad = np.deg2rad(lon_0)
    lat_0_rad = np.deg2rad(lat_0)
    lat_1_rad = np.deg2rad(lat_1)
    lat_2_rad = np.deg2rad(lat_2)

    def t_calc(phi_rad, e_val):
        sin_phi = np.sin(phi_rad)
        return np.tan(np.pi/4 - phi_rad/2) / ((1 - e_val*sin_phi)/(1 + e_val*sin_phi))**(e_val/2)

    m1 = np.cos(lat_1_rad) / np.sqrt(1 - e2 * np.sin(lat_1_rad)**2)
    m2 = np.cos(lat_2_rad) / np.sqrt(1 - e2 * np.sin(lat_2_rad)**2)

    t0 = t_calc(lat_0_rad, e)
    t1 = t_calc(lat_1_rad, e)
    t2 = t_calc(lat_2_rad, e)

    n = np.log(m1 / m2) / np.log(t1 / t2)
    F = m1 / (n * t1**n)
    rho_0 = a * F * t0**n

    # Adjust for false easting and northing
    x_adj = x - false_easting
    y_adj = rho_0 - (y - false_northing)

    rho_prime = np.sqrt(x_adj**2 + y_adj**2)
    
    # Handle case where rho_prime is zero
    rho_prime[rho_prime == 0] = 1e-10

    t_prime = (rho_prime / (a * F))**(1/n)

    # Iteratively solve for latitude
    phi = np.pi/2 - 2 * np.arctan(t_prime)
    for _ in range(5): # 5 iterations are generally sufficient
        sin_phi = np.sin(phi)
        phi_new = np.pi/2 - 2 * np.arctan(t_prime * ((1 - e*sin_phi)/(1 + e*sin_phi))**(e/2))
        if np.all(np.abs(phi_new - phi) < 1e-10):
            break
        phi = phi_new

    theta = np.arctan2(x_adj, y_adj)
    lon = theta / n + lon_0_rad

    return np.rad2deg(phi), np.rad2deg(lon)


def _lv03_to_wgs84_vec(east, north):
    """
    Vectorized conversion LV03 (EPSG:21781) -> WGS84 (EPSG:4326)
    east, north: scalar or numpy array (east = y, north = x in LV03 notation)
    returns (lon, lat)
    """
    east = np.asarray(east, dtype=float)
    north = np.asarray(north, dtype=float)

    y_aux = (east - 600000.0) / 1e6
    x_aux = (north - 200000.0) / 1e6

    lat = (16.9023892
           + 3.238272 * x_aux
           - 0.270978 * (y_aux**2)
           - 0.002528 * (x_aux**2)
           - 0.0447   * (y_aux**2) * x_aux
           - 0.0140   * (x_aux**3))

    lon = (2.6779094
           + 4.728982 * y_aux
           + 0.791484 * y_aux * x_aux
           + 0.1306   * y_aux * (x_aux**2)
           - 0.0436   * (y_aux**3))

    # convert from sexagesimal seconds to degrees
    lat = lat * 100.0 / 36.0
    lon = lon * 100.0 / 36.0

    return lon, lat


def epsg2056_point_to_wgs84(easting, northing):
    """
    EPSG:2056 (LV95) -> WGS84 lon/lat
    LV95 -> LV03 shift: easting - 2,000,000 ; northing - 1,000,000
    """
    # convert to LV03
    e_lv03 = np.asarray(easting, dtype=float) - 2_000_000.0
    n_lv03 = np.asarray(northing, dtype=float) - 1_000_000.0
    lon, lat = _lv03_to_wgs84_vec(e_lv03, n_lv03)

    return lat, lon
