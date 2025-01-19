
__all__ = ['utm_to_lat_lon', 'laea_to_wgs84']

import math

import numpy as np


def utm_to_lat_lon(easting, northing, zone:int):
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
