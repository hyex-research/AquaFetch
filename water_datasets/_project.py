
__all__ = ['utm_to_lat_lon']

import math



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


