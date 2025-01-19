
__all__ = ['netCDF4', 'plt', 'shapefile', 'xarray', 'matplotlib', 'easy_mpl', 'fiona', 'plt_Axes']

try:
    import netCDF4
except (ImportError, ModuleNotFoundError):
    netCDF4 = None

try:
    import matplotlib.pyplot as plt
    import matplotlib
except (ModuleNotFoundError, ImportError) as e:
    matplotlib, plt = None, None

if matplotlib is None:
    class plt_Axes: pass
else:
    plt_Axes = matplotlib.axes.Axes

try:
    import shapefile
except (ModuleNotFoundError, ImportError) as e:
    shapefile = None


try:
    import fiona
except (ModuleNotFoundError, ImportError):
    fiona = None


try:
    import xarray
except (ModuleNotFoundError, ImportError) as e:
    xarray = None


try:
    import shapely
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union
except (ModuleNotFoundError, OSError):
    shape, mapping, unary_union = None, None, None
    shapely = None


try:
    import easy_mpl
except (ModuleNotFoundError, ImportError) as e:
    easy_mpl = None