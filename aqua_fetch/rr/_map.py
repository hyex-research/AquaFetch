
# specifier_name_units
# specifier = method/aggregation_type/height
# method = method/source of calculation
# aggregation_type = min, max, mean, total, sum etc.
# height = height of the measurement like 2m/10m etc.

# TODO : differentiate between catchment averaged and temporal averaged features, the word 'mean' is ambiguous
# for example mean air temperature can mean catchment averaged or temporal averaged
# ****** Dynmaic Features *******

# %% streamflow

def observed_streamflow_cms()->str:
    return "q_cms_obs"


def simulated_streamflow_cms()->str:
    """
    cubic meter per second
    """
    return "q_cms_sim"


def observed_streamflow_mm()->str:
    """mm/timestep"""
    return "q_mm_obs"


def simulated_streamflow_mm()->str:
    """simulated (model) streamflow expressed as catchment-equivalent runoff
    depth in mm/timestep"""
    return "q_mm_sim"


def observed_water_level_cm()->str:
    """observed water level (gauge stage) in centimeters"""
    return "wl_cm_obs"


def observed_water_level_m()->str:
    """observed water level (gauge stage) in meters"""
    return "wl_m_obs"


def observed_water_level_ft()->str:
    """observed water level in feet. The reference datum is station-specific
    (e.g. stage above the local streambed/ground for stream gauges, or
    instantaneous surface level for a lake/reservoir gauge)."""
    return "wl_ft_obs"


# %% precpiation

def total_precipitation()->str:
    return "pcp_mm"


def total_precipitation_with_specifier(specifier:str)->str:
    return f"pcp_mm_{specifier}"


# %% 
# air temperature

def max_air_temp()->str:
    return "airtemp_C_max"


def max_air_temp_with_specifier(specifier:str)->str:
    return f"airtemp_C_{specifier}_max"


def min_air_temp()->str:
    return "airtemp_C_min"


def min_air_temp_with_specifier(specifier:str)->str:
    return f"airtemp_C_{specifier}_min"


def mean_air_temp()->str:
    # mean (daily) air temperature in degree celsius
    return "airtemp_C_mean"


def mean_air_temp_with_specifier(specifier:str)->str:
    return f"airtemp_C_mean_{specifier}"

# %%
# ground surface temperature

def mean_daily_ground_surface_temp()->str:
    return "gtemp_C"


def max_daily_ground_surface_temp()->str:
    return "gtemp_C_max"


def min_daily_ground_surface_temp()->str:
    return "gtemp_C_min"


# %%
# evpotranspiration

def mean_potential_evapotranspiration()->str:
    # total: is it mean or total?
    return "pet_mm"

def mean_potential_evapotranspiration_with_specifier(specifier:str)->str:
    # total: is it mean or total?
    return f"pet_mm_{specifier}"


def total_potential_evapotranspiration()->str:
    # mm/day
    return "pet_mm"


def total_potential_evapotranspiration_with_specifier(specifier:str)->str:
    # mm/day
    return f"pet_mm_{specifier}"


def mean_potential_evaporation()->str:
    # total: is it mean or total?
    return "pevap_mm"


def mean_potential_evaporation_with_specifier(specifier:str)->str:
    # total: is it mean or total?
    return f"pevap_mm_{specifier}"


def actual_evapotranspiration()->str:
    # units are mm/day
    return "aet_mm"


def actual_evapotranspiration_with_specifier(specifier:str)->str:
    """actual evapotranspiration units are mm/day"""
    return f"aet_mm_{specifier}"


def mean_daily_evaporation()->str:
    """catchment daily averaged evaporation (observations) mm/day"""
    return "evap_mm"


def mean_daily_evaporation_with_specifier(specifier:str)->str:
    """catchment daily averaged evaporation (observations) mm/day"""
    return f"evap_mm_{specifier}"

# %%
# wind speed

def mean_windspeed()->str:
    # daily averaged wind speed in meters per second
    return "windspeed_mps"


def mean_windspeed_with_specifier(specifier:str)->str:
    return f"windspeed_mps_{specifier}"


def max_windspeed()->str:
    return "windspeed_mps_max"


def min_windspeed()->str:
    return "windspeed_mps_min"


def max_wind_gust()->str:
    """maximum instantaneous wind speed (gust) in meters per second, e.g. daily
    maximum. Not the maximum of a sustained (averaged) wind speed, which is
    :func:`max_windspeed`."""
    return "windgust_mps_max"


def u_component_of_wind()->str:
    """
    u component of wind speed
    """
    return "windspeedu_mps"


def u_component_of_wind_at_10m()->str:
    """
    u component of wind speed at 10 meter height  # todo:
    """
    return "windspeedu_mps"


def v_component_of_wind_at_10m()->str:
    """
    v component of wind speed at 10 meter height  # todo:
    """
    return "windspeedv_mps"


def u_component_of_wind_with_specifier(specifier:str)->str:
    """
    u component of wind speed todo : at which height?
    """
    return f"windspeedu_mps_{specifier}"


def v_component_of_wind()->str:
    """ v component of wind speed
    """
    return "windspeedv_mps"


def v_component_of_wind_with_specifier(specifier:str)->str:
    """ v component of wind speed
    """
    return f"windspeedv_mps_{specifier}"


# %% relative humidity

def mean_rel_hum()->str:
    # mean relative humidity in percentage
    return "rh_%"


def rel_hum_with_specifier(specifier:str)->str:
    # in percentage
    return f"rh_%_{specifier}"


def mean_rel_hum_with_specifier(specifier:str)->str:
    """
    units are in percentage
    """
    return f"rh_%_{specifier}"


# mean specific humidity
def mean_specific_humidity()->str:
    return "spechum_gkg"


# %% air pressure
# todo: what is difference between surface pressure, air pressure and mean sea level pressure (CAMELS_AUS)?
# ground surface pressure (CCAM)

def mean_air_pressure()->str:
    """air pressure in hector pascal"""
    return "airpres_hpa"


def min_air_pressure()->str:
    """air pressure in hector pascal"""
    return "airpres_hpamin_"


# %% radiation
# ===========================================================================
# RADIATION NAMING CONVENTION
# ===========================================================================
#
#     <band><direction>rad_wm2 [_<stat>] [_<source>]
#
# Every radiation feature in the library is named by this formula. The first
# two slots are mandatory, the last two optional, and the order is fixed so a
# name can be parsed by position.
#
#   band       'sw' = shortwave, a.k.a. SOLAR      (~0.3-4 um, from the sun)
#              'lw' = longwave,  a.k.a. THERMAL    (~4-100 um, emitted by the
#                                                   surface and atmosphere)
#              'solar'/'shortwave' and 'thermal'/'longwave' are SYNONYMS, not
#              separate axes. The two vocabularies come from different
#              communities (ECMWF says solar/thermal, hydrology says sw/lw);
#              this library uses sw/lw throughout.
#
#   direction  'down' = flux arriving at the surface
#              'up'   = flux leaving the surface (reflected for sw, emitted
#                       for lw). Only LamaH-Ice ships one (outgoing longwave),
#                       left under its source name; see LamaHIce.dyn_map.
#              'net'  = down - up, POSITIVE TOWARD THE SURFACE (see SIGN below)
#
#   rad_wm2    invariant. 'rad' keeps the family greppable and, critically,
#              stops 'sw' colliding with swe_mm (snow water equivalent) and
#              with the 'soil water' reading of sw. It is fused to the
#              quantity rather than left as its own field so that the unit
#              stays at index 1 of name.split('_'), matching every other name
#              in this module (airtemp_C_mean, q_cms_obs, spechum_gkg).
#
#   stat       temporal (default, unmarked): mean min max med std
#              spatial  (explicitly marked) : spatmean spatmin spatmax
#                                             spatmed spatstd
#              window                       : daylight
#              A bare name is ALREADY a double mean -- the catchment spatial
#              mean of the timestep temporal mean -- so a dataset publishing
#              only a catchment mean uses the bare name and adds no token.
#              Temporal is unmarked because airtemp_C_max already means the
#              within-day maximum. The distinction matters: a within-day max
#              of shortwave is ~800 W m-2 (noon), a catchment-spatial max is
#              ~210 W m-2 (the mean plus grid noise) -- a factor of four.
#
#   source     silo awap agcd era5 era5land merra2 ens ref daymet ...
#              Disambiguates products WITHIN one dataset. Its absence does not
#              mean the provenance is unknown.
#
# Giving 6 quantities, of which rr populates 4:
#
#              |  shortwave (solar)   |  longwave (thermal)
#     ---------+----------------------+---------------------
#     down     |  swdownrad_wm2       |  lwdownrad_wm2
#     up       |  (reserved)          |  (reserved)
#     net      |  swnetrad_wm2        |  lwnetrad_wm2
#
# UNIT CONTRACT: the trailing ``wm2`` is a promise, not a label. A dataset
# whose native units are anything else (MJ m-2 day-1, J cm-2 day-1,
# J m-2 day-1 -- all of which occur) must declare a conversion in its
# ``dyn_factors``. Constants are provided below.
#
# SIGN CONVENTION: ``net`` is down minus up, positive toward the surface (CF).
# Net shortwave is therefore positive; net longwave is predominantly NEGATIVE
# (the surface loses heat to the sky), typically -20 to -100 W m-2 as a daily
# mean.
#
# Some sources publish net longwave positive-UPWARD instead, as a heat loss.
# That is a different convention, not an error, and this library does NOT
# normalise it: negating a published series would alter the data rather than
# re-express it, and only unit conversion is permitted here. Such a dataset's
# columns are instead left under their SOURCE names -- mapping them to
# ``lwnetrad_wm2`` would assert a convention the data does not follow -- and the
# class warns unconditionally so the user can convert if they choose.
# LamaH-CE is currently the only such dataset; see LamaHCE._warn_thermal_sign.
#
# DOWNWARD IS NOT NET: net = downward x (1 - albedo). Over 120 randomly
# sampled HYSETS watersheds, where both come from the same ERA5 fields, the
# net/downward ratio has a per-station median of 0.824 -- i.e. an albedo of
# about 0.18 -- so the two are never interchangeable. Downward shortwave is a forcing and is comparable across
# datasets; net shortwave embeds the producing model's albedo.
#
# ``swdownrad_wm2`` means global horizontal irradiance (direct + diffuse).
# Top-of-atmosphere, clear-sky, direct-normal and diffuse variants have no
# slot yet; add 'toa'/'clearsky'/'dni'/'diffuse' tokens if a dataset ships one.
# ===========================================================================

# Radiation unit conversions to the canonical W m-2. Defined here, next to the
# names that promise those units, so that a dataset never hard-codes a factor
# whose meaning has to be re-derived by the next reader.
MJ_M2_DAY_TO_WM2 = 1e6 / 86400.0   # 11.574074...   CAMELS_AUS (SILO/AWAP), CABra, CAMELS_PE, CAMELS_KR
KJ_M2_DAY_TO_WM2 = 1e3 / 86400.0   # 0.011574074... CAMELS_FI
J_CM2_DAY_TO_WM2 = 1e4 / 86400.0   # 0.11574074...  CAMELS_FR (SAFRAN)
J_M2_DAY_TO_WM2 = 1.0 / 86400.0    # ERA5/ERA5-Land daily accumulations (HYSETS)

# controlled vocabularies -- a token outside these is a bug, not a new feature
RAD_BANDS = ('sw', 'lw')
RAD_DIRECTIONS = ('down', 'up', 'net')
RAD_STATS = ('mean', 'min', 'max', 'med', 'std',
             'spatmean', 'spatmin', 'spatmax', 'spatmed', 'spatstd',
             'daylight')


def _rad(band: str, direction: str, stat: str = None, specifier: str = None) -> str:
    """
    Builds a radiation feature name from the convention documented above.

    Raises rather than asserts: ``assert`` is stripped under ``python -O``, and
    these checks are what stop two different quantities being minted under one
    name. A guard that disappears in optimised runs is not a guard.
    """
    if band not in RAD_BANDS:
        raise ValueError(f"unknown radiation band {band!r}, expected one of {RAD_BANDS}")
    if direction not in RAD_DIRECTIONS:
        raise ValueError(f"unknown radiation direction {direction!r}, "
                         f"expected one of {RAD_DIRECTIONS}")
    name = f"{band}{direction}rad_wm2"
    if stat is not None:
        if stat not in RAD_STATS:
            raise ValueError(f"unknown radiation stat {stat!r}, "
                             f"expected one of {RAD_STATS}")
        name = f"{name}_{stat}"
    if specifier is not None:
        if specifier in RAD_STATS:
            raise ValueError(
                f"{specifier!r} is an aggregation token, not a source; pass it "
                f"as `stat` instead, otherwise two different quantities get the "
                f"same name")
        name = f"{name}_{specifier}"
    return name


# --------------------------------------------------------------- shortwave
def solar_radiation() -> str:
    """Downward shortwave (solar) radiation, W m-2. Global horizontal irradiance."""
    return _rad('sw', 'down')


def solar_radiation_with_specifier(specifier: str) -> str:
    """Downward shortwave radiation from a named source, e.g. ``silo``, ``era5``."""
    return _rad('sw', 'down', specifier=specifier)


def max_solar_radiation() -> str:
    """Maximum (within the timestep) downward shortwave radiation, W m-2."""
    return _rad('sw', 'down', stat='max')


def min_solar_radiation() -> str:
    """Minimum (within the timestep) downward shortwave radiation, W m-2."""
    return _rad('sw', 'down', stat='min')


def solar_radiation_with_spatial_stat(stat: str) -> str:
    """
    Downward shortwave radiation aggregated ACROSS THE CATCHMENT, W m-2.

    ``stat`` is one of ``min``, ``max``, ``med``, ``std``. Note these describe
    the spread of the gridded forcing over the catchment, so they scale with
    catchment area and grid resolution rather than with climate; treat them as
    catchment metadata, not as forcing.
    """
    return _rad('sw', 'down', stat=f'spat{stat}')


def daylight_solar_radiation() -> str:
    """
    Downward shortwave radiation averaged over the DAYLIGHT hours only, W m-2.

    This is Daymet's native ``srad`` and is NOT comparable with the 24-h mean
    that :func:`solar_radiation` returns -- it is larger by 1/(daylight
    fraction), roughly 1.5-2.5x. Convert with ``srad * dayl / 86400``.
    """
    return _rad('sw', 'down', stat='daylight')


def net_solar_radiation() -> str:
    """Net shortwave (solar) radiation, W m-2. Downward minus reflected."""
    return _rad('sw', 'net')


def net_solar_radiation_with_specifier(specifier: str) -> str:
    """Net shortwave radiation from a named source, e.g. ``era5``, ``merra2``."""
    return _rad('sw', 'net', specifier=specifier)


def max_net_solar_radiation() -> str:
    """Maximum (within the timestep) net shortwave radiation, W m-2."""
    return _rad('sw', 'net', stat='max')


def min_net_solar_radiation() -> str:
    """Minimum (within the timestep) net shortwave radiation, W m-2."""
    return _rad('sw', 'net', stat='min')


# --------------------------------------------------------------- longwave
def downward_longwave_radiation() -> str:
    """Downward longwave (thermal) radiation, W m-2."""
    return _rad('lw', 'down')


def downward_longwave_radiation_with_specifier(specifier: str) -> str:
    """Downward longwave radiation from a named source."""
    return _rad('lw', 'down', specifier=specifier)


def net_longwave_radiation() -> str:
    """
    Net longwave (thermal) radiation, W m-2, positive toward the surface.
    Predominantly negative: the surface loses heat to the sky.
    """
    return _rad('lw', 'net')


def net_longwave_radiation_with_specifier(specifier: str) -> str:
    """Net longwave radiation from a named source, e.g. ``era5``, ``merra2``."""
    return _rad('lw', 'net', specifier=specifier)


def max_net_longwave_radiation() -> str:
    """Maximum (within the timestep) net longwave radiation, W m-2."""
    return _rad('lw', 'net', stat='max')


def min_net_longwave_radiation() -> str:
    """Minimum (within the timestep) net longwave radiation, W m-2."""
    return _rad('lw', 'net', stat='min')


# %% 
# snow water equivalent, depth, density

def snow_depth()->str:
    return "snowdepth_m"

def snow_water_equivalent()->str:
    # is it total or mean?
    return "swe_mm"


def snow_water_equivalent_with_specifier(specifier:str)->str:
    # is it total or mean?
    return f"swe_mm_{specifier}"


def max_snow_water_equivalent()->str:
    return "swe_mm_max"


def min_snow_water_equivalent()->str:
    return "swe_mm_min"


def snowfall()->str:
    """total snowfall mm per units of time"""
    return "snowfall_mm"


def snowmelt()->str:
    """total snowmelt mm per units of time"""
    return "snowmelt_mm"


def snow_density()->str:
    """Average daily snow density in kg m-3"""
    return "snowdensity_kgm3"

# %%

def leaf_area_index()->str:
    return "lai"


def groundwater_percentages()->str:
    return "gw_percent"


# %%
# soil moisture layer
# todo : is it same as soil water layer? 
# in section 2.6 of CAMELS-LUX documentation, it is mentioned that

def soil_moisture_layer1()->str:
    """ m3/m3"""
    return "sml1"


def soil_moisture_layer2()->str:
    """ m3/m3"""
    return "sml2"


def soil_moisture_layer3()->str:
    """ m3/m3"""
    return "sml3"


def soil_moisture_layer4()->str:
    """ m3/m3"""
    return "sml4"


# %% dew point temperature

def mean_dewpoint_temperature()->str:
    return "dptemp_C_mean"


def mean_dewpoint_temperature_at_2m()->str:
    return "dptemp_C_mean_2m"


def mean_dewpoint_temperature_with_specifier(specifier:str)->str:
    return f"dptemp_C_mean_{specifier}"


def max_dewpoint_temperature()->str:
    return "dptemp_C_max"


def max_dewpoint_temperature_at_2m()->str:
    return "dptemp_C_max_2m"


def max_dewpoint_temperature_with_specifier(specifier:str)->str:
    return f"dptemp_C_max_{specifier}"


def min_dewpoint_temperature()->str:
    return "dptemp_C_min"


def min_dewpoint_temperature_at_2m()->str:
    return "dptemp_C_min_2m"


def min_dewpoint_temperature_with_specifier(specifier:str)->str:
    return f"dptemp_C_min_{specifier}"

# %%
#  vapor pressure

def mean_vapor_pressure()->str:
    return "vp_hpa"


def mean_vapor_pressure_with_specifier(specifier)->str:
    return f"vp_hpa_{specifier}"


# %%
# sunshine duration

def sunshine_duration()->str:
    return "ssd_hr"

# %%
# cloud cover

def cloud_cover()->str:
    return "cloudcover"

# %%
# ****STATIC FEATURES****

def catchment_area()->str:
    return "area_km2"


def catchment_area_with_specifier(specifier:str)->str:
    """catchment area in square kilometers"""
    return f"area_km2_{specifier}"


def catchment_perimeter()->str:
    """Catchment perimeter in kilometers"""
    return "perimeter_km"


def gauge_latitude()->str:
    """in units of WGS84 (degrees)"""
    return "lat"

def gauge_longitude()->str:
    """in units of WGS84 (degrees)"""
    return "long"


def slope(unit)->str:
    """Average slope of the catchment"""
    return f"slope_{unit}"


def gauge_elevation_meters()->str:
    """elevation of the gauge station in meters (m a.s.l)"""
    return "elev_gauge_m"


def catchment_elevation_meters()->str:
    """mean elevation of the catchment in meters"""
    return "elev_catch_m"


def min_catchment_elevation_meters()->str:
    """minimum elevation of the catchment in meters"""
    return "elev_catch_min_m"


def max_catchment_elevation_meters()->str:
    """maximum elevation of the catchment in meters"""
    return "elev_catch_max_m"


def med_catchment_elevation_meters()->str:
    """median elevation of the catchment in meters"""
    return "elev_catch_med_m"


def urban_fraction()->str:
    """Fraction of urban area in the catchment"""
    return "urban_frac"


def urban_fraction_with_specifier(specifier:str)->str:
    """Fraction of urban area in the catchment with a specifier such as year"""
    return f"urban_frac_{specifier}"


def forest_fraction()->str:
    """Fraction of forest area in the catchment"""
    return "forest_frac"


def forest_fraction_with_specifier(specifier:str)->str:
    """Fraction of forest area in the catchment with a specifier such as year"""
    return f"forest_frac_{specifier}"


def grass_fraction()->str:
    """Fraction of grass area in the catchment"""
    return "grass_frac"


def grass_fraction_with_specifier(specifier:str)->str:
    """Fraction of grass area in the catchment with a specifier such as year.
    """
    return f"grass_frac_{specifier}"


def crop_fraction()->str:
    """Fraction of cropland area in the catchment.
    In CAMELS-LUX it is named as 'agricultural_land'
    """
    return "crop_frac"

def crop_fraction_with_specifier(specifier:str)->str:
    """Fraction of cropland area in the catchment with a specifier such as year.
    In CAMELS-LUX it is named as 'agricultural_land'
    """
    return f"crop_frac_{specifier}"


def impervious_fraction()->str:
    """Fraction of impervious area in the catchment"""
    return "imperv_frac"


def aridity_index()->str:
    """the ratio of mean daily & ERA5-Land potential 
    evapotranspiration to mean daily precipitation"""
    return "aridity"


def gauge_density()->str:
    return "gauge_density"


def baseflow_index()->str:
    return "bfi"


def catchment_centroid_latitude()->str:
    return "lat_catch"


def catchment_centroid_longitude()->str:
    return "long_catch"


def elong_ratio()->str:
    return "elong_ratio"


def silt_percentage() -> str:
    """Percentage of silt dominated soils of total area in %"""
    return "silt_perc"


def clay_percentage() -> str:
    """Percentage of clay dominated soils of total area in %"""
    return "clay_perc"


def soil_depth() -> str:
    """Mean soil depth to bedrock in meters"""
    return "soil_depth_m"


def population_density(year:int=None) -> str:
    """Population density in people per square kilometer"""
    if year:
        return f"pop_density_{year}_km2"
    return "pop_density_km2"


def population_density_with_specifier(specifier: str) -> str:
    """Population density in people per square kilometer with a specifier such as year"""
    return f"pop_density_{specifier}_km2"
