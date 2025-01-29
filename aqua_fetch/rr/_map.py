
# specifier_name_units
# specifier = method/aggregation_type/height
# method = method/source of calculation
# aggregation_type = min, max, mean, total, sum etc.
# height = height of the measurement like 2m/10m etc.

# ****** Dynmaic Features *******

# %% streamflow

def observed_streamflow_cms()->str:
    return "q_cms_obs"


def simulated_streamflow_cms()->str:
    """
    cubic meter per second
    """
    return "q_cms_sim"


def observed_streamflow_mmd()->str:
    """mm/day"""
    return "q_mmd_obs"


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


# %% solar radiation

def solar_radiation()->str:
    """
    # todo: is it mean or total?
    also know as
    shortwave radiation
    downard shortwave radiation
    net solar radiation
    """
    return "solrad_wm2"


def solar_radiation_with_specifier(specifier:str)->str:
    """also know as
    shortwave radiation
    downard shortwave radiation
    net solar radiation
    """
    return f"solrad_wm2_{specifier}"


def max_solar_radiation()->str:
    """
    also know as
    shortwave radiation
    downard shortwave radiation
    net solar radiation
    """
    return "solrad_wm2_max"


def min_solar_radiation()->str:
    """
    also know as
    shortwave radiation
    downard shortwave radiation
    net solar radiation
    """
    return "solrad_wm2_min"


def downward_longwave_radiation()->str:
    return "lwdownrad_wm2"


def download_longwave_radiation_with_specifier(specifier:str)->str:
    return f"lwdownrad_wm2_{specifier}"


# %%
# thermal radiation
def mean_thermal_radiation()->str:
    return "thermrad_wm2"


def max_thermal_radiation()->str:
    return "thermrad_wm2_max"


def min_themal_radiation()->str:
    return "thermrad_wm2_min"


# %% 
# snow water equivalent depth

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


# %%

def leaf_area_index()->str:
    return "lai"


def groundwater_percentages()->str:
    return "gw_percent"


# %%
# soil moisture layer
# todo : is it same as soil water layer?

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
# ****STATIC FEATURES****

def catchment_area()->str:
    return "area_km2"

def gauge_latitude()->str:
    return "lat"

def gauge_longitude()->str:
    return "long"

def slope(unit)->str:
    return f"slope_{unit}"


def gauge_elevation_meters()->str:
    """elevation of the gauge station in meters"""
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


def urban_fraction()->str:
    """Fraction of urban area in the catchment"""
    return "urban_frac"


def forest_fraction()->str:
    """Fraction of forest area in the catchment"""
    return "forest_frac"


def grass_fraction()->str:
    """Fraction of grass area in the
    catchment"""
    return "grass_frac"


def impervious_fraction()->str:
    """Fraction of impervious area in the catchment"""
    return "imperv_frac"


def aridity_index()->str:
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