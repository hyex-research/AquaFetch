
# specifier_name_units
# specifier = method/aggregation_type/height
# method = method/source of calculation
# aggregation_type = min, max, mean, total, sum etc.

# ****** Dynmaic Features *******

# %% streamflow

def observed_streamflow_cms()->str:
    return "obs_q_cms"


def simulated_streamflow_cms()->str:
    return "sim_q_cms"


# %% precpiation

def total_precipitation()->str:
    return "pcp_mm"


def total_precipitation_with_method(method:str)->str:
    return f"{method}_pcp_mm"


# %% air temperature

def max_air_temp()->str:
    return "max_temp_C"


def max_air_temp_with_method(method:str)->str:
    return f"{method}_max_temp_C"


def min_air_temp()->str:
    return "min_temp_C"


def min_air_temp_with_method(method:str)->str:
    return f"{method}_min_temp_C"


def mean_air_temp()->str:
    return "mean_temp_C"


def mean_air_temp_with_method(method:str)->str:
    return f"{method}_mean_temp_C"

# evpotranspiration

def mean_potential_evapotranspiration()->str:
    # total: is it mean or total?
    return "pet_mm"

def mean_potential_evapotranspiration_with_method(method:str)->str:
    # total: is it mean or total?
    return f"{method}_pet_mm"


def total_potential_evapotranspiration()->str:
    return "pet_mm"


def total_potential_evapotranspiration_with_method(method:str)->str:
    return f"{method}_pet_mm"


def mean_potential_evaporation()->str:
    # total: is it mean or total?
    return "pevap_mm"


def mean_potential_evaporation_with_method(method:str)->str:
    # total: is it mean or total?
    return f"{method}_pevap_mm"


def actual_evapotranspiration()->str:
    # units are mm/day
    return "aet_mm"


def actual_evapotranspiration_with_method(method:str)->str:
    # units are mm/day
    return f"{method}_aet_mm"

# wind speed

def mean_windspeed()->str:
    return "windspeed_mps"


def mean_windspeed_with_method(method:str)->str:
    return f"{method}_windspeed_mps"


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


def u_component_of_wind_with_method(method:str)->str:
    """
    u component of wind speed todo : at which height?
    """
    return f"{method}_windspeedu_mps"


def v_component_of_wind()->str:
    """ v component of wind speed
    """
    return "windspeedv_mps"


def v_component_of_wind_with_method(method:str)->str:
    """ v component of wind speed
    """
    return f"{method}_windspeedv_mps"


# relative humidity

def mean_rel_hum()->str:
    return "rh_%"

# air pressure

def mean_air_pressure()->str:
    """air pressure in hector pascal"""
    return "airpres_hpa"


def min_air_pressure()->str:
    """air pressure in hector pascal"""
    return "min_airpres_hpa"


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


def solar_radiation_with_method(method:str)->str:
    """also know as
    shortwave radiation
    downard shortwave radiation
    net solar radiation
    """
    return f"{method}_solrad_wm2"


def max_solar_radiation()->str:
    """
    also know as
    shortwave radiation
    downard shortwave radiation
    net solar radiation
    """
    return "max_solrad_wm2"


def min_solar_radiation()->str:
    """
    also know as
    shortwave radiation
    downard shortwave radiation
    net solar radiation
    """
    return "max_solrad_wm2"


def downard_longwave_radiation()->str:
    return "lwdownrad_wmd2"


def download_longwave_radiation_with_method(method:str)->str:
    return f"{method}_lwdownrad_wmd2"


# thermal radiation
def mean_thermal_radiation()->str:
    return "thermrad_wm2"


def max_thermal_radiation()->str:
    return "max_thermrad_wm2"


# %% snow water equivalent

def snow_water_equivalent()->str:
    # is it total or mean?
    return "swe_mm"


def snow_water_equivalent_with_method(method:str)->str:
    # is it total or mean?
    return f"{method}_swe_mm"


def leaf_area_index()->str:
    return "lai"


def groundwater_percentages()->str:
    return "gw_percent"


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
    return "mean_dewpoint_C"


def mean_dewpoint_temperature_at_2m()->str:
    return "mean_2m_dewpoint_C"


def mean_dewpoint_temperature_with_method(method:str)->str:
    return f"{method}_mean_dewpoint_C"


def max_dewpoint_temperature()->str:
    return "max_dewpoint_C"


def max_dewpoint_temperature_at_2m()->str:
    return "max_2m_dewpoint_C"


def max_dewpoint_temperature_with_method(method:str)->str:
    return f"{method}_max_dewpoint_C"


def min_dewpoint_temperature()->str:
    return "min_dewpoint_C"


def min_dewpoint_temperature_at_2m()->str:
    return "min_2m_dewpoint_C"


def min_dewpoint_temperature_with_method(method:str)->str:
    return f"{method}_min_dewpoint_C"


# ****STATIC FEATURES****

def catchment_area()->str:
    return "area_km2"

def gauge_latitude()->str:
    return "lat"

def gauge_longitude()->str:
    return "long"

def slope(unit)->str:
    return f"slope_{unit}"
