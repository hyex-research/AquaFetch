

# method/AggregationType_name_units

def max_daily_air_temp()->str:
    return "max_temp_C"


def min_daily_air_temp()->str:
    return "min_temp_C"


def mean_daily_air_temp()->str:
    return "mean_temp_C"


def mean_daily_air_temp_with_method(method:str)->str:
    return f"{method}_mean_temp_C"

def mean_daily_precipitation()->str:
    return "pcp_mm"


def mean_daily_precipitation_with_method(method:str)->str:
    # todo : is it mean or total?
    return f"{method}_pcp_mm"


def mean_daily_windspeed()->str:
    return "windspeed_mps"


def mean_daily_windspeed_with_method(method:str)->str:
    return f"{method}_windspeed_mps"


def u_component_of_daily_wind()->str:
    """ u component of wind speed
    """
    return "windspeedu_mps"


def u_component_of_daily_wind_with_method(method:str)->str:
    """ u component of wind speed
    """
    return f"{method}_windspeedu_mps"


def v_component_of_daily_wind()->str:
    """ v component of wind speed
    """
    return "windspeedv_mps"


def v_component_of_daily_wind_with_method(method:str)->str:
    """ v component of wind speed
    """
    return f"{method}_windspeedv_mps"


def mean_daily_rel_hum()->str:
    return "rh_%"


def mean_daily_potential_evpotranspiration()->str:
    # total: is it mean or total?
    return "pet_mm"

def daily_potential_evpotranspiration_with_method(method:str)->str:
    # total: is it mean or total?
    return f"{method}_pet_mm"

def daily_actual_evpotranspiration()->str:
    # units are mm/day
    return "aet_mm"

def daily_actual_evpotranspiration_with_method(method:str)->str:
    # units are mm/day
    return f"{method}_aet_mm"


def mean_daily_air_pressure()->str:
    return "airpres_hpa"


def min_daily_air_pressure()->str:
    return "min_airpres_hpa"


def solar_radiation()->str:
    """also know as
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


def downard_longwave_radiation()->str:
    return "lwdownrad_wmd2"


def download_longwave_radiation_with_method(method:str)->str:
    return f"{method}_lwdownrad_wmd2"


def snow_water_equivalent()->str:
    return "swe_mm"


def snow_water_equivalent_with_method(method:str)->str:
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


# ****STATIC FEATURES****

def catchment_area()->str:
    return "area_km2"

def gauge_latitude()->str:
    return "lat"

