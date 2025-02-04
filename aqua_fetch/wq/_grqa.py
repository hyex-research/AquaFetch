
__all__ = ["GRQA"]

import os
from typing import Union, List

import numpy as np
import pandas as pd

from .._datasets import Datasets
from ..utils import check_st_en, check_attributes


DTYPES = {
    'BOD5': {
        'obs_id': str,
        'obs_time_zone': str, 'site_id': str, 'site_name': str, 'site_country': str,
        'upstream_basin_area': np.float32, 'upstream_basin_area_unit': str,
             'drainage_region_name': str,
             'param_code': str, 'source_param_code': str, 'param_name': str, 'source_param_name': str,
             'obs_value': np.float32, 'source_obs_value': np.float32,
             'detection_limit_flag': str,
             'WATERBASE_meta_procedureAnalysedMatrix': str,
             'WATERBASE_meta_Remarks': str, 
             'WQP_meta_ResultAnalyticalMethod_MethodName': str,
             'WQP_meta_ResultLaboratoryCommentText': str,
             },
    'BOD': {
        'GEMSTAT_meta_Station_Narrative': str,
        'GEMSTAT_meta_Parameter_Description': str,
        'GEMSTAT_meta_Analysis_Method_Code': str,
        'GEMSTAT_meta_Method_Name': str, 'GEMSTAT_meta_Method_Description': str
        },
    'COD': {
        'GEMSTAT_meta_Station_Narrative': str,
        'GEMSTAT_meta_Parameter_Description': str,
        'GEMSTAT_meta_Analysis_Method_Code': str,
        'GEMSTAT_meta_Method_Name': str, 'GEMSTAT_meta_Method_Description': str        
    },
    'DIC': {
        'obs_id': str,
        'obs_time_zone': str, 'site_id': str, 'site_name': str, 'site_country': str,
        'upstream_basin_area': np.float32, 'upstream_basin_area_unit': str,
             'drainage_region_name': str,
             'param_code': str, 'source_param_code': str, 'param_name': str, 'source_param_name': str,
             'obs_value': np.float32, 'source_obs_value': np.float32,
             'detection_limit_flag': str,
             'source': str,
             'filtration': str,
             'obs_percentile': np.float32,
             'site_ts_continuity': np.float32,
             'GEMSTAT_meta_Station_Narrative': str, 
             'GEMSTAT_meta_Parameter_Description': str,
             'GEMSTAT_meta_Analysis_Method_Code': str,
             'GEMSTAT_meta_Method_Name': str,
             'GEMSTAT_meta_Method_Description': str,
             'GLORICH_meta_Value_remark_code': str,
             'GLORICH_meta_Meaning': str,
             'WQP_meta_ResultAnalyticalMethod_MethodName': str,
             'WQP_meta_ResultLaboratoryCommentText': str,
    },
    'DIP': {
        'obs_id': str,
        'obs_time_zone': str, 'site_id': str, 'site_name': str, 'site_country': str,
        'upstream_basin_area': np.float32, 'upstream_basin_area_unit': str,        
             'drainage_region_name': str,
             'param_code': str, 'source_param_code': str, 'param_name': str, 'source_param_name': str,
             'obs_value': np.float32, 'source_obs_value': np.float32,
             'detection_limit_flag': str,        
        'site_country': str,
             'filtration': str,
             'obs_percentile': np.float32,
             'site_ts_continuity': np.float32,
             'GEMSTAT_meta_Station_Narrative': str, 
             'GEMSTAT_meta_Parameter_Description': str,
             'GEMSTAT_meta_Analysis_Method_Code': str,
             'GEMSTAT_meta_Method_Name': str,
             'GEMSTAT_meta_Method_Description': str,
             'GLORICH_meta_Value_remark_code': str,
             'GLORICH_meta_Meaning': str,
    },
    'DKN': {
        'obs_id': str,
        'obs_time_zone': str, 'site_id': str, 'site_name': str, 'site_country': str,
        'upstream_basin_area': np.float32, 'upstream_basin_area_unit': str,        
             'drainage_region_name': str,
             'param_code': str, 'source_param_code': str, 'param_name': str, 'source_param_name': str,
             'obs_value': np.float32, 'source_obs_value': np.float32,
             'detection_limit_flag': str,        
        'site_country': str,
             'filtration': str,
             'obs_percentile': np.float32,
             'site_ts_continuity': np.float32,
             'GEMSTAT_meta_Station_Narrative': str, 
             'GEMSTAT_meta_Parameter_Description': str,
             'GEMSTAT_meta_Analysis_Method_Code': str,
             'GEMSTAT_meta_Method_Name': str,
             'GEMSTAT_meta_Method_Description': str,
             'GLORICH_meta_Value_remark_code': str,
             'GLORICH_meta_Meaning': str,        
    },
    'DOC': {
        'obs_id': str,
        'obs_time_zone': str, 'site_id': str, 'site_name': str, 'site_country': str,
        'upstream_basin_area': np.float32, 'upstream_basin_area_unit': str,        
             'drainage_region_name': str,
             'param_code': str, 'source_param_code': str, 'param_name': str, 'source_param_name': str,
             'obs_value': np.float32, 'source_obs_value': np.float32,
             'detection_limit_flag': str,        
            'site_country': str,
             'filtration': str,
             'obs_percentile': np.float32,
             'site_ts_continuity': np.float32,
             'GEMSTAT_meta_Station_Narrative': str, 
             'GEMSTAT_meta_Parameter_Description': str,
             'GEMSTAT_meta_Analysis_Method_Code': str,
             'GEMSTAT_meta_Method_Name': str,
             'GEMSTAT_meta_Method_Description': str,
             'GLORICH_meta_Value_remark_code': str,
             'GLORICH_meta_Meaning': str,          
             'WATERBASE_meta_procedureAnalysedMatrix': str,
             'WATERBASE_meta_Remarks': str,         
             'WQP_meta_ResultAnalyticalMethod_MethodName': str,
             'WQP_meta_ResultLaboratoryCommentText': str,
    },
    'DON': {
            'obs_id': str,
             'obs_time_zone': str, 'site_id': str, 'site_name': str, 'site_country': str,
             'upstream_basin_area': np.float32, 'upstream_basin_area_unit': str,        
             'drainage_region_name': str,
             'param_code': str, 'source_param_code': str, 'param_name': str, 'source_param_name': str,
             'obs_value': np.float32, 'source_obs_value': np.float32,
             'detection_limit_flag': str,        
             'site_country': str,
             'filtration': str,
             'obs_percentile': np.float32,
             'site_ts_continuity': np.float32,        
             'GEMSTAT_meta_Station_Narrative': str, 
             'GEMSTAT_meta_Parameter_Description': str,
             'GEMSTAT_meta_Analysis_Method_Code': str,
             'GEMSTAT_meta_Method_Name': str,
             'GEMSTAT_meta_Method_Description': str,
             'GLORICH_meta_Value_remark_code': str,
             'GLORICH_meta_Meaning': str,          
             'WQP_meta_ResultAnalyticalMethod_MethodName': str,
             'WQP_meta_ResultLaboratoryCommentText': str,
    },
    'DOSAT': {
            'obs_id': str,
             'obs_time_zone': str, 'site_id': str, 'site_name': str, 'site_country': str,
             'upstream_basin_area': np.float32, 'upstream_basin_area_unit': str,        
             'drainage_region_name': str,
             'param_code': str, 'source_param_code': str, 'param_name': str, 'source_param_name': str,
             'obs_value': np.float32, 'source_obs_value': np.float32,
             'detection_limit_flag': str,        
             'site_country': str,
             'filtration': str,
             'obs_percentile': np.float32,
             'site_ts_continuity': np.float32,        
             'GEMSTAT_meta_Station_Narrative': str, 
             'GEMSTAT_meta_Parameter_Description': str,
             'GEMSTAT_meta_Analysis_Method_Code': str,
             'GEMSTAT_meta_Method_Name': str,
             'GEMSTAT_meta_Method_Description': str,
             'GLORICH_meta_Value_remark_code': str,
             'GLORICH_meta_Meaning': str,       
             'WATERBASE_meta_procedureAnalysedMatrix': str,
             'WATERBASE_meta_Remarks': str,                 
             'WQP_meta_ResultAnalyticalMethod_MethodName': str,
             'WQP_meta_ResultLaboratoryCommentText': str,    
    },
    'NH4N': {
        'obs_id': str,
        'obs_time_zone': str, 'site_id': str, 'site_name': str, 'site_country': str,
        'upstream_basin_area': np.float32, 'upstream_basin_area_unit': str,        
             'drainage_region_name': str,
             'param_code': str, 'source_param_code': str, 'param_name': str, 'source_param_name': str,
             'obs_value': np.float32, 'source_obs_value': np.float32,
             'detection_limit_flag': str,        
            'site_country': str,
             'filtration': str,
             'obs_percentile': np.float32,
             'site_ts_continuity': np.float32,
             'GEMSTAT_meta_Station_Narrative': str, 
             'GEMSTAT_meta_Parameter_Description': str,
             'GEMSTAT_meta_Analysis_Method_Code': str,
             'GEMSTAT_meta_Method_Name': str,
             'GEMSTAT_meta_Method_Description': str,
             'GLORICH_meta_Value_remark_code': str,
             'GLORICH_meta_Meaning': str,          
             'WATERBASE_meta_procedureAnalysedMatrix': str,
             'WATERBASE_meta_Remarks': str,              
    },
'NO2N': {
            'obs_id': str,
             'obs_time_zone': str, 'site_id': str, 'site_name': str, 'site_country': str,
             'upstream_basin_area': np.float32, 'upstream_basin_area_unit': str,        
             'drainage_region_name': str,
             'param_code': str, 'source_param_code': str, 'param_name': str, 'source_param_name': str,
             'obs_value': np.float32, 'source_obs_value': np.float32,
             'detection_limit_flag': str,        
             'site_country': str,
             'filtration': str,
             'obs_percentile': np.float32,
             'site_ts_continuity': np.float32,        
             'GEMSTAT_meta_Station_Narrative': str, 
             'GEMSTAT_meta_Parameter_Description': str,
             'GEMSTAT_meta_Analysis_Method_Code': str,
             'GEMSTAT_meta_Method_Name': str,
             'GEMSTAT_meta_Method_Description': str,
             'GLORICH_meta_Value_remark_code': str,
             'GLORICH_meta_Meaning': str,          
             'WATERBASE_meta_procedureAnalysedMatrix': str,
             'WATERBASE_meta_Remarks': str,               
             'WQP_meta_ResultAnalyticalMethod_MethodName': str,
             'WQP_meta_ResultLaboratoryCommentText': str,   
             },
    'NO3N': {
        'obs_id': str,
        'obs_time_zone': str, 'site_id': str, 'site_name': str, 'site_country': str,
        'upstream_basin_area': np.float32, 'upstream_basin_area_unit': str,        
        'drainage_region_name': str,
        'param_code': str, 'source_param_code': str, 'param_name': str, 'source_param_name': str,
        'obs_value': np.float32, 'source_obs_value': np.float32,
        'detection_limit_flag': str,        
        'site_country': str,
        'filtration': str,
        'obs_percentile': np.float32,
        'site_ts_continuity': np.float32,        
        'GEMSTAT_meta_Station_Narrative': str, 
        'GEMSTAT_meta_Parameter_Description': str,
        'GEMSTAT_meta_Analysis_Method_Code': str,
        'GEMSTAT_meta_Method_Name': str,
        'GEMSTAT_meta_Method_Description': str,
        'GLORICH_meta_Value_remark_code': str,
        'GLORICH_meta_Meaning': str,          
        'WATERBASE_meta_procedureAnalysedMatrix': str,
        'WATERBASE_meta_Remarks': str,               
        'WQP_meta_ResultAnalyticalMethod_MethodName': str,
        'WQP_meta_ResultLaboratoryCommentText': str,   
    },
    'pH': {
        'obs_id': str,
        'obs_time_zone': str, 'site_id': str, 'site_name': str, 'site_country': str,
        'upstream_basin_area': np.float32, 'upstream_basin_area_unit': str,        
        'drainage_region_name': str,
        'param_code': str, 'source_param_code': str, 'param_name': str, 'source_param_name': str,
        'obs_value': np.float32, 'source_obs_value': np.float32,
        'detection_limit_flag': str,        
        'site_country': str,
        'source_unit': str,
        'filtration': str,
        'obs_percentile': np.float32,
        'site_ts_continuity': np.float32,        
        'GEMSTAT_meta_Station_Narrative': str, 
        'GEMSTAT_meta_Parameter_Description': str,
        'GEMSTAT_meta_Analysis_Method_Code': str,
        'GEMSTAT_meta_Method_Name': str,
        'GEMSTAT_meta_Method_Description': str,
        'GLORICH_meta_Value_remark_code': str,
        'GLORICH_meta_Meaning': str,          
        'WATERBASE_meta_procedureAnalysedMatrix': str,
        'WATERBASE_meta_Remarks': str,               
        'WQP_meta_ResultAnalyticalMethod_MethodName': str,
        'WQP_meta_ResultLaboratoryCommentText': str,   
    },
    'PN': {
        'drainage_region_name': str,
        'param_code': str, 'source_param_code': str, 
        'param_name': str, 'source_param_name': str,
        'obs_value': np.float32, 
        'source_obs_value': np.float32,
        'detection_limit_flag': str,        
        'site_country': str,
        'source_unit': str,
        'filtration': str,
        'obs_percentile': np.float32,
        'site_ts_continuity': np.float32,           
        'GEMSTAT_meta_Station_Narrative': str, 
        'GEMSTAT_meta_Parameter_Description': str,
        'GEMSTAT_meta_Analysis_Method_Code': str,
        'GEMSTAT_meta_Method_Name': str,
        'GEMSTAT_meta_Method_Description': str,
        'GLORICH_meta_Value_remark_code': str,
        'GLORICH_meta_Meaning': str,          
        'WQP_meta_ResultAnalyticalMethod_MethodName': str,
        'WQP_meta_ResultLaboratoryCommentText': str, 
    },
    'POC': {
        'obs_id': str,
        'obs_time_zone': str, 
        'site_id': str, 
        'site_name': str, 
        'site_country': str,
        'upstream_basin_area': np.float32, 
        'upstream_basin_area_unit': str,            
        'drainage_region_name': str,
        'param_code': str, 'source_param_code': str, 
        'param_name': str, 'source_param_name': str,
        'obs_value': np.float32, 
        'source_obs_value': np.float32,
        'detection_limit_flag': str,        
        'site_country': str,
        'source_unit': str,
        'filtration': str,
        'obs_percentile': np.float32,
        'site_ts_continuity': np.float32,           
        'GEMSTAT_meta_Station_Narrative': str, 
        'GEMSTAT_meta_Parameter_Description': str,
        'GEMSTAT_meta_Analysis_Method_Code': str,
        'GEMSTAT_meta_Method_Name': str,
        'GEMSTAT_meta_Method_Description': str,
        'GLORICH_meta_Value_remark_code': str,
        'GLORICH_meta_Meaning': str,          
        'WQP_meta_ResultAnalyticalMethod_MethodName': str,
        'WQP_meta_ResultLaboratoryCommentText': str, 
    },
    'TAN': {
        'upstream_basin_area': np.float32, 
        'upstream_basin_area_unit': str,            
        'drainage_region_name': str,
        'param_code': str, 'source_param_code': str, 
        'param_name': str, 'source_param_name': str,
        'obs_value': np.float32, 
        'source_obs_value': np.float32,
        'detection_limit_flag': str,        
        'site_country': str,
        'source_unit': str,
        'filtration': str,        
    },
    'TDN': {
        'obs_id': str,
        'obs_time_zone': str, 
        'site_id': str, 
        'site_name': str, 
        'site_country': str,
        'upstream_basin_area': np.float32, 
        'upstream_basin_area_unit': str,            
        'drainage_region_name': str,
        'param_code': str, 'source_param_code': str, 
        'param_name': str, 'source_param_name': str,
        'obs_value': np.float32, 
        'source_obs_value': np.float32,
        'detection_limit_flag': str,        
        'site_country': str,
        'source_unit': str,
        'filtration': str,
        'obs_percentile': np.float32,
        'site_ts_continuity': np.float32,           
        'GEMSTAT_meta_Station_Narrative': str, 
        'GEMSTAT_meta_Parameter_Description': str,
        'GEMSTAT_meta_Analysis_Method_Code': str,
        'GEMSTAT_meta_Method_Name': str,
        'GEMSTAT_meta_Method_Description': str,
        'GLORICH_meta_Value_remark_code': str,
        'GLORICH_meta_Meaning': str,          
        'WQP_meta_ResultAnalyticalMethod_MethodName': str,
        'WQP_meta_ResultLaboratoryCommentText': str, 
    },
    'TDP': {
        'obs_id': str,
        'obs_time_zone': str, 
        'site_id': str, 
        'site_name': str, 
        'site_country': str,
        'upstream_basin_area': np.float32, 
        'upstream_basin_area_unit': str,            
        'drainage_region_name': str,
        'param_code': str, 'source_param_code': str, 
        'param_name': str, 'source_param_name': str,
        'obs_value': np.float32, 
        'source_obs_value': np.float32,
        'detection_limit_flag': str,        
        'site_country': str,
        'source_unit': str,
        'filtration': str,
        'obs_percentile': np.float32,
        'site_ts_continuity': np.float32,           
        'GEMSTAT_meta_Station_Narrative': str, 
        'GEMSTAT_meta_Parameter_Description': str,
        'GEMSTAT_meta_Analysis_Method_Code': str,
        'GEMSTAT_meta_Method_Name': str,
        'GEMSTAT_meta_Method_Description': str,
        'GLORICH_meta_Value_remark_code': str,
        'GLORICH_meta_Meaning': str,          
        'WQP_meta_ResultAnalyticalMethod_MethodName': str,
        'WQP_meta_ResultLaboratoryCommentText': str, 
    },
    'TEMP': {
        'obs_id': str,
        'obs_time_zone': str, 'site_id': str, 'site_name': str, 'site_country': str,
        'upstream_basin_area': np.float32, 'upstream_basin_area_unit': str,        
        'drainage_region_name': str,
        'param_code': str, 'source_param_code': str, 'param_name': str, 'source_param_name': str,
        'obs_value': np.float32, 'source_obs_value': np.float32,
        'detection_limit_flag': str,        
        'site_country': str,
        'source_unit': str,
        'filtration': str,
        'obs_percentile': np.float32,
        'site_ts_continuity': np.float32,        
        'GEMSTAT_meta_Station_Narrative': str, 
        'GEMSTAT_meta_Parameter_Description': str,
        'GEMSTAT_meta_Analysis_Method_Code': str,
        'GEMSTAT_meta_Method_Name': str,
        'GEMSTAT_meta_Method_Description': str,
        'GLORICH_meta_Value_remark_code': str,
        'GLORICH_meta_Meaning': str,          
        'WATERBASE_meta_procedureAnalysedMatrix': str,
        'WATERBASE_meta_Remarks': str,               
        'WQP_meta_ResultAnalyticalMethod_MethodName': str,
        'WQP_meta_ResultLaboratoryCommentText': str,   
    },
    'TIC': {
        'upstream_basin_area': np.float32, 
        'upstream_basin_area_unit': str,            
        'drainage_region_name': str,
        'param_code': str, 'source_param_code': str, 
        'param_name': str, 'source_param_name': str,
        'obs_value': np.float32, 
        'source_obs_value': np.float32,
        'detection_limit_flag': str,        
        'site_country': str,
        'source_unit': str,
        'filtration': str,
        'obs_percentile': np.float32,
        'site_ts_continuity': np.float32,           
        'GEMSTAT_meta_Station_Narrative': str, 
        'GEMSTAT_meta_Parameter_Description': str,
        'GEMSTAT_meta_Analysis_Method_Code': str,
        'GEMSTAT_meta_Method_Name': str,
        'GEMSTAT_meta_Method_Description': str,
        'GLORICH_meta_Value_remark_code': str,
        'GLORICH_meta_Meaning': str,          
        'WQP_meta_ResultAnalyticalMethod_MethodName': str,
        'WQP_meta_ResultLaboratoryCommentText': str,         
    },
    'TIP': {
        'obs_id': str,
        'obs_time_zone': str, 'site_id': str, 'site_name': str, 'site_country': str,
        'upstream_basin_area': np.float32, 'upstream_basin_area_unit': str,        
        'drainage_region_name': str,
        'param_code': str, 'source_param_code': str, 'param_name': str, 'source_param_name': str,
        'obs_value': np.float32, 'source_obs_value': np.float32,
        'detection_limit_flag': str,        
        'site_country': str,
        'source_unit': str,
        'filtration': str,
        'obs_percentile': np.float32,
        'site_ts_continuity': np.float32,        
        'GEMSTAT_meta_Station_Narrative': str, 
        'GEMSTAT_meta_Parameter_Description': str,
        'GEMSTAT_meta_Analysis_Method_Code': str,
        'GEMSTAT_meta_Method_Name': str,
        'GEMSTAT_meta_Method_Description': str,
        'GLORICH_meta_Value_remark_code': str,
        'GLORICH_meta_Meaning': str,            
    },
    'TSS': {
        'obs_id': str,
        'obs_time_zone': str, 'site_id': str, 'site_name': str, 'site_country': str,
        'upstream_basin_area': np.float32, 'upstream_basin_area_unit': str,        
        'drainage_region_name': str,
        'param_code': str, 'source_param_code': str, 'param_name': str, 'source_param_name': str,
        'obs_value': np.float32, 'source_obs_value': np.float32,
        'detection_limit_flag': str,        
        'site_country': str,
        'source_unit': str,
        'filtration': str,
        'obs_percentile': np.float32,
        'site_ts_continuity': np.float32,        
        'GEMSTAT_meta_Station_Narrative': str, 
        'GEMSTAT_meta_Parameter_Description': str,
        'GEMSTAT_meta_Analysis_Method_Code': str,
        'GEMSTAT_meta_Method_Name': str,
        'GEMSTAT_meta_Method_Description': str,
        'GLORICH_meta_Value_remark_code': str,
        'GLORICH_meta_Meaning': str,          
        'WATERBASE_meta_procedureAnalysedMatrix': str,
        'WATERBASE_meta_Remarks': str,               
        'WQP_meta_ResultAnalyticalMethod_MethodName': str,
        'WQP_meta_ResultLaboratoryCommentText': str,  
    },
    'TP': {
        'obs_id': str,
        'obs_time_zone': str, 'site_id': str, 'site_name': str, 'site_country': str,
        'upstream_basin_area': np.float32, 'upstream_basin_area_unit': str,        
        'drainage_region_name': str,
        'param_code': str, 'source_param_code': str, 'param_name': str, 'source_param_name': str,
        'obs_value': np.float32, 'source_obs_value': np.float32,
        'detection_limit_flag': str,        
        'site_country': str,
        'source_unit': str,
        'filtration': str,
        'obs_percentile': np.float32,
        'site_ts_continuity': np.float32,        
        'GEMSTAT_meta_Station_Narrative': str, 
        'GEMSTAT_meta_Parameter_Description': str,
        'GEMSTAT_meta_Analysis_Method_Code': str,
        'GEMSTAT_meta_Method_Name': str,
        'GEMSTAT_meta_Method_Description': str,
        'GLORICH_meta_Value_remark_code': str,
        'GLORICH_meta_Meaning': str,          
        'WATERBASE_meta_procedureAnalysedMatrix': str,
        'WATERBASE_meta_Remarks': str,               
        'WQP_meta_ResultAnalyticalMethod_MethodName': str,
        'WQP_meta_ResultLaboratoryCommentText': str,  
    },
    'TON': {
        'obs_id': str,
        'obs_time_zone': str, 'site_id': str, 'site_name': str, 'site_country': str,
        'upstream_basin_area': np.float32, 'upstream_basin_area_unit': str,        
        'drainage_region_name': str,
        'param_code': str, 'source_param_code': str, 'param_name': str, 'source_param_name': str,
        'obs_value': np.float32, 'source_obs_value': np.float32,
        'detection_limit_flag': str,        
        'site_country': str,
        'source_unit': str,
        'filtration': str,
        'obs_percentile': np.float32,
        'site_ts_continuity': np.float32,        
        'GEMSTAT_meta_Station_Narrative': str, 
        'GEMSTAT_meta_Parameter_Description': str,
        'GEMSTAT_meta_Analysis_Method_Code': str,
        'GEMSTAT_meta_Method_Name': str,
        'GEMSTAT_meta_Method_Description': str,
        'GLORICH_meta_Value_remark_code': str,
        'GLORICH_meta_Meaning': str,          
        'WATERBASE_meta_procedureAnalysedMatrix': str,
        'WATERBASE_meta_Remarks': str,               
        'WQP_meta_ResultAnalyticalMethod_MethodName': str,
        'WQP_meta_ResultLaboratoryCommentText': str,  
    },
    'TOC': {
        'obs_id': str,
        'obs_time_zone': str, 'site_id': str, 'site_name': str, 'site_country': str,
        'upstream_basin_area': np.float32, 'upstream_basin_area_unit': str,        
        'drainage_region_name': str,
        'param_code': str, 'source_param_code': str, 'param_name': str, 'source_param_name': str,
        'obs_value': np.float32, 'source_obs_value': np.float32,
        'detection_limit_flag': str,        
        'site_country': str,
        'source_unit': str,
        'filtration': str,
        'obs_percentile': np.float32,
        'site_ts_continuity': np.float32,        
        'GEMSTAT_meta_Station_Narrative': str, 
        'GEMSTAT_meta_Parameter_Description': str,
        'GEMSTAT_meta_Analysis_Method_Code': str,
        'GEMSTAT_meta_Method_Name': str,
        'GEMSTAT_meta_Method_Description': str,
        'GLORICH_meta_Value_remark_code': str,
        'GLORICH_meta_Meaning': str,          
        'WATERBASE_meta_procedureAnalysedMatrix': str,
        'WATERBASE_meta_Remarks': str,               
        'WQP_meta_ResultAnalyticalMethod_MethodName': str,
        'WQP_meta_ResultLaboratoryCommentText': str, 
    },
    'TN': {
        'obs_id': str,
        'obs_time_zone': str, 'site_id': str, 'site_name': str, 'site_country': str,
        'upstream_basin_area': np.float32, 'upstream_basin_area_unit': str,        
        'drainage_region_name': str,
        'param_code': str, 'source_param_code': str, 'param_name': str, 'source_param_name': str,
        'obs_value': np.float32, 'source_obs_value': np.float32,
        'detection_limit_flag': str,        
        'site_country': str,
        'source_unit': str,
        'filtration': str,
        'obs_percentile': np.float32,
        'site_ts_continuity': np.float32,        
        'GEMSTAT_meta_Station_Narrative': str, 
        'GEMSTAT_meta_Parameter_Description': str,
        'GEMSTAT_meta_Analysis_Method_Code': str,
        'GEMSTAT_meta_Method_Name': str,
        'GEMSTAT_meta_Method_Description': str,
        'GLORICH_meta_Value_remark_code': str,
        'GLORICH_meta_Meaning': str,          
        'WATERBASE_meta_procedureAnalysedMatrix': str,
        'WATERBASE_meta_Remarks': str,               
        'WQP_meta_ResultAnalyticalMethod_MethodName': str,
        'WQP_meta_ResultLaboratoryCommentText': str, 
    },
    'TKN': {
        'obs_id': str,
        'obs_time_zone': str, 'site_id': str, 'site_name': str, 'site_country': str,
        'upstream_basin_area': np.float32, 'upstream_basin_area_unit': str,        
        'drainage_region_name': str,
        'param_code': str, 'source_param_code': str, 'param_name': str, 'source_param_name': str,
        'obs_value': np.float32, 'source_obs_value': np.float32,
        'detection_limit_flag': str,        
        'site_country': str,
        'source_unit': str,
        'filtration': str,
        'obs_percentile': np.float32,
        'site_ts_continuity': np.float32,        
        'GEMSTAT_meta_Station_Narrative': str, 
        'GEMSTAT_meta_Parameter_Description': str,
        'GEMSTAT_meta_Analysis_Method_Code': str,
        'GEMSTAT_meta_Method_Name': str,
        'GEMSTAT_meta_Method_Description': str,
        'GLORICH_meta_Value_remark_code': str,
        'GLORICH_meta_Meaning': str,          
        'WATERBASE_meta_procedureAnalysedMatrix': str,
        'WATERBASE_meta_Remarks': str,          
    },
}


class GRQA(Datasets):
    """
    Global River Water Quality Archive following the work of 
    `Virro et al., 2021 <https://essd.copernicus.org/articles/13/5483/2021/>`_ .
    This dataset comprises of 42 parameters for 94955 sites across 116 countries.

    Examples
    --------
    >>> from aqua_fetch import GRQA
    >>> ds = GRQA(path="/mnt/datawaha/hyex/atr/data")
    >>> ds.parameters
    ['TPP', 'PON', 'TEMP', 'TSS', ...]
    >>> print(len(ds.parameters))
    42
    >>> len(ds.countries)
    116
    >>> len(ds.stations())
    94955
    >>> len(ds.parameters)
    >>> coords = ds.stn_coords()
    >>> coords.shape
    (94955, 2)
    >>> country = "Pakistan"
    >>> len(ds.fetch_parameter('TEMP', country=country))
    1324
    >>> df = ds.fetch_parameter("TEMP", country=country)
    >>> print(df.shape)
    (1324, 38)
    >>> df = ds.fetch_parameter("NH4N", country=country)
    >>> print(df.shape)
    (28, 36)
    """

    url = 'https://zenodo.org/record/7056647#.YzBzDHZByUk'

    def __init__(
            self,
            download_source:bool = False,
            path = None,
            **kwargs):
        """
        parameters
        ----------
        download_source : bool
            whether to download source data or not
        """
        super().__init__(path=path, **kwargs)

        files = ['GRQA_data_v1.3.zip', 'GRQA_meta.zip']
        if download_source:
            files += ['GRQA_source_data.zip']
        self._download(include=files)

    @property
    def files(self):
        return os.listdir(os.path.join(self.path, "GRQA_data_v1.3", "GRQA_data_v1.3"))

    @property
    def parameters(self):
        return [f.split('_')[0] for f in self.files]
    
    def stations(self)->List[str]:
        """Returns names of stations/site_id"""
        return self.sites_data().index.tolist()
    
    @property
    def countries(self)->List[str]:
        return self.sites_data()['site_country'].dropna().unique().tolist()

    def fetch_parameter(
            self,
            parameter: str = "COD",
            site_name: Union[List[str], str] = None,
            country: Union[List[str], str] = None,
            st:Union[int, str, pd.DatetimeIndex] = None,
            en:Union[int, str, pd.DatetimeIndex] = None,
    )->pd.DataFrame:
        """
        parameters
        ----------
        parameter : str, optional
            name of parameter
        site_name : str/list, optional
            location for which data is to be fetched.
        country : str/list optional (default=None)
        st : str
            starting date date or index
        en : str
            end date or index

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame`

        Example
        --------
        >>> from aqua_fetch import GRQA
        >>> dataset = GRQA()
        >>> df = dataset.fetch_parameter()
        fetch data for only one country
        >>> cod_pak = dataset.fetch_parameter("COD", country="Pakistan")
        fetch data for only one site
        >>> cod_kotri = dataset.fetch_parameter("COD", site_name="Indus River - at Kotri")
        we can find out the number of data points and sites available for a specific country as below
        >>> for para in dataset.parameters:
        >>>     data = dataset.fetch_parameter(para, country="Germany")
        >>>     if len(data)>0:
        >>>         print(f"{para}, {df.shape}, {len(df['site_name'].unique())}")

        """

        assert isinstance(parameter, str)
        assert parameter in self.parameters

        if isinstance(site_name, str):
            site_name = [site_name]

        if isinstance(country, str):
            country = [country]

        df = self._load_df(parameter)

        if site_name is not None:
            assert isinstance(site_name, list)
            df = df[df['site_name'].isin(site_name)]
        if country is not None:
            assert isinstance(country, list)
            df = df[df['site_country'].isin(country)]

        df.index = pd.to_datetime(df.pop("obs_date") + " " + df.pop("obs_time"), errors='coerce')

        return check_st_en(df, st, en)

    def _load_df(self, parameter, **read_kws):
        if hasattr(self, f"_load_{parameter}"):
            return getattr(self, f"_load_{parameter}")()

        fname = os.path.join(self.path, "GRQA_data_v1.3", "GRQA_data_v1.3", f"{parameter}_GRQA.csv")
        if parameter in DTYPES:
            return pd.read_csv(fname, sep=";", dtype=DTYPES[parameter], **read_kws)
        return pd.read_csv(fname, sep=";", **read_kws)

    def _load_DO(self):
        # read_csv is causing mysterious errors

        f = os.path.join(self.path, "GRQA_data_v1.3",
                         "GRQA_data_v1.3", f"DO_GRQA.csv")
        lines = []
        with open(f, 'r', encoding='utf-8') as fp:
            for idx, line in enumerate(fp):
                lines.append(line.split(';'))

        return pd.DataFrame(lines[1:], columns=lines[0])
    
    def stn_coords(self):
        """
        Returns the coordinates of all the stations in the dataset

        Returns
        -------
        pd.DataFrame
            A dataframe with columns 'lat', 'long'
        """

        sites = self.sites_data()
        return sites[['lat', 'long']].dropna().astype(np.float32)

    def sites_data(self)->pd.DataFrame:
        """
        Returns the meta data for the dataset
        """

        fpath = os.path.join(self.path, 'sites.csv')

        if os.path.exists(fpath):
            if self.verbosity:
                print(f"loading from pre-existing{fpath}")
            return pd.read_csv(fpath, index_col=0)

        dfs = []

        cols = ['lat_wgs84', 'lon_wgs84', 'site_name', 'site_country',
                'upstream_basin_area', 'upstream_basin_area_unit']

        for idx, para in enumerate(self.parameters):

            df1 = self._load_df(para, 
                            usecols = ['site_id'] + cols
                            ).set_index('site_id')[cols]

            duplicates = df1.index.duplicated(keep='first')  # Keep the first occurrence, mark others as duplicate
            # Drop duplicates based on the index
            df2 = df1[~duplicates] 

            dfs.append(df2)
            if self.verbosity> 1: print(idx, para )

        df = pd.concat(dfs)
        print(df.shape)
        duplicates = df.index.duplicated(keep='first')  # Keep the first occurrence, mark others as duplicate
        # Drop duplicates based on the index
        df = df[~duplicates] 

        df.rename(columns={'lat_wgs84': 'lat', 'lon_wgs84': 'long',
                           'upstream_basin_area': 'basin_area_km2', 
                           'upstream_basin_area_unit': 'area_unit'}, inplace=True)
        
        df.replace('', np.nan, inplace=True)

        area_unit = df['area_unit']

        area_unit = area_unit.replace('', np.nan).dropna().unique()

        assert len(area_unit) == 1
        assert area_unit[0] == 'sq mi'

        # convert basin_area from sq mi to km2
        df['basin_area_km2'] = df['basin_area_km2'].astype(np.float32) * 2.58999

        df = df.drop(columns=['area_unit'])        

        df.to_csv(fpath, index=True)

        return df