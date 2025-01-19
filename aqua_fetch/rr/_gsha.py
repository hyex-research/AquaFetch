
__all__ = [
    'GSHA', 
    '_GSHA', 
    'Japan'
    'Arcticnet',
    'Spain',
    'Thailand'
    ]

import os
import time
from typing import List, Union, Dict
import concurrent.futures as cf

import numpy as np
import pandas as pd

from .._backend import xarray as xr, shapefile

from ..utils import get_cpus
from ..utils import check_attributes
from ..utils import merge_shapefiles
from .camels import Camels

from ._map import (
    total_precipitation_with_specifier,
    leaf_area_index,
    actual_evapotranspiration_with_specifier,
    total_potential_evapotranspiration_with_specifier,
    download_longwave_radiation_with_specifier,
    solar_radiation_with_specifier,
    snow_water_equivalent_with_specifier,
    mean_air_temp_with_specifier,
    mean_windspeed_with_specifier,
    u_component_of_wind_with_specifier,
    v_component_of_wind_with_specifier,
    groundwater_percentages,
    soil_moisture_layer1,
    soil_moisture_layer2,
    soil_moisture_layer3,
    soil_moisture_layer4,
    observed_streamflow_cms,
)

from ._map import (
    catchment_area,
    gauge_latitude,
    gauge_longitude,
    slope
    )


METEO_MAP = {
    'arcticnet': 'Meteorology_PartI_arcticnet_AFD_GRDC_IWRIS_MLIT/Meteorology_arcticnet_AFD_GRDC_IWRIS_MLIT',
    'AFD': 'Meteorology_PartI_arcticnet_AFD_GRDC_IWRIS_MLIT/Meteorology_arcticnet_AFD_GRDC_IWRIS_MLIT',
    'GRDC': 'Meteorology_PartI_arcticnet_AFD_GRDC_IWRIS_MLIT/Meteorology_arcticnet_AFD_GRDC_IWRIS_MLIT',
    'IWRIS': 'Meteorology_PartI_arcticnet_AFD_GRDC_IWRIS_MLIT/Meteorology_arcticnet_AFD_GRDC_IWRIS_MLIT',
    'MLIT': 'Meteorology_PartI_arcticnet_AFD_GRDC_IWRIS_MLIT/Meteorology_arcticnet_AFD_GRDC_IWRIS_MLIT',
    'HYDAT': 'Meteorology_ PartII_ANA_BOM_CCRR_HYDAT/Meteorology_ANA_BOM_CCRR_HYDAT',
    'ANA': 'Meteorology_ PartII_ANA_BOM_CCRR_HYDAT/Meteorology_ANA_BOM_CCRR_HYDAT',
    'BOM': 'Meteorology_ PartII_ANA_BOM_CCRR_HYDAT/Meteorology_ANA_BOM_CCRR_HYDAT',
    'CCRR': 'Meteorology_ PartII_ANA_BOM_CCRR_HYDAT/Meteorology_ANA_BOM_CCRR_HYDAT',
    'China': 'Meteorology_PartIII_China_CHP_RID_USGS/Meteorology_China_CHP_RID_USGS',
    'CHP': 'Meteorology_PartIII_China_CHP_RID_USGS/Meteorology_China_CHP_RID_USGS',
    'RID': 'Meteorology_PartIII_China_CHP_RID_USGS/Meteorology_China_CHP_RID_USGS',
    'USGS': 'Meteorology_PartIII_China_CHP_RID_USGS/Meteorology_China_CHP_RID_USGS',
}


class GSHA(Camels):
    """
    Global streamflow characteristics, hydrometeorology and catchment
    attributes following `Peirong et al., 2023 <https://doi.org/10.5194/essd-16-1559-2024>`_.
    The data is downloaded from its `zenodo repository <https://zenodo.org/record/8090704>`_.
    It should be noted that this dataset does not contain observed streamflow data.
    It has 21568 stations, 26 dynamic (meteorological + storage) features with daily timestep, 21 dynamic
    features (landcover + streamflow indices + reservoir) with yearly timestep and 35 static features.

    Examples
    --------
    >>> from water_datasets import GSHA
    >>> dataset = GSHA()
    >>> len(dataset.stations())
    21568
    >>> dataset.agencies
    ['arcticnet', 'AFD', 'GRDC', 'IWRIS', 'MLIT', 'HYDAT', 'ANA', 'BOM', 'CCRR', 'China', 'CHP', 'RID', 'USGS']
    >>> dataset.start
    Timestamp('1979-01-01 00:00:00')
    >>> dataset.end
    Timestamp('2022-12-31 00:00:00')
    >>> dataset.static_features
    ['ele_mt_uav', 'slp_dg_uav', 'lat', 'long', 'area', 'agency', ...]
    >>> len(dataset.dynamic_features)
    26
    >>> len(dataset.daily_dynamic_features)
    26
    >>> len(dataset.yearly_dynamic_features)
    21
    >>> dataset.fetch_static_features('1001_arcticnet')
    fetch static features for all stations of arcticnet agency
    >>> dataset.fetch_static_features(agency='arcticnet')
    fetch static features for all stations of arcticnet agency
    >>> ds.fetch_dynamic_features(agency='arcticnet')

    """
    url = "https://zenodo.org/record/8090704"

    def __init__(self,
                 path=None,
                 overwrite=False,
                 to_netcdf: bool = True,
                 **kwargs):
        """
        Parameters
        ----------
        to_netcdf : bool
            whether to convert all the data into one netcdf file or not.
            This will fasten repeated calls to fetch etc but will
            require netcdf5 package as well as xarry.
        """
        super(GSHA, self).__init__(path=path, to_netcdf=to_netcdf, **kwargs)
        self.path = path

        files = ['Global_files.zip',
                 'GSHAreadme.docx',
                 'LAI.zip',
                 'Landcover.zip',
                 'Meteorology_PartI_arcticnet_AFD_GRDC_IWRIS_MLIT.zip',
                 'Meteorology_ PartII_ANA_BOM_CCRR_HYDAT.zip',
                 'Meteorology_PartIII_China_CHP_RID_USGS.zip',
                 'Reservoir.zip',
                 'Storage.zip',
                 'StreamflowIndices.zip',
                 'WatershedPolygons.zip',
                 'WatershedsAll.csv'
                 ]
        # self._download(overwrite=overwrite, files_to_check=files)

        self._maybe_merge_shapefiles()

        fpath = os.path.join(self.path,
                             "Global_files",
                             "Global_files",
                             'WatershedsAll.csv')
        wsAll = pd.read_csv(fpath)
        wsAll.columns = ['station_id', 'lat', 'long', 'area', 'agency']
        wsAll.index = wsAll.pop('station_id')
        self.wsAll = wsAll[~wsAll.index.duplicated(keep='first')].copy()

        self.boundary_file = os.path.join(self.path, "boundaries.shp")
        self._create_boundary_id_map(self.boundary_file, 7)

        self._daily_dynamic_features = self.__daily_dynamic_features()
        self._yearly_dynamic_features = self.__yearly_dynamic_features()

        self._static_features = self.__static_features()

    @property
    def agencies(self) -> List[str]:
        """
        returns the names of agencies as list

            - ``arcticnet`` : Antarctica
            - ``AFD`` : Spain
            - ``GRDC`` : Global
            - ``IWRIS`` : India
            - ``MLIT`` : Japan
            - ``HYDAT`` : Canada
            - ``ANA``: Brazil
            - ``BOM`` : Australia
            - ``CCRR`` : Chile
            - ``China``
            - ``CHP`` : China
            - ``RID`` : Thailand
            - ``USGS``

        """
        return self.wsAll.loc[:, 'agency'].unique()

    @property
    def daily_dynamic_features(self) -> List[str]:
        return self._daily_dynamic_features

    @property
    def yearly_dynamic_features(self) -> List[str]:
        return self._yearly_dynamic_features

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'area': catchment_area(),
                'lat': gauge_latitude(),
                'slp_dg_uav': slope('degrees'),
                'long': gauge_longitude(),
        }

    @property
    def dyn_map(self):
        return {
            'EP_GLEAM': actual_evapotranspiration_with_specifier('gleam'),
            'EP_REA': actual_evapotranspiration_with_specifier('rea'),
            'GLEAM_PET': total_potential_evapotranspiration_with_specifier('gleam'),
            'GW': groundwater_percentages(),
            'HPET_PET': total_potential_evapotranspiration_with_specifier('hpet'),
            'LONGRAD_ERA': download_longwave_radiation_with_specifier('era5'),
            'LONGRAD_MERRA': download_longwave_radiation_with_specifier('merra2'),
            'P_EMEarth': total_precipitation_with_specifier('emearth'),
            'P_MSWEP': total_precipitation_with_specifier('mswep'),
            'SHORTRAD_ERA': solar_radiation_with_specifier('era5'),
            'SHORTRAD_MERRA': solar_radiation_with_specifier('merra2'),
            'SML1': soil_moisture_layer1(),
            'SML2': soil_moisture_layer2(),
            'SML3': soil_moisture_layer3(),
            'SML4': soil_moisture_layer4(),
            'SWDE': snow_water_equivalent_with_specifier('era5'),  # change m to mm
            'T_ERA': mean_air_temp_with_specifier('era5'),  # change from K to C
            'T_EUSTACE': mean_air_temp_with_specifier('eustace'),  # change from K to C
            'T_MERRA': mean_air_temp_with_specifier('merra2'),  # change from K to C
            'WINDERA': mean_windspeed_with_specifier('era5'),
            'WINDMERRA': mean_windspeed_with_specifier('merra'),
            'WINDU_ERA': u_component_of_wind_with_specifier('era5'),
            'WINDU_MERRA': u_component_of_wind_with_specifier('merra'),
            'WINDV_ERA': v_component_of_wind_with_specifier('era5'),
            'WINDV_MERRA': v_component_of_wind_with_specifier('merra'),
            'lai': leaf_area_index(),
        }

    @property
    def dyn_factors(self):
        return {
            snow_water_equivalent_with_specifier('era5'): lambda x: x * 1000,
            mean_air_temp_with_specifier('era5'): lambda x: x - 273.15,
            mean_air_temp_with_specifier('eustace'): lambda x: x - 273.15,
            mean_air_temp_with_specifier('merra2'): lambda x: x - 273.15,
        }

    def __daily_dynamic_features(self):
        return pd.concat(
            [self.meteo_vars_stn('1001_arcticnet'),
             self.storage_vars_stn('1001_arcticnet'),
             ],
            axis=1
        ).columns.tolist() + ['lai']

    def __yearly_dynamic_features(self):
        return pd.concat(
            [self.lc_variables_stn('1001_arcticnet'),
             self.streamflow_indices_stn('1001_arcticnet'),
             self.reservoir_variables_stn('1001_arcticnet')
             ],
            axis=1
        ).columns.tolist()

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp('1979-01-01')

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp('2022-12-31')

    @property
    def static_features(self) -> List[str]:
        return self._static_features

    @property
    def dynamic_features(self) -> List[str]:
        return self.daily_dynamic_features

    def __static_features(self):
        return pd.concat(
            [self.atlas('1001_arcticnet'),
             self.uncertainty('1001_arcticnet')
             ],
            axis=1
        ).columns.tolist() + self.wsAll.columns.tolist()

    def agency_of_stn(self, stn: str) -> str:
        """find the agency to which a station belongs """
        return self.wsAll.loc[stn, 'agency']

    def agency_stations(self, agency: str) -> List[str]:
        """returns the station ids from a particular agency"""
        return self.wsAll[self.wsAll['agency'] == agency].index.tolist()

    def _maybe_merge_shapefiles(self):

        shp_path = os.path.join(self.path, 'WatershedPolygons', 'WatershedPolygons')
        out_shapefile = os.path.join(self.path, 'boundaries.shp')

        if not os.path.exists(out_shapefile):
            shp_files = [os.path.join(shp_path, filename) for filename in os.listdir(shp_path) if
                         filename.endswith('.shp')]
            merge_shapefiles(shp_files, out_shapefile,
                             add_new_field=True)
        return

    def _get_stations(
            self,
            stations: Union[str, List[str]] = "all",
            agency: Union[str, List[str]] = "all"
    ) -> List[str]:
        if agency != "all" and stations != 'all':
            raise ValueError("Either provide agency or stations not both")

        if agency != "all":
            agency = check_attributes(agency, self.agencies, 'agency')
            stations = self.wsAll[self.wsAll['agency'].isin(agency)].index.tolist()
        else:
            stations = check_attributes(stations, self.stations(), 'stations')

        return stations

    def stn_coords(self, stations: List[str] = "all", agency: List[str] = "all") -> pd.DataFrame:
        """
        returns the latitude and longitude of stations

        Returns
        -------
        pd.DataFrame
            a pandas DataFrame of shape (n, 2) where n is the number of stations

        Examples
        --------
        >>> from water_datasets import GSHA
        >>> dataset = GSHA()
        >>> dataset.stn_coords('1001_arcticnet')
        >>> dataset.stn_coords(['1001_arcticnet', '1002_arcticnet'])
        get coordinates for all stations of arcticnet agency
        >>> dataset.stn_coords(agency='arcticnet')
        """
        stations = self._get_stations(stations, agency)
        return self.wsAll.loc[stations, ['lat', 'long']].copy()

    def stations(self, agency: str = "all") -> List[str]:
        """returns names of stations as list"""
        if agency != "all":
            agency = check_attributes(agency, self.agencies, 'agency')
            return self.wsAll[self.wsAll['agency'].isin(agency)].index.tolist()
        return self.wsAll.index.tolist()

    def area(self, stations: List[str] = "all", agency: List[str] = "all") -> pd.Series:
        """area of catchments"""
        stations = self._get_stations(stations, agency)
        return self.wsAll.loc[stations, 'area']

    def uncertainty(
            self,
            stations: List[str] = "all",
            agency: List[str] = "all"
    ) -> pd.DataFrame:
        """
        Uncertainty estimates of all meteorological variables over all watersheds

            - P_uncertainty (%)	Precipitation uncertainty estimates (in percentage). Uncertainties are calculated from EM-Earth deterministic and MSWEP datasets.
            - T_uncertainty (%)	Temperature uncertainty estimates (in percentage). Uncertainties are calculated from EUSTACE, MERRA-2, and ERA5 datasets.
            - EVP_uncertainty (%)	Actual evapotranspiration uncertainty estimates (in percentage). Uncertainties are calculated from GLEAM and REA datasets.
            - LRAD_uncertainty (%)	Downward longwave radiation uncertainty estimates (in percentage). Uncertainties are calculated from MERRA-2 and ERA5-land datasets.
            - SRAD_uncertainty (%)	Downward shortwave radiation uncertainty estimates (in percentage). Uncertainties are calculated from MERRA-2 and ERA5-land datasets.
            - wind_uncertainty (%)	Wind speed uncertainty estimates (in percentage). The u- and v- components are aggregated on each grid to obtain wind speed. Uncertainties are calculated from MERRA-2 and ERA5-land datasets.
            - pet_uncertainty (%)	Potential evapotranspiration uncertainty estimates (in percentage). Uncertainties are calculated from GLEAM and REA datasets.

        Returns
        -------
        pd.DataFrame
            a pandas DataFrame of shape (n, 7) where n is the number of stations
        """
        stations = self._get_stations(stations, agency)

        fpath = os.path.join(
            self.path,
            "Global_files",
            "Global_files",
            'Uncertainty.csv')
        df = pd.read_csv(fpath, index_col=0)
        df = df[~df.index.duplicated(keep='first')]
        return df.loc[stations, :]

    def atlas(self, stations: List[str] = "all", agency: List[str] = "all") -> pd.DataFrame:
        """
        The link table between GSHA watershed IDs and RiverATLAS
        river reach IDs, as well as the selected static attributes

        Returns
        -------
        pd.DataFrame
            a pandas DataFrame of shape (n, 24) where n is the number of stations
        """
        stations = self._get_stations(stations, agency)

        fpath = os.path.join(
            self.path,
            "Global_files",
            "Global_files",
            'GSHA_ATLAS.csv')
        df = pd.read_csv(fpath, index_col=0)

        df = df[~df.index.duplicated(keep='first')]
        return df.loc[stations, :]

    def lc_variables_stn(self, stn: str) -> pd.DataFrame:
        """
        Landcover variables for a given station which have yearly timestep.
        Following three landcover variables are provided:

            - urban_fraction(%):	Ratio of urban extent to the entire watershed area (percentage).
            - forest_fraction(%):	Ratio of forest extent to the entire watershed area (percentage).
            - cropland_fraction(%):	Ratio of cropland extent to the entire watershed area (percentage).

        Returns
        -------
        pd.DataFrame
            a pandas DataFrame of shape (n, 3) where n is the number of years
        """
        return lc_variable_stn(self.path, stn)

    def lc_variables(
            self,
            stations: List[str] = "all",
            agency: List[str] = "all"
    ):
        """
        Landcover variables for one or more than one station either
        as xr.Dataset or dictionary. The data has yearly timestep.
        """
        stations = self._get_stations(stations, agency)

        lc_vars = lc_vars_all_stns(self.path)
        if isinstance(lc_vars, xr.Dataset):
            return lc_vars[stations]
        else:
            return {stn: lc_vars[stn] for stn in stations}

    def reservoir_variables_stn(self, stn: str) -> pd.DataFrame:
        """
        Reservoir variables for a given station from 1979 to 2020 with yearly timestep.
        Following two reservoir variables are provided:

        - ``capacity``:	Reservoir capacity of the year in the watershed (m3). To avoid including too many missing values, we use the ICOLD capacity in the linked table of the GeoDAR dataset.
        - ``dor``:	Degree of regulation of the watershed (yearly reservoir capacity/yearly mean flow). If yearly mean flow is missing, the value is substituted with the average of all mean flow values.

        Returns
        -------
        pd.DataFrame
            a pandas DataFrame of shape (42, 2) where 42 is the number of years
        """
        return reservoir_vars_stn(self.path, stn)

    def reservoir_variables(
            self,
            stations: List[str] = "all",
            agency: List[str] = "all"
    ):
        """
        Reservoir variables for one or more than one station either
        as xr.Dataset or dictionary. The data has yearly timestep.
        """
        stations = self._get_stations(stations, agency)

        lc_vars = reservoir_vars_all_stns(self.path)
        if isinstance(lc_vars, xr.Dataset):
            return lc_vars[stations]
        else:
            return {stn: lc_vars[stn] for stn in stations}

    def streamflow_indices_stn(self, stn: str) -> pd.DataFrame:
        """
        Streamflow indices for a given station which have yearly timestep.

        Returns
        -------
        pd.DataFrame
            a pandas DataFrame of shape (n, 16) where n is the number of years
        """
        return streamflow_indices_stn(self.path, stn)

    def streamflow_indices(
            self,
            stations: List[str] = "all",
            agency: List[str] = "all"
    ):
        """
        Landcover variables for one or more than one station either
        as xr.Dataset or dictionary. The data has yearly timestep.
        """
        stations = self._get_stations(stations, agency)

        lc_vars = streamflow_indices_all_stations(
            self.path,
            to_netcdf=self.to_netcdf,
            verbosity=self.verbosity
        )
        if isinstance(lc_vars, xr.Dataset):
            return lc_vars[stations]
        else:
            return {stn: lc_vars[stn] for stn in stations}

    def lai_stn(self, stn: str) -> pd.Series:
        """
        Daily leaf area index. As per documentation, due to satellite data quality,
        some watersheds might have relatively serious data missing issue. The data is
        from 1981-01-01 to 2020-12-31.

        Returns
        -------
        pd.Series
            a pandas Series of shape (14571,) where 14571 is the number of days
        """
        return lai_stn(self.path, stn)

    def lai(
            self,
            stations: List[str] = "all",
            agency: List[str] = "all"
    ):
        """
        Leaf Area Index timeseries for one or more than one station either
        as xr.Dataset or pandas DataFrame. The data has daily timestep.
        """
        stations = self._get_stations(stations, agency)

        lai = lai_all_stns(
            self.path,
            to_netcdf=self.to_netcdf,
            verbosity=self.verbosity
        )
        return lai[stations]

    def meteo_vars_stn(self, stn: str) -> pd.DataFrame:
        """
        Daily meteorological variables from 1979-01-01 to 2022-12-31 for a given station.

        Returns
        -------
        pd.DataFrame
            a pandas DataFrame of shape (16071, 19) where n is the number of days
        """
        path = os.path.join(
            self.path,
            METEO_MAP[self.agency_of_stn(stn)],
            f'{stn}.csv'
        )
        return self._meteo_vars_stn(path)

    def meteo_vars_all_stns(self):
        """
        Meteorological variables from 1979-01-01 to 2022-12-31 for all stations either
        as xr.Dataset or dictionary. The data has daily timestep.
        """
        nc_path = os.path.join(self.path, 'meteo_vars.nc')

        if self.to_netcdf and os.path.exists(nc_path):
            if self.verbosity: print(f"Reading from pre-existing {nc_path}")
            return xr.open_dataset(nc_path)

        meteo_vars = {}
        paths = [os.path.join(
            self.path,
            METEO_MAP[self.agency_of_stn(stn)],
            f'{stn}.csv') for stn in self.stations()]

        cpus = self.processes or get_cpus() - 2
        start = time.time()

        if self.verbosity:
            print(f"Reading meteorological variables for {len(self.stations())} stations using {cpus} cpus")
        # takes ~ 1538 seconds with 110 cpus
        with cf.ProcessPoolExecutor(cpus) as executor:
            results = executor.map(
                self._meteo_vars_stn,
                paths
            )

        if self.verbosity: print(f"Time taken: {time.time() - start:.2f} seconds")

        if self.to_netcdf:
            for stn, df in zip(self.stations(), results):
                meteo_vars[stn] = df

            encoding = {stn: {'dtype': 'float32', 'zlib': True, 'complevel': 3} for stn in meteo_vars.keys()}

            ds = xr.Dataset(meteo_vars)

            if self.verbosity: print(f"Saving to {nc_path}")
            ds.to_netcdf(nc_path, encoding=encoding)
        else:

            ds = {stn: df for stn, df in zip(self.stations(), results)}

        return ds

    def meteo_vars(
            self,
            stations: List[str] = "all",
            agency: List[str] = "all"
    ):
        """
        Meteorological variables from 1979-01-01 to 2022-12-31 for one or more than one station either
        as xr.Dataset or dictionary. The data has daily timestep.
        """
        if agency != "all" and stations != 'all':
            raise ValueError("Either provide agency or stations not both")

        if agency != "all":
            agency = check_attributes(agency, self.agencies, 'agency')
            stations = self.wsAll[self.wsAll['agency'].isin(agency)].index.tolist()
        else:
            stations = check_attributes(stations, self.stations(), 'stations')

        meteo_vars = self.meteo_vars_all_stns()

        if isinstance(meteo_vars, xr.Dataset):
            return meteo_vars[stations]
        else:
            return {stn: meteo_vars[stn] for stn in stations}

    def storage_vars_stn(self, stn: str) -> pd.DataFrame:
        """
        Daily Water storage term variables from 1979-01-01 to 2021-12-31 for a given station.

            - SM_layer1:  0-7 cm soil moisture from ERA5 land soil water layer 1 (m3/m3) for 1979-2021.
            - SM_layer2:  7-28 cm soil moisture from ERA5 land soil water layer 2 (m3/m3) for 1979-2021.
            - SM_layer3:  28-100 cm soil moisture from ERA5 land soil water layer 3 (m3/m3) for 1979-2021.
            - SM_layer4:  100-289 cm soil moisture from ERA5 land soil water layer 4 (m3/m3) for 1979-2021.
            - SWDE:  Snow water equivalent from ERA5 snow depth water equivalent (m of water equivalent) for 1979-2021.
            - groundwater(%):  Groundwater percentage from GRACE-FO data assimilation (%) for 2003-2021 (weekly).

        Returns
        -------
        pd.DataFrame
            a pandas DataFrame of shape (15706, 6) where n is the number of days
        """
        path = os.path.join(
            self.path,
            "Storage",
            "Storage",
            f'{stn}.csv'
        )
        return self._storage_vars_stn(path)

    def storage_vars_all_stns(self):
        """
        Water storage term variables from 1979-01-01 to 2021-12-31 for all stations either
        as xr.Dataset or dictionary. The data has daily timestep.
        """
        nc_path = os.path.join(self.path, 'storage.nc')

        if self.to_netcdf and os.path.exists(nc_path):
            if self.verbosity: print(f"Reading from pre-existing {nc_path}")
            return xr.open_dataset(nc_path)

        storage_vars = {}
        paths = [os.path.join(
            self.path,
            "Storage",
            "Storage",
            f'{stn}.csv') for stn in self.stations()]

        cpus = self.processes or get_cpus() - 2
        start = time.time()

        if self.verbosity: print(f"Reading storage vars for {len(self.stations())} stations using {cpus} cpus")
        # takes ~ 975 seconds with 110 cpus
        with cf.ProcessPoolExecutor(cpus) as executor:
            results = executor.map(
                self._storage_vars_stn,
                paths
            )

        if self.verbosity: print(f"Time taken: {time.time() - start:.2f} seconds")

        if self.to_netcdf:

            encoding = {stn: {'dtype': 'float32', 'zlib': True, 'complevel': 3} for stn in self.stations()}

            for stn, df in zip(self.stations(), results):
                storage_vars[stn] = df

            ds = xr.Dataset(storage_vars)

            if self.verbosity: print(f"Saving to {nc_path}")
            ds.to_netcdf(nc_path, encoding=encoding)
        else:

            ds = {stn: df for stn, df in zip(self.stations(), results)}

        return ds

    def storage_vars(
            self,
            stations: List[str] = "all",
            agency: List[str] = "all"
    ):
        """
        Water storage term variables from 1979-01-01 to 2021-12-31 for one or more than one station either
        as xr.Dataset or dictionary. The data has daily timestep.
        """
        stations = self._get_stations(stations, agency)

        meteo_vars = self.storage_vars_all_stns()

        if isinstance(meteo_vars, xr.Dataset):
            return meteo_vars[stations]
        else:
            return {stn: meteo_vars[stn] for stn in stations}

    def fetch_static_features(
            self,
            stations: Union[str, List[str]] = "all",
            static_features: Union[str, List[str]] = "all",
            agency: List[str] = "all",
    ) -> pd.DataFrame:
        """
        Returns static features of one or more stations.

        Parameters
        ----------
            stations : str
                name/id of station/stations of which to extract the data
            static_features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.

        Returns
        -------
        pd.DataFrame
            a pandas dataframe of shape (stations, features)

        Examples
        ---------
        >>> from water_datasets import GSHA
        >>> dataset = GSHA()
        get the names of stations
        >>> stns = dataset.stations()
        >>> len(stns)
            21568
        get all static data of all stations
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (21568, 35)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('1001_arcticnet')
        >>> static_data.shape
           (1, 35)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['ele_mt_uav', 'slp_dg_uav'])
        >>> static_data.shape
           (21568, 2)
        >>> data = dataset.fetch_static_features('1001_arcticnet', static_features=['slp_dg_uav', 'slp_dg_uav'])
        >>> data.shape
           (1, 2)
        >>> out = ds.fetch_static_features(agency='arcticnet')
        >>> out.shape
        (106, 35
        """

        stations = self._get_stations(stations, agency)

        features = check_attributes(static_features, self.static_features, 'static_features')

        return pd.concat([
            self.atlas(stations),
            self.uncertainty(stations),
            self.wsAll.loc[stations, :]
        ],
            axis=1
        ).loc[:, features]

    def fetch_stn_dynamic_features(
            self,
            stn_id: str,
            dynamic_features='all',
    ) -> pd.DataFrame:
        """
        Fetches all or selected dynamic features of one station.

        Parameters
        ----------
            stn_id : str
                name/id of station of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                dynamic features are returned.

        Returns
        -------
        pd.DataFrame
            a pandas dataframe of shape (n, features) where n is the number of days

        Examples
        --------
        >>> from water_datasets import GSHA
        >>> camels = GSHA()
        >>> camels.fetch_stn_dynamic_features('1001_arcticnet').unstack()
        >>> camels.dynamic_features
        >>> camels.fetch_stn_dynamic_features('1001_arcticnet',
        ... features=['tmax_AWAP', 'vprp_AWAP']).unstack()
        """
        features = check_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')

        out = pd.concat(
            [self.meteo_vars_stn(stn_id),
             self.storage_vars_stn(stn_id),
             self.lai_stn(stn_id).rename('lai')
             ],
            axis=1
        ).loc[:, features]
        out.columns.name = 'dynamic_features'
        return out

    def fetch_dynamic_features(
            self,
            stations: Union[List[str], str] = "all",
            dynamic_features='all',
            st=None,
            en=None,
            as_dataframe=False,
            agency: List[str] = "all",
    ):
        """Fetches all or selected dynamic features of one station.

        Parameters
        ----------
            stations : str
                name/id of station of which to extract the data
            features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                dynamic features are returned.
            st : Optional (default=None)
                start time from where to fetch the data.
            en : Optional (default=None)
                end time untill where to fetch the data
            as_dataframe : bool, optional (default=False)
                if true, the returned data is pandas DataFrame otherwise it
                is xarray dataset

        Examples
        --------
            >>> from water_datasets import GSHA
            >>> camels = GSHA()
            >>> camels.fetch_dynamic_features('1001_arcticnet', as_dataframe=True).unstack()
            >>> camels.dynamic_features
            >>> camels.fetch_dynamic_features('1001_arcticnet',
            ... features=['tmax_AWAP', 'vprp_AWAP', 'streamflow_mmd'],
            ... as_dataframe=True).unstack()
        """

        stations = self._get_stations(stations, agency)

        features = check_attributes(dynamic_features, self.dynamic_features, 'dynamic_features')

        st, en = self._check_length(st, en)

        if len(stations) == 1:
            if as_dataframe:
                return self.fetch_stn_dynamic_features(stations[0], features)
            else:
                return xr.Dataset({stations[0]: xr.DataArray(self.fetch_stn_dynamic_features(stations[0], features))})

        if as_dataframe:
            raise NotImplementedError("as_dataframe=True is not implemented yet")

        meteo_vars = self.meteo_vars(stations)
        storage_vars = self.storage_vars(stations)
        # since lai does not have 'features' dimension, we need to add it
        lai = self.lai(stations).expand_dims({'features': ['lai']})

        ds = xr.concat([meteo_vars, storage_vars, lai], dim='features')
        ds = ds.rename({'features': 'dynamic_features'})
        return ds.sel(time=slice(st, en))

    def _meteo_vars_stn(self, fpath) -> pd.DataFrame:

        if not os.path.exists(fpath):
            raise FileNotFoundError(f"{fpath} not found")

        df = pd.read_csv(fpath, index_col=0,
                         )

        df.index = pd.to_datetime(df.index)

        df.index.name = 'time'
        df.columns.name = 'features'

        df.rename(columns=self.dyn_map, inplace=True)

        for col, func in self.dyn_factors.items():
            if col in df.columns:
                df[col] = df[col].apply(func)

        return df

    def _storage_vars_stn(self, fpath) -> pd.DataFrame:

        if not os.path.exists(fpath):
            raise FileNotFoundError(f"{fpath} not found")

        df = pd.read_csv(fpath, index_col=0,
                         dtype={'SWDE': np.float32,
                                'SML1': np.float32,
                                'SML2': np.float32,
                                'SML3': np.float32,
                                'SML4': np.float32,
                                'GW': np.float32, }
                         )

        df.index = pd.to_datetime(df.index)

        df.index.name = 'time'
        df.columns.name = 'features'

        df.rename(columns=self.dyn_map, inplace=True)

        for col, func in self.dyn_factors.items():
            if col in df.columns:
                df[col] = df[col].apply(func)

        return df


class _GSHA(Camels):
    """
    Parent class for those datasets which uses static and dynamic features from
    GSHA dataset . The following dataset classes are based on this class:

        - py:class:`water_datasets.Japan`
        - py:class:`water_datasets.Thailand`
        - py:class:`water_datasets.Spain`

    """

    def __init__(
            self,
            gsha_path: Union[str, os.PathLike] = None,
            verbosity: int = 1,
            **kwargs):
        super(_GSHA, self).__init__(verbosity=verbosity, **kwargs)

        if gsha_path is None:
            self.gsha_path = os.path.dirname(self.path)
        else:
            self.gsha_path = gsha_path

        self.gsha = GSHA(path=self.gsha_path, verbosity=verbosity)

        self.boundary_file = self.gsha.boundary_file
        self._stations = self.__stations()

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp('1979-01-01')

    @property
    def end(self) -> pd.Timestamp:
        return pd.Timestamp('2022-12-31')

    @property
    def dynamic_features(self) -> List[str]:
        return [observed_streamflow_cms()] + self.gsha.dynamic_features

    @property
    def static_features(self) -> List[str]:
        return self.gsha.static_features

    @property
    def _coords_name(self) -> List[str]:
        return ['lat', 'long']

    @property
    def _area_name(self) -> str:
        return 'area'

    @property
    def _q_name(self) -> str:
        return observed_streamflow_cms()

    def stations(self) -> List[str]:
        return self._stations

    def __stations(self) -> List[str]:
        """
        returns names of only those stations which are also documented
        by GSHA.
        """
        return [stn.split('_')[0] for stn in self.gsha.agency_stations(self.agency_name)]

    def get_boundary(
            self,
            catchment_id: str,
            as_type: str = 'numpy'
    ):
        """
        returns boundary of a catchment in a required format

        Parameters
        ----------
        catchment_id : str
            name/id of catchment
        as_type : str
            'numpy' or 'geopandas'

        Examples
        --------
        >>> from water_datasets import Japan
        >>> dataset = Japan()
        >>> dataset.get_boundary(dataset.stations()[0])
        """

        if shapefile is None:
            raise ModuleNotFoundError("shapefile module is not installed. Please install it to use boundary file")

        from shapefile import Reader

        bndry_sf = Reader(self.boundary_file)
        if self.agency_name == 'RID':
            catchment_id = catchment_id.replace('.', '_')
        bndry_shp = bndry_sf.shape(self.gsha.bndry_id_map[f"{catchment_id}_{self.agency_name}"])

        bndry_sf.close()

        xyz = np.array(bndry_shp.points)

        xyz = self.transform_coords(xyz)

        return xyz

    def _fetch_dynamic_features(
            self,
            stations: list,
            dynamic_features='all',
            st=None,
            en=None,
            as_dataframe=False,
            as_ts=False
    ):
        """Fetches dynamic features of station."""
        st, en = self._check_length(st, en)
        features = check_attributes(dynamic_features, self.dynamic_features.copy(), 'dynamic_features')

        daily_q = None

        if observed_streamflow_cms() in features:
            daily_q = self.get_q(as_dataframe)
            if isinstance(daily_q, xr.Dataset):
                daily_q = daily_q.sel(time=slice(st, en))[stations]
            else:
                daily_q = daily_q.loc[st:en, stations]

            features.remove(observed_streamflow_cms())

        if len(features) == 0:
            return daily_q

        stations_ = [f"{stn}_{self.agency_name}" for stn in stations]
        data = self.gsha.fetch_dynamic_features(stations_, features, st, en, as_dataframe)

        if daily_q is not None:
            if isinstance(daily_q, xr.Dataset):
                assert isinstance(data, xr.Dataset), "xarray dataset not supported"
                data = data.rename({stn: stn.split('_')[0] for stn in data.data_vars})

                # first create a new dimension in daily_q named dynamic_features
                daily_q = daily_q.expand_dims({'dynamic_features': [observed_streamflow_cms()]})
                data = xr.concat([data, daily_q], dim='dynamic_features').sel(time=slice(st, en))
            else:
                # -1 because the data in .nc files hysets starts with 0
                data.rename(columns={stn: stn.split('_')[0] for stn in stations}, inplace=True)
                assert isinstance(data.index, pd.MultiIndex)
                # data is multiindex dataframe but daily_q is not
                # first make daily_q multiindex
                daily_q['dynamic_features'] = 'daily_q'
                daily_q.set_index('dynamic_features', append=True, inplace=True)
                daily_q = daily_q.reorder_levels(['time', 'dynamic_features'])
                data = pd.concat([data, daily_q], axis=0).sort_index()

        return data

    def _fetch_static_features(
            self,
            station="all",
            static_features: Union[str, list] = 'all',
            st=None,
            en=None,
            as_ts=False
    ) -> pd.DataFrame:
        """Fetches static features of station."""
        if self.verbosity > 1:
            print('fetching static features')
        stations = check_attributes(station, self.stations(), 'stations')
        stations_ = [f"{stn}_{self.agency_name}" for stn in stations]
        static_feats = self.gsha.fetch_static_features(stations_, static_features).copy()
        static_feats.index = [stn.split('_')[0] for stn in static_feats.index]
        return static_feats

    def fetch_stations_features(
            self,
            stations: list,
            dynamic_features: Union[str, list, None] = 'all',
            static_features: Union[str, list, None] = None,
            st=None,
            en=None,
            as_dataframe: bool = False,
            **kwargs
    ):
        """
        returns features of multiple stations

        Examples
        --------
        >>> from water_datasets import Arcticnet
        >>> dataset = Arcticnet()
        >>> stations = dataset.stations()
        >>> features = dataset.fetch_stations_features(stations)
        """
        stations = check_attributes(stations, self.stations(), 'stations')

        if dynamic_features is not None:

            dyn = self._fetch_dynamic_features(stations=stations,
                                               dynamic_features=dynamic_features,
                                               as_dataframe=as_dataframe,
                                               st=st,
                                               en=en,
                                               **kwargs
                                               )

            if static_features is not None:  # we want both static and dynamic
                to_return = {}
                static = self._fetch_static_features(station=stations,
                                                     static_features=static_features,
                                                     st=st,
                                                     en=en,
                                                     **kwargs
                                                     )
                to_return['static'] = static
                to_return['dynamic'] = dyn
            else:
                to_return = dyn

        elif static_features is not None:
            # we want only static
            to_return = self._fetch_static_features(
                station=stations,
                static_features=static_features,
                **kwargs
            )
        else:
            raise ValueError(f"""
            static features are {static_features} and dynamic features are also {dynamic_features}""")

        return to_return

    def fetch_static_features(
            self,
            stations: Union[str, List[str]] = "all",
            static_features: Union[str, List[str]] = "all",
            st=None,
            en=None,
            as_ts=False
    ) -> pd.DataFrame:
        """
        returns static atttributes of one or multiple stations

        Parameters
        ----------
            stations : str
                name/id of station of which to extract the data
            static_features : list/str, optional (default="all")
                The name/names of features to fetch. By default, all available
                static features are returned.
            st :
            en :
            as_ts :

        Examples
        ---------
        >>> from water_datasets import Japan
        >>> dataset = Japan()
        get the names of stations
        >>> stns = dataset.stations()
        >>> len(stns)
            12004
        get all static data of all stations
        >>> static_data = dataset.fetch_static_features(stns)
        >>> static_data.shape
           (12004, 27)
        get static data of one station only
        >>> static_data = dataset.fetch_static_features('01010070')
        >>> static_data.shape
           (1, 27)
        get the names of static features
        >>> dataset.static_features
        get only selected features of all stations
        >>> static_data = dataset.fetch_static_features(stns, ['Drainage_Area_km2', 'Elevation_m'])
        >>> static_data.shape
           (12004, 2)
        """
        return self._fetch_static_features(stations, static_features, st, en, as_ts)


def streamflow_indices_all_stations(
        ds_path: Union[str, os.PathLike],
        cpus: int = None,
        to_netcdf: bool = True,
        verbosity: int = 1
):
    nc_path = os.path.join(ds_path, 'streamflow_indices.nc')

    if to_netcdf and os.path.exists(nc_path):
        if verbosity: print(f"Reading from pre-existing {nc_path}")
        return xr.open_dataset(nc_path)

    cpus = cpus or get_cpus() - 2

    start = time.time()

    stations = GSHA(os.path.dirname(ds_path)).stations()
    paths = [ds_path for _ in range(len(stations))]

    if verbosity: print(f"Reading streamflow indices for {len(stations)} stations using {cpus} cpus")
    # takes ~20 seconds with 110 cpus
    with cf.ProcessPoolExecutor(cpus) as executor:

        results = executor.map(
            streamflow_indices_stn,
            paths,
            stations,
        )

    if verbosity: print(f"Time taken: {time.time() - start:.2f} seconds")

    if to_netcdf:

        encoding = {var: {'dtype': 'float32', 'zlib': True, 'complevel': 3} for var in stations()}

        ds = xr.Dataset({str(stn): xr.DataArray(df) for stn, df in zip(stations, results)})
        if verbosity: print(f"Saving to {nc_path}")
        ds.to_netcdf(nc_path, encoding=encoding)
    else:
        ds = {stn: df for stn, df in zip(stations, results)}

    return ds


def streamflow_indices_stn(
        ds_path: Union[str, os.PathLike],
        stn: str
) -> pd.DataFrame:
    fpath = os.path.join(
        ds_path,
        "StreamflowIndices",
        "StreamflowIndices",
        f'{stn}.csv')

    if not os.path.exists(fpath):
        raise FileNotFoundError(f"{ds_path} for station {stn} not found")

    df = pd.read_csv(
        fpath, index_col=0,
        dtype={'1- percentile': np.float32,
               '10- percentile': np.float32,
               '25- percentile': np.float32,
               'median': np.float32,
               '75- percentile': np.float32,
               '90- percentile': np.float32,
               '99- percentile': np.float32,
               'mean': np.float32,
               'maximum (AMF)': np.float32,
               'AMF occurrence date': str,
               'frequency of high-flow days': np.int32,
               'average duration of high-flow events': np.float32,
               'frequency of low-flow days': np.int32,
               'average duration of low-flow events': np.float32,
               'number of days with Q=0 (days)': np.int32,
               'valid observation days (days)': np.int32,
               }
    )
    df.index = pd.to_datetime(df.index, format='%Y')

    df.index.name = 'years'
    df.columns.name = 'streamflow_indices'

    return df


def lc_variable_stn(ds_path, stn: str) -> pd.DataFrame:
    fpath = os.path.join(
        ds_path,
        "Landcover",
        "Landcover",
        f'{stn}.csv')

    if not os.path.exists(fpath):
        raise FileNotFoundError(f"{ds_path} for station {stn} not found")

    df = pd.read_csv(fpath, index_col=0,
                     # dtype={'1- percentile': np.float32}
                     )

    df.index = pd.to_datetime(df.pop('year'), format='%Y')

    df.index.name = 'years'
    df.columns.name = 'lc_variables'

    return df


def lc_vars_all_stns(
        ds_path: Union[str, os.PathLike],
        cpus: int = None,
        to_netcdf: bool = True,
        verbosity: int = 1
):
    nc_path = os.path.join(ds_path, 'lc_variables.nc')

    if to_netcdf and os.path.exists(nc_path):
        if verbosity: print(f"Reading from pre-existing {nc_path}")
        return xr.open_dataset(nc_path)

    cpus = cpus or get_cpus() - 2

    start = time.time()

    stations = GSHA(os.path.dirname(ds_path)).stations()
    paths = [ds_path for _ in range(len(stations))]

    if verbosity: print(f"Reading landcover variables for {len(stations)} stations using {cpus} cpus")

    with cf.ProcessPoolExecutor(cpus) as executor:

        results = executor.map(
            lc_variable_stn,
            paths,
            stations,
        )

    print(f"Time taken: {time.time() - start:.2f} seconds")

    if to_netcdf:

        encoding = {var: {'dtype': 'float32', 'zlib': True, 'complevel': 3} for var in stations()}

        ds = xr.Dataset({stn: xr.DataArray(val) for stn, val in zip(stations, results)})
        print(f"Saving to {nc_path}")
        ds.to_netcdf(nc_path, encoding=encoding)
    else:
        ds = {stn: df for stn, df in zip(stations, results)}
    return ds


def reservoir_vars_stn(ds_path, stn: str) -> pd.DataFrame:
    fpath = os.path.join(
        ds_path,
        "Reservoir",
        "Reservoir",
        f'{stn}.csv')

    if not os.path.exists(fpath):
        raise FileNotFoundError(f"{ds_path} for station {stn} not found")

    df = pd.read_csv(fpath, index_col=0,
                     dtype={'capacity': np.float32, 'dor': np.float32, 'year': np.int32}
                     )

    df.index = pd.to_datetime(df.index, format='%Y')

    df.index.name = 'years'
    df.columns.name = 'reservoir_variables'

    return df


def reservoir_vars_all_stns(
        ds_path: Union[str, os.PathLike],
        cpus: int = None,
        to_netcdf: bool = True,
        verbosity: int = 1
):
    nc_path = os.path.join(ds_path, 'reservoir_variables.nc')

    if to_netcdf and os.path.exists(nc_path):
        if verbosity: print(f"Reading from pre-existing {nc_path}")
        return xr.open_dataset(nc_path)

    cpus = cpus or get_cpus() - 2

    start = time.time()

    stations = GSHA(os.path.dirname(ds_path)).stations()
    paths = [ds_path for _ in range(len(stations))]

    if verbosity: print(f"Reading reservoir variables for {len(stations)} stations using {cpus} cpus")

    with cf.ProcessPoolExecutor(cpus) as executor:

        results = executor.map(
            reservoir_vars_stn,
            paths,
            stations,
        )

    print(f"Time taken: {time.time() - start:.2f} seconds")

    if to_netcdf:

        encoding = {var: {'dtype': 'float32', 'zlib': True, 'complevel': 3} for var in stations}

        ds = xr.Dataset({stn: xr.DataArray(val) for stn, val in zip(stations, results)})
        print(f"Saving to {nc_path}")
        ds.to_netcdf(nc_path, encoding=encoding)
    else:
        ds = {stn: df for stn, df in zip(stations, results)}
    return ds


def lai_stn(ds_path, stn: str) -> pd.Series:
    fpath = os.path.join(
        ds_path,
        "LAI",
        "LAI",
        f'{stn}.csv')

    if not os.path.exists(fpath):
        raise FileNotFoundError(f"{ds_path} for station {stn} not found")

    df = pd.read_csv(fpath, index_col='date',
                     dtype={stn: np.float32}
                     )

    df.index = pd.to_datetime(df.index)

    df.index.name = 'time'

    return df[stn]


def lai_all_stns(
        ds_path: Union[str, os.PathLike],
        cpus: int = None,
        to_netcdf: bool = True,
        verbosity: int = 1
):
    if to_netcdf:
        nc_path = os.path.join(ds_path, 'lai.nc')
        if os.path.exists(nc_path):
            if verbosity: print(f"Reading from pre-existing {nc_path}")
            return xr.open_dataset(nc_path)
    elif os.path.exists(os.path.join(ds_path, 'lai.csv')):
        if verbosity: print(f"Reading from pre-existing {ds_path}")
        return pd.read_csv(os.path.join(ds_path, 'lai.csv'), index_col=0)

    cpus = cpus or get_cpus() - 2

    start = time.time()

    stations = GSHA(os.path.dirname(ds_path)).stations()
    paths = [ds_path for _ in range(len(stations))]

    if verbosity: print(f"Reading lai for {len(stations)} stations using {cpus} cpus")

    with cf.ProcessPoolExecutor(cpus) as executor:

        results = executor.map(
            lai_stn,
            paths,
            stations,
        )

    if verbosity: print(f"Time taken: {time.time() - start:.2f} seconds")

    if to_netcdf:

        encoding = {stn: {'dtype': 'float32', 'zlib': True, 'complevel': 3} for stn in stations}

        nc_path = os.path.join(ds_path, 'lai.nc')
        ds = xr.Dataset({stn: xr.DataArray(val) for stn, val in zip(stations, results)})
        print(f"Saving to {nc_path}")
        ds.to_netcdf(nc_path, encoding=encoding)
    else:
        ds = pd.concat(results, axis=1)
        csv_path = os.path.join(ds_path, 'lai.csv')
        if verbosity: print(f"Saving to {csv_path}")
        ds.to_csv(csv_path, index=True)

    return ds


# the dates for data to be downloaded 
START_YEAR = 1979
END_YEAR = 2023


class Japan(_GSHA):
    """
    Data of 694 catchments of Japan from 
    `river.go.jp website <http://www1.river.go.jp>`_ .
    The meteorological data static catchment features and catchment boundaries 
    taken from `GSHA <https://doi.org/10.5194/essd-16-1559-2024>`_ project. Therefore,
    the number of staic features are 35 and dynamic features are 27 and the
    data is available from 1979-01-01 to 2022-12-31.
    """

    def __init__(
            self, 
            path:Union[str, os.PathLike] = None,
            gsha_path:Union[str, os.PathLike] = None,
            verbosity:int=1,
            **kwargs):
        super().__init__(path=path, verbosity=verbosity, 
                         gsha_path=gsha_path, **kwargs)
    
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'area': catchment_area(),
                'lat': gauge_latitude(),
                'long': gauge_longitude(),

        }

    @property
    def agency_name(self)->str:
        return 'MLIT'

    def _maybe_move_and_merge_shpfiles(self):

        out_shp_file = os.path.join(self.path, "boundaries.shp")

        if not os.path.exists(out_shp_file):
            df = self.gsha._coords()
            jpn_stns = df.loc[df['agency'] == 'MLIT']
            shp_path = os.path.join(self.gsha.path, "WatershedPolygons", 
                                    "WatershedPolygons")
            shp_files = [os.path.join(shp_path, f"{filename}.shp") for filename in jpn_stns['station_id'].values]
            for f in shp_files:
                assert os.path.exists(f)

            merge_shapefiles(shp_files, out_shp_file, add_new_field=True)
        return

    def get_q(self, as_dataframe:bool=True)->pd.DataFrame:
        """reads daily streamflow for all stations and puts them in a single
        file named data.csv. If data.csv is already present, then it is read
        and its contents are returned as dataframe.
        """

        if self.timestep in ('daily', 'D'):
            df = download_daily_data(
                self.stations(), 
                self.path, 
                verbosity=self.verbosity,
                cpus=self.processes
                )
        else:
            df = self.get_hourly_data(cpus=self.processes)

        df.index.name = 'time'

        if as_dataframe:
            return df
        
        df = xr.Dataset({stn: xr.DataArray(df.loc[:, stn]) for stn in df.columns})
        return df

    def get_hourly_data(self, cpus=None):

        hourly_file = os.path.join(self.path, 'hourly_data.csv')

        if os.path.exists(hourly_file):
            print(f"reading hourly data from {hourly_file}")
            return pd.read_csv(hourly_file, index_col=0)

        path = os.path.join(self.path, 'hourly_files')
        
        if self.verbosity>0: print(f"preparing hourly data using {cpus} cpus")

        stn_qs = []
        for idx, stn in enumerate(self.stations()):
            stn_q = download_hourly_stn(path, stn=stn, cpus=cpus, verbosity=self.verbosity)

            if self.verbosity>0: print(f"{idx} {stn}, {len(stn_q)}, {stn_q.index[0]}")
            
            stn_qs.append(stn_q)
        
        q = pd.concat(stn_qs, axis=1)

        q.to_csv(hourly_file)
        return q


def download_daily_stn_yr(
        stn:str="309191289913130",
        yr:int=1979,
)->pd.Series:
    """downloads daily data for a year for a station"""

    url = f'http://www1.river.go.jp/cgi-bin/DspWaterData.exe?KIND=7&ID={stn}&BGNDATE={yr}0131&ENDDATE={yr}1231&KAWABOU=NO'
    df = pd.read_html(url, encoding='EUC-JP')[1].loc[2:, 1:].reset_index(drop=True)

    # make a dictionary with months as keys and number of days as values
    days_in_month = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

    # if it is a leap year, change the number of days in February
    if yr % 4 == 0:
        days_in_month[2] = 29

    assert len(df)<13, len(df)

    yearly_data = []
    for i in range(0, len(df)):
        row = df.iloc[i, 0:days_in_month[i+1]]
        yearly_data.append(row)

    stn_data = pd.concat(yearly_data).reset_index(drop=True)
    stn_data.index = pd.date_range(start=f'{yr}-01-01', end=f'{yr}-12-31', freq='D')
    stn_data.name = stn

    return stn_data


def download_daily_data(
        stations:List[str], 
        path:Union[str, os.PathLike], 
        verbosity:int=1,
        cpus:int=None
        ):
    """downloads daily data for all stations"""
    csv_path = os.path.join(path, 'daily_q.csv')

    if os.path.exists(csv_path):
        if verbosity:
            print(f"reading daily data from {csv_path}")
        return pd.read_csv(csv_path, index_col=0, parse_dates=True)

    years = range(START_YEAR, END_YEAR+1)
    stations_ = [[stn]*len(years) for stn in stations]
    # flatten the list
    stations_ = [item for sublist in stations_ for item in sublist]
    years_ = list(years) * len(stations)

    cpus = cpus or get_cpus()-2

    if verbosity:
        print(f"downloading daily data for {len(stations)} stations from {years[0]} to {years[-1]}")
    
    if verbosity>1:
        print(f"Total function calls: {len(stations_)} with {cpus} cpus")

    start = time.time()
    with cf.ProcessPoolExecutor(cpus) as executor:
        results = executor.map(download_daily_stn_yr, stations_, years_)
    
    if verbosity:
        print(f"total time taken to download data: {time.time() - start}")

    all_data = []
    for stn in stations:
        stn_data = []
        for yr in years:
            stn_yr_data = next(results)
            stn_data.append(stn_yr_data)
        
        stn_data = pd.concat(stn_data, axis=0)
        stn_data.name = stn

        if verbosity>2:
            print(f"total number of years: {yr} for {stn} with shape {stn_data.shape}")

        all_data.append(stn_data)
    
    if verbosity>2:
        print(f"total number of stations: {len(all_data)} each with shape {all_data[0].shape}")
    
    all_data = pd.concat(all_data, axis=1)

    all_data = all_data.replace({'−': np.nan, '欠測': np.nan})
    all_data = all_data.astype(np.float32)    
    if verbosity:
        print(f"saving daily data to {csv_path} with shape {all_data.shape}")
    all_data.to_csv(csv_path)
    return


def download_hourly_stn_day(
        stn:str="309191289913130", 
        st:str="20211227",
        en:str="20211227"
        ):
    """download hourly data for a single day for a single station"""
    url = f"http://www1.river.go.jp/cgi-bin/SrchSiteSuiData2.exe?SUIKEI=90336000&BGNDATE={st}&ENDDATE={en}&ID={stn}:0202;"
    
    data = pd.read_html(url)[0]       
    df = data.iloc[7:]
    df.columns = ['date', 'time', stn]

    if len(df)>0:
        # make sure that we have data for all 24 hours
        assert len(df) == 24, len(df)
    else:
        df.pop(stn)
        df.insert(2, stn, [np.nan for _ in range(24)])

    df.index = pd.date_range(pd.Timestamp(st), periods=24, freq="H")

    return df[stn]


def download_hourly_stn(
        path:Union[str, os.PathLike],
        stn:str="301031281101030", 
        st_yr:int=1980, 
        en_yr:int=2024,
        cpus:int=64,
        verbosity:int = 1
        )->pd.Series:
    
    fpath = os.path.join(path, f"{stn}.csv")
    if os.path.exists(fpath):
        if verbosity>0: print(f"{stn} already exists")
        return pd.read_csv(fpath, index_col=0)
    
    starts, ends = [], [] 
    for yr in range(st_yr, en_yr):

        if yr % 4 == 0:
            n_days = 366
        else:
            n_days = 365

        for j_day in range(0, n_days):
            # convert jday to date
            date = pd.Timestamp(f"{yr}-01-01") + pd.Timedelta(days=j_day)
            st = date.strftime("%Y%m%d")
            en = date.strftime("%Y%m%d")

            starts.append(st)
            ends.append(en)

    stations = [stn for _ in range(len(starts))]
    with cf.ProcessPoolExecutor(cpus) as executor:
        results = executor.map(download_hourly_stn_day, stations, starts, ends)

    yr_df = []
    for res in results:
        yr_df.append(res)

    q = pd.concat(yr_df)

    # replace "欠測" with np.nan
    q = q.replace("欠測", np.nan)
    q = q.replace('-', np.nan)

    q = q.dropna().astype(np.float32).sort_index()

    # drop duplicated index
    q = q[~q.index.duplicated(keep='first')]

    q.to_csv(fpath)
    return q


class Arcticnet(_GSHA):
    """
    Data of 106 catchments of arctic region from 
    `r-arcticnet project <https://www.r-arcticnet.sr.unh.edu/v4.0/AllData/index.html>`_ .
    The meteorological data static catchment features and catchment boundaries 
    taken from `GSHA <https://doi.org/10.5194/essd-16-1559-2024>`_ project. Therefore,
    the number of staic features are 35 and dynamic features are 27 and the
    data is available from 1979-01-01 to 2003-12-31.
    """

    def __init__(
            self, 
            path:Union[str, os.PathLike] = None,
            gsha_path:Union[str, os.PathLike] = None,
            verbosity:int=1,
            **kwargs):
        super().__init__(path=path, gsha_path=gsha_path, verbosity=verbosity, **kwargs)
    
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.metadata = self.get_metadata()
        self._get_q()

        self._stations = [stn for stn in self.all_stations() if stn in self.gsha_arctic_stns()]

        self._static_features = self.gsha.static_features

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'area': catchment_area(),
                'lat': gauge_latitude(),
                'long': gauge_longitude(),
        }

    @property
    def end(self)->pd.Timestamp:
        return pd.Timestamp('2003-12-31')
    
    def stations(self)->List[str]:
        return self._stations

    def all_stations(self):
        return self.metadata.index.astype(str).tolist()

    @property
    def agency_name(self)->str:
        return 'arcticnet'

    def get_metadata(self):
        metadata_path = os.path.join(self.path, "metadata.csv")
        if not os.path.exists(metadata_path):
            df = pd.read_csv(
                "https://www.r-arcticnet.sr.unh.edu/v4.0/russia-arcticnet/Daily_SiteAttributes.txt", 
                #"https://www.r-arcticnet.sr.unh.edu/v4.0/data/SiteAttributes.txt",
                sep="\t",
                encoding_errors='ignore',
                )
            df.index = df.pop('Code')
            df.to_csv(metadata_path, index=True)
        else:
            df = pd.read_csv(metadata_path, index_col=0)
        return df

    def get_q(self, as_dataframe:bool=True):
        nc_path = os.path.join(self.path, "daily_q.nc")

        if os.path.exists(nc_path):
            if self.verbosity:
                print(f"Reading {nc_path}")
            q_ds = xr.open_dataset(nc_path)
        else:
            q_ds = xr.Dataset({stn:self.get_stn_q(stn) for stn in self.stations()})
            # rename dimension/coordinate to 'time' in q_ds
            q_ds = q_ds.rename({'dim_0':'time'})

            encoding = {stn: {'dtype': 'float32', 'zlib': True, 'complevel': 3} for stn in self.stations()}
            if self.verbosity:
                print(f"Writing {nc_path}")
            q_ds.to_netcdf(nc_path, encoding=encoding)

        if as_dataframe:
            q_ds = q_ds.to_dataframe()
        return q_ds

    def _get_q(self, as_dataframe:bool=True):
        q_path = os.path.join(self.path, "daily_q.csv")
        if not os.path.exists(q_path):
            df = pd.read_csv(
                "https://www.r-arcticnet.sr.unh.edu/v4.0/russia-arcticnet/discharge_m3_s_UNH-UCLA.txt", 
                #"https://www.r-arcticnet.sr.unh.edu/v4.0/data/Discharge_ms.txt",
                sep="\t")
            df.to_csv(q_path)
        else:
            df = pd.read_csv(q_path, index_col=0)
        return df

    def get_stn_q(self, stn: str):
        q = self._get_q()
        stn_q = q.loc[q['Code']==int(stn), :].copy()

        stn_q.drop('Code', axis=1, inplace=True)

        # Function to check leap year and adjust February days
        def adjust_feb_days(year):
            if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
                return 29
            return 28

        month_days1 = {
            'Jan': 31, 'Feb': 28, # February will be adjusted dynamically
            'Mar': 31, 'Apr': 30, 'May': 31, 'Jun': 30,
            'Jul': 31, 'Aug': 31, 'Sep': 30, 'Oct': 31,
            'Nov': 30, 'Dec': 31
        }

        # Prepare a mapping of month names to the number of days
        month_days2 = {
            1: 31, 2: 28, # February will be adjusted dynamically
            3: 31, 4: 30, 5: 31, 6: 30,
            7: 31, 8: 31, 9: 30, 10: 31,
            11: 30, 12: 31
        }

        # Melt the DataFrame to convert it from wide format to long format
        df_long = stn_q.melt(id_vars=['Year', 'Day'], value_vars=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                        var_name='Month', value_name='Value')

        month_to_int = {month: i + 1 for i, month in enumerate(month_days1.keys())}
        df_long['Month'] = df_long['Month'].map(month_to_int).astype(int)

        def get_days_in_month(row):
            try:
                if int(row.Month) == 2:
                    return adjust_feb_days(row.Year)
                return month_days2[int(row.Month)]
            except KeyError as e:
                print(f"KeyError encountered: {e} with row details: {row}")
                return None  # Or handle the error as needed

        df_long['DaysInMonth'] = df_long.apply(get_days_in_month, axis=1)

        # Filter out invalid days (e.g., February 30)
        df_long = df_long[df_long['Day'] <= df_long['DaysInMonth']]

        # Create a datetime column from the Year, Month, Day
        df_long.index = pd.to_datetime(df_long[['Year', 'Month', 'Day']])

        return pd.Series(df_long['Value'], name=str(stn))        

    def gsha_arctic_stns(self)->List[str]:

        df = self.gsha.wsAll.copy()
        return [stn.split('_')[0] for stn in df[df.index.str.endswith('_arcticnet')].index]


class Spain(_GSHA):
    """
    Data of 889 catchments of Spain from 
    `ceh-es <https://ceh-flumen64.cedex.es>`_ website.
    The meteorological data static catchment features and catchment boundaries 
    taken from `GSHA <https://doi.org/10.5194/essd-16-1559-2024>`_ project. Therefore,
    the number of staic features are 35 and dynamic features are 27 and the
    data is available from 1979-01-01 to 2020-09-30.
    """

    def __init__(
            self, 
            path:Union[str, os.PathLike] = None,
            gsha_path:Union[str, os.PathLike] = None,
            overwrite:bool=False,
            verbosity:int=1,
            **kwargs):
        super().__init__(
            path=path, 
            gsha_path=gsha_path, 
            overwrite=overwrite,
            verbosity=verbosity,
            **kwargs)

        self.areas = [
            "CANTABRICO", "DUERO", "EBRO", "GALICIA COSTA",
            "GUADALQUIVIR", "GUADIANA", "JUCAR", "MIÑO-SIL",
            "SEGURA", "TAJO"
        ]

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'area': catchment_area(),
                'lat': gauge_latitude(),
                'long': gauge_longitude(),
        }

    @property
    def end(self)->pd.Timestamp:
        return pd.Timestamp('2020-09-30')
    
    @property
    def agency_name(self)->str:
        return 'AFD'
    
    def daily_q_all_areas(self)->pd.DataFrame:
        """Daily data of gauging stations in river from all areas

        Retuns
        ------
        16_806_305 rows x 3
        """
        dfs = []
        for area in self.areas:
            df = self.daily_q_area(area)
            dfs.append(df)

        return pd.concat(dfs)

    def daily_q_area(self, area:str)->pd.DataFrame:
        """Reads Daily data of gauging stations in river which is in afliq.csv file"""

        url = f"https://ceh-flumen64.cedex.es/anuarioaforos//anuario-2019-2020/{area}/afliq.csv"

        df = pd.read_csv(url, #os.path.join(self.path, area, "afliq.csv"),
                         sep=';')

        idx = pd.to_datetime(df.pop('fecha'), dayfirst=True)
        df.index = idx
        df.index.name = "date"
        df.columns = ['stations', 'height_m', "q_cms"]
        return df

    def get_q(self, as_dataframe:bool=True):
        """
        returns daily q of all stations

        Returns
        -------
        pd.DataFrame
            a pandas dataframe of shape (39721, 1447)
        """

        fpath = os.path.join(self.path, 'daily_q.csv')

        if os.path.exists(fpath):
            data= pd.read_csv(fpath, index_col='Unnamed: 0')
            data.index = pd.to_datetime(data.index)
            data.index.name = 'time'
            if as_dataframe:
                return data
            return xr.Dataset({stn: xr.DataArray(data.loc[:, stn]) for stn in data.columns})

        q = self.daily_q_all_areas()

        st = []
        en = []
        for g_name, grp in q.groupby('stations'):
            st.append(grp.sort_index().index[0])
            en.append(grp.sort_index().index[-1])

        start = pd.to_datetime(st).sort_values()[0]
        end = pd.to_datetime(en).sort_values()[-1]

        daily_qs = []
        for stn, stn_df in q.groupby('stations'):
            q_ts = pd.Series(name=stn,
                             index=pd.date_range(start, end=end, freq="d"),
                             dtype=np.float32)
            q_ts[stn_df.index] = stn_df['q_cms']
            daily_qs.append(q_ts)

        data = pd.concat(daily_qs, axis=1)

        data.to_csv(fpath)

        data.index.name = 'time'

        if as_dataframe:
            return data
    
        return xr.Dataset({stn: xr.DataArray(data.loc[:, stn]) for stn in data.columns})


class Thailand(_GSHA):
    """
    Data of 73 catchments of Thailand from 
    `RID project <https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/rid-river/disc_d.html>`_ .
    The meteorological data static catchment features and catchment boundaries 
    taken from `GSHA <https://doi.org/10.5194/essd-16-1559-2024>`_ project. Therefore,
    the number of staic features are 35 and dynamic features are 27 and the
    data is available from 1980-01-01 to 1999-12-31.
    """
    url = {
'disc_d_1980_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1980_RIDall.zip',
'disc_d_1981_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1981_RIDall.zip',
'disc_d_1982_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1982_RIDall.zip',
'disc_d_1983_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1983_RIDall.zip',
'disc_d_1984_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1984_RIDall.zip',
'disc_d_1985_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1985_RIDall.zip',
'disc_d_1986_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1986_RIDall.zip',
'disc_d_1987_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1987_RIDall.zip',
'disc_d_1988_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1988_RIDall.zip',
'disc_d_1989_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1989_RIDall.zip',
'disc_d_1990_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1990_RIDall.zip',
'disc_d_1991_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1991_RIDall.zip',
'disc_d_1992_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1992_RIDall.zip',
'disc_d_1993_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1993_RIDall.zip',
'disc_d_1994_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1994_RIDall.zip',
'disc_d_1995_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1995_RIDall.zip',
'disc_d_1996_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1996_RIDall.zip',
'disc_d_1997_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1997_RIDall.zip',
'disc_d_1998_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1998_RIDall.zip',
'disc_d_1999_RIDall.zip': 'https://hydro.iis.u-tokyo.ac.jp/GAME-T/GAIN-T/routine/data/disc/disc_d_1999_RIDall.zip',
    }

    def __init__(
            self, 
            path:Union[str, os.PathLike] = None,
            gsha_path:Union[str, os.PathLike] = None,
            overwrite:bool=False,
            verbosity:int=1,
            **kwargs):
        super().__init__(
            path=path, 
            gsha_path=gsha_path, 
            overwrite=overwrite,
            verbosity=verbosity,
            **kwargs)

        self._download(overwrite=overwrite)

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'area': catchment_area(),
                'lat': gauge_latitude(),
                'long': gauge_longitude(),
        }

    @property
    def agency_name(self)->str:
        return 'RID'

    @property
    def start(self)->pd.Timestamp:
        return pd.Timestamp('1980-01-01')

    @property
    def end(self)->pd.Timestamp:
        return pd.Timestamp('1999-12-31')

    def get_q(self, as_dataframe:bool=True):
        """reads q"""

        fpath = os.path.join(self.path, 'daily_q.csv')

        if os.path.exists(fpath):
            if self.verbosity:
                print(f"Reading {fpath}")
            data = pd.read_csv(fpath, index_col=0, parse_dates=True)
            if as_dataframe:
                return data
            return xr.Dataset({stn: xr.DataArray(data.loc[:, stn]) for stn in data.columns})

        datas = []
        for year in range(1980, 2000):
            data = self._read_year(year)
            datas.append(data)

        data = pd.concat(datas)
        #data.columns = [column.replace('.', '_') for column in data.columns.tolist()]
        data.index.name = 'time'

        if self.verbosity:
            print(f"Writing {fpath}")
        data.to_csv(fpath)

        if as_dataframe:
            return data

        return xr.Dataset({stn: xr.DataArray(data.loc[:, stn]) for stn in data.columns})

    def _read_year(self, year:int):
        year_path = os.path.join(self.path, f'disc_d_{year}_RIDall')
        yr_dfs = []
        stn_ids = []

        ndays = 365
        if year%4==0:
            ndays = 366

        for file in os.listdir(year_path):
            fpath = os.path.join(year_path, file)

            df = pd.read_csv(
                fpath,
                sep='\t',
                names=['index', 'q_cms'],
                nrows=ndays,
                na_values=-9999.0,
            )

            df.index = pd.to_datetime(df.pop('index'))

            yr_dfs.append(df)

            stn_id = file.split('RID')[1].split('_m3s')[0]
            stn_ids.append(stn_id)

        df = pd.concat(yr_dfs, axis=1)
        df.columns = stn_ids
        return df
