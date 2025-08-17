
import os
from typing import Union, List, Dict

import pandas as pd

from .._datasets import Datasets
from ..utils import check_attributes


class CamelsChem(Datasets):
    """
    Water Quality data from USA following the works of `Sterle et al., 2024 <https://doi.org/10.5194/hess-28-611-2024>`_ .
    This dataset has 18 water chemistry parameters from 1980-01-01 - 2018-12-31.
    The data is is downloaded from `hydroshare <https://www.hydroshare.org/resource/841f5e85085c423f889ac809c1bed4ac/>`_
    Out of 671 stations, 155 stations have not water quality data.
    The wet depisition data consist of 12 parameters from 1985 - 2018.

    Examples
    --------
    >>> from aqua_fetch import CamelsChem
    >>> ds = CamelsChem(path='/path/to/dataset')
    >>> len(ds.stations())
    516
    >>> len(ds.parameters)
    28
    """
    url1 = {
        "Camels_chem_1980_2018.csv":
"https://www.hydroshare.org/resource/841f5e85085c423f889ac809c1bed4ac/data/contents/Camels_Chem_%20Dataset",
        "Gauge_and_region_names.csv":
"https://www.hydroshare.org/resource/841f5e85085c423f889ac809c1bed4ac/data/contents/Camels_Chem_%20Dataset",
        "Camel_Chem_Metrics.xlsx":
"https://www.hydroshare.org/resource/841f5e85085c423f889ac809c1bed4ac/data/contents/Camels_Chem_%20Dataset",
        "DepCon_671_1985_2018.xlsx":
"https://www.hydroshare.org/resource/841f5e85085c423f889ac809c1bed4ac/data/contents/Camels_Chem_%20Dataset",
        "DepCon_metadata.xlsx": 
"https://www.hydroshare.org/resource/841f5e85085c423f889ac809c1bed4ac/data/contents/Camels_Chem_%20Dataset",
        "Camels_topography.csv":
"https://www.hydroshare.org/resource/841f5e85085c423f889ac809c1bed4ac/data/contents/Catchment_attributes/",
    }

    def __init__(self, path=None, **kwargs):
        super().__init__(path=path, **kwargs)
        self._download()

        self._parameters = self.data().columns[5:-12].to_list()
        self._stations = self.gauge_and_region_names()['gauge_id'].tolist()

        self._atm_dep_parameters = self.__atm_dep_parameters()
    
    def _download(self, overwrite=False, **kwargs):
        
        for k,v in self.url1.items():
            url = f"{v}/{k}"

            if k.endswith('csv'):
                fpath = os.path.join(self.path, k)

                if not os.path.exists(fpath) or self.overwrite:
                    if self.verbosity:
                        print(f"Downloading {k} from {url}")
                    pd.read_csv(url).to_csv(fpath)
                else:
                    if self.verbosity:
                        print(f"{k} already exists in {self.path}")

            elif k.endswith('xlsx'):
                fpath = os.path.join(self.path, k)

                if not os.path.exists(fpath) or self.overwrite:
                    if self.verbosity:
                        print(f"Downloading {k} from {url}")
                    pd.read_excel(url).to_excel(fpath)
                else:
                    if self.verbosity:
                        print(f"{k} already exists in {self.path}")
        return

    @property
    def parameters(self)->List[str]:
        """
        returns the names of parameters in the dataset
        """
        return self._parameters

    @property
    def atm_dep_parameters(self)->List[str]:
        """
        returns the names of parameters in the atm_dep dataset
        """
        return self._atm_dep_parameters

    def topography(self)->pd.DataFrame:
        """
        reads the topography data
        """
        fname = os.path.join(self.path, 'Camels_topography.csv')
        df = pd.read_csv(fname, index_col='gauge_id', dtype={'gauge_id': str})
        df.pop('Unnamed: 0')
        return df

    def stn_coords(self)->pd.DataFrame:
        """
        Returns the coordinates of all the stations in the dataset in wgs84
        projection.

        Returns
        -------
        pd.DataFrame
            A dataframe with columns 'lat', 'long'
        """
        coords =  self.topography()[['gauge_lat', 'gauge_lon']]
        coords.columns = ['lat', 'long']
        return coords

    def stations(self)->List[str]:
        """
        returns the list of stations in the dataset
        """
        return self._stations

    def metrics(self):
        """
        reads metrics.xlsx which contains metadata
        """
        return pd.read_excel(os.path.join(self.path, "Camel_Chem_Metrics.xlsx"), index_col=0)
    
    def data(self)->pd.DataFrame:
        """
        reads the main dataset which has shape of (76284, 45)
        """
        data =  pd.read_csv(
            os.path.join(self.path, "Camels_chem_1980_2018.csv"), 
            index_col=1,
            dtype={'gauge_id': str}
            )
        
        metrics = self.metrics()

        units = {}
        for label, unit in zip(metrics['Column label'], metrics['Units'].to_list()):
            if isinstance(unit, str) and not unit.startswith('Y'):
                new_label = f"{label}_{unit}"
                units[label] = new_label
        data = data.rename(columns=units)     
        return data   

    def fetch(
            self,
            stations: Union[str, List[str]] = "all",
            parameters: Union[str, List[str]] = "all",
    )->Dict[str, pd.DataFrame]:
        """
        fetches the data for the given stations and parameters

        Parameters
        ----------
        stations: Union[str, List[str]]
            list of stations to fetch data for
        parameters: Union[str, List[str]]
            list of parameters to fetch data for
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            dictionary of dataframes for each station
        
        Examples
        --------
        >>> ds = CamelsChem(path='/mnt/datawaha/hyex/atr/data')
        >>> data = ds.fetch(stations=['1591400', '6350000'], parameters=['cl_mg/l', 'na_mg/l'])
        >>> data = ds.fetch('1591400', 'cl_mg/l')['1591400']
        >>> data.shape # (55, 1)        
        ... get all parameters for a station
        >>> data = ds.fetch('1591400')['1591400']
        >>> data.shape # (55, 28)
        >>> all_data = ds.fetch()  # get all parameters of all stations
        >>> len(all_data) # 516
        """
        parameters = check_attributes(parameters, self.parameters, 'parameters')
        stations = check_attributes(stations, self.stations())

        out = {}

        data = self.data()

        for stn in stations:

            if stn in data.index:

                stn_data = data.loc[stn, parameters + ['sample_timestamp']]
                if len(stn_data.shape) > 1:
                    stn_data.index = pd.to_datetime(stn_data['sample_timestamp'])
                else:
                    stn_data = pd.DataFrame(stn_data).transpose()
                    stn_data.index = stn_data['sample_timestamp']
                stn_data.drop(columns='sample_timestamp', inplace=True)
                out[stn] = stn_data

        return out
    
    def gauge_and_region_names(self)->pd.DataFrame:
        """
        reads the gauge and region names
        """
        return pd.read_csv(
            os.path.join(self.path, "Gauge_and_region_names.csv"), 
            index_col=0,
            dtype={'gauge_id': str}
            )

    def atm_dep_metadata(self)->pd.DataFrame:
        """
        reads the atm_dep_metadata
        """
        fpath = os.path.join(self.path, 'DepCon_metadata.xlsx')
        metadata = pd.read_excel(fpath, index_col=0)
        return metadata
    
    def atm_dep_data(self)->pd.DataFrame:
        """
        reads the atmospheric deposition data
        """
        fpath = os.path.join(self.path, 'DepCon_671_1985_2018.xlsx')
        atm_dep = pd.read_excel(fpath, index_col=1, dtype={'gauge_id': str})
        atm_dep.index = atm_dep.index.astype(str)
        atm_dep.drop(columns='Unnamed: 0', inplace=True)
        return atm_dep

    def __atm_dep_parameters(self)->List[str]:
        parameters = []
        for col in self.atm_dep_data().columns:
            parameter = col.split('_')[0]
            parameters.append(parameter)
        parameters = list(set(parameters))
        parameters.remove('Year')
        return parameters

    def fetch_atm_dep(
            self,
            stations: Union[str, List[str]] = "all",
            parameters: Union[str, List[str]] = "all",
    )->Dict[str, pd.DataFrame]:
        """
        fetches the data for the given stations and parameters

        Parameters
        ----------
        stations: Union[str, List[str]]
            list of stations to fetch data for
        parameters: Union[str, List[str]]
            list of parameters to fetch data for
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            dictionary of dataframes for each station

        Examples
        --------
        >>> ds = CamelsChem(path='/mnt/datawaha/hyex/atr/data')
        ... get data for a single station and a single parameter
        >>> data = ds.fetch_atm_dep(stations='1591400', parameters='cl')
        >>> print(data['1591400'].shape)  # (34, 8)
        ... get data for multiple stations and multiple parameters
        >>> data = ds.fetch_atm_dep(stations=['1591400', '6350000'], parameters=['cl', 'na'])
        >>> print(data['1591400'].shape)  # (34, 16)
        >>> print(data['6350000'].shape)  # (34, 16)
        .. get data for all stations and for all parameters
        >>> data = ds.fetch_atm_dep()
        >>> print(len(data))  # 671
        """
        parameters = check_attributes(parameters, self.atm_dep_parameters)
        stations = check_attributes(stations, self.stations())

        out = {}

        data = self.atm_dep_data()

        for stn in stations:

            if stn in data.index:

                cols = get_cols(parameters, data.columns.tolist())

                stn_data = data.loc[stn, cols + ['Year']]
                if len(stn_data.shape) > 1:
                    stn_data.index = pd.PeriodIndex(stn_data['Year'], freq='Y')
                else:
                    stn_data = pd.DataFrame(stn_data).transpose()
                    stn_data.index = stn_data['Year']
                stn_data.drop(columns='Year', inplace=True)
                out[stn] = stn_data

        return out


def get_cols(parameters:List[str], all_cols:List[str])->List[str]:
    cols = []
    for para in parameters:
        for col in all_cols:
            if col.startswith(para):
                cols.append(col)
    
    return cols


class CamelsCHChem(Datasets):
    """
    Water quality data for Switzerland following the work of 
    `Nascimento et al., 2025 <https://eartharxiv.org/repository/view/9046/>`_
    The dataset is downloaded from `zenodo <https://zenodo.org/records/16158375>`_ .
    It contains over 40 water quality parameters  for 115 Swiss catchments from 1980-01-01 - 2020-12-31.
    
    Examples
    --------
    >>> from aqua_fetch import CamelsCHChem
    >>> dataset = CamelsCHChem(path='/path/to/data')
    >>> stns = dataset.stations()
    >>> len(stns)
    115
    ... find out names of stations
    >>> stns[0:10]
    ['2009', '2011', '2016', '2018', ... '2044']
    ... get longitude and latitude of stations
    >>> coords = dataset.stn_coords()
    >>> coords.shape
    (115, 2)
    >>> data = dataset.fetch_catch_avg('2009')
    >>> type(data)
    dict
    >>> len(data)
    1
    >>> data.keys()
    '2009'
    >>> data['2009'].shape
    (209, 32)
    >>> data = dataset.fetch_catch_avg(['2009', '2011', '2018'])
    >>> data.keys()
    dict_keys(['2009', '2011', '2018'])
    >>> [val.shape for val in data.values()]
    [(209, 32), (209, 32), (209, 32)]
    >>> data['2009'].columns.tolist()
    ['cereal', 'maize', 'sugarbeet', ... 'gve_ha', 'delta_2h']
    ... find out start and end dates
    >>> data['2009'].index[0], data['2009'].index[-1]
    (Timestamp('1970-01-01'), Timestamp('2020-12-15'))
    ...
    ... get water quality time series
    >>> data = dataset.fetch_wq_ts(stations=['2009', '2011'])
    >>> data['2009'].shape
    (14610, 4)
    >>> data['2011'].shape 
    (14610, 4)
    >>> data['2011'].columns
    Index(['temp_sensor', 'pH_sensor', 'ec_sensor', 'O2C_sensor'], dtype='object')
    >>> data = dataset.fetch_wq_ts()
    >>> len(data) 
    115
    >>> data['2009'].index[0], data['2009'].index[-1]
    (Timestamp('1981-01-01 00:00:00'), Timestamp('2020-12-31 00:00:00'))
    ...
    # get isotope data
    >>> data = dataset.fetch_isotope(stations=['2009', '2016'])
    >>> data['2009'].shape 
    (452, 4)
    >>> data['2016'].shape 
    (450, 4)
    >>> data['2009'].columns
    Index(['date_start', 'date_end', 'delta_2h', 'delta_18o'], dtype='object')
    """

    url = "https://zenodo.org/records/16158375"

    def __init__(self, path=None, **kwargs):
        super().__init__(path=path, **kwargs)
        self._download()

        self._stations = self.metadata().index.tolist()

    def stations(self)->List[str]:
        """
        returns the list of stations in the dataset
        """
        return self._stations

    @property
    def gauge_md_file(self)->os.PathLike:
        """
        returns the gauge metadata file
        """
        return os.path.join(self.path, 
                            'camels-ch-chem', 
                            'gauges_metadata', 
                            'camels_ch_chem_gauges_metadata.csv')
    
    @property
    def catch_avg_data_path(self)->os.PathLike:
        """
        returns the path to the catchment average data
        """
        return os.path.join(self.path, 
                            'camels-ch-chem', 
                            'catchment_aggregated_data')
    
    def metadata(self)->pd.DataFrame:
        """
        reads the metadata file

        Returns
        -------
        pd.DataFrame
        """
        return pd.read_csv(self.gauge_md_file, 
                           index_col='gauge_id', 
                           dtype={'gauge_id': str})

    def stn_coords(self)->pd.DataFrame:
        """
        Returns the coordinates of all the stations in the dataset in wgs84
        projection.

        Returns
        -------
        pd.DataFrame
            A dataframe with columns 'lat', 'long'
        """
        coords =  self.metadata()[['gauge_lat', 'gauge_lon']]
        coords.columns = ['lat', 'long']
        return coords

    def fetch_catch_avg(
            self,
            stations: Union[str, List[str]] = "all",
    )->Dict[str, pd.DataFrame]:
        """
        fetches the catchment average data for the given stations

        Parameters
        ----------
        stations: Union[str, List[str]]
            list of stations to fetch data for

        Returns
        -------
        Dict[str, pd.DataFrame]
            dictionary of dataframes for each station
        
        Examples
        --------
        >>> ds = CamelsCHChem(path='/path/to/data')
        >>> data = ds.fetch_catch_avg(stations=['2009', '2011'])
        >>> print(data['2009'].shape)  # (209, 32)
        >>> print(data['2011'].shape)  # (209, 32)
        """
        stations = check_attributes(stations, self.stations(), 'stations')

        data = {}
        for stn in stations:

            stn_df = []

            for cat, key in {'agricultural_data': 'swisscrops', 
                        'atmospheric_deposition': 'atmdepo', 
                        'landcover_data': 'landcover', 
                        'livestock_data': 'livestock',
                        'rain_water_isotopes': 'rainisotopes'
                        }.items():

                fpath = os.path.join(self.catch_avg_data_path, cat, f'camels_ch_chem_{key}_{stn}.csv')
            
                if not os.path.exists(fpath):
                    cat_df = pd.DataFrame()
                else:
                    cat_df = pd.read_csv(fpath, index_col=0)
                    cat_df.index = pd.to_datetime(cat_df.index)
            
                stn_df.append(cat_df)
            
            if len(stn_df) > 0:
                stn_df = pd.concat(stn_df, axis=1)
            else:
                stn_df = pd.DataFrame()
            
            data[stn] = stn_df

        return data

    def fetch_wq_ts(
            self, 
            stations: Union[str, List[str]] = "all",
            timestep: str = 'D'
            )->Dict[str, pd.DataFrame]:
        """
        fetches the water quality time series data for the given station(s).
        This data consists of 'temp_sensor', 'ph_sensor', 'ec_sensor' and 'O2C_sensor' 
        parameters for the given station(s).

        Parameters
        ----------
        stn: Union[str, List[str]]
            station or list of stations to fetch data for
        timestep: str
            the timestep of the data, default is 'D' for daily data.
            Other option is ``H`` for hourly.
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            dictionary of dataframes for each station
        
        Examples
        --------
        >>> ds = CamelsCHChem(path='/path/to/data')
        >>> data = ds.fetch_wq_ts('2009')['2009']
        >>> print(data.shape)  # (14610, 4)
        """
        stations = check_attributes(stations, self.stations(), 'stations')

        subfolder = 'hourly' if timestep == 'H' else 'daily'

        data = {}
        for stn in stations:
            fpath = os.path.join(self.path, 
                                'camels-ch-chem', 
                                'stream_water_chemistry',
                                'timeseries',
                                subfolder,
                                f'camels_ch_chem_{subfolder}_{stn}.csv')
            
            if not os.path.exists(fpath):
                stn_df = pd.DataFrame()
            else:
                stn_df = pd.read_csv(fpath, index_col=0)
                stn_df.index = pd.to_datetime(stn_df.index)
            
            data[stn] = stn_df

        return data

    def fetch_isotope(
            self, 
            stations: Union[str, List[str]] = "all",
            category: str = "isot"
            )->Dict[str, pd.DataFrame]:

        stations = check_attributes(stations, self.stations(), 'stations')

        data = {}
        for stn in stations:
            fpath = os.path.join(self.path, 
                                'camels-ch-chem', 
                                'stream_water_isotopes',
                                category,
                                f'camels_ch_chem_{category}_{stn}.csv')
            
            if not os.path.exists(fpath):
                stn_df = pd.DataFrame()
            else:
                stn_df = pd.read_csv(fpath)
                stn_df['date_start'] = pd.to_datetime(stn_df['date_start'])
                stn_df['date_end'] = pd.to_datetime(stn_df['date_end'])

            data[stn] = stn_df

        return data
    