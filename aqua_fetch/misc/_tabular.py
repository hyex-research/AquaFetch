import os
import glob
import random
import warnings
from typing import Union, List

import numpy as np
import pandas as pd

from aqua_fetch._backend import netCDF4, xarray as xr
from aqua_fetch.utils import (
    download_and_unzip, 
    unzip_all_in_dir, 
    download,
    check_attributes
    )


SEP = os.sep

from aqua_fetch._datasets import Datasets


class GloHydroRes(Datasets):
    """
    Global dataset of hydropower plant (location, head, type) and reservoir
    (dam and reservoir location, dam height, reservoir depth, area, and volume)
    for 7,775 plants in 128 countries following the work of 
    `Shah et al., 2025 <https://doi.org/10.1038/s41597-025-04975-0>`_.
    """
    url = 'https://zenodo.org/records/14526360'



class Weisssee(Datasets):

    dynamic_attributes = ['Precipitation_measurements',
                          'long_wave_upward_radiation',
                          'snow_density_at_30cm',
                          'long_wave_downward_radiation'
                          ]

    url = '10.1594/PANGAEA.898217'

    def __init__(self, path=None, overwrite=False, **kwargs):
        super(Weisssee, self).__init__(path=path, **kwargs)
        #self.path = path
        self.download_from_pangaea(overwrite=overwrite)

    def fetch(self, **kwargs):
        """
        Examples
        --------
            >>> from aqua_fetch import Weisssee
            >>> dataset = Weisssee()
            >>> data = dataset.fetch()
        """

        data = {}
        for f in self.data_files:
            fpath = os.path.join(self.path, f)
            df = pd.read_csv(fpath, **kwargs)

            if 'index_col' in kwargs:
                df.index = pd.to_datetime(df.index)

            data[f.split('.txt')[0]] = df

        return data


class ETP_CHN_SEBAL(Datasets):

    url = "https://zenodo.org/record/4218413#.YBNhThZS-Ul"


class ISWDC(Datasets):

    url = "https://zenodo.org/record/2616035#.YBNl5hZS-Uk"


class WQJordan(Weisssee):
    """Jordan River water quality data of 9 variables for two variables."""
    url = 'https://doi.pangaea.de/10.1594/PANGAEA.919103'


class WQJordan2(Weisssee):
    """Stage and Turbidity data of Jordan River"""
    url = '10.1594/PANGAEA.919104'


class YamaguchiClimateJp(Weisssee):
    """Daily climate and flow data of Japan from 2006 2018"""
    url = "https://doi.pangaea.de/10.1594/PANGAEA.909880"


class FlowBenin(Weisssee):
    """Flow data"""
    url = "10.1594/PANGAEA.831196"


class HydrometricParana(Weisssee):
    """Daily and monthly water level and flow data of Parana river Argentina
    from 1875 to 2017."""
    url = "https://doi.pangaea.de/10.1594/PANGAEA.882613"


class RiverTempSpain(Weisssee):
    """Daily mean stream temperatures in Central Spain for different periods."""
    url = "https://doi.pangaea.de/10.1594/PANGAEA.879494"


class WQCantareira(Weisssee):
    """Water quality and quantity primary data from field campaigns in the Cantareira Water Supply System,
     period Oct. 2013 - May 2014"""
    url = "https://doi.pangaea.de/10.1594/PANGAEA.892384"


class RiverIsotope(Weisssee):
    """399 δ18O and δD values in river surface waters of Indian River"""
    url = "https://doi.pangaea.de/10.1594/PANGAEA.912582"


class EtpPcpSamoylov(Weisssee):
    """Evpotranspiration and Precipitation at station TOWER on Samoylov Island Russia
     from 20110524 to 20110819 with 30 minute frequency"""
    url = "10.1594/PANGAEA.811076"


class FlowSamoylov(Weisssee):
    """Net lateral flow at station INT2 on Samoylov Island Russia
    from 20110612 to 20110819 with 30 minute frequency"""
    url = "10.1594/PANGAEA.811072"


class FlowSedDenmark(Weisssee):
    """Flow and suspended sediment concentration fields over tidal bedforms, ADCP profile"""
    url = "10.1594/PANGAEA.841977"


class StreamTempSpain(Weisssee):
    """Daily Mean Stream Temperature at station Tormes3, Central Spain from 199711 to 199906."""
    url = "https://doi.pangaea.de/10.1594/PANGAEA.879507"


class RiverTempEroo(Weisssee):
    """Water temperature records in the Eroo River and some tributaries (Selenga River basin, Mongolia, 2011-2012)"""
    url = "10.1594/PANGAEA.890070"


class HoloceneTemp(Weisssee):
    """Holocene temperature reconstructions for northeastern North America and the northwestern Atlantic,
     core Big_Round_Lake."""
    url = "10.1594/PANGAEA.905446"


class FlowTetRiver(Weisssee):
    """Daily mean river discharge at meteorological station Perpignan upstream, Têt basin France from 1980
    to 2000."""
    url = "10.1594/PANGAEA.226925"


class SedimentAmersee(Weisssee):
    """Occurence of flood laminae in sediments of Ammersee"""
    url = "10.1594/PANGAEA.746240"


class HydrocarbonsGabes(Weisssee):
    """Concentration and geological parameters of n-alkanes and n-alkenes in surface sediments from the Gulf of Gabes,
     Tunisia"""
    url = "10.1594/PANGAEA.774595"


class WaterChemEcuador(Weisssee):
    """weekly and biweekly Water chemistry of cloud forest streams at baseflow conditions,
     Rio San Francisco, Ecuador"""
    url = "10.1594/PANGAEA.778629"


class WaterChemVictoriaLakes(Weisssee):
    """Surface water chemistry of northern Victoria Land lakes"""
    url = "10.1594/PANGAEA.807883"


class HydroChemJava(Weisssee):
    """Hydrochemical data from subsurface rivers, coastal and submarine springsin a karstic region
     in southern Java."""
    url = "10.1594/PANGAEA.882178"


class PrecipBerlin(Weisssee):
    """Sub-hourly Berlin Dahlem precipitation time-series 2001-2013"""
    url = "10.1594/PANGAEA.883587"


class GeoChemMatane(Weisssee):
    """Geochemical data collected in shallow groundwater and river water in a subpolar environment
     (Matane river, QC, Canada)."""
    url = "10.1594/PANGAEA.908290"


class HydroMeteorAndes(Datasets):
    """Hydrometeriological dataset of tropical Andes region"""
    url = ["https://springernature.figshare.com/ndownloader/files/10514506",
           "https://springernature.figshare.com/ndownloader/files/10514509"]


class WeatherJena(Datasets):
    """
    10 minute weather dataset of Jena, Germany hosted at https://www.bgc-jena.mpg.de/wetter/index.html
    from 2002 onwards.

        Examples
        --------
        >>> from aqua_fetch import WeatherJena
        >>> dataset = WeatherJena()
        >>> data = dataset.fetch()
        >>> data.sum()
    """
    url = "https://www.bgc-jena.mpg.de/wetter/weather_data.html"

    STARTS = {
        'roof': 2004,
        'saale': 2003,
        'soil': 2007
    }

    PREFIX = {
        'roof': 'mpi_roof',
        'saale': 'mpi_saale',
        'soil': 'MPI_Soil'
    }

    def __init__(self,
                 path=None,
                 obs_loc='roof'):
        """
        The ETP data is collected at three different locations i.e. roof, soil and saale(hall).

        Parameters
        ----------
            obs_loc : str, optional (default=roof)
                location of observation. It can be one of following
                    - roof
                    - soil
                    - saale
        """

        if obs_loc not in ['roof', 'soil', 'saale']:
            raise ValueError
        self.obs_loc = obs_loc

        super().__init__(path=path)

        sub_dir = os.path.join(self.path, self.obs_loc)

        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)

        self._download(sub_dir)

        if xr is None:
            warnings.warn("""
            loading data from csv files is slow. 
            Try installing xarray and netcdf for faster loading
            """)
            #download_all_http_directory(self.url, sub_dir, match_name=self.obs_loc)
            unzip_all_in_dir(sub_dir, 'zip')
        else:
            nc_path = os.path.join(sub_dir, "data.nc")
            if not os.path.exists(nc_path):
                #download_all_http_directory(self.url, sub_dir, match_name=self.obs_loc)
                unzip_all_in_dir(sub_dir, 'zip')
                print("converting data to netcdf file. This will happen only once.")
                df = self._read_as_df()
                ndf = pd.DataFrame()
                for _col in df.columns:
                    col = _col.replace("/", "_")
                    ndf[col] = df[_col].copy()

                ndf.replace('********', np.nan, inplace=True)
                ndf['Rn (W_m**2)'] = ndf['Rn (W_m**2)'].astype(np.float32)
                ndf.to_xarray().to_netcdf(nc_path)

    def _download(self, path):
        """downloads the dataset"""
        if os.path.exists(path) and len(os.listdir(path)) > 0:
            return
    
        for year in range(self.STARTS[self.obs_loc], 2024):
            for period in ['a', 'b']:
                url = f"https://www.bgc-jena.mpg.de/wetter/{self.PREFIX[self.obs_loc]}_{year}{period}.zip"
            
                download_and_unzip(path, 
                               url=url, name=f"mpi_{self.obs_loc}_{year}{period}.zip", verbosity=self.verbosity)
        return

    @property
    def dynamic_features(self)->List[str]:
        """returns names of features available"""
        return self.fetch().columns.tolist()

    def fetch(
            self,
            st: Union[str, int, pd.DatetimeIndex] = None,
            en: Union[str, int, pd.DatetimeIndex] = None
    ) -> pd.DataFrame:
        """
        Fetches the time series data between given period as :obj:`pandas.DataFrame`.

        Parameters
        ----------
            st : Optional
                start of data to be fetched. If None, the data from start (2003-01-01)
                will be retuned
            en : Optional
                end of data to be fetched. If None, the data from till (2021-12-31)
                end be retuned.

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of shape (972111, 21)

        Examples
        --------
            >>> from aqua_fetch import WeatherJena
            >>> dataset = WeatherJena()
            >>> data = dataset.fetch()
            >>> data.shape
            (972111, 21)
            ... # get data between specific period
            >>> data = dataset.fetch("20110101", "20201231")
            >>> data.shape
            (525622, 21)
        """

        sub_dir = os.path.join(self.path, self.obs_loc)

        if xr is None:
            df = self._read_as_df()
        else:
            nc_path = os.path.join(sub_dir, "data.nc")
            df = xr.load_dataset(nc_path).to_dataframe()
            if 'Date Time' in df:
                df.index = pd.to_datetime(df.pop('Date Time'))

        if isinstance(st, int):
            if en is None:
                en = len(df)
            assert isinstance(en, int)
            return df.iloc[st:en]
        elif st is not None:
            return df.loc[st:en]

        return df

    def _read_as_df(self)->pd.DataFrame:

        sub_dir = os.path.join(self.path, self.obs_loc)
        all_files = glob.glob(f"{sub_dir}/*.csv")

        dfs = []
        for fpath in all_files:
            f_df = pd.read_csv(fpath, index_col='Date Time',
                               encoding='unicode_escape', na_values=-9999)
            f_df.index = pd.to_datetime(f_df.index, format='%d.%m.%Y %H:%M:%S')
            dfs.append(f_df)  

        df = pd.concat(dfs)
        return df.sort_index()


class SWECanada(Datasets):
    """
    Daily Canadian historical Snow Water Equivalent dataset from 1928 to 2020
    from Brown_ et al., 2019 .

    Examples
    --------
        >>> from aqua_fetch import SWECanada
        >>> swe = SWECanada()
        ... # get names of all available stations
        >>> stns = swe.stations()
        >>> len(stns)
        2607
        ... # get data of one station
        >>> df1 = swe.fetch('SCD-NS010')
        >>> df1['SCD-NS010'].shape
        (33816, 3)
        ... # get data of 10 stations
        >>> df5 = swe.fetch(5, st='20110101')
        >>> df5.keys()
        ['YT-10AA-SC01', 'ALE-05CA805', 'SCD-NF078', 'SCD-NF086', 'INA-07RA01B']
        >>> [v.shape for v in df5.values()]
        [(3500, 3), (3500, 3), (3500, 3), (3500, 3), (3500, 3)]
        ... # get data of 0.1% of stations
        >>> df2 = swe.fetch(0.001, st='20110101')
        ... # get data of one stations starting from 2011
        >>> df3 = swe.fetch('ALE-05AE810', st='20110101')
        >>> df3.keys()
        >>> ['ALE-05AE810']
        >>> df4 = swe.fetch(stns[0:10], st='20110101')

    .. _Brown:
        https://doi.org/10.1080/07055900.2019.1598843

    """
    url = "https://zenodo.org/records/10835278"
    features = ['snw', 'snd', 'den']
    q_flags = ['data_flag_snw', 'data_flag_snd', 'qc_flag_snw', 'qc_flag_snd']

    def __init__(self, path=None, **kwargs):
        super().__init__(path=path, **kwargs)
        #self.path = path

        self._download()

    def stations(self) -> List[str]:
        nc = netCDF4.Dataset(os.path.join(self.path, 'CanSWE-CanEEN_1928-2023_v6.nc'))
        s = nc['station_id'][:]
        return s.tolist()

    @property
    def start(self):
        return '19280101'

    @property
    def end(self):
        return '20230731'

    def fetch(
            self,
            stations: Union[None, str, float, int, list] = None,
            features: Union[None, str, list] = None,
            q_flags: Union[None, str, list] = None,
            st=None,
            en=None
    ) -> dict:
        """
        Fetches time series data from selected stations.

        Parameters
        ----------
            stations :
                station/stations to be retrieved. In None, then data
                from all stations will be returned.
            features :
                Names of features to be retrieved. Following features
                are allowed:

                    - ``snw`` snow water equivalent kg/m3
                    - ``snd`` snow depth m
                    - ``den`` snowpack bulk density kg/m3

                If None, then all three features will be retrieved.
            q_flags :
                If None, then no qflags will be returned. Following q_flag
                values are available.
                    - ``data_flag_snw``
                    - ``data_flag_snd``
                    - ``qc_flag_snw``
                    - ``qc_flag_snd``
            st :
                start of data to be retrieved
            en :
                end of data to be retrived.

        Returns
        -------
        dict
            a dictionary of dataframes of shape (st:en, features + q_flags) whose
            length is equal to length of stations being considered.
        """
        # todo, q_flags not working

        if stations is None:
            stations = self.stations()
        elif isinstance(stations, str):
            stations = [stations]
        elif isinstance(stations, list):
            pass
        elif isinstance(stations, int):
            stations = random.sample(self.stations(), stations)
        elif isinstance(stations, float):
            num_stations = int(len(self.stations()) * stations)
            stations = random.sample(self.stations(), num_stations)

        stns = self.stations()
        stn_id_dict = {k: v for k, v in zip(stns, np.arange(len(stns)))}
        stn_id_dict_inv = {v: k for k, v in stn_id_dict.items()}
        stn_ids = [stn_id_dict[i] for i in stations]

        features = check_attributes(features, self.features)
        qflags = []
        if q_flags is not None:
            qflags = check_attributes(q_flags, self.q_flags)

        features_to_fetch = features + qflags

        all_stn_data = {}
        for stn in stn_ids:

            stn_df = self.fetch_station_attributes(stn, features_to_fetch, st=st, en=en)
            all_stn_data[stn_id_dict_inv[stn]] = stn_df

        return all_stn_data

    def fetch_station_attributes(self,
                                 stn,
                                 features_to_fetch,
                                 st=None,
                                 en=None,
                                 ) -> pd.DataFrame:
        """fetches attributes of one station"""

        # st, en = self._check_length(st, en)

        nc = netCDF4.Dataset(os.path.join(self.path, 'CanSWE-CanEEN_1928-2023_v6.nc'))

        stn_df = pd.DataFrame(columns=features_to_fetch)

        for var in nc.variables:
            if var in features_to_fetch:
                ma = np.array(nc[var][:])
                ma[ma == nc[var]._FillValue] = np.nan
                ta = ma[stn, :]  # target array of on station
                s = pd.Series(ta, index=pd.date_range(self.start, self.end, freq='D'), name=var)
                stn_df[var] = s[st:en]

        nc.close()

        return stn_df


class RRAlpineCatchments(Datasets):
    """
    Modelled runoff in contrasting Alpine catchments in Austria from 1981 to 2100
    using 14 models follwoing the work of Hanus et al., 2021 [12]_ .
    past 1981 - 2010
    future

    .. [12] https://hess.copernicus.org/preprints/hess-2021-92/
    """
    url = "https://zenodo.org/record/4539986"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._download()


class ETPAgroForestGermany(Datasets):
    """
    Evapotranspiration over agroforestry sites in Germany
    https://doi.org/10.5194/bg-17-5183-2020
    SiteName_Landuse_Content_Figures_Tables.csv
    """
    url = "https://zenodo.org/record/4038399"


class ETPTelesinaItaly(Datasets):
    """
    Daily rain and reference evapotranspiration for three years 2002-2004
    """
    url = "https://zenodo.org/record/3726856"


def gw_punjab(
        data_type:str = "full",
        country:str = None,
)->pd.DataFrame:
    """
    groundwater level (meters below ground level) dataset from Punjab region
    (Pakistan and north-west India) following the study of `MacAllister et al., 2022 <https://doi.org/10.1038/s41561-022-00926-1>`_.

    parameters
    ----------
    data_type : str (default="full")
        either ``full`` or ``LTS``. The ``full`` contains the
        full dataset, there are 68783 rows of observed groundwater level data from
        4028 individual sites. In ``LTS`` there are 7547 rows of groundwater
        level observations from 130 individual sites, which have water level data available
        for a period of more than 40 years and from which at least two thirds of the
        annual observations are available.
    country : str (default=None)
        the country for which data to retrieve. Either ``PAK`` or ``IND``.

    Returns
    -------
    pd.DataFrame
        a :obj:`pandas.DataFrame` with datetime index

    Examples
    ---------
    >>> from aqua_fetch import gw_punjab
    >>> full_data = gw_punjab()
    find out the earliest observation
    >>> print(full_data.sort_index().head(1))
    >>> lts_data = gw_punjab()
    >>> lts_data.shape
        (68782, 4)
    >>> df_pak = gw_punjab(country="PAK")
    >>> df_pak.sort_index().dropna().head(1)

    """
    f = 'https://webservices.bgs.ac.uk/accessions/download/167240?fileName=India_Pakistan_WL_NGDC.xlsx'

    ds_dir =os.path.join(os.path.dirname(__file__), "data", 'gw_punjab')
    if not os.path.exists(ds_dir):
        os.makedirs(ds_dir)

    fpath = os.path.join(ds_dir, "gw_punjab.xlsx")
    if not os.path.exists(fpath):
        print(f"downloading {fpath}")
        download(f, os.path.dirname(fpath), fname="gw_punjab.xlsx")

    assert data_type in ("full", "LTS")

    if data_type == "full":
        sheet_name = "Full_dataset"
    else:
        sheet_name = "LTS"

    df = pd.read_excel(fpath, sheet_name=sheet_name)

    if sheet_name == "LTS":
        df.iloc[5571, 3] = '01/10/1887'
        df.iloc[5572, 3] = '01/10/1892'
        df.iloc[6227, 3] = '01/10/1887'
        df.iloc[5511, 3] = '01/10/1887'
        df.iloc[5512, 3] = '01/10/1892'
        df.iloc[6228, 3] = '01/10/1892'

    df.index = pd.to_datetime(df.pop("DATE"))

    if country:
        if country == "PAK":
            pak_stations = [st for st in df['OW_ID'].unique() if st.startswith("PAK")]
            df = df[df['OW_ID'].isin(pak_stations)]
        else:
            pak_stations = [st for st in df['OW_ID'].unique() if st.startswith("IND")]
            df = df[df['OW_ID'].isin(pak_stations)]

    return df
