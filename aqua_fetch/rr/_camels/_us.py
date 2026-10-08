import os
import glob
from typing import Union, List, Dict, Tuple

import pandas as pd

from ..utils import _RainfallRunoff
from ...utils import download, unzip
from .._map import (
    observed_streamflow_cms,
    max_air_temp,
    min_air_temp,
    total_precipitation,
    total_potential_evapotranspiration,
    mean_vapor_pressure,
    solar_radiation,
    daylight_solar_radiation,
    snow_water_equivalent,
)
from .._map import catchment_area, gauge_latitude, gauge_longitude, slope

# directory separator
SEP = os.sep


class CAMELS_US(_RainfallRunoff):
    """
    This is a dataset of 671 US catchments with 59 static catchment features
    and 9 catchment averaged dynamic features for each catchment. The dynamic features are
    daily timeseries from 1980-01-01 to 2014-12-31. The data is downloaded
    from its `zenodo repository <https://zenodo.org/records/15529996>`_ . For more details
    on data refer to `Newman et al., 2015 <https://doi.org/10.5194/hess-19-209-2015>`_ ,
    `Newman et al., 2022 <https://gdex.ucar.edu/dataset/camels.html.>`_ and
    `Addor et al., 2017 <https://hess.copernicus.org/articles/21/5293/2017/>`_.

    Please note this data is also known as "CAMELS" however, we have named it CAMELS_US
    to differentiate it from other CAMELS like datasts from other parts of the world.

    Examples
    --------
    >>> from aqua_fetch import CAMELS_US
    >>> dataset = CAMELS_US()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='11478500', as_dataframe=True)
    >>> df = dynamic['11478500'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (12784, 9)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       671
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (67 out of 671)
       67
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(12784, 9), (12784, 9), (12784, 9),... (12784, 9), (12784, 9)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('11478500', as_dataframe=True,
    ...  dynamic_features=['pcp_mm', 'swdownrad_wm2', 'airtemp_C_max', 'airtemp_C_min', 'q_cms_obs'])
    >>> dynamic['11478500'].shape
       (12784, 5)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='11478500', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['11478500'].shape
    ((1, 59), 1, (12784, 9))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 12784, 'dynamic_features': 9})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (671, 2)
    >>> dataset.stn_coords('11478500')  # returns coordinates of station whose id is 11478500
        40.480419	-123.890877
    >>> dataset.stn_coords(['11478500', '14020000'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('11478500')
    # get coordinates of two stations
    >>> dataset.area(['11478500', '14020000'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('11478500')

    """
    DATASETS = ['CAMELS_US']

    url = {
        'camels_attributes_v2.0.pdf': 'https://zenodo.org/records/15529996/files/',
        'camels_attributes_v2.0.xlsx': 'https://zenodo.org/records/15529996/files/',
        'camels_clim.txt': 'https://zenodo.org/records/15529996/files/',
        'camels_geol.txt': 'https://zenodo.org/records/15529996/files/',
        'camels_hydro.txt': 'https://zenodo.org/records/15529996/files/',
        'camels_name.txt': 'https://zenodo.org/records/15529996/files/',
        'camels_soil.txt': 'https://zenodo.org/records/15529996/files/',
        'camels_topo.txt': 'https://zenodo.org/records/15529996/files/',
        'camels_vege.txt': 'https://zenodo.org/records/15529996/files/',
        'readme.txt': 'https://zenodo.org/records/15529996/files/',
        'basin_timeseries_v1p2_metForcing_obsFlow.zip': 'https://zenodo.org/records/15529996/files/',
        'basin_set_full_res.zip': 'https://zenodo.org/records/15529996/files/',
    }

    folders = {'daymet': f'basin_mean_forcing{SEP}daymet',
               'maurer': f'basin_mean_forcing{SEP}maurer',
               'nldas': f'basin_mean_forcing{SEP}nldas',
               'v1p15_daymet': f'basin_mean_forcing{SEP}v1p15{SEP}daymet',
               'v1p15_nldas': f'basin_mean_forcing{SEP}v1p15{SEP}nldas',
               'elev_bands': f'elev{SEP}daymet',
               'hru': f'hru_forcing{SEP}daymet'}

    dynamic_features_ = ['dayl(s)', 'prcp(mm/day)', 'srad(W/m2)',
                         'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)', 'Flow']

    def __init__(
            self,
            path:Union[str, os.PathLike]=None,
            data_source: str = 'daymet',
            **kwargs
    ):

        """
        parameters
        ----------
        path : str
            If the data is alredy downloaded then provide the complete
            path to it. If None, then the data will be downloaded.
            The data is downloaded once and therefore susbsequent
            calls to this class will not download the data unless
            ``overwrite`` is set to True.
        data_source : str
            source of meteorological timeseries data. Allowed values are
    
                - daymet
                - maurer
                - nldas
                - v1p15_daymet
                - v1p15_nldas
        """
        assert data_source in self.folders, f'allowed data sources are {self.folders.keys()}'
        self.data_source = data_source

        super().__init__(path=path, name="CAMELS_US", **kwargs)

        if not os.path.exists(self.path):
            os.makedirs(self.path)

        # An archive is satisfied (not needed) when *any* of these alternatives
        # exists on disk: the archive itself, its extracted sibling folder, or a
        # consolidated cache that fully captures its content. The third entry
        # is what allows free_disk_space("redundant") to delete the per-station
        # tree without triggering a re-download next time.
        satisfied_by_cache = {
            'basin_timeseries_v1p2_metForcing_obsFlow.zip': self.dyn_fpath,
        }

        for fname, url in self.url.items():

            fpath = os.path.join(self.path, fname)
            furl = f"{url}{fname}"

            if fname.endswith('.zip'):
                extracted = os.path.join(self.path, fname[:-len('.zip')])
                cache = satisfied_by_cache.get(fname)
                already_have = (
                    os.path.exists(fpath)
                    or os.path.exists(extracted)
                    or (cache is not None and os.path.exists(cache))
                )
            else:
                already_have = os.path.exists(fpath)

            if already_have and not self.overwrite:
                continue

            if self.verbosity:
                print(f"downloading {fname} from {url}")

            download(furl, self.path, fname, verbosity=self.verbosity)

            unzip(self.path, verbosity=self.verbosity)

        self.dataset_dir = os.path.join(
            self.path,
            f'basin_timeseries_v1p2_metForcing_obsFlow{SEP}basin_dataset_public_v1p2')

        self._static_features = self._static_data().columns.tolist()
        self._maybe_to_netcdf()

    @property
    def boundary_file(self) -> os.PathLike:
        return os.path.join(
            self.path,
            "basin_set_full_res",
            "HCDN_nhru_final_671.shp"
        )

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'area_gages2': catchment_area(),
                'slope_mean': slope('mkm-1'),
                'gauge_lat': gauge_latitude(),
                'gauge_lon': gauge_longitude(),
        }

    @property
    def dyn_map(self):
        return {
            'Flow': observed_streamflow_cms(),  # todo : check units
            'tmin(C)': min_air_temp(),
            'tmax(C)': max_air_temp(),
            'prcp(mm/day)': total_precipitation(),
            'swe(mm)': snow_water_equivalent(),
            'pet_mean': total_potential_evapotranspiration(),
            'vp(Pa)': mean_vapor_pressure(),  # todo: convert frmo Pa to hpa
            # Daymet's ``srad`` is averaged over the DAYLIGHT period, not over the
            # 24-h day, so it is not comparable with the daily-mean W m-2 that
            # every other dataset serves as ``swdownrad_wm2``. The native value is
            # kept faithfully under its own name and the 24-h mean is derived
            # from it in _read_stn_dyn(). The maurer and nldas forcings were
            # resampled into the Daymet format and share the convention
            # (verified: raw Kt 1.22 and 1.07, i.e. both impossible as 24-h
            # means). Note _read_stn_dyn currently hardcodes the daymet
            # filename, so those two sources cannot actually be read yet;
            # that is a pre-existing limitation, not one introduced here.
            'srad(W/m2)': daylight_solar_radiation(),
        }

    @property
    def dynamic_features(self) -> List[str]:
        # swdownrad_wm2 is derived (see _read_stn_dyn) so it has no raw counterpart
        # in dynamic_features_
        return [self.dyn_map.get(feat, feat) for feat in self.dynamic_features_] + [solar_radiation()]

    @property
    def dyn_factors(self) -> Dict[str, float]:
        return {
            observed_streamflow_cms(): 0.0283168,
        }

    @property
    def start(self):
        return pd.Timestamp("19800101")

    @property
    def end(self):
        return pd.Timestamp("20141231")

    @property
    def static_features(self)->List[str]:
        return self._static_features

    def stations(self) -> list:
        streamflow_dir = os.path.join(self.dataset_dir, 'usgs_streamflow')
        if os.path.exists(streamflow_dir):
            stns = []
            for _dir in os.listdir(streamflow_dir):
                cat = os.path.join(streamflow_dir, _dir)
                stns += [fname.split('_')[0] for fname in os.listdir(cat)]
        else:
            # Fallback when the per-station tree has been removed by
            # free_disk_space("redundant"): read station ids from the cached
            # static_features.csv, which contains them as the gauge_id index.
            static_fpath = os.path.join(self.path, 'static_features.csv')
            df = pd.read_csv(static_fpath, dtype={'gauge_id': str}, usecols=['gauge_id'])
            stns = df['gauge_id'].astype(str).tolist()

        # remove stations for which static values are not available
        for stn in ['06775500', '06846500', '09535100']:
            if stn in stns:
                stns.remove(stn)

        return stns

    def _redundant_after_consolidation(self) -> List[Tuple[str, str]]:
        """
        The per-station forcing and streamflow text files inside
        ``basin_timeseries_v1p2_metForcing_obsFlow/`` become redundant once
        ``camels_us_D_v2.nc`` exists, since all dynamic data is baked into that
        consolidated NetCDF.

        Caveat: the cache reflects whichever ``data_source`` was selected when
        it was first built. Once ``free_disk_space("redundant")`` removes the
        per-station tree, switching ``data_source`` (e.g. ``daymet`` ?
        ``nldas``) requires re-downloading the source archive to rebuild the
        cache. Run cleanup only after settling on a data source.
        """
        return [
            (os.path.join(self.path, 'basin_timeseries_v1p2_metForcing_obsFlow'),
             self.dyn_fpath),
        ]

    def _read_stn_dyn(
            self,
            stn:str,
    ):

        assert isinstance(stn, str)
        df = None
        dir_name = self.folders[self.data_source]
        for cat in os.listdir(os.path.join(self.dataset_dir, dir_name)):
            cat_dirs = os.listdir(os.path.join(self.dataset_dir, f'{dir_name}{SEP}{cat}'))
            stn_file = f'{stn}_lump_cida_forcing_leap.txt'
            if stn_file in cat_dirs:
                df = pd.read_csv(os.path.join(self.dataset_dir,
                                                f'{dir_name}{SEP}{cat}{SEP}{stn_file}'),
                                    sep="\s+|;|:",
                                    skiprows=4,
                                    engine='python',
                                    names=['Year', 'Mnth', 'Day', 'Hr', 'dayl(s)', 'prcp(mm/day)', 'srad(W/m2)',
                                        'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)'],
                                    )
                df.index = pd.to_datetime(
                    df['Year'].map(str) + '-' + df['Mnth'].map(str) + '-' + df['Day'].map(str))

        flow_dir = os.path.join(self.dataset_dir, 'usgs_streamflow')
        for cat in os.listdir(flow_dir):
            cat_dirs = os.listdir(os.path.join(flow_dir, cat))
            stn_file = f'{stn}_streamflow_qc.txt'
            if stn_file in cat_dirs:
                fpath = os.path.join(flow_dir, f'{cat}{SEP}{stn_file}')
                q_df = pd.read_csv(fpath,
                                    sep=r"\s+",
                                    names=['station', 'Year', 'Month', 'Day', 'Flow', 'Flag'],
                                    engine='python')
                q_df.index = pd.to_datetime(
                    q_df['Year'].map(str) + '-' + q_df['Month'].map(str) + '-' + q_df['Day'].map(str))

        stn_df = pd.concat([
            df[['dayl(s)', 'prcp(mm/day)', 'srad(W/m2)', 'swe(mm)', 'tmax(C)', 'tmin(C)', 'vp(Pa)']],
            q_df['Flow']],
            axis=1)

        stn_df.rename(columns=self.dyn_map, inplace=True)

        self._apply_dyn_factors(stn_df)

        # Daymet reports srad as the mean over the daylight hours only. Weighting
        # by the daylight fraction of the day gives the 24-h mean that the
        # canonical ``swdownrad_wm2`` promises and that every other dataset serves.
        stn_df[solar_radiation()] = (
            stn_df[daylight_solar_radiation()] * stn_df['dayl(s)'] / 86400.0
        )

        return stn_df

    def _static_data(self)->pd.DataFrame:
        static_fpath = os.path.join(self.path, 'static_features.csv')
        if not os.path.exists(static_fpath):
            files = glob.glob(f"{self.path}/*.txt")

            static_dfs = []
            for f in files:
                if not f.endswith('readme.txt'):
                    if self.verbosity>2:
                        print(f"reading {f}")
                    # index should be read as string

                    _df = pd.read_csv(f, sep=';', index_col='gauge_id', dtype={'gauge_id': str})

                    static_dfs.append(_df)
            static_df = pd.concat(static_dfs, axis=1)
            static_df.to_csv(static_fpath, index_label='gauge_id')
        else:  # index should be read as string bcs it has 0s at the start
            static_df = pd.read_csv(static_fpath, index_col='gauge_id', dtype={'gauge_id': str})

        static_df.index = static_df.index.astype(str)

        static_df = static_df#.loc[stn_id][features]
        if isinstance(static_df, pd.Series):
            static_df = pd.DataFrame(static_df).transpose()

        static_df.rename(columns=self.static_map, inplace=True)
        
        return static_df
