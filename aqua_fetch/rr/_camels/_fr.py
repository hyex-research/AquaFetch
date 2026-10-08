import os
import glob
import zlib
import shutil
import zipfile
import warnings
from typing import List, Dict

import pandas as pd

from ..utils import _RainfallRunoff
from ...utils import download
from .._map import (
    observed_streamflow_cms,
    observed_streamflow_mm,
    mean_air_temp,
    max_air_temp,
    min_air_temp,
    total_precipitation,
    total_precipitation_with_specifier,
    total_potential_evapotranspiration_with_specifier,
    mean_windspeed,
    solar_radiation,
    downward_longwave_radiation,
    J_CM2_DAY_TO_WM2,
    mean_specific_humidity,
)
from .._map import catchment_area, gauge_latitude, gauge_longitude, slope
from ._common import _remove_stale


class CAMELS_FR(_RainfallRunoff):
    """
    Dataset of 654 catchments from France following the works of
    `Delaigue et al., 2025 <https://doi.org/10.5194/essd-17-1461-2025>`_.
    The dataset consists of 344 static catchment features and 22 dynamic features.
    The dynamic features span from 1970101 to 20211231 with daily timestep.

    This is release 3.2 of the `Recherche Data Gouv record
    <https://doi.org/10.57745/WH7FJR>`_. All releases hold the same 654 stations,
    22 dynamic features, 344 static features and time span; release 3.0 corrected
    ``hym_q_questionable``, ``hym_q_unqualified`` and ``hym_q_anomaly_inrae``
    (percentages of streamflow values flagged as doubtful), which is the only
    difference in the data. Data downloaded by an earlier version of aqua_fetch is
    release 2.1; it is detected at initialization, and only its 9.4 MB attributes
    archive is downloaded again.

    Not provided: the monthly and yearly aggregates of the time series archive.

    Timings on a 48-core machine: the first initialization downloads 372 MB and
    builds a 2.2 GB netCDF cache, for which it reads the 654 daily csv files in
    18 s. An initialization that only upgrades release 2.1 takes 5 s. Afterwards
    initialization takes 0.5 s, fetching all 654 stations with all 22 dynamic
    features 0.15 s and with all 344 static features 0.45 s.

    Examples
    ---------
    >>> from aqua_fetch import CAMELS_FR
    >>> dataset = CAMELS_FR()
    ... # get data by station id
    >>> _, dynamic = dataset.fetch(stations='J421191001', as_dataframe=True)
    >>> df = dynamic['J421191001'] # dynamic is a dictionary of with keys as station names and values as DataFrames
    >>> df.shape
    (18993, 22)
    ...
    ... # get name of all stations as list
    >>> stns = dataset.stations()
    >>> len(stns)
       654
    ... # get data of 10 % of stations as dataframe
    >>> _, dynamic = dataset.fetch(0.1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 10% of stations (65 out of 654)
       65
    ...
    ... # dynamic is a dictionary whose values are dataframes of dynamic features
    >>> [df.shape for df in dynamic.values()]
        [(18993, 22), (18993, 22), (18993, 22),... (18993, 22), (18993, 22)]
    ...
    ... get the data of a single (randomly selected) station
    >>> _, dynamic = dataset.fetch(stations=1, as_dataframe=True)
    >>> len(dynamic)  # dynamic has data for 1 station
        1
    ... # get names of available dynamic features
    >>> dataset.dynamic_features
    ... # get only selected dynamic features
    >>> _, dynamic = dataset.fetch('J421191001', as_dataframe=True,
    ...  dynamic_features=['pcp_mm', 'spechum_gkg', 'airtemp_C_mean', 'pet_mm_pm', 'q_cms_obs'])
    >>> dynamic['J421191001'].shape
       (18993, 5)
    ...
    ... # get names of available static features
    >>> dataset.static_features
    ... # get data of 10 random stations
    >>> _, dynamic = dataset.fetch(10, as_dataframe=True)
    >>> len(dynamic)  # remember this is a dictionary with values as dataframe
       10
    ...
    # If we get both static and dynamic data
    >>> static, dynamic = dataset.fetch(stations='J421191001', static_features="all", as_dataframe=True)
    >>> static.shape, len(dynamic), dynamic['J421191001'].shape
    ((1, 344), 1, (18993, 22))
    ...
    # If we don't set as_dataframe=True and have xarray installed then the returned data will be a xarray Dataset
    >>> _, dynamic = dataset.fetch(10)
    ... type(dynamic)   
    xarray.core.dataset.Dataset
    ...
    >>> dynamic.dims
    FrozenMappingWarningOnValuesAccess({'time': 18993, 'dynamic_features': 22})
    ...
    >>> len(dynamic.data_vars)
    10
    ...
    >>> coords = dataset.stn_coords() # returns coordinates of all stations
    >>> coords.shape
        (654, 2)
    >>> dataset.stn_coords('J421191001')  # returns coordinates of station whose id is J421191001
        48.006298   -4.063848
    >>> dataset.stn_coords(['J421191001', '802'])  # returns coordinates of two stations
    ...
    # get area of a single station
    >>> dataset.area('J421191001')
    # get coordinates of two stations
    >>> dataset.area(['J421191001', '802'])
    ...
    # if fiona library is installed we can get the boundary as fiona Geometry
    >>> dataset.get_boundary('J421191001')
    """
    # file ids of the Recherche Data Gouv record doi:10.57745/WH7FJR, release 3.2.
    # Only the attributes archive and the README differ from release 2.1; the time
    # series, geography, licenses and description files are the same files.
    _DATAFILE = "https://entrepot.recherche.data.gouv.fr/api/access/datafile/"
    url = {
        "ADDITIONAL_LICENSES.zip": f"{_DATAFILE}343463",
        "CAMELS_FR_attributes.zip": f"{_DATAFILE}621683",
        'CAMELS_FR_geography.zip': f"{_DATAFILE}343465",
        'CAMELS_FR_time_series.zip': f"{_DATAFILE}343470",
        'README.md': f"{_DATAFILE}621685",
        'NEWS.md': f"{_DATAFILE}621689",
        'CAMELS-FR_description.ods': f"{_DATAFILE}348740",
    }

    # files of CAMELS_FR_attributes.zip and CAMELS_FR_geography.zip, from the
    # "File hierarchy convention" section of the README of the dataset. Used by
    # _check_manifest, so that an incomplete extraction is reported instead of
    # being taken for a smaller dataset.
    _STATIC_ATTR_FILES = (
        "00_description_geology_classes.txt",
        "00_description_land_cover_classes.txt",
        "CAMELS_FR_geology_attributes.csv",
        "CAMELS_FR_human_influences_dams.csv",
        "CAMELS_FR_hydrogeology_attributes.csv",
        "CAMELS_FR_land_cover_attributes.csv",
        "CAMELS_FR_site_general_attributes.csv",
        "CAMELS_FR_soil_general_attributes.csv",
        "CAMELS_FR_soil_quantiles_attributes.csv",
        "CAMELS_FR_station_general_attributes.csv",
        "CAMELS_FR_topography_general_attributes.csv",
        "CAMELS_FR_topography_quantiles_attributes.csv",
    )
    _TS_STAT_FILES = (
        "CAMELS_FR_climatic_statistics.csv",
        "CAMELS_FR_hydroclimatic_quantiles.csv",
        "CAMELS_FR_hydroclimatic_regimes_daily.csv",
        "CAMELS_FR_hydroclimatic_statistics_joint_availability_yearly.csv",
        "CAMELS_FR_hydroclimatic_statistics_timeseries_yearly.csv",
        "CAMELS_FR_hydrological_signatures.csv",
        "CAMELS_FR_hydrometry_statistics.csv",
    )
    _GEOG_FILES = (
        "CAMELS_FR_catchment_boundaries.gpkg",
        "CAMELS_FR_catchment_nestedness_information.csv",
        "CAMELS_FR_gauge_outlet.gpkg",
    )

    # size in bytes of CAMELS_FR_hydrometry_statistics.csv in release 2.1, which
    # aqua_fetch downloaded until now. Release 3.0 recomputed hym_q_questionable,
    # hym_q_unqualified and hym_q_anomaly_inrae, making this the only file of the
    # attributes archive whose content, and size, changed.
    _V21_HYDROMETRY_BYTES = 38976

    def __init__(self,
                 path=None,
                 overwrite=False,
                 **kwargs):
        """
        Parameters
        ----------
        path : str
            directory under which the data is (or will be) saved in a
            ``CAMELS_FR`` folder. If None, the default data directory of
            aqua_fetch is used.
        overwrite : bool
            if True, the archives, the extracted folders and the netCDF caches
            are deleted and downloaded/built again.
        **kwargs :
            any keyword argument of :py:class:`aqua_fetch.rr._RainfallRunoff`
            such as ``processes``, ``verbosity``, ``to_netcdf`` or ``remove_zip``.
        """
        super().__init__(path=path, overwrite=overwrite, **kwargs)

        self._download_camels_fr(overwrite=overwrite)

        self._stations = self.__stations()

        self._check_manifest()

        self._static_features = list(set(self._static_data().columns.to_list()))

        self._dynamic_features = self._read_stn_dyn(self.stations()[0]).columns.to_list()

        # if self.to_netcdf:
        self._maybe_to_netcdf()

    def _download_camels_fr(self, overwrite: bool = False):
        """
        Downloads and extracts only those archives whose extracted folder does
        not exist, so that an archive deleted after extraction
        (``remove_zip=True``) is not downloaded again, and downloads the plain
        files that are missing.

        When the attributes on disk are those of the superseded release 2.1 (see
        :meth:`_stale_attributes`), the 9.4 MB attributes archive and the README
        are downloaded again so that an existing installation is brought to
        release 3.2. Nothing else is touched, because the 361 MB time series
        archive, the geography archive and the netCDF cache, which holds dynamic
        data only, are the same in both releases.

        ``overwrite=True`` first deletes this dataset's archives, extracted
        folders, plain files and netCDF caches.
        """
        os.makedirs(self.path, exist_ok=True)

        archives = {fname: link for fname, link in self.url.items() if fname.endswith('.zip')}
        plain = {fname: link for fname, link in self.url.items() if not fname.endswith('.zip')}
        # each archive holds a folder of its own name, so CAMELS_FR_attributes.zip
        # is extracted to path/CAMELS_FR_attributes/CAMELS_FR_attributes/
        folder_of = {fname: os.path.join(self.path, fname[:-len('.zip')]) for fname in archives}

        if overwrite:
            caches = glob.glob(os.path.join(glob.escape(self.path),
                                            f"{self.name.lower()}_{self.timestep}*.nc"))
            _remove_stale([*(os.path.join(self.path, fname) for fname in self.url),
                           *folder_of.values(), *caches], self.verbosity)
        elif self._stale_attributes():
            # unconditional, because this silently changes the values that
            # static_features returns between two runs of the same code
            warnings.warn(
                f"The CAMELS-FR attributes in {self.path} are those of release 2.1, "
                "in which hym_q_questionable, hym_q_unqualified and "
                "hym_q_anomaly_inrae were miscalculated. Downloading the 9.4 MB "
                "attributes archive of release 3.2 to replace them; the time "
                "series, the boundaries and the netCDF cache are unaffected.",
                UserWarning)
            _remove_stale([folder_of['CAMELS_FR_attributes.zip'],
                           os.path.join(self.path, 'CAMELS_FR_attributes.zip'),
                           os.path.join(self.path, 'README.md')],
                          self.verbosity, reason="superseded release 2.1")

        for fname, link in archives.items():
            folder = folder_of[fname]
            if os.path.exists(folder):
                continue

            archive = os.path.join(self.path, fname)
            if not os.path.exists(archive):
                if self.verbosity:
                    print(f"downloading {link} to {archive}")
                download(link, outdir=self.path, fname=fname, verbosity=self.verbosity)

            # extracted into a temporary folder that is renamed once complete, so
            # that an interrupted extraction is redone instead of being taken as
            # complete at the next initialization
            if self.verbosity:
                print(f"extracting {archive}")
            partial = f"{folder}_extracting"
            shutil.rmtree(partial, ignore_errors=True)
            try:
                with zipfile.ZipFile(archive) as zf:
                    zf.extractall(partial)
            except (zipfile.BadZipFile, zlib.error, EOFError):  # e.g. an error page saved as the archive
                shutil.rmtree(partial, ignore_errors=True)
                os.remove(archive)
                raise ValueError(f"{archive} is corrupt and was deleted. "
                                 f"Initialize CAMELS_FR again to download it again.") from None
            os.replace(partial, folder)

        for fname, link in plain.items():
            fpath = os.path.join(self.path, fname)
            if not os.path.exists(fpath):
                if self.verbosity:
                    print(f"downloading {link} to {fpath}")
                download(link, outdir=self.path, fname=fname, verbosity=self.verbosity)

        if self.remove_zip:
            for fname in archives:
                archive = os.path.join(self.path, fname)
                if os.path.exists(archive):
                    if self.verbosity:
                        print(f"remove_zip=True: removing {archive}")
                    os.remove(archive)
        return

    @property
    def _hydrometry_file(self) -> os.PathLike:
        """the only file whose content differs between releases 2.1 and 3.2"""
        return os.path.join(self.ts_stat_path, "CAMELS_FR_hydrometry_statistics.csv")

    def _stale_attributes(self) -> bool:
        """
        Whether the extracted attributes are those of release 2.1.

        The two releases differ in one file only, whose size is 38976 bytes in
        2.1 and 39467 bytes in 3.2, so one ``stat`` call tells them apart without
        reading the file or contacting the server.
        """
        fpath = self._hydrometry_file
        return os.path.exists(fpath) and os.path.getsize(fpath) == self._V21_HYDROMETRY_BYTES

    def _check_manifest(self):
        """
        warns if files that the README of the dataset lists are missing, e.g.
        left out by an interrupted extraction or deleted by hand
        """
        files = [os.path.join(self.static_attr_path, fname) for fname in self._STATIC_ATTR_FILES]
        files += [os.path.join(self.ts_stat_path, fname) for fname in self._TS_STAT_FILES]
        files += [os.path.join(self.geog_path, fname) for fname in self._GEOG_FILES]
        missing = [fpath for fpath in files if not os.path.exists(fpath)]

        n_daily = len(glob.glob(os.path.join(glob.escape(self.daily_ts_path),
                                             "CAMELS_FR_tsd_*.csv")))
        if n_daily != len(self._stations):
            missing.append(f"{len(self._stations) - n_daily} of the {len(self._stations)} "
                           f"daily time series files in {self.daily_ts_path}")

        if missing:
            warnings.warn(
                f"CAMELS_FR: {len(missing)} expected files are missing: {missing}. "
                f"Use overwrite=True to download them again.", UserWarning)
        return

    @property
    def boundary_file(self) -> os.PathLike:        
        return os.path.join(
            self.path, 
            'CAMELS_FR_geography', 
            'CAMELS_FR_geography', 
            'CAMELS_FR_catchment_boundaries.gpkg')

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'hyd_slope_fdc': slope(''),
                # 'sit_latitude', 'sit_longitude', todo : what is difference between site and guage lat/lon?
                # gauge latitude/longitude in WGS84 (Hydroportail coordinates)
                'sta_x_w84': gauge_longitude(),
                'sta_y_w84': gauge_latitude(),
                # todo: should we use sta_x_w84_snap and sta_y_w84_snap which are 
                # gauge longitude in WGS84 (INRAE's own estimation, snapped on thorical river network)
                'sit_area_topo': catchment_area(),
        }

    @property
    def dyn_map(self)->Dict[str, str]:
        return {
            # streamflow in liters per second
            'tsd_q_l': observed_streamflow_cms(),
            # streamflow in milimeters per day
            'tsd_q_mm': observed_streamflow_mm(),
            'tsd_wind': mean_windspeed(),
            'tsd_temp_min': min_air_temp(),  # minimum air temperature over the period (18h day-1, 18h day]
            'tsd_temp_max': max_air_temp(),  # maximum air temperature over the period (18h day-1, 18h day]
            'tsd_temp': mean_air_temp(),  # mean air temperature over the period (18h day-1, 18h day]
            # short wave visible radiation over the period (0h day, 0h day+1]
            'tsd_rad_ssi': solar_radiation(),  # J cm-2 day-1 -> W m-2 in dyn_factors
            # long wave atmospheric radiation over the period (0h day, 0h day+1]
            'tsd_rad_dli': downward_longwave_radiation(),  # J cm-2 day-1 -> W m-2 in dyn_factors
            # specific air humidity over the period (0h day, 0h day+1]
            'tsd_humid': mean_specific_humidity(),
            # PET over the period (0h day, 0h day+1] (Penman-Monteith method with a modified albedo when snow lies on the ground)
            'tsd_pet_pm': total_potential_evapotranspiration_with_specifier('pm'),
            'tsd_pet_pe': total_potential_evapotranspiration_with_specifier('pe'),
            'tsd_pet_ou': total_potential_evapotranspiration_with_specifier('ou'),
            # total precipitation (liquid + solid) over the period (6h day, 6h day+1]
            'tsd_prec': total_precipitation(),
            # solid fraction of precipitation over the period (6h day, 6h day+1]
            'tsd_prec_solid_frac': total_precipitation_with_specifier('solfrac'),  # todo : check its units?
        }

    @property
    def dyn_factors(self) -> Dict[str, float]:
        # ``CAMELS-FR_description.ods`` gives both radiation series in J cm-2
        # accumulated over the day, while the canonical names promise W m-2.
        return {
            solar_radiation(): J_CM2_DAY_TO_WM2,
            downward_longwave_radiation(): J_CM2_DAY_TO_WM2,
        }

    @property
    def daily_ts_path(self) -> os.PathLike:
        return os.path.join(self.path, "CAMELS_FR_time_series", "CAMELS_FR_time_series", "daily")

    @property
    def attr_path(self) -> os.PathLike:
        return os.path.join(self.path, "CAMELS_FR_attributes", "CAMELS_FR_attributes")

    @property
    def static_attr_path(self) -> os.PathLike:
        return os.path.join(self.attr_path, "static_attributes")

    @property
    def ts_stat_path(self) -> os.PathLike:
        return os.path.join(self.attr_path, "time_series_statistics")

    @property
    def static_features(self) -> List[str]:
        """returns static features for Denmark catchments"""
        return self._static_features

    @property
    def dynamic_features(self) -> List[str]:
        """returns names of dynamic features"""
        return self._dynamic_features

    @property
    def start(self) -> pd.Timestamp:  # start of data
        return pd.Timestamp('1970-01-01')

    @property
    def end(self) -> pd.Timestamp:  # end of data
        return pd.Timestamp('2021-12-31')

    def __stations(self) -> List[str]:
        return pd.read_csv(os.path.join(
            self.static_attr_path,
            "CAMELS_FR_human_influences_dams.csv"),
            sep=";",
            index_col=0).index.to_list()

    def stations(self) -> List[str]:
        return self._stations

    @property
    def geog_path(self) -> os.PathLike:
        return os.path.join(self.path, "CAMELS_FR_geography", "CAMELS_FR_geography")

    def static_attrs(self) -> pd.DataFrame:
        """
        combination of topographic + soil + landuse + geology + climate + hydro
        + climate + anthropogenic features

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of static features of all catchments of shape (654, xxxx)
        """
        files = glob.glob(f"{self.static_attr_path}/*.csv")
        dfs = []
        for f in files:
            df = pd.read_csv(f, sep=";", index_col=0)
            df.index = df.index.astype(str)
            if len(df) == 654:
                dfs.append(df)
            elif self.verbosity > 1:
                print(f"skipping {os.path.basename(f)} as it has {len(df)} rows")

        static_attrs = pd.concat(dfs, axis=1)

        gen_attrs = pd.read_csv(
            os.path.join(self.static_attr_path, "CAMELS_FR_site_general_attributes.csv"),
            sep=";",
            index_col=0,
        )

        # in gen_attrs the stn_id has lenght of 8 while in static_attrs it is 10
        # so adding the last two digits to the gen_attrs
        _map = {stn[0:-2]: stn for stn in static_attrs.index}
        gen_attrs = gen_attrs.rename(index=_map)

        if self.verbosity:
            for stn in static_attrs.index:
                if stn not in gen_attrs.index:
                    print(stn, " not found in site_general_attributes.csv")

        static_attrs = pd.concat([gen_attrs, static_attrs], axis=1)
        return static_attrs

    def ts_attrs(self) -> pd.DataFrame:
        """
        daily_timeseries statistics of all catchments

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of static features of all catchments of shape (654, xxxx)
        """
        files = glob.glob(f"{self.ts_stat_path}/*.csv")
        dfs = []
        for f in files:
            df = pd.read_csv(f, sep=";", index_col=0)
            df.index = df.index.astype(str)
            if len(df) == 654:
                dfs.append(df)
            elif self.verbosity > 1:
                print(f"skipping {os.path.basename(f)} as it has {len(df)} rows")

        return pd.concat(dfs, axis=1)

    def _static_data(self) -> pd.DataFrame:
        """
        static attributes plus timeseries statistics

        Returns
        -------
        pd.DataFrame
            a :obj:`pandas.DataFrame` of static features of all catchments of shape (654, xxxx)
        """

        static_data = pd.concat([
            self.static_attrs(),
            self.ts_attrs()
        ], axis=1)
        # remove duplicated columns
        df = static_data.loc[:, ~static_data.columns.duplicated()].copy()

        df.rename(columns=self.static_map, inplace=True)

        return df

    def _read_stn_dyn(self, station: str) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(self.daily_ts_path, f"CAMELS_FR_tsd_{station}.csv"),
            sep=";",
            index_col=0,
            parse_dates=True,
            comment="#",
        )

        df.rename(columns=self.dyn_map, inplace=True)

        self._apply_dyn_factors(df)

        return df
