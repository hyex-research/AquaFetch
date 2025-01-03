
import os
import warnings
from typing import Union, List, Dict
from urllib.error import HTTPError
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from ._misc import _EStreams
from ..utils import get_cpus
from .._backend import xarray as xr

from ._map import (
    catchment_area,
    gauge_latitude,
    gauge_longitude,
    slope
    )


class Ireland(_EStreams):
    """
    Data of 464 catchments of Ireland. Out of these 464 catchments, 
    280 are from OPW and 184 are from EPA.
    The observed streamflow data for EPA stations is downloaded from 
    https://epawebapp.epa.ie/Hydronet/#Flow while the observed streamflow for OPW 
    stations is downloaded from https://waterlevel.ie/hydro-data/#/overview/Waterlevel.
    It should be that out of 280 OPW stations, streamflow data is available for only 129
    stations. 
    The meteorological data, static catchment 
    features and catchment boundaries are
    taken from :py:class:`water_quality.EStreams` follwoing the works
    of ` Nascimento et al., 2024 <https://doi.org/10.5194/hess-25-471-2021>`_ project. Therefore,
    the number of staic features are 35 and dynamic features are 27 and the
    data is available from 1992-01-01 to 2020-06-31.
    """
    def __init__(
            self, 
            path:Union[str, os.PathLike] = None,
            estreams_path:Union[str, os.PathLike] = None,
            verbosity:int=1,
            **kwargs):

        super().__init__(path=path, estreams_path=estreams_path, verbosity=verbosity, **kwargs)

    @property
    def static_map(self) -> Dict[str, str]:
        return {
                'area': catchment_area(),
                'lat': gauge_latitude(),
                'slope_sawicz': slope(''),
                'lon': gauge_longitude(),

        }

    @property
    def country_name(self)->str:
        return 'IE'
   
    @property
    def epa_stations(self)->List[str]:
        md = self.estreams.md
        stns = md.loc[(md['gauge_country']=='IE') & (md['gauge_provider']=='IE_EPA')]['gauge_id']
        epa_stns =  stns.tolist()
        if self.timestep in ('H', 'hourly'): epa_stns.remove('38004')  # todo: 
        return epa_stns

    @property
    def opw_stations(self)->List[str]:
        md = self.estreams.md
        stns = md.loc[(md['gauge_country']=='IE') & (md['gauge_provider']=='IE_OPW')]['gauge_id']
        return stns.tolist()
    
    def is_opw_station(self, stn)->bool:
        return stn in self.opw_stations

    def is_epa_station(self, stn)->bool:
        return stn in self.epa_stations

    def stations(self)->List[str]:
        return self._stations
    
    def gauge_id_basin_id_map(self)->dict:
        # guage_id '18118'
        # basin_id 'IEEP0281'
        # '18118' -> 'IEEP0281'
        return {k:v for v,k in self.md['gauge_id'].to_dict().items()}

    def basin_id_gauge_id_map(self)->dict:
        # guage_id '18118'
        # basin_id 'IEEP0281'
        # 'IEEP0281' -> '18118'
        return self.md['gauge_id'].to_dict()

    def get_q(
            self, 
            as_dataframe:bool=True,
            overwrite:bool=False, 
            ):
        fname = 'daily_q' if self.timestep in ["D", 'daily'] else 'hourly_q'
        ext = '.nc' if self.to_netcdf else '.csv'
    
        fpath = os.path.join(self.path, fname + ext)

        if not os.path.exists(fpath) or overwrite:

            cpus = self.processes or min(get_cpus() - 2, 16)

            if cpus > 1:
                epa_df = self.download_epa_data_parallel(cpus=cpus)
                opw_df = self.download_opw_data_parallel(cpus=cpus)
            else:
                epa_df = self.download_epa_data_seq()
                opw_df = self.download_opw_data_seq()

            data = pd.concat([epa_df, opw_df], axis=1)
            data.index.name = 'time'
            data.rename(columns=self.gauge_id_basin_id_map(), inplace=True)

            if ext == '.csv':
                data.to_csv(fpath, index_label="index")
            else:
                data = xr.Dataset({stn: xr.DataArray(data.loc[:, stn]) for stn in data.columns})
                data.to_netcdf(fpath)

        else:
            if self.verbosity > 1: print(f"Reading from pre-exising {fpath}")
            if ext == '.csv':
                data = pd.read_csv(fpath, index_col="index")
                data.index = pd.to_datetime(data.index)
                data.index.name = 'time'
                data.rename(columns=self.gauge_id_basin_id_map(), inplace=True)
            else:
                data = xr.open_dataset(fpath)
                if self.verbosity > 1: print(f"opened {fpath}")

        if isinstance(data, pd.DataFrame):
            if as_dataframe:
                return data
            return xr.Dataset({stn: xr.DataArray(data.loc[:, stn]) for stn in data.columns})
        else:
            if as_dataframe:
                data = data.to_dataframe()
                data.index.name = 'time'
                return data
            return data

    def download_epa_data_seq(self):
        """
        Examples
        ---------
        >>> epa_df = download_epa_data()
        """
        folder = {'D': 'daily', 'H': 'hourly'}[self.timestep]

        all_epa_data_file = os.path.join(self.path, f"epa_{folder}.csv")
        if os.path.exists(all_epa_data_file):
            if self.verbosity>1: print(f"{all_epa_data_file} already exists")
            df = pd.read_csv(all_epa_data_file, index_col=0, parse_dates=True)
            print(f"{all_epa_data_file} already exists")  
            return df

        print("Downloading EPA data Sequentially")

        epa_failiures = 0
        epa_dfs = []

        for idx, stn in enumerate(self.epa_stations):

            fpath = os.path.join(self.path, "EPA", folder, f"{stn}.csv")

            print(f"{idx}/{len(self.epa_stations)} Downloading {stn}")

            df, epa_failiures = _download_epa_stn_data(fpath, self.timestep, verbosity=self.verbosity-1)

            epa_dfs.append(df) 

        print(f'total epa failiures: {epa_failiures}')
        print(f'total epa dfs: {len(epa_dfs)}')

        df = pd.concat(epa_dfs, axis=1).astype('float32')

        if self.verbosity>1: print(f"Downloaded total epa dfs: {len(epa_dfs)} saving to {all_epa_data_file}")
        df.to_csv(all_epa_data_file)
        return df

    def download_epa_data_parallel(self, cpus=None):

        if cpus is None:
            cpus = min(get_cpus() - 2, 16)

        folder = {'D': 'daily', 'H': 'hourly'}[self.timestep]

        all_epa_data_file = os.path.join(self.path, f"epa_{folder}.csv")
        if os.path.exists(all_epa_data_file):
            df = pd.read_csv(all_epa_data_file, index_col=0, parse_dates=True)
            print(f"{all_epa_data_file} already exists")  
            return df

        timesteps = [self.timestep] * len(self.epa_stations)
        fpaths = [os.path.join(self.path, "EPA", folder, f"{stn}.csv") for stn in self.epa_stations]

        print(f"Downloading {len(fpaths)} EPA stations using {cpus} cpus at {os.path.join(self.path, 'EPA', folder)}")

        with ProcessPoolExecutor(cpus) as executor:
            epa_dfs = list(executor.map(_download_epa_stn_data, fpaths, timesteps))

        df = pd.concat([val[0] for val in epa_dfs], axis=1).astype('float32')

        if self.timestep in ["D", 'daily']:
            # remove hourly information from the index
            # 2000-01-01 01:00:00 -> 2000-01-01
            df.index = df.index.normalize()

        print(f'Downloaded total epa dfs: {len(epa_dfs)}')

        df.to_csv(all_epa_data_file)
        return df
    
    def download_opw_data_parallel(self, cpus=None):

        folder = {'D': 'daily', 'H': 'hourly'}[self.timestep]

        all_opw_data_file = os.path.join(self.path, f"opw_{folder}.csv")
        if os.path.exists(all_opw_data_file):
            df = pd.read_csv(all_opw_data_file, index_col=0, parse_dates=True)
            print(f"{all_opw_data_file} already exists")  
            return df

        fpaths = [os.path.join(self.path, "OPW", folder, f"{stn}.csv") for stn in self.opw_stations]

        if self.verbosity:
            print(f"Downloading {len(fpaths)} OPW stations using {cpus} cpus at {os.path.join(self.path, 'OPW', folder)}")

        with ProcessPoolExecutor(cpus) as executor:
            opw_dfs = list(executor.map(_download_opw_stn_data, fpaths, [self.timestep]*len(self.opw_stations)))

        opw_df = [df for df in opw_dfs]
        opw_df = pd.concat(opw_df, axis=1).astype('float32')

        if self.timestep in ("D", "daily"):
            opw_df.index = opw_df.index.tz_localize(None)

        if self.verbosity:
            print(f"Downloaded total opw dfs: {len(opw_dfs)}")
            print(f"Saving opw data {opw_df.shape} to {all_opw_data_file}")

        failures = [df for df in opw_dfs if len(df)==0]
        if len(failures)>0:
            self.opw_failures = [s.name for s in failures]
            warnings.warn(f"Failed to download {len(failures)} OPW stations")
    
        opw_df.to_csv(all_opw_data_file)

        return opw_df

    def download_opw_data_seq(self):
        """
        Examples
        ---------
        >>> opw_df = download_opw_data()
        """

        folder = {'D': 'daily', 'H': 'hourly'}[self.timestep]

        all_opw_data_file = os.path.join(self.path, f"opw_{folder}.csv")
        if os.path.exists(all_opw_data_file):
            df = pd.read_csv(all_opw_data_file, index_col=0, parse_dates=True)
            if self.verbosity: print(f"{all_opw_data_file} already exists")  
            return df

        if self.verbosity: print("Downloading OPW data")

        failiures = 0
        opw_dfs = []
        for idx, stn in enumerate(self.opw_stations):

            fpath = os.path.join(self.path, "OPW", folder, f"{stn}.csv")

            print(f"{idx}/{len(self.opw_stations)} Downloading {stn}")

            df = _download_opw_stn_data(fpath, self.timestep)
 
            opw_dfs.append(df)
            if len(df)==0:
                failiures += 1

        if self.verbosity:
            print(f"total failiures: {failiures}")
            print(f"total opw dfs: {len(opw_dfs)}")

        opw_dfs1 = [df for df in opw_dfs if len(df)>0]
        opw_df = pd.concat(opw_dfs1, axis=1).astype('float32')

        #if self.timestep in ("D", "daily"):
        opw_df.index = opw_df.index.tz_localize(None)

        if self.verbosity:
            print(f"Saving opw data {opw_df.shape} to {all_opw_data_file}")
        opw_df.to_csv(all_opw_data_file)

        return opw_df


def _download_epa_stn_data(fpath, timestep="D")->pd.Series:
    stn = os.path.basename(fpath).split('.')[0]
    if timestep in ("D", 'daily'):
        fname = "daymean.zip"
    else:
        fname = "15min.zip"

    epa_failiures = 0

    url = f"https://epawebapp.epa.ie/Hydronet/output/internet/stations/DUB/{stn}/Q/complete_{fname}"
    url2 = f"https://epawebapp.epa.ie/Hydronet/output/internet/stations/MON/{stn}/Q/complete_{fname}"
    url3 = f"https://epawebapp.epa.ie/Hydronet/output/internet/stations/ATH/{stn}/Q/complete_{fname}"
    url4 = f"https://epawebapp.epa.ie/Hydronet/output/internet/stations/COR/{stn}/Q/complete_{fname}"
    url5 = f"https://epawebapp.epa.ie/Hydronet/output/internet/stations/KIK/{stn}/Q/complete_{fname}"
    url6 = f"https://epawebapp.epa.ie/Hydronet/output/internet/stations/CAS/{stn}/Q/complete_{fname}"

    try:
        df = pd.read_csv(url, 
                    comment='#', 
                    sep=';',
                    names=["timestamp", stn, "qflag"]
                    )
    except HTTPError:
        try:
            df = pd.read_csv(url2, 
            comment='#', 
            sep=';',
            names=["timestamp", stn, "qflag"])
        except HTTPError:
            try:
                df = pd.read_csv(url3, 
                comment='#', 
                sep=';',
                names=["timestamp", stn, "qflag"])
            except HTTPError:
                try:
                    df = pd.read_csv(url4, 
                    comment='#', 
                    sep=';',
                    names=["timestamp", stn, "qflag"])
                except HTTPError:
                    try:
                        df = pd.read_csv(url5, 
                        comment='#', 
                        sep=';',
                        names=["timestamp", stn, "qflag"])
                    except HTTPError:
                        try:
                            df = pd.read_csv(url6, 
                            comment='#', 
                            sep=';',
                            names=["timestamp", stn, "qflag"])
                        except HTTPError:
                            print(f"Failed to download {stn}")
                            epa_failiures += 1
                            pass


    df.index = pd.to_datetime(df.pop('timestamp'))

    # considering quality codes https://epawebapp.epa.ie/Hydronet/#FAQ
    # Quality codes: Good, nan, Suspect, Extrapolated, Unchecked, Excellent, Estimated

    df = df.loc[~df['qflag'].isin(['Unchecked'])]
    
    if timestep == "D":
        return df[stn], epa_failiures

    return df[stn].resample(timestep).mean(), epa_failiures


def _download_opw_stn_data(fpath, timestep="D")->pd.Series:
    stn = os.path.basename(fpath).split('.')[0]
    # we don't/can't download daily data 
    if timestep == "daily":
        timestep = "D"
    elif timestep == "hourly":
        timestep = "H"
    elif timestep == "D":
        pass
    else:
        raise ValueError(f"timestep should be either 'D' or 'H' but it is {timestep}")

    url = f"https://waterlevel.ie/hydro-data/data/internet/stations/0/{stn}/Q/Discharge_complete.zip"

    try:
        df = pd.read_csv(url,
                        comment='#',
                        sep=';',
                        names=["timestamp", stn, "q_code"]
                        )
    except HTTPError:
        warnings.warn(f"Failed to download {stn}", UserWarning)
        df = pd.Series(name=stn)

    df.index = pd.to_datetime(df.pop('timestamp'))
    df.index = df.index.tz_localize(None)  

    # considering quality codes as given here https://waterlevel.ie/hydro-data/#/html/qualitycodes
    # df['q_code'] has following values : 36, 46, 31, 56, 96, 225, 101, 32, 99, 254
    # 96 and 254 are provisional values and can be changed!
    # 101 is erronous while 36 and 46 contain fair and siginificant errors respectively.

    # get rows where q_code is not 96 or 254
    df = df.loc[~df['q_code'].isin([96, 254])]
    
    stn_data = df[stn]
    #stn_data = stn_data.resample(timestep).apply(lambda subdata: tw_resampler(subdata, stn_data.sort_index(), timestep))    
    stn_data = stn_data.resample(timestep).mean()
    return stn_data

