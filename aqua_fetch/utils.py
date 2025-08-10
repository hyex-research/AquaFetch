
import re
import os
import sys
import glob
import shutil
import random
import datetime
import warnings
import zipfile
import tempfile
from io import BytesIO
import urllib.request as ulib
from typing import Union, List
import urllib.parse as urlparse

import requests

import numpy as np
import pandas as pd

from ._backend import shapefile, plt, xarray, netCDF4, matplotlib
from ._backend import fiona


COLORS = ['#CDC0B0', '#00FFFF', '#76EEC6', '#C1CDCD', '#E3CF57', '#EED5B7', '#8B7D6B', '#0000FF', '#8A2BE2', '#9C661F',
          '#FF4040', '#8A360F', '#98F5FF', '#FF9912', '#B23AEE', '#9BCD9B', '#8B8B00']


# following files must exist withing data folder for CAMELS-GB data
DATA_FILES = {
    'CAMELS-GB': [
        'CAMELS_GB_climatic_attributes.csv',
        'CAMELS_GB_humaninfluence_attributes.csv',
        'CAMELS_GB_hydrogeology_attributes.csv',
        'CAMELS_GB_hydrologic_attributes.csv',
        'CAMELS_GB_hydrometry_attributes.csv',
        'CAMELS_GB_landcover_attributes.csv',
        'CAMELS_GB_soil_attributes.csv',
        'CAMELS_GB_topographic_attributes.csv'
    ],
    'HYSETS': [  # following files must exist in a folder containing HYSETS dataset.
        'HYSETS_2020_ERA5.nc',
        'HYSETS_2020_ERA5Land.nc',
        'HYSETS_2020_ERA5Land_SWE.nc',
        'HYSETS_2020_Livneh.nc',
        'HYSETS_2020_nonQC_stations.nc',
        'HYSETS_2020_SCDNA.nc',
        'HYSETS_2020_SNODAS_SWE.nc',
        'HYSETS_elevation_bands_100m.csv',
        'HYSETS_watershed_boundaries.zip',
        'HYSETS_watershed_properties.txt'
    ]
}


def download(
        url: str,
        outdir: Union[str, os.PathLike] = None,
        fname: str = None,
        verbosity:int = 1,
        )->os.PathLike:
    """
    High level function, which downloads URL into tmp file in current
    directory and then moves and/or renames it to outdir/fname

    :param url:
    :param outdir: output directory
    :param fname: filename to save the downloaded file. If not given, then autodetected from either URL
        or HTTP headers.
    :return:    filepath] where URL is downloaded to
    """
    if outdir is None:
        outdir = os.getcwd()

    # get filename for temp file in current directory
    prefix = filename_from_url(url)
    (fd, tmpfile) = tempfile.mkstemp(".tmp", prefix=prefix, dir=".")
    os.close(fd)
    os.unlink(tmpfile)

    # set progress monitoring callback
    def callback_charged(blocks, block_size, total_size):
        # 'closure' to set bar drawing function in callback
        callback_progress(blocks, block_size, total_size, bar_function=bar)

    if verbosity:
        callback = callback_charged
    else:
        callback = None

    # Python 3 can not quote URL as needed
    binurl = list(urlparse.urlsplit(url))
    binurl[2] = urlparse.quote(binurl[2])
    binurl = urlparse.urlunsplit(binurl)

    try:
        (tmpfile, headers) = ulib.urlretrieve(binurl, tmpfile, callback)
    except ulib.HTTPError as e:
        print(f"HTTP Error for {url} to download {fname}")
        raise e
    
    filename = filename_from_url(url)

    if fname:
        filename = fname
    
    if filename is None:
        filename = 'temp'
        warnings.warn(f"saving file from {url} as {filename}")

    fpath = outdir + os.sep + filename

    # add numeric ' (x)' suffix if filename already exists
    if os.path.exists(fpath):
        fpath = fpath + '1'
    shutil.move(tmpfile, fpath)

    # print headers
    return fpath


__current_size = 0


def callback_progress(blocks, block_size, total_size, bar_function):
    """callback function for urlretrieve that is called when connection is
    created and when once for each block

    draws adaptive progress bar in terminal/console

    use sys.stdout.write() instead of "print,", because it allows one more
    symbol at the line end without linefeed on Windows

    :param blocks: number of blocks transferred so far
    :param block_size: in bytes
    :param total_size: in bytes, can be -1 if server doesn't return it
    :param bar_function: another callback function to visualize progress
    """
    global __current_size

    width = 100

    if sys.version_info[:3] == (3, 3, 0):  # regression workaround
        if blocks == 0:  # first call
            __current_size = 0
        else:
            __current_size += block_size
        current_size = __current_size
    else:
        current_size = min(blocks * block_size, total_size)
    progress = bar_function(current_size, total_size, width)
    if progress:
        sys.stdout.write("\r" + progress)


def filename_from_url(url):
    """:return: detected filename as unicode or None"""
    # [ ] test urlparse behavior with unicode url
    fname = os.path.basename(urlparse.urlparse(url).path)
    if len(fname.strip(" \n\t.")) == 0:
        return None

    if fname.startswith(":"):
        old_fname = fname
        fname = old_fname[1:]
        warnings.warn(f"fname changed from {old_fname} to {fname}")
    return fname


def bar(current_size, total_size, width):
    percent = current_size/total_size * 100
    if round(percent % 1, 4) == 0.0:
        print(f"{round(percent)}% of {round(total_size*1e-6, 2)} MB downloaded")
    return


def check_attributes(
        attributes:Union[str, List[str]],
        check_against: List[str],
        attribute_name:str = ''
        ) -> List[str]:

    if isinstance(attributes, str) and attributes == 'all':
        attributes = check_against
    elif not isinstance(attributes, list):
        assert isinstance(attributes, str), f"unknown type {type(attributes)} for {attribute_name}"
        assert attributes in check_against, f"invalid value {attributes} for {attribute_name}"
        attributes = [attributes]
    else:
        assert isinstance(attributes, list), f'unknown attributes {attributes}'

    if not all(elem in check_against for elem in attributes):
        print(f"Allowed {attribute_name} are {check_against}")
        print(f"Given {attribute_name} are {attributes}")
        raise ValueError(f"The names of some {attribute_name} are not valid/allowed")

    return attributes


def sanity_check(dataset_name, path, url=None):
    if dataset_name in DATA_FILES:
        if dataset_name == 'CAMELS-GB':
            if not os.path.exists(os.path.join(path, 'data')):
                raise FileNotFoundError(f"No folder named `data` exists inside {path}")
            else:
                data_path = os.path.join(path, 'data')
                for file in DATA_FILES[dataset_name]:
                    if not os.path.exists(os.path.join(data_path, file)):
                        raise FileNotFoundError(f"File {file} must exist inside {data_path}")
    _maybe_not_all_files_downloaded(path, url)
    return


def _maybe_not_all_files_downloaded(
        path:str,
        url:Union[str, list, dict]
):
    if isinstance(url, dict):
        available_files = os.listdir(path)

        for fname, link in url.items():
            if fname not in available_files:
                print(f"file {fname} is not available so downloading it now.")
                download_and_unzip(path, {fname:link})

    return


def check_st_en(
        df:pd.DataFrame,
        st:Union[int, str, pd.DatetimeIndex]=None,
        en:Union[int, str, pd.DatetimeIndex]=None
)->pd.DataFrame:
    """slices the :obj:`pandas.DataFrame` based upon st and en"""
    if isinstance(st, int):
        if en is None:
            en = len(df)
        else:
            assert isinstance(en, int)
        df = df.iloc[st:en]

    elif isinstance(st, (str, pd.DatetimeIndex)):
        if en is None:
            en = df.index[-1]
        df = df.loc[st:en]

    elif isinstance(en, int):
        st = 0 # st must be none here
        df = df.iloc[st:en]
    elif isinstance(en, (str, pd.DatetimeIndex)):
        st = df.index[0]
        df = df.loc[st:en]

    return df


def unzip_all_in_dir(dir_name, ext=".gz"):
    gz_files = glob.glob(f"{dir_name}/*{ext}")
    for f in gz_files:
        shutil.unpack_archive(f, dir_name)
    return


def maybe_download(
        path,
        url:Union[str, List[str], dict],
        overwrite:bool=False,
        name=None,
        include:list=None,
        files_to_check:list = None,
        verbosity:int = 1,
        **kwargs):
    """
    Parameters
    ----------
    path :
        The path where to download the files. If it already exists and is not empty,
        then the files will not be downloaded again unless overwrite is True
    url :
    overwrite :
    name :
    include :
    files_to_check : list
        if given, then even if the ds_dir exists, it will be checked that
        whetehr all files are present or not if not, then the files which
        are not present will be downloaded. This argument can be used to
        make sure that only undownloaded files are downloaded again instead
        of downloading all the files again
    verbosity : int
    **kwargs :
        any keyword arguments for download_and_unzip function
    """
    if os.path.exists(path) and len(os.listdir(path)) > 0:
        if overwrite:
            print(f"removing previous data directory {path} and downloading new")
            shutil.rmtree(path)
            download_and_unzip(path, 
                               url=url, 
                               include=include, 
                               verbosity=verbosity,
                               **kwargs)
        elif files_to_check:  # todo: not checking which files are present or not
            download_and_unzip(path, 
                               url=url,
                               files_to_check=files_to_check,
                               verbosity=verbosity,
                               **kwargs)
        else:
            if verbosity:
                print(f"""
        Not downloading the data since the directory 
        {path} already exists.
        Use overwrite=True to remove previously saved files and download again""")
            sanity_check(name, path, url)
    else:
        download_and_unzip(path, url=url, include=include,
                           verbosity=verbosity,
                            **kwargs)
    return


def download_and_unzip(
        path,
        url:Union[str, List[str], dict],
        include:List[str]=None,
        files_to_check:List[str] = None,
        verbosity:int = 1,
        **kwargs
        ):
    """

    parameters
    ----------
    path :
        The path where to download the files
    url :

    include :
        files to download. Files which are not in include will not be
        downloaded.
    files_to_check :
        This argument can be used to make sure that only undownloaded files
        are downloaded again instead of downloading all the files again
    **kwargs :
        any keyword arguments for download_from_zenodo function
    """
    from .download_zenodo import download_from_zenodo

    if not os.path.exists(path):
        os.makedirs(path)
    if isinstance(url, str):
        if verbosity>0: print(f"downloading {url} to {path}")
        if 'zenodo' in url:
            download_from_zenodo(path, 
                                 doi=url, 
                                 include=include,
                                 files_to_check=files_to_check,
                                 **kwargs)
        else:
            download(url, path, verbosity=verbosity)
        unzip(path, verbosity=verbosity)
    elif isinstance(url, list):
        if verbosity>0: print(f"downloading {len(url)} files to {path}")

        for url in url:
            if verbosity>0: print(f"downloading {url}")

            if 'zenodo' in url:
                download_from_zenodo(path, 
                                     doi=url, 
                                     include=include,
                                     files_to_check=files_to_check,
                                     **kwargs)
            else:
                download(url, path, verbosity=verbosity)
        unzip(path, verbosity=verbosity)
    elif isinstance(url, dict):
        if verbosity>0: print(f"downloading {len(url)} files to {path}")

        for fname, url in url.items():
            if verbosity>0: print(f"downloading {fname}")

            if 'zenodo' in url:
                download_from_zenodo(path, 
                                     doi=url, 
                                     include=include,
                                     files_to_check=files_to_check,
                                     **kwargs)
            elif 'drive.google' in url:
                download_from_google_drive(url, path, fname, verbosity=verbosity)   
            else:
                if include is not None or files_to_check is not None:
                    raise ValueError("include and files_to_check are available only for zenodo")
                download(url, path, fname, verbosity=verbosity)
        unzip(path, verbosity=verbosity)

    else:
        raise ValueError(f"Invalid url: {path}, {url}")

    return


def download_from_google_drive(url, path, fname, verbosity=1):

    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None

    URL = "https://drive.google.com/uc?export=download"
    # get the file id from url which exists between d/ and /view
    file_id = url.split('/')[-2]
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    destination = os.path.join(path, fname)
    CHUNK_SIZE = 32768  # The size of each chunk to write (you can adjust this)

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    return


def unzip(
        path:Union[str, os.PathLike], 
        overwrite:bool=False, 
        verbosity=1
        ):
    """unzip all the zipped files in a directory"""

    if verbosity>0: print(f"unzipping files in {path}")

    all_zip_files = glob.glob(f"{path}/*.zip")

    for zip_file_path in all_zip_files:

        src = os.path.basename(zip_file_path)
        trgt = src.split('.zip')[0]

        if os.path.exists(os.path.join(path, trgt)) and overwrite:
            if verbosity>0: print(f"removing pre-existing {trgt}")
            shutil.rmtree(os.path.join(path, trgt))

        if not os.path.exists(os.path.join(path, trgt)):

            if verbosity>0: print(f"unzipping {src} to {trgt}")

            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                try:
                    zip_ref.extractall(os.path.join(path, trgt))
                except OSError:
                    filelist = zip_ref.filelist
                    for _file in filelist:
                        if '.txt' in _file.filename or '.csv' in _file.filename or '.xlsx' in _file.filename:
                            zip_ref.extract(_file)
        else:
            if verbosity>0: print(f"{trgt} already exists")

    # extracting tar.gz files todo, check if zip files can also be unpacked by the following oneliner
    gz_files = glob.glob(f"{path}/*.gz")
    for gz_file in gz_files:
        shutil.unpack_archive(gz_file, path)

    sevenz_files = glob.glob(f"{path}/*.7z")

    if len(sevenz_files) > 0:
        try:
            import py7zr
        except (ModuleNotFoundError, ImportError):
            raise ImportError('py7zr is required to extract the .7z files. Please install it using `pip install py7zr`')

        for fpath in sevenz_files:
            if not os.path.exists(os.path.join(path, fpath.split('.7z')[0])):
                with py7zr.SevenZipFile(fpath, mode='r') as z:
                    z.extractall(path = path)
                    print(f'Extracted {fpath}')
    return


class OneHotEncoder(object):
    """
    >>> from aqua_fetch import mg_degradation
    >>> data, _ = mg_degradation()
    >>> cat_enc1 = OneHotEncoder()
    >>> cat_ = cat_enc1.fit_transform(data['catalyst_type'].values)
    >>> _cat = cat_enc1.inverse_transform(cat_)
    >>> all([a==b for a,b in zip(data['catalyst_type'].values, _cat)])
    """
    def fit(self, X:np.ndarray):
        assert len(X) == X.size
        categories, inverse = np.unique(X, return_inverse=True)
        X = np.eye(categories.shape[0])[inverse]
        self.categories_ = [categories]
        return X

    def transform(self, X):
        return X

    def fit_transform(self, X):
        return self.transform(self.fit(X))

    def inverse_transform(self, X):
        return pd.DataFrame(X, columns=self.categories_[0]).idxmax(1).values


class LabelEncoder(object):
    """
    >>> from aqua_fetch import mg_degradation
    >>> data, _ = mg_degradation()
    >>> cat_enc1 = LabelEncoder()
    >>> cat_ = cat_enc1.fit_transform(data['catalyst_type'].values)
    >>> _cat = cat_enc1.inverse_transform(cat_)
    >>> all([a==b for a,b in zip(data['catalyst_type'].values, _cat)])
    """
    def fit(self, X):
        assert len(X) == X.size
        categories, inverse = np.unique(X, return_inverse=True)
        self.categories_ = [categories]
        labels = np.unique(inverse)
        self.mapper_ = {label:category for category,label in zip(categories, labels)}
        return inverse

    def transform(self, X):
        return X

    def fit_transform(self, X):
        return self.transform(self.fit(X))

    def inverse_transform(self, X:np.ndarray):
        assert len(X) == X.size
        X = np.array(X).reshape(-1,)
        return pd.Series(X).map(self.mapper_).values


def ohe_column(df:pd.DataFrame, col_name:str)->tuple:
    """one hot encode a column in datatrame"""
    assert isinstance(col_name, str)
    assert isinstance(df, pd.DataFrame)

    encoder = OneHotEncoder()
    ohe_cat = encoder.fit_transform(df[col_name].values.reshape(-1, 1))
    cols_added = [f"{col_name}_{i}" for i in range(ohe_cat.shape[-1])]

    # todo : this can be problematic if index in df does not start from 0,1,...
    df = pd.concat([df, pd.DataFrame(ohe_cat, columns=cols_added)], axis=1)

    df.pop(col_name)

    return df, cols_added, encoder


def le_column(df:pd.DataFrame, col_name:str)->tuple:
    """label encode a column in dat:obj:`pandas.DataFrame`aframe"""
    encoder = LabelEncoder()
    index = df.columns.to_list().index(col_name)
    encoded = encoder.fit_transform(df[col_name])
    df.pop(col_name)
    df.insert(index, col_name, encoded)
    return df, None, encoder


def plot_polygon_feature(feature, n, bbox):
    f_if = feature.shape.__geo_interface__
    polys = len(f_if['coordinates'])
    def_col = n
    for i in range(polys):
        a = np.array(f_if['coordinates'][i])
        if a.ndim < 2 and len(a.shape) > 0:
            c = a
            m = max([len(ci) for ci in c])
            for ci in c:
                col = 'k' if len(ci) != m else def_col
                x = np.array([k[0] for k in ci])
                y = np.array([k[1] for k in ci])
                plt.plot(x, y, col, label="__none__", linewidth=0.5)

        elif len(a.shape) > 0:
            b = a.reshape(-1, 2)
            plt.plot(b[:, 0], b[:, 1], def_col)
        plt.ylim([bbox[1], bbox[3]])
        plt.xlim([bbox[0], bbox[2]])
    return


def dateandtime_now() -> str:
    """
    Returns the datetime in following format as string
    YYYYMMDD_HHMMSS
    """
    jetzt = datetime.datetime.now()
    dt = ''
    for time in ['year', 'month', 'day', 'hour', 'minute', 'second']:
        _time = str(getattr(jetzt, time))
        if len(_time) < 2:
            _time = '0' + _time
        if time == 'hour':
            _time = '_' + _time
        dt += _time
    return dt


def find_records(shp_file, record_name, feature_number):
    """find the metadata about feature given its feature number and column_name which contains the data"""
    assert os.path.exists(shp_file), f'{shp_file} does not exist'
    shp_reader = shapefile.Reader(shp_file)
    col_no = find_col_name(shp_reader, record_name)

    if col_no == -99:
        raise ValueError(f'no column named {record_name} found in {shp_reader.shapeName}')
    else:
        # print(col_no, 'is the col no')
        name = get_record_in_col(shp_reader, feature_number, col_no)
    return name


def find_col_name(shp_reader, field_name):
    _col_no = 0
    col_no = -99
    for fields in shp_reader.fields:
        _col_no += 1
        for field in fields:
            if field == field_name:
                col_no = _col_no
                break
    return col_no


def get_record_in_col(shp_reader, i, col_no):
    recs = shp_reader.records()
    col_no = col_no - 2  # -2, 1 for index reduction, 1 for a junk column shows up in records
    return recs[i][col_no]


class Resampler(object):
    """
    Resamples time-series data from one frequency to another frequency.
    """
    min_in_freqs = {
        'MIN': 1,
        'MINUTE': 1,
        'DAILY': 1440,
        'D': 1440,
        'HOURLY': 60,
        'HOUR': 60,
        'H': 60,
        'MONTHLY': 43200,
        'M': 43200,
        'YEARLY': 525600
        }

    def __init__(self, data, freq, how='mean', verbosity=1):
        """
        Arguments:
            data : data to use
            freq : frequency at which to transform/resample
            how : string or dictionary mapping to columns in data defining how to resample the data.
        """
        data = pd.DataFrame(data)
        self.orig_df = data.copy()
        self.target_freq = self.freq_in_mins_from_string(freq)
        self.how = self.check_how(how)
        self.verbosity = verbosity


    def __call__(self, *args, **kwargs):
        if self.target_freq > self.orig_freq:
            # we want to calculate at higher/larger time-step
            return self.downsample()

        else:
            # we want to calculate at smaller time-step
            return self.upsamle()

    @property
    def orig_freq(self):
        return self.freq_in_mins_from_string(pd.infer_freq(self.orig_df.index))

    @property
    def allowed_freqs(self):
        return self.min_in_freqs.keys()

    def check_how(self, how):
        if not isinstance(how, str):
            assert isinstance(how, dict)
            assert len(how) == len(self.orig_df.columns)
        else:
            assert isinstance(how, str)
            how = {col:how for col in self.orig_df.columns}
        return how

    def downsample(self):
        df = pd.DataFrame()
        for col in self.orig_df:
            _df = downsample_df(self.orig_df[col], how=self.how[col], target_freq=self.target_freq)
            df = pd.concat([df, _df], axis=1)

        return df

    def upsamle(self, drop_nan=True):
        df = pd.DataFrame()
        for col in self.orig_df:
            _df = upsample_df(self.orig_df[col], how=self.how[col], target_freq=self.target_freq)
            df = pd.concat([df, _df], axis=1)

        # concatenation of dataframes where one sample was upsampled with linear and the other with same, will result
        # in different length and thus concatenation will add NaNs to the smaller column.
        if drop_nan:
            df = df.dropna()
        return df

    def str_to_mins(self, input_string: str) -> int:

        return self.min_in_freqs[input_string]

    def freq_in_mins_from_string(self, input_string: str) -> int:

        if has_numbers(input_string):
            in_minutes = split_freq(input_string)
        elif input_string.upper() in ['D', 'H', 'M', 'DAILY', 'HOURLY', 'MONTHLY', 'YEARLY', 'MIN', 'MINUTE']:
            in_minutes = self.str_to_mins(input_string.upper())
        else:
            raise TypeError("invalid input string", input_string)

        return int(in_minutes)


def downsample_df(df, how, target_freq):

    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)

    assert how in ['mean', 'sum']
    # from low timestep to high timestep i.e from 1 hour to 24 hour
    # For quantities like temprature, relative humidity, Q, wind speed
    if how == 'mean':
        return df.resample(f'{target_freq}min').mean()
    # For quantities like 'rain', solar radiation', evapotranspiration'
    elif how == 'sum':
        return df.resample(f'{target_freq}min').sum()

def upsample_df(df,  how:str, target_freq:int):
    """drop_nan: if how='linear', we may """
    # from larger timestep to smaller timestep, such as from daily to hourly
    out_freq = str(target_freq) + 'min'

    if isinstance(df, pd.Series):
        df = pd.DataFrame(df)
    col_name  = df.columns[0]
    nan_idx = df.isna()  # preserving indices with nan values
    assert df.shape[1] <=1

    nan_idx_r = nan_idx.resample(out_freq).ffill()
    nan_idx_r = nan_idx_r.fillna(False)  # the first value was being filled with NaN, idk y?
    data_frame = df.copy()

    # For quantities like temprature, relative humidity, Q, wind speed, we would like to do an interpolation
    if how == 'linear':
        data_frame = data_frame.resample(out_freq).interpolate(method='linear')
        # filling those interpolated values with NaNs which were NaN before interpolation
        data_frame[nan_idx_r] = np.nan

    # For quantities like 'rain', solar radiation', evapotranspiration', we would like to distribute them equally
    # at smaller time-steps.
    elif how == 'same':
        # distribute rainfall equally to smaller time steps. like hourly 17.4 will be 1.74 at 6 min resolution
        idx = data_frame.index[-1] + get_offset(data_frame.index.freqstr)
        data_frame = data_frame.append(data_frame.iloc[[-1]].rename({data_frame.index[-1]: idx}))
        data_frame = add_freq(data_frame)
        df1 = data_frame.resample(out_freq).ffill().iloc[:-1]
        df1[col_name ] /= df1.resample(data_frame.index.freqstr)[col_name ].transform('size')
        data_frame = df1.copy()
        # filling those interpolated values with NaNs which were NaN before interpolation
        data_frame[nan_idx_r] = np.nan

    else:
        raise ValueError(f"unoknown method to transform '{how}'")

    return data_frame


def add_freq(df, assert_feq=False, freq=None, method=None):

    idx = df.index.copy()
    if idx.freq is None:
        _freq = pd.infer_freq(idx)
        idx.freq = _freq

        if idx.freq is None:
            if assert_feq:
                df = force_freq(df, freq, method=method)
            else:

                raise AttributeError('no discernible frequency found.  Specify'
                                     ' a frequency string with `freq`.'.format())
        else:
            df.index = idx
    return df


def force_freq(data_frame, freq_to_force, method=None):

    old_nan_counts = data_frame.isna().sum()
    old_shape = data_frame.shape
    dr = pd.date_range(data_frame.index[0], data_frame.index[-1], freq=freq_to_force)

    df_unique = data_frame[~data_frame.index.duplicated(keep='first')]  # first remove duplicate indices if present
    if method:
        df_idx_sorted = df_unique.sort_index()
        df_reindexed = df_idx_sorted.reindex(dr, method='nearest')
    else:
        df_reindexed = df_unique.reindex(dr, fill_value=np.nan)

    df_reindexed.index.freq = pd.infer_freq(df_reindexed.index)
    new_nan_counts = df_reindexed.isna().sum()
    print('Frequency {} is forced to dataframe, NaN counts changed from {} to {}, shape changed from {} to {}'
          .format(df_reindexed.index.freq, old_nan_counts.values, new_nan_counts.values,
                  old_shape, df_reindexed.shape))
    return df_reindexed


def split_freq(freq_str: str) -> int:
    match = re.match(r"([0-9]+)([a-z]+)", freq_str, re.I)
    if match:
        minutes, freq = match.groups()
        if freq.upper() in ['H', 'HOURLY', 'HOURS', 'HOUR']:
            minutes = int(minutes) * 60
        elif freq.upper() in ['D', 'DAILY', 'DAY', 'DAYS']:
            minutes = int(minutes) * 1440
        return int(minutes)
    else:
        raise NotImplementedError

TIME_STEP = {'D': 'Day', 'H': 'Hour', 'M': 'MonthEnd'}

def get_offset(freqstr: str) -> str:
    offset_step = 1
    if freqstr in TIME_STEP:
        freqstr = TIME_STEP[freqstr]
    elif has_numbers(freqstr):
        in_minutes = split_freq(freqstr)
        freqstr = 'Minute'
        offset_step = int(in_minutes)

    offset = getattr(pd.offsets, freqstr)(offset_step)

    return offset

def has_numbers(input_string: str) -> bool:
    return bool(re.search(r'\d', input_string))


def get_version_info()->dict:

    from .__init__ import __version__

    versions = {
        'numpy': np.__version__,
        'pandas': pd.__version__,
        'aqua_fetch': __version__,
        'python': sys.version,
        'os': os.name
    }

    if plt is not None:
        versions['matplotlib'] = matplotlib.__version__

    if shapefile is not None:
        versions['shapefile'] = shapefile.__version__

    if xarray is not None:
        versions['xarray'] = xarray.__version__
    
    if netCDF4 is not None:
        versions['netCDF4'] = netCDF4.__version__

    try:
        import scipy
        versions['scipy'] = scipy.__version__
    except (ImportError, ModuleNotFoundError):
        pass

    if fiona is not None:
        versions['fiona'] = fiona.__version__

    return versions


def hardware_info()->dict:
    mem_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')  # e.g. 4015976448
    mem_gib = mem_bytes / (1024. ** 3)  # e.g. 3.74
    return dict(
        tot_cpus= os.cpu_count(),
        avail_cpus = os.cpu_count() if os.name=="nt" else len(os.sched_getaffinity(0)),
        mem_gib=mem_gib,
    )


def print_info(
        include_run_time:bool=True,
        include_hardware_info:bool=True
        ):
    info = get_version_info()

    if include_run_time:

        jetzt = datetime.datetime.now()
        zeitpunkt = jetzt.strftime("%d %B %Y %H:%M:%S")
        info['Script Executed on: '] = zeitpunkt

    if include_hardware_info:
        info.update(hardware_info())

    for k, v in info.items():
        print(k, v)
    return


def get_cpus()->int:
    if os.name == "nt":
        return os.cpu_count()
    else:
        return len(os.sched_getaffinity(0))


def merge_shapefiles_fiona(shp_files, output_path):
    """
    merges shapefiles into one shapefile using fiona
    """

    if os.path.exists(output_path):
        return

    import fiona

    print(f"Merging {len(shp_files)} shapefiles into {output_path}")

    # Read schema and CRS from the first shapefile
    with fiona.open(shp_files[0], 'r') as first:
        schema = first.schema
        crs = first.crs
        geom_type = first.schema['geometry']

    # Define schema with one string property: the shapefile name
    schema = {
        'geometry': geom_type,
        'properties': {'gauge_id': 'str'}
    }

    # Write merged output
    with fiona.open(output_path, 'w', driver='ESRI Shapefile', crs=crs, schema=schema) as output:
        for idx, shp in enumerate(shp_files):

            base_name = os.path.splitext(os.path.basename(shp))[0]

            with fiona.open(shp, 'r') as src:
                for feature in src:
                    new_feature = {
                        'geometry': feature['geometry'],
                        'properties': {'gauge_id': base_name}
                    }
                    output.write(new_feature)

            if idx % 1000 == 0:
                print(f"Processed {idx + 1}/{len(shp_files)} shapefiles...")

    print(f"Merged {len(shp_files)} shapefiles into {output_path}")

    return


def encode_cols(
        df:pd.DataFrame,
        cols:List[str],
        encoding:str,
)->tuple:
    encoders = {}

    for col in cols:
        if encoding == "ohe":
            df, _, encoders[col] = ohe_column(df, col)
        elif encoding == "le":
            df, _, encoders[col] = le_column(df, col)

    return df, encoders


def maybe_download_and_read_data(
        url: str,
        file_name: str,
        **kwargs
) -> pd.DataFrame:
    """
    Download and read tabular (csv/xlsx) data from the given url
    """
    path = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(path):
        os.makedirs(path)
    fpath = os.path.join(path, file_name)

    if os.path.exists(fpath):
        df = pd.read_csv(fpath, index_col="index", **kwargs)
    else:
        if url.startswith("https://ars.els-cdn.com"):
            df = download_with_requests(url)
        elif url.startswith("https://pubs.acs.org"):
            df = download_with_requests(url)
        elif url.endswith(".csv"):
          df = pd.read_csv(url,**kwargs)
        elif url.endswith(".xlsx"):
            df = pd.read_excel(url, **kwargs)
        else:
            raise ValueError(f"Unknown extension: {url.split('.')}")

        df.to_csv(fpath, index=True, index_label="index")

    return df


def download_with_requests(url)->pd.DataFrame:
    # Add headers if needed (you may need to adjust these)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        # Use BytesIO for binary data/excel sheets
        data = BytesIO(response.content)
        df = pd.read_excel(data)
    else:
        raise ValueError(f"{response.status_code}: Failed to download data from {url}")  

    return df
