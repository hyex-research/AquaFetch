
import os
from typing import Union

import pandas as pd

from .._datasets import maybe_download

from ..utils import check_attributes


def ecoli_mekong(
        st: Union[str, pd.Timestamp, int] = "20110101",
        en: Union[str, pd.Timestamp, int] = "20211231",
        features:Union[str, list] = None,
        overwrite=False
)->pd.DataFrame:
    """
    E. coli data from Mekong river (Houay Pano) area from 2011 to 2021
    `Boithias et al., 2022 <https://doi.org/10.5194/essd-14-2883-2022>`_ .

    Parameters
    ----------
        st : optional
            starting time. The default starting point is 2011-05-25 10:00:00
        en : optional
            end time, The default end point is 2021-05-25 15:41:00
        features : str, optional
            names of features to use. use ``all`` to get all features. By default
            following input features are selected

                - ``station_name`` name of station/catchment where the observation was made
                - ``T`` temperature
                - ``EC`` electrical conductance
                - ``DOpercent`` dissolved oxygen concentration
                - ``DO`` dissolved oxygen saturation
                - ``pH`` pH
                - ``ORP`` oxidation-reduction potential
                - ``Turbidity`` turbidity
                - ``TSS`` total suspended sediment concentration
                - ``E-coli_4dilutions`` Eschrechia coli concentration

        overwrite : bool
            whether to overwrite the downloaded file or not

    Returns
    -------
    pd.DataFrame
        with default parameters, the shape is (1602, 10)

    Examples
    --------
        >>> from water_datasets import ecoli_mekong
        >>> ecoli_data = ecoli_mekong()
        >>> ecoli_data.shape
        (1602, 10)

    """
    ecoli = ecoli_houay_pano(st, en, features, overwrite=overwrite)
    ecoli1 = ecoli_mekong_2016(st, en, features, overwrite=overwrite)
    ecoli2 = ecoli_mekong_laos(st, en, features, overwrite=overwrite)
    return pd.concat([ecoli, ecoli1, ecoli2])


def ecoli_mekong_2016(
        st: Union[str, pd.Timestamp, int] = "20160101",
        en: Union[str, pd.Timestamp, int] = "20161231",
        features:Union[str, list] = None,
        overwrite=False
)->pd.DataFrame:
    """
    E. coli data from Mekong river from 2016 from 29 catchments

    Parameters
    ----------
        st :
            starting time
        en :
            end time
        features : str, optional
            names of features to use. use ``all`` to get all features.
        overwrite : bool
            whether to overwrite the downloaded file or not

    Returns
    -------
    pd.DataFrame
        with default parameters, the shape is (58, 10)

    Examples
    --------
        >>> from water_datasets import ecoli_mekong_2016
        >>> ecoli = ecoli_mekong_2016()
        >>> ecoli.shape
        (58, 10)

    .. url_
        https://dataverse.ird.fr/dataset.xhtml?persistentId=doi:10.23708/ZRSBM4
    """
    url = {"ecoli_mekong_2016.csv": "https://dataverse.ird.fr/api/access/datafile/8852"}

    path = os.path.join(os.path.dirname(__file__), 'data', 'ecoli_mekong_2016')

    return _fetch_ecoli(path, overwrite, url, None, features, st, en,
                        "ecoli_houay_pano_tab_file")


def ecoli_houay_pano(
        st: Union[str, pd.Timestamp, int] = "20110101",
        en: Union[str, pd.Timestamp, int] = "20211231",
        features:Union[str, list] = None,
        overwrite=False
)->pd.DataFrame:
    """
    E. coli data from Mekong river (Houay Pano) area.

    Parameters
    ----------
        st : optional
            starting time. The default starting point is 2011-05-25 10:00:00
        en : optional
            end time, The default end point is 2021-05-25 15:41:00
        features : str, optional
            names of features to use. use ``all`` to get all features. By default
            following input features are selected

                ``station_name`` name of station/catchment where the observation was made
                ``T`` temperature
                ``EC`` electrical conductance
                ``DOpercent`` dissolved oxygen concentration
                ``DO`` dissolved oxygen saturation
                ``pH`` pH
                ``ORP`` oxidation-reduction potential
                ``Turbidity`` turbidity
                ``TSS`` total suspended sediment concentration
                ``E-coli_4dilutions`` Eschrechia coli concentration

        overwrite : bool
            whether to overwrite the downloaded file or not

    Returns
    -------
    pd.DataFrame
        with default parameters, the shape is (413, 10)

    Examples
    --------
        >>> from water_datasets import ecoli_houay_pano
        >>> ecoli = ecoli_houay_pano()
        >>> ecoli.shape
        (413, 10)

    .. url_
        https://dataverse.ird.fr/dataset.xhtml?persistentId=doi:10.23708/EWOYNK
    """
    url = {"ecoli_houay_pano_file.csv": "https://dataverse.ird.fr/api/access/datafile/9230"}

    path = os.path.join(os.path.dirname(__file__), 'data', 'ecoli_houay_pano')

    return _fetch_ecoli(path, overwrite, url, None, features, st, en,
                        "ecoli_houay_pano_tab_file")


def ecoli_mekong_laos(
        st: Union[str, pd.Timestamp, int] = "20110101",
        en: Union[str, pd.Timestamp, int] = "20211231",
        features:Union[str, list] = None,
        station_name:str = None,
        overwrite=False
)->pd.DataFrame:
    """
    E. coli data from Mekong river (Northern Laos).

    Parameters
    ----------
        st :
            starting time
        en :
            end time
        station_name : str
        features : str, optional
        overwrite : bool
            whether to overwrite or not

    Returns
    -------
    pd.DataFrame
        with default parameters, the shape is (1131, 10)

    Examples
    --------
        >>> from water_datasets import ecoli_mekong_laos
        >>> ecoli = ecoli_mekong_laos()
        >>> ecoli.shape
        (1131, 10)

    .. url_
        https://dataverse.ird.fr/file.xhtml?fileId=9229&version=3.0
    """
    url = {"ecoli_mekong_loas_file.csv": "https://dataverse.ird.fr/api/access/datafile/9229"}

    path = os.path.join(os.path.dirname(__file__), 'data', 'ecoli_mekong_loas')

    return _fetch_ecoli(path, overwrite, url, station_name, features, st, en,
                        _name="ecoli_mekong_laos_tab_file")


def _fetch_ecoli(path, overwrite, url, station_name, features, st, en, _name):

    if not os.path.exists(path):
        os.makedirs(path)

    if not os.path.exists(os.path.join(path, list(url.keys())[0])):
        maybe_download(path, overwrite=overwrite, url=url, name=_name)
    all_files = os.listdir(path)
    assert len(all_files)==1
    fname = os.path.join(path, all_files[0])
    df = pd.read_csv(fname, sep='\t')

    df.index = pd.to_datetime(df['Date_Time'])

    if station_name is not None:
        assert station_name in df['River'].unique().tolist()
        df = df.loc[df['River']==station_name]

    if features is None:
        features = ['River', 'T', 'EC', 'DOpercent', 'DO', 'pH', 'ORP', 'Turbidity',
                    'TSS', 'E-coli_4dilutions']

    features = check_attributes(features, df.columns.tolist())
    df = df[features]

    # River is not a representative name
    df = df.rename(columns={"River": "station_name"})

    df = df.sort_index()

    if st:
        if isinstance(en, int):
            assert isinstance(en, int)
            df = df.iloc[st:en]
        else:
            df = df.loc[st:en]

    return df
