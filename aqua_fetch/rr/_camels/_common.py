import os
import zlib
import shutil
import zipfile
import warnings
from typing import Tuple

import pandas as pd


def _row_date(row: bytes, sep: bytes) -> str:
    """the date in the first field of a row of a time series file"""
    return row[:row.find(sep)].strip(b' "').decode()


def _first_and_last_row(fpath: str) -> Tuple[bytes, bytes]:
    """first row after the header and last row of a text file"""
    with open(fpath, 'rb') as f:
        f.readline()  # header
        first = f.readline()
        f.seek(0, os.SEEK_END)
        f.seek(max(0, f.tell() - 2**16))  # a row is at most a few kB long
        last = [row for row in f.read().splitlines() if row.strip()][-1]
    return first, last


def _remove_stale(paths, verbosity: int = 1, reason: str = "overwrite=True"):
    """
    Deletes every existing file or folder in ``paths``. Called before an
    ``overwrite=True`` re-download so that nothing stale survives: ``download``
    would save a second archive as ``<name>.zip1`` which nothing reads, an
    existing extracted folder would not be re-extracted, and the base
    ``_maybe_to_netcdf`` would rebuild the netCDF cache by reading the old cache.
    """
    for stale in paths:
        if os.path.lexists(stale):
            if verbosity:
                print(f"{reason}: removing stale {stale}")
            # a symlink is unlinked; its target is left alone
            if os.path.isdir(stale) and not os.path.islink(stale):
                shutil.rmtree(stale)
            else:
                os.remove(stale)
    return

def _extract_zip(archive: str, folder: str, verbosity: int = 1):
    """
    Extracts ``archive`` into a temporary folder which is renamed to ``folder``
    once complete, so that an interrupted extraction is redone on the next
    initialization instead of being taken as complete. A corrupt archive, e.g.
    an error page saved under its name, is deleted so that it is downloaded
    again.
    """
    tmp = f"{folder}_extracting"
    shutil.rmtree(tmp, ignore_errors=True)  # left by an interrupted extraction

    if verbosity:
        print(f"extracting {archive} to {folder}")

    try:
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(tmp)
    except (zipfile.BadZipFile, zlib.error, EOFError):
        shutil.rmtree(tmp, ignore_errors=True)
        os.remove(archive)
        raise ValueError(f"{archive} is corrupt and was deleted. Initialize the "
                         f"dataset again to download it again.") from None

    shutil.rmtree(folder, ignore_errors=True)  # os.replace needs a free name
    os.replace(tmp, folder)
    return


def _warn_duplicate_gauges(dataset: str, meta: pd.DataFrame, decimals: int = 3):
    """
    Warns (regardless of ``verbosity``) if two gauges share the same name and
    coordinates rounded to ``decimals``. ``meta`` must have the columns
    ``gauge_id``, ``gauge_name``, ``gauge_lat`` and ``gauge_lon``. Duplicates
    are only reported, never excluded.
    """
    key = (meta['gauge_name'].astype(str) + '_' +
           meta['gauge_lat'].round(decimals).astype(str) + '_' +
           meta['gauge_lon'].round(decimals).astype(str))
    dup_mask = key.duplicated(keep=False)
    if dup_mask.any():
        dups = meta.loc[dup_mask, 'gauge_id'].tolist()
        warnings.warn(
            f"{dataset}: {int(dup_mask.sum())} gauges appear to be "
            f"duplicates (same name and coordinates): {dups}. They are "
            f"kept in the dataset.", UserWarning)
    return
