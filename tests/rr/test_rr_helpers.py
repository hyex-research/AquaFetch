"""
Tests for the pure helpers of ``aqua_fetch/rr/utils.py``.

They need no data and no network, so they run anywhere. Currently they cover
``ymd_index``, which builds the time index of the CAMELS-BR and LamaH readers
from integer year/month/day(/hour/minute) columns.

Run as a script or with pytest.
"""

import os
import site
import logging

import numpy as np
import pandas as pd

wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

if __name__ == "__main__":
    logging.basicConfig(filename='test_rr_helpers.log', filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from aqua_fetch.rr.utils import ymd_index


def test_ymd_index_equals_pandas():
    """every day of two centuries, and a leap day, come out exactly as pandas
    parses them"""
    logger.info("test_ymd_index_equals_pandas")
    days = pd.date_range('1899-01-01', '2101-12-31', freq='D')
    built = ymd_index(days.year.to_numpy(), days.month.to_numpy(), days.day.to_numpy())
    assert built.equals(days), (len(built), len(days))

    minutes = pd.date_range('2020-02-28 22:00', '2020-03-01 02:00', freq='min')
    built = ymd_index(minutes.year.to_numpy(), minutes.month.to_numpy(),
                      minutes.day.to_numpy(), minutes.hour.to_numpy(),
                      minutes.minute.to_numpy())
    assert built.equals(minutes), (built[[0, -1]], minutes[[0, -1]])
    return


def test_ymd_index_rejects_impossible_dates():
    """a corrupt row raises instead of being rolled into the next month or day,
    which is what pandas does and what a silent mis-dating would hide"""
    logger.info("test_ymd_index_rejects_impossible_dates")
    impossible = [
        (2025, 2, 31),      # February has no 31st
        (2023, 2, 29),      # 2023 is not a leap year
        (2025, 4, 31),
        (2025, 13, 1),      # month out of range
        (2025, 0, 1),
        (2025, 1, 0),       # day out of range
        (2025, 1, -3),
    ]
    for year, month, day in impossible:
        try:
            built = ymd_index(np.array([year]), np.array([month]), np.array([day]))
        except ValueError:
            continue
        raise AssertionError(f"{year}-{month}-{day} became {built[0]}")

    # the leap day of a leap year is a date
    assert ymd_index(np.array([2024]), np.array([2]), np.array([29]))[0] == \
        pd.Timestamp('2024-02-29')

    for hour, minute in ((24, 0), (25, 0), (-1, 0), (0, 60), (0, -1)):
        try:
            built = ymd_index(np.array([2025]), np.array([1]), np.array([1]),
                              np.array([hour]), np.array([minute]))
        except ValueError:
            continue
        raise AssertionError(f"{hour}:{minute} became {built[0]}")

    assert ymd_index(np.array([2025]), np.array([1]), np.array([1]),
                     np.array([23]), np.array([59]))[0] == \
        pd.Timestamp('2025-01-01 23:59')
    return


def test_ymd_index_accepts_series_and_lists():
    """the columns of a DataFrame and plain lists are accepted, as the readers
    pass them"""
    logger.info("test_ymd_index_accepts_series_and_lists")
    frame = pd.DataFrame({'year': [1980, 2024], 'month': [1, 12], 'day': [1, 31]})
    from_frame = ymd_index(frame['year'], frame['month'], frame['day'])
    from_lists = ymd_index([1980, 2024], [1, 12], [1, 31])
    expected = pd.DatetimeIndex(['1980-01-01', '2024-12-31'])
    assert from_frame.equals(expected) and from_lists.equals(expected)
    return


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith('test_') and callable(test):
            test()
            print(f"{name} passed")
    print("all tests passed")
