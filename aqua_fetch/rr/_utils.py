
__all__ = [
    "tw_resampler",
]

from datetime import timedelta

import pandas as pd
import numpy as np


def filterByTime(
        pdf:pd.Series,
        start:pd.Timestamp,
        end:pd.Timestamp
)->pd.Series:
    """
    Filter subset by datetime specified its header

    Parameters:
        pdf :
            source data to filter
        start :
            start date / time
        end :
        end  date/time

    return:
      pandas.Series object
    """
    return pdf.loc[start:end-timedelta(seconds=1)]


def interpolateByTime(
        pdf0:pd.Series,
        pdf1:pd.Series,
        freq:str = "H"
)->float:
    """
    Interpolate data at 00:00

    Parameters:
      pdf0: data befor 0:00
        Type: pandas.DataFrame object
      pdf1: data after 0:00
        Type: pandas.DataFrame object
      freq
    return:
      value with a type of float
    """
    replace = dict(minute=0, second=0)
    if freq=="D":
        replace = dict(hour=0, minute=0)
    v0, t0 = pdf0.iloc[-1], pdf0.index[-1]
    v1, t1 = pdf1.iloc[0], pdf1.index[0]

    t =  t1.replace(**replace)

    dt0 = (t - t0).total_seconds()/3600
    dt1 = (t1 - t).total_seconds()/3600
    v = ( v0 * dt1 + v1 * dt0 ) / ( dt0 + dt1 )
    return np.float32(v)


def prepend(historical_data, subdata, freq:str = "H"):
    """
    Add first row at 0:00. When length of historical_data greater than 1, 
    interpolate it. Or, use first row of subdata.

    Parameters:
        historical_data: data befor 0:00
            Type: pandas.DataFrame object
        subdata: data after 0:00
            Type: pandas.DataFrame object
        freq

    return:
      pandas.DataFrame object
    """
    first = subdata.iloc[:1].copy()
    if freq == "H":
        # e.g 2024-01-01 04:10 -> 2023-01-01 04:00
        first.index = first.index.where([0], first.index[0].replace(minute=0, second=0))
    else:
        # e.g. 2024-01-02 08:00:00 -> 2024-01-02 00:00:00
        first.index = first.index.where([0], first.index[0].replace(hour=0, minute=0, second=0))
    if len(historical_data) > 1:
        first.loc[first.index[0]] = interpolateByTime(historical_data, subdata, freq=freq)
    pdf = pd.concat([subdata, first]).sort_index()
    return pdf


def append(
        subdata:pd.Series,
        future_data:pd.Series,
        freq:str="H"):
    """
    Add last row at 0:00. When length of pdf1 greater than 1, interpolate it.
    Or, use last row of pdf0.

    Parameters:
        subdata: data befor for the current step (day, hour)
            Type: pandas.DataFrame object
        future_data: data after 0:00
            Type: pandas.DataFrame object
        freq :
    return:
      pandas.DataFrame object
    """
    last = subdata.iloc[-1:].copy()
    if freq == "H":
        # e.g 2024-01-01 02:45 -> 2023-01-01 03:00
        last.index = last.index.where([0],
                                      last.index[0].replace(
                                          minute=0, second=0) + timedelta(hours=1))
    else:
        last.index = last.index.where([0],
                                      last.index[0].replace(
                                          hour=0, minute=0, second=0) + timedelta(days=1))
    if len(future_data) > 1:
        last.loc[last.index[0]] = interpolateByTime(subdata, future_data, freq=freq)
    pdf = pd.concat([subdata, last]).sort_index()
    return pdf


def weightsCalculator(index, freq:str="H"):
    """
    Calculate weights of time for variable(water level or flow). Actuall, they
    have a unit of hour.

    Parameters:
        index: time corresponding to the variable data
          Type: list of datetime or numpy array of datetime
        freq :
    return:
      numpy array in float/int with a unit of hour
    """
    _format = 'm'
    total = 120
    if freq=="D":
        _format = 'h'
        total = 48

    dt = np.array(index[1:]) - np.array(index[:-1])
    dt = [i/np.timedelta64(1, _format) for i in dt]	# convert timedelta object to number of hours
    weights = np.array([0] + dt) + np.array(dt + [0])
    assert int(round(weights.sum())) == total, int(round(weights.sum()))
    return weights


def tw_resampler(
        subdata:pd.Series,
        whole_data:pd.Series,
        freq:str = "H",
)->float:
    """
    resampler for time weighted average. Resamples from sub-daily to daily or
    sub-hourly to hourly considering time weighted average instead of simple average.

    Parameters
    ----------
    subdata : pd.Series
        The data for the current time step (day/hour). The index of subdata must be
        pandas DatetimeIndex.
    whole_data : pd.Series
        is used to get data for historical and next step for interpolation at start and end
        of current step if the data at start and end of current step is not available.
    freq : str
        must be either ``H`` or ``D``

    index = pd.to_datetime([
    '2024-01-01 00:00:00', '2024-01-01 08:00:00',
    '2024-01-01 02:00:00', '2024-01-01 08:20:00',
    '2024-01-01 02:45:00', '2024-01-01 09:45:00',
    '2024-01-01 04:10:00', '2024-01-01 09:50:00',
    '2024-01-01 04:20:00', '2024-01-01 11:10:00',
    '2024-01-01 04:35:00', '2024-01-01 11:15:00',
    '2024-01-01 04:45:00', '2024-01-01 11:45:00',
    '2024-01-01 04:55:00', '2024-01-01 12:15:00',
    '2024-01-01 05:15:00', '2024-01-01 12:25:00',
    '2024-01-01 06:15:00', '2024-01-01 12:35:00',

    '2024-01-01 13:00:00', '2024-01-01 19:00:00',
    '2024-01-01 13:10:00', '2024-01-01 19:10:00',
    '2024-01-01 13:45:00', '2024-01-01 19:45:00',
    '2024-01-01 14:30:00', '2024-01-01 19:50:00',
    '2024-01-01 14:50:00', '2024-01-01 21:30:00',
    '2024-01-01 15:15:00', '2024-01-01 21:45:00',
    '2024-01-01 15:35:00', '2024-01-01 22:15:00',
    '2024-01-01 17:15:00', '2024-01-01 22:55:00',
    '2024-01-01 17:16:00', '2024-01-01 23:17:00',
    '2024-01-01 17:17:00', '2024-01-01 23:19:00',
        ])

    df = pd.DataFrame(
    np.arange(len(index)),
    index=index,
    columns=['value']
    ).astype(np.float32)    
    val = df['value'].resample('H').apply(lambda subdata: tw_resampler(subdata, df['value'].sort_index()))

    np.testing.assert_array_almost_equal([val.sum()], [327.425], decimal=4)
    # resampling from sub-daily to daily time-step

    index = pd.to_datetime([
    '2024-01-01 00:00:00', '2024-01-02 08:00:00',
    '2024-01-01 02:00:00', '2024-01-02 08:20:00',
    '2024-01-01 02:45:00', '2024-01-02 09:45:00',
    '2024-01-01 04:10:00', '2024-01-02 09:50:00',
    '2024-01-01 04:20:00', '2024-01-02 11:10:00',
    '2024-01-01 04:35:00', '2024-01-02 11:15:00',
    '2024-01-01 04:45:00', '2024-01-02 11:45:00',
    '2024-01-01 04:55:00', '2024-01-02 12:15:00',
    '2024-01-01 05:15:00', '2024-01-02 12:25:00',
    '2024-01-01 06:15:00', '2024-01-02 12:35:00',

    '2024-01-03 13:00:00', '2024-01-04 19:00:00',
    '2024-01-03 13:10:00', '2024-01-04 19:10:00',
    '2024-01-03 13:45:00', '2024-01-04 19:45:00',
    '2024-01-03 14:30:00', '2024-01-05 19:50:00',
    '2024-01-03 14:50:00', '2024-01-07 21:30:00',
    '2024-01-03 15:15:00', '2024-01-08 21:45:00',
    '2024-01-03 15:35:00', '2024-01-08 22:15:00',
    '2024-01-03 17:15:00', '2024-01-08 22:55:00',
    '2024-01-03 17:16:00', '2024-01-08 23:17:00',
    '2024-01-03 17:17:00', '2024-01-08 23:19:00',
        ])

    df = pd.DataFrame(
    np.arange(len(index)),
    index=index,
    columns=['value']
    ).astype(np.float32)
    df['value'].resample('D').apply(lambda subdata: tw_resampler(subdata, df['value'].sort_index(), 'D'))

    """

    REPLACE= {
        'D': {'hour': 0, 'minute': 0, 'second': 0},
        'H': {'minute': 0, 'second': 0},
        'h': {'minute': 0, 'second': 0}
    }

    DELTA = {
        "D": {"d0": dict(days=1), "d3": dict(days=2)},
        "H": {"d0": dict(hours=1), "d3": dict(hours=2)},
        "h": {"d0": dict(hours=1), "d3": dict(hours=2)}
    }

    # case 0: do nothing
    if len(subdata) == 0: 
        return np.nan

    # case 1: use it as the daily average if only one gauged data
    if len(subdata) == 1: 
        record = round(subdata.iloc[0], 4)
    else:
        d1 = subdata.index[0].replace(**REPLACE[freq]) 	# start of the current step (d1)
        d0 = d1 -  timedelta(**DELTA[freq]['d0'])	# start of the previous step (d0)
        d2 = d1 +  timedelta(**DELTA[freq]['d0'])	# start of the next step (d2)
        d3 = d1 +  timedelta(**DELTA[freq]['d3'])	# start of the next two steps (d3)
        # first index is not starts with 00:00
        if subdata.index[0] != d1:	
            historical_data = whole_data.loc[d0:d1-timedelta(seconds=1)].dropna()#.sort_index()
            subdata = prepend(historical_data, subdata, freq=freq)

        # last index is not 00:00
        if subdata.index[-1] != d2:	
            future_data = whole_data.loc[d2:d3-timedelta(seconds=1)].dropna()#.sort_index()
            subdata = append(subdata, future_data, freq=freq)
      
        # weights
        weights = weightsCalculator(subdata.index.to_numpy(), freq=freq)
      
        # daily average
        record = np.average(subdata.values, weights=weights)

    return record
