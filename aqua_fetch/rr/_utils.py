
__all__ = [
    "tw_resampler",
    "tw_resampler_",
]

import warnings
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
    if freq.upper() == "H":
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
    if freq.upper() == "H":
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


def tw_resampler_(
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
    val = df['value'].resample('H').apply(lambda subdata: tw_resampler_(subdata, df['value'].sort_index()))

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
    df['value'].resample('D').apply(lambda subdata: tw_resampler_(subdata, df['value'].sort_index(), 'D'))

    """

    # normalise case so 'h'/'d' behave exactly like 'H'/'D' throughout
    # (prepend/append/weightsCalculator and the dicts below).
    freq = freq.upper()

    REPLACE = {
        'D': {'hour': 0, 'minute': 0, 'second': 0},
        'H': {'minute': 0, 'second': 0},
    }

    DELTA = {
        "D": {"d0": dict(days=1), "d3": dict(days=2)},
        "H": {"d0": dict(hours=1), "d3": dict(hours=2)},
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


def _tw_reference(series: pd.Series, freq_norm: str) -> pd.Series:
    """Exact per-bucket reference call pattern.

    Used by :func:`tw_resampler` to delegate the few input shapes its
    vectorised fast path cannot reproduce faithfully (sub-minute timestamps,
    a DST transition inside a tz-aware range, and unsorted input carrying
    duplicate timestamps). ``freq_norm`` must already be upper-cased.
    """
    rfreq = "h" if freq_norm == "H" else "D"
    return series.resample(rfreq).apply(
        lambda s: tw_resampler_(s, series.sort_index(), freq_norm))


def tw_resampler(
        series: pd.Series,
        freq: str = "H",
) -> pd.Series:
    """
    Vectorised trapezoidal time-weighted resampling of an irregularly-sampled
    series to a regular hourly or daily grid.

    Fast drop-in replacement for the per-bucket call pattern::

        series.resample(freq).apply(
            lambda s: tw_resampler_(s, series.sort_index(), freq))

    operating on the whole series in a single vectorised pass instead of one
    Python call per bucket. For the timestamp resolutions that occur in real
    hydrological data (15-minute and coarser, i.e. timestamps aligned to whole
    minutes) the output is identical to :func:`tw_resampler_` up to floating
    point summation order.

    Parameters
    ----------
    series : pd.Series
        Source values indexed by a (tz-naive or tz-aware) pandas DatetimeIndex.
        Need not be sorted; it is sorted internally.
    freq : {"H", "h", "D", "d"}
        Target resampling frequency (hourly or daily).

    Returns
    -------
    pd.Series
        Resampled series indexed by bucket starts, dtype float64. The name of
        the input series is preserved.

    Notes
    -----
    Semantics reproduced from :func:`tw_resampler_`, for each bucket ``[d1, d2)``:

    * zero observations -> ``NaN``.
    * one observation   -> that value rounded to 4 decimals (using the same
      :func:`round` and dtype as the source so the result is bit-identical).
    * a bucket that contains **any** ``NaN`` value -> ``NaN`` (a single missing
      value poisons the whole bucket, exactly as ``np.average`` does).
    * two or more values -> the bucket is padded with a synthetic left/right
      boundary point at ``d1``/``d2``. The boundary is linearly interpolated
      from the adjacent bucket when that neighbour holds 2+ non-NaN
      observations, otherwise it is constant-extrapolated from the first/last
      in-bucket value. Interpolated boundary values are truncated to
      ``float32`` to reproduce :func:`interpolateByTime` exactly. The result is
      the trapezoidal integral over ``[d1, d2]`` divided by ``(d2 - d1)``.

    Three input shapes are delegated to the exact per-bucket
    :func:`tw_resampler_` (correct, only slower) instead of being handled by the
    fast path:

    * **Sub-minute timestamps** — ``tw_resampler_`` derives bucket edges from
      ``index[0].replace(...)``, keeping the sub-minute part; that quirk only
      matters below minute resolution, which never occurs in the source data.
      Supplying it warns and delegates.
    * **A DST transition inside a tz-aware range** — the local wall-clock grid
      would then contain nonexistent/ambiguous hours and variable-length days.
    * **Duplicate timestamps** — the reference reorders equal-timestamp values
      through an unstable ``sort_index()`` (in prepend/append), so its result is
      order-dependent and has no single well-defined value; the only faithful
      answer is the reference's own. (Real per-timestamp-unique data never hits
      this.)

    For a ``float64`` input (every real dataset, read via pandas) the output is
    identical to :func:`tw_resampler_` up to float summation order (~1e-13). For
    a ``float32`` input the interpolated boundary values agree only to float32
    precision (~1e-6 relative): under numpy's NEP-50 rules the original
    evaluates ``interpolateByTime`` in float32, whereas this function widens to
    float64 before the final float32 truncation.

    Multiprocessing is intentionally not used: the work is a single dependent
    numpy pass and pickling the array to worker processes would be slower than
    running it in-process. Parallelism belongs one level up, across stations.
    """
    if not isinstance(freq, str) or freq.upper() not in ("H", "D"):
        raise ValueError(f"freq must be 'H'/'h' or 'D'/'d', got {freq!r}")
    freq_norm = freq.upper()
    step_ns = (3600 if freq_norm == "H" else 86400) * 1_000_000_000

    name = series.name
    orig_tz = series.index.tz

    if series.empty:
        empty_idx = pd.DatetimeIndex([])
        if orig_tz is not None:
            empty_idx = empty_idx.tz_localize(orig_tz)
        if hasattr(series.index, "unit"):          # pandas >= 2.0
            empty_idx = empty_idx.as_unit(series.index.unit)
        return pd.Series([], index=empty_idx, name=name, dtype=np.float64)

    if orig_tz is not None:
        # Bin on local wall-clock time, matching a tz-aware resample. Whether a
        # DST transition breaks the regular grid is checked below on the grid
        # itself (not on the observations, which may all sit on one side of it).
        times_ns = (series.index.tz_localize(None)
                    .values.astype("datetime64[ns]").astype(np.int64))
    else:
        times_ns = series.index.values.astype("datetime64[ns]").astype(np.int64)

    # The fast path is exact only for timestamps aligned to whole minutes
    # (the finest resolution real data ever has is 15-min). For anything finer
    # the boundary edges that tw_resampler_ derives via ``.replace()`` diverge,
    # so fall back to the exact per-bucket implementation instead of silently
    # returning a slightly different answer.
    if (times_ns % (60 * 1_000_000_000) != 0).any():
        warnings.warn(
            "tw_resampler received sub-minute timestamps; falling back to "
            "the slower per-bucket tw_resampler_ to preserve exact output.",
            stacklevel=2,
        )
        return _tw_reference(series, freq_norm)

    orig_vals = np.asarray(series.values)
    vals = orig_vals.astype(np.float64)

    order = np.argsort(times_ns, kind="stable")
    times_ns = times_ns[order]
    # tw_resampler_ reorders equal-timestamp values through an *unstable*
    # ``sort_index()`` (inside prepend/append and its whole_data lookup), so its
    # result is order-dependent whenever timestamps are duplicated -- there is
    # no single well-defined answer to match. Delegate so the output reproduces
    # whatever the reference produces for such (data-quality-anomalous) input.
    if times_ns.size > 1 and bool((np.diff(times_ns) == 0).any()):
        return _tw_reference(series, freq_norm)
    vals = vals[order]
    orig_vals = orig_vals[order]

    bucket_floor = times_ns // step_ns
    b_min = int(bucket_floor[0])
    b_max = int(bucket_floor[-1])
    n_buckets = b_max - b_min + 1
    bucket_off = (bucket_floor - b_min).astype(np.int64)

    if orig_tz is not None:
        # Localize the wall-clock grid to detect any DST/offset transition that
        # would break the fixed-step assumption: a nonexistent/ambiguous grid
        # edge, or a bucket whose real (UTC) length is not exactly one step,
        # means pandas would bin differently and we must delegate. The grid is
        # extended back to the first bucket's local midnight -- pandas resample
        # anchors bins at the first day's midnight (origin='start_day'), so a
        # *fractional*-hour transition between that anchor and b_min shifts every
        # bin edge off local ':00' without touching the [b_min, b_max] span.
        buckets_per_day = 86400_000_000_000 // step_ns
        grid_start = (b_min // buckets_per_day) * buckets_per_day
        edge_ns = np.arange(grid_start, b_max + 2, dtype=np.int64) * step_ns
        edges = pd.DatetimeIndex(edge_ns.astype("datetime64[ns]")).tz_localize(
            orig_tz, ambiguous="NaT", nonexistent="NaT")
        if edges.isna().any() or (np.diff(edges.asi8) != step_ns).any():
            return _tw_reference(series, freq_norm)
        head = b_min - grid_start
        out_index = edges[head:head + n_buckets]
    else:
        out_starts_ns = (b_min + np.arange(n_buckets, dtype=np.int64)) * step_ns
        out_index = pd.DatetimeIndex(out_starts_ns.astype("datetime64[ns]"))

    # Reproduce the reference index metadata so it is an exact drop-in: preserve
    # the input's datetime unit (so ``.equals()`` holds for coarser-than-ns
    # input; a no-op on pandas < 2.0 where every datetime is ns) and stamp the
    # resample frequency that ``.resample()`` sets.
    if hasattr(series.index, "unit"):              # pandas >= 2.0
        out_index = out_index.as_unit(series.index.unit)
    out_index.freq = (pd.tseries.offsets.Hour() if freq_norm == "H"
                      else pd.tseries.offsets.Day())

    occupied_off, first_in_bucket = np.unique(bucket_off, return_index=True)
    last_in_bucket = np.empty_like(first_in_bucket)
    last_in_bucket[:-1] = first_in_bucket[1:] - 1
    last_in_bucket[-1] = len(times_ns) - 1
    bucket_size = last_in_bucket - first_in_bucket + 1

    out_values = np.full(n_buckets, np.nan, dtype=np.float64)

    # --- buckets with exactly one observation -----------------------------
    # tw_resampler_ returns ``round(value, 4)``; reproduce it with Python's
    # round() on the original-dtype scalar so the result is bit-identical
    # (np.round uses a different algorithm at exact half-way values).
    single = bucket_size == 1
    if single.any():
        bk = occupied_off[single]
        sv = orig_vals[first_in_bucket[single]]
        out_values[bk] = [
            np.nan if v != v else round(v, 4) for v in sv
        ]

    multi = bucket_size >= 2
    if not multi.any():
        return pd.Series(out_values, index=out_index, name=name)

    m_off = occupied_off[multi]
    m_first = first_in_bucket[multi]
    m_last = last_in_bucket[multi]

    t_first = times_ns[m_first]
    v_first = vals[m_first]
    t_last = times_ns[m_last]
    v_last = vals[m_last]

    d1 = (b_min + m_off) * step_ns
    d2 = d1 + step_ns

    need_left = t_first != d1
    need_right = t_last != d2

    # per-bucket non-NaN bookkeeping (dropna() in tw_resampler_'s boundary logic)
    not_nan = ~np.isnan(vals)
    nonnan_count = np.zeros(n_buckets, dtype=np.int64)
    if not_nan.any():
        np.add.at(nonnan_count, bucket_off[not_nan], 1)

    first_nn_per_bucket = np.full(n_buckets, -1, dtype=np.int64)
    last_nn_per_bucket = np.full(n_buckets, -1, dtype=np.int64)
    nn_obs = np.where(not_nan)[0]
    if nn_obs.size:
        bo_nn = bucket_off[nn_obs]
        run_end = np.empty(bo_nn.size, dtype=bool)
        run_end[:-1] = bo_nn[:-1] != bo_nn[1:]
        run_end[-1] = True
        run_start = np.empty(bo_nn.size, dtype=bool)
        run_start[0] = True
        run_start[1:] = bo_nn[1:] != bo_nn[:-1]
        last_nn_per_bucket[bo_nn[run_end]] = nn_obs[run_end]
        first_nn_per_bucket[bo_nn[run_start]] = nn_obs[run_start]

    # boundary values: default is constant-extrapolation (first/last in-bucket
    # value); overwrite with the float32 linear interpolation where the
    # neighbouring bucket has 2+ non-NaN observations.
    left_v = v_first.copy()
    right_v = v_last.copy()

    prev_off = m_off - 1
    valid_prev_mask = (prev_off >= 0) & need_left
    if valid_prev_mask.any():
        candidates = np.where(valid_prev_mask)[0]
        eligible = nonnan_count[prev_off[candidates]] >= 2
        do = candidates[eligible]
        if do.size:
            hist_idx = last_nn_per_bucket[prev_off[do]]
            v_hist = vals[hist_idx]
            v_cur = v_first[do]
            # dt in hours from exact integer-ns differences, matching
            # interpolateByTime's ``(t - t0).total_seconds() / 3600`` so the
            # subsequent float32 truncation is bit-identical.
            dt_h = (d1[do] - times_ns[hist_idx]) / 1e9 / 3600
            dt_c = (t_first[do] - d1[do]) / 1e9 / 3600
            left_v[do] = np.float32((v_hist * dt_c + v_cur * dt_h) / (dt_h + dt_c))

    next_off = m_off + 1
    valid_next_mask = (next_off < n_buckets) & need_right
    if valid_next_mask.any():
        candidates = np.where(valid_next_mask)[0]
        eligible = nonnan_count[next_off[candidates]] >= 2
        do = candidates[eligible]
        if do.size:
            fut_idx = first_nn_per_bucket[next_off[do]]
            v_fut = vals[fut_idx]
            v_cur = v_last[do]
            # dt in hours from exact integer-ns differences (see left boundary).
            dt_c = (d2[do] - t_last[do]) / 1e9 / 3600
            dt_f = (times_ns[fut_idx] - d2[do]) / 1e9 / 3600
            right_v[do] = np.float32((v_cur * dt_f + v_fut * dt_c) / (dt_c + dt_f))

    # trapezoidal integral over [d1, d2] == 0.5 * sum (v_i + v_{i+1}) * dt
    integral = np.zeros(n_buckets, dtype=np.float64)

    with np.errstate(invalid="ignore"):
        if len(times_ns) > 1:
            dt_pair = (times_ns[1:] - times_ns[:-1]).astype(np.float64)
            v_pair = vals[1:] + vals[:-1]
            same = bucket_off[1:] == bucket_off[:-1]
            contrib = np.where(same, dt_pair * v_pair * 0.5, 0.0)
            np.add.at(integral, bucket_off[:-1], contrib)

        dt_left = (t_first - d1).astype(np.float64)
        contrib_left = np.where(need_left, dt_left * (left_v + v_first) * 0.5, 0.0)
        np.add.at(integral, m_off, contrib_left)

        dt_right = (d2 - t_last).astype(np.float64)
        contrib_right = np.where(need_right, dt_right * (v_last + right_v) * 0.5, 0.0)
        np.add.at(integral, m_off, contrib_right)

        out_values[m_off] = integral[m_off] / step_ns

    # a NaN anywhere in a multi-observation bucket poisons it (np.average would
    # return NaN); enforce it explicitly so the answer never depends on NaN
    # arithmetic happening to propagate.
    bucket_has_nan = np.zeros(n_buckets, dtype=bool)
    nan_mask = ~not_nan
    if nan_mask.any():
        bucket_has_nan[bucket_off[nan_mask]] = True
        dirty = m_off[bucket_has_nan[m_off]]
        out_values[dirty] = np.nan

    return pd.Series(out_values, index=out_index, name=name)
