"""
Parity + speed tests for the vectorised :func:`tw_resampler` against the
reference per-bucket :func:`tw_resampler_`.

The reference is the exact call pattern used in production
(``_estreams.py`` / ``_usgs.py``)::

    series.resample(freq).apply(lambda s: tw_resampler_(s, series.sort_index(), freq))

with an *uppercase* ``freq`` ("H"/"D"), which is how every real caller invokes
it. Real hydrological data is timestamped at whole minutes (15-min or coarser),
so all randomized data here is minute/15-min aligned. For that regime the two
implementations agree to float summation order for float64 input, and to
float32 precision for float32 input (see the module docstring of
``tw_resampler`` for why).

Runs stand-alone (``python tests/rr/test_tw_resampler_parity.py``) or under
pytest. No network or file I/O is touched.
"""
import os
import site
import time
import warnings

import numpy as np
import pandas as pd

wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

from aqua_fetch.rr._utils import tw_resampler_, tw_resampler  # noqa: E402

# realistic timestamp alignments (nanoseconds)
MIN = 60 * 10 ** 9
Q15 = 15 * MIN
SEC = 10 ** 9

# tolerances (justified empirically in the accompanying analysis)
F64_RTOL, F64_ATOL = 1e-9, 1e-8        # float64: bit-exact bar summation order
F32_RTOL, F32_ATOL = 1e-5, 1e-5        # float32: float32 precision at boundaries


def run_old(series: pd.Series, freq: str) -> pd.Series:
    """Reference: the exact production call pattern (uppercase freq)."""
    freq = freq.upper()
    rfreq = "h" if freq == "H" else "D"
    return series.resample(rfreq).apply(
        lambda s: tw_resampler_(s, series.sort_index(), freq)
    )


def assert_parity(new: pd.Series, old: pd.Series, label: str,
                  rtol: float = F64_RTOL, atol: float = F64_ATOL) -> None:
    """Index + value parity with explicit NaN handling; raises on mismatch."""
    assert new.index.equals(old.index), (
        f"{label}: index mismatch (len new={len(new)} old={len(old)})"
    )
    new_v = new.values.astype(np.float64)
    old_v = old.values.astype(np.float64)
    both_nan = np.isnan(new_v) & np.isnan(old_v)
    one_nan = np.isnan(new_v) ^ np.isnan(old_v)
    assert not one_nan.any(), (
        f"{label}: NaN disagreement at {list(new.index[np.where(one_nan)[0][:5]])}"
    )
    close = np.isclose(new_v, old_v, rtol=rtol, atol=atol, equal_nan=False)
    close |= both_nan
    if not close.all():
        bad = np.where(~close)[0]
        rows = "\n".join(
            f"  [{new.index[i]}] old={old_v[i]!r} new={new_v[i]!r} "
            f"diff={abs(old_v[i] - new_v[i]):.3e}"
            for i in bad[:10]
        )
        raise AssertionError(f"{label}: {bad.size} value mismatches\n{rows}")


def gen(seed: int, n: int, freq: str, span_days: int, align_ns: int = Q15,
        nan_rate: float = 0.0, dtype=np.float64, scale: float = 100.0):
    """Randomized, timestamp-aligned test series (deduplicated, sorted)."""
    rng = np.random.default_rng(seed)
    start = np.datetime64("2010-01-01", "s").astype("datetime64[ns]").astype(np.int64)
    span_ns = span_days * 86400 * 10 ** 9
    off = rng.integers(0, span_ns // align_ns, size=n, dtype=np.int64) * align_ns
    times = (start + off).astype("datetime64[ns]")
    vals = (rng.standard_normal(n) * scale).astype(dtype)
    if nan_rate > 0:
        vals[rng.random(n) < nan_rate] = np.nan
    s = pd.Series(vals, index=pd.DatetimeIndex(times))
    return s[~s.index.duplicated(keep="first")].sort_index()


# ---------------------------------------------------------------------------
# docstring examples
# ---------------------------------------------------------------------------
_HOURLY_IDX = pd.to_datetime([
    "2024-01-01 00:00:00", "2024-01-01 08:00:00", "2024-01-01 02:00:00",
    "2024-01-01 08:20:00", "2024-01-01 02:45:00", "2024-01-01 09:45:00",
    "2024-01-01 04:10:00", "2024-01-01 09:50:00", "2024-01-01 04:20:00",
    "2024-01-01 11:10:00", "2024-01-01 04:35:00", "2024-01-01 11:15:00",
    "2024-01-01 04:45:00", "2024-01-01 11:45:00", "2024-01-01 04:55:00",
    "2024-01-01 12:15:00", "2024-01-01 05:15:00", "2024-01-01 12:25:00",
    "2024-01-01 06:15:00", "2024-01-01 12:35:00", "2024-01-01 13:00:00",
    "2024-01-01 19:00:00", "2024-01-01 13:10:00", "2024-01-01 19:10:00",
    "2024-01-01 13:45:00", "2024-01-01 19:45:00", "2024-01-01 14:30:00",
    "2024-01-01 19:50:00", "2024-01-01 14:50:00", "2024-01-01 21:30:00",
    "2024-01-01 15:15:00", "2024-01-01 21:45:00", "2024-01-01 15:35:00",
    "2024-01-01 22:15:00", "2024-01-01 17:15:00", "2024-01-01 22:55:00",
    "2024-01-01 17:16:00", "2024-01-01 23:17:00", "2024-01-01 17:17:00",
    "2024-01-01 23:19:00",
])
_DAILY_IDX = pd.to_datetime([
    "2024-01-01 00:00:00", "2024-01-02 08:00:00", "2024-01-01 02:00:00",
    "2024-01-02 08:20:00", "2024-01-01 02:45:00", "2024-01-02 09:45:00",
    "2024-01-01 04:10:00", "2024-01-02 09:50:00", "2024-01-01 04:20:00",
    "2024-01-02 11:10:00", "2024-01-01 04:35:00", "2024-01-02 11:15:00",
    "2024-01-01 04:45:00", "2024-01-02 11:45:00", "2024-01-01 04:55:00",
    "2024-01-02 12:15:00", "2024-01-01 05:15:00", "2024-01-02 12:25:00",
    "2024-01-01 06:15:00", "2024-01-02 12:35:00", "2024-01-03 13:00:00",
    "2024-01-04 19:00:00", "2024-01-03 13:10:00", "2024-01-04 19:10:00",
    "2024-01-03 13:45:00", "2024-01-04 19:45:00", "2024-01-03 14:30:00",
    "2024-01-05 19:50:00", "2024-01-03 14:50:00", "2024-01-07 21:30:00",
    "2024-01-03 15:15:00", "2024-01-08 21:45:00", "2024-01-03 15:35:00",
    "2024-01-08 22:15:00", "2024-01-03 17:15:00", "2024-01-08 22:55:00",
    "2024-01-03 17:16:00", "2024-01-08 23:17:00", "2024-01-03 17:17:00",
    "2024-01-08 23:19:00",
])


def test_docstring_hourly():
    s = pd.Series(np.arange(len(_HOURLY_IDX)), index=_HOURLY_IDX).astype(np.float32)
    new = tw_resampler(s, "H")
    assert_parity(new, run_old(s, "H"), "docstring hourly", F32_RTOL, F32_ATOL)
    np.testing.assert_array_almost_equal([new.sum()], [327.425], decimal=4)


def test_docstring_daily():
    s = pd.Series(np.arange(len(_DAILY_IDX)), index=_DAILY_IDX).astype(np.float32)
    new = tw_resampler(s, "D")
    assert_parity(new, run_old(s, "D"), "docstring daily", F32_RTOL, F32_ATOL)


# ---------------------------------------------------------------------------
# randomized parity (float64 = production dtype -> tight tolerance)
# ---------------------------------------------------------------------------
def test_random_hourly_float64():
    for seed in range(8):
        for align in (Q15, MIN):
            s = gen(seed, 4000, "H", 40, align_ns=align)
            assert_parity(tw_resampler(s, "H"), run_old(s, "H"),
                          f"H f64 seed={seed} align={align}")


def test_random_daily_float64():
    for seed in range(8):
        for align in (Q15, MIN):
            s = gen(seed, 6000, "D", 400, align_ns=align)
            assert_parity(tw_resampler(s, "D"), run_old(s, "D"),
                          f"D f64 seed={seed} align={align}")


def test_random_with_nan_float64():
    for seed in range(6):
        for freq, span in (("H", 40), ("D", 400)):
            s = gen(seed, 4000, freq, span, nan_rate=0.15)
            assert_parity(tw_resampler(s, freq), run_old(s, freq),
                          f"{freq} nan f64 seed={seed}")


def test_random_large_and_small_scale_float64():
    for scale in (1e-3, 1.0, 1e5):
        s = gen(3, 4000, "H", 40, scale=scale)
        assert_parity(tw_resampler(s, "H"), run_old(s, "H"),
                      f"H f64 scale={scale}")


def test_sparse_single_obs_buckets():
    # many buckets hold exactly one observation -> exercises round() path
    for seed in range(4):
        s = gen(seed, 250, "H", 80)
        new, old = tw_resampler(s, "H"), run_old(s, "H")
        assert (old.notna().sum() > 0)
        assert_parity(new, old, f"sparse H seed={seed}")


def test_random_float32():
    for seed in range(6):
        for freq, span in (("H", 40), ("D", 400)):
            s = gen(seed, 4000, freq, span, dtype=np.float32)
            assert_parity(tw_resampler(s, freq), run_old(s, freq),
                          f"{freq} f32 seed={seed}", F32_RTOL, F32_ATOL)


# ---------------------------------------------------------------------------
# NaN semantics (hand-built, deterministic)
# ---------------------------------------------------------------------------
def test_nan_poisons_multi_obs_bucket():
    idx = pd.to_datetime([
        "2024-01-01 00:00", "2024-01-01 00:20", "2024-01-01 00:40",  # bucket 0
        "2024-01-01 01:15", "2024-01-01 01:45",                        # bucket 1
        "2024-01-01 02:30",                                            # bucket 2 (single)
    ])
    vals = np.array([1.0, np.nan, 3.0, 5.0, 7.0, np.nan])
    s = pd.Series(vals, index=idx)
    new = tw_resampler(s, "H")
    old = run_old(s, "H")
    assert_parity(new, old, "nan poisoning")
    # explicit expectations: bucket 0 poisoned by interior NaN, bucket 2 single-NaN
    assert np.isnan(new.loc["2024-01-01 00:00"])
    assert np.isnan(new.loc["2024-01-01 02:00"])
    assert not np.isnan(new.loc["2024-01-01 01:00"])


def test_nan_neighbor_does_not_leak():
    # a NaN in bucket B must not poison clean neighbour A
    idx = pd.to_datetime([
        "2024-01-01 00:10", "2024-01-01 00:50",   # bucket 0 clean
        "2024-01-01 01:10", "2024-01-01 01:50",   # bucket 1 clean
    ])
    s = pd.Series([1.0, 2.0, np.nan, 4.0], index=idx)
    new = tw_resampler(s, "H")
    old = run_old(s, "H")
    assert_parity(new, old, "nan no-leak")
    assert not np.isnan(new.loc["2024-01-01 00:00"])   # bucket 0 stays clean
    assert np.isnan(new.loc["2024-01-01 01:00"])       # bucket 1 poisoned


# ---------------------------------------------------------------------------
# boundary-interpolation eligibility (neighbour needs >= 2 non-NaN)
# ---------------------------------------------------------------------------
def test_boundary_eligibility():
    # prev bucket has a single obs -> constant extrapolation, not interpolation
    idx = pd.to_datetime([
        "2024-01-01 00:30",                        # prev bucket: 1 obs
        "2024-01-01 01:15", "2024-01-01 01:45",    # target bucket: 2 obs
        "2024-01-01 02:10", "2024-01-01 02:50",    # next bucket: 2 obs
    ])
    s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=idx)
    assert_parity(tw_resampler(s, "H"), run_old(s, "H"), "eligibility")


# ---------------------------------------------------------------------------
# edge cases
# ---------------------------------------------------------------------------
def test_empty_series():
    s = pd.Series([], index=pd.DatetimeIndex([]), dtype=np.float64)
    out = tw_resampler(s, "H")
    assert out.empty and out.dtype == np.float64


def test_single_observation_total():
    s = pd.Series([3.14159], index=pd.to_datetime(["2024-01-01 00:30"]))
    new = tw_resampler(s, "H")
    old = run_old(s, "H")
    assert_parity(new, old, "single total obs")
    assert len(new) == 1 and abs(new.iloc[0] - round(3.14159, 4)) < 1e-12


def test_all_nan_series():
    idx = pd.to_datetime(["2024-01-01 00:10", "2024-01-01 00:50",
                          "2024-01-01 01:30"])
    s = pd.Series([np.nan, np.nan, np.nan], index=idx)
    assert_parity(tw_resampler(s, "H"), run_old(s, "H"), "all nan")


def test_gaps_between_buckets():
    # empty buckets in the middle must appear as NaN on the regular grid
    idx = pd.to_datetime([
        "2024-01-01 00:10", "2024-01-01 00:50",
        "2024-01-01 05:10", "2024-01-01 05:50",   # 4h gap
    ])
    s = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
    new = tw_resampler(s, "H")
    old = run_old(s, "H")
    assert_parity(new, old, "gaps")
    assert new.isna().sum() == 4   # 01:00..04:00 empty


def test_pre_1970_dates():
    # long streamflow records predate the 1970 epoch (negative ns timestamps);
    # floor division must still bin like pandas resample, incl. across 1970.
    for start, (freq, span) in [
        ("1911-03-05", ("H", 20)), ("1911-03-05", ("D", 400)),
        ("1969-12-20", ("H", 25)), ("1969-12-20", ("D", 400)),
    ]:
        rng = np.random.default_rng(0)
        s0 = np.datetime64(start, "s").astype("datetime64[ns]").astype(np.int64)
        span_ns = span * 86400 * 10 ** 9
        off = rng.integers(0, span_ns // Q15, size=3000, dtype=np.int64) * Q15
        s = pd.Series(rng.standard_normal(3000) * 100,
                      index=pd.DatetimeIndex((s0 + off).astype("datetime64[ns]")))
        s = s[~s.index.duplicated(keep="first")].sort_index()
        assert_parity(tw_resampler(s, freq), run_old(s, freq),
                      f"pre-1970 {start} {freq}")


def test_duplicate_timestamps():
    # real data may carry duplicated timestamps; the reference is order-dependent
    # there (unstable sort_index in prepend/append), so the new impl delegates to
    # it. This holds even for SORTED input, hourly and daily.
    rng = np.random.default_rng(1)
    s0 = np.datetime64("2010-01-01", "s").astype("datetime64[ns]").astype(np.int64)
    off = rng.integers(0, 20 * 86400 // 900, size=2000, dtype=np.int64) * Q15
    s = pd.Series(rng.standard_normal(2000) * 100,
                  index=pd.DatetimeIndex((s0 + off).astype("datetime64[ns]"))).sort_index()
    assert s.index.duplicated().any()
    for freq in ("H", "D"):
        assert_parity(tw_resampler(s, freq), run_old(s, freq),
                      f"duplicate timestamps {freq}", rtol=0.0, atol=1e-12)


def test_sorted_duplicate_regression():
    # locked-in counterexample: a SORTED daily series with duplicate timestamps
    # that the fast path (stable sort) would answer 74.75 while the reference
    # (unstable sort) answers 38.81 -> must delegate and match the reference.
    idx = pd.DatetimeIndex([
        "2199-11-15 04:00", "2199-11-15 07:30", "2199-11-15 07:30",
        "2199-11-15 08:00", "2199-11-15 08:45", "2199-11-15 09:45",
        "2199-11-15 12:45", "2199-11-15 14:45", "2199-11-15 14:45",
        "2199-11-15 15:15", "2199-11-15 15:45", "2199-11-15 16:30",
        "2199-11-15 17:15", "2199-11-15 18:15", "2199-11-15 21:00",
        "2199-11-15 23:00"])
    vals = np.array([302.254545, 170.910170, -404.261690, -307.694602,
                     201.578829, 444.679058, -382.281465, 418.327785,
                     -1605.465267, 1610.018898, 282.738314, -187.845787,
                     352.976200, 325.852784, -44.492672, -865.780887])
    s = pd.Series(vals, index=idx, name="q")
    assert s.index.is_monotonic_increasing and s.index.duplicated().any()
    assert_parity(tw_resampler(s, "D"), run_old(s, "D"),
                  "sorted-dup regression", rtol=0.0, atol=1e-12)


def test_output_dtype_and_name():
    s = gen(0, 500, "H", 20)
    s.name = "discharge"
    new = tw_resampler(s, "H")
    assert new.dtype == np.float64
    assert new.name == "discharge"


def test_index_grid_matches_resample():
    for freq, span in (("H", 30), ("D", 200)):
        s = gen(1, 2500, freq, span)
        new, old = tw_resampler(s, freq), run_old(s, freq)
        assert new.index.equals(old.index)
        assert new.index.freq == old.index.freq   # freq metadata reproduced


def test_index_unit_and_freq_metadata():
    # .resample() preserves the input's datetime unit and sets .freq; the fast
    # path must reproduce both so .equals() holds even for coarser-than-ns input
    # (real data is ns, but this makes it a true drop-in).
    for unit in ("s", "ms", "us", "ns"):
        idx = pd.date_range("2020-01-01", "2020-01-06", freq="15min").as_unit(unit)
        s = pd.Series(np.arange(len(idx)) * 1.0, index=idx)
        for freq in ("H", "D"):
            new, old = tw_resampler(s, freq), run_old(s, freq)
            assert new.index.equals(old.index), f"unit={unit} freq={freq}"
            assert new.index.unit == old.index.unit == unit
            assert new.index.freq == old.index.freq


def test_input_not_mutated():
    s = gen(2, 1000, "H", 20)
    before_idx = s.index.copy()
    before_vals = s.values.copy()
    _ = tw_resampler(s, "H")
    assert s.index.equals(before_idx)
    np.testing.assert_array_equal(s.values, before_vals)


def test_unsorted_input():
    s = gen(3, 2000, "H", 30)
    shuffled = s.sample(frac=1.0, random_state=7)   # scramble row order
    new = tw_resampler(shuffled, "H")
    old = run_old(s, "H")                            # old on sorted reference
    assert_parity(new, old, "unsorted input")


def test_invalid_freq_raises():
    s = gen(0, 100, "H", 10)
    for bad in ("W", "M", "min", "15min", "", None, 3):
        try:
            tw_resampler(s, bad)
        except ValueError:
            continue
        raise AssertionError(f"freq={bad!r} should have raised ValueError")


# ---------------------------------------------------------------------------
# frequency spelling: lowercase must equal uppercase and match the (now fixed)
# reference. This also guards the prepend/append `freq.upper()=="H"` fix.
# ---------------------------------------------------------------------------
def test_lowercase_freq_equivalence():
    for freq_l, freq_u in (("h", "H"), ("d", "D")):
        span = 30 if freq_u == "H" else 300
        s = gen(5, 3000, freq_u, span)
        new_l = tw_resampler(s, freq_l)
        new_u = tw_resampler(s, freq_u)
        assert_parity(new_l, new_u, f"lowercase-{freq_l} vs upper")
        # reference with lowercase freq (exercises the prepend/append fix)
        old_l = s.resample(freq_l).apply(
            lambda x: tw_resampler_(x, s.sort_index(), freq_l))
        assert_parity(new_l, old_l, f"lowercase-{freq_l} vs old-lowercase")


# ---------------------------------------------------------------------------
# tz-aware parity + tz preservation
# ---------------------------------------------------------------------------
def test_tz_aware_fixed_offset_fast_path():
    # UTC / fixed-offset tz-aware data is handled by the fast path (bins on
    # local wall time) and matches the reference.
    s = gen(4, 2000, "H", 30)
    for tz in ("UTC", "Etc/GMT+5"):
        s_tz = s.tz_localize(tz)
        new = tw_resampler(s_tz, "H")
        assert str(new.index.tz) == str(s_tz.index.tz)
        assert_parity(new, run_old(s_tz, "H"), f"tz-aware {tz}")


def test_dst_transition_falls_back_and_matches():
    # a DST spring-forward inside a tz-aware range must delegate to the exact
    # reference (fast path would crash / use a wrong-length day).
    idx = pd.date_range("2024-03-30 20:00", "2024-04-01 06:00",
                        freq="15min", tz="Europe/Dublin")
    s = pd.Series(np.arange(len(idx)) * 1.0, index=idx, name="q")
    for freq in ("H", "D"):
        new = tw_resampler(s, freq)
        assert str(new.index.tz) == "Europe/Dublin"
        assert_parity(new, run_old(s, freq), f"DST spring {freq}",
                      rtol=0.0, atol=1e-12)


def test_dst_grid_crosses_but_observations_do_not():
    # the DST guard must be a property of the GRID, not the observations:
    # (A) a single 23h/25h DST day with all obs on one side -> must not use a
    #     wrong-length day silently; (B) an hourly grid spanning a transition
    #     while all obs share one offset -> must not crash.
    caseA = pd.Series(
        [10.0, 20.0],
        index=pd.DatetimeIndex(pd.to_datetime(
            ["2024-03-31 04:00", "2024-03-31 10:00"])).tz_localize("Europe/London"))
    assert_parity(tw_resampler(caseA, "D"), run_old(caseA, "D"),
                  "DST 23h-day grid", rtol=0.0, atol=1e-12)
    caseB = pd.Series(
        [1.0, 2.0],
        index=pd.DatetimeIndex(pd.to_datetime(
            ["2024-02-15 12:00", "2024-11-15 12:00"])).tz_localize("Europe/London"))
    assert_parity(tw_resampler(caseB, "H"), run_old(caseB, "H"),
                  "DST grid-spans-transition", rtol=0.0, atol=1e-12)


def test_fractional_hour_dst_zone_falls_back():
    # a fractional-hour DST shift (Lord Howe = 30 min) between the resample
    # origin (first day's local midnight) and the first observation moves every
    # bin edge off local ':00'; the guard must extend back to that midnight and
    # delegate, not silently return differently-binned values.
    idx = pd.DatetimeIndex(pd.to_datetime([
        "2020-04-05 03:10", "2020-04-05 04:35", "2020-04-05 06:00",
        "2020-04-05 07:25", "2020-04-05 08:50"])).tz_localize("Australia/Lord_Howe")
    s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0], index=idx, name="q")
    assert_parity(tw_resampler(s, "H"), run_old(s, "H"),
                  "fractional-DST hourly", rtol=0.0, atol=1e-12)


def test_dst_free_span_in_dst_zone_uses_fast_path():
    # a range with NO transition inside a DST-capable zone must still match
    # (and stay on the fast path).
    idx = pd.date_range("2015-06-01", "2015-08-01", freq="15min",
                        tz="Europe/London")
    s = pd.Series(np.arange(len(idx)) * 0.5, index=idx)
    assert str(tw_resampler(s, "H").index.tz) == "Europe/London"
    assert_parity(tw_resampler(s, "H"), run_old(s, "H"), "DST-free span")


def test_unsorted_with_duplicates_falls_back_and_matches():
    # unsorted input carrying duplicate timestamps delegates to the reference
    # (whose duplicate tie-order is otherwise not reproducible bit-for-bit).
    rng = np.random.default_rng(3)
    s0 = np.datetime64("2010-01-01", "s").astype("datetime64[ns]").astype(np.int64)
    off = rng.integers(0, 20 * 86400 // 900, size=800, dtype=np.int64) * Q15
    s = pd.Series(rng.standard_normal(800) * 100,
                  index=pd.DatetimeIndex((s0 + off).astype("datetime64[ns]")))
    assert s.index.duplicated().any() and not s.index.is_monotonic_increasing
    assert_parity(tw_resampler(s, "H"), run_old(s, "H"),
                  "unsorted+dup", rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# sub-minute fallback: warns and delegates to the exact per-bucket path
# ---------------------------------------------------------------------------
def test_subminute_fallback_matches_and_warns():
    # second-aligned (sub-minute) hourly data -> fast path invalid -> fallback
    rng = np.random.default_rng(0)
    start = np.datetime64("2010-01-01", "s").astype("datetime64[ns]").astype(np.int64)
    off = rng.integers(0, 30 * 86400, size=3000, dtype=np.int64) * SEC
    times = (start + off).astype("datetime64[ns]")
    s = pd.Series(rng.standard_normal(3000) * 100, index=pd.DatetimeIndex(times))
    s = s[~s.index.duplicated(keep="first")].sort_index()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        new = tw_resampler(s, "H")
    assert any("sub-minute" in str(w.message) for w in caught), "expected warning"
    old = run_old(s, "H")
    # fallback IS the reference call, so parity must be effectively exact
    assert_parity(new, old, "subminute fallback", rtol=0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# speed
# ---------------------------------------------------------------------------
def _benchmark(n=120_000, span_days=3650):
    s = gen(0, n, "H", span_days, align_ns=Q15)
    t0 = time.perf_counter(); new = tw_resampler(s, "H"); t_new = time.perf_counter() - t0
    t0 = time.perf_counter(); old = run_old(s, "H"); t_old = time.perf_counter() - t0
    assert_parity(new, old, f"benchmark n={len(s)}")
    return t_old, t_new, len(s), len(new)


def test_speed_faster_than_reference():
    t_old, t_new, n_obs, n_buckets = _benchmark()
    print(f"\n  n={n_obs:,} obs -> {n_buckets:,} hourly buckets")
    print(f"  old: {t_old:7.3f}s   new: {t_new:7.3f}s   speedup: {t_old / t_new:6.1f}x")
    assert t_new < t_old, "vectorised version should be faster than per-bucket"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print("\nall parity + edge-case tests passed")
