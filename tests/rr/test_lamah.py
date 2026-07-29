"""
Integration tests for the LamaH rainfall-runoff datasets:

    * ``LamaHCE``  - LamaH-CE, Central Europe (mainly Austria)
    * ``LamaHIce`` - LamaH-Ice, Iceland

Both datasets are exercised at daily (``'D'``) and hourly (``'H'``) timestep and
for the three basin delineations (``total_upstrm`` / ``intermediate_all`` /
``intermediate_lowimp``). Every combination is run through the shared,
comprehensive :func:`utils.test_dataset` suite, which checks the fetching
fidelity, the static/dynamic feature counts, coordinates, boundaries, ``q_mm``
and that no re-download/re-extraction happens.

Run directly (``python test_lamah.py``); the data is expected to be already
downloaded under ``GSCAD_PATH``.
"""

import os
import site
import logging

# add the repository root to the path so that ``aqua_fetch`` and the shared
# ``utils`` test helpers can be imported
wd_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
site.addsitedir(wd_dir)

if __name__ == "__main__":
    logging.basicConfig(filename='test_lamah.log', filemode='w',
                        level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

from aqua_fetch import LamaHCE, LamaHIce

from utils import test_dataset

# Root under which the (already downloaded) data lives. Each class appends its
# own sub-directory, so e.g. LamaH-CE daily is expected at
# ``<GSCAD_PATH>/LamaHCE_daily/LamaHCE/...``. Replace with the path on your
# machine; on another host the data is at
# ``/mnt/datawaha/hyex/atr/gscad_database/raw``.
GSCAD_PATH = '/mnt/storage1/atr/data/gscad_database/raw'

VERBOSITY = 4

# the three basin delineations, shared by both datasets and both timesteps
DATA_TYPES = ['total_upstrm', 'intermediate_all', 'intermediate_lowimp']


# ---------------------------------------------------------------------------
# LamaH-CE (Central Europe, mainly Austria)
# ---------------------------------------------------------------------------
# number of stations / static attributes per data_type (same for both timesteps)
LAMAHCE_NUM_STATIONS = [859, 859, 454]
LAMAHCE_NUM_STATIC = [84, 85, 85]


def test_lamahce_daily():
    """LamaH-CE at daily timestep, all three basin delineations."""
    for idx, data_type in enumerate(DATA_TYPES):
        logger.info(f"testing LamaHCE {data_type} at daily timestep")

        dataset = LamaHCE(path=os.path.join(GSCAD_PATH, 'LamaHCE_daily'),
                          timestep='D', data_type=data_type, verbosity=VERBOSITY)

        test_dataset(dataset,
                     num_stations=LAMAHCE_NUM_STATIONS[idx],
                     dyn_data_len=14244,
                     num_static_attrs=LAMAHCE_NUM_STATIC[idx],
                     num_dyn_attrs=22,
                     yearly_steps=366,
                     test_latlong_ranges=False)
    return


def test_lamahce_hourly():
    """LamaH-CE at hourly timestep, all three basin delineations."""
    for idx, data_type in enumerate(DATA_TYPES):
        logger.info(f"testing LamaHCE {data_type} at hourly timestep")

        dataset = LamaHCE(path=os.path.join(GSCAD_PATH, 'LamaHCE_hourly'),
                          timestep='H', data_type=data_type, verbosity=VERBOSITY)

        test_dataset(dataset,
                     num_stations=LAMAHCE_NUM_STATIONS[idx],
                     dyn_data_len=341856,
                     num_static_attrs=LAMAHCE_NUM_STATIC[idx],
                     num_dyn_attrs=16,
                     yearly_steps=8761,
                     test_latlong_ranges=False)
    return


# ---------------------------------------------------------------------------
# LamaH-Ice (Iceland)
# ---------------------------------------------------------------------------
# number of stations is the same for both timesteps; the static-attribute count
# differs because the daily ``total_upstrm`` product ships extra water-balance
# attributes that the hourly / intermediate products do not.
LAMAHICE_NUM_STATIONS = [111, 107, 86]
LAMAHICE_DAILY_NUM_STATIC = [154, 114, 114]
LAMAHICE_HOURLY_NUM_STATIC = [138, 114, 114]


def test_lamahice_hourly():
    """LamaH-Ice at hourly timestep, all three basin delineations."""
    for idx, data_type in enumerate(DATA_TYPES):
        logger.info(f"testing LamaHIce {data_type} at hourly timestep")

        dataset = LamaHIce(path=os.path.join(GSCAD_PATH, 'LamaHIce_hourly'),
                           timestep='H', data_type=data_type, verbosity=VERBOSITY)

        test_dataset(dataset,
                     num_stations=LAMAHICE_NUM_STATIONS[idx],
                     dyn_data_len=412848,
                     num_static_attrs=LAMAHICE_HOURLY_NUM_STATIC[idx],
                     num_dyn_attrs=28,
                     yearly_steps=8761,
                     test_latlong_ranges=False)
    return


def test_lamahice_daily():
    """LamaH-Ice at daily timestep, all three basin delineations."""
    for idx, data_type in enumerate(DATA_TYPES):
        logger.info(f"testing LamaHIce {data_type} at daily timestep")

        dataset = LamaHIce(path=os.path.join(GSCAD_PATH, 'LamaHIce_daily'),
                           timestep='D', data_type=data_type, verbosity=VERBOSITY)

        test_dataset(dataset,
                     num_stations=LAMAHICE_NUM_STATIONS[idx],
                     dyn_data_len=26298,
                     num_static_attrs=LAMAHICE_DAILY_NUM_STATIC[idx],
                     num_dyn_attrs=36,
                     yearly_steps=366,
                     test_latlong_ranges=False)
    return


if __name__ == "__main__":
    test_lamahce_daily()
    test_lamahce_hourly()
    test_lamahice_hourly()
    test_lamahice_daily()

    print("*** All LamaH tests passed ***")
